"""AssessVox — FastAPI server for the 3-panel oral-exam UI.

Endpoints:
  GET  /                           static UI
  POST /api/sessions               teacher submits question (+ optional grading prompt)
  GET  /api/sessions/{id}/flow     returns the generated FlowSpec
  GET  /api/sessions/{id}/report   returns the grading report (404 until ready)
  POST /api/offer                  WebRTC SDP exchange, spawns the bot
  PATCH /api/offer                 ICE candidate trickle
  POST /api/sessions/{id}/upload-recording  receive video blob from browser
  POST /api/sessions/{id}/grade-video       trigger multimodal video grading

Run with:  uv run assessvox.py
Then open: http://localhost:8000
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

# Pipecat WebRTC handler (mirrors pipecat/runner/run.py:_setup_webrtc_routes)
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# LiveKit — used for video-mode sessions. Import is soft so the server
# starts even if the package isn't installed (video sessions will simply
# return an error rather than crashing everything else).
try:
    from livekit.api import AccessToken, LiveKitAPI, VideoGrants
    from livekit.protocol.room import CreateRoomRequest
    _LIVEKIT_AVAILABLE = True
except ImportError:
    _LIVEKIT_AVAILABLE = False

# Local imports
import dynamic_exam
from dynamic_exam import (
    DIMENSION_GUIDE,
    FLOWS_DIR,
    GRADING_PROMPT,
    REPORTS_DIR,
)
from flow_planner import generate_flow
from flow_validator import FlowValidationError, validate_flow_spec
from template_loader import get_template, list_templates

load_dotenv(override=True)

WEB_DIR = Path(__file__).parent / "web"

# ─── In-memory session store ──────────────────────────────────────────────────
# session_id → {topic, duration, custom_grading_prompt, flow_spec, status, ...}
_SESSIONS: dict[str, dict[str, Any]] = {}


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="AssessVox")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/sessions", include_in_schema=False)
async def sessions_page():
    """Session-inspector dashboard."""
    return FileResponse(WEB_DIR / "sessions.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ─── Session timeline (dashboard) endpoints ──────────────────────────────────

import session_timeline as _session_timeline


@app.get("/api/session-timeline/list")
async def list_all_sessions():
    """Newest-first list of every session on disk for the dashboard picker."""
    return {"sessions": _session_timeline.list_sessions()}


@app.get("/api/session-timeline/{session_id}")
async def get_session_timeline(session_id: str):
    """Parsed timeline for one session: turns, LLM calls, node transitions,
    function calls, latency, slack events."""
    t = _session_timeline.build_timeline(session_id)
    if t is None:
        raise HTTPException(404, f"no log file found for session {session_id}")
    return _session_timeline.timeline_to_dict(t)


# ─── Session endpoints ────────────────────────────────────────────────────────

# Models that don't follow our long multi-step system prompts well enough.
# Repeatedly observed in logs: 3-attempt-cap rule ignored, autocompletion of
# student fragments, phantom function calls. Coerced to gemini-3.5-flash.
_LLM_MODEL_BLOCKLIST = frozenset()  # gemini-3.1-flash-lite re-enabled for testing
_LLM_MODEL_FALLBACK = "gemini-3.5-flash"

# Default conversation LLM. gemini-3.5-flash is accurate but slow (~4.4s TTFB /
# ~2.6s per call), which produces audible dead air mid-turn — the examiner
# pauses between sentences and around function-routing (two-call) turns, so the
# voice "breaks" in the middle of a response. A faster tier closes those gaps.
# Override in .env with GEMINI_CONVERSATION_MODEL (e.g. back to gemini-3.5-flash
# if exam-flow adherence regresses).
DEFAULT_LLM_MODEL = os.getenv("GEMINI_CONVERSATION_MODEL", "gemini-2.5-flash")


def _coerce_llm_model(provider: str, model: str) -> str:
    if (provider or "").lower() == "google" and model in _LLM_MODEL_BLOCKLIST:
        logger.warning(
            f"Coercing LLM {model!r} → {_LLM_MODEL_FALLBACK!r} "
            "(blocked for poor instruction-following on long flow prompts)"
        )
        return _LLM_MODEL_FALLBACK
    return model


# ─── Cartesia voices cache ──────────────────────────────────────────────────
# Server-side cache for the Cartesia voice library so the UI dropdown can
# populate dynamically (including new Indian / regional voices) without
# pelting the Cartesia API on every page load.

_CARTESIA_VOICES_CACHE: list[dict[str, Any]] = []
_CARTESIA_VOICES_FETCHED_AT: float = 0.0
_CARTESIA_VOICES_TTL_SECS: float = 3600.0   # refresh hourly
_CARTESIA_API_VERSION = "2026-03-01"


async def _fetch_cartesia_voices() -> list[dict[str, Any]]:
    """Page through Cartesia's GET /voices and return a flat list.

    Each returned dict has keys: id, name, description, gender, language,
    country. Only public voices are included. Raises on HTTP error.
    """
    api_key = os.getenv("CARTESIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CARTESIA_API_KEY not set in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Cartesia-Version": _CARTESIA_API_VERSION,
    }
    voices: list[dict[str, Any]] = []
    cursor: str | None = None

    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(20):  # safety cap — 20 × 100 = 2000 voices max
            params = {"limit": "100"}
            if cursor:
                params["starting_after"] = cursor
            r = await client.get(
                "https://api.cartesia.ai/voices",
                headers=headers,
                params=params,
            )
            r.raise_for_status()
            body = r.json()
            for v in body.get("data", []):
                voices.append({
                    "id": v.get("id"),
                    "name": v.get("name") or "(unnamed)",
                    "description": v.get("description") or "",
                    "gender": v.get("gender"),
                    "language": v.get("language"),
                    "country": v.get("country"),
                })
            if not body.get("has_more"):
                break
            cursor = body.get("next_page")
            if not cursor:
                break
    return voices


async def _get_cached_cartesia_voices() -> list[dict[str, Any]]:
    """Return cached voices, refreshing if the TTL has expired."""
    global _CARTESIA_VOICES_CACHE, _CARTESIA_VOICES_FETCHED_AT
    now = time.monotonic()
    if (
        _CARTESIA_VOICES_CACHE
        and (now - _CARTESIA_VOICES_FETCHED_AT) < _CARTESIA_VOICES_TTL_SECS
    ):
        return _CARTESIA_VOICES_CACHE
    voices = await _fetch_cartesia_voices()
    _CARTESIA_VOICES_CACHE = voices
    _CARTESIA_VOICES_FETCHED_AT = now
    logger.info(f"Cartesia voices fetched: {len(voices)} total")
    return voices


class CreateSessionRequest(BaseModel):
    # Question can be either:
    #   • Free-text question (template_id is None / empty) — uses flow_planner
    #     to LLM-generate the FlowSpec, as it has since day one.
    #   • An override for a template's default_question (template_id provided)
    #     — uses the template's pre-built probe pool + grading rubric,
    #     substituting the supplied question text.
    question: str = ""
    duration_minutes: int = 6
    custom_grading_prompt: str | None = None
    student_name: str | None = None
    # Template id for pre-built assessments. None / empty = the original
    # LLM-generated free-text flow. See template_loader for the registry.
    template_id: str | None = None
    # Optional id of a question chosen from the template's question library.
    # When provided alongside `template_id`, the library question's prompt
    # is used as the exam_question (ignoring any custom `question` text
    # unless the teacher edited it). When None, falls back to `question`
    # (custom text) or the template's flow_spec.exam_question default.
    question_id: str | None = None
    # LLM provider: "google" (Gemini), "groq" (Llama), "openai" (GPT), "anthropic" (Claude), "cerebras" (GPT-OSS 120B, free tier)
    llm_provider: str = "google"
    llm_model: str = DEFAULT_LLM_MODEL
    # STT vendor: "cartesia" (ink-whisper, default — cheapest + cleanest;
    # no Whisper hallucinations), "cartesia-ink-2" (May 2026 ink-2 — #1 on
    # streaming STT leaderboards, built-in turn detection, English-only),
    # "deepgram" (nova-3 — best keyterm-boost for domain vocabulary),
    # "elevenlabs" (scribe_v2_realtime, legacy path with commit/echo-flush
    # workarounds), "assemblyai" (Universal-3 Pro), or "xai" (Grok voice
    # WebSocket).
    stt_provider: str = "cartesia-ink-2"
    # ISO-639-1 STT language. "en" locks transcripts to English (the default).
    # Empty string / None re-enables auto-detect (where supported).
    stt_language: str | None = "en"
    # Cartesia voice ID for TTS. Default is the "Examiner" voice we've been
    # using; the UI dropdown exposes a curated set, but any valid Cartesia
    # voice ID works.
    tts_voice: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b"
    # LLM turn-marker mixin control (filter_incomplete_user_turns):
    #   "auto" — ON for STTs without native endpointing, OFF for assemblyai
    #            and cartesia-ink-2 (default)
    #   "on"   — force ON for every provider (costs +1 LLM round-trip per turn)
    #   "off"  — force OFF for every provider (skips the marker call; only safe
    #            when STT or VAD reliably detect end-of-turn on their own)
    use_turn_markers: str = "auto"
    # When True (default), Pipecat's assistant aggregator runs auto-context
    # summarization in the background — older turns are compressed via an LLM
    # call so the per-turn prompt stays bounded. Set False to disable on short
    # sessions or for debugging.
    enable_auto_summarization: bool = True


class CreateSessionResponse(BaseModel):
    session_id: str
    flow: dict
    image_url: str | None = None
    requires_video: bool = False
    livekit_url: str | None = None
    livekit_token: str | None = None
    livekit_room: str | None = None


@app.get("/api/grading-prompt", response_model=dict)
async def get_default_grading_prompt():
    """Return the default grading prompt (and the dimension guide) so the
    advanced-options panel can show it as editable default text."""
    return {"prompt": GRADING_PROMPT, "dimension_guide": DIMENSION_GUIDE}


@app.get("/api/cartesia-voices", response_model=dict)
async def get_cartesia_voices():
    """Return the Cartesia voice library so the UI can build a region-grouped
    dropdown dynamically. Cached for an hour on the server.

    Falls back to a small curated list on API failure so the UI never breaks.
    """
    try:
        voices = await _get_cached_cartesia_voices()
        return {"voices": voices, "source": "api"}
    except Exception as e:
        logger.warning(f"Cartesia voices API fetch failed, using fallback: {e}")
        return {
            "voices": [
                {"id": "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
                 "name": "Examiner (default)", "language": "en", "country": "US",
                 "gender": "feminine", "description": "Warm, neutral examiner voice"},
                {"id": "71a7ad14-091c-4e8e-a314-022ece01c121",
                 "name": "British Reading Lady", "language": "en", "country": "GB",
                 "gender": "feminine", "description": "Calm, articulate"},
            ],
            "source": "fallback",
            "error": str(e),
        }


@app.get("/api/templates", response_model=dict)
async def get_templates():
    """List available assessment templates so the UI can build the picker.

    Returns a list of templates plus the special "Custom (LLM-generated)"
    entry which is the legacy free-text flow path.
    """
    tpls = [t.to_dict() for t in list_templates()]
    return {"templates": tpls}


@app.get("/api/templates/{template_id}/questions", response_model=dict)
async def get_template_questions(template_id: str):
    """Return the question library for a given template.

    The UI populates a second dropdown from this — teachers pick a specific
    question for that assessment shape, or "Write custom" to type their own.
    """
    tpl = get_template(template_id)
    if tpl is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "unknown_template_id", "template_id": template_id},
        )
    return {
        "template_id": template_id,
        "questions": tpl.questions_to_dict(),
    }


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest):
    """Teacher submits a question → server generates + validates a FlowSpec.

    Two paths:
      • Template path (template_id provided): load pre-built FlowSpec from JSON
        template, optionally override the exam_question with req.question.
        Skips flow_planner entirely — deterministic, fast.
      • LLM path (template_id empty / None): legacy free-text path — call
        flow_planner.generate_flow which uses the LLM to invent the FlowSpec.
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    template_id = (req.template_id or "").strip() or None
    flow_spec = None
    template_grading_rubric: str | None = None
    duration_minutes = req.duration_minutes

    if template_id:
        # ─── Template path ─────────────────────────────────────────────────
        tpl = get_template(template_id)
        if tpl is None:
            raise HTTPException(
                status_code=400,
                detail={"error": "unknown_template_id", "template_id": template_id},
            )

        # Resolve the question. Priority:
        #   1. Custom text in req.question (teacher typed something)
        #   2. Library question by req.question_id
        #   3. Template's flow_spec.exam_question default
        question_id = (req.question_id or "").strip() or None
        question_text_override: str | None = (req.question or "").strip() or None
        image_url: str | None = None
        requires_video: bool = False
        template_video_rubric: str | None = None
        if question_id:
            q = tpl.get_question(question_id)
            if q is None:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "unknown_question_id",
                            "template_id": template_id,
                            "question_id": question_id},
                )
            image_url = q.get("image_url") or None
            requires_video = bool(q.get("requires_video"))
            if not question_text_override:
                question_text_override = q.get("prompt") or None

        logger.info(
            f"Creating session {session_id} | template={template_id} | "
            f"question_id={question_id!r} | "
            f"question={question_text_override or '(template default)'!r}"
        )
        try:
            flow_spec = tpl.build_flow_spec(question_override=question_text_override)
            validate_flow_spec(flow_spec)
        except FlowValidationError as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "template_invalid", "reasons": e.reasons,
                        "template_id": template_id},
            )
        template_grading_rubric = tpl.grading_rubric
        template_video_rubric = getattr(tpl, "video_grading_rubric", None)
        # Template-defined duration is the canonical one; req.duration_minutes
        # is only honoured for the custom (LLM) path.
        duration_minutes = flow_spec.duration_minutes
    else:
        # ─── Legacy LLM-generated path ─────────────────────────────────────
        image_url = None
        requires_video = False
        template_video_rubric = None
        logger.info(f"Creating session {session_id} | topic={req.question!r}")
        if not (req.question or "").strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "question_required",
                        "message": "Either provide template_id or a non-empty question."},
            )
        last_errors: list[str] = []
        for attempt in (1, 2):
            try:
                spec = await generate_flow(req.question, req.duration_minutes)
                validate_flow_spec(spec)
                flow_spec = spec
                break
            except FlowValidationError as e:
                last_errors = e.reasons
                logger.warning(f"attempt {attempt} validation failed: {last_errors}")
            except Exception as e:
                last_errors = [str(e)]
                logger.warning(f"attempt {attempt} generation failed: {last_errors}")
        if flow_spec is None:
            raise HTTPException(
                status_code=502,
                detail={"error": "flow_generation_failed", "reasons": last_errors},
            )

    # Persist spec to disk
    FLOWS_DIR.mkdir(parents=True, exist_ok=True)
    flow_path = FLOWS_DIR / f"flow_{session_id}.json"
    flow_path.write_text(flow_spec.model_dump_json(indent=2))

    # Custom grading prompt resolution (priority order):
    #   1. req.custom_grading_prompt — teacher-overridden in Advanced settings
    #   2. template_grading_rubric  — from the chosen template
    #   3. None — grading falls back to dynamic_exam.GRADING_PROMPT default
    custom_prompt = req.custom_grading_prompt or template_grading_rubric

    _SESSIONS[session_id] = {
        "session_id": session_id,
        "topic": req.question or flow_spec.topic,
        "duration_minutes": duration_minutes,
        "custom_grading_prompt": custom_prompt,
        "template_id": template_id,
        "student_name": (req.student_name or "").strip() or None,
        "flow_spec": flow_spec,
        "status": "ready",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "llm_provider": req.llm_provider,
        "llm_model": _coerce_llm_model(req.llm_provider, req.llm_model),
        "stt_provider": (req.stt_provider or "cartesia-ink-2").strip().lower(),
        "stt_language": (req.stt_language or "").strip() or None,
        "tts_voice": (req.tts_voice or "").strip()
                     or "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
        "use_turn_markers": (req.use_turn_markers or "auto").strip().lower(),
        "enable_auto_summarization": bool(req.enable_auto_summarization),
        "image_url": image_url,
        "requires_video": requires_video,
        "video_grading_rubric": template_video_rubric,
        "livekit_room": None,
    }

    # ─── Video branch: mint a LiveKit room + student token ─────────────────
    livekit_url: str | None = None
    livekit_token: str | None = None
    livekit_room: str | None = None
    if requires_video:
        livekit_room = f"assessvox-{session_id}"
        student_identity = (req.student_name or "").strip() or f"student-{session_id[-6:]}"
        exam_spec_for_agent = {
            "session_id": session_id,
            "exam_question": flow_spec.exam_question,
            "topic": flow_spec.topic,
            "examiner_tone": flow_spec.examiner_tone,
            "redirect_when_off_topic": flow_spec.redirect_when_off_topic,
            "duration_minutes": flow_spec.duration_minutes,
            "required_probes": list(flow_spec.required_probes),
            "skills_assessed": list(flow_spec.skills_assessed or []),
            "probe_pool": [p.model_dump() for p in flow_spec.probe_pool],
            "student_name": (req.student_name or "").strip() or None,
        }
        await _create_livekit_room(livekit_room, exam_spec_for_agent)
        livekit_token, livekit_url = _mint_livekit_token(livekit_room, student_identity)
        _SESSIONS[session_id]["livekit_room"] = livekit_room
        logger.info(
            f"Session {session_id} VIDEO — LiveKit room {livekit_room!r} "
            f"created, student identity={student_identity!r}"
        )

    logger.info(
        f"Session {session_id} ready | {len(flow_spec.probe_pool)} probes | "
        f"required={flow_spec.required_probes} | "
        f"source={'template:' + template_id if template_id else 'llm-generated'}"
    )
    return CreateSessionResponse(
        session_id=session_id,
        flow=flow_spec.model_dump(),
        image_url=image_url,
        requires_video=requires_video,
        livekit_url=livekit_url,
        livekit_token=livekit_token,
        livekit_room=livekit_room,
    )


RECORDINGS_DIR = Path(__file__).parent / "recordings"


def _find_recording(session_id: str) -> Path | None:
    for ext in ("mp4", "webm", "mov"):
        p = RECORDINGS_DIR / f"{session_id}.{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


async def _create_livekit_room(room_name: str, exam_spec: dict) -> None:
    if not _LIVEKIT_AVAILABLE:
        raise HTTPException(500, {"error": "livekit_not_installed",
                                  "message": "Install livekit and livekit-api packages."})
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    url = os.getenv("LIVEKIT_URL")
    if not (api_key and api_secret and url):
        raise HTTPException(500, {"error": "livekit_not_configured"})
    http_url = url.replace("wss://", "https://").replace("ws://", "http://")
    lk = LiveKitAPI(url=http_url, api_key=api_key, api_secret=api_secret)
    try:
        await lk.room.create_room(CreateRoomRequest(
            name=room_name,
            metadata=json.dumps(exam_spec),
            # Persist the room (and its exam-spec metadata) for an hour. At the
            # old 300s, if the student generated a flow and didn't Start Exam
            # within 5 min, LiveKit garbage-collected the empty room; joining
            # later auto-recreated it WITHOUT metadata, so the agent fell back to
            # the generic "which assessment?" prompt instead of running the exam.
            empty_timeout=3600,
            max_participants=4,
        ))
    finally:
        await lk.aclose()


def _mint_livekit_token(room_name: str, identity: str) -> tuple[str, str]:
    if not _LIVEKIT_AVAILABLE:
        raise HTTPException(500, {"error": "livekit_not_installed"})
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    url = os.getenv("LIVEKIT_URL")
    if not (api_key and api_secret and url):
        raise HTTPException(500, {"error": "livekit_not_configured",
                                  "message": "LIVEKIT_URL / LIVEKIT_API_KEY / "
                                             "LIVEKIT_API_SECRET must be set in .env."})
    token = (
        AccessToken(api_key=api_key, api_secret=api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=room_name,
                                 can_publish=True, can_subscribe=True))
        .with_ttl(dt.timedelta(hours=2))
    )
    return token.to_jwt(), url


@app.post("/api/sessions/{session_id}/upload-recording")
async def upload_recording(session_id: str, recording: UploadFile = File(...)):
    """Receive a recording blob from the browser (client-side MediaRecorder)."""
    if session_id not in _SESSIONS:
        raise HTTPException(404, f"unknown session {session_id}")
    if not _SESSIONS[session_id].get("requires_video"):
        raise HTTPException(400, f"session {session_id} is not a video session")
    mt = (recording.content_type or "").lower()
    ext = "mp4" if "mp4" in mt else ("mov" if "quicktime" in mt or "mov" in mt else "webm")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RECORDINGS_DIR / f"{session_id}.{ext}"
    bytes_written = 0
    with out_path.open("wb") as f:
        while True:
            chunk = await recording.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            bytes_written += len(chunk)
    logger.info(f"Recording uploaded for {session_id}: {bytes_written / 1_000_000:.1f} MB → {out_path}")
    return {"session_id": session_id, "path": str(out_path),
            "bytes": bytes_written, "content_type": recording.content_type}


@app.post("/api/sessions/{session_id}/grade-video")
async def grade_video_session(session_id: str, background_tasks: BackgroundTasks):
    """Trigger multimodal video grading after a LiveKit session ends."""
    s = _SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, f"unknown session {session_id}")
    if not s.get("requires_video"):
        raise HTTPException(400, f"session {session_id} is not a video session")

    report_path = REPORTS_DIR / f"report_{session_id}.json"
    video_path = _find_recording(session_id)
    flow_spec = s["flow_spec"]
    metadata = {
        "session_id": session_id,
        "student_name": s.get("student_name"),
        "topic": flow_spec.topic,
        "duration_minutes": s.get("duration_minutes"),
        "cancelled": False,
        "video_mode": True,
    }

    if video_path is None:
        logger.warning(f"grade-video: no recording found for {session_id}")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        placeholder = {
            "session_id": session_id, "metadata": metadata,
            "headline": "Recording unavailable — grading skipped",
            "summary": "No recording file found. Upload via /api/sessions/{id}/upload-recording first.",
            "dimensions": {}, "strengths": [], "growth_areas": [], "overall_score": None,
        }
        report_path.write_text(json.dumps(placeholder, indent=2))
        return {"status": "no_recording", "report_path": str(report_path)}

    async def _run_grading():
        try:
            await dynamic_exam.grade_exam(
                wav_path=Path("/dev/null"),
                metadata=metadata,
                report_path=report_path,
                transcript_turns=[],
                node_trace=[],
                duration_seconds=float((s.get("duration_minutes") or 10) * 60),
                exam_question=flow_spec.exam_question,
                topic=flow_spec.topic,
                video_path=video_path,
                custom_video_prompt=s.get("video_grading_rubric"),
            )
        except Exception:
            logger.exception(f"video grading failed for {session_id}")

    background_tasks.add_task(_run_grading)
    return {"status": "grading_started", "video_path": str(video_path)}


# ─── LiveKit worker ───────────────────────────────────────────────────────────

_LK_WORKER_PROC: "subprocess.Popen | None" = None


def _kill_stale_livekit_workers() -> None:
    """Kill any leftover agent.py workers from a previous run before spawning.

    Each stale worker stays registered with LiveKit Cloud and can be dispatched
    the next video session — running old code or with detached output — so a new
    exam silently goes to the wrong worker. Killing them first guarantees only
    the current build serves video rooms.
    """
    try:
        import psutil
    except Exception:
        return
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = p.info.get("cmdline") or []
            if p.pid != me and any("agent.py" in str(c) for c in cmdline):
                p.kill()
                logger.info(f"Killed stale LiveKit worker pid={p.pid}")
        except Exception:
            pass


def _spawn_livekit_worker() -> None:
    global _LK_WORKER_PROC
    if os.getenv("ASSESSVOX_DISABLE_LK_WORKER", "").strip() in ("1", "true", "yes"):
        logger.info("ASSESSVOX_DISABLE_LK_WORKER set; skipping worker spawn")
        return
    _kill_stale_livekit_workers()
    if not (os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_API_KEY")
            and os.getenv("LIVEKIT_API_SECRET")):
        logger.warning("LiveKit credentials missing — video sessions will have no examiner agent.")
        return
    agent_path = Path(__file__).parent / "support-agent" / "agent.py"
    if not agent_path.exists():
        logger.warning(f"Agent worker not found at {agent_path} — video sessions disabled")
        return
    cmd = [sys.executable, str(agent_path), "dev"]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    logger.info(f"Spawning LiveKit agent worker: {' '.join(cmd)}")
    _LK_WORKER_PROC = subprocess.Popen(
        cmd, cwd=str(Path(__file__).parent), env=env,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    logger.info(f"LiveKit worker spawned (pid={_LK_WORKER_PROC.pid})")


def _shutdown_livekit_worker() -> None:
    global _LK_WORKER_PROC
    if _LK_WORKER_PROC is None or _LK_WORKER_PROC.poll() is not None:
        return
    logger.info(f"Terminating LiveKit worker (pid={_LK_WORKER_PROC.pid})")
    try:
        _LK_WORKER_PROC.terminate()
        try:
            _LK_WORKER_PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _LK_WORKER_PROC.kill()
    except Exception as e:
        logger.warning(f"Error stopping LiveKit worker: {e}")


import atexit
atexit.register(_shutdown_livekit_worker)


_WORKER_WATCHDOG_TASK: "asyncio.Task | None" = None


async def _worker_watchdog() -> None:
    """Respawn the LiveKit worker if it dies.

    The worker gives up and exits after ~16 failed connection attempts (e.g. a
    transient DNS/network blip reaching LiveKit Cloud). Without this, one hiccup
    silently takes video mode down until the whole server is restarted — the
    examiner simply never joins the room and never speaks.
    """
    while True:
        await asyncio.sleep(15)
        proc = _LK_WORKER_PROC
        if proc is not None and proc.poll() is not None:
            logger.warning(
                f"LiveKit worker (pid={proc.pid}) exited (code={proc.returncode}); "
                "respawning so video sessions keep working"
            )
            _spawn_livekit_worker()


@app.on_event("startup")
async def _on_startup() -> None:
    global _WORKER_WATCHDOG_TASK
    _spawn_livekit_worker()
    _WORKER_WATCHDOG_TASK = asyncio.create_task(_worker_watchdog())


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    if _WORKER_WATCHDOG_TASK is not None:
        _WORKER_WATCHDOG_TASK.cancel()
    _shutdown_livekit_worker()


@app.get("/api/sessions/{session_id}/flow")
async def get_session_flow(session_id: str):
    s = _SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "unknown session")
    return s["flow_spec"].model_dump()


@app.get("/api/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: str):
    """Live transcript polled by the UI during the call.

    Priority of sources:
      1. The finalised report file (canonical, post-grading).
      2. The live TranscriptCollector (real-time TranscriptionFrame +
         TTSTextFrame events).
      3. The LLMContext messages (turns committed by pipecat).
    """
    if session_id not in _SESSIONS:
        raise HTTPException(404, f"unknown session {session_id}")

    # 1. Post-grading: report file is canonical
    report_path = REPORTS_DIR / f"report_{session_id}.json"
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text())
            return {"turns": data.get("transcript", []), "final": True}
        except Exception:
            pass

    # 2. Live frame-level collector (most up-to-date during the call)
    if dynamic_exam._CURRENT_SESSION_ID == session_id:
        latency = dynamic_exam.get_last_turn_latency()
        live = dynamic_exam.get_live_transcript()
        if live:
            return {"turns": live, "final": False, "last_turn_latency": latency}
        # 3. Fall back to LLMContext if collector empty (e.g. before STT)
        ctx_turns = dynamic_exam.get_current_transcript()
        return {"turns": ctx_turns, "final": False, "last_turn_latency": latency}

    return {"turns": [], "final": False, "last_turn_latency": None}


@app.get("/api/sessions/{session_id}/report")
async def get_session_report(session_id: str):
    report_path = REPORTS_DIR / f"report_{session_id}.json"
    # Serve the report if the file exists, regardless of whether the session
    # is still in memory (handles page-refresh after server restart).
    if report_path.exists():
        return JSONResponse(json.loads(report_path.read_text()))
    # Only 404 with "unknown session" if the session was never created.
    # Otherwise return "not ready yet" so the poller keeps going.
    if session_id not in _SESSIONS:
        raise HTTPException(404, "unknown session")
    raise HTTPException(404, "report not ready yet")


# ─── WebRTC handshake ─────────────────────────────────────────────────────────

_webrtc_handler = SmallWebRTCRequestHandler()


def _build_transport(connection: SmallWebRTCConnection) -> SmallWebRTCTransport:
    # RNNoise was added to cut ElevenLabs scribe_v2 hallucinations from
    # background noise. With Cartesia STT as default it's not needed, and
    # its 16k↔48k resampling can degrade fricatives ("ch" vs "j"). Re-enable
    # by uncommenting `audio_in_filter=RNNoiseFilter()` if noise issues recur.
    return SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            # audio_in_filter=RNNoiseFilter(),   # see comment above
            audio_out_enabled=True,
            video_out_enabled=False,
        ),
    )


async def _bot_for_session(connection: SmallWebRTCConnection, session_id: str) -> None:
    """Background task: build transport, run the dynamic_exam bot, grade afterwards."""
    s = _SESSIONS.get(session_id)
    if not s:
        logger.error(f"_bot_for_session: unknown session {session_id}")
        return

    transport = _build_transport(connection)
    runner_args = SmallWebRTCRunnerArguments(
        webrtc_connection=connection,
        body={"session_id": session_id},
    )
    try:
        await dynamic_exam.run_session(
            transport=transport,
            runner_args=runner_args,
            session_id=session_id,
            flow_spec=s["flow_spec"],
            custom_grading_prompt=s.get("custom_grading_prompt"),
            student_name=s.get("student_name"),
            llm_provider=s.get("llm_provider", "google"),
            llm_model=s.get("llm_model", DEFAULT_LLM_MODEL),
            stt_provider=s.get("stt_provider", "cartesia-ink-2"),
            stt_language=s.get("stt_language", "en"),
            tts_voice=s.get("tts_voice", "e07c00bc-4134-4eae-9ea4-1a55fb45746b"),
            use_turn_markers=s.get("use_turn_markers", "auto"),
            enable_auto_summarization=bool(s.get("enable_auto_summarization", True)),
        )
    except Exception:
        logger.exception(f"bot run failed for session {session_id}")


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Browser sends SDP offer + session_id (in request_data). We look up the
    session, create a connection, spawn the bot, return SDP answer."""
    rd = request.request_data or {}
    session_id = rd.get("session_id") if isinstance(rd, dict) else None
    if not session_id or session_id not in _SESSIONS:
        raise HTTPException(400, f"unknown or missing session_id: {session_id!r}")

    async def _on_connection(connection: SmallWebRTCConnection):
        background_tasks.add_task(_bot_for_session, connection, session_id)

    answer = await _webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=_on_connection,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await _webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("ASSESSVOX_HOST", "0.0.0.0")
    port = int(os.getenv("ASSESSVOX_PORT", "8000"))
    logger.info(f"AssessVox starting on http://{host}:{port}")
    uvicorn.run("assessvox:app", host=host, port=port, reload=False)
