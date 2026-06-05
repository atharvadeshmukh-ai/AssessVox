"""AssessVox — FastAPI server for the 3-panel oral-exam UI.

Endpoints:
  GET  /                       static UI
  POST /api/sessions           teacher submits question (+ optional grading prompt)
  GET  /api/sessions/{id}/flow returns the generated FlowSpec
  GET  /api/sessions/{id}/report returns the grading report (404 until ready)
  POST /api/offer              WebRTC SDP exchange, spawns the bot
  PATCH /api/offer             ICE candidate trickle

Run with:  uv run assessvox.py
Then open: http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
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


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ─── Session endpoints ────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    question: str
    duration_minutes: int = 6
    custom_grading_prompt: str | None = None
    student_name: str | None = None
    llm_provider: str = "google"
    llm_model: str = "gemini-3.5-flash"
    # ISO-639-1 STT language. "en" locks transcripts to English (the default).
    # Empty string / None re-enables ElevenLabs auto-detect.
    stt_language: str | None = "en"


class CreateSessionResponse(BaseModel):
    session_id: str
    flow: dict


@app.get("/api/grading-prompt", response_model=dict)
async def get_default_grading_prompt():
    """Return the default grading prompt (and the dimension guide) so the
    advanced-options panel can show it as editable default text."""
    return {"prompt": GRADING_PROMPT, "dimension_guide": DIMENSION_GUIDE}


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest):
    """Teacher submits a question → server generates + validates a FlowSpec."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    logger.info(f"Creating session {session_id} | topic={req.question!r}")

    # Generate flow (retry once on validation failure)
    last_errors: list[str] = []
    flow_spec = None
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

    _SESSIONS[session_id] = {
        "session_id": session_id,
        "topic": req.question,
        "duration_minutes": req.duration_minutes,
        "custom_grading_prompt": req.custom_grading_prompt,
        "student_name": (req.student_name or "").strip() or None,
        "flow_spec": flow_spec,
        "status": "ready",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "llm_provider": req.llm_provider,
        "llm_model": req.llm_model,
        "stt_language": (req.stt_language or "").strip() or None,
    }

    logger.info(
        f"Session {session_id} ready | {len(flow_spec.probe_pool)} probes | "
        f"required={flow_spec.required_probes}"
    )
    return CreateSessionResponse(session_id=session_id, flow=flow_spec.model_dump())


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
    s = _SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "unknown session")
    report_path = REPORTS_DIR / f"report_{session_id}.json"
    if not report_path.exists():
        raise HTTPException(404, "report not ready yet")
    return JSONResponse(json.loads(report_path.read_text()))


# ─── WebRTC handshake ─────────────────────────────────────────────────────────

_webrtc_handler = SmallWebRTCRequestHandler()


def _build_transport(connection: SmallWebRTCConnection) -> SmallWebRTCTransport:
    # RNNoise strips steady-state background noise (fans, HVAC, hum) from the
    # mic feed before it reaches VAD/STT. Documented free alternative to Krisp
    # VIVA — cuts the spurious "Hello?" / "Yeah." hallucinations that ElevenLabs
    # scribe_v2 produces from ambient noise.
    return SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_in_filter=RNNoiseFilter(),
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
            llm_model=s.get("llm_model", "gemini-3.5-flash"),
            stt_language=s.get("stt_language", "en"),
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
