"""LiveKit worker for AssessVox video-mode oral exams.

Runs as a long-lived worker that LiveKit Cloud dispatches into any room
named `assessvox-*`. For each room, it reads the exam spec from
`ctx.room.metadata`, builds a dynamic system prompt, and runs a Gemini Live
realtime session with video_input=True so the model can see the student.

Function tools registered on ExamAgent:
  - record_student_name    — called after name capture (1 call only)
  - advance_probe          — called when student answers a probe substantively
  - off_topic_strike       — called when student is clearly evading
  - end_exam               — called when time is up OR student requests to stop

Run with:
    uv run python support-agent/agent.py dev
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl as _ssl
import sys
from pathlib import Path
from typing import Annotated

# ── Windows: never block on a crash dialog ───────────────────────────────────
# livekit_ffi.dll's bundled soxr resampler can hit a native assertion
# ("FFT_LEN == -1") on this box. By default that pops a modal Visual C++ dialog
# (Abort/Retry/Ignore) which FREEZES the worker until someone clicks it — the
# whole video exam hangs. Suppress the OS error boxes and route CRT abort to
# stderr so the worker instead dies fast and the server's watchdog respawns it.
if sys.platform == "win32":
    try:
        import ctypes
        # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        for _crt in ("ucrtbase", "msvcrt"):
            try:
                _lib = ctypes.CDLL(_crt)
                # _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT):
                # clear both bits so abort() does not raise a dialog.
                _lib._set_abort_behavior(0, 0x1 | 0x2)
            except OSError:
                pass
    except Exception:
        pass

# ── Fast TLS, scoped to Gemini ───────────────────────────────────────────────
# On this Windows / Python 3.14 box, ssl.create_default_context() loads the
# certifi CA bundle in 3-8 SECONDS. google-genai rebuilds that context on every
# Gemini Live (re)connection, each rebuild FREEZING the agent event loop — which
# lags the conversation and trips the WebSocket heartbeat (more drops, more
# rebuilds). We build ONE context (once, at worker warm-up, off the hot path)
# and hand it ONLY to the Gemini client via http_options, so there is no
# per-connection rebuild and — unlike monkeypatching ssl globally — no single
# SSL context object shared across every other library in the process.
_GEMINI_SSL_CTX = None


def _gemini_ssl_ctx():
    global _GEMINI_SSL_CTX
    if _GEMINI_SSL_CTX is None:
        _GEMINI_SSL_CTX = _ssl.create_default_context()
    return _GEMINI_SSL_CTX

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, JobContext, llm, room_io
from livekit.agents.voice.agent_session import VoiceActivityVideoSampler
from livekit.rtc.room import ConnectError
from livekit.plugins import google, silero
from livekit.agents.utils.images import EncodeOptions, ResizeOptions
from google.genai import types as genai_types

sys.path.insert(0, str(Path(__file__).parent))
from prompts import INTERVIEW_INSTRUCTIONS, build_exam_instructions

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env.local")

logger = logging.getLogger("assessvox-video-agent")
logger.setLevel(logging.INFO)


def _media_resolution():
    """Resolve GEMINI_LIVE_MEDIA_RESOLUTION (LOW/MEDIUM/HIGH) to the enum.

    HIGH reads fine detail (handwriting, diagrams) but makes Gemini do more work
    per frame, which delays its audio replies. MEDIUM is the balanced default —
    still far clearer than Gemini's low default, without the audio lag.
    """
    name = os.getenv("GEMINI_LIVE_MEDIA_RESOLUTION", "MEDIUM").upper()
    return {
        "LOW": genai_types.MediaResolution.MEDIA_RESOLUTION_LOW,
        "MEDIUM": genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "HIGH": genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    }.get(name, genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM)


class ExamAgent(Agent):
    """Agent subclass that carries the exam spec and exposes structured tools.

    Tools are discovered automatically by find_function_tools(self) — any
    method decorated with @llm.function_tool is registered with the
    Gemini Live session at start.
    """

    def __init__(self, spec: dict | None, ctx: JobContext):
        instructions = build_exam_instructions(spec) if spec else INTERVIEW_INSTRUCTIONS
        # Ensure our scoped SSL context exists before building the Gemini client
        # (prewarm normally has it ready; this guards the case where this job
        # process didn't run prewarm, so the build still happens here, not on the
        # first conversational turn).
        _gemini_ssl_ctx()
        # Video examiner (Gemini Live) model: the UI's per-exam choice (spec
        # "live_model") wins, then GEMINI_LIVE_MODEL, then the 3.8 Live default.
        live_model = (
            (spec or {}).get("live_model")
            or os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
        )
        logger.info(f"Video examiner model: {live_model}")
        super().__init__(
            instructions=instructions,
            llm=google.realtime.RealtimeModel(
                model=live_model,
                # Female examiner voice. Gemini Live female voices: Aoede, Kore,
                # Leda, Zephyr (male: Puck, Charon, Fenrir, Orus). Override with
                # GEMINI_LIVE_VOICE in .env.
                voice=os.getenv("GEMINI_LIVE_VOICE", "Aoede"),
                temperature=0.6,
                api_key=os.getenv("GEMINI_API_KEY"),
                # Visual detail vs audio latency trade-off (see _media_resolution).
                # MEDIUM by default; set GEMINI_LIVE_MEDIA_RESOLUTION=HIGH to read
                # finer detail at the cost of slower replies.
                media_resolution=_media_resolution(),
                # Encode each video frame at 512px, not the 1024px default. JPEG
                # encoding runs synchronously on the audio event loop; at 1024px
                # each frame blocked it ~200ms (the voice broke). 512px is ~4x
                # cheaper (under the block threshold) and still readable.
                image_encode_options=EncodeOptions(
                    resize_options=ResizeOptions(
                        width=int(os.getenv("GEMINI_LIVE_IMG_SIZE", "512")),
                        height=int(os.getenv("GEMINI_LIVE_IMG_SIZE", "512")),
                        strategy="scale_aspect_fit",
                    ),
                    quality=75,
                ),
                # Hand Gemini our single, pre-built SSL context (warmed in
                # prewarm, off the audio path) so it SKIPS its own
                # ssl.create_default_context() — which on this box costs 0.6-8s
                # and otherwise runs at session start AND on every reconnect,
                # freezing the voice. google-genai has TWO separate SSL paths
                # that read DIFFERENT keys, so we must set both:
                #   • the Live WEBSOCKET reads async_client_args["ssl"]
                #   • the HTTPX client reads client_args/async_client_args["verify"]
                # Setting only "ssl" (as we first did) left the httpx path still
                # rebuilding (~624ms block at Client.__init__, seen in the logs).
                # This stays scoped to the Gemini client — no global ssl patch
                # shared with every other library in the worker.
                http_options=genai_types.HttpOptions(
                    client_args={"verify": _gemini_ssl_ctx()},
                    async_client_args={"ssl": _gemini_ssl_ctx(), "verify": _gemini_ssl_ctx()},
                ),
            ),
        )
        self._spec = spec or {}
        self._ctx = ctx

        # Mutable exam state
        self._student_name: str = (spec or {}).get("student_name", "") or "the student"
        # Keep the base instructions so we can re-issue them with the student's
        # name appended once it's known (see record_student_name). Gemini Live
        # does not reliably remember a name captured via a tool-call result, so
        # we pin it into the system instructions instead.
        self._base_instructions: str = instructions
        self._probes_covered: list[str] = []
        self._required_probes: set[str] = set((spec or {}).get("required_probes") or [])
        self._off_topic_strikes: int = 0
        self._exam_ended: bool = False

    # ─── Tool 1: record student name ──────────────────────────────────────────

    @llm.function_tool
    async def record_student_name(
        self,
        name: Annotated[str, "The student's confirmed first name."],
    ) -> str:
        """Call after the student confirms or corrects their name.
        Call again if they give a different name than the one already recorded.
        Store it so subsequent probes can address them personally.
        """
        self._student_name = name.strip().title()
        logger.info(f"Student name recorded: {self._student_name!r}")
        # Pin the name into the live system instructions. Relying on the model to
        # remember a tool-call result is flaky on Gemini Live — we saw it record
        # a name and then, a turn later, claim "I don't have your name yet." The
        # plugin supports mutable instructions, so this makes the name durable.
        try:
            await self.update_instructions(
                f"{self._base_instructions}\n\n"
                f"# STUDENT NAME — CONFIRMED\n"
                f"The student's name is {self._student_name}. You DO know their "
                f"name. Address them as {self._student_name} where natural, and "
                f"NEVER say you don't know or don't have their name."
            )
        except Exception as e:
            logger.warning(f"Could not pin student name into instructions: {e}")
        return f"Name recorded: {self._student_name}"

    # ─── Tool 2: advance probe ────────────────────────────────────────────────

    @llm.function_tool
    async def advance_probe(
        self,
        probe_id: Annotated[str, "ID of the probe just completed."],
        depth: Annotated[str, "Depth of the student's answer. Must be exactly 'surface' (brief or partial answer, names a real concept) or 'developed' (detailed answer with reasoning and examples)."],
        gold_moment: Annotated[str, "A specific vivid insight the student gave (empty string if none)."],
        recommended_next: Annotated[str, "Probe ID you suggest asking next (from the probe pool)."],
    ) -> str:
        """Call when the student has given a substantive answer to a probe —
        even a short one. This logs progress and tells you what probe to
        move to next. Do NOT call if the student has not yet answered.

        After calling this, immediately ask the recommended_next probe.
        """
        if probe_id not in self._probes_covered:
            self._probes_covered.append(probe_id)
        logger.info(
            f"Probe advanced: {probe_id!r} depth={depth} "
            f"gold={gold_moment!r} next={recommended_next!r} "
            f"covered={self._probes_covered}"
        )

        remaining_required = [p for p in self._required_probes if p not in self._probes_covered]
        status = (
            f"Covered so far: {self._probes_covered}. "
            f"Still required: {remaining_required or 'none — all required probes done'}."
        )
        if gold_moment:
            status += f" Gold moment noted: '{gold_moment}'."
        return status

    # ─── Tool 3: off-topic strike ─────────────────────────────────────────────

    @llm.function_tool
    async def off_topic_strike(
        self,
        reason: Annotated[str, "Brief description of why this is off-topic."],
    ) -> str:
        """Call when the student deliberately evades the question or goes on
        a completely unrelated tangent. Do NOT call for genuine confusion
        or nervousness — only for deliberate evasion.

        After calling, redirect the student back to the question.
        After 3 strikes the exam should be terminated via end_exam.
        """
        self._off_topic_strikes += 1
        logger.info(f"Off-topic strike #{self._off_topic_strikes}: {reason!r}")

        if self._off_topic_strikes >= 3:
            return (
                f"Strike {self._off_topic_strikes}/3. "
                "MANDATORY: immediately call end_exam with "
                "reason='repeated_off_topic' and speak a firm closing line before ending."
            )
        return (
            f"Strike {self._off_topic_strikes}/3. "
            f"Redirect the student back to the question."
        )

    # ─── Tool 4: end exam ─────────────────────────────────────────────────────

    @llm.function_tool
    async def end_exam(
        self,
        reason: Annotated[
            str,
            "Why the exam is ending: 'completed', 'student_request', "
            "'time_up', or 'repeated_off_topic'.",
        ],
        closing_words: Annotated[
            str,
            "The exact sentence(s) to speak to close the session warmly. "
            "For student_request: acknowledge their wish and thank them. "
            "For completed: congratulate and thank them. "
            "For repeated_off_topic: firm but respectful close. "
            "Keep it to 1-2 sentences.",
        ],
    ) -> str:
        """Call when:
        - All required probes are covered and you have wrapped up naturally (reason='completed')
        - The student explicitly asks to stop / end / finish (reason='student_request')
        - The time budget is exhausted (reason='time_up')
        - 3 off-topic strikes have been reached (reason='repeated_off_topic')

        IMPORTANT: speak the closing_words to the student BEFORE this tool
        fires — the session will close after a short delay.
        Do NOT call this more than once.
        """
        if self._exam_ended:
            return "Exam already ended."
        self._exam_ended = True
        logger.info(f"Exam ending: reason={reason!r}, closing={closing_words!r}")

        # Give TTS a moment to finish speaking the closing line before we drop.
        async def _delayed_shutdown():
            # Give TTS ~3s to finish speaking the closing words, then signal
            # the browser before dropping the room so it can trigger the
            # upload+grade flow on the client side.
            await asyncio.sleep(3.0)
            try:
                import json as _json
                payload = _json.dumps({
                    "type": "exam_ended",
                    "reason": reason,
                }).encode()
                await self._ctx.room.local_participant.publish_data(
                    payload,
                    reliable=True,
                )
                logger.info("Sent exam_ended data message to browser.")
            except Exception as e:
                logger.warning(f"Failed to send exam_ended data message: {e}")
            await asyncio.sleep(1.0)   # small buffer after message before drop
            logger.info("Shutting down session after closing words.")
            self._ctx.shutdown(reason=f"exam_ended:{reason}")

        asyncio.create_task(_delayed_shutdown())
        return f"Session closing in ~4 seconds. Reason: {reason}."


def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load()
    # Warm heavy, synchronous work off the hot path. Done lazily on the first
    # turn these each block the agent event loop for seconds (SSL context ~4s,
    # PIL ~1s), delaying the examiner's audio and the first video frame.
    try:
        # Build + cache our scoped Gemini SSL context now, off the hot path. It
        # is handed to the Gemini Live client via http_options (see ExamAgent),
        # which then skips its own create_default_context() on every reconnect
        # (3-8s each on this box).
        _gemini_ssl_ctx()
        import PIL.Image
        PIL.Image.preinit()
        # Warm numpy.fft — the transcription speaking-rate step runs an FFT on
        # the event loop; the first import alone stalled it ~0.5s.
        import numpy as _np
        _np.fft.rfft(_np.zeros(8))
    except Exception:
        pass


async def entrypoint(ctx: JobContext):
    logger.info(f"AssessVox video agent joining room: {ctx.room.name}")

    for attempt in range(1, 4):
        try:
            await ctx.connect()
            break
        except ConnectError as e:
            if attempt == 3:
                logger.error(f"Failed to connect after 3 attempts: {e}")
                raise
            wait = attempt * 2
            logger.warning(f"Connect failed (attempt {attempt}/3), retrying in {wait}s: {e}")
            await asyncio.sleep(wait)

    spec: dict | None = None
    md = (ctx.room.metadata or "").strip()
    if md:
        try:
            spec = json.loads(md)
            logger.info(
                f"Loaded exam spec for room {ctx.room.name}: "
                f"question={spec.get('exam_question', '')!r}, "
                f"probes={len(spec.get('probe_pool', []))}"
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Room metadata is not valid JSON: {e!r}; using fallback prompt")
    else:
        logger.info("Room metadata empty; using fallback prompt")

    vad = ctx.proc.userdata.get("vad") if hasattr(ctx, "proc") else None
    if vad is None:
        vad = silero.VAD.load()

    # Keep the model's view of the camera CURRENT. The default sampler sends
    # video at 1 fps while speaking and only 0.3 fps while silent (a frame every
    # ~3s), so when the student silently holds up work the model reacts to a
    # frame from seconds ago ("I can see the page" after it's gone). Sample
    # faster — especially while silent, since students show work without talking.
    # Each sampled frame is JPEG-encoded synchronously on the audio event loop,
    # so higher fps = more audio stutter. 1.5/1.0 keeps the view fresh (vs the
    # 1.0/0.3 default) without starving audio. Tune via the env vars.
    # Keep fps modest: every sampled frame is resized+encoded on the audio loop.
    # Combined with the small camera resolution (480x360 from the browser) and
    # 512px encode, this keeps the per-frame cost under the block threshold.
    video_sampler = VoiceActivityVideoSampler(
        speaking_fps=float(os.getenv("GEMINI_LIVE_SPEAKING_FPS", "1.0")),
        silent_fps=float(os.getenv("GEMINI_LIVE_SILENT_FPS", "0.5")),
    )
    session = AgentSession(vad=vad, video_sampler=video_sampler)
    agent = ExamAgent(spec=spec, ctx=ctx)

    await session.start(
        room=ctx.room,
        agent=agent,
        # Split input/output options (NOT the unified room_options) so that
        # sync_transcription=False is actually honored. The unified form ignored
        # it, leaving the transcript synchronizer's speaking-rate FFT running on
        # the audio event loop (~1-1.7s freezes → the voice breaks). We only need
        # the transcript text, not millisecond audio alignment.
        room_input_options=room_io.RoomInputOptions(
            video_enabled=True,
            audio_enabled=True,
        ),
        room_output_options=room_io.RoomOutputOptions(
            audio_enabled=True,
            transcription_enabled=True,
            sync_transcription=False,
        ),
    )

    # Forward each COMPLETED conversation turn to the AssessVox server, so the UI
    # shows the full, correctly-ordered transcript. The browser's live
    # transcription stream can drop the start of a turn (the greeting showed only
    # its question tail); these committed items carry the whole turn.
    _sid = (spec or {}).get("session_id")
    _server_url = os.getenv("ASSESSVOX_SERVER_URL", "http://127.0.0.1:8000")
    if _sid:
        import httpx as _httpx

        async def _post_turn(role: str, text: str) -> None:
            try:
                async with _httpx.AsyncClient(timeout=5) as _c:
                    await _c.post(
                        f"{_server_url}/api/sessions/{_sid}/video-turn",
                        json={"role": role, "text": text},
                    )
            except Exception:
                pass

        @session.on("conversation_item_added")
        def _forward_turn(ev) -> None:
            try:
                role = getattr(ev.item, "role", "") or ""
                text = (ev.item.text_content or "").strip()
            except Exception:
                return
            if text and role in ("assistant", "user"):
                asyncio.create_task(_post_turn(role, text))

    # The examiner opens the conversation. Gemini Live otherwise waits silently
    # for the student to speak first, so neither side talks and the exam never
    # starts ("conversation not being established"). Trigger an opening turn.
    #
    # Only restate the exam task when we actually have the exam details (spec).
    # With no spec, the worker was started without an assessment selected, so
    # inventing a task would mislead the student — greet and ask which
    # assessment they want instead.
    if spec:
        opening = (
            "Greet the student warmly by name, say you can see them on camera, "
            "briefly restate the exam task in one sentence, and invite them to "
            "begin. Say this ONCE, in two or three sentences, then stop and wait "
            "for their response. Do NOT repeat yourself or re-ask. Do NOT claim "
            "to see any diagram, apparatus, or work yet — they haven't shown it."
        )
    else:
        opening = (
            "Greet the student warmly, say you can see them on camera, and ask "
            "which assessment they would like to do today. Say this ONCE, in one "
            "or two sentences, then stop and wait for their answer. Do NOT invent "
            "or assume an exam task, and do NOT claim to see any diagram or work "
            "yet — they haven't shown it."
        )
    await session.generate_reply(instructions=opening)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        num_idle_processes=int(os.getenv("ASSESSVOX_LK_IDLE_PROCESSES", "1")),
    ))
