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

# ── Fast TLS: memoise the SSL context ────────────────────────────────────────
# On this Windows / Python 3.14 box, ssl.create_default_context() loads the
# certifi CA bundle in 3-8 SECONDS. google-genai rebuilds that context on every
# Gemini Live (re)connection, and each rebuild FREEZES the agent event loop for
# seconds — which lags the conversation AND trips the WebSocket heartbeat,
# causing more drops, more reconnects, more rebuilds (we saw 28 of them, up to
# 7.7s each). Memoise so the context is built at most once per arg-set and
# reused instantly on every reconnect.
_SSL_CTX_CACHE: dict = {}
_orig_create_default_context = _ssl.create_default_context


def _memoised_create_default_context(purpose=_ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    key = (purpose, cafile, capath, cadata if isinstance(cadata, (str, bytes, type(None))) else id(cadata))
    ctx = _SSL_CTX_CACHE.get(key)
    if ctx is None:
        ctx = _orig_create_default_context(purpose, cafile=cafile, capath=capath, cadata=cadata)
        _SSL_CTX_CACHE[key] = ctx
    return ctx


_ssl.create_default_context = _memoised_create_default_context

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
        super().__init__(
            instructions=instructions,
            llm=google.realtime.RealtimeModel(
                model=os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview"),
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
            ),
        )
        self._spec = spec or {}
        self._ctx = ctx

        # Mutable exam state
        self._student_name: str = (spec or {}).get("student_name", "") or "the student"
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
        # Build + cache the SSL context now (memoised at import). google-genai
        # reuses it on every Gemini reconnect instead of rebuilding (3-8s each).
        _ssl.create_default_context()
        try:
            import certifi
            _ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
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

    # The examiner opens the conversation. Gemini Live otherwise waits silently
    # for the student to speak first, so neither side talks and the exam never
    # starts ("conversation not being established"). Trigger an opening turn.
    await session.generate_reply(
        instructions=(
            "Greet the student warmly by name, say you can see them on camera, "
            "briefly restate the exam task in one sentence, and invite them to "
            "begin. Say this ONCE, in two or three sentences, then stop and wait "
            "for their response. Do NOT repeat yourself or re-ask. Do NOT claim "
            "to see any diagram, apparatus, or work yet — they haven't shown it."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        num_idle_processes=int(os.getenv("ASSESSVOX_LK_IDLE_PROCESSES", "1")),
    ))
