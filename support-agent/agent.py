"""LiveKit worker for AssessVox video-mode oral exams.

Runs as a long-lived worker that LiveKit Cloud dispatches into any room
named `assessvox-*`. For each room, it reads the exam spec from
`ctx.room.metadata` (written by AssessVox when the room was created),
builds a dynamic system prompt, and runs a Gemini Live realtime session
with `video_input=True` so the model can see the student.

Run with:
    uv run python support-agent/agent.py dev

The `dev` subcommand registers the worker with LiveKit Cloud and waits
for dispatched rooms.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, JobContext, room_io
from livekit.rtc.room import ConnectError
from livekit.plugins import google, silero

# Allow `python support-agent/agent.py` from the repo root by adding our
# own directory to the path so `prompts` resolves.
sys.path.insert(0, str(Path(__file__).parent))
from prompts import INTERVIEW_INSTRUCTIONS, build_exam_instructions

# Load env from AssessVox root first (where LIVEKIT + GEMINI_API_KEY live),
# then from our own .env.local for backwards compat.
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env.local")

logger = logging.getLogger("assessvox-video-agent")
logger.setLevel(logging.INFO)


def _build_agent_for_spec(spec: dict | None) -> Agent:
    instructions = build_exam_instructions(spec) if spec else INTERVIEW_INSTRUCTIONS
    return Agent(
        instructions=instructions,
        llm=google.realtime.RealtimeModel(
            # Per ai.google.dev/gemini-api/docs/models (June 2026), the public
            # Live API models are gemini-3.1-flash-live-preview and
            # gemini-2.5-flash-native-audio-preview-12-2025. The 3.1 preview is
            # the canonical one. Override via GEMINI_LIVE_MODEL env var.
            model=os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview"),
            voice=os.getenv("GEMINI_LIVE_VOICE", "Puck"),
            temperature=0.6,
            api_key=os.getenv("GEMINI_API_KEY"),
        ),
    )


def prewarm(proc):
    """Called once per warmed child process BEFORE any job is assigned.

    We pre-load Silero VAD here so it's ready in memory when an exam
    session lands. The model is cached on ``proc.userdata`` so the
    entrypoint can grab it instead of re-loading.

    With ``num_idle_processes >= 1`` the LiveKit agents framework keeps
    that many already-prewarmed children sitting in a pool — when a room
    is dispatched, we pop a hot child instead of forking + importing for
    ~2.8 seconds. Cuts the cold start from ~3-5s down to ~50ms.
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"AssessVox video agent joining room: {ctx.room.name}")

    # Connect with retry — LiveKit Cloud occasionally rejects the first SDP.
    for attempt in range(1, 4):
        try:
            await ctx.connect()
            break
        except ConnectError as e:
            if attempt == 3:
                logger.error(f"Failed to connect after 3 attempts: {e}")
                raise
            import asyncio
            wait = attempt * 2
            logger.warning(f"Connect failed (attempt {attempt}/3), retrying in {wait}s: {e}")
            await asyncio.sleep(wait)

    # Read the exam spec from room metadata. AssessVox writes JSON there
    # when it creates the room.
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

    # Reuse the pre-warmed VAD instance if available (loaded in prewarm()).
    # Falls back to a fresh load if we somehow ended up in a non-prewarmed
    # process (e.g. running under `dev` without a pool).
    vad = ctx.proc.userdata.get("vad") if hasattr(ctx, "proc") else None
    if vad is None:
        vad = silero.VAD.load()
    session = AgentSession(vad=vad)

    await session.start(
        room=ctx.room,
        agent=_build_agent_for_spec(spec),
        room_options=room_io.RoomOptions(
            video_input=True,
            audio_input=True,
            audio_output=True,
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        # Keep one child process already forked + VAD-loaded so the next
        # room dispatch skips the ~2.8s cold start. Tune up if you expect
        # concurrent sessions; each warmed process costs ~150 MB RAM.
        num_idle_processes=int(os.getenv("ASSESSVOX_LK_IDLE_PROCESSES", "1")),
    ))
