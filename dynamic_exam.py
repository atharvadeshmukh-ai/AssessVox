#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dynamic IB MYP Oral Exam Bot with Grading.

End-to-end pipeline:
  1. Teacher question (env var EXAM_TOPIC) → flow_planner.generate_flow
  2. Validate the generated FlowSpec (graph-level checks). Retry once on
     failure with the validator errors appended to the prompt.
  3. Persist the FlowSpec to flows/flow_<session_id>.json for reproducibility.
  4. Pipecat conversation: dynamic probe routing using the primitive catalog.
  5. Record audio → recordings/exam_<session_id>.wav
  6. Grade audio + transcript → reports/report_<session_id>.json
     (same 7-dimension holistic format as teacher2.py)

Required env vars:
  EXAM_TOPIC                 — the teacher's question (default: gene editing)
  EXAM_DURATION_MINUTES      — target duration (default: 6)
  GOOGLE_API_KEY             — used for planning, conversation, grading
  ELEVENLABS_API_KEY         — STT (ElevenLabs Scribe v2 realtime)
  CARTESIA_API_KEY           — TTS
  GEMINI_GRADER_MODEL        — optional, default 'gemini-3.5-flash'

Run with:
    uv run dynamic_exam.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError

print("Starting Dynamic IB MYP Oral Examiner bot (with grading)...")
print("Loading models and imports (first run may take ~20 seconds)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
logger.info("Local Smart Turn Analyzer V3 loaded")

logger.info("Loading Silero VAD...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
logger.info("Silero VAD loaded")

logger.info("Loading pipeline components...")
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
# Kept-but-unused imports — used by the commented-out turn-management config
# in run_bot. Leave imported so re-enabling lines is a no-op edit.
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    FilterIncompleteUserTurnStrategies,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_mute import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
    FunctionCallUserMuteStrategy,
)
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.cartesia.stt import CartesiaSTTService
# Deepgram, AssemblyAI, and xAI are lazy-imported inside _build_stt to avoid
# hard dependency on their SDK extras when the user picks a different provider.
from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService


class AssessVoxGoogleLLMService(GoogleLLMService):
    """GoogleLLMService whose TTFB metric is deferred to the first non-thought token.

    Gemini 2.5+ models emit thought chunks before the actual answer when
    thinking is enabled. The base class stops the TTFB clock on the very first
    chunk regardless of type, which makes latency look ~20x better than what
    the student actually experiences. This subclass gates stop_ttfb_metrics()
    on the first real content token — a no-op when thinking is disabled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ttft_armed = False

    async def _process_context(self, context):
        self._ttft_armed = False
        await super()._process_context(context)

    async def stop_ttfb_metrics(self, *, end_time: float | None = None):
        if self._ttft_armed:
            await super().stop_ttfb_metrics(end_time=end_time)

    async def push_frame(self, frame, direction=None):
        from pipecat.frames.frames import LLMTextFrame
        if not self._ttft_armed and isinstance(frame, LLMTextFrame) and frame.text:
            self._ttft_armed = True
        if direction is not None:
            await super().push_frame(frame, direction)
        else:
            await super().push_frame(frame)

from websockets.asyncio.client import connect as _websocket_connect



class FallbackCartesiaTTSService(CartesiaTTSService):
    """Cartesia TTS that rotates through a list of API keys when one is
    rejected (HTTP 402 credits-exhausted, 401 invalid, 403 forbidden).

    The first key in the list is used by default. If the WebSocket
    handshake fails with a key-rejection error, the next key is tried.
    Once a key works, it sticks for the rest of the session — the index
    only advances, never resets.
    """

    def __init__(self, *, api_keys: list[str], voice_id: str, **kwargs):
        if not api_keys:
            raise ValueError("FallbackCartesiaTTSService needs at least one api_key")
        super().__init__(
            api_key=api_keys[0],
            settings=CartesiaTTSService.Settings(voice=voice_id),
            **kwargs,
        )
        self._api_keys: list[str] = list(api_keys)
        self._key_index: int = 0

    @staticmethod
    def _is_rotatable_error(msg: str) -> bool:
        # Rotate on key-rejection (402/401/403) AND on any transient connection
        # failure — timed-out handshakes, refused connections, DNS hiccups, etc.
        # These often mean "this key/region is down right now," and the next key
        # may go through a different gateway and succeed. Worst case we waste a
        # second retry; best case we recover a session that would otherwise be
        # silent.
        msg_l = msg.lower()
        if any(token in msg for token in ("HTTP 402", "HTTP 401", "HTTP 403")):
            return True
        return any(token in msg_l for token in (
            "timed out", "timeout",
            "handshake",
            "connection refused", "connection reset", "connection closed",
            "name or service not known", "temporary failure in name resolution",
            "no route to host",
        ))

    @staticmethod
    def _mask(key: str) -> str:
        return key[:6] + "..." + key[-4:] if len(key) > 10 else "(short)"

    async def _connect_websocket(self):
        last_err: Exception | None = None
        attempts = 0
        while self._key_index < len(self._api_keys):
            self._api_key = self._api_keys[self._key_index]
            attempts += 1
            try:
                if self._websocket and self._websocket.state.name == "OPEN":
                    return
                logger.debug(
                    f"Connecting to Cartesia TTS with key #{self._key_index + 1}/"
                    f"{len(self._api_keys)} ({self._mask(self._api_key)})"
                )
                self._websocket = await _websocket_connect(
                    f"{self._url}?api_key={self._api_key}&cartesia_version={self._cartesia_version}"
                )
                await self._call_event_handler("on_connected")
                if self._key_index > 0:
                    logger.warning(
                        f"Cartesia fell back to key #{self._key_index + 1} "
                        f"({self._mask(self._api_key)})"
                    )
                return
            except Exception as e:
                last_err = e
                msg = str(e)
                if self._is_rotatable_error(msg):
                    logger.warning(
                        f"Cartesia key #{self._key_index + 1} "
                        f"({self._mask(self._api_key)}) failed ({msg[:120]}) — "
                        f"trying next key"
                    )
                    self._key_index += 1
                    self._websocket = None
                    continue
                logger.error(
                    f"Cartesia non-rotatable error on key #{self._key_index + 1}: {msg[:200]}"
                )
                # Still advance so the giving-up log shows the right count.
                self._key_index += 1
                break
        logger.error(
            f"Cartesia TTS giving up after trying {attempts} key(s); "
            f"last error: {last_err}"
        )
        await self.push_error(
            error_msg=f"All Cartesia keys exhausted: {last_err}", exception=last_err
        )
        self._websocket = None
        await self._call_event_handler("on_connection_error", f"{last_err}")


def _load_cartesia_keys() -> list[str]:
    """Read the primary + up-to-two-fallback Cartesia keys from env, in order."""
    candidates = [
        os.getenv("CARTESIA_API_KEY"),
        os.getenv("CARTESIA_API_KEY_2"),
        os.getenv("CARTESIA_API_KEY_3"),
    ]
    return [k for k in candidates if k and k.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# KEPT-BUT-UNUSED in default config (commented out of the active pipeline /
# STT factory). Each class below was added to chase a specific symptom over
# several sessions; the canonical Pipecat-flows config in run_bot doesn't
# need them. Leave the code so re-enabling is a 1-line edit if a symptom
# returns. See run_bot for the commented config blocks.
# ═══════════════════════════════════════════════════════════════════════════


class SmartTurnElevenLabsSTT(ElevenLabsRealtimeSTTService):
    """[Active only when stt_provider='elevenlabs'] ElevenLabs Realtime STT
    with a session-started gate for MANUAL commits.

    ElevenLabs' WebSocket is TCP-open ~0.7 s before it sends `session_started`.
    If a VADUserStoppedSpeakingFrame arrives in that window (e.g. user speaks
    immediately), the commit is sent before ElevenLabs is ready and gets dropped,
    producing an empty transcript.

    Fix: buffer the commit flag at startup. When `session_started` arrives, flush
    any pending commit. After that the gate is open and commits flow normally.
    """

    # ElevenLabs rejects commits with < 0.3 s of uncommitted audio. Track when
    # the bot started speaking so we can skip the echo flush if the bot's turn
    # was too short to have accumulated enough audio to commit.
    BOT_SPEECH_MIN_FOR_FLUSH_S = 0.35
    # Minimum elapsed time since the previous commit before sending a new one.
    # Same 300 ms ElevenLabs requirement + 50 ms safety margin.
    COMMIT_COOLDOWN_S = 0.35
    # How quickly to retry connect() after an unexpected mid-session disconnect.
    RECONNECT_BACKOFF_S = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._session_ready: bool = False
        self._pending_commit: bool = False
        self._bot_was_speaking: bool = False
        self._bot_started_at: float = 0.0
        # Timestamp until which the next committed_transcript should be discarded
        # (echo flush window). 0.0 means no discard pending.
        self._discard_echo_until: float = 0.0
        # Time of the last commit sent — used to gate watchdog commits so we
        # don't trip ElevenLabs's "0.28s of uncommitted audio" rejection.
        self._last_commit_at: float = 0.0
        # Set to True between StartFrame and EndFrame/CancelFrame. When True
        # and the websocket dies (server-initiated close), we proactively
        # reconnect rather than wait for the next inbound audio frame to
        # noticed it. Prevents the dead-air gap that ended last session.
        self._exam_active: bool = False
        self._reconnect_watcher_task: asyncio.Task | None = None

    async def _send_flush_commit(self, reason: str):
        """Send a zero-audio commit to flush ElevenLabs' buffer.

        Marks a 2-second window during which the next committed_transcript is
        silently dropped — it will contain bot audio (echo), not user speech.
        Using a timed window rather than a boolean ensures the flag auto-expires
        even if ElevenLabs never responds to an empty/silent flush.
        """
        if self._websocket:
            try:
                await self._websocket.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",
                    "commit": True,
                    "sample_rate": self.sample_rate,
                }))
                self._discard_echo_until = time.monotonic() + 0.5
                self._last_commit_at = time.monotonic()
                logger.debug(f"SmartTurnElevenLabsSTT: echo-flush sent ({reason}), discard window open for 0.5s")
            except Exception as e:
                logger.warning(f"SmartTurnElevenLabsSTT: echo-flush failed: {e}")

    async def _process_response(self, data: dict):
        mtype = data.get("message_type", "")

        if mtype == "session_started":
            self._session_ready = True
            logger.debug("SmartTurnElevenLabsSTT: session_started — commit gate open")
            if self._pending_commit:
                self._pending_commit = False
                if self._websocket:
                    try:
                        await self._websocket.send(json.dumps({
                            "message_type": "input_audio_chunk",
                            "audio_base_64": "",
                            "commit": True,
                            "sample_rate": self.sample_rate,
                        }))
                        logger.debug("SmartTurnElevenLabsSTT: flushed deferred commit")
                    except Exception as e:
                        logger.warning(f"SmartTurnElevenLabsSTT: deferred commit failed: {e}")

        elif mtype in ("committed_transcript", "committed_transcript_with_timestamps"):
            if self._discard_echo_until and time.monotonic() < self._discard_echo_until:
                self._discard_echo_until = 0.0  # consume the window
                logger.debug("SmartTurnElevenLabsSTT: discarded bot-echo flush transcript (within window)")
                return  # drop it — don't let parent emit a TranscriptionFrame

        await super()._process_response(data)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_was_speaking = True
            self._bot_started_at = time.monotonic()

        elif isinstance(frame, BotStoppedSpeakingFrame) and self._bot_was_speaking:
            # Bot just stopped (possibly interrupted by user barge-in).
            # ElevenLabs still has bot audio in its buffer. Flush it now so the
            # user's subsequent speech starts from a clean slate.
            spoke_for = time.monotonic() - self._bot_started_at
            self._bot_was_speaking = False
            if self._session_ready and spoke_for >= self.BOT_SPEECH_MIN_FOR_FLUSH_S:
                await self._send_flush_commit(f"bot stopped speaking ({spoke_for:.2f}s)")
            elif self._session_ready:
                logger.debug(
                    f"SmartTurnElevenLabsSTT: skipping echo-flush — bot only "
                    f"spoke {spoke_for:.2f}s (< {self.BOT_SPEECH_MIN_FOR_FLUSH_S}s, "
                    "below ElevenLabs commit minimum)"
                )

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # Record turn start regardless of session readiness — used by latency tracker
            _TURN_TIMING.clear()
            _TURN_TIMING["vad_stop_at"] = time.perf_counter()
            # The parent's process_frame will send a commit on this frame
            # (MANUAL strategy). Record the time so the watchdog branch can
            # gate itself against the cooldown.
            if self._commit_strategy == CommitStrategy.MANUAL and self._session_ready:
                self._last_commit_at = time.monotonic()

        # Belt-and-braces commit: if the user turn closes WITHOUT a VAD stop
        # (e.g. the pipecat watchdog firing `strategy: None` after 5s, or
        # MinWords opening a turn from a single interim word that VAD never
        # registered), ElevenLabs never gets `VADUserStoppedSpeakingFrame` and
        # the audio in its buffer never commits → no final transcript → bot
        # stuck. Send a manual commit on the generic UserStoppedSpeakingFrame
        # too. Safe: if a VAD-driven commit already fired, this is a no-op on
        # an empty buffer.
        #
        # Cooldown guard: ElevenLabs rejects commits when there's < 0.3s of new
        # uncommitted audio since the last one. Skip if we just committed —
        # this is what's been firing the "0.28s" ErrorFrames every session.
        if (
            isinstance(frame, UserStoppedSpeakingFrame)
            and not isinstance(frame, VADUserStoppedSpeakingFrame)
            and self._commit_strategy == CommitStrategy.MANUAL
            and self._session_ready
            and self._websocket
        ):
            since_last_commit = time.monotonic() - self._last_commit_at
            if since_last_commit < self.COMMIT_COOLDOWN_S:
                logger.debug(
                    f"SmartTurnElevenLabsSTT: skipping watchdog commit — only "
                    f"{since_last_commit:.2f}s since last commit "
                    f"(< {self.COMMIT_COOLDOWN_S}s, would be rejected)"
                )
            else:
                try:
                    await self._websocket.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": "",
                        "commit": True,
                        "sample_rate": self.sample_rate,
                    }))
                    self._last_commit_at = time.monotonic()
                    logger.debug(
                        "SmartTurnElevenLabsSTT: sent watchdog commit "
                        "(UserStoppedSpeakingFrame without preceding VAD stop)"
                    )
                except Exception as e:
                    logger.warning(f"SmartTurnElevenLabsSTT: watchdog commit failed: {e}")

        if (
            isinstance(frame, VADUserStoppedSpeakingFrame)
            and self._commit_strategy == CommitStrategy.MANUAL
            and not self._session_ready
        ):
            # Session not yet confirmed — defer the commit, still pass frame downstream.
            # Temporarily mask commit_strategy so the parent skips its commit branch.
            self._commit_strategy = CommitStrategy.VAD
            try:
                await super().process_frame(frame, direction)
            finally:
                self._commit_strategy = CommitStrategy.MANUAL
            self._pending_commit = True
            logger.debug("SmartTurnElevenLabsSTT: deferred commit (session_started not yet received)")
        else:
            await super().process_frame(frame, direction)

    async def _connect(self):
        self._session_ready = False
        self._pending_commit = False
        self._bot_was_speaking = False
        self._bot_started_at = 0.0
        self._discard_echo_until = 0.0
        self._last_commit_at = 0.0
        await super()._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._exam_active = True
        logger.info("SmartTurnElevenLabsSTT: exam active — reconnect-on-drop enabled")

    async def stop(self, frame: EndFrame):
        self._exam_active = False
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._exam_active = False
        await super().cancel(frame)

    async def _disconnect_websocket(self):
        """Override to detect unexpected mid-session disconnects.

        ElevenLabs closes the websocket on its own after extended idle periods
        or repeated bad commits. The base class accepts the close and does
        nothing — leaving the pipeline in a dead-air state until the student
        gives up and refreshes. We instead schedule a fresh `_connect()` so
        the session keeps running.
        """
        # Capture state BEFORE the base class clears self._websocket.
        was_active = self._exam_active and not self._reconnect_in_progress
        await super()._disconnect_websocket()

        if was_active:
            logger.warning(
                "SmartTurnElevenLabsSTT: ElevenLabs websocket dropped mid-exam — "
                f"scheduling reconnect in {self.RECONNECT_BACKOFF_S}s"
            )
            self._reconnect_watcher_task = self.create_task(
                self._auto_reconnect(), name="elevenlabs-auto-reconnect"
            )

    async def _auto_reconnect(self):
        """Background task: sleep briefly then call _connect() to revive STT."""
        try:
            await asyncio.sleep(self.RECONNECT_BACKOFF_S)
            if not self._exam_active:
                logger.info("SmartTurnElevenLabsSTT: auto-reconnect skipped — exam ended")
                return
            try:
                await self._connect()
                logger.info("SmartTurnElevenLabsSTT: auto-reconnect succeeded")
            except Exception as e:
                logger.error(f"SmartTurnElevenLabsSTT: auto-reconnect failed: {e}")
        finally:
            self._reconnect_watcher_task = None


# ─── Per-turn latency tracking ────────────────────────────────────────────────
# Shared mutable dict; SmartTurnElevenLabsSTT and TurnLatencyTracker both write here.
_TURN_TIMING: dict = {}
_LAST_TURN_LATENCY: dict | None = None


def get_last_turn_latency() -> dict | None:
    return _LAST_TURN_LATENCY


class TurnLatencyTracker(FrameProcessor):
    """[UNUSED — not in active pipeline; was emitting no logs in practice]
    Tracks per-turn latency and logs a breakdown after each bot response starts.

    Timing model:
      vad_stop_at        — user stopped speaking (set by SmartTurnElevenLabsSTT)
      transcript_at      — committed transcript received from STT
      llm_start_at       — LLM started generating (LLMFullResponseStartFrame)
      bot_started_at     — bot started speaking (BotStartedSpeakingFrame)

    Total perceived latency = bot_started_at - vad_stop_at.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        global _LAST_TURN_LATENCY

        if isinstance(frame, TranscriptionFrame) and "vad_stop_at" in _TURN_TIMING:
            _TURN_TIMING["transcript_at"] = time.perf_counter()

        elif isinstance(frame, LLMFullResponseStartFrame) and "vad_stop_at" in _TURN_TIMING:
            _TURN_TIMING["llm_start_at"] = time.perf_counter()

        elif isinstance(frame, BotStartedSpeakingFrame) and "vad_stop_at" in _TURN_TIMING:
            now = time.perf_counter()
            _TURN_TIMING["bot_started_at"] = now

            t0 = _TURN_TIMING["vad_stop_at"]
            stt_ms   = round((_TURN_TIMING.get("transcript_at", now) - t0) * 1000)
            llm_ms   = round((_TURN_TIMING.get("llm_start_at",  now) - _TURN_TIMING.get("transcript_at", t0)) * 1000)
            tts_ms   = round((now - _TURN_TIMING.get("llm_start_at", _TURN_TIMING.get("transcript_at", t0))) * 1000)
            total_ms = round((now - t0) * 1000)

            _LAST_TURN_LATENCY = {
                "total_ms": total_ms,
                "stt_ms":   stt_ms,
                "llm_ms":   llm_ms,
                "tts_ms":   tts_ms,
            }
            logger.info(
                f"[LATENCY] total={total_ms}ms  "
                f"stt={stt_ms}ms  llm={llm_ms}ms  tts={tts_ms}ms"
            )

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class BargeinGuard(FrameProcessor):
    """[UNUSED — not in active pipeline; problem now solved by
    filter_incomplete_user_turns gating bot responses on the ✓ marker]
    Suppress LLM re-runs from in-flight function calls when the user barges in.

    Why: `FunctionCallResultFrame` is an `UninterruptibleFrame` — it survives
    `InterruptionFrame`. If `advance_from_probe` (or any flow function) finishes
    in the same few ms the user starts barging in, its result reaches the
    assistant aggregator before `UserStartedSpeakingFrame` does, so the aggregator's
    `not self._user_speaking` guard fails. The aggregator then pushes a fresh
    context upstream and the bot starts speaking again right over the user.

    Fix: place this processor immediately upstream of `context_aggregator.assistant()`.
    On any sign of barge-in (`InterruptionFrame`, `VADUserStartedSpeakingFrame`,
    or `UserStartedSpeakingFrame`) open a short window. While the window is open,
    rewrite any passing `FunctionCallResultFrame` to `run_llm=False` so the
    aggregator updates context silently without re-triggering the LLM.
    The window closes when the user stops speaking.
    """

    BARGEIN_WINDOW_S = 3.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._suppress_until: float = 0.0

    def _open_window(self, reason: str):
        self._suppress_until = time.monotonic() + self.BARGEIN_WINDOW_S
        logger.debug(f"BargeinGuard: window opened ({reason}) for {self.BARGEIN_WINDOW_S}s")

    def _close_window(self, reason: str):
        if self._suppress_until:
            logger.debug(f"BargeinGuard: window closed ({reason})")
        self._suppress_until = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (InterruptionFrame, VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            self._open_window(type(frame).__name__)
        elif isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            self._close_window(type(frame).__name__)
        elif isinstance(frame, FunctionCallResultFrame):
            if self._suppress_until and time.monotonic() < self._suppress_until:
                if frame.run_llm is not False:
                    logger.info(
                        f"BargeinGuard: suppressing LLM re-run for "
                        f"{frame.function_name}:{frame.tool_call_id} (barge-in active)"
                    )
                    frame.run_llm = False

        await self.push_frame(frame, direction)


class PlaceholderFilter(FrameProcessor):
    """[UNUSED — not in active pipeline; the "Empty Message" symptom was a
    Flash-Lite quirk, doesn't recur on gemini-3.5-flash]
    Drop bot text frames whose content is a known LLM placeholder string.

    Why: Gemini occasionally interprets "wait silently after a function call"
    instructions by emitting literal placeholder text like "Empty Message",
    "(silent)", or "..." as the assistant turn. Without this filter, that
    text flows through TTS — so the bot literally speaks "Empty Message" to
    the student — and then lands in the transcript snapshot / grading
    report.

    Placement: sit BEFORE the TTS service in the pipeline, after the LLM.
    Drops the offending TextFrame/TTSTextFrame outright so it never reaches
    Cartesia or any downstream transcript collector.
    """

    _PLACEHOLDERS: set[str] = {
        "empty message", "empty_message", "(empty)", "[empty]",
        "no response", "no_response", "(no response)", "[no response]",
        "(silent)", "[silent]", "silent", "silence",
        "(wait)", "[wait]", "wait",
        "...", "…", ".", "...",
        "(pass)", "[pass]", "pass",
        "(noop)", "[noop]", "noop", "no-op",
        "null", "none",
    }

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (TextFrame, TTSTextFrame)):
            text = (getattr(frame, "text", "") or "").strip().lower()
            if text in self._PLACEHOLDERS:
                logger.warning(
                    f"PlaceholderFilter: dropped bot placeholder text "
                    f"{getattr(frame, 'text', '')!r} (would have spoken/logged)"
                )
                return  # do not forward

        await self.push_frame(frame, direction)


class MarkerStripFilter(FrameProcessor):
    """Belt-and-braces strip of ✓ / ○ / ◐ turn-completion markers from text frames.

    Workaround for a confirmed upstream issue where
    ``UserTurnCompletionLLMServiceMixin._push_turn_text`` doesn't reliably
    consume the leading marker on the ``GoogleLLMService`` streaming path
    when ``filter_incomplete_user_turns=True``. Symptom (seen in session
    20260606_032629_5a8f63): bot turns appear in the transcript and TTS
    feed as ``"✓ Hello..."`` or just ``"○"`` / ``"◐"`` on their own.

    Place between the LLM and the TTS service. Removes the marker from
    LLMTextFrame / TTSTextFrame / TextFrame and drops bare-marker frames
    so the TTS never speaks them.

    Idempotent: if the marker has already been stripped upstream, this is
    a no-op. Strips at most one leading marker per frame.
    """

    _MARKERS: tuple[str, ...] = ("✓", "○", "◐")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMTextFrame, TTSTextFrame, TextFrame)):
            text = getattr(frame, "text", "") or ""
            # Only inspect for a marker; do NOT lstrip the frame unconditionally.
            # Streaming providers (Groq/Llama, OpenAI) emit leading-space tokens
            # like " Hello", " world" as natural word boundaries — stripping them
            # collapses the output into "Helloworld...".
            leading_ws_len = len(text) - len(text.lstrip())
            after_ws = text[leading_ws_len:]
            marker_found = False
            for marker in self._MARKERS:
                if after_ws.startswith(marker):
                    # Strip leading whitespace + marker + any whitespace right after it.
                    new_text = after_ws[len(marker):].lstrip()
                    logger.debug(
                        f"MarkerStripFilter: stripped marker from "
                        f"{type(frame).__name__}; before={text!r} after={new_text!r}"
                    )
                    frame.text = new_text
                    marker_found = True
                    break
            # Drop the frame entirely if all that's left is whitespace —
            # this is the "bare ○" / "bare ◐" suppress-only case where
            # the LLM emitted nothing but the marker.
            if marker_found and not (frame.text or "").strip():
                return

        await self.push_frame(frame, direction)


from pipecat_flows import FlowManager, FlowsFunctionSchema, FlowArgs, NodeConfig

from flow_planner import generate_flow
from flow_primitives import ALLOWED_PROBE_TYPES, render_probe_instructions
from flow_spec_schema import FlowSpec, ProbeNode
from flow_validator import FlowValidationError, validate_flow_spec

logger.info("All components loaded successfully!")

load_dotenv(override=True)

# ─── Paths ─────────────────────────────────────────────────────────────────────
RECORDINGS_DIR  = Path(__file__).parent / "recordings"
REPORTS_DIR     = Path(__file__).parent / "reports"
FLOWS_DIR       = Path(__file__).parent / "flows"
LOGS_DIR        = Path(__file__).parent / "logs"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"

# ─── Module-level flow state (populated at startup) ────────────────────────────
_FLOW_SPEC: FlowSpec | None = None
_PROBE_FACTORIES: dict[str, Callable] = {}
_REQUIRED_PROBES: set[str] = set()
_CUSTOM_GRADING_PROMPT: str | None = None

# Set when a session is running, so the server can read the live transcript
_CURRENT_CONTEXT = None
_CURRENT_SESSION_ID: str | None = None


def get_current_transcript() -> list[dict]:
    """Return diarized turns from the active session's LLMContext.
    Empty list if no session is running."""
    if _CURRENT_CONTEXT is None:
        return []
    try:
        return _extract_transcript(_CURRENT_CONTEXT)
    except Exception:
        return []


# Optional teacher-provided student name. When set, run_bot skips the
# STT-driven name-capture greeting entirely.
_PROVIDED_STUDENT_NAME: str | None = None


def set_active_flow(
    flow_spec: FlowSpec,
    custom_grading_prompt: str | None = None,
    student_name: str | None = None,
) -> None:
    """Configure module-level state for a session. Called by the server before
    spawning a bot, or by bot() in CLI mode after generating the flow."""
    global _FLOW_SPEC, _PROBE_FACTORIES, _REQUIRED_PROBES, _CUSTOM_GRADING_PROMPT
    global _PROVIDED_STUDENT_NAME
    _FLOW_SPEC = flow_spec
    _PROBE_FACTORIES = build_probe_factories(flow_spec)
    _REQUIRED_PROBES = set(flow_spec.required_probes)
    _CUSTOM_GRADING_PROMPT = custom_grading_prompt
    _PROVIDED_STUDENT_NAME = (student_name or "").strip() or None

# ─── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a warm, encouraging oral examiner for an IB MYP language class "
    "(students aged 11 to 16). Your role is to facilitate a structured oral discussion. "
    "You MUST ALWAYS call the provided function to advance the conversation. "
    "STRICT RULES — never break these: "
    "1. Do NOT correct grammar, pronunciation, or word choice. "
    "2. Do NOT share your own opinions on the exam topic. "
    "3. Do NOT accept facts as a substitute for opinion — redirect: "
    "   'That is good background — but what is YOUR view on this?' "
    "4. A SHORT ANSWER IS NOT A THIN ANSWER. If the student gives one phrase that "
    "   names a real reason, example, or concept (e.g. 'comfort', 'productivity', "
    "   'social bonding'), ACCEPT IT and advance. Only push for elaboration if "
    "   the answer has zero content (e.g. 'um', 'maybe', 'I guess'). Do NOT "
    "   interrogate every utterance. "
    "5. NEVER repeat the same follow-up phrase twice. NEVER push deeper on the "
    "   same micro-concept the student is struggling with. If you must ask again, "
    "   move SIDEWAYS to a different aspect of the topic — never just rephrase. "
    "6. INTENT DETECTION: distinguish deliberate evasion from genuine confusion. "
    "   Evasion = unrelated jokes/tangents. Confusion = asking to clarify the question. "
    "   Treat confusion generously — help, do not strike. "
    "7. FUNCTION RESULT RULE: after any function call, read the NEW task instructions "
    "   and speak ONLY natural conversational sentences. "
    "   If result has next_action='wait_for_student', return ZERO output — no text, "
    "   no acknowledgement, no placeholder. Do NOT write 'Empty Message', '...', "
    "   '(silent)', '[wait]', 'No response', or any other stand-in. The correct "
    "   response is literally nothing — your assistant turn must be empty. "
    "8. NEVER output raw JSON, status codes, function names, curly braces, or technical content. "
    "9. Call any strike/transition function at most ONCE per student message. "
    "10. Voice conversation only — no markdown, no bullet points, no emojis. "
    "11. Keep your turns short — facilitate, not teach. "
    "12. NEVER invent a topic. Ask ONLY the exact exam question given in the task instructions. "
    "13. EMPATHY ON STRUGGLE: when the student signals difficulty ('I don't know', "
    "    'I'm not sure', 'I forget', 'I can't recall', visible hesitation), your "
    "    FIRST 5–8 words MUST be a warm, brief acknowledgment composed FRESH each "
    "    time. NEVER reuse the same opener you used in your previous turn. Compose "
    "    naturally from any of these styles, mixing them up: "
    "    • normalising: 'No worries', 'Totally fine', 'Don't worry about that', "
    "      'Happens to all of us', 'It's a tricky one'. "
    "    • valuing partial effort: 'Whatever comes to mind works', 'Any small "
    "      thought helps', 'Even half an idea is great'. "
    "    • encouragement: 'You're doing well', 'Good attempt', 'I appreciate "
    "      you thinking it through'. "
    "    BANNED phrasings (you have overused these): "
    "    'There is no perfect answer here', 'That is totally fine — there is no…'. "
    "    NEVER lead with another demand when the student has just signalled struggle. "
    "14. ZERO REPETITION: never start two consecutive turns with the same opener. "
    "    Each new turn should open with a different acknowledgement, transition, "
    "    or question lead-in than your last one. Variety is mandatory. "
    "15. THINKING ALOUD: if the student speaks in short fragments across multiple "
    "    utterances, they are thinking aloud. Do NOT interrupt to demand depth. "
    "    Let them finish the thought. You may quietly say 'take your time' once "
    "    if there is a long pause, but otherwise wait."
)
ROLE_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]

ADVANCE_PROBE_SCHEMA = {
    "depth": {
        "type": "string",
        "enum": ["surface", "developed"],
        "description": "Depth of the student's response",
    },
    "topics_raised": {
        "type": "array", "items": {"type": "string"},
        "description": "Themes the student mentioned in this response",
    },
    "topics_missing": {
        "type": "array", "items": {"type": "string"},
        "description": "Important themes from the exam NOT yet discussed",
    },
    "recommended_next": {
        "type": "string",
        "description": "Probe ID from the available list you recommend as the best next step",
    },
    "gold_moment": {
        "type": "string",
        "description": (
            "A specific, vivid idea the student said worth following up. "
            "Empty string if nothing stands out."
        ),
    },
    "student_struggling": {
        "type": "boolean",
        "description": "True if the student needed significant scaffolding",
    },
    "student_consented_to_advance": {
        "type": "boolean",
        "description": (
            "True if the student explicitly said they want to move on, skip, "
            "give up on this probe, or accepted an offer to switch questions. "
            "False otherwise. Counts toward a per-exam skip limit."
        ),
    },
    "vague_terms": {
        "type": "array", "items": {"type": "string"},
        "description": "Imprecise words or phrases the student used",
    },
}


# ─── Off-topic guardrail ───────────────────────────────────────────────────────

def _off_topic_redirect_text(count: int, name: str) -> str:
    """The canned line spoken by the off-topic handler (NOT by the LLM).

    Kept out of the LLM prompt because Gemini Flash has been observed to
    call the function and then return zero completion tokens, leaving the
    student in silence. The handler speaks this via TTSSpeakFrame so the
    redirect is guaranteed regardless of Gemini's whim.
    """
    redirect = _FLOW_SPEC.redirect_when_off_topic if _FLOW_SPEC else (
        "Let's bring it back to the main question."
    )
    if count == 0:
        return redirect
    if count == 1:
        return f"{name}, I need you to stay on topic. Please answer the question I asked."
    if count == 2:
        return (
            f"{name}, this is your final warning. If you go off-topic again, "
            "I will have to cancel this exam."
        )
    return (
        f"{name}, the exam is now being cancelled due to repeated off-topic behaviour."
    )


def _off_topic_instruction(count: int, name: str) -> str:
    """System-prompt fragment that tells the LLM WHEN to call the function.

    Note: the redirect TEXT is NOT included — the handler speaks it directly.
    This avoids the 0-completion-tokens deadlock where Gemini called the
    function first, saw next_action='wait_for_student', and refused to speak
    the redirect at all.
    """
    # If the student is asking ABOUT the exam (META), the LLM should explain
    # briefly rather than strike. Inject the exam's own topic + skills so the
    # explanation is precise to this assessment, not a generic stock line.
    topic = (_FLOW_SPEC.topic if _FLOW_SPEC else "this assessment").strip()
    skills = (_FLOW_SPEC.skills_assessed if _FLOW_SPEC else []) or []
    skills_str = "; ".join(skills) if skills else "reasoning and language use"
    meta_check = (
        f"META CHECK first: if the student is asking ABOUT the question "
        f"('why ask this?', 'what does this mean?', 'is this graded?', "
        f"'can you repeat?'), do NOT strike. Briefly say this assesses "
        f"{skills_str} for {topic}, reassure no single right answer, then "
        f"re-pose the original question. Otherwise:\n"
    )

    if count == 0:
        return meta_check + (
            "OFF-TOPIC TRACKER [strikes: 0]: if the student's most recent message "
            "is off-topic (unrelated tangent, complaining about the question, "
            "ignoring it), call off_topic_strike. The redirect line will be "
            "spoken automatically — you do NOT need to write it yourself. "
            "After the function call, wait for the student's next response."
        )
    if count == 1:
        return meta_check + (
            "OFF-TOPIC TRACKER [strikes: 1]: if still off-topic, call off_topic_strike. "
            "The firm redirect will be spoken automatically. Then wait."
        )
    if count == 2:
        return meta_check + (
            "OFF-TOPIC TRACKER [strikes: 2]: if still off-topic, call off_topic_strike. "
            "The FINAL WARNING will be spoken automatically. Then wait."
        )
    return meta_check + (
        f"OFF-TOPIC TRACKER [strikes: {count}]: if still off-topic, call off_topic_strike. "
        "The cancellation line will be spoken automatically."
    )


async def handle_global_off_topic(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig | None]:
    count = flow_manager.state.get("off_topic_count", 0) + 1
    flow_manager.state.update({"off_topic_count": count})
    logger.info(f"Off-topic strike #{count}")

    # Speak the canned redirect directly via TTS so the student always hears
    # something even if Gemini Flash returns 0 completion tokens after the
    # function call (a known failure mode — see session 20260615_030219).
    name = flow_manager.state.get("name", "")
    redirect_text = _off_topic_redirect_text(count - 1, name)
    if redirect_text:
        try:
            await flow_manager.task.queue_frame(TTSSpeakFrame(redirect_text))
            logger.info(
                f"Handler emitted off-topic redirect via TTS: {redirect_text!r}"
            )
        except Exception as e:
            logger.warning(f"Failed to push TTSSpeakFrame for off-topic redirect: {e}")

    if count >= 4:
        return ({"status": "exam_cancelled"}, create_cancel_exam_node(flow_manager))

    current = flow_manager.state.get("current_probe", "opening")
    if current == "opening":
        return ({"next_action": "wait_for_student"}, create_opening_node(flow_manager))
    if current == "followup_probe":
        # Dynamic node — rebuild it from the saved gold moment so a single
        # off-topic strike doesn't end the exam.
        gold = flow_manager.state.get("current_gold_moment", "")
        if gold:
            return (
                {"next_action": "wait_for_student"},
                create_followup_probe_node(flow_manager, gold),
            )
        # No gold cached — stay on the current node (no transition).
        logger.warning(
            "Off-topic during followup_probe with no cached gold — staying on node"
        )
        return ({"next_action": "wait_for_student"}, None)
    if current in _PROBE_FACTORIES:
        return ({"next_action": "wait_for_student"}, _PROBE_FACTORIES[current](flow_manager))
    # Unknown current_probe — DO NOT close the exam on a single strike. Stay
    # on the current node and let the LLM redirect the student back on topic.
    logger.warning(
        f"Off-topic on unknown probe {current!r} — staying on node (was closing exam)"
    )
    return ({"next_action": "wait_for_student"}, None)


# ─── Cancel node ───────────────────────────────────────────────────────────────

def create_cancel_exam_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    return NodeConfig(
        name="cancelled",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Inform {name} in one firm, calm sentence that the exam is being cancelled "
                "due to repeated off-topic behaviour. Do not apologise. Ask no questions."
            ),
        }],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )


# ─── Greeting ──────────────────────────────────────────────────────────────────

def create_greeting_node(flow_manager: FlowManager = None) -> NodeConfig:
    count = flow_manager.state.get("greeting_strikes", 0) if flow_manager else 0

    if count == 0:
        evasion_tone = "Say: 'Just your first name is fine — what shall I call you?'"
    elif count == 1:
        evasion_tone = (
            "Say firmly: 'I can see you are not answering. "
            "This is a formal exam — I need your first name now.'"
        )
    else:
        evasion_tone = (
            "Say: 'This is your last chance. Tell me your name "
            "or I will have to end this session.'"
        )

    return NodeConfig(
        name="greeting",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                "STEP 1 — INITIAL GREETING (no function call yet).\n"
                "If there is NO user message in the conversation history (i.e. the "
                "student has not spoken yet), simply greet the student warmly. "
                "Introduce yourself as their IB MYP oral examiner. Explain you will "
                "have a short discussion and that there are no right or wrong "
                "answers — you want to hear their thoughts. Ask for their name. "
                "Then STOP and WAIT silently. Do NOT call assess_name_response. "
                "Do NOT call any function. Do NOT say anything else. WAIT.\n\n"
                "STEP 2 — AFTER STUDENT SPEAKS.\n"
                "Only AFTER the student has actually sent a message in response, "
                "call assess_name_response ONCE with:\n"
                "  intent='gave_name'           — student gave a real first name\n"
                "  intent='needs_clarification' — confused or asking about the question\n"
                "  intent='evasion'             — unrelated tangent or deliberate dodge\n\n"
                f"For 'evasion': {evasion_tone} Then call assess_name_response.\n"
                "For 'needs_clarification': respond helpfully, then call assess_name_response.\n\n"
                f"GREETING STRIKES SO FAR: {count}.\n\n"
                "CRITICAL RULES:\n"
                "• NEVER call assess_name_response on your own assistant message. "
                "Only call it in response to a real student utterance.\n"
                "• Call assess_name_response at most ONCE per student message.\n"
                "• Do NOT accept single words like 'Hi' or 'okay' as a name."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="assess_name_response",
                handler=handle_greeting_routing,
                description="Call after every student message in the greeting phase.",
                properties={
                    "intent": {
                        "type": "string",
                        "enum": ["gave_name", "needs_clarification", "evasion"],
                        "description": "Classification of the student's most recent message.",
                    },
                    "name_if_given": {
                        "type": "string",
                        "description": "The name the student gave. Empty string if none.",
                    },
                },
                required=["intent", "name_if_given"],
            ),
        ],
    )


# ─── Name confirmation + spell-out fallback ────────────────────────────────────

def create_known_name_greeting_node(flow_manager: FlowManager) -> NodeConfig:
    """Greeting used when the teacher provided the student's name via the UI.
    Greets by name, introduces the examiner, then asks 'Are you ready to begin?'
    and waits for the student's reply before transitioning to the first
    question. Skips STT-based name capture entirely.
    """
    name = flow_manager.state.get("name", "student")
    return NodeConfig(
        name="greeting_known_name",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"The student's name is {name}. Greet {name} warmly by name in "
                "one or two short sentences, introduce yourself as their IB MYP "
                "oral examiner, and reassure them there are no right or wrong "
                "answers — you want to hear their thoughts. End your greeting "
                "by asking clearly: 'Are you ready to begin?'\n\n"
                "DO NOT ask the exam question yet. DO NOT call any function "
                "until the student has spoken. Wait for the student's reply.\n\n"
                "After the student replies, call begin_exam ONCE with:\n"
                "  intent='ready'     — student said yes / ready / sure / okay / "
                "any clear affirmative.\n"
                "  intent='not_ready' — student said no / wait / give me a moment / "
                "asked a question / expressed nervousness.\n\n"
                "Do NOT ask for the student's name; you already have it."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="begin_exam",
                handler=handle_begin_exam,
                description=(
                    "Call once after the student replies to 'Are you ready to "
                    "begin?'. Passes their readiness intent."
                ),
                properties={
                    "intent": {
                        "type": "string",
                        "enum": ["ready", "not_ready"],
                        "description": "Whether the student is ready to start the exam.",
                    },
                },
                required=["intent"],
            ),
        ],
    )


def create_confirm_name_node(flow_manager: FlowManager) -> NodeConfig:
    """After the greeting captures a name, read it back and ask the student to
    confirm. If they reject, fall through to a spell-out node so a garbled STT
    pass doesn't lock in the wrong name for the whole session."""
    candidate = flow_manager.state.get("name_candidate", "")
    return NodeConfig(
        name="confirm_name",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"You just heard the student's name as '{candidate}'. Read it "
                f"back warmly in one short sentence to confirm — e.g. "
                f"\"Great — I'll call you {candidate}. Did I get that right?\" "
                "Wait for the student's reply, then call assess_name_confirmation "
                "ONCE with one of:\n"
                "  intent='confirmed'    — student agreed (yes / correct / that's right / nods verbally)\n"
                "  intent='corrected'    — student rejected it or gave a different name\n"
                "If 'corrected' and the student clearly stated a new name in the "
                "same reply, pass it as corrected_name. Otherwise leave "
                "corrected_name empty and we'll ask them to spell it."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="assess_name_confirmation",
                handler=handle_name_confirmation,
                description="Call once after the student replies to the confirmation prompt.",
                properties={
                    "intent": {
                        "type": "string",
                        "enum": ["confirmed", "corrected"],
                        "description": "Whether the student confirmed the read-back name.",
                    },
                    "corrected_name": {
                        "type": "string",
                        "description": (
                            "If the student stated a new/corrected name in the same "
                            "reply, put it here. Otherwise empty string."
                        ),
                    },
                },
                required=["intent", "corrected_name"],
            ),
        ],
    )


def create_spell_name_node(flow_manager: FlowManager) -> NodeConfig:
    """Asks the student to spell their first name letter by letter. STT is far
    more accurate on individual letters than on a single-word proper noun, so
    this is the recovery branch from the confirmation node."""
    return NodeConfig(
        name="spell_name",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                "Apologise briefly for the mishearing in one short sentence, "
                "then ask the student to spell their first name out letter by "
                "letter — e.g. \"Sorry about that — could you spell your first "
                "name for me, letter by letter?\". Wait for the student's "
                "spelling, then call record_spelled_name ONCE with the "
                "assembled name in title case. If the student's reply does not "
                "look like a spelling (e.g. they just say a name again), still "
                "accept that name."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="record_spelled_name",
                handler=handle_spelled_name,
                description="Call once after the student spells (or restates) their name.",
                properties={
                    "name": {
                        "type": "string",
                        "description": (
                            "The student's first name, assembled from the "
                            "letters they spelled, title-cased."
                        ),
                    },
                },
                required=["name"],
            ),
        ],
    )


# ─── Opening (uses generated exam_question) ────────────────────────────────────

def create_opening_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    count = flow_manager.state.get("off_topic_count", 0)
    if not _FLOW_SPEC:
        raise RuntimeError("_FLOW_SPEC is None — flow was not initialised")
    exam_question = _FLOW_SPEC.exam_question
    available = [p.id for p in _FLOW_SPEC.probe_pool]
    time_notice = _time_notice_prefix(flow_manager)
    support_notice = _support_notice_prefix(flow_manager)
    no_repeat = _no_repeat_prefix(flow_manager)

    return NodeConfig(
        name="opening",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"{time_notice}{support_notice}{no_repeat}"
                f"Ask {name} the following exam question, word for word:\n"
                f"'{exam_question}'\n\n"
                "STEP 1 — CLARIFY STANCE.\n"
                "If the student gives only facts or no clear position, redirect: "
                "'That is good background — but what is YOUR view? "
                "Do you think it is mainly positive, negative, or complicated?' "
                "Do not move on until they have a position.\n\n"
                "STEP 2 — ACCEPT OR PROBE.\n"
                "If they have a position AND have named at least ONE reason (even "
                "as a short phrase — 'because it's faster', 'because of comfort', "
                "'fairness'), ACCEPT IT and call advance_from_probe with depth='surface' "
                "or 'developed' as appropriate. Do NOT demand more depth from a real "
                "answer just because it is brief.\n\n"
                "STEP 3 — CALL FUNCTION.\n"
                f"Available probe IDs for recommended_next: {available}.\n\n"
                "DISENGAGE SIGNALS — mandatory, overrides everything else:\n"
                f"If {name}'s most recent message contains ANY of these intents — "
                "even paraphrased — you MUST immediately accept their previous answer "
                "at its current depth and call advance_from_probe with "
                "student_consented_to_advance=true and student_struggling=true. "
                "Do NOT push back. Just acknowledge briefly and advance.\n"
                "  • 'next question' / 'move on' / 'skip' / 'I don't know'\n"
                "  • 'I don't want to answer' / 'pass' / 'I'm done with this'\n"
                "  • 'can we continue?' / 'what's next?'\n"
                "  • 'I can't recall' / 'nothing comes to mind'\n"
                "A frustrated student is a failed exam — honouring disengage is "
                "more important than extracting depth.\n\n"
                "TIERED ATTEMPT POLICY (count follow-ups within this opening probe):\n"
                f"  • Attempt 1: ask {name} normally per the question above.\n"
                "  • Attempt 2: if still thin, move SIDEWAYS to a different aspect — "
                "not the same micro-concept. Offer a graceful out: "
                "'Would you like to try a different question, or stay on this one?' "
                "If they consent, advance with student_consented_to_advance=true.\n"
                "  • Attempt 3+: ACCEPT whatever they have said. Call advance_from_probe "
                "with depth='surface', student_struggling=true. Acknowledge warmly: "
                "'That is a tricky one — let's move on.' Do NOT keep pushing.\n\n"
                "If the result has next_action='wait_for_student', stop and wait."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_probe",
                handler=handle_transition,
                description="Call once student has a clear position and a reason.",
                properties=ADVANCE_PROBE_SCHEMA,
                required=list(ADVANCE_PROBE_SCHEMA.keys()),
            ),
        ],
    )


# ─── Time-notice helper (shared by all probe nodes) ────────────────────────────

def _time_notice_prefix(flow_manager: FlowManager) -> str:
    """If the time-warning flag is set but hasn't been announced yet, return a
    one-time instruction prefix to inject into the next node's task_messages.
    Marks it announced so it never fires twice."""
    if not flow_manager.state.get("time_warning_given"):
        return ""
    if flow_manager.state.get("time_warning_announced"):
        return ""
    flow_manager.state.update({"time_warning_announced": True})
    return (
        "TIME NOTICE: at the start of your next turn, give the student ONE warm, "
        "brief warning that the discussion is winding down. Use a variation of: "
        "'We're nearing the end of our discussion — let's finish with one or two "
        "more questions.' Do NOT mention exact minutes. Then proceed with the "
        "probe instructions below.\n\n"
    )


def _revisit_notice_prefix(flow_manager: FlowManager) -> str:
    """Fires when handle_transition has chosen to revisit a covered probe
    because all probes are exhausted but time remains. Tells the LLM to
    explore a brand-new angle, not repeat what was asked before."""
    if not flow_manager.state.get("in_revisit"):
        return ""
    return (
        "REVISIT NOTICE: we have already covered the planned probes once, but "
        "there is still time in the exam. Treat THIS probe as a REVISIT — "
        "explore the topic from a COMPLETELY DIFFERENT angle than what you "
        "asked before. Open with a brief transition like 'Earlier we talked "
        "about X — let's look at it from a different side.' Be extra patient "
        "and supportive. Use the SIMPLEST possible scaffolds. Do NOT honour "
        "a single disengage signal immediately — try one supportive scaffold "
        "first.\n\n"
    )


def _recent_examiner_openings(n: int = 3, max_words: int = 9) -> list[str]:
    """Return the opening ~max_words of the last `n` EXAMINER turns from the
    live transcript collector, deduplicated (case-insensitive). Used to feed
    back into the next probe so the LLM can see (and avoid) phrases it just
    used."""
    if _CURRENT_TRANSCRIPT is None:
        return []
    examiner = [t["text"] for t in _CURRENT_TRANSCRIPT.turns if t["speaker"] == "EXAMINER"]
    out: list[str] = []
    seen: set[str] = set()
    for text in reversed(examiner):
        words = text.strip().split()
        if len(words) < 3:
            continue
        snippet = " ".join(words[:max_words]).rstrip(".,;:!?-—")
        key = snippet.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(snippet)
        if len(out) >= n:
            break
    return list(reversed(out))


def _no_repeat_prefix(flow_manager: FlowManager) -> str:
    """Inject a 'do not reuse these openers' block into the next probe's
    task_messages, populated with the openings of the assistant's last few
    turns. Hard counter to the LLM's tendency to glom onto one phrasing."""
    openings = _recent_examiner_openings()
    if not openings:
        return ""
    bullets = "\n".join(f"  • '{o}…'" for o in openings)
    return (
        "DO NOT REUSE YOUR RECENT OPENERS — you have already used these in "
        "your last few turns. Open this turn with a COMPLETELY different "
        "acknowledgement / transition / lead-in. Variety is mandatory.\n"
        f"{bullets}\n\n"
    )


def _support_notice_prefix(flow_manager: FlowManager) -> str:
    """Fires when the student has signalled disengage on 2+ consecutive
    probes. Tells the LLM to scaffold heavily instead of advancing again."""
    streak = int(flow_manager.state.get("consecutive_disengages", 0))
    if streak < 2:
        return ""
    return (
        f"SUPPORTIVE MODE — the student has just disengaged on {streak} probes "
        "in a row. Do NOT honour another disengage signal on this probe. "
        "Instead, scaffold heavily: ask the SIMPLEST possible version of the "
        "question (e.g. 'What's one word that comes to mind?'). Open with a "
        "warm acknowledgement composed FRESH — do NOT reuse 'there is no perfect "
        "answer here', 'that is totally fine', or any opener you used in your "
        "last turn. Stay on this probe until you get ANY substantive content, "
        "then advance with student_struggling=true.\n\n"
    )


# ─── Dynamic probe node (built from FlowSpec via primitives) ──────────────────

def create_dynamic_probe_node(
    flow_manager: FlowManager,
    probe: ProbeNode,
) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    count = flow_manager.state.get("off_topic_count", 0)
    covered = set(flow_manager.state.get("covered_probes", []))
    available = [p.id for p in _FLOW_SPEC.probe_pool if p.id not in covered]

    instructions = render_probe_instructions(probe, name, available)
    time_notice = _time_notice_prefix(flow_manager)
    revisit_notice = _revisit_notice_prefix(flow_manager)
    support_notice = _support_notice_prefix(flow_manager)
    no_repeat = _no_repeat_prefix(flow_manager)

    return NodeConfig(
        name=probe.id,
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"{time_notice}{revisit_notice}{support_notice}{no_repeat}{instructions}"
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_probe",
                handler=handle_transition,
                description="Call after the student has responded substantively to this probe.",
                properties=ADVANCE_PROBE_SCHEMA,
                required=list(ADVANCE_PROBE_SCHEMA.keys()),
            ),
        ],
    )


# ─── Gold-moment follow-up ─────────────────────────────────────────────────────

def create_followup_probe_node(
    flow_manager: FlowManager,
    gold_moment: str,
) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    count = flow_manager.state.get("off_topic_count", 0)
    covered = set(flow_manager.state.get("covered_probes", []))
    available = [p.id for p in _FLOW_SPEC.probe_pool if p.id not in covered]
    time_notice = _time_notice_prefix(flow_manager)
    support_notice = _support_notice_prefix(flow_manager)
    no_repeat = _no_repeat_prefix(flow_manager)

    return NodeConfig(
        name="followup_probe",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"{time_notice}{support_notice}{no_repeat}"
                f"{name} just said something worth following up on:\n"
                f"'{gold_moment}'\n\n"
                f"Ask {name} ONE question to dig deeper into that specific idea. "
                "Be genuinely curious, not leading. Examples:\n"
                "  'Can you say more about that?'\n"
                "  'What makes you think that?'\n"
                "  'Can you give me a concrete example of what you mean?'\n\n"
                f"AVAILABLE NEXT PROBES AFTER THIS: {available}\n\n"
                "Once they have elaborated, call advance_from_probe."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_probe",
                handler=handle_transition,
                description="Call after the student elaborates on their insight.",
                properties=ADVANCE_PROBE_SCHEMA,
                required=list(ADVANCE_PROBE_SCHEMA.keys()),
            ),
        ],
    )


# ─── Closing ───────────────────────────────────────────────────────────────────

def create_closing_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    ended_for_time = flow_manager.state.get("ended_early_for_time", False)

    if ended_for_time:
        content = (
            f"We have run out of time, so this is the end of the discussion. "
            f"Thank {name} warmly in one or two short sentences — acknowledge "
            "that you covered a lot of ground together and that their answers "
            "were thoughtful. Do NOT mention the time running out as a problem. "
            "Do NOT give feedback on language, performance, or scores. "
            "Do NOT ask any further questions."
        )
    else:
        content = (
            f"Thank {name} warmly in one or two short sentences for their thoughtful "
            "responses. Let them know the oral discussion is now complete. "
            "Do NOT give feedback on language, performance, or scores. "
            "Do NOT ask any further questions."
        )

    return NodeConfig(
        name="closing",
        role_messages=ROLE_MESSAGES,
        task_messages=[{"role": "system", "content": content}],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )


# ─── Time-budget helper ────────────────────────────────────────────────────────

TIME_WARNING_THRESHOLD = 0.75   # at 75% of budget, inject the "wrap up" notice
TIME_HARD_THRESHOLD = 1.00      # at 100% of budget, jump to closing


def _compute_time_state(flow_manager: FlowManager) -> dict:
    """Return {elapsed, budget, remaining, fraction, hit_warning, hit_hard}.
    All times in seconds. If the budget or start time is missing, returns
    a permissive state (no warning, no force-close)."""
    started_iso = flow_manager.state.get("started_at_iso")
    duration_min = (_FLOW_SPEC.duration_minutes if _FLOW_SPEC else None) or 6
    budget = duration_min * 60
    if not started_iso:
        return {
            "elapsed": 0.0, "budget": budget, "remaining": budget,
            "fraction": 0.0, "hit_warning": False, "hit_hard": False,
        }
    try:
        started = datetime.fromisoformat(started_iso)
    except Exception:
        return {
            "elapsed": 0.0, "budget": budget, "remaining": budget,
            "fraction": 0.0, "hit_warning": False, "hit_hard": False,
        }
    elapsed = (datetime.now() - started).total_seconds()
    frac = elapsed / budget if budget > 0 else 0.0
    return {
        "elapsed": elapsed, "budget": budget, "remaining": max(0.0, budget - elapsed),
        "fraction": frac,
        "hit_warning": frac >= TIME_WARNING_THRESHOLD,
        "hit_hard": frac >= TIME_HARD_THRESHOLD,
    }


# ─── Smart transition handler ──────────────────────────────────────────────────

VOLUNTARY_SKIP_LIMIT = 2  # Beyond this, treat further skips as struggle, not voluntary.
PROBE_ATTEMPT_CAP = 3     # Matches _ATTEMPT_CAP_RULE in flow_primitives.
REVISIT_LIMIT = 2         # How many times we'll loop back to covered probes if time remains.


def _pick_revisit_probe(flow_manager: FlowManager) -> str | None:
    """Pick the covered probe with the lowest attempt count — the one that
    got least exploration first time around — as the revisit target.
    Returns None if no probe is suitable."""
    if not _FLOW_SPEC:
        return None
    covered = set(flow_manager.state.get("covered_probes", []))
    attempts = flow_manager.state.get("probe_attempts", {})
    # Candidates: probes in the pool that were covered.
    candidates = [(p.id, int(attempts.get(p.id, 0))) for p in _FLOW_SPEC.probe_pool if p.id in covered]
    if not candidates:
        return None
    # Sort by attempts ascending — least-explored probe gets revisited first.
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


async def handle_transition(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    current = flow_manager.state.get("current_probe", "opening")
    covered = set(flow_manager.state.get("covered_probes", []))
    covered.add(current)
    node_count = flow_manager.state.get("node_count", 0) + 1

    all_terms = list(set(
        flow_manager.state.get("vague_terms", []) + args.get("vague_terms", [])
    ))

    # ── Voluntary skip tracking (B) ───────────────────────────────────────
    consented = bool(args.get("student_consented_to_advance", False))
    voluntary_skips = flow_manager.state.get("voluntary_skips", 0)
    consecutive_disengages = flow_manager.state.get("consecutive_disengages", 0)
    if consented:
        voluntary_skips += 1
        consecutive_disengages += 1
        logger.info(
            f"Voluntary skip #{voluntary_skips} (limit {VOLUNTARY_SKIP_LIMIT}) "
            f"on probe {current!r} — consecutive_disengages={consecutive_disengages}"
        )
    else:
        consecutive_disengages = 0  # Reset on a real (non-consent) advance

    # ── Cap-violation detection (C) — observational only ─────────────────
    probe_attempts = flow_manager.state.get("probe_attempts", {})
    attempts_for_current = int(probe_attempts.get(current, 0))
    cap_violations: list[dict] = list(flow_manager.state.get("cap_violations", []))
    if attempts_for_current > PROBE_ATTEMPT_CAP:
        violation = {
            "probe": current,
            "attempts": attempts_for_current,
            "cap": PROBE_ATTEMPT_CAP,
        }
        logger.warning(
            f"CAP VIOLATION on probe {current!r}: {attempts_for_current} attempts "
            f"(cap is {PROBE_ATTEMPT_CAP}). LLM ignored the tiered cap rule."
        )
        cap_violations.append(violation)

    flow_manager.state.update({
        "covered_probes": list(covered),
        "node_count": node_count,
        "vague_terms": all_terms,
        "voluntary_skips": voluntary_skips,
        "consecutive_disengages": consecutive_disengages,
        "cap_violations": cap_violations,
    })

    # ── Time budget ───────────────────────────────────────────────────────
    time_state = _compute_time_state(flow_manager)
    logger.info(
        f"Transition from '{current}' | covered={covered} | "
        f"node_count={node_count} | attempts_for_current={attempts_for_current} | "
        f"consented={consented} | voluntary_skips={voluntary_skips}/{VOLUNTARY_SKIP_LIMIT} | "
        f"elapsed={time_state['elapsed']:.0f}s ({time_state['fraction']*100:.0f}% of "
        f"{time_state['budget']:.0f}s) | gold='{args.get('gold_moment', '')}'"
    )

    # If the student has used up their voluntary skips, the next consented
    # advance is still honoured BUT the grader sees it as struggle, not consent.
    # (We don't block the advance — that would defeat the empathetic intent —
    # we just stop letting consent inflate engagement metrics.)
    if consented and voluntary_skips > VOLUNTARY_SKIP_LIMIT:
        logger.info(
            f"Voluntary skip limit exceeded — treating advance as struggle "
            f"(voluntary_skips={voluntary_skips})"
        )

    # Hard time cap: skip to closing after the current probe
    if time_state["hit_hard"]:
        logger.warning(
            f"Time budget exhausted ({time_state['elapsed']:.0f}s >= "
            f"{time_state['budget']:.0f}s) — forcing closing"
        )
        flow_manager.state.update({"ended_early_for_time": True})
        return ({}, create_closing_node(flow_manager))

    # Soft time warning: fire ONCE when we cross 75% of budget. We don't
    # build the next node yet — we just set the flag so the next node's
    # factory (e.g. create_dynamic_probe_node) injects the TIME NOTICE
    # instruction into its task_messages.
    warning_given = flow_manager.state.get("time_warning_given", False)
    if time_state["hit_warning"] and not warning_given:
        logger.info(
            f"Time warning threshold crossed ({time_state['fraction']*100:.0f}%) — "
            "next probe will tell the student to wrap up"
        )
        flow_manager.state.update({"time_warning_given": True})

    # ── Routing decisions (unchanged below) ───────────────────────────────
    gold = args.get("gold_moment", "").strip()
    gold_followed = flow_manager.state.get("gold_followed", 0)
    if gold and gold_followed < 1:
        flow_manager.state.update({
            "gold_followed": gold_followed + 1,
            "current_probe": "followup_probe",
            # Persist gold so off-topic / re-entry can rebuild the same node
            # without losing the moment we were following.
            "current_gold_moment": gold,
        })
        logger.info(f"Following gold moment: {gold!r}")
        return ({}, create_followup_probe_node(flow_manager, gold))

    missing_required = _REQUIRED_PROBES - covered
    if missing_required:
        next_id = sorted(missing_required)[0]
        logger.info(f"Required probe not yet covered: {next_id!r}")
        flow_manager.state.update({"current_probe": next_id})
        return ({}, _PROBE_FACTORIES[next_id](flow_manager))

    recommended = args.get("recommended_next", "").strip()
    if recommended and recommended not in covered and recommended in _PROBE_FACTORIES:
        logger.info(f"Using LLM recommendation: {recommended!r}")
        flow_manager.state.update({"current_probe": recommended})
        return ({}, _PROBE_FACTORIES[recommended](flow_manager))

    remaining = [p.id for p in _FLOW_SPEC.probe_pool if p.id not in covered]
    if not remaining:
        # All probes covered — but should we really close, or revisit?
        # Close only if we're past the 75% time-warning threshold OR we've
        # already revisited the max number of times. Otherwise revisit the
        # least-explored covered probe with a "supportive" notice.
        revisit_count = flow_manager.state.get("revisit_count", 0)
        if time_state["hit_warning"] or revisit_count >= REVISIT_LIMIT:
            logger.info(
                f"All probes covered + (time_warning={time_state['hit_warning']} or "
                f"revisits={revisit_count}/{REVISIT_LIMIT}) — closing"
            )
            return ({}, create_closing_node(flow_manager))

        revisit_id = _pick_revisit_probe(flow_manager)
        if revisit_id is None:
            logger.info("All probes covered, no revisit candidate — closing")
            return ({}, create_closing_node(flow_manager))

        flow_manager.state.update({
            "current_probe": revisit_id,
            "in_revisit": True,
            "revisit_count": revisit_count + 1,
        })
        # Reset attempt count for the revisited probe so the tier policy
        # starts fresh, not on top of the old failed attempts.
        attempts = dict(flow_manager.state.get("probe_attempts", {}))
        attempts[revisit_id] = 0
        flow_manager.state.update({"probe_attempts": attempts})
        logger.info(
            f"Time remains ({time_state['fraction']*100:.0f}% used) and probes exhausted "
            f"— REVISIT #{revisit_count + 1} on probe {revisit_id!r}"
        )
        return ({}, _PROBE_FACTORIES[revisit_id](flow_manager))

    next_id = remaining[0]
    logger.info(f"Picking next uncovered probe: {next_id!r}")
    flow_manager.state.update({"current_probe": next_id})
    return ({}, _PROBE_FACTORIES[next_id](flow_manager))


# ─── Greeting routing handler ──────────────────────────────────────────────────

async def handle_greeting_routing(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig | None]:
    intent = args["intent"]
    name_given = args.get("name_if_given", "").strip()

    # Phantom-call guard: the LLM sometimes calls this function on its own
    # initial greeting before the student has uttered anything. Detect that and
    # silently drop the call (return next_node=None → no transition, no new LLM
    # generation). Without this guard the bot ends up apologising to itself in
    # turn 2 and the greeting loops.
    student_utterances = flow_manager.state.get("student_utterance_count", 0)
    if student_utterances == 0 and not (intent == "gave_name" and name_given):
        logger.warning(
            f"Greeting: ignoring phantom assess_name_response (intent={intent!r}, "
            f"no student message in context yet)"
        )
        return ({"next_action": "wait_for_student", "ignored": True}, None)

    if intent == "gave_name" and name_given:
        logger.info(f"Name candidate captured: {name_given!r} → confirm")
        flow_manager.state.update({"name_candidate": name_given})
        return ({"next_action": "confirm_name"}, create_confirm_name_node(flow_manager))

    if intent == "needs_clarification":
        logger.info("Greeting: clarification needed, no strike")
        return ({"next_action": "wait_for_student"}, create_greeting_node(flow_manager))

    count = flow_manager.state.get("greeting_strikes", 0) + 1
    flow_manager.state.update({"greeting_strikes": count})
    logger.info(f"Greeting evasion strike #{count}")
    if count >= 3:
        return ({"status": "exam_cancelled"}, create_cancel_exam_node(flow_manager))
    return ({"next_action": "wait_for_student"}, create_greeting_node(flow_manager))


async def handle_name_confirmation(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    intent = args["intent"]
    corrected = args.get("corrected_name", "").strip()
    candidate = flow_manager.state.get("name_candidate", "").strip()

    if intent == "confirmed" and candidate:
        logger.info(f"Name confirmed: {candidate!r}")
        flow_manager.state.update({
            "name": candidate,
            "off_topic_count": 0,
            "current_probe": "opening",
        })
        return ({}, create_opening_node(flow_manager))

    # Student rejected the read-back. If they supplied a corrected name in the
    # same turn, use it directly; otherwise fall through to spell-out.
    if corrected:
        logger.info(f"Name corrected inline: {corrected!r}")
        flow_manager.state.update({
            "name": corrected,
            "off_topic_count": 0,
            "current_probe": "opening",
        })
        return ({}, create_opening_node(flow_manager))

    logger.info("Name rejected, asking student to spell it out")
    return ({"next_action": "spell_name"}, create_spell_name_node(flow_manager))


async def handle_begin_exam(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    """Used by create_known_name_greeting_node. If the student is ready,
    transitions to the opening exam question. If not, stays in the greeting
    node so the bot can reassure them and re-ask."""
    intent = args.get("intent", "ready")
    if intent == "not_ready":
        # Re-render the greeting so the LLM can offer reassurance and re-ask.
        # The reply will route here again with a fresh intent.
        flow_manager.state["not_ready_count"] = (
            flow_manager.state.get("not_ready_count", 0) + 1
        )
        logger.info(
            f"Student not yet ready (count={flow_manager.state['not_ready_count']}); "
            "staying in greeting."
        )
        return (
            {"next_action": "wait_for_student"},
            create_known_name_greeting_node(flow_manager),
        )

    logger.info("Student is ready; advancing to opening question.")
    flow_manager.state.update({
        "off_topic_count": 0,
        "current_probe": "opening",
    })
    return ({}, create_opening_node(flow_manager))


async def handle_spelled_name(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    name = args.get("name", "").strip().title()
    if not name:
        # Defensive: shouldn't happen given required=["name"], but if the LLM
        # gives an empty string fall back to a generic address rather than loop.
        name = "student"
    logger.info(f"Name recorded via spelling: {name!r}")
    flow_manager.state.update({
        "name": name,
        "off_topic_count": 0,
        "current_probe": "opening",
    })
    return ({}, create_opening_node(flow_manager))


# ─── Flow-spec generation with validation + retry + persistence ───────────────

def _print_flow_spec(spec: FlowSpec, flow_path: Path) -> None:
    """Pretty-print the generated flow to the console so the teacher can see
    exactly what the LLM produced before the live session starts."""
    bar = "═" * 72
    print()
    print(bar)
    print(" GENERATED FLOW SPEC")
    print(bar)
    print(f"  Topic            : {spec.topic}")
    print(f"  Exam question    : {spec.exam_question}")
    print(f"  Duration         : {spec.duration_minutes} min")
    print(f"  Examiner tone    : {spec.examiner_tone}")
    if spec.skills_assessed:
        print(f"  Skills assessed  : {', '.join(spec.skills_assessed)}")
    print(f"  Off-topic redir. : {spec.redirect_when_off_topic}")
    print(f"  Required probes  : {spec.required_probes}")
    print(f"  Probe pool       : {len(spec.probe_pool)} probes")
    print("─" * 72)
    for i, probe in enumerate(spec.probe_pool, 1):
        marker = " ★" if probe.id in spec.required_probes else "  "
        print(f"  {i}.{marker} [{probe.type:<10}] {probe.id}")
        # wrap the prompt at ~66 chars for readability
        words = probe.prompt.split()
        line = "         "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                print(line)
                line = "         " + w
            else:
                line = (line + " " + w) if line.strip() else line + w
        if line.strip():
            print(line)
        if probe.scaffold:
            print(f"         scaffold: {probe.scaffold}")
    print("─" * 72)
    print(f"  Saved to : {flow_path}")
    print(f"  ★ = required probe (must be covered every session)")
    print(bar)
    print()


async def _generate_and_validate(topic: str, duration: int, session_id: str) -> FlowSpec:
    """Generate → validate → retry once on failure → persist."""
    last_errors: list[str] = []

    for attempt in (1, 2):
        try:
            spec = await generate_flow(topic, duration)
        except (ValidationError, json.JSONDecodeError) as e:
            last_errors = [f"could not parse planner output: {e}"]
            logger.warning(f"Planner attempt {attempt}: {last_errors}")
            continue

        try:
            validate_flow_spec(spec)
        except FlowValidationError as e:
            last_errors = e.reasons
            logger.warning(f"Validation attempt {attempt} failed: {last_errors}")
            continue

        FLOWS_DIR.mkdir(parents=True, exist_ok=True)
        flow_path = FLOWS_DIR / f"flow_{session_id}.json"
        flow_path.write_text(spec.model_dump_json(indent=2))
        logger.info(f"FlowSpec persisted → {flow_path}")
        _print_flow_spec(spec, flow_path)
        return spec

    raise FlowValidationError(last_errors or ["unknown planner failure"])


def build_probe_factories(flow_spec: FlowSpec) -> dict[str, Callable]:
    return {
        probe.id: (lambda fm, p=probe: create_dynamic_probe_node(fm, p))
        for probe in flow_spec.probe_pool
    }


# ─── Grading (ported from teacher2.py) ────────────────────────────────────────

GRADER_MODEL = os.getenv("GEMINI_GRADER_MODEL", "gemini-3.5-flash")

DIMENSIONS = {
    "clarity":       {"weight": 0.10, "label": "Clarity of Position"},
    "reasoning":     {"weight": 0.25, "label": "Reasoning & Critical Thinking"},
    "examples":      {"weight": 0.20, "label": "Examples & Conceptual Depth"},
    "engagement":    {"weight": 0.10, "label": "Engagement & Focus"},
    "fluency":       {"weight": 0.15, "label": "Fluency & Pacing"},
    "pronunciation": {"weight": 0.10, "label": "Pronunciation & Intelligibility"},
    "confidence":    {"weight": 0.10, "label": "Confidence & Prosody"},
}
DIMENSION_GUIDE = """\
  clarity        — How directly did the student state and maintain their position?
  reasoning      — Quality of argumentation AND counterargument handling.
                   Does the student justify their view and seriously consider
                   the other side instead of dismissing it?
  examples       — Real-world grounding plus conceptual depth. Are examples
                   specific (named places, programs) or vague? Does the student
                   engage with concepts like scale, fairness, sustainability?
  engagement     — On-topic discipline AND willingness to attempt every probe.
                   Penalise students who barely spoke or refused to engage.
  fluency        — Flow, pacing, and pause patterns. Estimate WPM from the audio
                   (ideal 130–150).
  pronunciation  — Intelligibility and word-choice precision.
  confidence     — Prosody, conviction, and filler control (um/uh/like/well).
"""

GRADING_PROMPT = """You are an IB MYP oral exam evaluator producing a HOLISTIC
report on a single 4–6 minute discussion. Do not grade turn-by-turn. Read the
whole exam first, then write one integrated judgment per dimension.

Two voices: an AI EXAMINER and the STUDENT. Grade ONLY the student.

Exam question (LLM-generated for this session):
  "{exam_question}"

Topic context:
  "{topic}"

Exam metadata (JSON):
{metadata}

Diarized transcript — authoritative source for evidence:
{transcript}

Node trace (which probes were reached):
{node_trace}

DIMENSIONS — score each 1–4 (1 Emerging · 2 Developing · 3 Proficient · 4 Mastery)
{dimension_guide}

For each dimension produce:
  - score:    integer 1–4
  - analysis: 2–4 sentences of integrated commentary across the WHOLE exam.
  - evidence: 1–3 short student quotes from the transcript. Never quote the
              examiner. Never write "Entire recording".

Then produce:
  - headline: one sentence (≤ 20 words) overall impression
  - summary:  4–6 sentence holistic narrative
  - strengths:    2–3 specific items
  - growth_areas: 2–3 concrete next steps
  - overall_score: a single number 0–100 reflecting overall performance.
                   Use your own judgement; weigh reasoning and examples most.
                   If metadata.cancelled is true, set this to 0.

If metadata.cancelled is true, set every dimension score to 1 too.

Return ONLY valid JSON, no markdown:
{{
  "headline": str,
  "summary": str,
  "dimensions": {{
    "clarity":       {{"score": int, "analysis": str, "evidence": [str, ...]}},
    "reasoning":     {{...}},
    "examples":      {{...}},
    "engagement":    {{...}},
    "fluency":       {{...}},
    "pronunciation": {{...}},
    "confidence":    {{...}}
  }},
  "strengths": [str, ...],
  "growth_areas": [str, ...],
  "overall_score": float
}}
"""


def _write_wav(path: Path, pcm: bytes, sample_rate: int, num_channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


# Markers used by pipecat's filter_incomplete_user_turns mixin to flag
# turn state in LLM output: ✓ complete, ○ incomplete-short, ◐ incomplete-long.
# MarkerStripFilter cleans the TTS path, but the assistant aggregator
# captures the raw LLM text into context.messages BEFORE the strip filter
# runs. Strip here so reports and grader prompts never see them.
_TURN_MARKERS = ("✓", "○", "◐")


def _strip_turn_markers(text: str) -> str:
    s = text or ""
    # Remove markers anywhere in the string (they can appear mid-sentence
    # when the LLM emits them as soft prefixes to chunked outputs).
    for mk in _TURN_MARKERS:
        s = s.replace(mk, "")
    # Collapse the double-space holes left behind.
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def _extract_transcript(context) -> list[dict]:
    msgs = None
    for attr in ("get_messages", "messages"):
        obj = getattr(context, attr, None)
        if callable(obj):
            try:
                msgs = obj()
                break
            except Exception:
                continue
        elif obj is not None:
            msgs = obj
            break
    if not msgs:
        return []

    turns: list[dict] = []
    for m in msgs:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(content, list):
            content = " ".join(
                (p.get("text", "") if isinstance(p, dict) else str(p)) for p in content
            )
        if not content or role not in ("user", "assistant"):
            continue
        speaker = "STUDENT" if role == "user" else "EXAMINER"
        text = _strip_turn_markers(str(content))
        if not text:
            continue
        turns.append({"speaker": speaker, "text": text})
    return turns


def _format_transcript(turns: list[dict]) -> str:
    if not turns:
        return "(empty — STT produced no turns)"
    return "\n".join(f"[{t['speaker']}] {t['text']}" for t in turns)




def _grade_exam_sync(
    wav_path: Path,
    metadata: dict,
    transcript_str: str,
    node_trace_str: str,
    exam_question: str,
    topic: str,
) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    audio_bytes = wav_path.read_bytes()

    prompt_template = _CUSTOM_GRADING_PROMPT or GRADING_PROMPT
    try:
        prompt = prompt_template.format(
            exam_question=exam_question,
            topic=topic,
            metadata=json.dumps(metadata, indent=2),
            transcript=transcript_str,
            node_trace=node_trace_str,
            dimension_guide=DIMENSION_GUIDE,
        )
    except KeyError as e:
        logger.warning(f"Custom grading prompt missing placeholder {e}; falling back to default")
        prompt = GRADING_PROMPT.format(
            exam_question=exam_question,
            topic=topic,
            metadata=json.dumps(metadata, indent=2),
            transcript=transcript_str,
            node_trace=node_trace_str,
            dimension_guide=DIMENSION_GUIDE,
        )

    response = client.models.generate_content(
        model=GRADER_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=8192,   # 7-dim report can be long; default cap truncates
        ),
    )
    text = (response.text or "{}").strip()

    # Defensive: some models wrap JSON in ```json ... ``` despite the mime hint.
    if text.startswith("```"):
        # Strip leading ``` and optional language tag
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    # Defensive: trim anything before first { and after last }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
    else:
        candidate = text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.error(
            f"Grader returned non-JSON ({e}). First 1000 chars:\n{text[:1000]}"
        )
        return {"error": "invalid_json", "raw": text}


async def grade_exam(
    wav_path: Path,
    metadata: dict,
    report_path: Path,
    transcript_turns: list[dict],
    node_trace: list[dict],
    duration_seconds: float,
    exam_question: str,
    topic: str,
) -> None:
    logger.info(f"Grading exam → sending {wav_path.name} to {GRADER_MODEL}")

    transcript_str = _format_transcript(transcript_turns)
    node_trace_str = (
        "\n".join(f"  {n['ts']}  → {n['node']}" for n in node_trace)
        if node_trace else "(no transitions recorded)"
    )

    try:
        report = await asyncio.to_thread(
            _grade_exam_sync,
            wav_path, metadata, transcript_str, node_trace_str,
            exam_question, topic,
        )
    except Exception as e:
        logger.exception(f"Grading failed: {e}")
        report = {"error": str(e)}

    # No post-processing: the grader's scores and overall_score pass through unchanged.
    report["metadata"] = metadata
    report["audio_file"] = str(wav_path)
    report["transcript"] = transcript_turns
    report["node_trace"] = node_trace
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Report written → {report_path}")


# ─── Per-probe attempt counter ─────────────────────────────────────────────────

class LiveTranscript:
    """Shared turn list written by two TranscriptCollector instances —
    one before context_aggregator.user() (catches TranscriptionFrame from
    STT), another after TTS (catches TTSTextFrame). Consecutive same-speaker
    frames are coalesced."""

    def __init__(self) -> None:
        self.turns: list[dict] = []

    def append(self, speaker: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self.turns and self.turns[-1]["speaker"] == speaker:
            self.turns[-1]["text"] = (self.turns[-1]["text"] + " " + text).strip()
        else:
            self.turns.append({"speaker": speaker, "text": text})


class TranscriptCollector(FrameProcessor):
    """Captures frames into a shared LiveTranscript. The capture filter is
    passed in so we can deploy one instance for STT (TranscriptionFrame →
    STUDENT) and another for TTS (TTSTextFrame → EXAMINER)."""

    def __init__(self, live: LiveTranscript, *, capture_user: bool, capture_bot: bool):
        super().__init__()
        self._live = live
        self._capture_user = capture_user
        self._capture_bot = capture_bot

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self._capture_user and isinstance(frame, TranscriptionFrame):
            self._live.append("STUDENT", getattr(frame, "text", ""))
        elif self._capture_bot and isinstance(frame, TTSTextFrame):
            self._live.append("EXAMINER", getattr(frame, "text", ""))
        await self.push_frame(frame, direction)


# Live collector for the active session (set in run_bot)
_CURRENT_TRANSCRIPT: LiveTranscript | None = None


def get_live_transcript() -> list[dict]:
    """Server-facing accessor for the in-flight transcript."""
    if _CURRENT_TRANSCRIPT is None:
        return []
    return list(_CURRENT_TRANSCRIPT.turns)


class ProbeAttemptCounter(FrameProcessor):
    """Increments flow_manager.state['probe_attempts'][current_probe] on every
    final user transcription. Provides ground truth for the cap (the LLM still
    enforces it from prompt text — this is for logs, metadata, and debugging)."""

    def __init__(self):
        super().__init__()
        self._flow_manager: FlowManager | None = None

    def attach(self, flow_manager: FlowManager) -> None:
        self._flow_manager = flow_manager

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and self._flow_manager is not None:
            state = self._flow_manager.state
            # Global counter — incremented for every final student transcription,
            # used by handle_greeting_routing to detect phantom function calls
            # the LLM makes on its own greeting before the student has spoken.
            state.update({
                "student_utterance_count": state.get("student_utterance_count", 0) + 1
            })
            current = state.get("current_probe")
            if current:
                attempts: dict[str, int] = dict(state.get("probe_attempts", {}))
                attempts[current] = attempts.get(current, 0) + 1
                state.update({"probe_attempts": attempts})
                logger.info(f"Probe attempt: {current} #{attempts[current]}")
        await self.push_frame(frame, direction)


# ─── Pipecat bot setup ─────────────────────────────────────────────────────────



def _build_stt(
    stt_provider: str,
    stt_language: str | None,
    keyterms: list[str] | None = None,
):
    """Instantiate the configured STT service.

    Supported providers:
      - "elevenlabs":     SmartTurnElevenLabsSTT wrapping scribe_v2_realtime
        (includes our commit cooldown, echo flush, auto-reconnect glue).
      - "cartesia":       CartesiaSTTService with ink-whisper (cheap, low-latency,
        no Whisper-family hallucinations from noise; native keepalive).
      - "cartesia-ink-2": CartesiaSTTService with ink-2 (May 2026 release; #1 on
        Artificial Analysis streaming STT leaderboards; built-in turn detection,
        eager mode, semantic endpointing. English-only.)
      - "deepgram":       DeepgramSTTService with nova-3 (best keyterm-boosting
        for domain vocabulary).
      - "assemblyai":     AssemblyAISTTService with Universal-3 Pro (u3-rt-pro).
      - "xai":            XAISTTService using xAI's Grok voice WebSocket.

    Language defaults to 'en'. Pass empty string / None to enable auto-detect
    (only honoured by services that support it).
    """
    lang_code = (stt_language or "").strip() or None
    provider = (stt_provider or "elevenlabs").lower()

    if provider in ("cartesia", "cartesia-ink-whisper", "cartesia-ink-2"):
        if not os.getenv("CARTESIA_API_KEY"):
            raise RuntimeError("CARTESIA_API_KEY not set in .env (required for Cartesia STT)")
        # ink-2 is English-only at release (May 2026); ink-whisper is multilingual.
        cartesia_model = "ink-2" if provider == "cartesia-ink-2" else "ink-whisper"
        cartesia_lang = "en" if cartesia_model == "ink-2" else (lang_code or "en")
        logger.info(
            f"STT provider: cartesia / {cartesia_model} | "
            f"language={cartesia_lang}"
        )
        return CartesiaSTTService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            settings=CartesiaSTTService.Settings(
                model=cartesia_model,
                language=cartesia_lang,
            ),
        )

    if provider == "deepgram":
        if not os.getenv("DEEPGRAM_API_KEY"):
            raise RuntimeError("DEEPGRAM_API_KEY not set in .env (required for Deepgram STT)")
        try:
            from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
        except ImportError as e:
            raise RuntimeError(
                "Deepgram STT not installed — run: uv add 'pipecat-ai[deepgram]'"
            ) from e
        # Nova-3 supports keyterm-boosting for domain-specific vocabulary —
        # fixes the "child marriage" → "chat manage" class of mis-transcription.
        kt = keyterms or []
        logger.info(
            f"STT provider: deepgram / nova-3 | language={lang_code or 'en'} | "
            f"keyterms={len(kt)} ({kt[:8]}{'…' if len(kt) > 8 else ''})"
        )
        return DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                model="nova-3",
                language=lang_code or "en",
                keyterm=kt or None,
                smart_format=True,
                punctuate=True,
            ),
        )

    # Default: ElevenLabs
    if provider == "assemblyai":
        if not os.getenv("ASSEMBLYAI_API_KEY"):
            raise RuntimeError("ASSEMBLYAI_API_KEY not set in .env (required for AssemblyAI STT)")
        try:
            from pipecat.services.assemblyai.stt import AssemblyAISTTService
        except ImportError as e:
            raise RuntimeError(
                "AssemblyAI STT not installed — run: uv add 'pipecat-ai[assemblyai]'"
            ) from e
        logger.info(
            f"STT provider: assemblyai / universal-3-pro | language={lang_code or 'auto'} "
            f"| native turn-detection ON"
        )
        # u3-rt-pro has a proprietary end-of-turn detector that's strictly
        # better than Smart Turn V3 for our use case. Use AssemblyAI's native
        # turn-detection mode (vad_force_turn_endpoint=False) so the model
        # decides when the student is done, instead of forcing finals every
        # 100ms and stacking Smart Turn + LLM markers on top.
        #
        # Per the installed source (stt.py:188-195):
        #   vad_force_turn_endpoint=False is "AssemblyAI turn detection
        #   mode, u3-rt-pro only" — AssemblyAI's model controls turn endings,
        #   emits UserStarted/StoppedSpeakingFrame from STT.
        return AssemblyAISTTService(
            api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            vad_force_turn_endpoint=False,
            should_interrupt=True,
            settings=AssemblyAISTTService.Settings(
                model="u3-rt-pro",
                language=lang_code or None,
                # IB-domain biasing — boosts rubric vocabulary.
                keyterms_prompt=(
                    list(keyterms or [])
                    + ["IB", "MYP", "DP", "TOK", "rubric", "criterion"]
                ),
                # Higher confidence threshold = more conservative end-of-turn.
                # API default ~0.5; 0.7 reduces "Hi." / single-filler being
                # judged complete after a substantive question.
                end_of_turn_confidence_threshold=0.7,
            ),
        )

    if provider == "xai":
        if not os.getenv("XAI_API_KEY"):
            raise RuntimeError("XAI_API_KEY not set in .env (required for xAI STT)")
        try:
            from pipecat.services.xai.stt import XAISTTService
        except ImportError as e:
            raise RuntimeError(
                "xAI STT not installed — run: uv add 'pipecat-ai[xai]'"
            ) from e
        logger.info(
            f"STT provider: xai / grok | language={lang_code or 'auto'}"
        )
        return XAISTTService(
            api_key=os.getenv("XAI_API_KEY"),
            settings=XAISTTService.Settings(
                language=lang_code or None,
            ),
        )

    if provider != "elevenlabs":
        logger.warning(
            f"Unknown stt_provider {stt_provider!r}, falling back to ElevenLabs"
        )
    if not os.getenv("ELEVENLABS_API_KEY"):
        raise RuntimeError("ELEVENLABS_API_KEY not set in .env (required for ElevenLabs STT)")
    if lang_code:
        logger.info(f"STT provider: elevenlabs / scribe_v2_realtime | language locked to {lang_code!r}")
    else:
        logger.info("STT provider: elevenlabs / scribe_v2_realtime | language: auto-detect")
    return SmartTurnElevenLabsSTT(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        commit_strategy=CommitStrategy.MANUAL,
        params=ElevenLabsRealtimeSTTService.InputParams(
            language_code=lang_code,
        ),
        settings=ElevenLabsRealtimeSTTService.Settings(
            model="scribe_v2_realtime",
        ),
    )


# Tokens that don't add domain info to STT keyterms biasing. Used by
# _extract_keyterms for the Deepgram provider.
_STT_STOPWORDS: set[str] = {
    "what", "when", "where", "which", "while", "would", "could", "should",
    "their", "there", "these", "those", "this", "that", "with", "from",
    "into", "your", "yours", "about", "ask", "asking", "tell", "telling",
    "student", "students", "answer", "answers", "examiner", "exam", "topic",
    "probe", "prompt", "scaffold", "respond", "response", "based", "they",
    "them", "then", "than", "have", "has", "had", "been", "being", "will",
    "you", "yes", "the", "and", "for", "are", "but", "not", "one", "all",
    "any", "out", "off", "use", "way", "may", "say", "see",
}


def _extract_keyterms_from_flowspec(flow_spec: FlowSpec | None, limit: int = 100) -> list[str]:
    """Pull domain vocabulary from the exam's FlowSpec to bias STT models
    that support keyterm-prompting (Deepgram nova-3). For ElevenLabs and
    Cartesia we currently don't pass keyterms — Cartesia's API doesn't expose
    them, and ElevenLabs's keyterm field has a small documented benefit."""
    if not flow_spec:
        return []
    pool: set[str] = set()
    sources = [flow_spec.topic, flow_spec.exam_question]
    for p in flow_spec.probe_pool:
        sources.append(p.prompt)
        if p.scaffold:
            sources.append(p.scaffold)
    for text in sources:
        for w in re.findall(r"[A-Za-z][A-Za-z\-']{3,}", text):
            lw = w.lower()
            if lw in _STT_STOPWORDS or len(lw) > 30:
                continue
            pool.add(lw)
    return sorted(pool)[:limit]


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    session_id: str,
    llm_provider: str = "google",
    llm_model: str = "gemini-3.5-flash",
    stt_provider: str = "cartesia-ink-2",
    stt_language: str | None = "en",
    tts_voice: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    use_turn_markers: str = "auto",
):
    # Per-session log file — captures every loguru log line emitted during this
    # session so we can post-mortem any one run without scrolling stdout.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    session_log_path = LOGS_DIR / f"session_{session_id}.log"
    _session_log_sink_id = logger.add(
        str(session_log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
    )

    logger.info(
        f"Starting dynamic oral exam bot | session={session_id} | "
        f"llm={llm_provider}:{llm_model} | stt={stt_provider}"
    )

    keyterms = _extract_keyterms_from_flowspec(_FLOW_SPEC)
    stt = _build_stt(stt_provider, stt_language, keyterms=keyterms)

    cartesia_keys = _load_cartesia_keys()
    if not cartesia_keys:
        raise RuntimeError(
            "No Cartesia API key configured — set CARTESIA_API_KEY in .env"
        )
    logger.info(
        f"Cartesia TTS configured with {len(cartesia_keys)} key(s): "
        f"primary={FallbackCartesiaTTSService._mask(cartesia_keys[0])}"
        + (f", fallbacks={[FallbackCartesiaTTSService._mask(k) for k in cartesia_keys[1:]]}" if len(cartesia_keys) > 1 else " (no fallbacks)")
        + f" | voice={tts_voice}"
    )
    tts = FallbackCartesiaTTSService(
        api_keys=cartesia_keys,
        voice_id=tts_voice,
    )

    if llm_provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        llm = GroqLLMService(
            api_key=groq_key,
            settings=GroqLLMService.Settings(model=llm_model),
        )
    elif llm_provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        llm = OpenAILLMService(
            api_key=openai_key,
            settings=OpenAILLMService.Settings(model=llm_model),
        )
    elif llm_provider == "anthropic":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        llm = AnthropicLLMService(
            api_key=anthropic_key,
            settings=AnthropicLLMService.Settings(model=llm_model),
        )
    elif llm_provider == "cerebras":
        from pipecat.services.cerebras.llm import CerebrasLLMService
        cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
        if not cerebras_key:
            raise RuntimeError("CEREBRAS_API_KEY not set in .env")
        # gpt-oss-120b ships with reasoning_effort="medium" by default. That
        # adds ~300 reasoning tokens per response and several seconds of LLM
        # time even on short replies — wasted on an oral examiner that just
        # needs to acknowledge / probe / advance. "low" is the minimum
        # supported value (Cerebras does not allow disabling reasoning
        # entirely on gpt-oss-120b). The `extra` dict is spread into the
        # OpenAI-compatible chat-completions call by CerebrasLLMService.
        llm = CerebrasLLMService(
            api_key=cerebras_key,
            settings=CerebrasLLMService.Settings(
                model=llm_model,
                extra={"reasoning_effort": "low"},
            ),
        )
    else:
        llm = AssessVoxGoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            settings=AssessVoxGoogleLLMService.Settings(model=llm_model),
        )

    context = LLMContext()
    # ─── Turn-taking config (canonical Pipecat-flows shape) ─────────────────
    # Matches pipecat-flows/examples/insurance_quote.py verbatim. SileroVAD
    # at defaults + boolean `filter_incomplete_user_turns=True` is the
    # well-trodden path that the reference example uses with pipecat-flows.
    # Avoids the marker-leak we saw with the newer strategy-based API on
    # GoogleLLMService.
    #
    # ───────────── PREVIOUS (over-engineered) CONFIG — COMMENTED OUT ─────────────
    # We accumulated layers over several sessions to fight ElevenLabs-specific
    # quirks + Flash-Lite instruction-following failures. Most of these aren't
    # needed once we (a) use Cartesia STT as default and (b) gemini-3.5-flash.
    # Re-enable individual lines if a specific symptom returns.
    # ─────────────────────────────────────────────────────────────────────────
    # context_aggregator = LLMContextAggregatorPair(
    #     context,
    #     user_params=LLMUserAggregatorParams(
    #         vad_analyzer=SileroVADAnalyzer(params=VADParams(
    #             stop_secs=0.2,
    #             start_secs=0.3,
    #             confidence=0.7,
    #             min_volume=0.6,
    #         )),
    #         user_turn_strategies=FilterIncompleteUserTurnStrategies(
    #             start=[MinWordsUserTurnStartStrategy(min_words=3)],
    #             stop=[
    #                 TurnAnalyzerUserTurnStopStrategy(
    #                     turn_analyzer=LocalSmartTurnAnalyzerV3(
    #                         params=SmartTurnParams(
    #                             stop_secs=5.0,
    #                             pre_speech_ms=800,
    #                             max_duration_secs=12.0,
    #                         ),
    #                     ),
    #                 ),
    #             ],
    #             config=UserTurnCompletionConfig(
    #                 incomplete_short_timeout=5.0,
    #                 incomplete_long_timeout=10.0,
    #             ),
    #         ),
    #         user_mute_strategies=[
    #             MuteUntilFirstBotCompleteUserMuteStrategy(),
    #             FunctionCallUserMuteStrategy(),
    #         ],
    #         user_turn_stop_timeout=5.0,
    #     ),
    # )

    # LLM turn-marker mixin: asks the LLM to emit ✓/○/◐ before every spoken
    # reply, costing one extra LLM round-trip per user turn (~900 ms TTFB on
    # Anthropic). Worth it when STT has no native endpointing; pure waste when
    # STT does its own end-of-turn detection.
    #
    # use_turn_markers:
    #   "auto" (default) — ON for everything except STTs with native turn
    #                      detection (AssemblyAI Universal-3 Pro, Cartesia Ink-2)
    #   "on"             — force ON (debugging / fall-back path)
    #   "off"            — force OFF (skip the marker LLM call — caller takes
    #                      responsibility for endpointing)
    _stt_lower = (stt_provider or "").lower()
    _stt_has_native_endpointing = _stt_lower in ("assemblyai", "cartesia-ink-2")
    _marker_mode = (use_turn_markers or "auto").lower()
    if _marker_mode == "on":
        _use_marker_mixin = True
    elif _marker_mode == "off":
        _use_marker_mixin = False
    else:
        _use_marker_mixin = not _stt_has_native_endpointing
    logger.info(
        f"Turn detection: "
        f"mode={_marker_mode} | "
        f"{'LLM markers ON (filter_incomplete_user_turns=True)' if _use_marker_mixin else 'native endpointing (markers OFF)'} "
        f"| stt={_stt_lower}"
    )
    # Custom turn-analyzer config. Pipecat's default is SmartTurn V3 with
    # stop_secs=3.0 — the analyzer waits 3 s of silence before force-committing
    # a turn it scored INCOMPLETE. With the marker mixin OFF (mode=off /
    # auto with native-endpoint STT), there's nothing to overrule SmartTurn,
    # so any borderline INCOMPLETE turn pays the full 3 s slack.
    #
    # stop_secs=2.0 — tuned for the IB oral exam pace where students often
    # take 0.5-1.5 s mid-thought pauses. At 1.0 s SmartTurn was committing
    # turns mid-breath, fragmenting one student response into 3-4 separate
    # LLM calls and producing felt-latency stutter. 3.0 s (pipecat default)
    # was too loose — added a full 3 s wait whenever the analyzer scored
    # INCOMPLETE. 2.0 s absorbs natural mid-sentence pauses while keeping
    # the worst-case ceiling well below the old default. SmartTurn still
    # short-circuits instantly when it scores COMPLETE (most turns).
    _turn_analyzer = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(stop_secs=2.0)
    )
    _user_turn_strategies = UserTurnStrategies(
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=_turn_analyzer)],
    )

    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            filter_incomplete_user_turns=_use_marker_mixin,
            # Explicit SmartTurn config — see above. Override default
            # stop_secs=3.0 → 1.0 to keep slack bounded when markers are OFF.
            user_turn_strategies=_user_turn_strategies,
            # Defaults are 5 s / 10 s — tuned for snappy dialogue. For an oral
            # exam where students think mid-sentence, the 5 s timer fired as a
            # "Take your time, I'm listening" re-prompt right when the student
            # was actually still thinking. Stretch them so a real thinking
            # pause doesn't get nagged. (Only consulted when the mixin is on.)
            user_turn_completion_config=UserTurnCompletionConfig(
                incomplete_short_timeout=12.0,   # ↑ from 5 — mid-sentence pause
                incomplete_long_timeout=25.0,    # ↑ from 10 — "let me think"
            ),
        ),
    )

    # Publish for /api/sessions/{id}/transcript polling
    global _CURRENT_CONTEXT, _CURRENT_SESSION_ID, _CURRENT_TRANSCRIPT
    _CURRENT_CONTEXT = context
    _CURRENT_SESSION_ID = session_id
    _CURRENT_TRANSCRIPT = LiveTranscript()
    user_transcript_collector = TranscriptCollector(
        _CURRENT_TRANSCRIPT, capture_user=True, capture_bot=False,
    )
    bot_transcript_collector = TranscriptCollector(
        _CURRENT_TRANSCRIPT, capture_user=False, capture_bot=True,
    )
    rtvi = RTVIProcessor()
    audio_buffer = AudioBufferProcessor(sample_rate=16000, num_channels=1)
    attempt_counter = ProbeAttemptCounter()
    marker_strip = MarkerStripFilter()
    # Optional processors kept around but not in the active pipeline. Each
    # was added to chase a specific symptom; if it recurs, instantiate the
    # one you need and add it back to the Pipeline below.
    # latency_tracker = TurnLatencyTracker()
    # bargein_guard = BargeinGuard()
    # placeholder_filter = PlaceholderFilter()

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        # latency_tracker,             # observability — was emitting no logs
        attempt_counter,
        user_transcript_collector,     # captures TranscriptionFrame BEFORE the user
                                        # aggregator consumes it
        context_aggregator.user(),
        llm,
        marker_strip,                  # strips ✓/○/◐ from text frames — works
                                        # around upstream bug in GoogleLLMService
                                        # marker mixin (see MarkerStripFilter doc)
        # placeholder_filter,          # only relevant for Flash-Lite "Empty Message" leak
        tts,
        bot_transcript_collector,      # captures TTSTextFrame after TTS
        transport.output(),
        audio_buffer,
        # bargein_guard,               # FunctionCallResultFrame race — solved by
                                        # filter_incomplete_user_turns gating responses
        context_aggregator.assistant(),
    ])

    metrics_logger = MetricsLogObserver()
    latency_observer = UserBotLatencyObserver(enable_metrics=True)

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_seconds):
        logger.info(f"[LATENCY] voice-to-voice: {latency_seconds * 1000:.0f} ms")

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        # pipecat 1.3.0 renamed LatencyBreakdown.ttfb_metrics → .ttfb
        # (list[TTFBBreakdownMetrics]). See user_bot_latency_observer.py:107.
        parts = " | ".join(
            f"{m.processor.split('.')[-1]}={m.duration_secs * 1000:.0f}ms"
            for m in breakdown.ttfb
        )
        logger.info(f"[LATENCY] breakdown: {parts}")

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),
        observers=[RTVIObserver(rtvi), metrics_logger, latency_observer],
    )

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        global_functions=[
            FlowsFunctionSchema(
                name="off_topic_strike",
                handler=handle_global_off_topic,
                description=(
                    "Call when the student goes off-topic, after delivering the "
                    "redirect or warning specified by the OFF-TOPIC TRACKER. "
                    "Call at most once per student message."
                ),
                properties={},
                required=[],
            )
        ],
    )
    attempt_counter.attach(flow_manager)

    wav_path    = RECORDINGS_DIR / f"exam_{session_id}.wav"
    report_path = REPORTS_DIR    / f"report_{session_id}.json"
    started_at  = {"t": None}
    graded      = {"done": False}

    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio: bytes, sample_rate: int, num_channels: int):
        if graded["done"]:
            return
        graded["done"] = True
        if not audio:
            logger.warning("on_audio_data fired with no audio — skipping grading")
            return

        _write_wav(wav_path, audio, sample_rate, num_channels)
        logger.info(
            f"Audio saved → {wav_path} "
            f"({len(audio) / (sample_rate * 2 * num_channels):.1f}s)"
        )

        state = flow_manager.state
        strike_count = state.get("off_topic_count", 0) + state.get("greeting_strikes", 0)
        cancelled = strike_count >= 4 or state.get("greeting_strikes", 0) >= 3
        duration_s = (
            (datetime.now() - started_at["t"]).total_seconds()
            if started_at["t"] else 0.0
        )
        transcript_turns = _extract_transcript(context)
        node_trace = state.get("node_trace", [])
        budget_seconds = (
            (_FLOW_SPEC.duration_minutes if _FLOW_SPEC else 6) * 60
        )
        metadata = {
            "session_id": session_id,
            "student_name": state.get("name"),
            "strike_count": strike_count,
            "cancelled": cancelled,
            "duration_seconds": duration_s,
            "budget_seconds": budget_seconds,
            "time_warning_given": state.get("time_warning_given", False),
            "ended_early_for_time": state.get("ended_early_for_time", False),
            "covered_probes": state.get("covered_probes", []),
            "vague_terms": state.get("vague_terms", []),
            "probe_attempts": state.get("probe_attempts", {}),
            "voluntary_skips": state.get("voluntary_skips", 0),
            "consecutive_disengages_final": state.get("consecutive_disengages", 0),
            "revisit_count": state.get("revisit_count", 0),
            "cap_violations": state.get("cap_violations", []),
            "flow_spec_file": str(FLOWS_DIR / f"flow_{session_id}.json"),
        }
        await grade_exam(
            wav_path, metadata, report_path,
            transcript_turns, node_trace, duration_s,
            _FLOW_SPEC.exam_question, _FLOW_SPEC.topic,
        )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Student connected — starting audio recording")
        started_at["t"] = datetime.now()
        # Promote to flow_manager.state so handle_transition / _compute_time_state
        # can read it (started_at[] is local to this function).
        flow_manager.state.update({
            "started_at_iso": started_at["t"].isoformat(timespec="seconds"),
        })
        await audio_buffer.start_recording()
        if _PROVIDED_STUDENT_NAME:
            flow_manager.state.update({"name": _PROVIDED_STUDENT_NAME})
            logger.info(
                f"Using prefilled student name {_PROVIDED_STUDENT_NAME!r}; "
                "skipping STT name capture."
            )
            await flow_manager.initialize(
                create_known_name_greeting_node(flow_manager)
            )
        else:
            await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Student disconnected")
        await task.cancel()

    # Safety net: fires when a user turn was opened but no usable transcript
    # arrived within `user_turn_stop_timeout` (default 5s). With Cartesia STT
    # this can happen if the audio track briefly hiccups. Don't crash — just
    # log it so we can see in the per-session log.
    @context_aggregator.user().event_handler("on_user_turn_stop_timeout")
    async def on_user_turn_stop_timeout(aggregator):
        logger.warning(
            "User turn stop-timeout fired (turn opened but no final transcript)"
        )

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    try:
        await runner.run(task)
    finally:
        # Always dump a transcript snapshot — even if the session was cancelled
        # before grading ran. This gives a paste-free way to inspect any session
        # (failed, cancelled, or graded) by reading transcripts/transcript_<id>.json.
        try:
            TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_path = TRANSCRIPTS_DIR / f"transcript_{session_id}.json"
            snapshot = {
                "session_id":      session_id,
                "ended_at":        datetime.now().isoformat(timespec="seconds"),
                "student_name":    flow_manager.state.get("name"),
                "current_probe":   flow_manager.state.get("current_probe"),
                "node_trace":      flow_manager.state.get("node_trace", []),
                "covered_probes":  flow_manager.state.get("covered_probes", []),
                "off_topic_count": flow_manager.state.get("off_topic_count", 0),
                "greeting_strikes": flow_manager.state.get("greeting_strikes", 0),
                "turns":           _extract_transcript(context),
                "log_file":        str(session_log_path),
                "recording_file":  str(wav_path),
                "report_file":     str(report_path) if report_path.exists() else None,
                "flow_spec_file":  str(FLOWS_DIR / f"flow_{session_id}.json"),
            }
            snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str))
            logger.info(f"Transcript snapshot saved → {snapshot_path}")
        except Exception:
            logger.exception("Failed to write transcript snapshot")
        # Detach the per-session log sink so the next session opens a fresh file.
        try:
            logger.remove(_session_log_sink_id)
        except Exception:
            pass


async def run_session(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    session_id: str,
    flow_spec: FlowSpec,
    custom_grading_prompt: str | None = None,
    student_name: str | None = None,
    llm_provider: str = "google",
    llm_model: str = "gemini-3.5-flash",
    stt_provider: str = "cartesia-ink-2",
    stt_language: str | None = "en",
    tts_voice: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    use_turn_markers: str = "auto",
) -> None:
    """Server-friendly entry: caller provides transport, session_id, and a
    pre-generated FlowSpec. CLI mode uses bot() instead."""
    set_active_flow(
        flow_spec,
        custom_grading_prompt=custom_grading_prompt,
        student_name=student_name,
    )
    logger.info(
        f"Flow ready | session={session_id} | {len(flow_spec.probe_pool)} probes | "
        f"required={flow_spec.required_probes} | question={flow_spec.exam_question!r}"
    )
    await run_bot(
        transport, runner_args, session_id,
        llm_provider=llm_provider, llm_model=llm_model,
        stt_provider=stt_provider,
        stt_language=stt_language,
        tts_voice=tts_voice,
        use_turn_markers=use_turn_markers,
    )


async def bot(runner_args: RunnerArguments):
    """CLI entry point: generate + validate flow, then run the pipeline."""
    topic = os.getenv(
        "EXAM_TOPIC",
        "Should gene editing in humans be allowed?",
    )
    duration = int(os.getenv("EXAM_DURATION_MINUTES", "6"))
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Generating flow | session={session_id} | topic={topic!r}")
    flow_spec = await _generate_and_validate(topic, duration, session_id)
    set_active_flow(flow_spec)

    transport_params = {
        "daily": lambda: DailyParams(
            api_key=os.getenv("DAILY_API_KEY"),
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
            video_out_bitrate=2_000_000,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args, session_id)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
