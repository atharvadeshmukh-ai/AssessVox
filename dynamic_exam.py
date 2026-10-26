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
import contextvars
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
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMTextFrame,
    MetricsFrame,
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
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)
# Kept-but-unused imports — used by the commented-out turn-management config
# in run_bot. Leave imported so re-enabling lines is a no-op edit.
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    FilterIncompleteUserTurnStrategies,
    ExternalUserTurnStrategies,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_mute import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
    FunctionCallUserMuteStrategy,
)
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    STTUsageMetricsData,
    TTSUsageMetricsData,
)
from pricing import SessionUsage
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService
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
        # Gemini rejects contexts that start with a model turn.
        # Happens when an async tool result re-triggers the LLM but the
        # preceding user turn got dropped from the context window.
        #
        # context.messages can contain either dict-shaped LLMStandardMessage
        # or LLMSpecificMessage dataclasses (whose .message field holds the
        # provider-native payload — for Gemini that's a `Content` object).
        # Probe both shapes safely.
        if context.messages:
            first = context.messages[0]
            role = None
            if isinstance(first, dict):
                role = first.get("role")
            else:
                # LLMSpecificMessage(llm=..., message=<provider-native>).
                # For Gemini the inner message exposes `.role`; for other
                # providers it may differ — getattr-with-default is fine.
                inner = getattr(first, "message", None)
                role = getattr(inner, "role", None) or getattr(first, "role", None)
            if role == "model":
                context.messages.insert(0, {
                    "role": "user",
                    "parts": [{"text": "[conversation continued]"}]
                })
        await super()._process_context(context)

    async def stop_ttfb_metrics(self, *, end_time: float | None = None):
        if self._ttft_armed:
            await super().stop_ttfb_metrics(end_time=end_time)

    async def push_frame(self, frame, direction=None):
        # Arm TTFB on the first real output — either a text token (normal
        # speech response) or a function call (tool-use response). Without
        # the FunctionCallInProgressFrame arm, tool-use calls produce no
        # LLMTextFrame so _ttft_armed stays False and the entire Gemini
        # thinking time (sometimes 7s+) is silently swallowed from metrics.
        if not self._ttft_armed and (
            (isinstance(frame, LLMTextFrame) and frame.text)
            or isinstance(frame, FunctionCallInProgressFrame)
        ):
            self._ttft_armed = True
        if direction is not None:
            await super().push_frame(frame, direction)
        else:
            await super().push_frame(frame)


# ─── LLM call dumper ─────────────────────────────────────────────────────────
#
# Per-session JSONL of every LLM round-trip: input messages, system prompt,
# tools list, output text, tool_use blocks. Powers the /sessions dashboard's
# "raw LLM I/O" inline expander.
#
# Hooked via _process_context (captures input) + push_frame (accumulates
# streamed output text + tool calls; flushes record on LLMFullResponseEndFrame).
# Designed as a mixin so it works for every provider (Anthropic, Google,
# OpenAI, Groq, Cerebras) without per-provider subclassing.

class LLMCallDumperMixin:
    """Append a JSONL record of every LLM round-trip to llm_calls/calls_<session>.jsonl.

    Captures: timestamp, provider class, model, input messages (via the
    adapter's own logging serializer), output text concatenated across stream
    chunks, and any tool_use blocks the LLM emitted.

    Safe to mix into any pipecat LLM service that uses `_process_context` and
    pushes `LLMFullResponseStartFrame`/`LLMFullResponseEndFrame`. All other
    providers in this codebase follow that contract.
    """

    def __init__(
        self,
        *args,
        _dump_path: Path | None = None,
        _dump_kind: str = "conversation",   # or "summarization"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._dump_path = _dump_path
        self._dump_kind = _dump_kind
        self._dump_lock = asyncio.Lock()
        # Active record (one in flight at a time per LLM call).
        self._cur_record: dict | None = None
        self._cur_text_chunks: list[str] = []
        self._cur_tool_calls: list[dict] = []
        self._cur_usage: dict | None = None
        # Finished record waiting for its token usage: pipecat pushes the usage
        # MetricsFrame only AFTER LLMFullResponseEndFrame, so writing at End
        # always recorded usage=None ("tokens —" on the dashboard).
        self._pending_record: dict | None = None

    async def _process_context(self, context):
        if self._pending_record is not None:
            # Next call started before usage arrived — write without it.
            rec, self._pending_record = self._pending_record, None
            asyncio.create_task(self._write_record(rec))
        if self._dump_path is not None:
            try:
                adapter = self.get_llm_adapter()
                # `get_messages_for_logging` returns a safely-redacted list
                # the SDK itself uses for debug logs. Plenty good for our
                # inspector and avoids leaking large internal cache state.
                messages = adapter.get_messages_for_logging(context)
                # Some adapters return strings already; normalize to list.
                if isinstance(messages, str):
                    try:
                        messages = json.loads(messages)
                    except Exception:
                        messages = [{"raw": messages}]
                # Serialize tool definitions via the adapter so we get the
                # actual function-declaration JSON the LLM sees, not the
                # opaque ToolsSchema object reference.
                tools = getattr(context, "tools", None)
                tools_repr = []
                try:
                    if tools is not None:
                        from pipecat.adapters.schemas.tools_schema import NOT_GIVEN
                        if tools is not NOT_GIVEN:
                            # adapter.to_provider_tools_format returns the
                            # provider-specific list (e.g. Gemini's
                            # [{function_declarations:[...]}]).
                            formatted = adapter.to_provider_tools_format(tools)
                            tools_repr = json.loads(json.dumps(formatted, default=str))
                except Exception:
                    try:
                        # Fallback: grab standard_tools directly
                        st = getattr(tools, "standard_tools", None)
                        if st:
                            tools_repr = json.loads(json.dumps([
                                {"name": fn.name, "description": getattr(fn, "description", "")}
                                for fn in st
                            ], default=str))
                    except Exception:
                        tools_repr = [{"repr_failed": str(tools)[:200]}]
                system = getattr(context, "system_message", None) or getattr(
                    self, "_system_instruction", None
                ) or getattr(self._settings, "system_instruction", None) if hasattr(self, "_settings") else None
                self._cur_record = {
                    "ts": datetime.now().isoformat(timespec="milliseconds"),
                    "kind": self._dump_kind,
                    "service": type(self).__name__,
                    "model": getattr(self._settings, "model", None) if hasattr(self, "_settings") else None,
                    "input": {
                        "system": str(system) if system else None,
                        "messages": messages,
                        "tools": tools_repr,
                    },
                    "output": {"text": "", "tool_calls": []},
                    "usage": None,
                }
                self._cur_text_chunks = []
                self._cur_tool_calls = []
                self._cur_usage = None
            except Exception as e:
                logger.warning(f"LLMCallDumper: input capture failed: {e}")
                self._cur_record = None
        return await super()._process_context(context)

    async def run_function_calls(self, function_calls):
        # Capture tool_use blocks here — for many providers (notably Gemini)
        # the LLM emits tool_use as a streamed content_block, and pipecat
        # routes it through this method without ever pushing a
        # FunctionCallInProgressFrame downstream. The push_frame hook
        # therefore misses it. This is the canonical capture point.
        if self._cur_record is not None and function_calls:
            try:
                for fc in function_calls:
                    self._cur_tool_calls.append({
                        "name": getattr(fc, "function_name", None),
                        "arguments": getattr(fc, "arguments", None),
                        "tool_call_id": getattr(fc, "tool_call_id", None),
                    })
            except Exception:
                pass
        return await super().run_function_calls(function_calls)

    @staticmethod
    def _usage_from_metrics(frame) -> dict | None:
        """LLM token usage from a MetricsFrame, or None if it carries none."""
        for entry in (getattr(frame, "data", None) or []):
            if entry.__class__.__name__ == "LLMUsageMetricsData":
                val = getattr(entry, "value", None) or entry
                return {
                    "prompt_tokens": getattr(val, "prompt_tokens", None),
                    "completion_tokens": getattr(val, "completion_tokens", None),
                    "total_tokens": getattr(val, "total_tokens", None),
                    "cache_read_input_tokens": getattr(val, "cache_read_input_tokens", None),
                    "cache_creation_input_tokens": getattr(val, "cache_creation_input_tokens", None),
                    "reasoning_tokens": getattr(val, "reasoning_tokens", None),
                }
        return None

    async def _flush_pending_later(self, rec: dict, delay: float = 2.0):
        """Write a held record even if its usage never arrives."""
        await asyncio.sleep(delay)
        if self._pending_record is rec:
            self._pending_record = None
            await self._write_record(rec)

    async def push_frame(self, frame, direction=None):
        # Usage for the call that just finished — attach it and write.
        if self._pending_record is not None and isinstance(frame, MetricsFrame):
            try:
                usage = self._usage_from_metrics(frame)
                if usage is not None:
                    rec, self._pending_record = self._pending_record, None
                    rec["usage"] = usage
                    asyncio.create_task(self._write_record(rec))
            except Exception:
                pass
        # Accumulate streamed text + tool calls between Start and End frames.
        if self._cur_record is not None:
            if isinstance(frame, LLMTextFrame) and frame.text:
                self._cur_text_chunks.append(frame.text)
            elif isinstance(frame, FunctionCallInProgressFrame):
                # Tool call emitted by the LLM via downstream frame. Some
                # providers DO push this; capture if so (de-duped by
                # tool_call_id at write time would be cleaner but in practice
                # only one of the two paths fires per call).
                try:
                    seen_ids = {tc.get("tool_call_id") for tc in self._cur_tool_calls}
                    new_id = getattr(frame, "tool_call_id", None)
                    if new_id not in seen_ids:
                        self._cur_tool_calls.append({
                            "name": getattr(frame, "function_name", None),
                            "arguments": getattr(frame, "arguments", None),
                            "tool_call_id": new_id,
                        })
                except Exception:
                    pass
            elif isinstance(frame, MetricsFrame):
                # Some providers report usage before End — keep it if so.
                try:
                    usage = self._usage_from_metrics(frame)
                    if usage is not None:
                        self._cur_usage = usage
                except Exception:
                    pass
            elif isinstance(frame, LLMFullResponseEndFrame):
                # Flush.
                rec = self._cur_record
                rec["output"]["text"] = "".join(self._cur_text_chunks)
                rec["output"]["tool_calls"] = list(self._cur_tool_calls)
                rec["usage"] = self._cur_usage
                self._cur_record = None
                self._cur_text_chunks = []
                self._cur_tool_calls = []
                self._cur_usage = None
                # Best-effort async write — never block the audio pipeline.
                if rec["usage"] is not None:
                    asyncio.create_task(self._write_record(rec))
                else:
                    # Usage normally follows End; hold the record until it lands.
                    self._pending_record = rec
                    asyncio.create_task(self._flush_pending_later(rec))
        # Defer to the existing push_frame chain (incl. AssessVoxGoogleLLMService).
        if direction is not None:
            await super().push_frame(frame, direction)
        else:
            await super().push_frame(frame)

    async def _write_record(self, record: dict):
        if self._dump_path is None:
            return
        try:
            async with self._dump_lock:
                # Synchronous file write inside async lock — JSONL append is
                # one syscall per call (cheap, and we already paid the LLM
                # round-trip cost). Avoids needing aiofiles as a dep.
                self._dump_path.parent.mkdir(parents=True, exist_ok=True)
                with self._dump_path.open("a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"LLMCallDumper: write failed: {e}")


class UsageCollectorObserver(BaseObserver):
    """Tallies LLM tokens / TTS characters / STT seconds for ONE session.

    Observers are per-PipelineTask, so concurrent exams never mix — unlike the
    session log files, which receive every exam's lines. The tally is saved to
    usage/usage_<session>.json at session end and priced by the dashboard.
    """

    def __init__(self, usage: SessionUsage, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self._usage = usage

    async def on_push_frame(self, data: FramePushed):
        if not isinstance(data.frame, MetricsFrame):
            return
        for m in data.frame.data:
            try:
                if isinstance(m, LLMUsageMetricsData):
                    v = m.value
                    self._usage.add_llm(
                        "conversation", m.model,
                        prompt=v.prompt_tokens or 0,
                        completion=v.completion_tokens or 0,
                        cache_read=v.cache_read_input_tokens or 0,
                        cache_write=v.cache_creation_input_tokens or 0,
                        reasoning=v.reasoning_tokens or 0,
                    )
                elif isinstance(m, TTSUsageMetricsData):
                    self._usage.add_tts(m.model, m.value)
                elif isinstance(m, STTUsageMetricsData):
                    self._usage.add_stt(m.model, m.value.audio_seconds)
            except Exception as e:
                logger.warning(f"UsageCollector: failed to record {type(m).__name__}: {e}")


def _gemini_usage(response, model: str) -> dict | None:
    """Pull token usage off a google-genai response and log it in the same
    '📊 ... LLM TOKEN USAGE' shape pipecat uses (so log-based estimates see it)."""
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return None
    usage = {
        "model": model,
        "prompt": um.prompt_token_count or 0,
        "completion": um.candidates_token_count or 0,
        "reasoning": getattr(um, "thoughts_token_count", None) or 0,
        "cache_read": getattr(um, "cached_content_token_count", None) or 0,
    }
    logger.info(
        f"📊 [Grader] LLM TOKEN USAGE ({model}): prompt: {usage['prompt']}, "
        f"completion: {usage['completion']}, cache_read: {usage['cache_read']}, "
        f"reasoning: {usage['reasoning']}"
    )
    return usage


# Concrete dumper-enabled subclasses, one per provider. Mixin must precede
# the provider class in MRO so its method overrides take effect.
class _DumpingAssessVoxGoogleLLMService(LLMCallDumperMixin, AssessVoxGoogleLLMService):
    pass


class _DumpingAnthropicLLMService(LLMCallDumperMixin, AnthropicLLMService):
    pass


class _DumpingOpenAILLMService(LLMCallDumperMixin, OpenAILLMService):
    pass


class _DumpingGroqLLMService(LLMCallDumperMixin, GroqLLMService):
    pass


from websockets.asyncio.client import connect as _websocket_connect



class FallbackCartesiaTTSService(CartesiaTTSService):
    """Cartesia TTS that rotates through a list of API keys when one is
    rejected (HTTP 402 credits-exhausted, 401 invalid, 403 forbidden).

    The first key in the list is used by default. If the WebSocket
    handshake fails with a key-rejection error, the next key is tried.
    Once a key works, it sticks for the rest of the session — the index
    only advances, never resets.
    """

    def __init__(self, *, api_keys: list[str], voice_id: str, model: str | None = None, **kwargs):
        if not api_keys:
            raise ValueError("FallbackCartesiaTTSService needs at least one api_key")
        settings = (
            CartesiaTTSService.Settings(voice=voice_id, model=model)
            if model
            else CartesiaTTSService.Settings(voice=voice_id)
        )
        super().__init__(
            api_key=api_keys[0],
            settings=settings,
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

            # Strip markdown emphasis/code characters the LLM sometimes emits
            # despite the "no markdown" instruction (*bold*, **word**, `code`).
            # These are never spoken and surface as stray '*' in the transcript
            # and TTS. Removing the bare characters is safe for a voice feed.
            cur = frame.text if marker_found else text
            if "*" in cur or "`" in cur:
                # Remove ONLY the markdown characters themselves and keep every
                # other character — crucially spaces. Dropping a whole fragment
                # that is just "* " (streaming splits emphasis off on its own)
                # would glue neighbours together: "really important" ->
                # "reallyimportant". So we never drop the frame here.
                # Also leave an asterisk that sits directly between digits alone
                # so multiplication ("3*4") isn't mangled into "34".
                cleaned = re.sub(r"(?<!\d)\*(?!\d)", "", cur).replace("`", "")
                frame.text = cleaned

        await self.push_frame(frame, direction)


from pipecat_flows import FlowManager, FlowsFunctionSchema, FlowArgs, NodeConfig

from flow_planner import generate_flow, planned_question_count
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
LLM_CALLS_DIR   = Path(__file__).parent / "llm_calls"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"

# Which exam the current asyncio task belongs to — set in run_bot and read by
# each per-session log file's filter. A task that misses it (e.g. created
# before run_bot) logs as None and its lines go to every open session file.
_LOG_SESSION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "assessvox_log_session", default=None
)

# ─── Module-level flow state (populated at startup) ────────────────────────────
# NOTE: these are module globals, so they are shared by all voice bots in the
# process. That is safe for one exam at a time, but concurrent voice exams in the
# same process would overwrite each other. (A ContextVar was tried but does not
# propagate into pipecat's transport event handlers; the correct per-session fix
# is to carry this on flow_manager.state — a tested follow-up.)
_FLOW_SPEC: FlowSpec | None = None
_PROBE_FACTORIES: dict[str, Callable] = {}
_REQUIRED_PROBES: set[str] = set()
_CUSTOM_GRADING_PROMPT: str | None = None

# Set when a session is running, so the server can read the live transcript.
# _CURRENT_* remain as a single-session fallback; _LIVE_SESSIONS is the
# per-session registry (keyed by session_id) that makes CONCURRENT voice exams
# each expose their OWN live transcript to the /transcript endpoint instead of
# all reading one shared global (which would leak one student's conversation
# into another's panel).
_CURRENT_CONTEXT = None
_CURRENT_SESSION_ID: str | None = None
_LIVE_SESSIONS: dict = {}   # session_id -> {"context": ctx, "transcript": LiveTranscript}


def _register_live_session(session_id, context, transcript) -> None:
    _LIVE_SESSIONS[session_id] = {"context": context, "transcript": transcript}


def _unregister_live_session(session_id) -> None:
    _LIVE_SESSIONS.pop(session_id, None)


def get_current_transcript(session_id: str | None = None) -> list[dict]:
    """Return diarized turns from a session's LLMContext (by session_id when
    given, else the single-session fallback). Empty list if none."""
    ctx = None
    if session_id is not None and session_id in _LIVE_SESSIONS:
        ctx = _LIVE_SESSIONS[session_id]["context"]
    if ctx is None:
        ctx = _CURRENT_CONTEXT
    if ctx is None:
        return []
    try:
        return _extract_transcript(ctx)
    except Exception:
        return []


# Optional teacher-provided student name. When set, run_bot skips the
# STT-driven name-capture greeting entirely.
_PROVIDED_STUDENT_NAME: str | None = None
# IB programme of the chosen template ("MYP" | "DP" | ...). None for free-text
# questions, which keep the original MYP framing.
_PROGRAMME: str | None = None


def set_active_flow(
    flow_spec: FlowSpec,
    custom_grading_prompt: str | None = None,
    student_name: str | None = None,
    programme: str | None = None,
) -> None:
    """Configure module-level state for a session. Called by the server before
    spawning a bot, or by bot() in CLI mode after generating the flow."""
    global _FLOW_SPEC, _PROBE_FACTORIES, _REQUIRED_PROBES, _CUSTOM_GRADING_PROMPT
    global _PROVIDED_STUDENT_NAME, _PROGRAMME
    _FLOW_SPEC = flow_spec
    _PROBE_FACTORIES = build_probe_factories(flow_spec)
    _REQUIRED_PROBES = set(flow_spec.required_probes)
    _CUSTOM_GRADING_PROMPT = custom_grading_prompt
    _PROVIDED_STUDENT_NAME = (student_name or "").strip() or None
    _PROGRAMME = (programme or "").strip().upper() or None


# ─── Per-session accessors (scalability / concurrency) ─────────────────────────
# The per-session flow data is ALSO seeded onto flow_manager.state in run_bot.
# Unlike a module global (shared by every concurrent exam in the process) or a
# ContextVar (which does NOT reach pipecat's transport event handlers),
# flow_manager.state is a per-exam dict that pipecat makes available to EVERY
# node builder and handler. These accessors read it from there, falling back to
# the module global when flow_manager is absent/unseeded — so single-exam
# behaviour is byte-identical, while concurrent exams each read their OWN state.
def _fspec(fm=None):
    if fm is not None:
        v = fm.state.get("flow_spec")
        if v is not None:
            return v
    return _FLOW_SPEC


def _pfact(fm=None):
    if fm is not None:
        v = fm.state.get("probe_factories")
        if v:
            return v
    return _PROBE_FACTORIES


def _rprobes(fm=None):
    if fm is not None:
        v = fm.state.get("required_probes")
        if v is not None:
            return v
    return _REQUIRED_PROBES


def _cgp(fm=None):
    if fm is not None and "custom_grading_prompt" in fm.state:
        return fm.state.get("custom_grading_prompt")
    return _CUSTOM_GRADING_PROMPT


def _psn(fm=None):
    if fm is not None and "provided_student_name" in fm.state:
        return fm.state.get("provided_student_name")
    return _PROVIDED_STUDENT_NAME


def _opening_script(fm=None) -> str:
    """Scripted opening for role-play / listen-and-respond exams ('' otherwise)."""
    spec = _fspec(fm)
    if spec is None or spec.question_style == "discussion":
        return ""
    return (spec.opening_script or "").strip()


async def _speak_opening_script(flow_manager) -> None:
    """Read the scripted opening word for word (TTS, no LLM paraphrase). Called
    by every handler that moves into the opening node."""
    script = _opening_script(flow_manager)
    if not script:
        return
    try:
        await flow_manager.task.queue_frame(TTSSpeakFrame(script))
        logger.info(f"Spoke opening script ({_fspec(flow_manager).question_style})")
    except Exception as e:
        logger.warning(f"opening script TTSSpeakFrame failed: {e}")


def _speak_question_line(fm=None) -> str:
    """The 'THE EXAM QUESTION' line of the greeting / name-confirmation speak
    instructions. With a scripted opening, the LLM only acknowledges — the
    script itself is spoken by the handler."""
    if _opening_script(fm):
        return (
            "THE EXAM QUESTION: scripted — it is played automatically right after "
            "your speak field. For 'ready' / 'confirmed' your speak must be ONLY a "
            "warm acknowledgement of 3-6 words (e.g. 'Wonderful, let's begin.'). "
            "Do NOT say the question, scenario or message yourself.\n"
        )
    spec = _fspec(fm)
    return f"THE EXAM QUESTION (shorten to its core if long): '{spec.exam_question if spec else ''}'\n"


def _image_note(fm=None) -> str:
    """Private description of the picture on the student's screen."""
    spec = _fspec(fm)
    desc = (spec.image_description or "").strip() if spec else ""
    if not desc:
        return ""
    return (
        "PICTURE ON THE STUDENT'S SCREEN (private — for you only): " + desc + "\n"
        "The student can see this picture; you cannot. Use this description only to "
        "check what they say and to ask about details they have missed. NEVER read it "
        "out, describe the picture for them, or hint at details they have not "
        "mentioned. Their description must come from them.\n\n"
    )


def _format_banner(fm=None) -> str:
    """Exam-format rules for role-play / listen-and-respond, shown at every step.
    They override the opinion-style wording of the shared probe primitives."""
    spec = _fspec(fm)
    if spec is None or spec.question_style == "discussion":
        return ""
    brief = (spec.examiner_brief or "").strip()
    if spec.question_style == "role_play":
        return (
            "EXAM FORMAT — ROLE-PLAY (overrides any probe wording below): you are "
            "playing a character, not interviewing. Stay in character for the whole "
            "exam and speak as the character would, in 1-2 short natural sentences. "
            "Carry out each probe TASK in character. Do NOT ask for the student's "
            "personal opinion, a counter-argument or 'what is YOUR view' unless the "
            "TASK says so. Never correct their language. Only step out of character "
            "for the closing.\n"
            + (f"EXAMINER BRIEF: {brief}\n" if brief else "")
            + "\n"
        )
    script = (spec.opening_script or "").strip()
    return (
        "EXAM FORMAT — LISTEN AND RESPOND: a short message was read aloud to the "
        "student at the start. Base comprehension follow-ups on that message and "
        "never give the answers away. If the student asks to hear it again, read it "
        "ONCE more word for word (only once in the whole exam).\n"
        + (f"MESSAGE AS READ ALOUD: {script}\n" if script else "")
        + (f"EXAMINER BRIEF: {brief}\n" if brief else "")
        + "\n"
    )


def _opening_steps(fm, name: str, exam_question: str) -> str:
    """Opening-node instructions for the exam's format."""
    style = _fspec(fm).question_style if _fspec(fm) else "discussion"
    if style == "role_play":
        return (
            f"The role-play scenario has ALREADY been given to {name} (see your "
            "previous message). Do NOT repeat it.\n"
            f"SCENARIO (for your reference): '{exam_question}'\n\n"
            "STEP 1 — STAY IN CHARACTER.\n"
            f"Reply to {name} as your character would, in 1-2 short natural "
            f"sentences. Let {name} lead their side. If they seem unsure what to "
            "do, remind them of their goal in ONE sentence and invite them to start.\n\n"
            "STEP 2 — ACCEPT OR PROBE.\n"
            f"Once {name} has made their main request or point in role with at "
            "least one detail (what, why, when or how), call advance_from_probe "
            "with depth='surface' or 'developed'. Do NOT demand perfect language.\n\n"
        )
    if style == "listen_respond":
        return (
            f"The message and a first question about it have ALREADY been given to "
            f"{name} (see your previous message). Do NOT read the message again "
            "unless they ask.\n"
            f"TASK (for your reference): '{exam_question}'\n\n"
            "STEP 1 — CHECK UNDERSTANDING.\n"
            f"Listen to {name}'s answer about the message. If they clearly "
            "misunderstood or ask to hear it again, read the message ONCE more, "
            "word for word, then re-ask the question.\n\n"
            "STEP 2 — ACCEPT OR PROBE.\n"
            f"If {name} gets the main point (even partly, in simple words), call "
            "advance_from_probe with depth='surface' or 'developed'. After one "
            "repeat, accept whatever they say. Never give the answer away.\n\n"
        )
    return (
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
    )


def _examiner_title(fm=None) -> str:
    """'IB DP' / 'IB PYP' / 'IB MYP' — how the examiner introduces itself."""
    prog = fm.state.get("programme") if fm is not None and "programme" in fm.state else _PROGRAMME
    return {"DP": "IB DP", "PYP": "IB PYP"}.get(prog or "", "IB MYP")


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
    "   NEVER ask a question that seeks the same information as one you already "
    "   asked, even in different words, and NEVER ask about something the student "
    "   already explained (even in passing) — build on it instead. "
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
    "   Never thank the student for the session or say goodbye yourself while "
    "   calling a function — the closing step does that, once. "
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
    "    if there is a long pause, but otherwise wait. "
    "16. PLAIN, STUDENT-FRIENDLY LANGUAGE — mandatory: these students are 11–16 and "
    "    many are not native English speakers. Speak in simple, everyday words and "
    "    SHORT sentences. Avoid academic jargon, complex or rare vocabulary, idioms, "
    "    and long multi-clause sentences. Ask ONE clear thing at a time. If a "
    "    subject term is unavoidable, explain it in a few plain words right after. "
    "    Aim for language a 12-year-old can follow easily on first hearing — if a "
    "    simpler word works, always use it. If a student seems confused by your "
    "    wording, rephrase it even more simply. "
    "17. STOPPING ON REQUEST — mandatory: if the student clearly asks to STOP or "
    "    FINISH the whole exam (e.g. 'I want to stop', 'can we end now', 'I'm "
    "    done', 'I'm finished', 'that's all I have', 'I don't want to continue', "
    "    'end the exam'), speak ONE short warm closing line and call end_exam "
    "    right away. Respect their wish — do NOT argue, bargain, or ask them to "
    "    continue. (Only skipping one question = advance_from_probe, not end_exam.)"
)
ROLE_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]

_MYP_AUDIENCE = "an IB MYP language class (students aged 11 to 16)"
_PROGRAMME_AUDIENCE = {
    "DP": "an IB Diploma Programme (DP) class (students aged 16 to 19)",
    "PYP": "an IB PYP class (students aged 3 to 12)",
}


def _role_messages_for(programme: str | None) -> list[dict]:
    """ROLE_MESSAGES with the audience line matching the template's programme."""
    audience = _PROGRAMME_AUDIENCE.get(programme or "")
    if not audience:
        return list(ROLE_MESSAGES)
    return [{"role": "system", "content": SYSTEM_PROMPT.replace(_MYP_AUDIENCE, audience)}]

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

def _off_topic_redirect_text(count: int, name: str, fm=None) -> str:
    """The canned line spoken by the off-topic handler (NOT by the LLM).

    Kept out of the LLM prompt because Gemini Flash has been observed to
    call the function and then return zero completion tokens, leaving the
    student in silence. The handler speaks this via TTSSpeakFrame so the
    redirect is guaranteed regardless of Gemini's whim.
    """
    redirect = _fspec(fm).redirect_when_off_topic if _fspec(fm) else (
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


def _off_topic_instruction(count: int, name: str, fm=None) -> str:
    """System-prompt fragment that tells the LLM WHEN to call the function.

    Note: the redirect TEXT is NOT included — the handler speaks it directly.
    This avoids the 0-completion-tokens deadlock where Gemini called the
    function first, saw next_action='wait_for_student', and refused to speak
    the redirect at all.
    """
    # If the student is asking ABOUT the exam (META), the LLM should explain
    # briefly rather than strike. Inject the exam's own topic + skills so the
    # explanation is precise to this assessment, not a generic stock line.
    topic = (_fspec(fm).topic if _fspec(fm) else "this assessment").strip()
    skills = (_fspec(fm).skills_assessed if _fspec(fm) else []) or []
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


def _state_block(flow_manager: FlowManager) -> str:
    """Render the load-bearing facts as a compact block.

    Pipecat's auto-summarizer compresses verbatim history, which can drop
    rare-but-important facts ("student's position", "name confirmed") across
    repeated summary cycles. Mitigation per the 2026 voice-agent playbook:
    keep load-bearing facts in a structured State block prepended to the
    system prompt every turn. State is NEVER summarised — only the
    conversation transcript is.
    """
    if flow_manager is None:
        return ""
    s = flow_manager.state or {}
    name = s.get("name") or "(not yet given)"
    current = s.get("current_probe") or "opening"
    covered = list(s.get("covered_probes") or [])
    strikes = int(s.get("off_topic_count") or 0)
    gold = (s.get("current_gold_moment") or "").strip()

    lines = [
        "[SESSION STATE — load-bearing facts. Refer to these even if they fell off transcript.]",
        f"• Student name: {name}",
        f"• Current probe: {current}",
        f"• Probes covered: {', '.join(covered) if covered else 'none yet'}",
        f"• Off-topic strikes: {strikes}",
    ]
    if gold:
        # Truncate gold moment so the state block stays bounded.
        if len(gold) > 240:
            gold = gold[:237] + "…"
        lines.append(f"• Last gold moment from student: \"{gold}\"")
    lines.append("[/SESSION STATE]")
    return "\n".join(lines) + "\n\n" + _format_banner(flow_manager) + _image_note(flow_manager)


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
    redirect_text = _off_topic_redirect_text(count - 1, name, flow_manager)
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
    if current in _pfact(flow_manager):
        return ({"next_action": "wait_for_student"}, _pfact(flow_manager)[current](flow_manager))
    # Unknown current_probe — DO NOT close the exam on a single strike. Stay
    # on the current node and let the LLM redirect the student back on topic.
    logger.warning(
        f"Off-topic on unknown probe {current!r} — staying on node (was closing exam)"
    )
    return ({"next_action": "wait_for_student"}, None)


async def handle_student_end_request(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig | None]:
    """Student explicitly asked to STOP / FINISH the whole exam → close it.

    Transitions straight to the closing node, which thanks the student warmly
    and ends the conversation (which triggers grading). This is distinct from
    advance_from_probe, which only skips the current question.
    """
    name = flow_manager.state.get("name", "")
    logger.info(f"Student requested to end the exam (name={name!r}); closing.")
    flow_manager.state.update({"ended_by_student": True})
    return ({"status": "exam_ended_by_student"}, create_closing_node(flow_manager))


# ─── Cancel node ───────────────────────────────────────────────────────────────

def create_cancel_exam_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    return NodeConfig(
        name="cancelled",
        role_messages=[],  # seeded once in LLMContext at session start
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
        role_messages=[],  # seeded once in LLMContext at session start
        task_messages=[{
            "role": "system",
            "content": (
                "STEP 1 — INITIAL GREETING (no function call yet).\n"
                "If there is NO user message in the conversation history (i.e. the "
                "student has not spoken yet), simply greet the student warmly. "
                f"Introduce yourself as their {_examiner_title(flow_manager)} oral examiner. Explain you will "
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
                "• Do NOT accept single words like 'Hi' or 'okay' as a name.\n\n"
                "SPEAK FIELD — mandatory for every call:\n"
                "For intent='gave_name': write a warm one-sentence confirmation, "
                "e.g. \"Great — I'll call you [name]. Did I get that right?\"\n"
                "For intent='needs_clarification': write a brief helpful reply "
                "that clarifies what you need.\n"
                "For intent='evasion': write the redirect line specified above.\n"
                "This text goes directly to TTS — no second LLM call is made."
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
                    "speak": {
                        "type": "string",
                        "description": (
                            "The exact sentence to speak aloud right now. "
                            "For gave_name: a warm one-sentence confirmation reading "
                            "back the name, e.g. 'Great — I'll call you Atharva. "
                            "Did I get that right?'. For other intents: the "
                            "appropriate redirect or clarification line."
                        ),
                    },
                },
                required=["intent", "name_if_given", "speak"],
            ),
        ],
    )


# ─── Name confirmation + spell-out fallback ────────────────────────────────────

def create_known_name_greeting_node(
    flow_manager: FlowManager, respond_immediately: bool = True
) -> NodeConfig:
    """Greeting used when the teacher provided the student's name via the UI.
    Greets by name, introduces the examiner, then asks 'Are you ready to begin?'
    and waits for the student's reply before transitioning to the first
    question. Skips STT-based name capture entirely.

    respond_immediately=False re-enters the node silently (the handler already
    spoke via TTS) and waits for the student's next utterance.
    """
    name = flow_manager.state.get("name", "student")
    return NodeConfig(
        name="greeting_known_name",
        respond_immediately=respond_immediately,
        role_messages=[],  # seeded once in LLMContext at session start
        task_messages=[{
            "role": "system",
            "content": (
                f"The student's name is {name}. Greet {name} warmly by name in "
                f"one or two short sentences, introduce yourself as their {_examiner_title(flow_manager)} "
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
                "Do NOT ask for the student's name; you already have it.\n\n"
                f"SPEAK FIELD — mandatory:\n"
                "For 'ready': ONE SHORT flowing statement — warm acknowledgement "
                "(3-5 words) + the central question only, in a single clear "
                "sentence. KEEP IT SMALL. If the question text below is long or "
                "bundles several asks or instructions (e.g. 'be ready to "
                "defend...', 'engage with... too', 'consider both X and Y', "
                "'choose 3 objects and...'), do NOT read the whole paragraph — "
                "pose ONLY the core question now; the other parts come up "
                "naturally as follow-ups. Never deliver a big paragraph or "
                "multiple questions at once. "
                "Example: 'Wonderful! Here's our question: [the one core question]'\n"
                f"{_speak_question_line(flow_manager)}"
                "For 'not_ready': a gentle reassurance."
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
                    "speak": {
                        "type": "string",
                        "description": (
                            "The exact sentence to say right now. "
                            "For ready: brief warm transition e.g. 'Wonderful — let's begin.' "
                            "For not_ready: gentle reassurance."
                        ),
                    },
                },
                required=["intent", "speak"],
            ),
        ],
    )


def create_confirm_name_node(flow_manager: FlowManager) -> NodeConfig:
    """After the greeting captures a name, read it back and ask the student to
    confirm. If they reject, fall through to a spell-out node so a garbled STT
    pass doesn't lock in the wrong name for the whole session.

    On 'confirmed' the speak field must include BOTH the warm acknowledgement
    AND the exam question introduction — the opening node will load with
    respond_immediately=False, so this is the only LLM call that speaks.
    """
    candidate = flow_manager.state.get("name_candidate", "")
    exam_question = _fspec(flow_manager).exam_question if _fspec(flow_manager) else ""
    return NodeConfig(
        name="confirm_name",
        role_messages=[],  # seeded once in LLMContext at session start
        respond_immediately=False,  # greeting handler speaks via TTS directly
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
                "corrected_name empty and we'll ask them to spell it.\n\n"
                "SPEAK FIELD — mandatory:\n"
                "For 'confirmed': write ONE SHORT flowing statement that (a) warmly "
                "acknowledges the confirmed name in 3-5 words, then (b) poses only "
                "the CORE question in a single clear sentence. KEEP IT SMALL. If "
                "the question text below is long or bundles several asks (e.g. "
                "'be ready to defend...', 'engage with... too', 'choose 3 objects "
                "and...'), do NOT read the whole paragraph — pose only the core "
                "question now; other parts come up as follow-ups. Never deliver a "
                "big paragraph or multiple questions at once. Example: "
                "'Wonderful, [name]! Here's our question: [the one core question]' "
                f"{_speak_question_line(flow_manager)}"
                "For 'corrected' with a new name: confirm the new name warmly. "
                "For 'corrected' without a name: ask them to spell it."
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
                    "speak": {
                        "type": "string",
                        "description": (
                            "The exact text to speak aloud right now. "
                            "For confirmed: ONE SHORT statement — warm acknowledgement "
                            "(3-5 words) + only the CORE question in one clear sentence "
                            "(if the question is long or bundles several asks, pose only "
                            "its central question, not the whole paragraph), e.g. "
                            "'Wonderful! Here's our question: [the one core question]'. "
                            "For corrected with new name: warm confirmation of the "
                            "new name. For corrected without name: ask them to spell."
                        ),
                    },
                },
                required=["intent", "corrected_name", "speak"],
            ),
        ],
    )


def create_spell_name_node(flow_manager: FlowManager) -> NodeConfig:
    """Asks the student to spell their first name letter by letter. STT is far
    more accurate on individual letters than on a single-word proper noun, so
    this is the recovery branch from the confirmation node."""
    return NodeConfig(
        name="spell_name",
        role_messages=[],  # seeded once in LLMContext at session start
        respond_immediately=False,  # confirmation handler speaks via TTS directly
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
                    "speak": {
                        "type": "string",
                        "description": (
                            "Confirm the recorded name warmly in 3-5 words ONLY — "
                            "the exam opening is played automatically right after."
                            if _opening_script(flow_manager) else
                            "ONE flowing statement: confirm the recorded name warmly "
                            "(3-5 words) then immediately pose the exam question "
                            f"verbatim: '{_fspec(flow_manager).exam_question if _fspec(flow_manager) else ''}'"
                        ),
                    },
                },
                required=["name", "speak"],
            ),
        ],
    )


# ─── Opening (uses generated exam_question) ────────────────────────────────────

def create_opening_node(flow_manager: FlowManager, respond_immediately: bool = True) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    count = flow_manager.state.get("off_topic_count", 0)
    if not _fspec(flow_manager):
        raise RuntimeError("_fspec(flow_manager) is None — flow was not initialised")
    exam_question = _fspec(flow_manager).exam_question
    available = [p.id for p in _fspec(flow_manager).probe_pool]
    time_notice = _time_notice_prefix(flow_manager)
    support_notice = _support_notice_prefix(flow_manager)
    no_repeat = _no_repeat_prefix(flow_manager)

    return NodeConfig(
        name="opening",
        role_messages=[],  # seeded once in LLMContext at session start
        respond_immediately=respond_immediately,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_state_block(flow_manager)}"
                f"{_off_topic_instruction(count, name, flow_manager)}\n\n"
                f"{time_notice}{_pacing_prefix(flow_manager)}{support_notice}{no_repeat}"
                f"{_opening_steps(flow_manager, name, exam_question)}"
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


def _recent_examiner_openings(n: int = 3, max_words: int = 9, fm=None) -> list[str]:
    """Return the opening ~max_words of the last `n` EXAMINER turns from the
    live transcript collector, deduplicated (case-insensitive). Used to feed
    back into the next probe so the LLM can see (and avoid) phrases it just
    used."""
    lt = (fm.state.get("live_transcript") if fm is not None else None) or _CURRENT_TRANSCRIPT
    if lt is None:
        return []
    examiner = [t["text"] for t in lt.turns if t["speaker"] == "EXAMINER"]
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


def _asked_questions(fm=None, n: int = 12, max_words: int = 25) -> list[str]:
    """Every question sentence the examiner has asked so far (most recent `n`),
    deduplicated. Read from the live transcript, which is never summarised, so
    the list survives auto-context compression of older turns."""
    lt = (fm.state.get("live_transcript") if fm is not None else None) or _CURRENT_TRANSCRIPT
    if lt is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in lt.turns:
        if t["speaker"] != "EXAMINER":
            continue
        for q in re.findall(r"[^.?!]*\?", t["text"]):
            words = q.strip().split()
            if len(words) < 4 or "ready to begin" in q.lower():
                continue
            q = " ".join(words[:max_words])
            if q.lower() in seen:
                continue
            seen.add(q.lower())
            out.append(q)
    return out[-n:]


def _no_repeat_prefix(flow_manager: FlowManager) -> str:
    """Inject an anti-repetition block into the next probe's task_messages:
    the questions already asked (so the LLM can't re-ask them, even reworded)
    plus the openings of its last few turns (so it varies its phrasing)."""
    asked = _asked_questions(fm=flow_manager)
    openings = _recent_examiner_openings(fm=flow_manager)
    out = ""
    if asked:
        bullets = "\n".join(f"  • {q}" for q in asked)
        out += (
            "QUESTIONS YOU HAVE ALREADY ASKED — do NOT ask any of these again, "
            "and do NOT ask anything that seeks the SAME information in different "
            "words (a reworded question is still a repeat). Also do NOT ask about "
            "anything the student has already explained in the conversation, even "
            "in passing — build on it instead. Your next question must target a "
            "genuinely NEW aspect.\n"
            f"{bullets}\n\n"
        )
    if openings:
        bullets = "\n".join(f"  • '{o}…'" for o in openings)
        out += (
            "DO NOT REUSE YOUR RECENT OPENERS — you have already used these in "
            "your last few turns. Open this turn with a COMPLETELY different "
            "acknowledgement / transition / lead-in. Variety is mandatory.\n"
            f"{bullets}\n\n"
        )
    return out


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
    available = [p.id for p in _fspec(flow_manager).probe_pool if p.id not in covered]

    instructions = render_probe_instructions(probe, name, available)
    time_notice = _time_notice_prefix(flow_manager)
    revisit_notice = _revisit_notice_prefix(flow_manager)
    support_notice = _support_notice_prefix(flow_manager)
    no_repeat = _no_repeat_prefix(flow_manager)

    return NodeConfig(
        name=probe.id,
        role_messages=[],  # seeded once in LLMContext at session start
        task_messages=[{
            "role": "system",
            "content": (
                f"{_state_block(flow_manager)}"
                f"{_off_topic_instruction(count, name, flow_manager)}\n\n"
                f"{time_notice}{_pacing_prefix(flow_manager)}{revisit_notice}"
                f"{support_notice}{no_repeat}{instructions}"
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
    available = [p.id for p in _fspec(flow_manager).probe_pool if p.id not in covered]
    time_notice = _time_notice_prefix(flow_manager)
    support_notice = _support_notice_prefix(flow_manager)
    no_repeat = _no_repeat_prefix(flow_manager)

    return NodeConfig(
        name="followup_probe",
        role_messages=[],  # seeded once in LLMContext at session start
        task_messages=[{
            "role": "system",
            "content": (
                f"{_state_block(flow_manager)}"
                f"{_off_topic_instruction(count, name, flow_manager)}\n\n"
                f"{time_notice}{_pacing_prefix(flow_manager)}{support_notice}{no_repeat}"
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
    # The examiner sometimes thanks the student in the same turn it advances
    # to closing, which made the closing a second "thank you".
    content += (
        " IMPORTANT: if your previous message already thanked the student, do "
        "NOT thank them again — say only, in one short sentence, that the oral "
        "discussion is now complete."
    )

    return NodeConfig(
        name="closing",
        role_messages=[],  # seeded once in LLMContext at session start
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
    duration_min = (_fspec(flow_manager).duration_minutes if _fspec(flow_manager) else None) or 6
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


# ─── Pacing ────────────────────────────────────────────────────────────────────
# The exam should END on time NATURALLY, not be cut off. Each question gets a
# share of the remaining time; the examiner is told its share up front, nudged
# to move on (after the student's current answer) once the share is used, and
# the exam closes at a question boundary once there's no time for another one.
CLOSE_RESERVE_SECS = 30   # kept free at the end for the closing words
MIN_QUESTION_SECS = 50    # don't start a new question with less than this left
PACING_FAILSAFE_SECS = 120  # overtime after which we close even mid-question


def _pacing(flow_manager: FlowManager) -> dict:
    """Return {time_left, questions_left, slice}: questioning seconds left
    (before the closing reserve), questions still planned, and the time share
    for the next question."""
    ts = _compute_time_state(flow_manager)
    planned = planned_question_count(ts["budget"] / 60)
    asked = len(flow_manager.state.get("covered_probes", []))
    questions_left = max(1, planned - asked)
    time_left = max(0.0, ts["budget"] - ts["elapsed"] - CLOSE_RESERVE_SECS)
    return {
        "time_left": time_left,
        "questions_left": questions_left,
        "slice": time_left / questions_left,
    }


def _pacing_prefix(flow_manager: FlowManager) -> str:
    """Tell the examiner how much time this question gets, so it paces the
    follow-ups itself instead of being stopped by the clock."""
    if not flow_manager.state.get("started_at_iso"):
        return ""
    p = _pacing(flow_manager)
    if p["time_left"] < 2 * MIN_QUESTION_SECS:
        return (
            "PACING: this is the LAST question of the exam. Ask it, let the "
            "student answer fully, then call advance_from_probe — the exam will "
            "close after it. Do NOT mention time or the clock.\n\n"
        )
    return (
        f"PACING: about {p['time_left'] / 60:.1f} minutes of questioning remain — "
        f"roughly {p['questions_left']} more question(s), ~{p['slice']:.0f} seconds "
        "each. Keep this question to the main ask plus at most ONE short "
        "follow-up, then call advance_from_probe. Never rush the student or "
        "mention the time.\n\n"
    )


# ─── Smart transition handler ──────────────────────────────────────────────────

VOLUNTARY_SKIP_LIMIT = 2  # Beyond this, treat further skips as struggle, not voluntary.
PROBE_ATTEMPT_CAP = 3     # Matches _ATTEMPT_CAP_RULE in flow_primitives.
REVISIT_LIMIT = 2         # How many times we'll loop back to covered probes if time remains.

# ─── Idle / no-response escalation ──────────────────────────────────────────
# When the student goes silent after the examiner speaks, escalate instead of
# hanging forever: strike 1 nudges ("are you there?"), strike 2 moves to the
# next question, strike 3 thanks and ends (which triggers grading). The timer
# is the per-silence window and only runs after the bot finishes speaking.
IDLE_TIMEOUT_SECS = float(os.getenv("EXAM_IDLE_TIMEOUT_SECS", "15"))


def _pick_revisit_probe(flow_manager: FlowManager) -> str | None:
    """Pick the covered probe with the lowest attempt count — the one that
    got least exploration first time around — as the revisit target.
    Returns None if no probe is suitable."""
    if not _fspec(flow_manager):
        return None
    covered = set(flow_manager.state.get("covered_probes", []))
    attempts = flow_manager.state.get("probe_attempts", {})
    # Candidates: probes in the pool that were covered.
    candidates = [(p.id, int(attempts.get(p.id, 0))) for p in _fspec(flow_manager).probe_pool if p.id in covered]
    if not candidates:
        return None
    # Sort by attempts ascending — least-explored probe gets revisited first.
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


async def _flush_transcript(flow_manager: FlowManager) -> None:
    """[DISABLED — DO NOT CALL FROM handle_transition]

    Used to wipe the user/assistant conversation history at a probe
    boundary, but this breaks Gemini's function-call invariant: when the
    flush runs during an active function-call/response pair (which is
    exactly what's happening when `handle_transition` fires for
    `advance_from_probe`), the next LLM call hits:

        400 INVALID_ARGUMENT: Please ensure that function call turn comes
        immediately after a user turn or after a function response turn.

    Pipecat's docs explicitly do NOT guarantee safety of
    LLMMessagesUpdateFrame during tool execution; see Pipecat issues
    #3290 and #3557 for related upstream fragility.

    If you want to re-introduce a flush, options are:
      (a) Defer to the NEXT UserStartedSpeakingFrame after the function
          chain completes (set a `_pending_flush` flag in flow_manager.state
          and consume it from a FrameProcessor).
      (b) Use LLMMessagesTransformFrame to keep the trailing user/fn-call/
          fn-response triple intact and trim only older messages.

    Until either of those lands, prompt-size growth is bounded by
    auto-summarization + the state block alone.
    """
    try:
        await flow_manager.task.queue_frame(
            LLMMessagesUpdateFrame(messages=[], run_llm=False)
        )
        logger.info("Checkpoint flush: cleared user/assistant transcript")
    except Exception as e:
        logger.warning(f"Transcript flush failed (continuing anyway): {e}")


async def handle_transition(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    # Reset the no-response strike counter: the student engaged enough to
    # advance. Skip the reset when THIS advance was itself triggered by the
    # idle escalation (strike 2), so a continued silence still escalates to end.
    if flow_manager.state.pop("idle_advance_pending", False):
        pass
    else:
        flow_manager.state.update({"idle_strikes": 0})

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

    # Natural end: the student has just finished an answer. If there isn't
    # room for another question before the closing reserve, close now — at a
    # question boundary, so nothing is cut off and the exam ends on time.
    pace = _pacing(flow_manager)
    if pace["time_left"] < MIN_QUESTION_SECS:
        logger.info(
            f"Pacing: {pace['time_left']:.0f}s of questioning left "
            f"(< {MIN_QUESTION_SECS}s) — closing after this answer"
        )
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

    # NOTE: We previously called `_flush_transcript(flow_manager)` here as a
    # "checkpoint flush" at every probe transition. Removed because it broke
    # Gemini's function-call invariant — wiping the messages list mid-
    # function-call returned 400 INVALID_ARGUMENT: "Please ensure that
    # function call turn comes immediately after a user turn or after a
    # function response turn." See sessions 20260619_094503 and 095433 for
    # the crash trace, and Pipecat issues #3290 / #3557 for the upstream
    # context. Bounding prompt growth now relies on auto-summarization
    # (LLMAutoContextSummarizationConfig) + the state block, both of which
    # respect Gemini's ordering rules.

    # ── Routing decisions (unchanged below) ───────────────────────────────
    gold = args.get("gold_moment", "").strip()
    gold_followed = flow_manager.state.get("gold_followed", 0)
    # A gold-moment follow-up is an extra question — only if time allows.
    if gold and gold_followed < 1 and pace["time_left"] >= 2 * MIN_QUESTION_SECS:
        flow_manager.state.update({
            "gold_followed": gold_followed + 1,
            "current_probe": "followup_probe",
            # Persist gold so off-topic / re-entry can rebuild the same node
            # without losing the moment we were following.
            "current_gold_moment": gold,
        })
        logger.info(f"Following gold moment: {gold!r}")
        return ({}, create_followup_probe_node(flow_manager, gold))

    missing_required = set(_rprobes(flow_manager)) - covered
    if missing_required:
        next_id = sorted(missing_required)[0]
        logger.info(f"Required probe not yet covered: {next_id!r}")
        flow_manager.state.update({"current_probe": next_id})
        return ({}, _pfact(flow_manager)[next_id](flow_manager))

    recommended = args.get("recommended_next", "").strip()
    if recommended and recommended not in covered and recommended in _pfact(flow_manager):
        logger.info(f"Using LLM recommendation: {recommended!r}")
        flow_manager.state.update({"current_probe": recommended})
        return ({}, _pfact(flow_manager)[recommended](flow_manager))

    remaining = [p.id for p in _fspec(flow_manager).probe_pool if p.id not in covered]
    if not remaining:
        # All probes covered but time remains (the pacing check above already
        # closed if it didn't). Revisit the least-explored covered probe from
        # a new angle so the exam fills its duration, up to REVISIT_LIMIT.
        revisit_count = flow_manager.state.get("revisit_count", 0)
        if revisit_count >= REVISIT_LIMIT:
            logger.info(
                f"All probes covered + revisits={revisit_count}/{REVISIT_LIMIT} — closing"
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
        return ({}, _pfact(flow_manager)[revisit_id](flow_manager))

    next_id = remaining[0]
    logger.info(f"Picking next uncovered probe: {next_id!r}")
    flow_manager.state.update({"current_probe": next_id})
    return ({}, _pfact(flow_manager)[next_id](flow_manager))


# ─── Greeting routing handler ──────────────────────────────────────────────────

async def handle_greeting_routing(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig | None]:
    intent = args["intent"]
    name_given = args.get("name_if_given", "").strip()
    speak_text = args.get("speak", "").strip()

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

    # Speak the LLM-generated line directly — skips the second LLM call.
    if speak_text:
        try:
            await flow_manager.task.queue_frame(TTSSpeakFrame(speak_text))
            logger.info(f"Greeting handler spoke directly: {speak_text!r}")
        except Exception as e:
            logger.warning(f"Greeting handler TTSSpeakFrame failed: {e}")

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
    speak_text = args.get("speak", "").strip()

    # Speak the LLM-generated line directly — skips the second LLM call.
    if speak_text:
        try:
            await flow_manager.task.queue_frame(TTSSpeakFrame(speak_text))
            logger.info(f"Confirmation handler spoke directly: {speak_text!r}")
        except Exception as e:
            logger.warning(f"Confirmation handler TTSSpeakFrame failed: {e}")

    if intent == "confirmed" and candidate:
        logger.info(f"Name confirmed: {candidate!r}")
        flow_manager.state.update({
            "name": candidate,
            "off_topic_count": 0,
            "current_probe": "opening",
        })
        # exam question already spoken in the speak field — opening loads
        # its probe-handling prompt silently, waits for student to respond.
        await _speak_opening_script(flow_manager)
        return ({}, create_opening_node(flow_manager, respond_immediately=False))

    # Student rejected the read-back. If they supplied a corrected name in the
    # same turn, use it directly; otherwise fall through to spell-out.
    if corrected:
        logger.info(f"Name corrected inline: {corrected!r}")
        flow_manager.state.update({
            "name": corrected,
            "off_topic_count": 0,
            "current_probe": "opening",
        })
        # Same: exam question was in the speak field, opening is silent.
        await _speak_opening_script(flow_manager)
        return ({}, create_opening_node(flow_manager, respond_immediately=False))

    logger.info("Name rejected, asking student to spell it out")
    return ({"next_action": "spell_name"}, create_spell_name_node(flow_manager))


async def handle_begin_exam(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    """Used by create_known_name_greeting_node. If the student is ready,
    transitions to the opening exam question. If not, stays in the greeting
    node so the bot can reassure them and re-ask."""
    intent = args.get("intent", "ready")
    speak_text = args.get("speak", "").strip()

    # Phantom-call guard — THE EXAMINER MUST GREET FIRST. The known-name greeting
    # node fires respond_immediately on connect, but some models (notably Haiku
    # 5.5) call begin_exam right then — before the student has said anything and
    # WITHOUT speaking the greeting — which skips the whole greeting and jumps
    # straight to the question. If begin_exam arrives with no student utterance
    # yet, speak the greeting ourselves and STAY in the greeting node so the
    # examiner opens the conversation and then waits for the student's reply.
    if flow_manager.state.get("student_utterance_count", 0) == 0:
        name = (flow_manager.state.get("name") or "there").strip() or "there"
        greeting = (
            f"Hello {name}! I'm your {_examiner_title(flow_manager)} oral examiner for today. This is a "
            "short, relaxed discussion — there are no right or wrong answers, I "
            "just want to hear your thoughts. Are you ready to begin?"
        )
        try:
            await flow_manager.task.queue_frame(TTSSpeakFrame(greeting))
            logger.info(
                "begin_exam phantom-called before greeting; spoke greeting and "
                "staying in greeting node to wait for the student."
            )
        except Exception as e:
            logger.warning(f"greeting TTSSpeakFrame failed: {e}")
        # Must return a node, not None: pipecat-flows re-runs the LLM after any
        # non-transition function result, and the model then "apologises" for
        # starting early and asks the question itself — leaving the whole exam
        # stuck in the greeting node. A silent re-entry suppresses that re-run.
        return (
            {"next_action": "wait_for_student"},
            create_known_name_greeting_node(flow_manager, respond_immediately=False),
        )

    if speak_text:
        try:
            await flow_manager.task.queue_frame(TTSSpeakFrame(speak_text))
            logger.info(f"begin_exam handler spoke directly: {speak_text!r}")
        except Exception as e:
            logger.warning(f"begin_exam TTSSpeakFrame failed: {e}")

    if intent == "not_ready":
        flow_manager.state["not_ready_count"] = (
            flow_manager.state.get("not_ready_count", 0) + 1
        )
        logger.info(
            f"Student not yet ready (count={flow_manager.state['not_ready_count']}); "
            "staying in greeting."
        )
        # Reassurance was already spoken above — re-enter silently so the LLM
        # doesn't generate a second one on top of it.
        return (
            {"next_action": "wait_for_student"},
            create_known_name_greeting_node(
                flow_manager, respond_immediately=not speak_text
            ),
        )

    logger.info("Student is ready; advancing to opening question.")
    flow_manager.state.update({
        "off_topic_count": 0,
        "current_probe": "opening",
        # Start the exam proper with a clean no-response count so any silences
        # during the greeting don't carry over into the first question.
        "idle_strikes": 0,
        "idle_advance_pending": False,
    })
    # exam question already spoken in speak field — opening is silent.
    await _speak_opening_script(flow_manager)
    return ({}, create_opening_node(flow_manager, respond_immediately=False))


async def handle_spelled_name(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    name = args.get("name", "").strip().title()
    speak_text = args.get("speak", "").strip()
    if not name:
        name = "student"
    logger.info(f"Name recorded via spelling: {name!r}")

    if speak_text:
        try:
            await flow_manager.task.queue_frame(TTSSpeakFrame(speak_text))
            logger.info(f"spelled_name handler spoke directly: {speak_text!r}")
        except Exception as e:
            logger.warning(f"spelled_name TTSSpeakFrame failed: {e}")

    flow_manager.state.update({
        "name": name,
        "off_topic_count": 0,
        "current_probe": "opening",
    })
    # exam question already spoken in speak field — opening is silent.
    await _speak_opening_script(flow_manager)
    return ({}, create_opening_node(flow_manager, respond_immediately=False))


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


# Single source of truth for the conversation LLM default, shared by run_bot,
# run_session, and the server (assessvox imports both). Defined once so every
# entry point runs the same provider+model regardless of how the exam started.
# Default = Claude Haiku 4.5: fastest measured time-to-first-token (~0.7s vs
# ~4.7s for gemini-3.5-flash on the same network) and it handles the exam's
# function-calling. Override per-exam from the UI dropdown, or globally via the
# DEFAULT_LLM_PROVIDER / DEFAULT_LLM_MODEL env vars.
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "claude-haiku-4-5")


def _gemini_thinking_level(model: str) -> str:
    """Pick a thinking level the given Gemini model actually supports.

    We want the LOWEST level to suppress the thought-signature blocks that
    otherwise bloat context (3K -> 19K+ tokens). Older flash models (2.5, 3.5)
    accept "minimal", which is ideal. gemini-3.8+ dropped "minimal" — the API
    returns 400 INVALID_ARGUMENT ("Thinking level MINIMAL is not supported") —
    and its floor is "low", so use that there. Picking by model keeps every
    option in the UI dropdown working instead of crashing the voice bot.
    """
    m = (model or "").lower()
    if "gemini-3.8" in m or "gemini-4" in m:
        return "low"
    return "minimal"


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


# ─── Video grading (LiveKit + Gemini multimodal) ─────────────────────────────

VIDEO_GRADER_MODEL = os.getenv("GEMINI_VIDEO_GRADER_MODEL", "gemini-3.5-flash")

VIDEO_DIMENSION_GUIDE = """\
AUDIO DIMENSIONS (same as text grading — judge from spoken content):
  clarity, reasoning, examples, engagement, fluency, pronunciation, confidence

VIDEO DIMENSIONS (judge from what you SEE in the video):
  technique               — Procedural correctness of any demonstration.
                            For non-demo questions score 4 by default and note "n/a".
  visual_clarity          — Was what the student showed legible? Camera angle appropriate?
  verbal_visual_alignment — Does what the student SAID match what they DID?
"""

VIDEO_GRADING_PROMPT = """You are an IB oral exam evaluator grading a VIDEO recording
of a single oral assessment. Watch end-to-end before scoring.
Two voices: an AI EXAMINER and the STUDENT. Grade ONLY the student.
Ground every claim in what you can SEE or HEAR. Do not invent observations.

Exam question: "{exam_question}"
Topic: "{topic}"
Metadata: {metadata}
Node trace: {node_trace}

DIMENSIONS — score each 1-4:
{dimension_guide}

For each dimension produce score, analysis (2-4 sentences), evidence (1-3 timestamped objects).
Then: headline (≤20 words), summary (4-6 sentences), strengths (2-3), growth_areas (2-3),
overall_score (0-100).

Return ONLY valid JSON:
{{
  "headline": str, "summary": str,
  "dimensions": {{
    "clarity": {{"score": int, "analysis": str, "evidence": [{{"timestamp": str, "observation": str}}]}},
    "reasoning": {{...}}, "examples": {{...}}, "engagement": {{...}},
    "fluency": {{...}}, "pronunciation": {{...}}, "confidence": {{...}},
    "technique": {{...}}, "visual_clarity": {{...}}, "verbal_visual_alignment": {{...}}
  }},
  "strengths": [str], "growth_areas": [str], "overall_score": float
}}
"""


def _grade_call_with_retry(fn, *, attempts: int = 4, what: str = "grading"):
    """Run a Gemini grading call (generate_content / file upload) with retries
    on transient network failures.

    The grading request occasionally dies mid-flight with a connection reset
    (WinError 10054 / httpx.ReadError) — a flaky hop to Google, not a real error.
    google-genai's own retry doesn't cover read-resets, so one blip failed the
    whole report. We retry transient network/5xx errors with backoff, but NOT
    4xx (bad key, quota, malformed request), which won't improve on retry.
    """
    import time as _time
    try:
        import httpx
        _net_errs = (httpx.TransportError, ConnectionError, TimeoutError)
    except Exception:
        _net_errs = (ConnectionError, TimeoutError)
    try:
        from google.genai import errors as _genai_errors
    except Exception:
        _genai_errors = None

    def _is_retriable(e: Exception) -> bool:
        # Retry transport/network failures...
        if isinstance(e, _net_errs):
            return True
        # ...and SERVER errors (HTTP 5xx) reported by the google-genai library,
        # classified by STATUS CODE, not by words in the message. Never retry a
        # client error (4xx: bad key, 400 bad request, 402/429 quota) — it can't
        # succeed on retry and only wastes quota and delays the failure.
        if _genai_errors is not None and isinstance(e, getattr(_genai_errors, "APIError", ())):
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            try:
                return int(code) >= 500
            except (TypeError, ValueError):
                return isinstance(e, getattr(_genai_errors, "ServerError", ()))
        return False

    last_exc = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — classify below, re-raise non-retriable
            if not _is_retriable(e):
                raise
            last_exc = e
            if i < attempts:
                wait = 2 ** (i - 1)  # 1s, 2s, 4s
                logger.warning(
                    f"{what}: transient error (attempt {i}/{attempts}): "
                    f"{type(e).__name__}: {str(e)[:120]}; retrying in {wait}s"
                )
                _time.sleep(wait)
    raise last_exc


def _grade_exam_video_sync(
    video_path: Path,
    metadata: dict,
    node_trace_str: str,
    exam_question: str,
    topic: str,
    custom_prompt: str | None,
) -> dict:
    """Grade a video recording via Gemini multimodal Files API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    logger.info(f"Uploading {video_path.name} to Gemini Files API…")
    video_file = _grade_call_with_retry(
        lambda: client.files.upload(file=str(video_path)), what="video upload")

    waited = 0.0
    while getattr(video_file, "state", None) and video_file.state.name == "PROCESSING":
        if waited > 300:
            raise TimeoutError(f"Gemini file processing exceeded 5 min for {video_path.name}")
        import time as _time; _time.sleep(3.0)
        waited += 3.0
        video_file = client.files.get(name=video_file.name)
    if getattr(video_file, "state", None) and video_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini failed to process {video_path.name}")
    logger.info(f"Video ready ({video_file.name}); calling {VIDEO_GRADER_MODEL}…")

    prompt_template = custom_prompt or VIDEO_GRADING_PROMPT
    try:
        prompt = prompt_template.format(
            exam_question=exam_question, topic=topic,
            metadata=json.dumps(metadata, indent=2),
            node_trace=node_trace_str,
            dimension_guide=VIDEO_DIMENSION_GUIDE,
        )
    except KeyError as e:
        logger.warning(f"Custom video grading prompt missing placeholder {e}; using default")
        prompt = VIDEO_GRADING_PROMPT.format(
            exam_question=exam_question, topic=topic,
            metadata=json.dumps(metadata, indent=2),
            node_trace=node_trace_str,
            dimension_guide=VIDEO_DIMENSION_GUIDE,
        )

    response = _grade_call_with_retry(lambda: client.models.generate_content(
        model=VIDEO_GRADER_MODEL,
        contents=[prompt, video_file],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=12288,
        ),
    ), what="video grading")
    grader_usage = _gemini_usage(response, VIDEO_GRADER_MODEL)
    text = (response.text or "{}").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    first, last = text.find("{"), text.rfind("}")
    candidate = text[first:last + 1] if (first != -1 and last > first) else text
    try:
        return {**json.loads(candidate), "_grader_usage": grader_usage}
    except json.JSONDecodeError as e:
        logger.error(f"Video grader returned non-JSON ({e}). First 500: {text[:500]}")
        return {"error": "invalid_json", "raw": text, "_grader_usage": grader_usage}


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

    response = _grade_call_with_retry(lambda: client.models.generate_content(
        model=GRADER_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=8192,   # 7-dim report can be long; default cap truncates
        ),
    ), what="audio grading")
    grader_usage = _gemini_usage(response, GRADER_MODEL)
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
        return {**json.loads(candidate), "_grader_usage": grader_usage}
    except json.JSONDecodeError as e:
        logger.error(
            f"Grader returned non-JSON ({e}). First 1000 chars:\n{text[:1000]}"
        )
        return {"error": "invalid_json", "raw": text, "_grader_usage": grader_usage}


async def grade_exam(
    wav_path: Path,
    metadata: dict,
    report_path: Path,
    transcript_turns: list[dict],
    node_trace: list[dict],
    duration_seconds: float,
    exam_question: str,
    topic: str,
    video_path: Path | None = None,
    custom_video_prompt: str | None = None,
    usage: SessionUsage | None = None,
) -> None:
    node_trace_str = (
        "\n".join(f"  {n['ts']}  → {n['node']}" for n in node_trace)
        if node_trace else "(no transitions recorded)"
    )

    if video_path is not None and video_path.exists():
        logger.info(f"Grading exam (video) → sending {video_path.name} to {VIDEO_GRADER_MODEL}")
        try:
            report = await asyncio.to_thread(
                _grade_exam_video_sync,
                video_path, metadata, node_trace_str,
                exam_question, topic, custom_video_prompt,
            )
        except Exception as e:
            logger.exception(f"Video grading failed: {e}")
            report = {"error": str(e)}
        report["video_file"] = str(video_path)
    else:
        logger.info(f"Grading exam → sending {wav_path.name} to {GRADER_MODEL}")
        transcript_str = _format_transcript(transcript_turns)
        try:
            report = await asyncio.to_thread(
                _grade_exam_sync,
                wav_path, metadata, transcript_str, node_trace_str,
                exam_question, topic,
            )
        except Exception as e:
            logger.exception(f"Grading failed: {e}")
            report = {"error": str(e)}
        report["audio_file"] = str(wav_path)

    grader_usage = report.pop("_grader_usage", None)
    if usage is not None and grader_usage:
        usage.add_llm(
            "grading", grader_usage["model"],
            prompt=grader_usage["prompt"], completion=grader_usage["completion"],
            cache_read=grader_usage["cache_read"], reasoning=grader_usage["reasoning"],
        )

    report["metadata"] = metadata
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


def get_live_transcript(session_id: str | None = None) -> list[dict]:
    """Server-facing accessor for a session's in-flight transcript (by
    session_id when given, else the single-session fallback)."""
    lt = None
    if session_id is not None and session_id in _LIVE_SESSIONS:
        lt = _LIVE_SESSIONS[session_id]["transcript"]
    if lt is None:
        lt = _CURRENT_TRANSCRIPT
    if lt is None:
        return []
    return list(lt.turns)


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
            # The student spoke, so any accumulated silence no longer counts —
            # reset the no-response escalation here (NOT only when the exam moves
            # to a new question). Also clear the idle-advance marker so a leftover
            # one can never make the student's next real answer skip its reset.
            if state.get("idle_strikes") or state.get("idle_advance_pending"):
                state.update({"idle_strikes": 0, "idle_advance_pending": False})
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

    if provider == "cartesia-ink-2":
        if not os.getenv("CARTESIA_API_KEY"):
            raise RuntimeError("CARTESIA_API_KEY not set in .env (required for Cartesia STT)")
        # CartesiaTurnsSTTService uses Cartesia's v2 WebSocket API (/stt/turns/websocket).
        # Ink-2 drives turn boundaries natively — turn.start / turn.update / turn.end /
        # turn.eager_end — and pushes UserStartedSpeakingFrame / UserStoppedSpeakingFrame
        # directly from semantic endpointing. No Silero VAD or SmartTurn V3 needed;
        # ExternalUserTurnStrategies lets those external frames control turn flow.
        # This eliminates the mid-thought fragmentation that caused 3-6 LLM calls
        # per turn when students paused between phrases.
        # Keyterm boosting: bias recognition toward this exam's domain vocabulary
        # (topic terms + student/character names) so words like "elitist" aren't
        # misheard as "elite test". ink-2 supports the same keyterm prompting as
        # Deepgram nova-3; without it, rare/academic words get phonetic lookalikes.
        kt = keyterms or []
        logger.info(
            f"STT provider: cartesia / ink-2 (turns v2) | native semantic "
            f"endpointing | keyterms={len(kt)} ({kt[:8]}{'…' if len(kt) > 8 else ''})"
        )
        return CartesiaTurnsSTTService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            settings=CartesiaTurnsSTTService.Settings(
                language=lang_code or "en",
                keyterm=kt or None,
            ),
        )

    if provider in ("cartesia", "cartesia-ink-whisper"):
        if not os.getenv("CARTESIA_API_KEY"):
            raise RuntimeError("CARTESIA_API_KEY not set in .env (required for Cartesia STT)")
        cartesia_lang = lang_code or "en"
        cartesia_model = "ink-whisper"
        logger.info(
            f"STT provider: cartesia / {cartesia_model} | language={cartesia_lang}"
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
    that support keyterm-prompting. Deepgram nova-3 AND Cartesia ink-2 both
    accept a keyterm list (ink-2 via CartesiaTurnsSTTService.Settings.keyterm),
    so academic/topic words aren't misheard as phonetic lookalikes (e.g.
    'elitist' -> 'elite test')."""
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
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: str = DEFAULT_LLM_MODEL,
    stt_provider: str = "cartesia-ink-2",
    stt_language: str | None = "en",
    tts_voice: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    use_turn_markers: str = "auto",
    enable_auto_summarization: bool = True,
):
    # Capture THIS session's flow state into locals NOW, before any await. The
    # module globals hold this exam's data at run_bot entry (set_active_flow ran
    # synchronously just before), but a CONCURRENT exam could overwrite the
    # globals the moment we await. We seed these onto flow_manager.state below,
    # which pipecat makes available to every handler — so each exam reads its own
    # state. (This is what makes concurrent exams in one process safe.)
    _sess_flow_spec = _FLOW_SPEC
    _sess_probe_factories = _PROBE_FACTORIES
    _sess_required_probes = _REQUIRED_PROBES
    _sess_custom_grading = _CUSTOM_GRADING_PROMPT
    _sess_provided_name = _PROVIDED_STUDENT_NAME
    _sess_programme = _PROGRAMME

    # Per-session log file — captures every loguru log line emitted during this
    # session so we can post-mortem any one run without scrolling stdout.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    session_log_path = LOGS_DIR / f"session_{session_id}.log"
    # Tag this exam's asyncio task (and every task it spawns) so the file keeps
    # only this exam's lines when several exams run at once. Untagged lines
    # (server-level, or tasks created before this point) are still kept.
    _LOG_SESSION.set(session_id)
    _session_log_sink_id = logger.add(
        str(session_log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
        filter=lambda record, sid=session_id: _LOG_SESSION.get() in (sid, None),
    )

    logger.info(
        f"Starting dynamic oral exam bot | session={session_id} | "
        f"llm={llm_provider}:{llm_model} | stt={stt_provider}"
    )

    keyterms = _extract_keyterms_from_flowspec(_sess_flow_spec)
    stt = _build_stt(stt_provider, stt_language, keyterms=keyterms)

    cartesia_keys = _load_cartesia_keys()
    if not cartesia_keys:
        raise RuntimeError(
            "No Cartesia API key configured — set CARTESIA_API_KEY in .env"
        )
    cartesia_tts_model = os.getenv("CARTESIA_TTS_MODEL", "sonic-3.6")
    logger.info(
        f"Cartesia TTS configured with {len(cartesia_keys)} key(s): "
        f"primary={FallbackCartesiaTTSService._mask(cartesia_keys[0])}"
        + (f", fallbacks={[FallbackCartesiaTTSService._mask(k) for k in cartesia_keys[1:]]}" if len(cartesia_keys) > 1 else " (no fallbacks)")
        + f" | voice={tts_voice} | model={cartesia_tts_model}"
    )
    tts = FallbackCartesiaTTSService(
        api_keys=cartesia_keys,
        voice_id=tts_voice,
        model=cartesia_tts_model,
    )

    # LLM call dumper: write every round-trip's input+output to
    # llm_calls/calls_<session>.jsonl. The /sessions dashboard reads this to
    # show raw I/O per turn.
    LLM_CALLS_DIR.mkdir(parents=True, exist_ok=True)
    _llm_dump_path = LLM_CALLS_DIR / f"calls_{session_id}.jsonl"

    # The examiner rulebook goes in as the service's system_instruction, NOT as
    # a system message in the context. pipecat-flows registers its functions as
    # async tools, which makes pipecat compose its own system_instruction; on
    # Anthropic and Gemini a context system message is then silently DROPPED
    # (the "Both system_instruction and an initial system message" warning), so
    # the rulebook never reached the model. Setting it here keeps both.
    _role_prompt = _role_messages_for(_sess_programme)[0]["content"]

    if llm_provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        llm = _DumpingGroqLLMService(
            api_key=groq_key,
            settings=GroqLLMService.Settings(model=llm_model, system_instruction=_role_prompt),
            _dump_path=_llm_dump_path,
        )
    elif llm_provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        llm = _DumpingOpenAILLMService(
            api_key=openai_key,
            settings=OpenAILLMService.Settings(model=llm_model, system_instruction=_role_prompt),
            _dump_path=_llm_dump_path,
        )
    elif llm_provider == "anthropic":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        llm = _DumpingAnthropicLLMService(
            api_key=anthropic_key,
            settings=AnthropicLLMService.Settings(model=llm_model, system_instruction=_role_prompt),
            _dump_path=_llm_dump_path,
        )
    elif llm_provider == "cerebras":
        from pipecat.services.cerebras.llm import CerebrasLLMService
        cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
        if not cerebras_key:
            raise RuntimeError("CEREBRAS_API_KEY not set in .env")
        # Wrap CerebrasLLMService with the dumper. Built dynamically because
        # it's imported lazily — keeps Cerebras as an optional extra.
        _DumpingCerebrasLLMService = type(
            "_DumpingCerebrasLLMService",
            (LLMCallDumperMixin, CerebrasLLMService),
            {},
        )
        # gpt-oss-120b ships with reasoning_effort="medium" by default. That
        # adds ~300 reasoning tokens per response and several seconds of LLM
        # time even on short replies — wasted on an oral examiner that just
        # needs to acknowledge / probe / advance. "low" is the minimum
        # supported value (Cerebras does not allow disabling reasoning
        # entirely on gpt-oss-120b). The `extra` dict is spread into the
        # OpenAI-compatible chat-completions call by CerebrasLLMService.
        llm = _DumpingCerebrasLLMService(
            api_key=cerebras_key,
            settings=CerebrasLLMService.Settings(
                model=llm_model,
                system_instruction=_role_prompt,
                extra={"reasoning_effort": "low"},
            ),
            _dump_path=_llm_dump_path,
        )
    else:
        # gemini-3.5-flash defaults to thinking_level="high", which causes
        # Gemini to emit thought_signature blocks on every model response.
        # Those signatures are large base64 traces that get replayed in
        # context on every subsequent call, bloating prompts from ~3K to
        # 19K+ tokens. Set thinking_level="minimal" to disable thinking-
        # mode artifacts entirely. The model still produces high-quality
        # output for our oral-exam workload (no chain-of-thought needed).
        _gemini_settings = AssessVoxGoogleLLMService.Settings(
            model=llm_model,
            system_instruction=_role_prompt,
            thinking=AssessVoxGoogleLLMService.ThinkingConfig(
                # Model-aware: gemini-3.8+ rejects "minimal" (400), floors at
                # "low"; older flash keeps "minimal". See _gemini_thinking_level.
                thinking_level=_gemini_thinking_level(llm_model),
                include_thoughts=False,
            ),
        )
        llm = _DumpingAssessVoxGoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            settings=_gemini_settings,
            _dump_path=_llm_dump_path,
        )

    # Seed the context with the persona/system message ONCE. pipecat-flows'
    # _update_llm_context uses ContextStrategy.APPEND on every _set_node call,
    # which re-injects role_messages into context.messages each time. With 9
    # nodes and ~3500 tokens of ROLE_MESSAGES, that accumulates to 6+ copies
    # (≈21K tokens) by the end of a full exam — confirmed in session dumps.
    # Fix: seed the context here at creation so ROLE_MESSAGES is already
    # present, then pass role_messages=[] on all nodes so _set_node has
    # nothing new to append.
    # Rulebook lives in llm system_instruction (see _role_prompt above).
    context = LLMContext(messages=[])
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
    # Turn strategies:
    #
    # cartesia-ink-2 → CartesiaTurnsSTTService (v2 API) emits
    #   UserStartedSpeakingFrame / UserStoppedSpeakingFrame directly from
    #   Ink-2's semantic endpointing (turn.start / turn.end events). Silero
    #   VAD and SmartTurn V3 are redundant and harmful — they would commit
    #   mid-thought fragments before Ink-2 decides the turn is over.
    #   ExternalUserTurnStrategies tells the aggregator: "let the STT service
    #   drive turn boundaries; don't apply VAD/SmartTurn at all."
    #
    # All other STTs → SmartTurn V3 with stop_secs=1.2 as the worst-case cap.
    _use_external_turn_strategies = _stt_lower == "cartesia-ink-2"
    if _use_external_turn_strategies:
        _user_turn_strategies = ExternalUserTurnStrategies()
        logger.info("Turn strategies: ExternalUserTurnStrategies (Ink-2 drives turn boundaries)")
    else:
        _turn_analyzer = LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams(stop_secs=1.2)
        )
        _user_turn_strategies = UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=_turn_analyzer)],
        )

    # Pipecat auto-context summarization: when the conversation grows past
    # ``max_context_tokens``, an async background task asks the LLM to
    # compress older turns into a short summary, replacing them in-place.
    # Without this, the prompt grew from 1.3K (turn 1) to 19.2K tokens
    # (turn 23) — see session 20260615_033722.
    #
    # Tuning notes:
    #   • Earlier thresholds (4K tokens / 14 msgs) fired every probe-advance
    #     during a 7-minute exam — 4–5 summarizations per session, ~12 s each.
    #     Net negative for latency. We raise to 15K / 50 so summarization
    #     fires at most once on a typical exam (and often never).
    #   • Per pipecat PR #3855, if ``summary_config.llm`` is left None the
    #     summarization request is sent through the SAME LLMService that's
    #     producing the spoken response. Same API client, same rate limit
    #     → the two requests compete and tail-latency on the speech path
    #     spikes. We route summarization to a DEDICATED Cerebras LLM
    #     (cheap + fast + separate API key) so it never touches the main
    #     pipeline.
    #
    # Fallback: if CEREBRAS_API_KEY is missing, fall back to the legacy
    # in-pipeline path with a warning. Better than crashing the bot.
    _summary_llm = None
    _cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
    if _cerebras_key:
        try:
            from pipecat.services.cerebras.llm import CerebrasLLMService

            # The dedicated summarizer LLM is invoked by pipecat via
            # `_generate_summary()` directly — bypassing the pipeline entirely,
            # so the standard LLMCallDumperMixin (which hooks into push_frame)
            # would never fire. Subclass and override _generate_summary to
            # capture input + output + token usage into the JSONL dump.
            _dump_path_for_summarizer = _llm_dump_path

            class _SummarizingCerebrasLLM(CerebrasLLMService):
                async def _generate_summary(self, frame):
                    # Snapshot input (the messages about to be summarized) so
                    # we can write it AFTER super() returns. Util lives at
                    # pipecat.utils.context.llm_context_summarization in
                    # pipecat 1.3+.
                    from pipecat.utils.context.llm_context_summarization import (
                        LLMContextSummarizationUtil,
                    )
                    input_snapshot = None
                    try:
                        sel = LLMContextSummarizationUtil.get_messages_to_summarize(
                            frame.context, frame.min_messages_to_keep
                        )
                        input_snapshot = {
                            "system": frame.summarization_prompt,
                            "messages": [
                                m if isinstance(m, dict) else {"role": getattr(m, "role", "?"), "raw": str(m)}
                                for m in sel.messages
                            ],
                            "target_context_tokens": frame.target_context_tokens,
                            "min_messages_to_keep": frame.min_messages_to_keep,
                        }
                    except Exception as e:
                        logger.debug(f"summarizer dump: snapshot failed: {e}")

                    ts_in = datetime.now().isoformat(timespec="milliseconds")
                    try:
                        result = await super()._generate_summary(frame)
                        # result = (summary_text, last_summarized_index)
                        summary_text, last_idx = result
                    except Exception:
                        # Still log the failed attempt so the dashboard shows it.
                        rec = {
                            "ts": ts_in,
                            "kind": "summarization",
                            "service": type(self).__name__,
                            "model": getattr(self._settings, "model", None),
                            "input": input_snapshot,
                            "output": {"text": "", "tool_calls": []},
                            "usage": None,
                            "error": "_generate_summary raised",
                        }
                        if _dump_path_for_summarizer is not None:
                            try:
                                _dump_path_for_summarizer.parent.mkdir(parents=True, exist_ok=True)
                                with _dump_path_for_summarizer.open("a") as f:
                                    f.write(json.dumps(rec, default=str) + "\n")
                            except Exception as we:
                                logger.warning(f"summarizer dump write failed: {we}")
                        raise

                    rec = {
                        "ts": ts_in,
                        "kind": "summarization",
                        "service": type(self).__name__,
                        "model": getattr(self._settings, "model", None),
                        "input": input_snapshot,
                        "output": {"text": summary_text, "tool_calls": [],
                                   "last_summarized_index": last_idx},
                        "usage": None,   # run_inference path doesn't emit MetricsFrame
                    }
                    if _dump_path_for_summarizer is not None:
                        try:
                            _dump_path_for_summarizer.parent.mkdir(parents=True, exist_ok=True)
                            with _dump_path_for_summarizer.open("a") as f:
                                f.write(json.dumps(rec, default=str) + "\n")
                        except Exception as we:
                            logger.warning(f"summarizer dump write failed: {we}")
                    return result

            _summary_llm = _SummarizingCerebrasLLM(
                api_key=_cerebras_key,
                settings=CerebrasLLMService.Settings(
                    model="gpt-oss-120b",
                    extra={"reasoning_effort": "low"},
                ),
            )
            logger.info(
                "Auto-summarization: dedicated LLM = cerebras/gpt-oss-120b "
                "(off-pipeline, won't compete with main LLM rate limit)"
            )
        except Exception as e:
            logger.warning(
                f"Could not build dedicated Cerebras summarizer LLM: {e}. "
                "Falling back to in-pipeline summarization."
            )
            _summary_llm = None
    else:
        logger.warning(
            "CEREBRAS_API_KEY not set — auto-summarization will share the "
            "main pipeline LLM and may add tail latency. Set CEREBRAS_API_KEY "
            "to route summaries off-pipeline."
        )

    _auto_summary_cfg = LLMAutoContextSummarizationConfig(
        # 8K trigger — Gemini Flash TTFB stays flat below ~8K tokens; above
        # that each extra 1K token costs ~200ms vs ~50ms below. Observed
        # spike to 3.9s TTFB at 13.5K in session 20260620_214804_b11b11.
        # Firing at 8K keeps us in the fast zone throughout the exam.
        # (Previously 15K — didn't fire often enough on fragmented sessions.)
        max_context_tokens=8000,
        max_unsummarized_messages=None,    # token count drives it, not msg count
        summary_config=LLMContextSummaryConfig(
            # Compress aggressively to 3K — gives ~5K headroom before the
            # next trigger. At ~400 tok/probe-transition, that's ~12 more
            # turns before summarization fires again (rarely twice in a
            # typical 7-min exam).
            target_context_tokens=3000,
            min_messages_after_summary=8,  # keep last 8 messages verbatim
            summary_message_template=(
                "Conversation summary so far: {summary}"
            ),
            llm=_summary_llm,              # dedicated Cerebras — off-pipeline
        ),
    )

    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Ink-2 (CartesiaTurnsSTTService) drives its own turn detection —
            # SileroVAD would re-interrupt turns that Ink-2 is still holding
            # open for mid-thought pauses. Pass None to skip Silero entirely.
            vad_analyzer=None if _use_external_turn_strategies else SileroVADAnalyzer(),
            filter_incomplete_user_turns=_use_marker_mixin,
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
            # Fire on_user_turn_idle after this many seconds of student silence
            # following the examiner's turn. Drives the no-response escalation
            # (nudge → next question → end), so a gap can never run forever.
            user_idle_timeout=IDLE_TIMEOUT_SECS,
        ),
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=enable_auto_summarization,
            auto_context_summarization_config=_auto_summary_cfg,
        ),
    )
    logger.info(
        f"Auto-context summarization: {'ON (LLM compresses older turns at ~4K tokens)' if enable_auto_summarization else 'OFF (full transcript kept verbatim)'}"
    )

    # Publish for /api/sessions/{id}/transcript polling. Use a LOCAL transcript
    # object and register it per-session (keyed by session_id) so concurrent
    # voice exams each expose their own; also keep the single-session globals as
    # a fallback, and stash the transcript on flow_manager.state for in-exam
    # readers (e.g. anti-repetition).
    live_transcript = LiveTranscript()
    _register_live_session(session_id, context, live_transcript)
    # NOTE: flow_manager doesn't exist yet here — live_transcript is stashed onto
    # flow_manager.state in the seeding block AFTER FlowManager is created.
    global _CURRENT_CONTEXT, _CURRENT_SESSION_ID, _CURRENT_TRANSCRIPT
    _CURRENT_CONTEXT = context
    _CURRENT_SESSION_ID = session_id
    _CURRENT_TRANSCRIPT = live_transcript
    user_transcript_collector = TranscriptCollector(
        live_transcript, capture_user=True, capture_bot=False,
    )
    bot_transcript_collector = TranscriptCollector(
        live_transcript, capture_user=False, capture_bot=True,
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

    class _Ink2LatencyObserver(UserBotLatencyObserver):
        """UserBotLatencyObserver that works with CartesiaTurnsSTTService.

        The stock observer starts timing on VADUserStoppedSpeakingFrame.
        With ExternalUserTurnStrategies + Ink-2, Silero is disabled so that
        frame never fires. This subclass falls back to UserStoppedSpeakingFrame
        as the start trigger when no VAD frame has set _user_stopped_time yet.
        """

        async def on_push_frame(self, data):
            # If Ink-2 emits UserStoppedSpeakingFrame and _user_stopped_time
            # hasn't been set by a prior VAD frame, record it now so the
            # parent class can complete the latency measurement.
            from pipecat.frames.frames import UserStoppedSpeakingFrame as _USF
            if (
                isinstance(data.frame, _USF)
                and not isinstance(data.frame, VADUserStoppedSpeakingFrame)
                and self._user_stopped_time is None
            ):
                self._user_stopped_time = time.time()
                self._user_turn_start_time = self._user_stopped_time
            await super().on_push_frame(data)

    latency_observer = _Ink2LatencyObserver(enable_metrics=True)
    # Per-session token / character / audio-second tally → usage/usage_<id>.json
    session_usage = SessionUsage()

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
        observers=[
            RTVIObserver(rtvi), metrics_logger, latency_observer,
            UsageCollectorObserver(session_usage),
        ],
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
            ),
            FlowsFunctionSchema(
                name="end_exam",
                handler=handle_student_end_request,
                description=(
                    "Call IMMEDIATELY when the student clearly asks to STOP or "
                    "FINISH the WHOLE exam — e.g. 'I want to stop', 'can we stop', "
                    "'can we end now', 'I'm done', 'I am finished', 'let's finish', "
                    "'that's all I have', 'I don't want to continue', 'end the "
                    "exam', 'I want to leave'. This ENDS the session and starts "
                    "grading. Before calling, speak ONE short warm closing line "
                    "(e.g. 'Of course — thank you for your thoughts today.'). "
                    "Do NOT call this when the student only wants to skip the "
                    "current question or move to the next one — that is "
                    "advance_from_probe, not end_exam."
                ),
                properties={},
                required=[],
            ),
        ],
    )
    attempt_counter.attach(flow_manager)

    # Seed THIS exam's flow state onto flow_manager.state (from the locals we
    # captured at entry, not the possibly-racing globals). Every node builder and
    # handler reads these via _fspec()/_pfact()/etc., so concurrent exams in one
    # process each use their own state instead of clobbering module globals.
    flow_manager.state.update({
        "flow_spec": _sess_flow_spec,
        "probe_factories": _sess_probe_factories,
        "required_probes": _sess_required_probes,
        "custom_grading_prompt": _sess_custom_grading,
        "provided_student_name": _sess_provided_name,
        "programme": _sess_programme,
        # Per-session live transcript (for in-exam readers like anti-repetition).
        "live_transcript": live_transcript,
    })

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
            (_sess_flow_spec.duration_minutes if _sess_flow_spec else 6) * 60
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
            "image_description": _sess_flow_spec.image_description or None,
        }
        await grade_exam(
            wav_path, metadata, report_path,
            transcript_turns, node_trace, duration_s,
            _sess_flow_spec.exam_question, _sess_flow_spec.topic,
            usage=session_usage,
        )

    # Pacing monitor. handle_transition only acts when the examiner moves to
    # the next question, so a long run of follow-ups inside one question could
    # overrun. This never cuts the conversation off — it adds a note the
    # examiner reads on its NEXT turn (after the student's current answer):
    #   question share used     → "acknowledge and move on to the next question"
    #   closing reserve reached → "acknowledge and advance" (routing then closes)
    #   PACING_FAILSAFE_SECS over budget → close anyway (LLM ignored the notes)
    exam_clock: dict = {"task": None}

    async def _pacing_note(text: str) -> None:
        await task.queue_frame(LLMMessagesAppendFrame(
            messages=[{"role": "system", "content": text}], run_llm=False,
        ))

    async def _exam_clock():
        loop = asyncio.get_running_loop()
        q_id, q_start, q_slice, q_nudged = None, 0.0, 0.0, False
        time_up_sent = False
        try:
            while not task.has_finished():
                await asyncio.sleep(3)
                node = flow_manager.current_node or ""
                if node in ("closing", "cancelled"):
                    return
                # Only pace the exam proper, not the greeting / name capture.
                if node not in ("opening", "followup_probe") and node not in _pfact(flow_manager):
                    continue
                cur = flow_manager.state.get("current_probe")
                if cur != q_id:
                    q_id, q_start, q_nudged = cur, loop.time(), False
                    q_slice = max(_pacing(flow_manager)["slice"], MIN_QUESTION_SECS)
                ts = _compute_time_state(flow_manager)
                remaining = ts["budget"] - ts["elapsed"]
                if remaining <= -PACING_FAILSAFE_SECS:
                    logger.warning("Pacing failsafe: well over time — closing")
                    flow_manager.state.update({"ended_early_for_time": True})
                    await flow_manager.set_node_from_config(create_closing_node(flow_manager))
                    return
                if remaining <= CLOSE_RESERVE_SECS:
                    if not time_up_sent:
                        time_up_sent = True
                        logger.info("Pacing: exam time reached — closing after this answer")
                        await _pacing_note(
                            "[PACING] The exam time is up. Do NOT ask another "
                            "question. When the student finishes their current "
                            "answer, call advance_from_probe straight away — do "
                            "NOT thank them or say goodbye; the closing step "
                            "does that."
                        )
                elif not q_nudged and loop.time() - q_start > q_slice:
                    q_nudged = True
                    logger.info(f"Pacing: question {q_id!r} used its {q_slice:.0f}s share")
                    await _pacing_note(
                        "[PACING] This question has had its share of the exam "
                        "time. Do NOT ask another follow-up on it. When the "
                        "student finishes their current answer, acknowledge it "
                        "briefly and call advance_from_probe."
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("exam clock failed")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Student connected — starting audio recording")
        started_at["t"] = datetime.now()
        # Promote to flow_manager.state so handle_transition / _compute_time_state
        # can read it (started_at[] is local to this function).
        flow_manager.state.update({
            "started_at_iso": started_at["t"].isoformat(timespec="seconds"),
        })
        exam_clock["task"] = asyncio.create_task(_exam_clock())
        await audio_buffer.start_recording()
        if _sess_provided_name:
            flow_manager.state.update({"name": _sess_provided_name})
            logger.info(
                f"Using prefilled student name {_sess_provided_name!r}; "
                "skipping STT name capture."
            )
            await flow_manager.initialize(
                create_known_name_greeting_node(flow_manager)
            )
        else:
            await flow_manager.initialize(create_greeting_node(flow_manager))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Student disconnected")
        if exam_clock["task"]:
            exam_clock["task"].cancel()
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

    # No-response escalation: fires after IDLE_TIMEOUT_SECS of student silence
    # once the examiner has finished speaking. Escalates across consecutive
    # silences so an unanswered question never hangs forever.
    #   strike 1 → nudge "are you still there?"
    #   strike 2 → acknowledge + move to the next question
    #   strike 3 → thank, close the exam (→ grading starts)
    # The counter resets when the student actually answers (handle_transition).
    @context_aggregator.user().event_handler("on_user_turn_idle")
    async def on_user_turn_idle(aggregator):
        strikes = int(flow_manager.state.get("idle_strikes", 0)) + 1
        flow_manager.state.update({"idle_strikes": strikes})
        name = (flow_manager.state.get("name") or "there").strip() or "there"
        node = flow_manager.current_node or ""
        logger.info(f"Student idle — strike {strikes}/3 (node={node!r})")

        # Already closing/cancelling — let it finish, don't nag.
        if node in ("closing", "cancelled"):
            return

        try:
            if strikes == 1:
                await flow_manager.task.queue_frame(TTSSpeakFrame(
                    f"Are you still there, {name}? Take your time — "
                    "just let me know when you're ready."
                ))
            elif strikes == 2:
                # During greeting / name-capture there is no advance_from_probe
                # function registered yet, so injecting a call to it makes the
                # model reply "I don't have that function." Just nudge again
                # (strike 3 will end the session) until the exam proper begins.
                _PRE_EXAM_NODES = {"greeting", "greeting_known_name",
                                   "confirm_name", "spell_name"}
                if node in _PRE_EXAM_NODES:
                    await flow_manager.task.queue_frame(TTSSpeakFrame(
                        f"{name}, are you there? Whenever you're ready, just say "
                        "so and we'll begin."
                    ))
                else:
                    # Idle-driven advance: flag it so handle_transition does NOT
                    # reset the strike counter (this isn't a real answer), letting
                    # continued silence escalate to the end on the next timeout.
                    flow_manager.state.update({"idle_advance_pending": True})
                    await flow_manager.task.queue_frame(TTSSpeakFrame(
                        f"No problem, {name} — let's move on to the next question."
                    ))
                    await flow_manager.task.queue_frame(LLMMessagesAppendFrame(
                        messages=[{"role": "system", "content": (
                            "[SILENCE] The student did not respond to the previous "
                            "question. Call advance_from_probe now with "
                            "student_struggling=true and "
                            "student_consented_to_advance=true, then ask the next "
                            "question in one short sentence."
                        )}],
                        run_llm=True,
                    ))
            else:
                # strike >= 3: thank and end → grading.
                flow_manager.state.update({"ended_idle": True})
                logger.info("Ending exam: student unresponsive after 3 prompts")
                await flow_manager.set_node_from_config(
                    create_closing_node(flow_manager)
                )
        except Exception:
            logger.exception("idle-escalation handler failed")

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
        try:
            usage_path = session_usage.save(session_id)
            logger.info(f"Usage saved → {usage_path}")
        except Exception:
            logger.exception("Failed to write usage file")
        # Drop this session's live-transcript registry entry (the report file is
        # canonical from here on) so the registry doesn't accumulate.
        _unregister_live_session(session_id)
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
    programme: str | None = None,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: str = DEFAULT_LLM_MODEL,
    stt_provider: str = "cartesia-ink-2",
    stt_language: str | None = "en",
    tts_voice: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    use_turn_markers: str = "auto",
    enable_auto_summarization: bool = True,
) -> None:
    """Server-friendly entry: caller provides transport, session_id, and a
    pre-generated FlowSpec. CLI mode uses bot() instead."""
    set_active_flow(
        flow_spec,
        custom_grading_prompt=custom_grading_prompt,
        student_name=student_name,
        programme=programme,
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
        enable_auto_summarization=enable_auto_summarization,
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
