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
  DEEPGRAM_API_KEY           — STT
  CARTESIA_API_KEY           — TTS
  GEMINI_GRADER_MODEL        — optional, default 'gemini-3.1-pro-preview'

Run with:
    uv run dynamic_exam.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.google.llm import GoogleLLMService

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
        # Initialise parent with the first key; we'll rotate self._api_key
        # ourselves in _connect_websocket.
        super().__init__(api_key=api_keys[0], voice_id=voice_id, **kwargs)
        self._api_keys: list[str] = list(api_keys)
        self._key_index: int = 0

    @staticmethod
    def _is_rotatable_error(msg: str) -> bool:
        return any(token in msg for token in ("HTTP 402", "HTTP 401", "HTTP 403"))

    @staticmethod
    def _mask(key: str) -> str:
        return key[:6] + "..." + key[-4:] if len(key) > 10 else "(short)"

    async def _connect_websocket(self):
        last_err: Exception | None = None
        while self._key_index < len(self._api_keys):
            self._api_key = self._api_keys[self._key_index]
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
                        f"({self._mask(self._api_key)}) rejected: {msg[:120]} — "
                        f"trying next key"
                    )
                    self._key_index += 1
                    self._websocket = None
                    continue
                # Non-rotatable error — surface immediately
                logger.error(f"Cartesia non-rotatable error: {msg[:200]}")
                break
        # All keys exhausted or non-rotatable failure
        logger.error(
            f"Cartesia TTS giving up after trying {self._key_index} key(s); "
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

from pipecat_flows import FlowManager, FlowsFunctionSchema, FlowArgs, NodeConfig

from flow_planner import generate_flow
from flow_primitives import ALLOWED_PROBE_TYPES, render_probe_instructions
from flow_spec_schema import FlowSpec, ProbeNode
from flow_validator import FlowValidationError, validate_flow_spec

logger.info("All components loaded successfully!")

load_dotenv(override=True)

# ─── Paths ─────────────────────────────────────────────────────────────────────
RECORDINGS_DIR = Path(__file__).parent / "recordings"
REPORTS_DIR    = Path(__file__).parent / "reports"
FLOWS_DIR      = Path(__file__).parent / "flows"

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


def set_active_flow(
    flow_spec: FlowSpec,
    custom_grading_prompt: str | None = None,
) -> None:
    """Configure module-level state for a session. Called by the server before
    spawning a bot, or by bot() in CLI mode after generating the flow."""
    global _FLOW_SPEC, _PROBE_FACTORIES, _REQUIRED_PROBES, _CUSTOM_GRADING_PROMPT
    _FLOW_SPEC = flow_spec
    _PROBE_FACTORIES = build_probe_factories(flow_spec)
    _REQUIRED_PROBES = set(flow_spec.required_probes)
    _CUSTOM_GRADING_PROMPT = custom_grading_prompt

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
    "   If result has next_action='wait_for_student', stop and wait silently. "
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

def _off_topic_instruction(count: int, name: str) -> str:
    redirect = _FLOW_SPEC.redirect_when_off_topic if _FLOW_SPEC else (
        "Let us bring it back to the main question."
    )
    guard = (
        "CRITICAL: call off_topic_strike at most ONCE per student message. "
        "After calling it, check the result — if next_action='wait_for_student', "
        "stop immediately and wait for their next response."
    )
    if count == 0:
        return (
            f"OFF-TOPIC TRACKER [strikes: 0]: if the student's most recent message "
            f"is off-topic, redirect once warmly: '{redirect}' Then call off_topic_strike. {guard}"
        )
    if count == 1:
        return (
            f"OFF-TOPIC TRACKER [strikes: 1]: if off-topic, say firmly: "
            f"'{name}, I need you to stay on topic. Please answer the question I asked.' "
            f"Then call off_topic_strike. {guard}"
        )
    if count == 2:
        return (
            f"OFF-TOPIC TRACKER [strikes: 2]: if off-topic, issue a FINAL WARNING: "
            f"'{name}, this is your final warning. If you go off-topic again, "
            f"I will have to cancel this exam.' Then call off_topic_strike. {guard}"
        )
    return (
        f"OFF-TOPIC TRACKER [strikes: {count}]: if off-topic, say: "
        f"'{name}, the exam is now being cancelled due to repeated off-topic behaviour.' "
        f"Then call off_topic_strike. {guard}"
    )


async def handle_global_off_topic(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    count = flow_manager.state.get("off_topic_count", 0) + 1
    flow_manager.state.update({"off_topic_count": count})
    logger.info(f"Off-topic strike #{count}")

    if count >= 4:
        return ({"status": "exam_cancelled"}, create_cancel_exam_node(flow_manager))

    current = flow_manager.state.get("current_probe", "opening")
    if current == "opening":
        return ({"next_action": "wait_for_student"}, create_opening_node(flow_manager))
    if current in _PROBE_FACTORIES:
        return ({"next_action": "wait_for_student"}, _PROBE_FACTORIES[current](flow_manager))
    return ({}, create_closing_node(flow_manager))


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
                "Greet the student warmly. Introduce yourself as their IB MYP oral examiner. "
                "Explain you will have a short discussion and that there are no right or "
                "wrong answers — you want to hear their thoughts. Ask for their name.\n\n"
                f"GREETING STRIKES SO FAR: {count}.\n\n"
                "After EVERY student message, call assess_name_response ONCE with:\n"
                "  intent='gave_name'           — student gave a real first name\n"
                "  intent='needs_clarification' — confused or asking about the question\n"
                "  intent='evasion'             — unrelated tangent or deliberate dodge\n\n"
                f"For 'evasion': {evasion_tone} Then call assess_name_response.\n"
                "For 'needs_clarification': respond helpfully, then call assess_name_response.\n\n"
                "CRITICAL: call assess_name_response at most ONCE per student message. "
                "Do NOT accept single words like 'Hi' or 'okay' as a name."
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
) -> tuple[dict, NodeConfig]:
    intent = args["intent"]
    name_given = args.get("name_if_given", "").strip()

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

GRADER_MODEL = os.getenv("GEMINI_GRADER_MODEL", "gemini-3.1-pro-preview")

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
        turns.append({"speaker": speaker, "text": str(content).strip()})
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
            current = state.get("current_probe")
            if current:
                attempts: dict[str, int] = dict(state.get("probe_attempts", {}))
                attempts[current] = attempts.get(current, 0) + 1
                state.update({"probe_attempts": attempts})
                logger.info(f"Probe attempt: {current} #{attempts[current]}")
        await self.push_frame(frame, direction)


# ─── Pipecat bot setup ─────────────────────────────────────────────────────────

_STT_STOPWORDS = {
    "what", "when", "where", "which", "while", "would", "could", "should",
    "their", "there", "these", "those", "this", "that", "with", "from",
    "into", "your", "yours", "about", "ask", "asking", "tell", "telling",
    "student", "students", "answer", "answers", "examiner", "exam", "topic",
    "probe", "prompt", "scaffold", "respond", "response", "based", "they",
    "them", "then", "than", "have", "has", "had", "been", "being", "will",
    "you", "yes", "the", "and", "for", "are", "but", "not", "one", "all",
    "any", "out", "off", "use", "way", "may", "say", "see", "ask",
}


# Common Indian/South-Asian first names — biases Deepgram toward proper-noun
# recognition during the greeting node. Capitalisation is preserved by keyterm
# prompting, so the bot gets the name back already title-cased.
_NAME_KEYTERMS = [
    "Aarav", "Aanya", "Aarush", "Aditi", "Aditya", "Advait", "Advika", "Akash",
    "Aman", "Amrita", "Ananya", "Anika", "Anish", "Anjali", "Ansh", "Anushka",
    "Arjun", "Arnav", "Aryan", "Atharva", "Avani", "Avinash", "Ayaan", "Ayush",
    "Bhavya", "Chirag", "Daksh", "Darsh", "Devansh", "Dhruv", "Diya", "Esha",
    "Gauri", "Harini", "Harsh", "Hiya", "Ishaan", "Ishita", "Ivaan", "Jay",
    "Kabir", "Kairav", "Kavya", "Khushi", "Kiaan", "Krish", "Krisha", "Laksh",
    "Lavanya", "Manav", "Mira", "Mithra", "Myra", "Naina", "Navya", "Neil",
    "Nikhil", "Nisha", "Nitya", "Om", "Pari", "Parth", "Pranav", "Prisha",
    "Priya", "Purva", "Raghav", "Rahul", "Reyansh", "Rhea", "Riya", "Rohan",
    "Rudra", "Saanvi", "Sahana", "Sahil", "Samaira", "Samar", "Sanvi", "Sara",
    "Shaurya", "Shivansh", "Shreya", "Siddharth", "Sneha", "Soham", "Tanvi",
    "Tara", "Tejas", "Vaani", "Vansh", "Veer", "Vihaan", "Vivaan", "Yash",
    "Zara",
]


def _extract_keyterms(flow_spec: FlowSpec, limit: int = 100) -> list[str]:
    """Pull domain-specific words from the FlowSpec to bias Deepgram STT
    toward terms likely to appear in this exam. Lower-cases, dedupes, drops
    common English stopwords + short words."""
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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments, session_id: str):
    logger.info(f"Starting dynamic oral exam bot | session={session_id}")

    # Per-session keyterms biasing Deepgram toward this exam's vocabulary.
    # Prepend common South-Asian first names so the greeting node lands the
    # student's name correctly (Nova-3 keyterm prompting preserves casing and
    # gives a documented uplift on proper-noun entity recognition).
    domain_keyterms = _extract_keyterms(_FLOW_SPEC) if _FLOW_SPEC else []
    # Deepgram caps keyterms at ~100; keep names first, top up with domain words.
    keyterms = (_NAME_KEYTERMS + domain_keyterms)[:100]
    logger.info(
        f"STT keyterms ({len(keyterms)}): {len(_NAME_KEYTERMS)} name(s) + "
        f"{len(keyterms) - len(_NAME_KEYTERMS)} domain term(s); "
        f"sample={keyterms[:10]}…"
    )

    # "multi" → Nova-3 multilingual (March 2026) with multilingual keyterm
    # prompting. Handles Indian-English phonotactics for student names far
    # better than the US-leaning default. Override via STT_LANGUAGE=en-IN or =en
    # if needed.
    stt_language = os.getenv("STT_LANGUAGE", "multi")
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-3",
            language=stt_language,
            keyterm=keyterms or None,
            smart_format=True,
            punctuate=True,
        ),
    )
    cartesia_keys = _load_cartesia_keys()
    if not cartesia_keys:
        raise RuntimeError(
            "No Cartesia API key configured — set CARTESIA_API_KEY in .env"
        )
    logger.info(
        f"Cartesia TTS configured with {len(cartesia_keys)} key(s): "
        f"primary={FallbackCartesiaTTSService._mask(cartesia_keys[0])}"
        + (f", fallbacks={[FallbackCartesiaTTSService._mask(k) for k in cartesia_keys[1:]]}" if len(cartesia_keys) > 1 else " (no fallbacks)")
    )
    tts = FallbackCartesiaTTSService(
        api_keys=cartesia_keys,
        voice_id="e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    )
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-3.5-flash",
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

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

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        attempt_counter,
        user_transcript_collector,     # captures TranscriptionFrame BEFORE the user
                                        # aggregator consumes it
        context_aggregator.user(),
        llm,
        tts,
        bot_transcript_collector,      # captures TTSTextFrame after TTS
        transport.output(),
        audio_buffer,
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),
        observers=[RTVIObserver(rtvi)],
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
        await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Student disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    try:
        await runner.run(task)
    finally:
        # Keep context readable briefly after disconnect so the UI can render
        # the final transcript before grading completes, then clear.
        pass  # _CURRENT_CONTEXT is cleared when the next run_bot starts


async def run_session(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    session_id: str,
    flow_spec: FlowSpec,
    custom_grading_prompt: str | None = None,
) -> None:
    """Server-friendly entry: caller provides transport, session_id, and a
    pre-generated FlowSpec. CLI mode uses bot() instead."""
    set_active_flow(flow_spec, custom_grading_prompt=custom_grading_prompt)
    logger.info(
        f"Flow ready | session={session_id} | {len(flow_spec.probe_pool)} probes | "
        f"required={flow_spec.required_probes} | question={flow_spec.exam_question!r}"
    )
    await run_bot(transport, runner_args, session_id)


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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args, session_id)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
