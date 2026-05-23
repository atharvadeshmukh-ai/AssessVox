#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""IB MYP Environmental Responsibility Oral Exam Bot.

A voice AI oral examiner for an IB MYP language class (ages 11–16) conducting
a structured 4–6 minute discussion on Environmental Responsibility.

Exam topic:
  "Who do you think is most responsible for solving climate change —
  individuals, governments, or big companies? Why?"

Flow (7 nodes):
  1. Greeting          — introduce exam, collect student's name
  2. Opening Question  — main stance + depth assessment
  3. Counter Probe     — "Can you think of a counterargument to your position?"
  4. Example Probe     — "Is there something your school or city does that relates?"
  5. Effectiveness     — "Do you think that actually makes a difference? Why?"
  6. Comparison Probe  — local vs global action, fairness angle
  7. Closing           — thank student, end conversation

Guardrail (all nodes 2–6):
  Each no=de exposes an off_topic_strike function.
  Strike 1 → warm redirect (count=1)
  Strike 2 → firm redirect (count=2)
  Strike 3 → final warning: exam will be cancelled (count=3)
  Strike 4 → exam is cancelled, conversation ends (count≥4)

Required AI services:
  - Deepgram (Speech-to-Text)
  - Google Gemini 2.5 Flash Lite (LLM)
  - Cartesia (Text-to-Speech)

Run with:
    uv run teacher2.py
"""

import asyncio
import json
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from loguru import logger

print("Starting IB MYP Environmental Responsibility Oral Examiner bot...")
print("Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
logger.info("Local Smart Turn Analyzer V3 loaded")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
logger.info("Silero VAD model loaded")

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService

from pipecat_flows import FlowManager, FlowsFunctionSchema, FlowArgs, NodeConfig

logger.info("All components loaded successfully!")

load_dotenv(override=True)

# ─── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a warm, encouraging oral examiner for an IB MYP language class "
    "(students aged 11 to 16). Your role is to facilitate a 4 to 6 minute "
    "discussion on Environmental Responsibility. "
    "You MUST ALWAYS call the provided function to advance the conversation. "
    "STRICT RULES — never break these: "
    "1. Do NOT correct grammar, pronunciation, or word choice. "
    "2. Do NOT share your own opinion on who is responsible for climate change. "
    "3. Do NOT accept a list of environmental facts as a substitute for opinion. "
    "   If the student gives facts instead of opinions, redirect: "
    "   'That is good background — but coming back to the question of responsibility, "
    "   who should actually act on that knowledge?' "
    "4. Do NOT accept single-word answers — always ask the student to elaborate. "
    "5. NEVER repeat the same follow-up phrase twice in a row. Vary your language. "
    "6. INTENT DETECTION: classify the student's intent before acting. "
    "   'needs_clarification' = student is confused or asking about the question format "
    "   (e.g. 'do you want my full name?', 'what do you mean?') — respond helpfully, do not strike. "
    "   'evasion' = unrelated jokes, tangents, or questions that have nothing to do with "
    "   the exam topic — redirect firmly and call the appropriate function. "
    "   When in doubt, treat the student generously: if they seem to be trying to engage, "
    "   do NOT count it as evasion. "
    "7. FUNCTION RESULT RULE: after calling any function, check the result. "
    "   If the result contains 'next_action: wait_for_student', stop immediately. "
    "   Do NOT speak again. Do NOT call another function. "
    "   Wait silently until the student sends their next message. "
    "8. OFF-TOPIC HANDLING: follow the OFF-TOPIC TRACKER instruction in your task. "
    "   Call off_topic_strike or assess_name_response at most ONCE per student message. "
    "   Never chain function calls. "
    "8. This is a voice conversation — speak naturally and warmly. "
    "   No bullet points, no markdown, no emojis. "
    "9. Keep your own turns short — you are facilitating, not teaching."
)
ROLE_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]

EXAM_QUESTION = (
    "Who do you think is most responsible for solving climate change — "
    "individuals, governments, or big companies? Why?"
)


# ─── Off-topic Guardrail Helpers ───────────────────────────────────────────────

def _off_topic_instruction(count: int, name: str) -> str:
    """Return the exact instruction the LLM should follow for off-topic behavior
    based on how many strikes have already been recorded."""
    guard = (
        "CRITICAL: only call off_topic_strike if the student's MOST RECENT message "
        "is off-topic. Call it at most ONCE per student message, then stop and wait "
        "for their next response before evaluating again."
    )
    if count == 0:
        return (
            "OFF-TOPIC TRACKER [strikes so far: 0]: "
            "If the student's most recent message is off-topic, redirect ONCE warmly — "
            "acknowledge briefly then bring them back. Then call off_topic_strike. "
            f"{guard}"
        )
    elif count == 1:
        return (
            "OFF-TOPIC TRACKER [strikes so far: 1]: "
            f"If the student's most recent message is off-topic, drop all warmth and say: "
            f"'{name}, I need you to stay on topic. Please answer the question I asked.' "
            f"Then call off_topic_strike. {guard}"
        )
    elif count == 2:
        return (
            "OFF-TOPIC TRACKER [strikes so far: 2]: "
            f"If the student's most recent message is off-topic, issue a FINAL WARNING: "
            f"'{name}, this is your final warning. If you go off-topic one more time, "
            f"I will have to cancel this exam.' Then call off_topic_strike. {guard}"
        )
    else:
        return (
            f"OFF-TOPIC TRACKER [strikes so far: {count}]: "
            f"If the student's most recent message is off-topic, say: "
            f"'{name}, you have gone off-topic too many times. I am cancelling this exam.' "
            f"Then call off_topic_strike. {guard}"
        )


def make_off_topic_handler(
    node_factory: Callable[[FlowManager], NodeConfig]
) -> Callable:
    """Returns a handler that increments the off-topic counter and either
    re-enters the same node (with updated counter) or routes to the cancel node."""
    async def handler(args: FlowArgs, flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
        count = flow_manager.state.get("off_topic_count", 0) + 1
        flow_manager.state.update({"off_topic_count": count})
        logger.info(f"Off-topic strike #{count}")
        result = {
            "status": "warning_delivered",
            "strike_count": count,
            "next_action": "wait_for_student",
        }
        if count >= 4:
            return (result | {"status": "exam_cancelled"}, create_cancel_exam_node(flow_manager))
        return (result, node_factory(flow_manager))
    return handler


def off_topic_function(node_factory: Callable[[FlowManager], NodeConfig]) -> FlowsFunctionSchema:
    """Build the off_topic_strike FlowsFunctionSchema for any node."""
    return FlowsFunctionSchema(
        name="off_topic_strike",
        handler=make_off_topic_handler(node_factory),
        description=(
            "Call this whenever the student goes off-topic, after delivering the "
            "appropriate redirect or warning based on the OFF-TOPIC TRACKER instruction."
        ),
        properties={},
        required=[],
    )


# ─── Cancel Exam Node ──────────────────────────────────────────────────────────

def create_cancel_exam_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    return NodeConfig(
        name="cancelled",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Inform {name} in one firm, calm sentence that the exam is being cancelled "
                "due to repeated off-topic behavior. Do not apologise or soften it. "
                "Do not ask any questions."
            ),
        }],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )


# ─── Node 1: Greeting ──────────────────────────────────────────────────────────

def create_greeting_node(flow_manager: FlowManager = None) -> NodeConfig:
    count = flow_manager.state.get("greeting_strikes", 0) if flow_manager else 0

    if count == 0:
        evasion_tone = "Tone: warm. Say 'Just your first name is fine — what shall I call you?'"
    elif count == 1:
        evasion_tone = (
            "Tone: firm, no warmth. Say 'I can see you are not answering. "
            "This is a formal exam — I need your first name now.'"
        )
    else:
        evasion_tone = (
            "Tone: final warning. Say 'This is your last chance. "
            "Tell me your name or I will end this session.'"
        )

    return NodeConfig(
        name="greeting",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                "Greet the student warmly. Introduce yourself as their IB MYP oral examiner. "
                "Explain you will have a short discussion about environmental responsibility "
                "and that there are no right or wrong answers — you want to hear their thinking. "
                "Ask for their name.\n\n"
                f"GREETING STRIKES SO FAR: {count}.\n\n"
                "After EVERY student message, call assess_name_response exactly ONCE with:\n"
                "  intent='gave_name'           — student gave a real first name\n"
                "  intent='needs_clarification' — student is confused or asking about the "
                "question (e.g. 'Hi', 'sorry what?', 'full name or nickname?') — respond "
                "helpfully before calling the function\n"
                "  intent='evasion'             — completely unrelated tangent or joke\n\n"
                f"For 'evasion': {evasion_tone} Then call assess_name_response.\n\n"
                "CRITICAL: call assess_name_response at most ONCE per student message. "
                "If the function result contains next_action='wait_for_student', "
                "stop immediately — do NOT speak again, do NOT call the function again. "
                "Wait silently for the student's next message.\n\n"
                "Do NOT accept single non-name words ('Hi', 'okay', 'yes') as a name."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="assess_name_response",
                handler=handle_greeting_routing,
                description=(
                    "Call this after every student message. "
                    "Classify intent: 'gave_name', 'needs_clarification', or 'evasion'. "
                    "Set name_if_given to the name if one was provided."
                ),
                properties={
                    "intent": {
                        "type": "string",
                        "enum": ["gave_name", "needs_clarification", "evasion"],
                        "description": (
                            "gave_name: real first name provided. "
                            "needs_clarification: trying to engage but confused. "
                            "evasion: deliberate off-topic behavior."
                        ),
                    },
                    "name_if_given": {
                        "type": "string",
                        "description": "The name the student gave, or empty string.",
                    },
                },
                required=["intent", "name_if_given"],
            ),
        ],
    )


# ─── Node 2: Opening Question ──────────────────────────────────────────────────

def create_opening_question_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    count = flow_manager.state.get("off_topic_count", 0)
    return NodeConfig(
        name="opening_question",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"Ask {name} the following exam question, word for word: "
                f"'{EXAM_QUESTION}' "
                "Give the student time to respond. Then:\n\n"
                "STEP 1 — CLARIFY STANCE.\n"
                "If the student does not clearly pick one actor (individuals, governments, "
                "or companies), or gives only facts without an opinion, redirect: "
                "'That is good background — but coming back to the question of responsibility, "
                "who should act? Individuals, governments, or companies?' "
                "Do not move on until they name at least one actor.\n\n"
                "STEP 2 — BUILD REASONING.\n"
                "Once they have a stance, if the reasoning is thin (one sentence), "
                "encourage more with varied phrases: "
                "'Why do you think that?' / 'What makes you say that?' / "
                "'Can you say a bit more about why you chose that group?' "
                "If after TWO attempts they still cannot reason, scaffold: "
                "'Think about who has the most power to make changes — "
                "is it a single person, a law, or a company policy?' "
                "or offer a starter: 'I think governments are most responsible because...'\n\n"
                "STEP 3 — CALL FUNCTION.\n"
                "Only call advance_from_opening once the student has a clear stance AND "
                "at least one reason. "
                "Set student_stance to 'individuals', 'governments', 'companies', or 'mixed'. "
                "Set depth to 'surface' (thin reasoning) or 'developed' (explained reasoning). "
                "Set vague_term to any imprecise word worth probing later, or empty string."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_opening",
                handler=handle_advance_from_opening,
                description=(
                    "Call once student has clear stance + at least one reason. "
                    "student_stance: 'individuals'/'governments'/'companies'/'mixed'. "
                    "depth: 'surface' or 'developed'."
                ),
                properties={
                    "student_stance": {
                        "type": "string",
                        "enum": ["individuals", "governments", "companies", "mixed"],
                        "description": "Which actor the student holds most responsible",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["surface", "developed"],
                        "description": "Depth of the student's reasoning",
                    },
                    "vague_term": {
                        "type": "string",
                        "description": "Vague word to probe later. Empty string if none.",
                    },
                },
                required=["student_stance", "depth", "vague_term"],
            ),
            off_topic_function(create_opening_question_node),
        ],
    )


# ─── Node 3: Counter-argument Probe ───────────────────────────────────────────

def create_counter_probe_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    stance = flow_manager.state.get("student_stance", "that group")
    count = flow_manager.state.get("off_topic_count", 0)
    return NodeConfig(
        name="counter_probe",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"Acknowledge {name}'s view briefly, then ask them to steelman the other side. "
                "Use a variation of: "
                "'Can you think of a counterargument to your own position?' "
                f"or: 'Someone who disagrees with putting the responsibility on {stance} "
                "might say — what do you think their argument would be?' "
                "If the student says they cannot think of anything, scaffold: "
                "'Think about the other two groups — individuals, governments, companies — "
                "what argument could each of them make that it is not their job?' "
                "If the student introduces a specific insight (e.g. 'individuals do not "
                "have enough power'), ask ONE follow-up to deepen it before calling. "
                "Call advance_from_counter once the student has made a genuine attempt."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_counter",
                handler=handle_advance_from_counter,
                description="Call once the student has attempted a counter-argument.",
                properties={},
                required=[],
            ),
            off_topic_function(create_counter_probe_node),
        ],
    )


# ─── Node 4: Real-world Example Probe ─────────────────────────────────────────

def create_example_probe_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    count = flow_manager.state.get("off_topic_count", 0)
    return NodeConfig(
        name="example_probe",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"Bridge naturally and ask {name} for a real-world connection. "
                "Use a variation of: "
                "'Is there something your school, your city, or your community does "
                "that relates to what you have been saying?' "
                "or: 'Can you think of an example from your own life or neighbourhood "
                "that connects to this idea of responsibility?' "
                "If they say they cannot think of one, scaffold: "
                "'Think about things like recycling programmes, school campaigns, "
                "local policies, or things your family does — does anything come to mind?' "
                "Once the student names a concrete example, call advance_from_example."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_example",
                handler=handle_advance_from_example,
                description="Call once the student has given a concrete real-world example.",
                properties={},
                required=[],
            ),
            off_topic_function(create_example_probe_node),
        ],
    )


# ─── Node 5: Effectiveness Probe ──────────────────────────────────────────────

def create_effectiveness_probe_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    count = flow_manager.state.get("off_topic_count", 0)
    return NodeConfig(
        name="effectiveness_probe",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"Follow up on the example {name} just gave by questioning its impact. "
                "Use a variation of: "
                "'Do you think that actually makes a difference? Why or why not?' "
                "or: 'How effective do you think that is in the bigger picture?' "
                "If they say yes without explaining, push: "
                "'What kind of difference do you think it makes — is it big or small?' "
                "If they say no, ask: 'So what would make a real difference then?' "
                "If the student introduces a cause-and-effect idea, ask ONE follow-up "
                "to explore it. "
                "Call advance_from_effectiveness once the student has evaluated the "
                "real-world impact of their example."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_effectiveness",
                handler=handle_advance_from_effectiveness,
                description="Call once the student has evaluated the effectiveness of their example.",
                properties={},
                required=[],
            ),
            off_topic_function(create_effectiveness_probe_node),
        ],
    )


# ─── Node 6: Comparison / Fairness Probe ──────────────────────────────────────

def create_comparison_probe_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    stance = flow_manager.state.get("student_stance", "that group")
    count = flow_manager.state.get("off_topic_count", 0)
    return NodeConfig(
        name="comparison_probe",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"Invite {name} to think about fairness and local vs global action. "
                "Use a variation of: "
                "'Do you think it is fair to expect ordinary people to change their "
                "lifestyle when big companies cause far more pollution?' "
                "or: 'Is tackling climate change something that can be solved locally, "
                "or does it need global cooperation? What do you think?' "
                f"If the student just restates their earlier stance (that {stance} should "
                "be responsible), push them to compare: "
                "'How does what one individual does compare to what one government can do?' "
                "Encourage use of contrast language: however, on the other hand, whereas. "
                "If the student uses a connector naturally, acknowledge it briefly "
                "('good point, and the contrast you just drew is important'). "
                "Call advance_from_comparison once the student has compared at least "
                "two different actors or scales of action."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_comparison",
                handler=handle_advance_from_comparison,
                description="Call once the student has compared different actors or scales of responsibility.",
                properties={},
                required=[],
            ),
            off_topic_function(create_comparison_probe_node),
        ],
    )


# ─── Node 7: Closing ───────────────────────────────────────────────────────────

def create_closing_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    return NodeConfig(
        name="closing",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Thank {name} warmly for their thoughtful responses in one or two short "
                "sentences. Let them know the oral discussion is now complete. "
                "Do NOT give feedback on their language, performance, or scores. "
                "Do NOT ask any further questions."
            ),
        }],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )


# ─── Handlers ──────────────────────────────────────────────────────────────────

async def handle_greeting_routing(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    intent = args["intent"]
    name_given = args.get("name_if_given", "").strip()

    if intent == "gave_name" and name_given:
        logger.info(f"Name recorded: {name_given!r}")
        flow_manager.state.update({"name": name_given, "off_topic_count": 0})
        _trace_node(flow_manager, "opening_question")
        return (
            {"status": "name_recorded", "name": name_given},
            create_opening_question_node(flow_manager),
        )

    if intent == "needs_clarification":
        logger.info("Greeting: student needs clarification, no strike")
        return (
            {"status": "clarifying", "next_action": "wait_for_student"},
            create_greeting_node(flow_manager),
        )

    # intent == "evasion"
    count = flow_manager.state.get("greeting_strikes", 0) + 1
    flow_manager.state.update({"greeting_strikes": count})
    logger.info(f"Greeting evasion strike #{count}")
    result = {"status": "warning_delivered", "strike": count, "next_action": "wait_for_student"}
    if count >= 3:
        return (result | {"status": "exam_cancelled"}, create_cancel_exam_node(flow_manager))
    return (result, create_greeting_node(flow_manager))


def _trace_node(flow_manager: FlowManager, node_name: str) -> None:
    trace = flow_manager.state.setdefault("node_trace", [])
    trace.append({"node": node_name, "ts": datetime.now().isoformat(timespec="seconds")})


async def handle_advance_from_opening(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[None, NodeConfig]:
    stance = args["student_stance"]
    depth = args["depth"]
    vague_term = args.get("vague_term", "")
    logger.info(f"Opening — stance: {stance!r} | depth: {depth!r} | vague: {vague_term!r}")
    flow_manager.state.update({
        "student_stance": stance,
        "opening_depth": depth,
        "vague_term": vague_term,
    })
    _trace_node(flow_manager, "counter_probe")
    return (None, create_counter_probe_node(flow_manager))


async def handle_advance_from_counter(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[None, NodeConfig]:
    logger.info("Counter probe complete")
    _trace_node(flow_manager, "example_probe")
    return (None, create_example_probe_node(flow_manager))


async def handle_advance_from_example(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[None, NodeConfig]:
    logger.info("Example probe complete")
    _trace_node(flow_manager, "effectiveness_probe")
    return (None, create_effectiveness_probe_node(flow_manager))


async def handle_advance_from_effectiveness(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[None, NodeConfig]:
    logger.info("Effectiveness probe complete")
    _trace_node(flow_manager, "comparison_probe")
    return (None, create_comparison_probe_node(flow_manager))


async def handle_advance_from_comparison(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[None, NodeConfig]:
    logger.info("Comparison probe complete")
    _trace_node(flow_manager, "closing")
    return (None, create_closing_node(flow_manager))


# ─── Grading ──────────────────────────────────────────────────────────────────

RECORDINGS_DIR = Path(__file__).parent / "recordings"
REPORTS_DIR = Path(__file__).parent / "reports"
GRADER_MODEL = os.getenv("GEMINI_GRADER_MODEL", "gemini-3-pro-preview")

DIMENSIONS = {
    "clarity":       {"weight": 0.10, "label": "Clarity of Position"},
    "reasoning":     {"weight": 0.25, "label": "Reasoning & Critical Thinking"},
    "examples":      {"weight": 0.20, "label": "Examples & Conceptual Depth"},
    "engagement":    {"weight": 0.10, "label": "Engagement & Focus"},
    "fluency":       {"weight": 0.15, "label": "Fluency & Pacing"},
    "pronunciation": {"weight": 0.10, "label": "Pronunciation & Intelligibility"},
    "confidence":    {"weight": 0.10, "label": "Confidence & Prosody"},
}
MIN_STUDENT_WORDS_FOR_ENGAGEMENT = 50

DIMENSION_GUIDE = """\
  clarity        — How directly and unambiguously did the student state and
                   maintain their position? Look for an explicit stance and
                   consistent reference to it.
  reasoning      — Quality of argumentation AND treatment of counterarguments.
                   Does the student justify their view, use because/however/
                   on the other hand, and seriously consider the other side
                   instead of dismissing it?
  examples       — Real-world grounding plus conceptual depth. Are examples
                   specific (named places, programs, observations) or vague?
                   Does the student engage with concepts like scale, fairness,
                   local vs global, sustainability?
  engagement     — On-topic discipline and willingness to attempt every probe.
                   Use strike_count from metadata AND student_word_count.
                   Very low word count = low engagement even if on-topic.
  fluency        — Flow, pacing, and pause patterns. Use student_wpm verbatim
                   (ideal 130–150). Note fragmented or mid-sentence pauses.
  pronunciation  — Intelligibility and word-choice precision from the
                   transcript and audio.
  confidence     — Prosody, conviction, and filler control (um/uh/like and
                   crutch words like 'well').
"""

GRADING_PROMPT = """You are an IB MYP oral exam evaluator producing a HOLISTIC
report on a single 4–6 minute discussion. Do not grade turn-by-turn. Read the
whole exam first, then write one integrated judgment per dimension.

The attached audio is a full recording. Two voices: an AI EXAMINER and the
STUDENT. Grade ONLY the student.

Exam question:
  "Who do you think is most responsible for solving climate change —
   individuals, governments, or big companies? Why?"

Exam metadata (JSON):
{metadata}

Diarized transcript — authoritative source for evidence:
{transcript}

Node trace (which probes were reached):
{node_trace}

Pre-computed numerics — use verbatim, do not re-estimate:
  student_word_count:       {student_word_count}
  student_speaking_seconds: {student_speaking_seconds}
  student_wpm:              {student_wpm}

────────────────────────────────────────────────────────────────────────────
DIMENSIONS — score each 1–4 (1 Emerging · 2 Developing · 3 Proficient · 4 Mastery)
────────────────────────────────────────────────────────────────────────────
{dimension_guide}

For each dimension produce:
  - score:    integer 1–4
  - analysis: 2–4 sentences of integrated commentary across the WHOLE exam.
              Reference patterns, not single moments. Be specific.
  - evidence: a list of 1–3 short student quotes from the transcript that
              best support the score. Never quote the examiner. Never write
              "Entire recording".

Then produce ONE holistic narrative `summary` (4–6 sentences) describing the
student's overall performance as a coherent picture — voice, thinking,
engagement — not a list of dimension recaps.

Also produce:
  - strengths:    2–3 specific strengths (one sentence each)
  - growth_areas: 2–3 concrete, actionable next steps (one sentence each)
  - headline:     one sentence (≤ 20 words) capturing the overall impression

Do NOT compute overall_score — leave it at 0.0; it is computed
deterministically afterwards from your dimension scores.

If metadata.cancelled is true, set every dimension score to 1 and explain
in summary that the exam was terminated.

Return ONLY valid JSON, no markdown, exactly this shape:
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
  "overall_score": 0.0
}}
"""


def _score_to_pct(score: int) -> float:
    return (max(1, min(4, score)) - 1) / 3 * 100


def _compute_overall(report: dict, student_word_count: int) -> dict:
    """Compute overall_score from dimension scores using fixed weights."""
    dims = report.get("dimensions") or {}
    weights = {k: v["weight"] for k, v in DIMENSIONS.items()}

    # If student barely spoke, drop engagement so silence doesn't read as focus.
    dropped = []
    if student_word_count < MIN_STUDENT_WORDS_FOR_ENGAGEMENT:
        weights.pop("engagement", None)
        dropped.append("engagement")

    total_w = sum(weights.values()) or 1.0
    overall = 0.0
    for key, w in weights.items():
        cell = dims.get(key)
        if not isinstance(cell, dict) or "score" not in cell:
            continue
        score = int(cell["score"])
        cell["score_pct"] = round(_score_to_pct(score), 1)
        cell["label"] = DIMENSIONS[key]["label"]
        cell["weight"] = w
        overall += _score_to_pct(score) * w
    overall = overall / total_w

    # Decorate dropped dimensions so they still appear but don't affect score.
    for d in dropped:
        cell = dims.get(d)
        if isinstance(cell, dict) and "score" in cell:
            cell["score_pct"] = round(_score_to_pct(int(cell["score"])), 1)
            cell["label"] = DIMENSIONS[d]["label"]
            cell["weight"] = 0.0
            cell["excluded_reason"] = (
                f"student_word_count={student_word_count} < "
                f"{MIN_STUDENT_WORDS_FOR_ENGAGEMENT}"
            )

    report["overall_score"] = round(overall, 1)
    report["scoring_notes"] = {
        "weights_used": {k: DIMENSIONS[k]["weight"] for k in DIMENSIONS},
        "excluded_dimensions": dropped,
    }
    return report


def _extract_transcript(context) -> list[dict]:
    """Pull diarized turns from the LLM context. user=STUDENT, assistant=EXAMINER."""
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


def _student_word_count(turns: list[dict]) -> int:
    return sum(len(t["text"].split()) for t in turns if t["speaker"] == "STUDENT")


def _write_wav(path: Path, pcm: bytes, sample_rate: int, num_channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _grade_exam_sync(
    wav_path: Path,
    metadata: dict,
    transcript_str: str,
    node_trace_str: str,
    student_word_count: int,
    student_speaking_seconds: float,
    student_wpm: float,
) -> dict:
    """Blocking Gemini call. Run via asyncio.to_thread."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    audio_bytes = wav_path.read_bytes()

    prompt = GRADING_PROMPT.format(
        metadata=json.dumps(metadata, indent=2),
        transcript=transcript_str,
        node_trace=node_trace_str,
        student_word_count=student_word_count,
        student_speaking_seconds=round(student_speaking_seconds, 1),
        student_wpm=round(student_wpm, 1),
        dimension_guide=DIMENSION_GUIDE,
    )

    response = client.models.generate_content(
        model=GRADER_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    text = response.text or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Grader returned non-JSON: {text[:500]}")
        return {"error": "invalid_json", "raw": text}


async def grade_exam(
    wav_path: Path,
    metadata: dict,
    report_path: Path,
    transcript_turns: list[dict],
    node_trace: list[dict],
    duration_seconds: float,
) -> None:
    logger.info(f"Grading exam → sending {wav_path.name} to {GRADER_MODEL}")

    student_words = _student_word_count(transcript_turns)
    # Approximate student speaking time: assume student spoke for the
    # portion of duration proportional to their share of words. Crude
    # but deterministic, and only used for WPM sanity.
    total_words = sum(len(t["text"].split()) for t in transcript_turns) or 1
    student_speaking_seconds = (duration_seconds or 0) * (student_words / total_words)
    student_wpm = (
        (student_words / student_speaking_seconds * 60)
        if student_speaking_seconds > 0 else 0.0
    )

    transcript_str = _format_transcript(transcript_turns)
    node_trace_str = (
        "\n".join(f"  {n['ts']}  → {n['node']}" for n in node_trace)
        if node_trace else "(no transitions recorded)"
    )

    try:
        report = await asyncio.to_thread(
            _grade_exam_sync,
            wav_path,
            metadata,
            transcript_str,
            node_trace_str,
            student_words,
            student_speaking_seconds,
            student_wpm,
        )
    except Exception as e:
        logger.exception(f"Grading failed: {e}")
        report = {"error": str(e)}

    if "error" not in report:
        report = _compute_overall(report, student_words)

    report["metadata"] = metadata
    report["audio_file"] = str(wav_path)
    report["transcript"] = transcript_turns
    report["node_trace"] = node_trace
    report["computed"] = {
        "student_word_count": student_words,
        "student_speaking_seconds": round(student_speaking_seconds, 1),
        "student_wpm": round(student_wpm, 1),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Report written → {report_path}")


# ─── Bot Setup ─────────────────────────────────────────────────────────────────

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting IB MYP environmental responsibility examiner bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-3.1-flash-lite-preview",
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)
    rtvi = RTVIProcessor()

    audio_buffer = AudioBufferProcessor(sample_rate=16000, num_channels=1)

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        context_aggregator.user(),
        llm,
        tts,
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
    )

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = RECORDINGS_DIR / f"exam_{session_id}.wav"
    report_path = REPORTS_DIR / f"report_{session_id}.json"
    started_at = {"t": None}
    graded = {"done": False}

    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio: bytes, sample_rate: int, num_channels: int):
        # Fired automatically when the pipeline ends (EndFrame triggers stop_recording).
        # Buffers are cleared right after this handler returns, so we must capture here.
        if graded["done"]:
            return
        graded["done"] = True

        if not audio:
            logger.warning("on_audio_data fired with no audio — skipping grading")
            return

        _write_wav(wav_path, audio, sample_rate, num_channels)
        logger.info(
            f"Audio saved → {wav_path} "
            f"({len(audio) / (sample_rate * 2 * num_channels):.1f}s "
            f"@ {sample_rate}Hz, {num_channels}ch)"
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
        metadata = {
            "session_id": session_id,
            "student_name": state.get("name"),
            "student_stance": state.get("student_stance"),
            "opening_depth": state.get("opening_depth"),
            "vague_term": state.get("vague_term"),
            "strike_count": strike_count,
            "cancelled": cancelled,
            "duration_seconds": duration_s,
            "nodes_reached": [n["node"] for n in node_trace],
        }
        await grade_exam(
            wav_path, metadata, report_path,
            transcript_turns, node_trace, duration_s,
        )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Student connected — starting audio recording")
        started_at["t"] = datetime.now()
        await audio_buffer.start_recording()
        await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Student disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
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
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
