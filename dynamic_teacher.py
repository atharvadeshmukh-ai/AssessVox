#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dynamic IB MYP Oral Exam Bot.

Accepts any exam topic via the EXAM_TOPIC environment variable.
Before the voice session starts, an LLM generates a FlowSpec (probe pool,
required probes, exam question). During the session, a smart handler routes
dynamically between probes based on what the student actually says.

Flow guarantees (no infinite loops):
  - covered_probes set grows monotonically — never revisit a completed probe
  - node_count hard cap at 12 → forced closing
  - gold_followed cap at 1 → prevents follow-up chains
  - Pipecat idle detection (300 s) → EndFrame fallback

Required env vars:
  EXAM_TOPIC       — the topic string (default: gene editing question)
  GOOGLE_API_KEY   — used for both planning and conversation LLM
  DEEPGRAM_API_KEY — STT
  CARTESIA_API_KEY — TTS

Run with:
    uv run dynamic_teacher.py
"""

import asyncio
import os
from typing import Callable

from dotenv import load_dotenv
from loguru import logger

print("Starting Dynamic IB MYP Oral Examiner bot...")
print("Loading models and imports (first run may take ~20 seconds)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
logger.info("Local Smart Turn Analyzer V3 loaded")

logger.info("Loading Silero VAD...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
logger.info("Silero VAD loaded")

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

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService

from pipecat_flows import FlowManager, FlowsFunctionSchema, FlowArgs, NodeConfig

from flow_spec_schema import FlowSpec, ProbeNode
from flow_planner import generate_flow

logger.info("All components loaded successfully!")

load_dotenv(override=True)

# ─── Module-level state (populated at startup, before first connection) ────────
_FLOW_SPEC: FlowSpec = None
_PROBE_FACTORIES: dict[str, Callable] = {}
_REQUIRED_PROBES: set[str] = set()

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
    "4. Do NOT accept single-word answers — always ask the student to elaborate. "
    "5. NEVER repeat the same follow-up phrase twice. Vary your language. "
    "6. INTENT DETECTION: distinguish deliberate evasion from genuine confusion. "
    "   Evasion = unrelated jokes/tangents. Confusion = asking to clarify the question. "
    "   Treat confusion generously — help the student, do not strike. "
    "7. FUNCTION RESULT RULE: after any function call, read the NEW task instructions "
    "   that follow and speak ONLY natural conversational sentences. "
    "   If the result has next_action='wait_for_student', stop and wait silently. "
    "   Do NOT call another function until the student speaks again. "
    "8. NEVER output raw JSON, status codes, function names, curly braces, or any "
    "   technical content. Speak only natural English sentences. "
    "9. Call any strike/transition function at most ONCE per student message. "
    "10. Voice conversation only — no markdown, no bullet points, no emojis. "
    "11. Keep your turns short — you are facilitating, not teaching. "
    "12. NEVER invent a topic. Ask ONLY the exact exam question given in the task instructions."
)
ROLE_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]

# ─── Shared transition function schema (used by all probe nodes) ───────────────
ADVANCE_PROBE_SCHEMA = {
    "depth": {
        "type": "string",
        "enum": ["surface", "developed"],
        "description": "Depth of the student's response",
    },
    "topics_raised": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Themes the student mentioned in this response",
    },
    "topics_missing": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Important themes from the exam NOT yet discussed",
    },
    "recommended_next": {
        "type": "string",
        "description": "Probe ID from the available list you recommend as the best next step",
    },
    "gold_moment": {
        "type": "string",
        "description": (
            "A specific, vivid idea the student said that is worth following up on. "
            "Empty string if nothing stands out."
        ),
    },
    "student_struggling": {
        "type": "boolean",
        "description": "True if the student needed significant scaffolding",
    },
    "vague_terms": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Imprecise words or phrases the student used",
    },
}


# ─── Off-topic instruction (per strike level) ──────────────────────────────────

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
    elif count == 1:
        return (
            f"OFF-TOPIC TRACKER [strikes: 1]: if the student's most recent message "
            f"is off-topic, say firmly: '{name}, I need you to stay on topic. "
            f"Please answer the question I asked.' Then call off_topic_strike. {guard}"
        )
    elif count == 2:
        return (
            f"OFF-TOPIC TRACKER [strikes: 2]: if the student's most recent message "
            f"is off-topic, issue a FINAL WARNING: '{name}, this is your final warning. "
            f"If you go off-topic again, I will have to cancel this exam.' "
            f"Then call off_topic_strike. {guard}"
        )
    else:
        return (
            f"OFF-TOPIC TRACKER [strikes: {count}]: if the student's most recent message "
            f"is off-topic, say: '{name}, the exam is now being cancelled due to repeated "
            f"off-topic behaviour.' Then call off_topic_strike. {guard}"
        )


# ─── Global off-topic handler (re-enters current probe) ───────────────────────

async def handle_global_off_topic(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    count = flow_manager.state.get("off_topic_count", 0) + 1
    flow_manager.state.update({"off_topic_count": count})
    logger.info(f"Off-topic strike #{count}")

    logger.info(f"Off-topic strike #{count}")

    if count >= 4:
        return ({}, create_cancel_exam_node(flow_manager))

    # Re-enter the current probe
    current = flow_manager.state.get("current_probe", "opening")
    if current == "opening":
        return ({}, create_opening_node(flow_manager))
    if current in _PROBE_FACTORIES:
        return ({}, _PROBE_FACTORIES[current](flow_manager))
    return ({}, create_closing_node(flow_manager))


# ─── Cancel exam node ──────────────────────────────────────────────────────────

def create_cancel_exam_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    return NodeConfig(
        name="cancelled",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Inform {name} in one firm, calm sentence that the exam is being cancelled "
                "due to repeated off-topic behaviour. Do not apologise. Do not ask any questions."
            ),
        }],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )


# ─── Node 1: Greeting ──────────────────────────────────────────────────────────

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
                "Explain you will have a short discussion and that there are no right or wrong "
                "answers — you want to hear their thoughts and opinions. "
                "Ask for their name.\n\n"
                f"GREETING STRIKES SO FAR: {count}.\n\n"
                "After EVERY student message, call assess_name_response ONCE with:\n"
                "  intent='gave_name'           — student gave a real first name\n"
                "  intent='needs_clarification' — confused or asking about the question\n"
                "  intent='evasion'             — unrelated tangent or deliberate dodge\n\n"
                f"For 'evasion': {evasion_tone} Then call assess_name_response.\n"
                "For 'needs_clarification': respond helpfully, then call assess_name_response.\n\n"
                "CRITICAL: call assess_name_response at most ONCE per student message. "
                "If the result has next_action='wait_for_student', stop and wait. "
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
                        "description": (
                            "gave_name: real first name provided. "
                            "needs_clarification: trying to engage but confused. "
                            "evasion: deliberate off-topic."
                        ),
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


# ─── Node 2: Opening Question ──────────────────────────────────────────────────

def create_opening_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    count = flow_manager.state.get("off_topic_count", 0)
    if not _FLOW_SPEC:
        logger.error("_FLOW_SPEC is None when building opening node — flow was not initialised")
    exam_question = _FLOW_SPEC.exam_question if _FLOW_SPEC else "What do you think about this topic?"
    available = [p.id for p in _FLOW_SPEC.probe_pool] if _FLOW_SPEC else []

    return NodeConfig(
        name="opening",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"Ask {name} the following exam question, word for word:\n"
                f"'{exam_question}'\n\n"
                "STEP 1 — CLARIFY STANCE.\n"
                "If the student gives only facts or no clear position, redirect: "
                "'That is good background — but what is YOUR view? "
                "Do you think it is mainly positive, negative, or complicated?' "
                "Do not move on until they have a position.\n\n"
                "STEP 2 — BUILD DEPTH.\n"
                "If the answer lacks reasoning, encourage with varied phrases: "
                "'Why do you feel that way?' / 'What makes you say that?' / "
                "'Can you say a bit more about that?' "
                "If after TWO attempts they still cannot elaborate, offer a scaffold: "
                "'Think about who this affects most — does that help?'\n\n"
                "STEP 3 — CALL FUNCTION.\n"
                "Only call advance_from_probe once the student has a clear position "
                "AND at least one reason. "
                f"Available probe IDs for recommended_next: {available}. "
                "If the result has next_action='wait_for_student', stop and wait."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_probe",
                handler=handle_transition,
                description=(
                    "Call once student has a clear position and at least one reason. "
                    "Fill all fields based on the full conversation so far."
                ),
                properties=ADVANCE_PROBE_SCHEMA,
                required=list(ADVANCE_PROBE_SCHEMA.keys()),
            ),
        ],
    )


# ─── Dynamic probe node (built from FlowSpec at runtime) ──────────────────────

def create_dynamic_probe_node(
    flow_manager: FlowManager,
    probe: ProbeNode,
) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    count = flow_manager.state.get("off_topic_count", 0)
    covered = set(flow_manager.state.get("covered_probes", []))
    available = [p.id for p in _FLOW_SPEC.probe_pool if p.id not in covered]

    scaffold_line = (
        f"SCAFFOLD (use if {name} is completely stuck): {probe.scaffold}\n\n"
        if probe.scaffold else ""
    )

    return NodeConfig(
        name=probe.id,
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"CURRENT PROBE: {probe.id} (type: {probe.type})\n\n"
                f"TASK: {probe.prompt}\n\n"
                f"{scaffold_line}"
                f"AVAILABLE NEXT PROBES: {available}\n\n"
                f"After {name} responds substantively, call advance_from_probe with:\n"
                "  depth: 'surface' or 'developed'\n"
                "  topics_raised: themes mentioned in this response\n"
                "  topics_missing: important exam themes not yet discussed\n"
                "  recommended_next: best next probe ID from the available list above\n"
                "  gold_moment: specific insight worth following up (empty string if none)\n"
                "  student_struggling: true if student needed heavy scaffolding\n"
                "  vague_terms: imprecise words or phrases student used\n\n"
                "If the result has next_action='wait_for_student', stop and wait."
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


# ─── Follow-up probe (dynamically built from gold_moment) ─────────────────────

def create_followup_probe_node(
    flow_manager: FlowManager,
    gold_moment: str,
) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    count = flow_manager.state.get("off_topic_count", 0)
    covered = set(flow_manager.state.get("covered_probes", []))
    available = [p.id for p in _FLOW_SPEC.probe_pool if p.id not in covered]

    return NodeConfig(
        name="followup_probe",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"{_off_topic_instruction(count, name)}\n\n"
                f"{name} just said something worth following up on:\n"
                f"'{gold_moment}'\n\n"
                f"Ask {name} ONE question to dig deeper into that specific idea. "
                "Be genuinely curious — not leading. Examples:\n"
                "  'Can you say more about that?'\n"
                "  'What makes you think that?'\n"
                "  'Can you give me a concrete example of what you mean?'\n\n"
                f"AVAILABLE NEXT PROBES AFTER THIS: {available}\n\n"
                "Once they have elaborated, call advance_from_probe. "
                "If result has next_action='wait_for_student', stop and wait."
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


# ─── Closing node ──────────────────────────────────────────────────────────────

def create_closing_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name", "student")
    return NodeConfig(
        name="closing",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Thank {name} warmly in one or two short sentences for their thoughtful "
                "responses. Let them know the oral discussion is now complete. "
                "Do NOT give feedback on language, performance, or scores. "
                "Do NOT ask any further questions."
            ),
        }],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )


# ─── Smart transition handler ──────────────────────────────────────────────────

async def handle_transition(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[dict, NodeConfig]:
    # 1. Mark current probe as covered
    current = flow_manager.state.get("current_probe", "opening")
    covered = set(flow_manager.state.get("covered_probes", []))
    covered.add(current)

    # 2. Hard node cap
    node_count = flow_manager.state.get("node_count", 0) + 1

    # 3. Accumulate vague terms
    all_terms = list(set(
        flow_manager.state.get("vague_terms", []) + args.get("vague_terms", [])
    ))

    flow_manager.state.update({
        "covered_probes": list(covered),
        "node_count": node_count,
        "vague_terms": all_terms,
    })

    logger.info(
        f"Transition from '{current}' | covered={covered} | "
        f"node_count={node_count} | gold='{args.get('gold_moment', '')}'"
    )

    if node_count > 12:
        logger.warning("Hard node cap reached — forcing closing")
        return ({}, create_closing_node(flow_manager))

    # 4. Follow a gold moment (max once per session)
    gold = args.get("gold_moment", "").strip()
    gold_followed = flow_manager.state.get("gold_followed", 0)
    if gold and gold_followed < 1:
        flow_manager.state.update({"gold_followed": gold_followed + 1, "current_probe": "followup_probe"})
        logger.info(f"Following gold moment: {gold!r}")
        return ({}, create_followup_probe_node(flow_manager, gold))

    # 5. Required probes first
    missing_required = _REQUIRED_PROBES - covered
    if missing_required:
        next_id = sorted(missing_required)[0]  # deterministic ordering
        logger.info(f"Required probe not yet covered: {next_id!r}")
        flow_manager.state.update({"current_probe": next_id})
        return ({}, _PROBE_FACTORIES[next_id](flow_manager))

    # 6. Use LLM recommendation if valid and uncovered
    recommended = args.get("recommended_next", "").strip()
    if recommended and recommended not in covered and recommended in _PROBE_FACTORIES:
        logger.info(f"Using LLM recommendation: {recommended!r}")
        flow_manager.state.update({"current_probe": recommended})
        return ({}, _PROBE_FACTORIES[recommended](flow_manager))

    # 7. Pick first uncovered probe from pool
    remaining = [p.id for p in _FLOW_SPEC.probe_pool if p.id not in covered]
    if not remaining:
        logger.info("All probes covered — closing")
        return ({}, create_closing_node(flow_manager))

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
        logger.info(f"Name recorded: {name_given!r}")
        flow_manager.state.update({
            "name": name_given,
            "off_topic_count": 0,
            "current_probe": "opening",
        })
        return ({}, create_opening_node(flow_manager))

    if intent == "needs_clarification":
        logger.info("Greeting: clarification needed, no strike")
        return ({}, create_greeting_node(flow_manager))

    # evasion
    count = flow_manager.state.get("greeting_strikes", 0) + 1
    flow_manager.state.update({"greeting_strikes": count})
    logger.info(f"Greeting evasion strike #{count}")
    if count >= 3:
        return ({}, create_cancel_exam_node(flow_manager))
    return ({}, create_greeting_node(flow_manager))


# ─── Probe factory builder ─────────────────────────────────────────────────────

def build_probe_factories(flow_spec: FlowSpec) -> dict[str, Callable]:
    """Create a factory function for each probe in the FlowSpec."""
    return {
        probe.id: (lambda fm, p=probe: create_dynamic_probe_node(fm, p))
        for probe in flow_spec.probe_pool
    }


# ─── Bot setup ─────────────────────────────────────────────────────────────────

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting dynamic oral examiner bot")

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        language="en-IN",
        model="nova-3",
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash-lite",
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)
    rtvi = RTVIProcessor()

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Student connected")
        await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Student disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main entry point. Generates the FlowSpec before starting the pipeline."""
    global _FLOW_SPEC, _PROBE_FACTORIES, _REQUIRED_PROBES

    topic = os.getenv(
        "EXAM_TOPIC",
        "Should gene editing in humans be allowed?",
    )
    duration = int(os.getenv("EXAM_DURATION_MINUTES", "6"))

    logger.info(f"Generating flow for: {topic!r}")
    _FLOW_SPEC = await generate_flow(topic, duration)
    _PROBE_FACTORIES = build_probe_factories(_FLOW_SPEC)
    _REQUIRED_PROBES = set(_FLOW_SPEC.required_probes)

    logger.info(f"Flow ready — {len(_PROBE_FACTORIES)} probes, required: {_REQUIRED_PROBES}")
    logger.info(f"Exam question: {_FLOW_SPEC.exam_question}")

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
