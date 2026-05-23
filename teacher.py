#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""IB MYP Cultural Identity & Globalization Oral Exam Bot.

A voice AI oral examiner for an IB MYP language class (ages 11–16) conducting
a structured 4–6 minute discussion on Cultural Identity & Globalization.

Exam topic:
  "Many people say globalization is making cultures around the world more
  similar. Do you think this is a positive or negative development?"

Flow (6 nodes):
  1. Greeting       — introduce exam, collect student's name
  2. Opening        — pose the question, assess depth of first response
  3. Example Probe  — "Can you give me a specific example from your own life?"
  4. Counter Probe  — "Some people would disagree — what might they say?"
  5. Precision Probe — challenge a vague term the student used
  6. Closing        — thank the student, end conversation

Required AI services:
  - Deepgram (Speech-to-Text)
  - Google Gemini 2.5 Flash Lite (LLM)
  - Cartesia (Text-to-Speech)

Run with:
    uv run teacher.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

print("Starting IB MYP Oral Examiner bot...")
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

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService

from pipecat_flows import FlowManager, FlowsFunctionSchema, FlowArgs, NodeConfig

logger.info("All components loaded successfully!")

load_dotenv(override=True)

# ─── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a friendly but structured oral examiner for an IB MYP language class "
    "(students aged 11 to 16). Your role is to facilitate a natural 4 to 6 minute "
    "discussion on Cultural Identity and Globalization. "
    "You MUST ALWAYS call the provided function when directed to advance the conversation. "
    "STRICT RULES — never break these: "
    "1. Do NOT correct grammar, pronunciation, or word choice mid-conversation. "
    "2. Do NOT share your own opinion on whether globalization is positive or negative. "
    "3. Do NOT lecture, give long explanations, or fill silence with information. "
    "4. Do NOT accept a single-word or one-phrase answer — always ask the student to "
    "   elaborate before calling any transition function. "
    "5. NEVER repeat the same follow-up question or phrase twice in a row. "
    "   Vary your language every time you prompt for more. "
    "6. OFF-TOPIC ESCALATION — follow this strictly: "
    "   First time the student goes off-topic: redirect once, briefly and warmly. "
    "   Second time and every time after that: drop ALL warmth and pleasantries immediately. "
    "   Do NOT comment on, engage with, or compliment their off-topic content in any way. "
    "   Say firmly and directly, with clear displeasure in your tone: "
    "   '[Name], I need you to stay focused. Please answer the question I asked.' "
    "   or: '[Name], we are not here to discuss that. Answer the question, please.' "
    "   Do NOT move forward or call any function until they answer the actual question. "
    "   Repeat the exact question if necessary. "
    "7. This is a voice conversation — speak naturally. "
    "   No bullet points, no markdown, no emojis. "
    "8. Keep your own turns short — you are facilitating, not teaching."
)
ROLE_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]

EXAM_QUESTION = (
    "Many people say globalization is making cultures around the world more similar. "
    "Do you think this is a positive or negative development?"
)


# ─── Node 1: Greeting ──────────────────────────────────────────────────────────

def create_greeting_node() -> NodeConfig:
    return NodeConfig(
        name="greeting",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                "Greet the student warmly. Introduce yourself as their IB MYP oral examiner. "
                "Briefly explain that you will have a short discussion today about culture and "
                "globalization and that there are no right or wrong answers — you are simply "
                "interested in their thoughts and opinions. "
                "Ask for their name. "
                "IMPORTANT: You are ONLY collecting the student's name here. Nothing else. "
                "If the student says anything other than their name — jokes, questions, "
                "tangents, silly answers — redirect ONCE warmly: 'I just need your name to get "
                "started.' If they persist, say firmly: '[Student], please just tell me your "
                "name. We cannot begin until you do.' "
                "Do NOT accept a non-name word like 'pickle' or 'unicorn' as a name. "
                "Only call record_name when the student gives a genuine first name."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="record_name",
                handler=handle_name,
                description="Call this as soon as the student provides their name.",
                properties={
                    "name": {
                        "type": "string",
                        "description": "The student's first name or preferred name",
                    },
                },
                required=["name"],
            )
        ],
    )


# ─── Node 2: Opening Question ──────────────────────────────────────────────────

def create_opening_question_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    return NodeConfig(
        name="opening_question",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Ask {name} the following exam question, word for word: "
                f"'{EXAM_QUESTION}' "
                "Give the student time to respond. Then follow these steps carefully:\n\n"
                "STEP 1 — CHECK POSITION CLARITY.\n"
                "If the student's position is unclear or contradictory "
                "(e.g. they switch between positive and negative, use confusing vocabulary, "
                "or say something you cannot interpret), do NOT accept it yet. "
                "Gently clarify first with a variation of: "
                "'It sounds like you see more than one side — which way do you lean, "
                "positive or negative?' "
                "or: 'Just to make sure I understand — do you think it is mainly good, "
                "mainly bad, or a bit of both?' "
                "Do not move on until you have a clear position.\n\n"
                "STEP 2 — BUILD DEPTH.\n"
                "Once the position is clear, if the answer lacks reasoning (one sentence, "
                "no explanation), encourage elaboration. Use DIFFERENT phrasings each time — "
                "never repeat the same question twice. Rotate through options like: "
                "'Why do you feel that way?' / "
                "'What makes you say that?' / "
                "'Can you say a bit more about that idea?' / "
                "'What comes to mind when you think about it?' "
                "If after TWO different follow-up attempts the student still cannot elaborate, "
                "provide a gentle scaffold: 'Think about things like language, traditions, "
                "clothing, or food — does any of those connect to what you are thinking?' "
                "or offer a sentence starter: 'Maybe you could start with: "
                "I think it is positive or negative because ...'\n\n"
                "STEP 3 — CALL FUNCTION.\n"
                "Only call advance_from_opening once: "
                "(a) the student has a clear position AND "
                "(b) they have given at least one reason or idea to support it. "
                "Set student_depth to: "
                "  'surface'   — position clear but reasoning is thin (one bare idea); "
                "  'developed' — clear opinion with at least one explained reason or example. "
                "Set vague_term to any imprecise word or phrase worth probing for precision "
                "(e.g. 'lost', 'better', 'fades', 'bad'). Empty string if none stands out. "
                "Set student_position to 'positive', 'negative', or 'mixed'."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_opening",
                handler=handle_advance_from_opening,
                description=(
                    "Call this only once the student has a clear position AND at least one "
                    "supporting idea. student_depth: 'surface' or 'developed'. "
                    "student_position: 'positive', 'negative', or 'mixed'."
                ),
                properties={
                    "student_depth": {
                        "type": "string",
                        "enum": ["surface", "developed"],
                        "description": "Depth of the student's opening response",
                    },
                    "student_position": {
                        "type": "string",
                        "enum": ["positive", "negative", "mixed"],
                        "description": "The position the student ultimately settled on",
                    },
                    "vague_term": {
                        "type": "string",
                        "description": (
                            "A vague word or phrase the student used that could be probed later "
                            "(e.g. 'lost', 'better', 'fades'). Empty string if none."
                        ),
                    },
                },
                required=["student_depth", "student_position", "vague_term"],
            )
        ],
    )


# ─── Node 3: Example Probe ─────────────────────────────────────────────────────

def create_example_probe_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    return NodeConfig(
        name="example_probe",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Acknowledge {name}'s point briefly (one short sentence), then ask for a "
                "specific example using a variation of: "
                "'Can you give me a specific example from your own life or community?' "
                "If the student responds vaguely (for example: 'like food or music'), "
                "gently push: 'Can you be a bit more specific — what exactly comes to mind?' "
                "Once the student has given a concrete, specific example with at least a little "
                "detail, call advance_from_example immediately."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_example",
                handler=handle_advance_from_example,
                description="Call this once the student has provided a specific, concrete example.",
                properties={},
                required=[],
            )
        ],
    )


# ─── Node 4: Counter-Argument Probe ───────────────────────────────────────────

def create_counter_probe_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    return NodeConfig(
        name="counter_probe",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Acknowledge {name}'s point in one short sentence, then introduce the opposing "
                "perspective using a natural variation of: "
                "'Some people would disagree with you — what do you think they might say?' "
                "If the student says they cannot think of anything, scaffold gently with: "
                "'Think about someone who loves experiencing food, music, or films from other "
                "cultures — how might they see globalization differently?' "
                "If the student's response contains a specific, insightful idea "
                "(for example: naming something concrete like 'music blending' or 'grassroot "
                "culture'), do NOT transition immediately — ask ONE follow-up to deepen it, "
                "such as: 'That is a really interesting point — can you give me an example of "
                "that?' or 'What do you mean by that exactly?' "
                "After that one follow-up, call advance_from_counter."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_from_counter",
                handler=handle_advance_from_counter,
                description="Call this once the student has attempted to state a counter-argument.",
                properties={},
                required=[],
            )
        ],
    )


# ─── Node 5: Precision Probe ───────────────────────────────────────────────────

def create_precision_probe_node(flow_manager: FlowManager) -> NodeConfig:
    name = flow_manager.state.get("name")
    vague_term = flow_manager.state.get("vague_term", "")

    if vague_term:
        probe_instruction = (
            f"Earlier, {name} used the word or phrase '{vague_term}'. "
            f"Ask them to explain more precisely what they meant, for example: "
            f"'You mentioned {vague_term} — what exactly do you mean by that in this context?' "
        )
    else:
        probe_instruction = (
            f"Ask {name} to define or explain what they mean by 'globalization' or a related "
            "concept they discussed. Use a variation of: "
            "'When you talk about globalization, what exactly do you mean? "
            "What does it actually look like in people's everyday lives?' "
        )

    return NodeConfig(
        name="precision_probe",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                probe_instruction +
                "This is about precision of language and ideas — be encouraging and curious, "
                "not critical. "
                "If the student's answer introduces a new, specific concept or vivid image "
                "(for example: 'grassroot culture fades away', 'our language disappears', "
                "'traditions are replaced'), treat it as a gold moment — ask ONE follow-up "
                "to explore it before closing, such as: "
                "'That is a really powerful idea — can you give me an example of a grassroot "
                "tradition you think is at risk?' "
                "or: 'What kind of culture do you mean? Can you paint a picture of it?' "
                "Only call advance_to_closing after that follow-up exchange."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="advance_to_closing",
                handler=handle_advance_to_closing,
                description="Call this once the student has given a more precise explanation.",
                properties={},
                required=[],
            )
        ],
    )


# ─── Node 6: Closing ───────────────────────────────────────────────────────────

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
                "Do NOT give any feedback on their language, performance, or scores. "
                "Do NOT ask any further questions."
            ),
        }],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )


# ─── Handlers ──────────────────────────────────────────────────────────────────

async def handle_name(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    name = args["name"]
    logger.info(f"Student name recorded: {name}")
    flow_manager.state.update({"name": name})
    return (None, create_opening_question_node(flow_manager))


async def handle_advance_from_opening(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[None, NodeConfig]:
    depth = args["student_depth"]
    position = args.get("student_position", "mixed")
    vague_term = args.get("vague_term", "")
    logger.info(f"Opening depth: {depth!r} | position: {position!r} | vague term: {vague_term!r}")
    flow_manager.state.update({"opening_depth": depth, "student_position": position, "vague_term": vague_term})
    if depth == "surface":
        # Ask for an example before moving to the counter-argument
        return (None, create_example_probe_node(flow_manager))
    else:
        # Response was already developed — skip to counter-argument
        return (None, create_counter_probe_node(flow_manager))


async def handle_advance_from_example(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[None, NodeConfig]:
    logger.info("Example probe complete")
    return (None, create_counter_probe_node(flow_manager))


async def handle_advance_from_counter(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[None, NodeConfig]:
    logger.info("Counter-argument probe complete")
    return (None, create_precision_probe_node(flow_manager))


async def handle_advance_to_closing(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[None, NodeConfig]:
    logger.info("Precision probe complete, moving to closing")
    return (None, create_closing_node(flow_manager))


# ─── Bot Setup ─────────────────────────────────────────────────────────────────

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting IB MYP oral examiner bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

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
    """Main bot entry point."""
    transport_params = {
        "daily": lambda: DailyParams(
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
