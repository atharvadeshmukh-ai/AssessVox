#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Flows Teacher Agent.

A voice AI teacher bot that asks students a math question based on their age.
- Under 18: asks "What is 2 + 2?"
- 18 and over: asks "What is 9 + 9?"

Required AI services:
- Deepgram (Speech-to-Text)
- Google Gemini 2.5 Flash Lite (LLM)
- Cartesia (Text-to-Speech)

Run the bot using:

    uv run bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
logger.info("✅ Local Smart Turn Analyzer V3 loaded")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
logger.info("✅ Silero VAD model loaded")

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

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# ─── System Prompt ────────────────────────────────────────────────────────────
# Tells the LLM its persona. The critical line is the second sentence:
# "You MUST ALWAYS call the provided function" — this prevents the LLM from
# chatting freely instead of advancing the flow.
SYSTEM_PROMPT = (
    "You are a friendly but firm teacher speaking to a student. "
    "You MUST ALWAYS call the provided function to move the conversation forward. "
    "You MUST stay strictly on the current task. "
    "If the student refuses to answer, goes off-topic, or tries to change the subject, "
    "do NOT follow them. Politely but firmly redirect them back to the question you asked. "
    "Never skip a step or change the topic. "
    "This is a voice conversation — keep responses short, warm, and natural. "
    "Do not use emojis or markdown."
)
ROLE_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]


# ─── Node Definitions ─────────────────────────────────────────────────────────

def create_greeting_node() -> NodeConfig:
    """Node 1 — Collect student's name only."""
    return NodeConfig(
        name="greeting",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                "Greet the student warmly and introduce yourself as their teacher. "
                "Ask only for their name. "
                "Once the student tells you their name, call record_name immediately."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="record_name",
                handler=handle_name,
                description="Call this as soon as the student says their name.",
                properties={
                    "name": {"type": "string", "description": "The student's name"},
                },
                required=["name"],
            )
        ],
    )


def create_ask_age_node(flow_manager: FlowManager) -> NodeConfig:
    """Node 2 — Collect student's age only."""
    name = flow_manager.state.get("name")
    return NodeConfig(
        name="ask_age",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Ask {name} how old they are. "
                "If they refuse, question why, or change the subject, "
                "calmly explain that you need their age to continue the lesson and ask again. "
                "Do NOT proceed or change topic until they give you a number. "
                "Once they provide their age, call record_age immediately."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="record_age",
                handler=handle_age,
                description="Call this as soon as the student says their age.",
                properties={
                    "age": {"type": "integer", "description": "The student's age in years"},
                },
                required=["age"],
            )
        ],
    )


def create_question_node(flow_manager: FlowManager) -> NodeConfig:
    """Node 3 — Ask the age-appropriate math question."""
    name = flow_manager.state.get("name")
    age = flow_manager.state.get("age")

    if age < 18:
        question = "What is 2 plus 2?"
        fn_name = "record_answer_child"
        handler = handle_answer_child
        description = "Call this as soon as the student answers 2 + 2."
    else:
        question = "What is 9 plus 9?"
        fn_name = "record_answer_adult"
        handler = handle_answer_adult
        description = "Call this as soon as the student answers 9 + 9."

    return NodeConfig(
        name="question",
        role_messages=ROLE_MESSAGES,
        task_messages=[{
            "role": "system",
            "content": (
                f"Ask {name} this math question: '{question}' "
                f"Once the student gives their answer, call {fn_name} immediately."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name=fn_name,
                handler=handler,
                description=description,
                properties={
                    "answer": {"type": "number", "description": "The student's numeric answer"},
                },
                required=["answer"],
            )
        ],
    )


def create_farewell_node(flow_manager: FlowManager) -> NodeConfig:
    """Node 4 — Praise or correct the student, then end the conversation.
    post_actions end_conversation lets the bot finish speaking before shutting down.
    """
    name = flow_manager.state.get("name")
    is_correct = flow_manager.state.get("is_correct")

    if is_correct:
        task = (
            f"Praise {name} warmly — their answer was absolutely correct. "
            "Say goodbye in one short sentence. Do not ask any more questions."
        )
    else:
        correct = flow_manager.state.get("correct_answer")
        task = (
            f"Tell {name} their answer was wrong and that the correct answer is {correct}. "
            "Encourage them to keep practicing and say goodbye. "
            "Do not ask any more questions."
        )

    return NodeConfig(
        name="farewell",
        role_messages=ROLE_MESSAGES,
        task_messages=[{"role": "system", "content": task}],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )


# ─── Handlers ─────────────────────────────────────────────────────────────────

async def handle_name(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    name = args["name"]
    logger.info(f"Name recorded: {name}")
    flow_manager.state.update({"name": name})
    return (None, create_ask_age_node(flow_manager))


async def handle_age(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    age = int(args["age"])
    logger.info(f"Age recorded: {age}")
    flow_manager.state.update({"age": age})
    return (None, create_question_node(flow_manager))


async def handle_answer_child(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    answer = args["answer"]
    correct = 4
    is_correct = (answer == correct)
    logger.info(f"Answer to 2+2: {answer} — {'correct' if is_correct else 'wrong'}")
    flow_manager.state.update({"answer": answer, "is_correct": is_correct, "correct_answer": correct})
    return (None, create_farewell_node(flow_manager))


async def handle_answer_adult(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    answer = args["answer"]
    correct = 18
    is_correct = (answer == correct)
    logger.info(f"Answer to 9+9: {answer} — {'correct' if is_correct else 'wrong'}")
    flow_manager.state.update({"answer": answer, "is_correct": is_correct, "correct_answer": correct})
    return (None, create_farewell_node(flow_manager))


# ─── Bot Setup ────────────────────────────────────────────────────────────────

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

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
        logger.info("Client connected")
        await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
