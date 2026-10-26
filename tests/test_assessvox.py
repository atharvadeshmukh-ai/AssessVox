"""Regression tests for AssessVox.

No network or API calls — everything here runs offline. Importing
dynamic_exam loads the local turn/VAD models, so the first run takes ~15s.

Run:  python -m uv run pytest
"""

import asyncio
import json
import re
from pathlib import Path

import pytest

import dynamic_exam as de
import pricing
import session_timeline
from flow_validator import validate_flow_spec
from template_loader import _load_all, get_template, list_templates

ROOT = Path(__file__).resolve().parent.parent
_load_all()


# ─── Helpers ──────────────────────────────────────────────────────────────────


class FakeTask:
    """Stands in for flow_manager.task — records what would be spoken."""

    def __init__(self):
        self.spoken: list[str] = []

    async def queue_frame(self, frame):
        self.spoken.append(getattr(frame, "text", type(frame).__name__))


class FakeFlowManager:
    def __init__(self, spec, **state):
        self.state = {
            "flow_spec": spec,
            "name": "Asha",
            "student_utterance_count": 1,
            "probe_factories": de.build_probe_factories(spec),
            "required_probes": set(spec.required_probes),
            **state,
        }
        self.task = FakeTask()


def spec_for(template_id, question_id=None):
    t = get_template(template_id)
    q = t.get_question(question_id) if question_id else None
    return t.build_flow_spec(question_override=q["prompt"] if q else None, question=q)


ALL_TEMPLATES = [t.id for t in list_templates()]


# ─── Templates and questions ──────────────────────────────────────────────────


@pytest.mark.parametrize("template_id", ALL_TEMPLATES)
def test_every_question_builds_a_valid_flow(template_id):
    t = get_template(template_id)
    validate_flow_spec(t.build_flow_spec())
    for q in t.questions:
        validate_flow_spec(t.build_flow_spec(question_override=q["prompt"], question=q))


@pytest.mark.parametrize("template_id", ALL_TEMPLATES)
def test_grading_rubric_formats(template_id):
    rubric = get_template(template_id).grading_rubric
    if rubric:
        rubric.format(exam_question="q", topic="t", metadata="{}", transcript="x",
                      node_trace="n", dimension_guide=de.DIMENSION_GUIDE)


# Official TOK exhibition IA prompts used by the template (IB TOK guide list).
TOK_OFFICIAL = {
    5: "What counts as good evidence for a claim?",
    6: "How does the way that we organise or classify knowledge affect what we know?",
    9: "Are some types of knowledge less open to interpretation than others?",
    11: "Can new knowledge change established values or beliefs?",
    12: "Is bias inevitable in the production of knowledge?",
    16: "Should some knowledge not be sought on ethical grounds?",
    17: "Why do we seek knowledge?",
    19: "What counts as a good justification for a claim?",
    22: "What role do experts play in influencing our consumption or acquisition of knowledge?",
    29: "Who owns knowledge?",
    30: "What role does imagination play in producing knowledge about the world?",
    34: "In what ways do our values affect our acquisition of knowledge?",
}


def test_tok_prompts_match_official_number_and_wording():
    for q in get_template("dp-tok-exhibition-defense").questions:
        number = int(re.search(r"IA #(\d+)", q["label"]).group(1))
        quoted = re.match(r'^"([^"]*)"', q["prompt"]).group(1)
        assert TOK_OFFICIAL.get(number) == quoted, q["id"]


def test_language_b_picture_questions_are_private():
    for q in get_template("dp-language-b-sl-oral").questions:
        assert q.get("image_url"), q["id"]
        assert q.get("image_description"), q["id"]
        assert "Describe the image:" not in q["prompt"], q["id"]
        assert q["prompt"].startswith("Look at the picture on your screen"), q["id"]


def test_custom_text_drops_template_scripts():
    t = get_template("myp-language-acquisition-role-play")
    spec = t.build_flow_spec(question_override="Role-play: ordering food at a cafe.")
    assert spec.opening_script == "" and spec.examiner_brief == "" and spec.image_description == ""


# ─── Greeting ─────────────────────────────────────────────────────────────────


def test_phantom_begin_exam_greets_once_and_waits_silently():
    fm = FakeFlowManager(spec_for("myp-personal-project-interview"), student_utterance_count=0)
    _, node = asyncio.run(de.handle_begin_exam({"intent": "ready", "speak": "x"}, fm))
    assert len(fm.task.spoken) == 1 and "Are you ready to begin?" in fm.task.spoken[0]
    # Returning a node (not None) is what stops pipecat-flows re-running the LLM.
    assert node is not None and node.get("respond_immediately") is False


@pytest.mark.parametrize("programme, title", [("DP", "IB DP"), ("MYP", "IB MYP"), (None, "IB MYP")])
def test_examiner_title_follows_programme(programme, title):
    fm = FakeFlowManager(spec_for("myp-personal-project-interview"), programme=programme)
    assert de._examiner_title(fm) == title
    greeting = de.create_known_name_greeting_node(fm)["task_messages"][0]["content"]
    assert f"{title} oral examiner" in greeting


def test_dp_rulebook_has_dp_audience():
    assert "Diploma Programme" in de._role_messages_for("DP")[0]["content"]
    assert "IB MYP language class" in de._role_messages_for("MYP")[0]["content"]


def test_rulebook_is_passed_as_system_instruction_for_every_provider():
    # A context system message is dropped by pipecat on Anthropic/Gemini once
    # async tools are registered — the rulebook must go in system_instruction.
    import inspect
    src = inspect.getsource(de.run_bot)
    assert src.count("system_instruction=_role_prompt") == 5
    assert "LLMContext(messages=[])" in src


# ─── Role-play / listen-and-respond ───────────────────────────────────────────


@pytest.mark.parametrize("template_id, question_id, starts", [
    ("myp-language-acquisition-role-play", "rp-lost-bag", "This is a role-play"),
    ("myp-language-acquisition-listen-respond", "lr-teen-sleep", "Listen carefully"),
])
def test_scripted_opening_is_read_word_for_word(template_id, question_id, starts):
    fm = FakeFlowManager(spec_for(template_id, question_id))
    _, node = asyncio.run(de.handle_begin_exam({"intent": "ready", "speak": "Let's begin."}, fm))
    assert fm.task.spoken[0] == "Let's begin."
    assert fm.task.spoken[1].startswith(starts)
    opening = node["task_messages"][0]["content"]
    assert "EXAM FORMAT" in opening and "CLARIFY STANCE" not in opening


def test_discussion_question_keeps_original_opening():
    fm = FakeFlowManager(spec_for("myp-language-acquisition-oral", "fairness-voting-age"))
    _, node = asyncio.run(de.handle_begin_exam({"intent": "ready", "speak": "Let's begin."}, fm))
    assert fm.task.spoken == ["Let's begin."]
    opening = node["task_messages"][0]["content"]
    assert "CLARIFY STANCE" in opening and "EXAM FORMAT" not in opening


def test_listening_passage_is_not_shown_in_question_text():
    for q in get_template("myp-language-acquisition-listen-respond").questions:
        assert "Listen carefully" not in q["prompt"]
        assert q["opening_script"].startswith("Listen carefully")


# ─── Picture questions ────────────────────────────────────────────────────────


def test_picture_description_goes_to_examiner_but_is_never_spoken():
    q = get_template("dp-language-b-sl-oral").get_question("identities-passport-stack")
    fm = FakeFlowManager(spec_for("dp-language-b-sl-oral", q["id"]), programme="DP")
    assert "PICTURE ON THE STUDENT'S SCREEN" in de._state_block(fm)
    spoken_question = de._speak_question_line(fm)
    assert "Spanish" not in spoken_question and "boarding pass" not in spoken_question


# ─── Pricing / dashboard ──────────────────────────────────────────────────────


def test_compute_cost_uses_list_prices():
    u = pricing.SessionUsage()
    u.add_llm("conversation", "claude-haiku-5-5", prompt=50_000, completion=10_000)
    u.add_llm("grading", "gemini-3.5-flash", prompt=9000, completion=1500, reasoning=500)
    u.add_tts("sonic-turbo", 1000)
    u.add_stt("ink-2", 60)
    cost = pricing.compute_cost(u.to_dict())
    by_cat = {line["category"]: line["usd"] for line in cost["lines"]}
    assert by_cat["conversation"] == pytest.approx((50_000 * 0.10 + 10_000 * 0.50) / 1e6)
    assert by_cat["grading"] == pytest.approx((9000 * 1.50 + 2000 * 9.00) / 1e6)
    assert by_cat["tts"] == pytest.approx(1000 * pricing.CARTESIA_USD_PER_CREDIT)
    assert by_cat["stt"] == pytest.approx(60 * 3 * pricing.CARTESIA_USD_PER_CREDIT)
    assert cost["unpriced"] == []


def test_haiku_long_prompt_is_billed_at_5x():
    u = pricing.SessionUsage()
    u.add_llm("conversation", "claude-haiku-5-5", prompt=200_000, completion=1000)
    cost = pricing.compute_cost(u.to_dict())
    assert cost["total_usd"] == pytest.approx((200_000 * 0.10 + 1000 * 0.50) * 5 / 1e6)


def test_unknown_model_is_flagged_not_priced():
    u = pricing.SessionUsage()
    u.add_llm("conversation", "some-new-model", prompt=10, completion=10)
    cost = pricing.compute_cost(u.to_dict())
    assert cost["total_usd"] is None and cost["unpriced"] == ["some-new-model"]


def test_usage_rebuilt_from_log_lines(tmp_path):
    log = tmp_path / "session_x.log"
    log.write_text("\n".join([
        "2026-10-26 15:35:42.376 | DEBUG | m:f:1 - 📊 [_DumpingAnthropicLLMService#2] LLM TOKEN USAGE "
        "(claude-haiku-5-5): prompt: 3218, completion: 132, total: 3350, cache_read: 0, cache_creation: 0 at 2.78s",
        "2026-10-26 15:35:42.441 | DEBUG | m:f:1 - 📊 [FallbackCartesiaTTSService#2] TTS USAGE (sonic-turbo): 183 characters at 2.849s",
        "2026-10-26 15:36:20.524 | DEBUG | m:f:1 - 📊 [CartesiaTurnsSTTService#2] STT USAGE (ink-2): 40.019s audio at 40.933s",
        "2026-10-26 15:37:20.000 | INFO | m:f:1 - 📊 [Grader] LLM TOKEN USAGE (gemini-3.5-flash): prompt: 6260, completion: 2000, cache_read: 0, reasoning: 180",
    ]), encoding="utf-8")
    usage = pricing.usage_from_log(log)
    llm = {(r["category"], r["model"]): r for r in usage["llm"]}
    assert llm[("conversation", "claude-haiku-5-5")]["input"] == 3218
    assert llm[("grading", "gemini-3.5-flash")]["output"] == 2180
    assert usage["tts"] == [{"model": "sonic-turbo", "characters": 183}]
    assert usage["stt"][0]["seconds"] == pytest.approx(40.019)


def test_session_duration_prefers_report(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"metadata": {"duration_seconds": 98.4}}))
    assert session_timeline._session_duration(report, None, None) == 98.4
    missing = tmp_path / "none.json"
    assert session_timeline._session_duration(
        missing, "2026-10-26 15:35:38.318", "2026-10-26T15:37:34") == pytest.approx(115.682)


def test_usage_collector_tallies_metrics():
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import (LLMTokenUsage, LLMUsageMetricsData, STTUsage,
                                         STTUsageMetricsData, TTSUsageMetricsData)

    class Pushed:
        def __init__(self, frame):
            self.frame = frame

    u = pricing.SessionUsage()
    frame = MetricsFrame(data=[
        LLMUsageMetricsData(processor="l", model="claude-haiku-5-5",
                            value=LLMTokenUsage(prompt_tokens=4000, completion_tokens=100, total_tokens=4100)),
        TTSUsageMetricsData(processor="t", model="sonic-turbo", value=120),
        STTUsageMetricsData(processor="s", model="ink-2", value=STTUsage(audio_seconds=30.0)),
    ])
    asyncio.run(de.UsageCollectorObserver(u).on_push_frame(Pushed(frame)))
    d = u.to_dict()
    assert d["llm"][0]["input"] == 4000 and d["tts"][0]["characters"] == 120 and d["stt"][0]["seconds"] == 30.0


# ─── Recording of LLM calls ───────────────────────────────────────────────────


def test_llm_call_record_gets_usage_that_arrives_after_end(tmp_path):
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMTextFrame, MetricsFrame
    from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData

    out = tmp_path / "calls.jsonl"

    class Base:
        async def push_frame(self, frame, direction=None):
            pass

    class Dumper(de.LLMCallDumperMixin, Base):
        def __init__(self):
            de.LLMCallDumperMixin.__init__(self, _dump_path=out)

    def start(d, ts):
        d._cur_record = {"ts": ts, "output": {"text": "", "tool_calls": []}, "usage": None}
        d._cur_text_chunks, d._cur_tool_calls, d._cur_usage = [], [], None

    usage = MetricsFrame(data=[LLMUsageMetricsData(
        processor="x", model="m", value=LLMTokenUsage(prompt_tokens=7696, completion_tokens=204, total_tokens=7900))])

    async def run():
        d = Dumper()
        start(d, "with-usage")
        await d.push_frame(LLMTextFrame(text="Hello"))
        await d.push_frame(LLMFullResponseEndFrame())
        await d.push_frame(usage)  # pipecat's real order: usage AFTER End
        start(d, "no-usage")
        await d.push_frame(LLMFullResponseEndFrame())
        await asyncio.sleep(2.3)  # held record is flushed after 2s

    asyncio.run(run())
    records = {r["ts"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert records["with-usage"]["usage"]["prompt_tokens"] == 7696
    assert records["with-usage"]["output"]["text"] == "Hello"
    assert records["no-usage"]["usage"] is None


# ─── Per-exam log files ───────────────────────────────────────────────────────


def test_concurrent_exams_write_separate_logs(tmp_path):
    from loguru import logger

    async def exam(sid):
        path = tmp_path / f"session_{sid}.log"
        de._LOG_SESSION.set(sid)
        sink = logger.add(str(path), level="DEBUG", enqueue=True, format="{message}",
                          filter=lambda r, s=sid: de._LOG_SESSION.get() in (s, None))

        async def child():
            for i in range(3):
                logger.info(f"{sid} child {i}")
                await asyncio.sleep(0.01)

        task = asyncio.create_task(child())
        for i in range(3):
            logger.info(f"{sid} main {i}")
            await asyncio.sleep(0.01)
        await asyncio.to_thread(lambda: logger.info(f"{sid} grading"))
        await task
        return sink, path

    async def run():
        results = await asyncio.gather(exam("AAA"), exam("BBB"))
        await logger.complete()
        for sink, _ in results:
            logger.remove(sink)
        return results

    for _, path in asyncio.run(run()):
        own = path.stem[-3:]
        other = "BBB" if own == "AAA" else "AAA"
        text = path.read_text()
        assert text.count(own) == 7
        assert other not in text
