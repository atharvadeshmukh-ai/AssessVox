"""Session timeline parser for the /sessions dashboard.

Reads a session's log file, transcript snapshot, and FlowSpec, and produces a
structured timeline of turns annotated with LLM calls, node transitions,
function calls, latency, and slack events.

The transcript JSON is the source of truth for what was said. The log file
is the source of truth for *when* things happened and how much they cost.
We stitch them together by timestamp.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pricing import session_cost

LOGS_DIR = Path(__file__).parent / "logs"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
FLOWS_DIR = Path(__file__).parent / "flows"
REPORTS_DIR = Path(__file__).parent / "reports"
LLM_CALLS_DIR = Path(__file__).parent / "llm_calls"

# ─── Log regex ────────────────────────────────────────────────────────────────

_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"

# 2026-06-17 06:21:05.764 | INFO | dynamic_exam:on_latency_breakdown:2733 - [LATENCY] breakdown: CartesiaSTTService#1=304ms | AnthropicLLMService#0=1233ms | AnthropicLLMService#0=680ms | FallbackCartesiaTTSService#1=107ms
_RE_LATENCY_BREAKDOWN = re.compile(
    rf"{_TS}\s+\|\s+INFO\s+\|.*\[LATENCY\] breakdown: (?P<parts>.+)$"
)
_RE_LATENCY_V2V = re.compile(
    rf"{_TS}\s+\|\s+INFO\s+\|.*\[LATENCY\] voice-to-voice: (?P<ms>\d+) ms"
)
# 📊 [AnthropicLLMService#0] TTFB (claude-haiku-4-5): 1.2330ms at 27.85s
# Service names may have a leading underscore when wrapped by the
# LLMCallDumperMixin subclasses (e.g. "_DumpingAnthropicLLMService#0").
_RE_TTFB = re.compile(
    rf"{_TS}\s+\|\s+DEBUG\s+\|.*\[(?P<svc>_?[A-Za-z]+(?:LLM|STT|TTS)Service#\d+)\] TTFB"
    rf"(?:\s+\((?P<model>[^)]+)\))?:\s+(?P<secs>[\d.]+)s"
)
# 2026-06-17 06:21:05.000 | DEBUG | pipecat_flows.manager:_set_node:655 - Setting node: confirm_name
_RE_SET_NODE = re.compile(
    rf"{_TS}\s+\|\s+DEBUG\s+\|.*Setting node:\s+(?P<node>\S+)"
)
# pipecat_flows logs "Function called: <name>" at DEBUG level, but the same
# label appears in dynamic_exam at INFO too. Accept either.
_RE_FUNCTION_CALL = re.compile(
    rf"{_TS}\s+\|\s+(?:DEBUG|INFO)\s+\|.*Function called:\s+(?P<fn>\S+)"
)
# 03:46:46.117 | End of Turn complete due to stop_secs. Silence in ms: 3000.0
_RE_SLACK = re.compile(
    rf"{_TS}\s+\|\s+DEBUG\s+\|.*End of Turn complete due to stop_secs\. Silence in ms:\s+(?P<ms>[\d.]+)"
)
# Starting dynamic oral exam bot | session=... | llm=anthropic:claude-haiku-4-5 | stt=cartesia-ink-2
_RE_STARTUP = re.compile(
    rf"{_TS}\s+\|\s+INFO\s+\|.*Starting dynamic oral exam bot\s+\|\s+session=(?P<sid>\S+)\s+\|\s+llm=(?P<llm>\S+)\s+\|\s+stt=(?P<stt>\S+)"
)
# [LATENCY] real wait: 3200 ms | endpointing wait: 600 ms   (student's actual last word -> examiner audio)
_RE_REAL_WAIT = re.compile(
    rf"{_TS}\s+\|\s+INFO\s+\|.*\[LATENCY\] real wait: (?P<ms>\d+) ms \| endpointing wait: (?P<ep>\d+) ms"
)
# [TURN] cut-off: ...   (examiner took the turn while the student was not done)
_RE_CUTOFF = re.compile(rf"{_TS}\s+\|\s+INFO\s+\|.*\[TURN\] cut-off")
# Turn detection: mode=off | native endpointing (markers OFF) | stt=cartesia-ink-2
_RE_TURN_MODE = re.compile(
    rf"{_TS}\s+\|\s+INFO\s+\|.*Turn detection:\s+mode=(?P<mode>\S+)"
)


def _ts_to_seconds(ts: str) -> float:
    """Parse a loguru timestamp into a float (epoch seconds)."""
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").timestamp()


# ─── Dataclasses returned to the UI ───────────────────────────────────────────


@dataclass
class LLMCallEvent:
    """One LLM round-trip — either a function-routing call or a spoken-response call."""
    ts: str                # ISO timestamp the TTFB metric fired
    service: str           # e.g. "AnthropicLLMService#0"
    model: str | None      # e.g. "claude-haiku-4-5"
    ttfb_ms: int           # 1233
    role: str              # "function" | "speech" | "unknown"


@dataclass
class TurnEvent:
    """One pipeline event we want to attribute to a turn."""
    ts: str
    kind: str              # "stt" | "llm" | "tts" | "function" | "node" | "slack"
    label: str             # display label
    duration_ms: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """One bot-turn = one student utterance + one bot reply.

    Each turn has the latency breakdown bracketed by the [LATENCY] log line at
    the end. We attribute every event between the prior turn's breakdown line
    and this one to this turn.
    """
    index: int
    student_text: str | None
    bot_text: str | None
    started_at: str | None       # ts of first event in the bracket
    ended_at: str | None         # ts of the [LATENCY] line
    voice_to_voice_ms: int | None
    llm_calls: list[LLMCallEvent] = field(default_factory=list)
    stt_ms: int | None = None
    tts_ms: int | None = None
    function_calls: list[str] = field(default_factory=list)
    node_transitions: list[str] = field(default_factory=list)
    slack_events_ms: list[float] = field(default_factory=list)
    classification: str = "single-call"   # "single-call" | "two-call" | "slack-hit" | "incomplete"
    raw_calls: list[dict[str, Any]] = field(default_factory=list)  # raw LLM I/O records from llm_calls/*.jsonl
    real_wait_ms: int | None = None      # student's actual last word -> examiner audio
    endpointing_ms: int | None = None    # part of real_wait spent waiting for Ink-2's turn end


@dataclass
class SummarizationEvent:
    """One inter-turn summarization. Falls between turn `after_turn_index` and the next turn."""
    ts: str
    after_turn_index: int   # -1 if before any turn (rare); else 0..N-1
    service: str
    model: str | None
    input: dict[str, Any] | None
    output: dict[str, Any] | None
    usage: dict[str, Any] | None
    error: str | None = None


@dataclass
class SessionTimeline:
    session_id: str
    started_at: str | None
    ended_at: str | None
    student_name: str | None
    llm: str | None
    stt: str | None
    turn_marker_mode: str | None
    template_id: str | None
    exam_question: str | None
    flow_nodes: list[dict[str, Any]] = field(default_factory=list)
    flow_edges: list[dict[str, str]] = field(default_factory=list)
    visited_nodes: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    summarization_events: list[SummarizationEvent] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    has_transcript: bool = False
    has_report: bool = False
    is_complete: bool = False
    has_raw_calls: bool = False
    cost: dict[str, Any] | None = None   # see pricing.session_cost


# ─── Log scanning ─────────────────────────────────────────────────────────────


def _scan_log(log_path: Path) -> dict[str, Any]:
    """Single pass over a session log; returns a dict of indexed events."""
    out: dict[str, Any] = {
        "startup": None,
        "turn_mode": None,
        "ttfbs": [],          # (ts_secs, ts_str, svc, model, ms)
        "set_nodes": [],      # (ts_secs, ts_str, node)
        "function_calls": [], # (ts_secs, ts_str, fn)
        "v2vs": [],           # (ts_secs, ts_str, ms)
        "breakdowns": [],     # (ts_secs, ts_str, parts_str)
        "slacks": [],         # (ts_secs, ts_str, ms)
        "real_waits": [],     # (ts_secs, ts_str, ms, endpointing_ms)
        "cutoffs": [],        # (ts_secs, ts_str)
    }
    try:
        text = log_path.read_text(errors="replace")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        if m := _RE_STARTUP.search(line):
            d = m.groupdict()
            d["ts"] = m.group(1)
            out["startup"] = d
        elif m := _RE_TURN_MODE.search(line):
            out["turn_mode"] = m.group("mode")
        elif m := _RE_TTFB.search(line):
            ts = m.group(1)
            out["ttfbs"].append((
                _ts_to_seconds(ts), ts,
                m.group("svc"), m.group("model"),
                int(float(m.group("secs")) * 1000),
            ))
        elif m := _RE_SET_NODE.search(line):
            ts = m.group(1)
            out["set_nodes"].append((_ts_to_seconds(ts), ts, m.group("node")))
        elif m := _RE_FUNCTION_CALL.search(line):
            ts = m.group(1)
            out["function_calls"].append((_ts_to_seconds(ts), ts, m.group("fn")))
        elif m := _RE_LATENCY_V2V.search(line):
            ts = m.group(1)
            out["v2vs"].append((_ts_to_seconds(ts), ts, int(m.group("ms"))))
        elif m := _RE_LATENCY_BREAKDOWN.search(line):
            ts = m.group(1)
            out["breakdowns"].append((_ts_to_seconds(ts), ts, m.group("parts")))
        elif m := _RE_REAL_WAIT.search(line):
            ts = m.group(1)
            out["real_waits"].append((_ts_to_seconds(ts), ts, int(m.group("ms")), int(m.group("ep"))))
        elif m := _RE_CUTOFF.search(line):
            ts = m.group(1)
            out["cutoffs"].append((_ts_to_seconds(ts), ts))
        elif m := _RE_SLACK.search(line):
            ts = m.group(1)
            out["slacks"].append((_ts_to_seconds(ts), ts, float(m.group("ms"))))
    return out


def _parse_breakdown(parts: str) -> tuple[int | None, int | None, list[tuple[str, str | None, int]]]:
    """Parse a breakdown 'parts' string into (stt_ms, tts_ms, llm_calls).

    Format: 'CartesiaSTTService#1=304ms | AnthropicLLMService#0=1233ms | ...'
    """
    stt_ms: int | None = None
    tts_ms: int | None = None
    llms: list[tuple[str, str | None, int]] = []
    for chunk in parts.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"(?P<svc>_?[A-Za-z]+(?:LLM|STT|TTS)Service#\d+)=(?P<ms>\d+)ms", chunk)
        if not m:
            continue
        svc, ms = m.group("svc"), int(m.group("ms"))
        if "STT" in svc:
            stt_ms = ms
        elif "TTS" in svc:
            # multiple TTS entries can appear (sub-utterances) — keep the first as the headline.
            if tts_ms is None:
                tts_ms = ms
        elif "LLM" in svc:
            llms.append((svc, None, ms))
    return stt_ms, tts_ms, llms


# ─── Public API ───────────────────────────────────────────────────────────────


def list_sessions() -> list[dict[str, Any]]:
    """Return a newest-first list of session summaries for the picker."""
    sessions: list[dict[str, Any]] = []
    if not LOGS_DIR.exists():
        return sessions

    for log_path in sorted(LOGS_DIR.glob("session_*.log"), reverse=True):
        sid = log_path.stem.removeprefix("session_")
        transcript_path = TRANSCRIPTS_DIR / f"transcript_{sid}.json"
        report_path = REPORTS_DIR / f"report_{sid}.json"

        student_name = None
        turn_count = 0
        ended_at = None
        is_complete = transcript_path.exists()
        if is_complete:
            try:
                t = json.loads(transcript_path.read_text())
                student_name = t.get("student_name")
                turn_count = len(t.get("turns", []))
                ended_at = t.get("ended_at")
            except Exception:
                is_complete = False

        # Cheap header extraction — just first ~40 lines of the log.
        llm = stt = started_at = None
        try:
            with log_path.open() as f:
                for i, line in enumerate(f):
                    if i > 40:
                        break
                    if m := _RE_STARTUP.search(line):
                        started_at = m.group(1)
                        llm = m.group("llm")
                        stt = m.group("stt")
                        break
        except Exception:
            pass

        sessions.append({
            "id": sid,
            "started_at": started_at,
            "ended_at": ended_at,
            "student_name": student_name,
            "llm": llm,
            "stt": stt,
            "turn_count": turn_count,
            "is_complete": is_complete,
            "has_report": report_path.exists(),
        })
    return sessions


def build_timeline(session_id: str) -> SessionTimeline | None:
    """Parse a single session's log + transcript + flow into a SessionTimeline."""
    log_path = LOGS_DIR / f"session_{session_id}.log"
    if not log_path.exists():
        return None

    transcript_path = TRANSCRIPTS_DIR / f"transcript_{session_id}.json"
    flow_path = FLOWS_DIR / f"flow_{session_id}.json"
    report_path = REPORTS_DIR / f"report_{session_id}.json"

    transcript_data: dict[str, Any] = {}
    if transcript_path.exists():
        try:
            transcript_data = json.loads(transcript_path.read_text())
        except Exception:
            transcript_data = {}

    flow_data: dict[str, Any] = {}
    if flow_path.exists():
        try:
            flow_data = json.loads(flow_path.read_text())
        except Exception:
            pass

    scan = _scan_log(log_path)
    startup = scan["startup"] or {}

    # Load the per-call dump if present. Records have kind="conversation"
    # (default) or kind="summarization" (dedicated summarizer LLM).
    llm_calls_path = LLM_CALLS_DIR / f"calls_{session_id}.jsonl"
    raw_call_records: list[tuple[float, dict]] = []
    summarization_records: list[tuple[float, dict]] = []
    if llm_calls_path.exists():
        for raw_line in llm_calls_path.read_text(errors="replace").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                rec = json.loads(raw_line)
                ts_str = rec.get("ts")
                if not ts_str:
                    continue
                secs = datetime.fromisoformat(ts_str).timestamp()
                if rec.get("kind") == "summarization":
                    summarization_records.append((secs, rec))
                else:
                    raw_call_records.append((secs, rec))
            except Exception:
                continue

    timeline = SessionTimeline(
        session_id=session_id,
        started_at=startup.get("ts") or (scan["breakdowns"][0][1] if scan["breakdowns"] else None),
        ended_at=transcript_data.get("ended_at"),
        student_name=transcript_data.get("student_name"),
        llm=startup.get("llm"),
        stt=startup.get("stt"),
        turn_marker_mode=scan["turn_mode"],
        template_id=None,
        exam_question=flow_data.get("exam_question"),
        has_transcript=bool(transcript_data),
        has_report=report_path.exists(),
        is_complete=bool(transcript_data),
        has_raw_calls=bool(raw_call_records),
    )

    # Build flow graph: hard-coded scaffold nodes (greeting → confirm_name →
    # opening → closing) plus dynamic probes from the FlowSpec, plus any
    # node names the log saw that aren't in the scaffold.
    scaffold_nodes = ["greeting", "confirm_name", "spell_name", "opening", "closing", "cancel_exam"]
    probe_ids = [p.get("id") for p in flow_data.get("probe_pool", []) if p.get("id")]
    visited = [n for (_, _, n) in scan["set_nodes"]]
    seen_set: set[str] = set()
    nodes: list[dict[str, Any]] = []
    for n in scaffold_nodes + probe_ids + visited:
        if n and n not in seen_set:
            seen_set.add(n)
            kind = ("probe" if n in probe_ids else
                    "scaffold" if n in scaffold_nodes else "other")
            nodes.append({"id": n, "kind": kind, "visited": n in visited})

    # Edges = consecutive transitions actually seen in the log.
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for i in range(len(visited) - 1):
        a, b = visited[i], visited[i + 1]
        if (a, b) not in seen_edges:
            seen_edges.add((a, b))
            edges.append({"from": a, "to": b})

    timeline.flow_nodes = nodes
    timeline.flow_edges = edges
    timeline.visited_nodes = visited

    # Build turns. Each [LATENCY] breakdown line closes a turn; events
    # between the prior breakdown and this one belong to this turn.
    student_turns = [t for t in transcript_data.get("turns", []) if t.get("speaker") == "STUDENT"]
    bot_turns = [t for t in transcript_data.get("turns", []) if t.get("speaker") == "EXAMINER"]

    breakdowns = scan["breakdowns"]
    # voice-to-voice and breakdown are logged as two separate events that fire
    # back-to-back from the same handler, so their timestamps differ by 1–2ms.
    # Match by nearest timestamp within a tolerance rather than exact string
    # equality (which silently drops every v2v when the ms digit differs).
    v2v_events = sorted(scan["v2vs"])  # (secs, ts, ms)

    prior_secs = 0.0
    for idx, (secs, ts, parts) in enumerate(breakdowns):
        stt_ms, tts_ms, _ = _parse_breakdown(parts)
        # All LLM TTFBs falling in (prior_secs, secs] belong to this turn.
        # Exclude the 0ms startup artifact that pipecat emits when the pipeline
        # first initialises every service — these are not real LLM calls.
        bracket_llms = [
            (ts_, svc, model, ms) for (ts_secs, ts_, svc, model, ms) in scan["ttfbs"]
            if "LLM" in svc and prior_secs < ts_secs <= secs and ms > 0
        ]
        # function calls + node transitions in the same bracket
        bracket_fns = [fn for (ts_secs, _, fn) in scan["function_calls"] if prior_secs < ts_secs <= secs]
        bracket_nodes = [n for (ts_secs, _, n) in scan["set_nodes"] if prior_secs < ts_secs <= secs]
        bracket_slacks = [ms for (ts_secs, _, ms) in scan["slacks"] if prior_secs < ts_secs <= secs]

        # Classify each LLM call: a function-routing call typically precedes
        # a function-called log line; the call right after a Setting node line
        # is the spoken response.
        llm_events: list[LLMCallEvent] = []
        for i, (lts, svc, model, ms) in enumerate(bracket_llms):
            role = "speech"
            if len(bracket_llms) >= 2 and i < len(bracket_llms) - 1:
                role = "function"
            llm_events.append(LLMCallEvent(
                ts=lts, service=svc, model=model, ttfb_ms=ms, role=role,
            ))

        bracket_raw_calls = [
            rec for (rec_secs, rec) in raw_call_records
            if prior_secs < rec_secs <= secs
        ]

        # Prefer per-call dump data for transcript text — the dumper captures
        # the exact user message the LLM saw and the exact assistant text it
        # produced, with no offset issues. Fall back to transcript[idx] only
        # when no dump exists (legacy sessions).
        #
        # Heuristic: ignore "user" messages that are obviously the system
        # bootstrap (e.g. flow-manager's seed message that starts with
        # "STEP 1 —" or contains "INITIAL GREETING"). Those aren't real
        # student utterances.
        def _looks_like_bootstrap(s: str) -> bool:
            head = (s or "")[:200]
            return ("STEP 1 —" in head
                    or "INITIAL GREETING" in head
                    or head.lstrip().startswith("If there is NO user message"))

        student_text = None
        bot_text = None
        if bracket_raw_calls:
            # Use the first call's last "user"-role message for student text,
            # and the last call's output text for bot text.
            first_msgs = (bracket_raw_calls[0].get("input") or {}).get("messages") or []
            for m in reversed(first_msgs):
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if role == "user":
                    # Message content can be a str or a list of parts.
                    c = m.get("content") or m.get("parts")
                    candidate = None
                    if isinstance(c, str):
                        candidate = c
                    elif isinstance(c, list):
                        bits = []
                        for p in c:
                            if isinstance(p, dict) and p.get("text"):
                                bits.append(p["text"])
                            elif isinstance(p, str):
                                bits.append(p)
                        candidate = " ".join(bits) if bits else None
                    if candidate and not _looks_like_bootstrap(candidate):
                        student_text = candidate
                    break
            # Bot text = output text of the last call in the bracket (since
            # function-routing calls have empty output, the speech call's
            # text is what was actually spoken).
            for rec in reversed(bracket_raw_calls):
                out_text = ((rec.get("output") or {}).get("text") or "").strip()
                if out_text:
                    bot_text = out_text
                    break
        if not student_text:
            student_text = student_turns[idx - 1]["text"] if idx > 0 and (idx - 1) < len(student_turns) else None
        if not bot_text:
            bot_text = bot_turns[idx]["text"] if idx < len(bot_turns) else None

        # Pair this turn's breakdown with the closest v2v event (within 0.5s).
        # Turns are seconds/minutes apart, so a 0.5s window can only ever match
        # the v2v line emitted alongside this same breakdown. The greeting turn
        # has no preceding student speech and correctly stays None.
        v2v_ms = None
        if v2v_events:
            near_secs, _, near_ms = min(v2v_events, key=lambda e: abs(e[0] - secs))
            if abs(near_secs - secs) <= 0.5:
                v2v_ms = near_ms
        # Real wait is logged at the same moment (examiner starts speaking).
        real_wait_ms = endpointing_ms = None
        if scan["real_waits"]:
            rw_secs, _, rw_ms, rw_ep = min(scan["real_waits"], key=lambda e: abs(e[0] - secs))
            if abs(rw_secs - secs) <= 0.5:
                real_wait_ms, endpointing_ms = rw_ms, rw_ep
        # Classification rules:
        # - two-call if there were >=2 LLM TTFB events OR a function call fired
        #   (covers the Gemini case where TTFB metric is deferred and the
        #    function-routing call doesn't emit its own metric — see
        #    AssessVoxGoogleLLMService).
        # - slack-hit if SmartTurn stop_secs cap fired.
        # - otherwise single-call.
        # Turn 0 (greeting) has a node transition at startup — not a real
        # two-call routing turn. Only classify as two-call on turns where
        # the student actually spoke (student_text is set) or where ≥2 real
        # LLM calls fired. The startup node-set on Turn 0 is a false positive.
        _has_real_fn_call = len(bracket_fns) > 0 and (student_text is not None or len(llm_events) >= 2)
        if len(llm_events) >= 2 or _has_real_fn_call:
            classification = "two-call"
        elif bracket_slacks:
            classification = "slack-hit"
        else:
            classification = "single-call"
        turn = Turn(
            index=idx,
            student_text=student_text,
            bot_text=bot_text,
            started_at=ts,    # we don't have a precise turn-start ts; the breakdown ts is good enough for ordering
            ended_at=ts,
            voice_to_voice_ms=v2v_ms,
            llm_calls=llm_events,
            stt_ms=stt_ms,
            tts_ms=tts_ms,
            function_calls=bracket_fns,
            node_transitions=bracket_nodes,
            slack_events_ms=bracket_slacks,
            classification=classification,
            raw_calls=bracket_raw_calls,
            real_wait_ms=real_wait_ms,
            endpointing_ms=endpointing_ms,
        )
        timeline.turns.append(turn)
        prior_secs = secs

    # Attribute summarization events to inter-turn positions. A summarization
    # logged at ts S falls between turn N and turn N+1 where turn N is the
    # last turn ending at or before S.
    turn_end_secs = [_ts_to_seconds(t.ended_at) for t in timeline.turns if t.ended_at]
    for s_secs, s_rec in summarization_records:
        after_idx = -1
        for i, te in enumerate(turn_end_secs):
            if te <= s_secs:
                after_idx = i
            else:
                break
        timeline.summarization_events.append(SummarizationEvent(
            ts=s_rec.get("ts", ""),
            after_turn_index=after_idx,
            service=s_rec.get("service", ""),
            model=s_rec.get("model"),
            input=s_rec.get("input"),
            output=s_rec.get("output"),
            usage=s_rec.get("usage"),
            error=s_rec.get("error"),
        ))

    # Summary stats
    v2vs = [t.voice_to_voice_ms for t in timeline.turns if t.voice_to_voice_ms is not None]
    v2vs_sorted = sorted(v2vs)
    median = v2vs_sorted[len(v2vs_sorted) // 2] if v2vs_sorted else None
    timeline.summary = {
        "turn_count": len(timeline.turns),
        "two_call_turns": sum(1 for t in timeline.turns if t.classification == "two-call"),
        "slack_hits": sum(1 for t in timeline.turns if t.slack_events_ms),
        "median_v2v_ms": median,
        "max_v2v_ms": max(v2vs) if v2vs else None,
        "min_v2v_ms": min(v2vs) if v2vs else None,
        "summarization_count": len(timeline.summarization_events),
    }
    real = sorted(t.real_wait_ms for t in timeline.turns if t.real_wait_ms is not None)
    eps = sorted(t.endpointing_ms for t in timeline.turns if t.endpointing_ms is not None)
    timeline.summary.update({
        "median_real_wait_ms": real[len(real) // 2] if real else None,
        "max_real_wait_ms": real[-1] if real else None,
        "median_endpointing_ms": eps[len(eps) // 2] if eps else None,
        # None for older sessions recorded before cut-off tracking existed.
        "cutoff_count": len(scan["cutoffs"]) if real or scan["cutoffs"] else None,
    })
    timeline.summary["duration_seconds"] = _session_duration(
        report_path, timeline.started_at, timeline.ended_at
    )
    timeline.cost = session_cost(session_id, log_path)
    return timeline


def _session_duration(report_path: Path, started_at: str | None, ended_at: str | None) -> float | None:
    """Seconds the session was live. Prefers the report's duration (student
    connected → call ended); falls back to bot start → transcript snapshot,
    which also includes grading time."""
    if report_path.exists():
        try:
            meta = json.loads(report_path.read_text(encoding="utf-8")).get("metadata") or {}
            if meta.get("duration_seconds"):
                return float(meta["duration_seconds"])
        except Exception:
            pass
    if started_at and ended_at:
        try:
            start = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S.%f")
            return max((datetime.fromisoformat(ended_at) - start).total_seconds(), 0.0)
        except ValueError:
            pass
    return None


def timeline_to_dict(t: SessionTimeline) -> dict[str, Any]:
    """JSON-serializable dict for the API response."""
    d = asdict(t)
    return d
