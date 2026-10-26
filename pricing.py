"""Per-session usage tally + cost estimate for the /sessions dashboard.

Prices are PUBLIC LIST PRICES as of Oct 2026 — edit the tables below to match
your own plan / contract. Anything with no price entry is still counted but
shown as "no price set" and left out of the total.

Usage comes from two places:
  - usage/usage_<session>.json, written at the end of each exam by
    SessionUsage (exact, per-session — preferred).
  - the session log, for older runs that predate the usage file (estimate:
    the log can also contain lines from other exams running at the same time).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

USAGE_DIR = Path(__file__).parent / "usage"

# ─── Prices ───────────────────────────────────────────────────────────────────

# USD per 1M tokens. Matched by prefix, longest first, so "claude-haiku-5-5"
# also covers dated variants like "claude-haiku-5-5-20261007".
LLM_PRICES: dict[str, dict[str, float]] = {
    # Haiku 5.5: prompts over 100K tokens are billed at 5x (long-context tier).
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01,
                         "cache_write": 0.125, "long_threshold": 100_000, "long_multiplier": 5},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
    # Gemini bills audio input at the same rate as text on 3.5 Flash.
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00, "cache_read": 0.15},
    "gpt-4o-mini":      {"input": 0.15, "output": 0.60, "cache_read": 0.075},
    "gpt-oss-120b":     {"input": 0.25, "output": 0.69},
}

# Cartesia bills in credits. Scale plan: $299 for 8M credits.
CARTESIA_USD_PER_CREDIT = 299 / 8_000_000
TTS_CREDITS_PER_CHAR: dict[str, float] = {"sonic": 1.0}          # all Sonic models
STT_CREDITS_PER_SECOND: dict[str, float] = {"ink-2": 3.0, "ink-whisper": 1.0}

CATEGORY_LABELS = {
    "conversation": "Examiner AI (LLM)",
    "grading": "Grading (Gemini)",
    "summarization": "Context summarizer",
    "tts": "Voice (text-to-speech)",
    "stt": "Listening (speech-to-text)",
}


def _lookup(table: dict[str, Any], model: str | None) -> Any:
    if not model:
        return None
    m = model.lower()
    for key in sorted(table, key=len, reverse=True):
        if m.startswith(key) or key in m:
            return table[key]
    return None


# ─── Usage tally (lives for one exam session) ─────────────────────────────────


class SessionUsage:
    """Accumulates LLM tokens, TTS characters and STT seconds for one session."""

    def __init__(self) -> None:
        self.llm: dict[tuple[str, str], dict[str, int]] = {}
        self.tts: dict[str, int] = {}
        self.stt: dict[str, float] = {}

    def add_llm(self, category: str, model: str | None, *, prompt: int = 0,
                completion: int = 0, cache_read: int = 0, cache_write: int = 0,
                reasoning: int = 0) -> None:
        # Anthropic reports prompt tokens net of cache; Gemini/OpenAI include
        # cached tokens in the prompt count. Normalise to "uncached input".
        model = model or "unknown"
        uncached = prompt if model.startswith("claude") else max(prompt - cache_read, 0)
        b = self.llm.setdefault((category, model), {
            "calls": 0, "input": 0, "output": 0, "cache_read": 0,
            "cache_write": 0, "long_calls_input": 0, "long_calls_output": 0,
        })
        b["calls"] += 1
        out = completion + reasoning
        total_in = uncached + cache_read + cache_write
        price = _lookup(LLM_PRICES, model) or {}
        if price.get("long_threshold") and total_in > price["long_threshold"]:
            b["long_calls_input"] += uncached
            b["long_calls_output"] += out
        else:
            b["input"] += uncached
            b["output"] += out
        b["cache_read"] += cache_read
        b["cache_write"] += cache_write

    def add_tts(self, model: str | None, characters: int) -> None:
        key = model or "unknown"
        self.tts[key] = self.tts.get(key, 0) + int(characters)

    def add_stt(self, model: str | None, seconds: float) -> None:
        key = model or "unknown"
        self.stt[key] = self.stt.get(key, 0.0) + float(seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm": [{"category": c, "model": m, **v} for (c, m), v in self.llm.items()],
            "tts": [{"model": m, "characters": n} for m, n in self.tts.items()],
            "stt": [{"model": m, "seconds": round(s, 3)} for m, s in self.stt.items()],
        }

    def save(self, session_id: str) -> Path:
        USAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = USAGE_DIR / f"usage_{session_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


# ─── Cost computation ─────────────────────────────────────────────────────────


def compute_cost(usage: dict[str, Any]) -> dict[str, Any]:
    """Turn a SessionUsage.to_dict() into priced line items + a total."""
    lines: list[dict[str, Any]] = []

    for row in usage.get("llm", []):
        p = _lookup(LLM_PRICES, row["model"])
        usd = None
        if p:
            mult = p.get("long_multiplier", 1)
            usd = (
                row["input"] * p["input"]
                + row["output"] * p["output"]
                + row.get("long_calls_input", 0) * p["input"] * mult
                + row.get("long_calls_output", 0) * p["output"] * mult
                + row["cache_read"] * p.get("cache_read", p["input"])
                + row["cache_write"] * p.get("cache_write", p["input"])
            ) / 1_000_000
        in_tok = row["input"] + row.get("long_calls_input", 0) + row["cache_read"] + row["cache_write"]
        out_tok = row["output"] + row.get("long_calls_output", 0)
        lines.append({
            "category": row["category"],
            "label": CATEGORY_LABELS.get(row["category"], row["category"]),
            "model": row["model"],
            "quantity": f"{row['calls']} calls · {in_tok:,} in / {out_tok:,} out tokens",
            "usd": usd,
        })

    for row in usage.get("tts", []):
        rate = _lookup(TTS_CREDITS_PER_CHAR, row["model"])
        lines.append({
            "category": "tts",
            "label": CATEGORY_LABELS["tts"],
            "model": row["model"],
            "quantity": f"{row['characters']:,} characters",
            "usd": row["characters"] * rate * CARTESIA_USD_PER_CREDIT if rate is not None else None,
        })

    for row in usage.get("stt", []):
        rate = _lookup(STT_CREDITS_PER_SECOND, row["model"])
        lines.append({
            "category": "stt",
            "label": CATEGORY_LABELS["stt"],
            "model": row["model"],
            "quantity": f"{row['seconds'] / 60:.1f} min audio",
            "usd": row["seconds"] * rate * CARTESIA_USD_PER_CREDIT if rate is not None else None,
        })

    priced = [l["usd"] for l in lines if l["usd"] is not None]
    return {
        "lines": lines,
        "total_usd": sum(priced) if priced else None,
        "unpriced": [l["model"] for l in lines if l["usd"] is None],
    }


# ─── Fallback: rebuild usage from a session log (older runs) ──────────────────

_RE_LLM_USAGE = re.compile(
    r"\[(?P<svc>[^\]]+)\] LLM TOKEN USAGE \((?P<model>[^)]+)\): (?P<fields>.+?)(?: at [\d.]+s)?$"
)
_RE_TTS_USAGE = re.compile(r"TTS USAGE \((?P<model>[^)]+)\): (?P<n>\d+) characters")
_RE_STT_USAGE = re.compile(r"STT USAGE \((?P<model>[^)]+)\): (?P<s>[\d.]+)s audio")


def usage_from_log(log_path: Path) -> dict[str, Any] | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    u = SessionUsage()
    found = False
    for line in text.splitlines():
        if "📊" not in line:
            continue
        if m := _RE_LLM_USAGE.search(line):
            f = dict(
                (k.strip(), int(v)) for k, v in
                (kv.split(":") for kv in m.group("fields").split(",") if ":" in kv)
                if v.strip().isdigit()
            )
            category = "grading" if m.group("svc") == "Grader" else "conversation"
            u.add_llm(category, m.group("model"),
                      prompt=f.get("prompt", 0), completion=f.get("completion", 0),
                      cache_read=f.get("cache_read", 0), cache_write=f.get("cache_creation", 0),
                      reasoning=f.get("reasoning", 0))
            found = True
        elif m := _RE_TTS_USAGE.search(line):
            u.add_tts(m.group("model"), int(m.group("n")))
            found = True
        elif m := _RE_STT_USAGE.search(line):
            u.add_stt(m.group("model"), float(m.group("s")))
            found = True
    return u.to_dict() if found else None


def session_cost(session_id: str, log_path: Path) -> dict[str, Any] | None:
    """Cost for one session: exact usage file if present, else log estimate."""
    usage_path = USAGE_DIR / f"usage_{session_id}.json"
    source = "usage_file"
    usage = None
    if usage_path.exists():
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        except Exception:
            usage = None
    if usage is None:
        usage = usage_from_log(log_path)
        source = "log_estimate"
    if usage is None:
        return None
    cost = compute_cost(usage)
    cost["source"] = source
    return cost
