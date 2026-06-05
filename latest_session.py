"""Quick helper to surface the most recent session's artefacts.

Usage:
    uv run latest_session.py            # just print the session id + paths
    uv run latest_session.py --log      # also print the full session log
    uv run latest_session.py --turns    # also print the transcript turns
    uv run latest_session.py --all      # everything
    uv run latest_session.py <id>       # use a specific session id

Designed so you can paste "analyse the latest session" to Claude and Claude can
just read transcripts/transcript_<id>.json + logs/session_<id>.log directly,
no copy-paste of long logs required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
LOGS_DIR        = ROOT / "logs"
TRANSCRIPTS_DIR = ROOT / "transcripts"
REPORTS_DIR     = ROOT / "reports"
RECORDINGS_DIR  = ROOT / "recordings"
FLOWS_DIR       = ROOT / "flows"


def find_latest_session_id() -> str | None:
    """Latest session id, picked from whichever directory has the freshest file."""
    candidates: list[tuple[float, str]] = []
    for d, prefix, suffix in [
        (TRANSCRIPTS_DIR, "transcript_", ".json"),
        (LOGS_DIR,        "session_",    ".log"),
        (REPORTS_DIR,     "report_",     ".json"),
        (FLOWS_DIR,       "flow_",       ".json"),
    ]:
        if not d.exists():
            continue
        for p in d.glob(f"{prefix}*{suffix}"):
            sid = p.stem[len(prefix):]
            candidates.append((p.stat().st_mtime, sid))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def session_paths(session_id: str) -> dict[str, Path]:
    return {
        "log":        LOGS_DIR        / f"session_{session_id}.log",
        "transcript": TRANSCRIPTS_DIR / f"transcript_{session_id}.json",
        "report":     REPORTS_DIR     / f"report_{session_id}.json",
        "recording":  RECORDINGS_DIR  / f"exam_{session_id}.wav",
        "flow":       FLOWS_DIR       / f"flow_{session_id}.json",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id", nargs="?", default=None)
    ap.add_argument("--log",    action="store_true")
    ap.add_argument("--turns",  action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--all",    action="store_true")
    args = ap.parse_args()

    sid = args.session_id or find_latest_session_id()
    if not sid:
        print("No sessions found.", file=sys.stderr)
        return 1

    paths = session_paths(sid)
    print(f"session_id: {sid}")
    for name, p in paths.items():
        marker = "✓" if p.exists() else "·"
        print(f"  {marker} {name:<10} {p}")

    show_log    = args.log    or args.all
    show_turns  = args.turns  or args.all
    show_report = args.report or args.all

    if show_turns and paths["transcript"].exists():
        data = json.loads(paths["transcript"].read_text())
        print("\n── TRANSCRIPT ──")
        print(f"student: {data.get('student_name')!r}")
        print(f"covered_probes: {data.get('covered_probes')}")
        print(f"off_topic_strikes: {data.get('off_topic_count')}, "
              f"greeting_strikes: {data.get('greeting_strikes')}")
        for turn in data.get("turns", []):
            role = turn.get("role", "?")
            text = turn.get("content", "")
            print(f"  [{role}] {text}")

    if show_report and paths["report"].exists():
        print("\n── REPORT ──")
        print(paths["report"].read_text())

    if show_log and paths["log"].exists():
        print("\n── LOG ──")
        print(paths["log"].read_text())

    return 0


if __name__ == "__main__":
    sys.exit(main())
