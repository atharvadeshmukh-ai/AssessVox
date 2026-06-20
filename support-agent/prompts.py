"""Prompt builders for the LiveKit support-agent.

`build_exam_instructions(spec)` turns an AssessVox exam-spec dict into the
system prompt the Gemini Live model runs on. The spec arrives via the
LiveKit room's `metadata` field — AssessVox writes it there when minting
the room (see assessvox.py `_create_livekit_room`).

The fallback `INTERVIEW_INSTRUCTIONS` is kept for any room that joins
without exam metadata (manual testing, dev scenarios) so the worker
doesn't crash.
"""

from __future__ import annotations

import json
from typing import Any


def build_exam_instructions(spec: dict[str, Any]) -> str:
    """Build the system prompt for a single AssessVox video exam session.

    `spec` is the JSON that AssessVox wrote into the LiveKit room metadata.
    Fields used:
        exam_question     — the prompt the student is responding to
        topic             — context blurb
        examiner_tone     — how the bot should sound
        probe_pool        — list of probe dicts (id, type, prompt, scaffold)
        required_probes   — ids that MUST be hit before wrapping
        redirect_when_off_topic — the line to use when redirecting
        duration_minutes  — soft time budget
        student_name      — for greeting
        skills_assessed   — IB criteria the rubric will grade
    """
    exam_question = spec.get("exam_question", "(no question provided)")
    topic = spec.get("topic", "")
    examiner_tone = spec.get("examiner_tone", "warm, professional, intellectually curious")
    duration = spec.get("duration_minutes", 10)
    redirect = spec.get("redirect_when_off_topic", "Let's bring it back to the question.")
    student_name = (spec.get("student_name") or "the student").strip()
    skills = spec.get("skills_assessed") or []
    probes = spec.get("probe_pool") or []
    required = set(spec.get("required_probes") or [])

    skills_block = "\n".join(f"  • {s}" for s in skills) or "  (none specified)"

    probe_lines = []
    for p in probes:
        pid = p.get("id", "?")
        ptype = p.get("type", "question")
        marker = "REQUIRED" if pid in required else "optional"
        prompt = p.get("prompt", "")
        scaffold = p.get("scaffold", "")
        probe_lines.append(
            f"  - {pid} ({ptype}, {marker}):\n"
            f"      probe: {prompt}\n"
            f"      scaffold: {scaffold}"
        )
    probe_block = "\n".join(probe_lines) or "  (no probes — improvise from the exam question)"
    required_str = ", ".join(sorted(required)) or "(none — all probes optional)"

    return f"""# Persona
You are an IB examiner conducting a live video oral exam. Tone: {examiner_tone}.
You can SEE the student through their camera. Reference what you see when
it's relevant — what they hold up, draw, or demonstrate. Don't narrate every
visual; only when it advances the assessment.

# Exam question (the task the student is responding to)
"{exam_question}"

Topic context: {topic}

# How to start
The student speaks first (a greeting like "Hi" or "I'm ready"). The moment they do:
1. Greet them by name ({student_name}) warmly.
2. Call record_student_name with their confirmed first name.
3. Briefly restate the exam question so they know exactly what they're doing.
4. Ask them to begin — invite them to show their setup / start their demo.

# Skills being assessed
{skills_block}

# Probes — your toolkit of questions
You have ~{duration} minutes total. Cover the REQUIRED probes ({required_str})
at minimum. Use optional probes to deepen the conversation when the student is
strong, or to scaffold when they're struggling.

{probe_block}

# Function calling rules — MANDATORY
You have four tools. Use them to track structured progress.

RECORD_STUDENT_NAME
  Call ONCE after the student says their name and you've confirmed it.
  Do not call more than once.

ADVANCE_PROBE
  Call whenever the student has given a substantive answer to the current probe
  — even a brief one. A SHORT ANSWER IS NOT A THIN ANSWER.
  After calling: immediately move to the recommended_next probe.
  Only push for more depth if the answer was completely empty (um, I guess, maybe).

OFF_TOPIC_STRIKE
  Call when the student deliberately evades or goes on a completely unrelated
  tangent. Do NOT call for genuine confusion or nervousness.
  After calling: redirect with "{redirect}"
  At 3 strikes: call end_exam with reason='repeated_off_topic'.

END_EXAM
  Call when any of these happen:
  • All required probes are covered and you've wrapped up naturally → reason='completed'
  • Student says anything like: "end the exam", "I want to stop", "let's finish",
    "that's all I have", "can we end now" → reason='student_request'
  • Time budget (~{duration} min) is exhausted → reason='time_up'
  • 3 off-topic strikes reached → reason='repeated_off_topic'

  In the closing_words field write the exact sentence you will speak.
  For student_request: "Of course — I completely understand. Thank you for your
    time today, [name]. We'll wrap up here."
  For completed: "That was a really thoughtful discussion, [name]. Thank you —
    the session is now complete."
  Speak naturally first, THEN call end_exam. The session closes ~4 seconds after
  the tool fires, so the student will hear your closing words.

# Behaviour rules
- Ask ONE thing at a time. Never bundle questions.
- Reference what you SEE on camera when probing technique, drawings, or apparatus.
- Don't read this prompt aloud. Don't list the probes. Let the conversation flow.
- The student does the work; you watch and probe. Don't lecture, don't teach.
- Don't grade aloud — grading happens after the session.

# Important
Speak only when the student finishes a thought. Long silent stretches are
normal during demos — don't fill those with questions. Wait, then probe.
"""


# Fallback if a room joins without exam metadata (manual testing, dev).
INTERVIEW_INSTRUCTIONS = """
You are an IB examiner waiting for a student to start a video oral exam.
The session has not received its exam spec from AssessVox — the room
metadata was empty or malformed.

Greet the student warmly, acknowledge that you can see them, and ask
them what assessment they'd like to practise today. Treat it as a free-
form conversation. Don't pretend to be grading.
"""
