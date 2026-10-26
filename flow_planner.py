"""FlowPlanner — calls Google Gemini to generate a FlowSpec JSON from a topic string."""

import json
import os

from google import genai
from google.genai import types
from loguru import logger

from flow_spec_schema import FlowSpec

# Pacing model shared with the live examiner (dynamic_exam): each question —
# ask, answer, at most one short follow-up — takes ~75s, plus ~60s for the
# greeting and closing. The planner sizes the probe pool to this so the
# conversation naturally fills the requested duration.
SECS_PER_QUESTION = float(os.getenv("EXAM_SECS_PER_QUESTION", "75"))
EXAM_OVERHEAD_SECS = 60


def planned_question_count(duration_minutes: float) -> int:
    """How many questions (opening question + probes) fit the duration."""
    return max(2, round((duration_minutes * 60 - EXAM_OVERHEAD_SECS) / SECS_PER_QUESTION))


PLANNER_PROMPT = """
You are designing an IB MYP oral exam conversation flow.

Topic: {topic}
Target duration: {duration} minutes

Generate a FlowSpec JSON with EXACTLY these fields:
{{
  "topic": "{topic}",
  "exam_question": "<one clear open-ended opinion question — must invite a YES/NO/CONDITIONAL stance plus a WHY>",
  "duration_minutes": {duration},
  "examiner_tone": "warm and encouraging",
  "skills_assessed": ["<skill1>", "<skill2>", "<skill3>"],
  "redirect_when_off_topic": "<one sentence to redirect an off-topic student back to the exam question>",
  "required_probes": ["<probe_id_1>", "<probe_id_2>"],
  "probe_pool": [
    {{
      "id": "<short_snake_case_id>",
      "type": "<question|example|counter|comparison|precision|synthesis>",
      "prompt": "<full task instruction for the AI examiner — what to ask, when to push, when to scaffold>",
      "scaffold": "<one sentence to say if student is completely stuck. Empty string if not needed.>"
    }}
  ]
}}

Rules:
- exam_question must be opinion-based and open-ended (not factual)
- exam_question must be SHORT and SINGLE-FOCUS: ONE question, at most ~25 words,
  one sentence. Do NOT bundle multiple questions, and do NOT append extra
  instructions like "be ready to defend...", "engage with... too", or
  "consider both X and Y" — those belong in the probes, not the opening
  question. The student should hear one clear, answerable question.
- Include EXACTLY {n_probes} probes in probe_pool covering different angles of the
  topic. This count is sized so the exam fills {duration} minutes: each probe is
  about 75 seconds of conversation (one question, the answer, at most one short
  follow-up), so each probe must be answerable in that time
- Every probe must target information NO other probe asks for. No two probes may
  be the same question reworded (e.g. "why did you choose X" and "what motivated
  you to pick X" are duplicates — keep only one). No probe may re-ask the
  exam_question itself; probes go beyond it (evidence, counterpoint, comparison,
  limitation, consequence, precision)
- required_probes: exactly 2 probe IDs that MUST be covered in every session
- precision probe type must appear exactly once and should be last in the pool
- prompt is an instruction to the AI examiner, NOT the actual words to say to the student
- scaffold is a simplified help phrase if the student says "I don't know"
- probe IDs must be unique snake_case strings (e.g. "ethics_probe", "access_probe")

Return ONLY valid JSON. No markdown fences. No explanation. No extra text.
"""


def _question_too_long(question: str, max_words: int = 25) -> bool:
    """True if the opening question is over the word cap or is compound.

    The planner is told to keep exam_question short and single-focus, but it
    sometimes returns a long or multi-part question anyway (two '?'s, or "and"
    joining two asks). Those overwhelm students mid-exam, so we catch them here
    and regenerate once.
    """
    q = (question or "").strip()
    if len(q.split()) > max_words:
        return True
    if q.count("?") > 1:  # more than one question packed in
        return True
    return False


async def generate_flow(topic: str, duration: int = 6) -> FlowSpec:
    """Call Gemini to generate a FlowSpec for the given topic."""
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    logger.info(f"Generating flow for topic: {topic!r}")

    # Questions after the opening one, plus a spare so the examiner can skip a
    # probe that doesn't fit; clamped to flow_validator's 3-10 range.
    n_probes = min(10, max(3, planned_question_count(duration)))
    prompt = PLANNER_PROMPT.format(topic=topic, duration=duration, n_probes=n_probes)
    spec: FlowSpec | None = None
    # One regeneration attempt if the opening question comes back too long or
    # compound. We nudge the second call with an explicit, stronger constraint.
    for attempt in range(2):
        contents = prompt
        if attempt == 1:
            contents = (
                prompt
                + "\n\nIMPORTANT: the previous exam_question was too long or "
                "asked more than one thing. Return exam_question as ONE single "
                "question of at most 25 words, with exactly one '?'."
            )
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        candidate = FlowSpec.model_validate(json.loads(response.text))
        spec = candidate
        if not _question_too_long(candidate.exam_question):
            break
        logger.warning(
            f"Exam question too long/compound ({len(candidate.exam_question.split())} "
            f"words), regenerating: {candidate.exam_question!r}"
        )

    logger.info(f"Flow generated: {len(spec.probe_pool)} probes, required={spec.required_probes}")
    logger.debug(f"Exam question: {spec.exam_question}")
    return spec
