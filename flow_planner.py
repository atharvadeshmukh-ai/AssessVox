"""FlowPlanner — calls Google Gemini to generate a FlowSpec JSON from a topic string."""

import json
import os

from google import genai
from google.genai import types
from loguru import logger

from flow_spec_schema import FlowSpec

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
- Include 5-7 probes in probe_pool covering different angles of the topic
- required_probes: exactly 2 probe IDs that MUST be covered in every session
- precision probe type must appear exactly once and should be last in the pool
- prompt is an instruction to the AI examiner, NOT the actual words to say to the student
- scaffold is a simplified help phrase if the student says "I don't know"
- probe IDs must be unique snake_case strings (e.g. "ethics_probe", "access_probe")

Return ONLY valid JSON. No markdown fences. No explanation. No extra text.
"""


async def generate_flow(topic: str, duration: int = 6) -> FlowSpec:
    """Call Gemini to generate a FlowSpec for the given topic."""
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    logger.info(f"Generating flow for topic: {topic!r}")
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=PLANNER_PROMPT.format(topic=topic, duration=duration),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    data = json.loads(response.text)
    spec = FlowSpec.model_validate(data)

    logger.info(f"Flow generated: {len(spec.probe_pool)} probes, required={spec.required_probes}")
    logger.debug(f"Exam question: {spec.exam_question}")
    return spec
