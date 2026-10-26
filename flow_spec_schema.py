"""FlowSpec schema — Pydantic models for dynamic exam flow generation."""

from typing import Literal
from pydantic import BaseModel, Field


class ProbeNode(BaseModel):
    id: str
    type: Literal["question", "example", "counter", "comparison", "precision", "synthesis"]
    prompt: str   # task instruction for the AI examiner
    scaffold: str = ""  # what to say if student is completely stuck


class FlowSpec(BaseModel):
    topic: str
    exam_question: str
    duration_minutes: int = 6
    examiner_tone: str = "warm and encouraging"
    skills_assessed: list[str] = Field(default_factory=list)
    redirect_when_off_topic: str = (
        "That is interesting — but let us bring it back to the main question."
    )
    required_probes: list[str] = Field(default_factory=list)
    probe_pool: list[ProbeNode] = Field(default_factory=list)
    # Exam format. "discussion" = opinion question (default). For "role_play"
    # and "listen_respond", `opening_script` is read aloud word for word to start
    # the exam (scenario + first in-character line, or the listening message +
    # first question), and `examiner_brief` is private guidance kept in front of
    # the examiner for the whole exam (character, passage text, etc.).
    question_style: Literal["discussion", "role_play", "listen_respond"] = "discussion"
    opening_script: str = ""
    examiner_brief: str = ""
    # Accurate description of the picture shown to the student (image
    # questions). Private: the examiner uses it to check the student's
    # description and the grader to judge accuracy — it is never read aloud.
    image_description: str = ""
