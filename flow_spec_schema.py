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
