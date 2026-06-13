"""Template loader for AssessVox.

Templates are pre-built assessment specifications stored as JSON files in
``templates/``. Each template provides:

  • Metadata for the UI dropdown (display_name, programme, subject, level)
  • A FlowSpec (topic, question, probes, required probes, examiner tone, etc.)
  • A template-specific grading rubric (mirrors the assessment's IB criteria)

Why templates exist:
  Different IB assessments have wildly different shapes (MYP Language Acquisition
  Oral ≠ DP Language B SL Oral ≠ TOK Oral Defense ≠ Personal Project Interview).
  Each has its own probe structure, examiner tone, time budget, and IB grading
  criteria. Templates let a teacher pick "DP Lang B SL Oral" and get a
  pre-validated FlowSpec without having to write a free-text question and rely
  on the LLM to generate the structure.

  The free-text path (``template_id == None``) is preserved for ad-hoc questions.

Schema:
  {
    "id": "dp-language-b-sl-oral",        # used in API + filenames
    "programme": "DP",                     # MYP | DP | PYP
    "subject": "Language B",
    "level": "SL",                         # SL | HL | N/A
    "assessment_type": "Individual Oral",
    "display_name": "DP Language B SL — Individual Oral",
    "description": "15-min oral with visual-stimulus prompt + theme discussion.",
    "flow_spec": { ...same shape as flow_spec_schema.FlowSpec... },
    "grading_rubric": "<full grading prompt string with {placeholders}>"
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from flow_spec_schema import FlowSpec

TEMPLATES_DIR = Path(__file__).parent / "templates"


class Template:
    """In-memory representation of a loaded template + its question library."""

    def __init__(self, raw: dict[str, Any], path: Path):
        self.raw = raw
        self.path = path
        self.id: str = raw["id"]
        self.programme: str = raw.get("programme", "Custom")
        self.subject: str = raw.get("subject", "")
        self.level: str = raw.get("level", "N/A")
        self.assessment_type: str = raw.get("assessment_type", "")
        self.display_name: str = raw.get("display_name", self.id)
        self.description: str = raw.get("description", "")
        self.grading_rubric: str | None = raw.get("grading_rubric") or None
        # Optional video-specific rubric. Used only when a session is graded
        # via the video path (i.e. the picked question's requires_video=True
        # AND a recording file is present). When None, the global
        # VIDEO_GRADING_PROMPT default applies.
        self.video_grading_rubric: str | None = raw.get("video_grading_rubric") or None
        # Validate the FlowSpec immediately — fails fast at load time, not at
        # session-creation time.
        self.flow_spec: FlowSpec = FlowSpec(**raw["flow_spec"])
        # Question library: each entry has id, label, prompt, optional theme /
        # global_context / image_url / notes. The teacher picks one from the
        # UI, or writes a custom prompt (question_id=None).
        self.questions: list[dict[str, Any]] = list(raw.get("questions") or [])

    def get_question(self, question_id: str) -> dict[str, Any] | None:
        """Return a question dict from the library by id, or None."""
        for q in self.questions:
            if q.get("id") == question_id:
                return q
        return None

    def build_flow_spec(self, question_override: str | None = None) -> FlowSpec:
        """Return a FlowSpec, optionally overriding the exam question.

        Resolution order for the exam_question that ends up in the FlowSpec:
          1. `question_override` if provided and non-empty (this comes from
             either a library question's prompt, or a custom teacher text).
          2. `flow_spec.exam_question` (the template's built-in default).
        Probe pool, required probes, examiner tone, and the redirect line
        always stay from the template.
        """
        if not question_override or not question_override.strip():
            return self.flow_spec.model_copy(deep=True)
        return self.flow_spec.model_copy(update={"exam_question": question_override.strip()})

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe shape for the /api/templates endpoint."""
        return {
            "id": self.id,
            "programme": self.programme,
            "subject": self.subject,
            "level": self.level,
            "assessment_type": self.assessment_type,
            "display_name": self.display_name,
            "description": self.description,
            "duration_minutes": self.flow_spec.duration_minutes,
            "default_question": self.flow_spec.exam_question,
            "question_count": len(self.questions),
        }

    def questions_to_dict(self) -> list[dict[str, Any]]:
        """JSON-safe library for /api/templates/{id}/questions.

        Returns each question with id, label, prompt, optional grouping keys
        (theme / global_context), image_url, and notes. Order preserved
        from the JSON source.
        """
        out: list[dict[str, Any]] = []
        for q in self.questions:
            out.append({
                "id": q.get("id"),
                "label": q.get("label") or q.get("id") or "(unnamed)",
                "prompt": q.get("prompt") or "",
                "theme": q.get("theme"),
                "global_context": q.get("global_context"),
                "image_url": q.get("image_url"),
                "requires_video": bool(q.get("requires_video")),
                "notes": q.get("notes") or "",
            })
        return out


_TEMPLATES: dict[str, Template] = {}
_LOADED = False


def _load_all() -> None:
    """Scan TEMPLATES_DIR for *.json files and populate the registry."""
    global _LOADED
    _TEMPLATES.clear()
    if not TEMPLATES_DIR.exists():
        logger.warning(f"Templates directory not found: {TEMPLATES_DIR}")
        _LOADED = True
        return

    for path in sorted(TEMPLATES_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
            tpl = Template(raw, path)
        except Exception as e:
            logger.error(f"Failed to load template {path.name}: {e}")
            continue
        if tpl.id in _TEMPLATES:
            logger.warning(
                f"Duplicate template id {tpl.id!r} (loading {path.name}, "
                f"already have {_TEMPLATES[tpl.id].path.name})"
            )
        _TEMPLATES[tpl.id] = tpl
        logger.info(f"Loaded template: {tpl.id} ({tpl.display_name})")
    _LOADED = True


def list_templates() -> list[Template]:
    """Return all loaded templates in display-friendly order.

    Sorted by programme (MYP → DP → others) then assessment type then level.
    """
    if not _LOADED:
        _load_all()
    order = {"MYP": 0, "DP": 1, "PYP": 2}
    return sorted(
        _TEMPLATES.values(),
        key=lambda t: (order.get(t.programme, 99), t.subject, t.assessment_type, t.level),
    )


def get_template(template_id: str) -> Template | None:
    """Look up a template by id. Returns None if not found."""
    if not _LOADED:
        _load_all()
    return _TEMPLATES.get(template_id)


def reload() -> None:
    """Re-scan the templates directory. Useful for dev / hot-edit."""
    global _LOADED
    _LOADED = False
    _load_all()
