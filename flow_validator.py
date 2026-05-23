"""Graph-level validation for a generated FlowSpec.

Pydantic catches per-field problems. This module catches problems only
visible across the whole graph: duplicate IDs, dangling required-probe
references, unsupported probe types, etc.

`validate_flow_spec(spec)` raises `FlowValidationError` with a list of
human-readable reasons that can be fed back to the LLM for a retry.
"""

from __future__ import annotations

import re

from flow_primitives import ALLOWED_PROBE_TYPES
from flow_spec_schema import FlowSpec

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


class FlowValidationError(ValueError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("FlowSpec failed validation:\n  - " + "\n  - ".join(reasons))


def validate_flow_spec(spec: FlowSpec) -> None:
    reasons: list[str] = []

    # 0. Pool size sanity
    if len(spec.probe_pool) < 3:
        reasons.append(
            f"probe_pool too small ({len(spec.probe_pool)}); need at least 3"
        )
    if len(spec.probe_pool) > 10:
        reasons.append(
            f"probe_pool too large ({len(spec.probe_pool)}); cap is 10"
        )

    ids = [p.id for p in spec.probe_pool]

    # 1. ID format
    bad_ids = [i for i in ids if not _ID_RE.match(i)]
    if bad_ids:
        reasons.append(
            f"invalid probe ids {bad_ids}; must be snake_case "
            "(lowercase, starts with letter, 2–40 chars, [a-z0-9_])"
        )

    # 2. Unique IDs
    seen: set[str] = set()
    dupes = sorted({i for i in ids if (i in seen) or seen.add(i)})  # type: ignore[func-returns-value]
    if dupes:
        reasons.append(f"duplicate probe ids: {dupes}")

    # 3. probe.type ∈ catalog
    bad_types = sorted({p.type for p in spec.probe_pool if p.type not in ALLOWED_PROBE_TYPES})
    if bad_types:
        reasons.append(
            f"unsupported probe types {bad_types}; allowed: {list(ALLOWED_PROBE_TYPES)}"
        )

    # 4. required_probes must exist in pool
    id_set = set(ids)
    missing_required = [r for r in spec.required_probes if r not in id_set]
    if missing_required:
        reasons.append(
            f"required_probes reference unknown ids: {missing_required}; "
            f"pool has: {sorted(id_set)}"
        )

    # 5. required_probes uniqueness
    if len(set(spec.required_probes)) != len(spec.required_probes):
        reasons.append(f"required_probes contains duplicates: {spec.required_probes}")

    # 6. Prompt substantive
    short_prompts = [p.id for p in spec.probe_pool if len(p.prompt.strip()) < 20]
    if short_prompts:
        reasons.append(
            f"probe prompts too short (< 20 chars) for ids: {short_prompts}"
        )

    # 7. Exam question substantive
    if len(spec.exam_question.strip()) < 15:
        reasons.append("exam_question must be at least 15 characters")

    # 8. At most one precision probe; if present, must be last
    precision_indices = [i for i, p in enumerate(spec.probe_pool) if p.type == "precision"]
    if len(precision_indices) > 1:
        reasons.append(
            f"more than one precision probe at indices {precision_indices}; only one allowed"
        )
    elif len(precision_indices) == 1 and precision_indices[0] != len(spec.probe_pool) - 1:
        reasons.append(
            f"precision probe should be LAST in probe_pool "
            f"(found at index {precision_indices[0]}, pool size {len(spec.probe_pool)})"
        )

    if reasons:
        raise FlowValidationError(reasons)
