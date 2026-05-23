"""Primitive catalog for dynamic oral exam flows.

A primitive is a typed template that turns a ProbeNode spec into the
`task_messages` content for a NodeConfig. The flow generator may only
compose probes whose `type` is a key in PRIMITIVES — everything else
is rejected by the validator. This is the closed allow-list.
"""

from __future__ import annotations

from typing import Callable

from flow_spec_schema import ProbeNode


_ADVANCE_INSTRUCTIONS = (
    "After {name} responds substantively, call advance_from_probe with:\n"
    "  depth: 'surface' or 'developed'\n"
    "  topics_raised: themes mentioned in this response\n"
    "  topics_missing: important exam themes not yet discussed\n"
    "  recommended_next: best next probe ID from the available list\n"
    "  gold_moment: a specific, vivid insight worth following up (empty string if none)\n"
    "  student_struggling: true if the student needed heavy scaffolding\n"
    "  vague_terms: imprecise words or phrases the student used\n\n"
    "If the result has next_action='wait_for_student', stop and wait."
)

_ACCEPTANCE_RULE = (
    "ACCEPTANCE — read this BEFORE deciding to ask a follow-up:\n"
    "A short answer is NOT automatically a thin answer. If {name}'s response "
    "NAMES a real reason, example, comparison, or specific concept relevant "
    "to the probe — even in one short phrase — that is a substantive answer. "
    "ACCEPT IT and immediately call advance_from_probe with depth='surface'. "
    "Do NOT demand elaboration on every utterance. Only push for more depth "
    "if the answer is genuinely empty (filler with no content, e.g. 'um yeah', "
    "'I guess', 'maybe', a single non-content word).\n"
    "Examples of acceptable short answers that should ADVANCE:\n"
    "  • 'It's about comfort and not having to travel.'  → accept, advance\n"
    "  • 'Productivity is higher at home.'                → accept, advance\n"
    "  • 'Social bonding gets weaker.'                    → accept, advance\n"
    "Do not interrogate a real answer just because it is brief.\n"
)

_ATTEMPT_CAP_RULE = (
    "TIERED ATTEMPT POLICY — mandatory:\n"
    "Count how many times you have asked {name} a follow-up WITHIN THIS CURRENT "
    "PROBE since the last function call (look back through the conversation).\n"
    "\n"
    "  ATTEMPT 1 — Ask the probe normally per the TASK instructions above.\n"
    "\n"
    "  ATTEMPT 2 — The first answer was thin. Do NOT re-ask the same question. "
    "Move SIDEWAYS to a DIFFERENT aspect of the probe topic, not deeper into the "
    "same micro-concept. If they mentioned 'comfort', do NOT ask 'how does comfort "
    "help?' — ask about a different angle like productivity, communication, or "
    "social impact. ALSO, offer the student a graceful out: 'Would you like to "
    "try a different question, or shall we stay on this one?' If they say yes / "
    "okay / move on, immediately call advance_from_probe with "
    "student_consented_to_advance=true and student_struggling=true.\n"
    "\n"
    "  ATTEMPT 3+ — You have now asked twice and the student is still struggling. "
    "ACCEPT whatever they have said and call advance_from_probe with "
    "depth='surface' and student_struggling=true. Acknowledge warmly: "
    "'That is a tricky one — let's move on.' Do NOT keep pushing. Do NOT "
    "rephrase the same demand again.\n"
)

_DISENGAGE_RULE = (
    "DISENGAGE SIGNALS — mandatory, overrides everything else:\n"
    "If {name}'s most recent message contains ANY of the following intents — "
    "even paraphrased — you MUST immediately accept their previous answer at its "
    "current depth and call advance_from_probe with student_consented_to_advance=true "
    "and student_struggling=true. Do NOT push back. Do NOT say 'however, we need "
    "to explore...'. Do NOT ask another follow-up. Just acknowledge in one short "
    "sentence and advance.\n"
    "Disengage intents:\n"
    "  • 'next question' / 'next one' / 'move on' / 'move to the next' / 'skip'\n"
    "  • 'I don't know' / 'I don't know more' / 'I have nothing to add'\n"
    "  • 'I don't want to answer' / 'I'd rather not' / 'pass'\n"
    "  • 'can we continue?' / 'let's keep going' / 'what's next?'\n"
    "  • 'I'm done with this question' / 'I'm finished' / 'that's all I have'\n"
    "  • 'I can't recall' / 'I can't think of one' / 'nothing comes to mind'\n"
    "Honouring disengage signals is non-negotiable — it is more important than "
    "extracting depth. A frustrated student is a failed exam.\n"
)


def _scaffold_line(name: str, probe: ProbeNode) -> str:
    if not probe.scaffold:
        return ""
    return f"SCAFFOLD (use only if {name} is completely stuck): {probe.scaffold}\n\n"


def _question_primitive(name: str, probe: ProbeNode, available: list[str]) -> str:
    return (
        f"CURRENT PROBE: {probe.id} (open question)\n\n"
        f"TASK: {probe.prompt}\n\n"
        "Ask the question naturally and give the student room to think. "
        "If they answer with only facts and no opinion, redirect: "
        f"'That is good background — but what is YOUR view, {name}?' "
        "If they answer in one or two words, ask them to elaborate.\n\n"
        f"{_scaffold_line(name, probe)}"
        f"AVAILABLE NEXT PROBES: {available}\n\n"
        f"{_ACCEPTANCE_RULE.format(name=name)}\n"
        f"{_DISENGAGE_RULE.format(name=name)}\n"
        f"{_ATTEMPT_CAP_RULE.format(name=name)}\n"
        f"{_ADVANCE_INSTRUCTIONS.format(name=name)}"
    )


def _example_primitive(name: str, probe: ProbeNode, available: list[str]) -> str:
    return (
        f"CURRENT PROBE: {probe.id} (real-world example)\n\n"
        f"TASK: {probe.prompt}\n\n"
        f"Ask {name} for a CONCRETE example from their own life, school, "
        "city, or news. Push back on vague answers — if they say 'people do "
        "things', ask 'which people? what things?'. A good example names a "
        "place, a person or group, and a specific action.\n\n"
        f"{_scaffold_line(name, probe)}"
        f"AVAILABLE NEXT PROBES: {available}\n\n"
        f"{_ACCEPTANCE_RULE.format(name=name)}\n"
        f"{_DISENGAGE_RULE.format(name=name)}\n"
        f"{_ATTEMPT_CAP_RULE.format(name=name)}\n"
        f"{_ADVANCE_INSTRUCTIONS.format(name=name)}"
    )


def _counter_primitive(name: str, probe: ProbeNode, available: list[str]) -> str:
    return (
        f"CURRENT PROBE: {probe.id} (counterargument)\n\n"
        f"TASK: {probe.prompt}\n\n"
        f"Acknowledge {name}'s position briefly, then ask them to steelman "
        "the opposing view: 'What is the strongest argument someone who "
        "disagrees with you would make?' Do NOT accept dismissals like "
        "'that argument is wrong' — push back: 'Try to put yourself in "
        "their shoes. What would they say is the reason?' "
        "Only accept the answer when the student has genuinely articulated "
        "the other side, even if they still reject it.\n\n"
        f"{_scaffold_line(name, probe)}"
        f"AVAILABLE NEXT PROBES: {available}\n\n"
        f"{_ACCEPTANCE_RULE.format(name=name)}\n"
        f"{_DISENGAGE_RULE.format(name=name)}\n"
        f"{_ATTEMPT_CAP_RULE.format(name=name)}\n"
        f"{_ADVANCE_INSTRUCTIONS.format(name=name)}"
    )


def _comparison_primitive(name: str, probe: ProbeNode, available: list[str]) -> str:
    return (
        f"CURRENT PROBE: {probe.id} (comparison)\n\n"
        f"TASK: {probe.prompt}\n\n"
        f"Invite {name} to compare two scales, groups, or approaches side "
        "by side. Encourage contrast language: 'however', 'on the other "
        "hand', 'whereas'. If they just restate their earlier view, push "
        "them to put the two options against each other directly. "
        "Acknowledge briefly when they use a contrast connector naturally.\n\n"
        f"{_scaffold_line(name, probe)}"
        f"AVAILABLE NEXT PROBES: {available}\n\n"
        f"{_ACCEPTANCE_RULE.format(name=name)}\n"
        f"{_DISENGAGE_RULE.format(name=name)}\n"
        f"{_ATTEMPT_CAP_RULE.format(name=name)}\n"
        f"{_ADVANCE_INSTRUCTIONS.format(name=name)}"
    )


def _precision_primitive(name: str, probe: ProbeNode, available: list[str]) -> str:
    return (
        f"CURRENT PROBE: {probe.id} (precision)\n\n"
        f"TASK: {probe.prompt}\n\n"
        f"Pick ONE vague word {name} has used (e.g. 'chaos', 'a lot', "
        "'people', 'stuff') and ask them to define it precisely. "
        "'You said X — what exactly do you mean by that?' "
        "Do not accept circular definitions ('chaos means a lot of chaos').\n\n"
        f"{_scaffold_line(name, probe)}"
        f"AVAILABLE NEXT PROBES: {available}\n\n"
        f"{_ACCEPTANCE_RULE.format(name=name)}\n"
        f"{_DISENGAGE_RULE.format(name=name)}\n"
        f"{_ATTEMPT_CAP_RULE.format(name=name)}\n"
        f"{_ADVANCE_INSTRUCTIONS.format(name=name)}"
    )


def _synthesis_primitive(name: str, probe: ProbeNode, available: list[str]) -> str:
    return (
        f"CURRENT PROBE: {probe.id} (synthesis)\n\n"
        f"TASK: {probe.prompt}\n\n"
        f"Ask {name} to pull their answers together: 'Putting everything "
        "you have said today into one sentence, what is your final view?' "
        "Do not let them simply repeat one earlier answer — push for a "
        "synthesis that touches at least two of the ideas already raised.\n\n"
        f"{_scaffold_line(name, probe)}"
        f"AVAILABLE NEXT PROBES: {available}\n\n"
        f"{_ACCEPTANCE_RULE.format(name=name)}\n"
        f"{_DISENGAGE_RULE.format(name=name)}\n"
        f"{_ATTEMPT_CAP_RULE.format(name=name)}\n"
        f"{_ADVANCE_INSTRUCTIONS.format(name=name)}"
    )


PrimitiveFn = Callable[[str, ProbeNode, list[str]], str]

PRIMITIVES: dict[str, PrimitiveFn] = {
    "question":   _question_primitive,
    "example":    _example_primitive,
    "counter":    _counter_primitive,
    "comparison": _comparison_primitive,
    "precision":  _precision_primitive,
    "synthesis":  _synthesis_primitive,
}

ALLOWED_PROBE_TYPES: tuple[str, ...] = tuple(PRIMITIVES.keys())


def render_probe_instructions(
    probe: ProbeNode, name: str, available_next: list[str]
) -> str:
    """Render the per-probe instruction body using the matching primitive."""
    fn = PRIMITIVES.get(probe.type, _question_primitive)
    return fn(name, probe, available_next)
