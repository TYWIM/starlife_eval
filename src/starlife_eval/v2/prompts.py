"""Prompt templates and answer parsing for v2 tasks.

Three prompt variants are used per task (round-robin by task index) so the report can measure
template sensitivity instead of silently averaging over one phrasing.
"""

from __future__ import annotations

import json
import re
from typing import Any

PROMPT_TEMPLATES = ("direct", "reasoned", "minimal")

_INSTRUCTION = {
    "choice": (
        "Answer with the letter of exactly one option. Reply with a single line: \"Answer: <letter>\"."
    ),
    "short": (
        "Answer with a short phrase on a single line: \"Answer: <your answer>\"."
    ),
    "json": (
        "Answer with one JSON object only, no prose and no code fences."
    ),
}


def build_prompt(task, template: str = "direct") -> str:
    """Build the user prompt for one task (images are passed separately)."""
    if template not in PROMPT_TEMPLATES:
        raise ValueError(f"unknown template {template!r}")
    parts: list[str] = []
    if template == "reasoned":
        parts.append("Look at the image(s) carefully, reason step by step about the spatial layout, "
                     "then give a final answer.")
    elif template == "minimal":
        parts.append("Spatial reasoning question.")
    views = task.meta.get("views")
    if views:
        labels = ", ".join(f"image {i + 1} = {v}" for i, v in enumerate(str(views).split(",")))
        parts.append(f"Images are provided in this order: {labels}.")
    parts.append(task.question.strip())
    if task.answer_type == "choice":
        letters = [chr(ord("A") + i) for i in range(len(task.choices))]
        parts.append("Options:")
        for letter, text in zip(letters, task.choices):
            parts.append(f"{letter}. {text}")
        if template == "minimal":
            parts.append("Reply with the letter only.")
    parts.append(_INSTRUCTION[task.answer_type])
    if task.answer_type == "json" and task.meta.get("schema"):
        parts.append("Expected JSON schema: " + json.dumps(task.meta["schema"], ensure_ascii=False))
    return "\n".join(parts)


_ANSWER_LINE = re.compile(r"answer\s*(?:is)?\s*[:：]?\s*\(?([A-Za-z])\)?", re.IGNORECASE)
_LETTER = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")


def _normalise(text: str) -> str:
    import unicodedata

    t = unicodedata.normalize("NFKC", text)
    t = t.replace("*", "").replace("`", "")
    return t


def parse_choice(raw: str, n_choices: int) -> int | None:
    """Extract a 0-based choice index from a model reply, or None if unparsable.

    Handles: "Answer: C", "answer is (c)", "**C**", "C.", "Option D", trailing explanations.
    When several standalone letters appear, the one nearest the answer keyword wins; otherwise the
    last one (models usually conclude with the answer).
    """
    if not raw:
        return None
    text = _normalise(raw).strip()
    m = _ANSWER_LINE.search(text)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        return idx if 0 <= idx < n_choices else None
    candidates = [ord(c.upper()) - ord("A") for c in _LETTER.findall(text)]
    candidates = [i for i in candidates if 0 <= i < n_choices]
    if not candidates:
        return None
    return candidates[-1]


def parse_short(raw: str) -> str | None:
    if not raw:
        return None
    m = _ANSWER_LINE.search(raw)
    text = (m.group(1) if m else raw).strip()
    return text or None


def parse_json_answer(raw: str) -> Any:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                return None
        return None
