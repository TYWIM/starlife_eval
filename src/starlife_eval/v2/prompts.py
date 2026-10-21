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


_ANSWER_LINE = re.compile(r"answer\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LETTER = re.compile(r"\b([A-Z])\b")


def parse_choice(raw: str, n_choices: int) -> int | None:
    """Extract a 0-based choice index from a model reply, or None if unparsable."""
    if not raw:
        return None
    text = raw.strip()
    m = _ANSWER_LINE.search(text)
    if m:
        text = m.group(1).strip()
    # strip leading punctuation / "option"
    text = re.sub(r"^(option|choice)\s*", "", text, flags=re.IGNORECASE).strip(" .;:")
    if len(text) == 1 and text.isalpha():
        idx = ord(text.upper()) - ord("A")
        return idx if 0 <= idx < n_choices else None
    m = re.search(r"\b([A-Z])\b", text, re.IGNORECASE)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        return idx if 0 <= idx < n_choices else None
    return None


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
