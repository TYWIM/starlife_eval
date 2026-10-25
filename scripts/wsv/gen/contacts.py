"""Visual-QA contact sheets (local artefacts, not part of the shipped dataset)."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE_W = 480
TILE_H = 360
LABEL_H = 26
COLUMNS = 4
MARGIN = 8
TITLE_H = 34


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def make_contact_sheet(items: list[tuple[Path, str]], out_path: Path, title: str | None = None) -> Path:
    """Compose a grid contact sheet of ``(image_path, label)`` tiles."""
    if not items:
        raise ValueError("contact sheet needs at least one tile")
    rows = math.ceil(len(items) / COLUMNS)
    width = MARGIN * 2 + COLUMNS * (TILE_W + MARGIN)
    height = MARGIN * 2 + (TITLE_H if title else 0) + rows * (TILE_H + LABEL_H + MARGIN)
    canvas = Image.new("RGB", (width, height), (26, 28, 32))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((MARGIN, MARGIN + 6), title, font=_font(18), fill=(235, 235, 235))
    top = MARGIN + (TITLE_H if title else 0)
    label_font = _font(13)
    for index, (path, label) in enumerate(items):
        column = index % COLUMNS
        row = index // COLUMNS
        x = MARGIN + column * (TILE_W + MARGIN)
        y = top + row * (TILE_H + LABEL_H + MARGIN)
        with Image.open(path) as source:
            tile = source.convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS)
        canvas.paste(tile, (x, y))
        draw.rectangle([x, y, x + TILE_W - 1, y + TILE_H - 1], outline=(72, 78, 88))
        draw.text((x + 3, y + TILE_H + 5), label, font=label_font, fill=(222, 224, 228))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def _select_rows(rows: list[dict], preference: list[tuple[str, int, str]], count: int) -> list[dict]:
    """Pick ``count`` rows, honouring a per-label preference list, then filling in order."""
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        meta = row.get("meta") or {}
        label = str(meta.get(preference[0][2]) if preference else "")
        by_label.setdefault(label, []).append(row)
    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    for label, wanted, _key in preference:
        for row in by_label.get(label, [])[:wanted]:
            if row["task_id"] not in chosen_ids:
                chosen.append(row)
                chosen_ids.add(row["task_id"])
    for row in rows:
        if len(chosen) >= count:
            break
        if row["task_id"] not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(row["task_id"])
    return chosen[:count]


def t5_sheet_items(rows: list[dict], data_root: Path, count: int = 6) -> list[tuple[Path, str]]:
    """6 T5 tasks x (before, after) = 12 tiles covering all change classes."""
    preference = [("appeared", 2, "change_class"), ("disappeared", 1, "change_class"),
                  ("changed", 2, "change_class"), ("stayed_same", 1, "change_class")]
    chosen = _select_rows(rows, preference, count)
    items: list[tuple[Path, str]] = []
    for row in chosen:
        task_id = str(row["task_id"])
        change_class = str(row["meta"]["change_class"])
        for frame_name, relative in (("before", row["images"][0]), ("after", row["images"][1])):
            items.append((data_root / relative, f"{task_id} {frame_name} [{change_class}]"))
    return items


def t7_sheet_items(rows: list[dict], data_root: Path, count: int = 6) -> list[tuple[Path, str]]:
    """6 T7 tasks x (frame1, frame2) = 12 tiles covering all directions."""
    preference = [(direction, 1, "direction") for direction in ("left", "right", "above", "below")]
    chosen = _select_rows(rows, preference, count)
    items: list[tuple[Path, str]] = []
    for row in chosen:
        task_id = str(row["task_id"])
        direction = str(row["meta"]["direction"])
        for frame_name, relative in (("f1", row["images"][0]), ("f2", row["images"][1])):
            items.append((data_root / relative, f"{task_id} {frame_name} [{direction}]"))
    return items


__all__ = ["make_contact_sheet", "t5_sheet_items", "t7_sheet_items"]
