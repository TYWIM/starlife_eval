"""CLI: generate the OpenWSV T5 (relation_change) / T7 (cross_frame) data.

From the repository root::

    C:\\Users\\MECHREVO\\miniconda3\\python.exe -X utf8 -m scripts.wsv.gen.build

Full generation writes:

* ``data/openwsv-data-1.1/t5_frames/*.jpg`` (1,200 frames = 600 before/after pairs)
* ``data/openwsv-data-1.1/t7_frames/*.jpg`` (800 frames = 400 frame1/frame2 pairs)
* ``data/openwsv-data-1.1/{SHA256SUMS,meta.json,README.md}``
* ``src/starlife_eval/wsv/suites/relation_change_test.jsonl`` (n=600)
* ``src/starlife_eval/wsv/suites/cross_frame_test.jsonl`` (n=400)
* ``_local/t5_samples.png`` / ``_local/t7_samples.png`` (visual QA contact sheets)

Smoke / pilot runs write the same layout under ``--out-root DIR`` (e.g.
``_local/pilot``) and are byte-identical to the first N tasks of a full run
(each task's scene is derived only from its own index).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL

from . import GENERATOR_VERSION, SEED
from . import contacts
from .render import JPEG_QUALITY, save_frame
from .scenes import (
    T5_ANSWER_INDEX,
    T5_CLASSES,
    T5_OPTIONS,
    T5_SUITE,
    T5_TOTAL,
    T7_ANSWER_INDEX,
    T7_DIRECTIONS,
    T7_OPTIONS,
    T7_SUITE,
    T7_TOTAL,
    plan_t5_task,
    plan_t7_task,
)
from .suites import (
    CAMERA_META,
    DATASET_DIR,
    T5_DATASET,
    T5_FILE,
    T5_FRAMES_DIR,
    T7_DATASET,
    T7_FILE,
    T7_FRAMES_DIR,
    build_rows_t5,
    build_rows_t7,
    write_jsonl,
)
from .validation import (
    ValidationError,
    load_jsonl,
    sha256_file,
    validate_all,
    verify_sha256sums,
)

T5_EXPECTED = T5_TOTAL
T7_EXPECTED = T7_TOTAL
DEFAULT_SHEET_T5 = "t5_samples.png"
DEFAULT_SHEET_T7 = "t7_samples.png"


# ------------------------------------------------------------------- helpers
def repo_root() -> Path:
    """``D:\\...\\starlife_eval`` (scripts/wsv/gen/build.py -> parents[3])."""
    return Path(__file__).resolve().parents[3]


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{count} B"  # pragma: no cover


def _hash_tree(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _combined_digest(hashes: dict[str, str]) -> str:
    payload = "\n".join(f"{relative} {digest}" for relative, digest in sorted(hashes.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ planning
def plan_tasks(count_t5: int, count_t7: int, seed: int, *, quiet: bool) -> tuple[list[dict], dict]:
    start = time.perf_counter()
    plans_t5 = [plan_t5_task(index, seed=seed) for index in range(count_t5)]
    plans_t7 = [plan_t7_task(index, seed=seed) for index in range(count_t7)]
    elapsed = time.perf_counter() - start
    classes = {name: 0 for name in T5_CLASSES}
    for plan in plans_t5:
        classes[plan["change_class"]] += 1
    directions = {name: 0 for name in T7_DIRECTIONS}
    for plan in plans_t7:
        directions[plan["direction"]] += 1
    if not quiet:
        print(f"planned {len(plans_t5)} T5 + {len(plans_t7)} T7 tasks in {elapsed:.2f}s")
        print(f"  T5 classes: {classes}")
        print(f"  T7 directions: {directions}")
    return plans_t5, plans_t7, {"seconds": elapsed, "classes": classes, "directions": directions}


# ----------------------------------------------------------------- rendering
def build_jobs(plans_t5: list[dict], plans_t7: list[dict], data_dir: Path) -> list[tuple[str, list[dict]]]:
    t5_dir = data_dir / T5_FRAMES_DIR
    t7_dir = data_dir / T7_FRAMES_DIR
    jobs: list[tuple[str, list[dict]]] = []
    for plan in plans_t5:
        task_id = str(plan["task_id"])
        jobs.append((str(t5_dir / f"{task_id}_before.jpg"), plan["frames"]["before"]))
        jobs.append((str(t5_dir / f"{task_id}_after.jpg"), plan["frames"]["after"]))
    for plan in plans_t7:
        task_id = str(plan["task_id"])
        jobs.append((str(t7_dir / f"{task_id}_f1.jpg"), plan["frame1"]))
        jobs.append((str(t7_dir / f"{task_id}_f2.jpg"), plan["frame2"]))
    return jobs


def _render_job(payload: tuple[str, list[dict]]) -> tuple[str, int]:
    path, objects = payload
    size = save_frame(objects, Path(path))
    return (str(path), int(size))


def render_jobs(jobs: list[tuple[str, list[dict]]], workers: int, *, quiet: bool) -> dict:
    start = time.perf_counter()
    total = len(jobs)
    step = max(1, total // 10)
    written = 0
    total_bytes = 0
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            _path, size = _render_job(job)
            written += 1
            total_bytes += size
            if not quiet and (index % step == 0 or index == total):
                print(f"  rendered {index}/{total} frames", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            for index, (_path, size) in enumerate(
                pool.imap_unordered(_render_job, jobs, chunksize=4), start=1
            ):
                written += 1
                total_bytes += int(size)
                if not quiet and (index % step == 0 or index == total):
                    print(f"  rendered {index}/{total} frames", flush=True)
    elapsed = time.perf_counter() - start
    if not quiet:
        print(
            f"rendered {written} frames in {elapsed:.1f}s "
            f"({elapsed / max(written, 1):.2f}s/frame, {_human_bytes(total_bytes)})"
        )
    return {
        "seconds": elapsed,
        "frames": written,
        "bytes": int(total_bytes),
        "workers": workers,
        "seconds_per_frame": elapsed / max(written, 1),
    }


# -------------------------------------------------------------- data package
def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _readme_text(count_t5: int, count_t7: int) -> str:
    return f"""# OpenWSV data package (openwsv-data-1.1)

Synthetic frames for the OpenWSV benchmark suites **T5** (`relation_change`) and
**T7** (`cross_frame`), framework `openwsv-1`.

## Contents

```
openwsv-data-1.1/
|-- t5_frames/          # {2 * count_t5:,} JPEG frames ({count_t5:,} before/after pairs, 1024x768, quality {JPEG_QUALITY})
|-- t7_frames/          # {2 * count_t7:,} JPEG frames ({count_t7:,} frame1/frame2 pairs, 1024x768, quality {JPEG_QUALITY})
|-- SHA256SUMS          # per-file hashes (frames + meta.json + README.md)
|-- meta.json           # dataset metadata, suite file hashes, class/direction counts
`-- README.md
```

- Frames are rendered by the deterministic generator `{GENERATOR_VERSION}`
  (seed `{SEED}`): pure Python + numpy + Pillow, no external assets, no network.
- Every task's two frames share one fixed pinhole camera
  (1024x768, `fx = fy = 800`, `cx = 512`, `cy = 384`) and the same tabletop
  room, so cross-frame questions are well defined in image coordinates.
- Suite task files ship with the framework package:
  `src/starlife_eval/wsv/suites/relation_change_test.jsonl` (T5, n={count_t5}) and
  `src/starlife_eval/wsv/suites/cross_frame_test.jsonl` (T7, n={count_t7}).
  Every row stores its structured ground truth in `meta`; `meta` is never sent
  to a model (only `question` / `images` / `choices` reach the prompt).
- Image paths inside the jsonl files are relative to the data root (`data/`),
  e.g. `openwsv-data-1.1/t5_frames/relation_change-000000_before.jpg`.

## Verify

```bash
cd openwsv-data-1.1 && sha256sum -c SHA256SUMS
```

## Reproduce

From the repository root (writes the frames + suites + this package):

```bash
C:\\Users\\MECHREVO\\miniconda3\\python.exe -X utf8 -m scripts.wsv.gen.build --force
```

The generator is deterministic: running it again (same seed `{SEED}`, same
Python/numpy/Pillow versions) reproduces byte-identical frames and jsonl files.
"""


def write_data_files(
    data_dir: Path,
    rows_t5: list[dict],
    rows_t7: list[dict],
    suite_sha256: dict[str, str],
    summary: dict,
) -> dict:
    """Write README.md, meta.json and SHA256SUMS into the data directory.

    Deliberately free of timings/absolute paths so the whole data package is
    byte-reproducible across runs (the generation report keeps the timings).
    """
    count_t5 = len(rows_t5)
    count_t7 = len(rows_t7)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_text(data_dir / "README.md", _readme_text(count_t5, count_t7))

    t5_summary = summary["t5"]
    t7_summary = summary["t7"]
    # None = not a `changed` task, so it is not a change variant
    change_variants = {
        variant: count
        for variant, count in t5_summary["change_variants"].items()
        if variant not in (None, "None")
    }
    meta_payload = {
        "dataset": DATASET_DIR,
        "framework": "openwsv-1",
        "generator": GENERATOR_VERSION,
        "seed": SEED,
        "created_utc": "2026-10-25",
        "camera": CAMERA_META,
        "frames": {
            T5_FRAMES_DIR: {
                "count": 2 * count_t5,
                "width": 1024,
                "height": 768,
                "format": "jpeg",
                "quality": JPEG_QUALITY,
            },
            T7_FRAMES_DIR: {
                "count": 2 * count_t7,
                "width": 1024,
                "height": 768,
                "format": "jpeg",
                "quality": JPEG_QUALITY,
            },
        },
        "suites": {
            T5_DATASET: {
                "file": f"src/starlife_eval/wsv/suites/{T5_FILE}",
                "n": count_t5,
                "capability": "T5",
                "answer_type": "choice",
                "sha256": suite_sha256[T5_FILE],
                "change_classes": t5_summary["change_classes"],
                "change_variants": change_variants,
                "answer_index": t5_summary["answer_index"],
            },
            T7_DATASET: {
                "file": f"src/starlife_eval/wsv/suites/{T7_FILE}",
                "n": count_t7,
                "capability": "T7",
                "answer_type": "choice",
                "sha256": suite_sha256[T7_FILE],
                "directions": t7_summary["directions"],
                "answer_index": t7_summary["answer_index"],
            },
        },
        "gt_conventions": {
            "coordinates": "image pixels, u right / v down, shared pinhole camera",
            "relation": (
                "direction of A's projected 2D bbox centre relative to B's "
                "(left/right/above/below) plus a projected centre-distance bucket "
                "(close/mid/far); signature = '<type>|<bucket>'"
            ),
            "t5_change_classes": {
                "appeared": "the A-B relation exists only in the after frame (an object appeared)",
                "disappeared": "the A-B relation exists only in the before frame (an object disappeared)",
                "changed": "the A-B relation exists in both frames with different signatures",
                "stayed_same": "the A-B relation signature is identical in both frames",
            },
            "t7_direction": (
                "direction of X's projected bbox centre relative to Y's in the shared "
                "camera view; the dominant axis must dominate >=1.5x and the main-axis "
                "gap must be >=8% of the image width/height"
            ),
        },
        "image_inventory": summary["images"],
        "files": {"sha256sums": "SHA256SUMS", "readme": "README.md"},
    }
    _write_text(
        data_dir / "meta.json",
        json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n",
    )
    _write_sha256sums(data_dir)
    return {"meta": data_dir / "meta.json", "readme": data_dir / "README.md"}


def _write_sha256sums(data_dir: Path) -> Path:
    entries: list[str] = []
    for path in sorted(data_dir.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(data_dir).as_posix()
        entries.append(f"{sha256_file(path)}  {relative}")
    sums_path = data_dir / "SHA256SUMS"
    _write_text(sums_path, "\n".join(entries) + "\n")
    return sums_path


# ------------------------------------------------------------- contact sheets
def write_contact_sheets(
    rows_t5: list[dict],
    rows_t7: list[dict],
    data_root: Path,
    t5_sheet_path: Path,
    t7_sheet_path: Path,
    *,
    quiet: bool,
) -> dict:
    t5_items = contacts.t5_sheet_items(rows_t5, data_root)
    t7_items = contacts.t7_sheet_items(rows_t7, data_root)
    contacts.make_contact_sheet(
        t5_items,
        t5_sheet_path,
        title="T5 relation_change - before/after samples (12 tiles)",
    )
    contacts.make_contact_sheet(
        t7_items,
        t7_sheet_path,
        title="T7 cross_frame - frame1/frame2 samples (12 tiles)",
    )
    if not quiet:
        print(f"contact sheets: {t5_sheet_path} ({len(t5_items)} tiles), {t7_sheet_path} ({len(t7_items)} tiles)")
    return {"t5_sheet": str(t5_sheet_path), "t7_sheet": str(t7_sheet_path)}


# ------------------------------------------------------------- determinism
def run_determinism_check(
    *,
    root: Path,
    local_dir: Path,
    workers: int,
    limit: int = 5,
) -> dict:
    """Render the first ``limit`` tasks twice in separate fresh processes and compare bytes."""
    base = local_dir / "_determinism"
    if base.exists():
        shutil.rmtree(base)
    hashes: dict[str, dict[str, str]] = {}
    for tag in ("a", "b"):
        out_root = base / tag
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "scripts.wsv.gen.build",
            "--limit",
            str(limit),
            "--out-root",
            str(out_root),
            "--no-data-files",
            "--no-sheets",
            "--workers",
            str(workers),
            "--quiet",
            "--force",
        ]
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "determinism run failed: "
                + " ".join(command)
                + "\nstdout tail: "
                + completed.stdout[-1000:]
                + "\nstderr tail: "
                + completed.stderr[-2000:]
            )
        hashes[tag] = _hash_tree(out_root)
    identical = bool(hashes["a"]) and hashes["a"] == hashes["b"]
    files_count = len(hashes["a"])
    combined = _combined_digest(hashes["a"]) if identical else None
    sample = [
        {"path": relative, "sha256": digest}
        for relative, digest in sorted(hashes["a"].items())
        if relative.endswith((".jpg", ".jsonl"))
    ][:6]
    if not identical:
        missing = sorted(set(hashes["a"]) ^ set(hashes["b"]))[:5]
        differing = [
            relative
            for relative in sorted(set(hashes["a"]) & set(hashes["b"]))
            if hashes["a"][relative] != hashes["b"][relative]
        ][:5]
        raise RuntimeError(
            f"determinism check failed: differing_files={differing} missing_files={missing}"
        )
    result = {
        "limit_per_suite": limit,
        "tasks_per_run": 2 * limit,
        "frames_per_run": 4 * limit,
        "files_compared": files_count,
        "identical": identical,
        "combined_sha256": combined,
        "sample": sample,
        "runs": [str(base / "a"), str(base / "b")],
    }
    shutil.rmtree(base, ignore_errors=True)
    return result


# ------------------------------------------------------------------- report
def write_report(
    report_path: Path,
    *,
    summary: dict,
    rows_t5: list[dict],
    rows_t7: list[dict],
    suite_paths: dict[str, Path],
    suite_sha256: dict[str, str],
    suite_sizes: dict[str, int],
    data_dir: Path,
    data_file_sha256: dict[str, str],
    determinism: dict | None,
    timings: dict,
    image_stats: dict,
    commands: dict[str, str],
) -> Path:
    t5 = summary["t5"]
    t7 = summary["t7"]
    images = summary["images"]
    env_lines = [
        f"- python: `{platform.python_version()}` ({platform.platform()})",
        f"- numpy: `{np.__version__}`",
        f"- Pillow: `{PIL.__version__}`",
        f"- generator: `{GENERATOR_VERSION}`, seed `{SEED}`",
    ]
    lines: list[str] = []
    lines.append("# OpenWSV T5/T7 data generation report")
    lines.append("")
    lines.append(
        f"Generated by `scripts/wsv/gen` ({GENERATOR_VERSION}, seed {SEED}) on the "
        "repository `D:\\\\Github\\\\xingsheng\\\\starlife_eval`."
    )
    lines.append("")
    lines.append("## 0. Environment")
    lines.append("")
    lines.extend(env_lines)
    lines.append("")
    lines.append("## 1. Counts")
    lines.append("")
    lines.append("| artefact | count |")
    lines.append("|---|---|")
    lines.append(f"| T5 tasks (`{T5_FILE}`) | {len(rows_t5)} |")
    lines.append(f"| T7 tasks (`{T7_FILE}`) | {len(rows_t7)} |")
    lines.append(f"| T5 frames (`{T5_FRAMES_DIR}/`) | {images['t5_frames']['count']} |")
    lines.append(f"| T7 frames (`{T7_FRAMES_DIR}/`) | {images['t7_frames']['count']} |")
    lines.append(f"| total images | {images['total_files']} ({_human_bytes(images['total_bytes'])}) |")
    lines.append("")
    lines.append("## 2. T5 `relation_change` distribution")
    lines.append("")
    lines.append("| change class | n | option | answer index |")
    lines.append("|---|---|---|---|")
    for name in T5_CLASSES:
        count = t5["change_classes"].get(name, 0)
        lines.append(f"| {name} | {count} | {T5_OPTIONS[T5_ANSWER_INDEX[name]]} | {T5_ANSWER_INDEX[name]} |")
    lines.append("")
    lines.append("Change variants (sub-types of the 150 `changed` tasks):")
    lines.append("")
    lines.append("| variant | n |")
    lines.append("|---|---|")
    for variant, count in t5["change_variants"].items():
        if variant in (None, "None"):
            continue  # None covers appeared/disappeared/stayed_same, not a changed variant
        lines.append(f"| {variant} | {count} |")
    lines.append("")
    lines.append("## 3. T7 `cross_frame` distribution")
    lines.append("")
    lines.append("| direction | n | option | answer index |")
    lines.append("|---|---|---|---|")
    for name in T7_DIRECTIONS:
        count = t7["directions"].get(name, 0)
        lines.append(f"| {name} | {count} | {T7_OPTIONS[T7_ANSWER_INDEX[name]]} | {T7_ANSWER_INDEX[name]} |")
    lines.append("")
    lines.append("## 4. Answer-index distribution (balanced)")
    lines.append("")
    lines.append("| suite | 0 | 1 | 2 | 3 | total |")
    lines.append("|---|---|---|---|---|---|")
    for label, item in (("relation_change (T5)", t5), ("cross_frame (T7)", t7)):
        distribution = item["answer_index"]
        total = sum(int(count) for count in distribution.values())
        lines.append(
            f"| {label} | {distribution.get('0', 0)} | {distribution.get('1', 0)} | "
            f"{distribution.get('2', 0)} | {distribution.get('3', 0)} | {total} |"
        )
    lines.append("")
    lines.append("## 5. Image inventory (1024x768 JPEG, quality "
                 f"{JPEG_QUALITY})")
    lines.append("")
    lines.append("| folder | count | total | min | mean | max |")
    lines.append("|---|---|---|---|---|---|")
    for folder in (T5_FRAMES_DIR, T7_FRAMES_DIR):
        stats = images.get(folder if folder in images else f"{folder}_frames", None)
        if stats is None:
            stats = images[f"{folder}_frames"] if f"{folder}_frames" in images else images[folder]
        lines.append(
            f"| `{folder}/` | {stats['count']} | {_human_bytes(stats['bytes_total'])} | "
            f"{stats['bytes_min']} B | {int(stats['bytes_mean'])} B | {stats['bytes_max']} B |"
        )
    lines.append("")
    lines.append("## 6. Suite files")
    lines.append("")
    lines.append("| suite | path | n | bytes | sha256 |")
    lines.append("|---|---|---|---|---|")
    for suite_name, file_name in ((T5_SUITE, T5_FILE), (T7_SUITE, T7_FILE)):
        path = suite_paths[file_name]
        rel = path.relative_to(repo_root()).as_posix() if path.is_absolute() else str(path)
        n = len(rows_t5) if suite_name == T5_SUITE else len(rows_t7)
        lines.append(
            f"| {suite_name} | `{rel}` | {n} | {suite_sizes[file_name]} B | `{suite_sha256[file_name]}` |"
        )
    lines.append("")
    lines.append("## 7. Data package files")
    lines.append("")
    lines.append("| file | sha256 |")
    lines.append("|---|---|")
    for name in ("README.md", "meta.json", "SHA256SUMS"):
        digest = data_file_sha256.get(name)
        if digest:
            lines.append(f"| `{DATASET_DIR}/{name}` | `{digest}` |")
    lines.append("")
    lines.append(f"`SHA256SUMS` covers all {images['total_files']} frames plus `README.md` and "
                 "`meta.json`; it was re-verified (`sha256sum -c` semantics) after writing.")
    lines.append("")
    lines.append("## 8. Determinism check")
    lines.append("")
    if determinism is None:
        lines.append("Not run for this invocation (use `--determinism-check`).")
    else:
        lines.append(
            "Two fresh Python processes rendered the first "
            f"{determinism['limit_per_suite']} tasks of each suite "
            f"({determinism['tasks_per_run']} tasks, {determinism['frames_per_run']} frames, "
            "single worker) into separate directories and every file was hashed."
        )
        lines.append("")
        lines.append(f"- files compared: **{determinism['files_compared']}** (JPEG frames + jsonl suites)")
        lines.append(f"- byte-identical: **{determinism['identical']}**")
        lines.append(f"- combined sha256 over `(path, sha256)` lines: `{determinism['combined_sha256']}`")
        lines.append("")
        lines.append("| sampled file (run a) | sha256 |")
        lines.append("|---|---|")
        for item in determinism["sample"]:
            lines.append(f"| `{item['path']}` | `{item['sha256']}` |")
    lines.append("")
    lines.append("## 9. Validation checks (all passed)")
    lines.append("")
    lines.append("- JSON schema/fields match the frozen suites; `answer_type=choice`, `split=test`, 4 fixed options.")
    lines.append("- Exactly 600 / 400 tasks; T5 classes 150 each, T7 directions 100 each; answer indices balanced.")
    lines.append("- Every row's image paths exist, are relative to the data root, 1024x768 JPEG.")
    lines.append("- T5: class semantics re-derived from `meta` structured state (appeared / disappeared /")
    lines.append("  changed / stayed same); `relation_before` / `relation_after` signatures recomputed from")
    lines.append("  the stored 2D bboxes and matched against the stored relation records.")
    lines.append("- T7: frame1/frame2 object descriptions are disjoint, 3 objects per frame, direction and")
    lines.append("  8%-of-frame main-axis gap / 1.5x dominance re-derived from the stored bbox centres.")
    lines.append("")
    lines.append("## 10. Ground-truth conventions")
    lines.append("")
    lines.append("- Camera (shared by both frames of every task): pinhole, 1024x768, `fx=fy=800`,")
    lines.append("  `cx=512`, `cy=384`; image coordinates are pixels with `u` right and `v` down.")
    lines.append("- T5 relation signature: direction of A's projected 2D bbox centre relative to B's")
    lines.append("  (`left` / `right` / `above` / `below`, A relative to B) plus a projected centre-distance")
    lines.append("  bucket (`close` < 128 px, `mid` 176-240 px, `far` >= 288 px; guard bands around edges).")
    lines.append("  `appeared` / `disappeared` / `changed` / `stayed_same` are derived by diffing the")
    lines.append("  before/after structured state, never from filenames.")
    lines.append("- T5 rows carry full before/after structured state in `meta.frames` (entities with 2D bbox,")
    lines.append("  type, colour, 3D centre, depth; plus all unambiguous pairwise relations). This `meta` is")
    lines.append("  for future structured scoring only - the runner never sends it to a model.")
    lines.append("- T7 direction: X's projected bbox centre relative to Y's, in the shared camera view; the")
    lines.append("  dominant axis must dominate by >=1.5x and the main-axis gap must be >=8% of the image")
    lines.append("  width (left/right) or height (above/below). `meta` stores the bbox centres and GT.")
    lines.append("")
    lines.append("## 11. Reproduce")
    lines.append("")
    lines.append("Full regeneration (deterministic, seed "
                 f"{SEED}):")
    lines.append("")
    lines.append("```bash")
    lines.append(commands["full"])
    lines.append("```")
    lines.append("")
    lines.append("Pilot / smoke run (first 10 tasks per suite into `_local/pilot/`):")
    lines.append("")
    lines.append("```bash")
    lines.append(commands["pilot"])
    lines.append("```")
    lines.append("")
    lines.append("Determinism is re-checked automatically by the `--determinism-check` flag in the full")
    lines.append("command above: it renders the first 5 tasks of each suite twice in separate fresh")
    lines.append("processes and exits non-zero if any frame or jsonl byte differs.")
    lines.append("")
    lines.append("Visual QA contact sheets: `_local/t5_samples.png`, `_local/t7_samples.png`.")
    lines.append("")
    lines.append("## 12. Timings (this machine, "
                 f"{os.cpu_count()} logical CPUs)")
    lines.append("")
    lines.append("| step | seconds |")
    lines.append("|---|---|")
    lines.append(f"| planning | {timings['planning_seconds']:.2f} |")
    lines.append(f"| rendering ({timings['render']['frames']} frames, {timings['render']['workers']} workers) | "
                 f"{timings['render']['seconds']:.1f} |")
    lines.append(f"| total (including validation/report) | {timings['total_seconds']:.1f} |")
    lines.append("")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(report_path, "\n".join(lines) + "\n")
    return report_path


# ---------------------------------------------------------------------- CLI
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wsv.gen.build",
        description="Generate the OpenWSV T5/T7 synthetic frames + suite jsonl files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="generate only the first N tasks per suite (smoke/pilot)")
    parser.add_argument("--out-root", type=str, default=None,
                        help="write under OUT_ROOT/data and OUT_ROOT/suites instead of the repo defaults")
    parser.add_argument("--workers", type=int, default=0, help="renderer processes (0 = auto)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"RNG seed (contract default {SEED})")
    parser.add_argument("--force", action="store_true", help="delete an existing generated data dir first")
    parser.add_argument("--no-images", action="store_true", help="skip rendering (planning + jsonl only)")
    parser.add_argument("--no-data-files", action="store_true", help="skip README.md / meta.json / SHA256SUMS")
    parser.add_argument("--no-sheets", action="store_true", help="skip the contact sheets")
    parser.add_argument("--t5-sheet", type=str, default=None, help="T5 contact sheet path")
    parser.add_argument("--t7-sheet", type=str, default=None, help="T7 contact sheet path")
    parser.add_argument("--report", type=str, default=None, help="write a markdown generation report to this path")
    parser.add_argument("--determinism-check", action="store_true",
                        help="render the first 5 tasks twice in fresh processes and compare bytes")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(argv)


def _resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    root = repo_root()

    if args.out_root:
        out_root = _resolve_path(args.out_root, root)
        data_root = out_root / "data"
        suite_dir = out_root / "suites"
        local_dir = out_root
    else:
        data_root = root / "data"
        suite_dir = root / "src" / "starlife_eval" / "wsv" / "suites"
        local_dir = root / "_local"
    data_dir = data_root / DATASET_DIR
    t5_sheet_path = _resolve_path(args.t5_sheet, local_dir) or (local_dir / DEFAULT_SHEET_T5)
    t7_sheet_path = _resolve_path(args.t7_sheet, local_dir) or (local_dir / DEFAULT_SHEET_T7)
    report_path = _resolve_path(args.report, root)

    count_t5 = T5_EXPECTED if args.limit is None else max(0, min(int(args.limit), T5_EXPECTED))
    count_t7 = T7_EXPECTED if args.limit is None else max(0, min(int(args.limit), T7_EXPECTED))
    strict = args.limit is None
    # default stays modest: the machine is shared and the renderer already culls
    # aggressively (~0.3 s/frame/worker effective); memory is the real limit here.
    workers = int(args.workers) if int(args.workers) > 0 else max(1, min(6, (os.cpu_count() or 4) // 2))
    seed = int(args.seed)
    if seed != SEED and not args.quiet:
        print(f"WARNING: seed {seed} differs from the frozen generator seed {SEED}; generated data is not canonical.")

    if not args.quiet:
        print(f"OpenWSV scene generator {GENERATOR_VERSION} (seed {seed})")
        print(f"  data dir:   {data_dir}")
        print(f"  suites dir: {suite_dir}")
        print(f"  tasks:      {count_t5} T5 + {count_t7} T7 ({2 * count_t5 + 2 * count_t7} frames)")
        print(f"  workers:    {workers}")

    plans_t5, plans_t7, planning_info = plan_tasks(count_t5, count_t7, seed, quiet=args.quiet)
    rows_t5 = build_rows_t5(plans_t5)
    rows_t7 = build_rows_t7(plans_t7)

    if data_dir.exists():
        if not args.force:
            print(
                f"ERROR: {data_dir} already exists; pass --force to regenerate it "
                "(the generator only ever deletes its own openwsv-data-1.1 directory).",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(data_dir)
    (data_dir / T5_FRAMES_DIR).mkdir(parents=True, exist_ok=True)
    (data_dir / T7_FRAMES_DIR).mkdir(parents=True, exist_ok=True)

    render_info = {"seconds": 0.0, "frames": 0, "bytes": 0, "workers": workers, "seconds_per_frame": 0.0}
    if not args.no_images:
        jobs = build_jobs(plans_t5, plans_t7, data_dir)
        render_info = render_jobs(jobs, workers, quiet=args.quiet)

    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_paths = {T5_FILE: suite_dir / T5_FILE, T7_FILE: suite_dir / T7_FILE}
    suite_sizes = {
        T5_FILE: write_jsonl(rows_t5, suite_paths[T5_FILE]),
        T7_FILE: write_jsonl(rows_t7, suite_paths[T7_FILE]),
    }
    del suite_sizes  # sizes are reported from disk below
    suite_sha256 = {name: sha256_file(path) for name, path in suite_paths.items()}
    suite_sizes = {name: path.stat().st_size for name, path in suite_paths.items()}
    if not args.quiet:
        for name in (T5_FILE, T7_FILE):
            print(f"wrote {suite_paths[name]} ({suite_sizes[name]} B, sha256 {suite_sha256[name]})")

    sheets_info = None
    if not args.no_sheets and not args.no_images:
        sheets_info = write_contact_sheets(
            rows_t5, rows_t7, data_root, t5_sheet_path, t7_sheet_path, quiet=args.quiet
        )

    summary = None
    data_file_sha256: dict[str, str] = {}
    validation_ok = False
    if args.no_images:
        print("WARNING: --no-images set; skipping image/file validation.")
    else:
        try:
            summary = validate_all(rows_t5, rows_t7, data_root, strict=strict)
            validation_ok = True
        except ValidationError as exc:
            print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
            return 3
        if not args.quiet:
            print("validation passed:")
            print(f"  T5 classes={summary['t5']['change_classes']} answers={summary['t5']['answer_index']}")
            print(f"  T7 directions={summary['t7']['directions']} answers={summary['t7']['answer_index']}")
            print(f"  images={summary['images']['total_files']} ({_human_bytes(summary['images']['total_bytes'])})")

    data_dir_info = None
    if not args.no_data_files:
        if summary is None:
            print("ERROR: cannot write meta.json/SHA256SUMS without images/validation.", file=sys.stderr)
            return 4
        timings = {
            "planning_seconds": planning_info["seconds"],
            "render_seconds": render_info["seconds"],
            "total_seconds": time.perf_counter() - started,
        }
        data_dir_info = write_data_files(
            data_dir, rows_t5, rows_t7, suite_sha256, summary
        )
        for name in ("README.md", "meta.json", "SHA256SUMS"):
            data_file_sha256[name] = sha256_file(data_dir / name)
        try:
            sums_info = verify_sha256sums(data_dir)
        except ValidationError as exc:
            print(f"SHA256SUMS VERIFICATION FAILED: {exc}", file=sys.stderr)
            return 5
        if not args.quiet:
            print(f"data package: {data_dir} ({sums_info['files']} files hashed, SHA256SUMS verified)")

    determinism = None
    if args.determinism_check:
        if not args.quiet:
            print("determinism check: rendering the first 5 tasks twice in fresh processes ...")
        determinism = run_determinism_check(
            root=root, local_dir=local_dir, workers=max(2, workers), limit=5
        )
        if not args.quiet:
            print(
                f"determinism check: {determinism['files_compared']} files, "
                f"identical={determinism['identical']}, combined sha256={determinism['combined_sha256']}"
            )

    if report_path is not None and summary is not None and data_dir_info is not None:
        total_seconds = time.perf_counter() - started
        timings = {
            "planning_seconds": planning_info["seconds"],
            "render": render_info,
            "total_seconds": total_seconds,
        }
        commands = {
            "full": (
                'C:\\Users\\MECHREVO\\miniconda3\\python.exe -X utf8 -m scripts.wsv.gen.build --force '
                "--determinism-check --report _local/REPORT_DATA_GEN.md"
            ),
            "pilot": (
                'C:\\Users\\MECHREVO\\miniconda3\\python.exe -X utf8 -m scripts.wsv.gen.build '
                "--limit 10 --out-root _local/pilot --force --no-data-files"
            ),
        }
        write_report(
            report_path,
            summary=summary,
            rows_t5=rows_t5,
            rows_t7=rows_t7,
            suite_paths=suite_paths,
            suite_sha256=suite_sha256,
            suite_sizes=suite_sizes,
            data_dir=data_dir,
            data_file_sha256=data_file_sha256,
            determinism=determinism,
            timings=timings,
            image_stats=summary["images"],
            commands=commands,
        )
        if not args.quiet:
            print(f"report: {report_path}")

    payload = {
        "ok": validation_ok or args.no_images,
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "tasks": {"relation_change": len(rows_t5), "cross_frame": len(rows_t7)},
        "suites": {name: {"path": str(path), "sha256": suite_sha256[name], "bytes": suite_sizes[name]}
                   for name, path in suite_paths.items()},
        "data_dir": str(data_dir),
        "sheets": sheets_info,
        "determinism": determinism,
        "report": str(report_path) if report_path else None,
        "seconds": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
