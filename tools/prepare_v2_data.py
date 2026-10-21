"""Build the v2 task bundles from downloaded official releases.

Pipeline (run on a machine that has network access to the HF mirror):

    python tools/prepare_v2_data.py --raw-root data/raw --out data/v2 --dataset all --package

Layout:
    data/v2/tasks/<dataset>.jsonl          unified task files (one per dataset)
    data/v2/images/<dataset>/...           images referenced by the tasks
    data/v2/manifest.json                  counts, licenses, sha256 of every task file
    data/v2/release/<dataset>.tar          optional release bundles (tasks + images)

Nothing is uploaded automatically; see datasets/ATTRIBUTION.md for what is redistributable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from starlife_eval.v2.converters import CONVERTERS, convert  # noqa: E402
from starlife_eval.v2.schema import load_tasks  # noqa: E402

DATASETS = ("spatialmqa", "spatial457", "mindedit", "spatialdise")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_one(raw: Path, out: Path, dataset: str, images_dir: Path) -> dict:
    tasks_path = out / "tasks" / f"{dataset}.jsonl"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    n = convert(dataset, raw, tasks_path)
    tasks = load_tasks(tasks_path)
    # collect images next to the task file (first image per task is enough for release bundles)
    ds_images = images_dir / dataset
    ds_images.mkdir(parents=True, exist_ok=True)
    copied = 0
    for t in tasks:
        new_images = []
        for img in t.images:
            src = Path(img)
            if src.exists():
                dst = ds_images / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                new_images.append(str(dst))
            else:
                new_images.append(img)
        if new_images != list(t.images):
            t = type(t)(**{**t.__dict__, "images": tuple(new_images)})  # noqa: SLF001 - frozen dataclass
            # rewrite is not needed for scoring; keep original file (paths stay valid on the build host)
    return {"dataset": dataset, "tasks": n, "task_file": str(tasks_path),
            "task_sha256": sha256(tasks_path), "images_copied": copied,
            "image_dir": str(ds_images)}


def package(out: Path, datasets: list[str], release_dir: Path) -> list[dict]:
    release_dir.mkdir(parents=True, exist_ok=True)
    bundles = []
    for ds in datasets:
        tar = release_dir / f"{ds}.tar"
        items = [out / "tasks" / f"{ds}.jsonl"]
        img_dir = out / "images" / ds
        if img_dir.exists():
            items.append(img_dir)
        cmd = ["tar", "-cf", str(tar)] + [str(p) for p in items]
        subprocess.run(cmd, check=True)
        bundles.append({"dataset": ds, "bundle": str(tar), "size": tar.stat().st_size,
                        "sha256": sha256(tar)})
    return bundles


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="convert official releases into v2 task bundles")
    ap.add_argument("--raw-root", default="data/raw",
                    help="root containing <dataset>/ directories with the official release")
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--dataset", default="all", help="dataset id or 'all'")
    ap.add_argument("--package", action="store_true", help="also build release tars")
    args = ap.parse_args(argv)

    raw_root = Path(args.raw_root)
    out = Path(args.out)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    registry = json.loads((REPO / "datasets" / "registry.json").read_text(encoding="utf-8"))
    licenses = {e["id"]: e.get("license") for e in registry["datasets"]}

    results, built = [], []
    for name in names:
        raw = raw_root / name
        if not raw.exists():
            print(f"skip {name}: no raw release under {raw}", file=sys.stderr)
            continue
        try:
            info = convert_one(raw, out, name, out / "images")
            info["license"] = licenses.get(name)
            results.append(info)
            built.append(name)
            print(f"{name}: {info['tasks']} tasks, sha256={info['task_sha256'][:16]}")
        except Exception as exc:  # noqa: BLE001
            print(f"convert {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    manifest = {"version": "v2.0-draft", "datasets": results}
    if args.package and built:
        manifest["bundles"] = package(out, built, out / "release")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
