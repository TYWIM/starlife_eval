"""Fetch the official releases used by the v2 benchmark and convert them to unified task JSONL.

Usage:
    python tools/fetch_benchmarks.py --list
    python tools/fetch_benchmarks.py --dataset mindedit --data data
    python tools/fetch_benchmarks.py --dataset spatialmqa --data data --convert

Nothing downloaded here is redistributed; see datasets/registry.json for licenses.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "datasets" / "registry.json"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url} -> {dst}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "starlife-eval-v2-fetch"})
    with urllib.request.urlopen(req, timeout=300) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="download official spatial-reasoning benchmarks")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--data", default="data", help="output root for raw + unified task files")
    ap.add_argument("--convert", action="store_true", help="also convert the JSONL release into unified tasks")
    ap.add_argument("--raw", default="", help="path to an already downloaded JSONL release")
    args = ap.parse_args(argv)

    reg = load_registry()
    entries = {e["id"]: e for e in reg["datasets"]}
    if args.list:
        for e in reg["datasets"]:
            print(f"{e['id']:14s} {e['capability']:5s} n={e.get('expected_size')} "
                  f"status={e.get('status')}  license={e.get('license')}")
        return 0
    if args.dataset not in entries:
        print(f"unknown dataset {args.dataset!r}; --list to see options", file=sys.stderr)
        return 2

    entry = entries[args.dataset]
    data_root = Path(args.data)
    raw_path = Path(args.raw) if args.raw else None
    if raw_path is None:
        home = entry.get("home")
        if not home:
            print(f"{args.dataset}: no public download URL yet (status: {entry.get('status')})", file=sys.stderr)
            return 3
        if "huggingface.co/datasets" in home:
            print("Hugging Face datasets are downloaded with: pip install datasets && "
                  f"python -c \"from datasets import load_dataset; load_dataset('{home.split('/datasets/')[1]}')\"")
            return 4
        print(f"{args.dataset}: manual download required ({home}); pass --raw <file> after downloading",
              file=sys.stderr)
        return 4

    if args.convert:
        from starlife_eval.v2.converters import convert

        out = data_root / "tasks" / f"{args.dataset}.jsonl"
        n = convert(args.dataset, raw_path, out)
        print(f"converted {n} tasks -> {out}")
    else:
        print(f"raw: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
