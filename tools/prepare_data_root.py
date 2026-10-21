"""把服务器工作区的真实产物整理成 ARCHITECTURE.md §1 的标准 data-root 布局。

不复制大文件（默认 symlink），不跑渲染、不写死机器路径：
  <data-root>/
    dataset_spatial_v2 -> <src>/dataset_spatial_v2
    features/spatial_v2.npz          -> <src>/features/spatial_v2.npz
    features/spatial_v2.meta.json    -> <src>/features/spatial_v2.meta.json
    checkpoints/pilot_v2g            -> <src>/checkpoints/pilot_v2g
    cloud/eval_cloud_raw.jsonl       -> <src>/eval_cloud_raw.jsonl
    cloud/eval_compare_v2g.json      -> <src>/eval_compare_v2g.json
    bench/bench_pipeline.json        -> <src>/bench_pipeline.json

用法（服务器上）：
    python tools/prepare_data_root.py --src <data_root> --dst <data_root>/starlife_eval_data
或本地（复制模式）：
    python tools/prepare_data_root.py --src D:/data/ws --dst D:/data/eval_root --copy
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ITEMS = [
    ("dataset_spatial_v2", "dataset_spatial_v2", "dir"),
    ("features/spatial_v2.npz", "features/spatial_v2.npz", "file"),
    ("features/spatial_v2.meta.json", "features/spatial_v2.meta.json", "file"),
    ("checkpoints/pilot_v2g", "checkpoints/pilot_v2g", "dir"),
    ("checkpoints/pilot_v2", "checkpoints/pilot_v2", "dir"),
    ("eval_cloud_raw.jsonl", "cloud/eval_cloud_raw.jsonl", "file"),
    ("eval_compare_v2g.json", "cloud/eval_compare_v2g.json", "file"),
    ("eval_compare_v2.json", "cloud/eval_compare_v2.json", "file"),
    ("bench_pipeline.json", "bench/bench_pipeline.json", "file"),
]


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copied"
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return "linked"
    except (OSError, NotImplementedError):
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copied(fallback)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始产物目录（服务器上是工作区根目录）")
    ap.add_argument("--dst", required=True, help="标准 data-root 目标目录")
    ap.add_argument("--copy", action="store_true", help="复制而不是 symlink")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"src 不存在: {src}")
    missing = []
    for rel_src, rel_dst, _kind in ITEMS:
        s = src / rel_src
        if not s.exists():
            missing.append(rel_src)
            continue
        how = link_or_copy(s, dst / rel_dst, args.copy)
        print(f"{how:16s} {rel_dst}")
    if missing:
        print("缺失（可接受，评测会标 N/A）:", ", ".join(missing))
    print(f"data-root ready: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
