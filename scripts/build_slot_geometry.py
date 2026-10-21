"""从计划文件 + 数据集 manifest 生成 slot 级几何特征（给 change 头补位移信息）。

输出 npz（与 extract_features 的 spatial npz 的 slot 顺序一致，按 manifest 行序）：
  slot_geom      (N, 13) float32  [pres_b, pres_a, dx, dy, dz, |d|, dyaw, dtilt, droll,
                                   scale_ratio, w/W, h/H, 1/(1+area)]
  slot_geom_pair (N,) int32       所属 pair 在 manifest 中的行号
  slot_geom_slot (N,) int32       manifest 里的 slot 编号

用法：
  python build_slot_geometry.py --plan synth_spatial/work/plan.json \
      --dataset dataset_spatial_v3 --out features/spatial_v3_geom.npz
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

W, H = 1024.0, 768.0


def wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--dataset", required=True, help="含 manifest.jsonl 的数据集目录")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(pathlib.Path(args.plan).read_text(encoding="utf-8"))
    by_pair = {str(p["pair_id"]): p for p in plan}
    manifest = [json.loads(l) for l in pathlib.Path(args.dataset, "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    geoms, pair_idx, slots = [], [], []
    for pi, row in enumerate(manifest):
        pair_id = str(row["pair_id"])
        plan_row = by_pair.get(pair_id)
        if plan_row is None:
            raise SystemExit(f"plan 缺少 pair {pair_id}")
        plan_slots = {int(e["slot"]): e for e in plan_row["entities"]}
        for ent in row["entities"]:
            slot = int(ent["slot"])
            pb = plan_slots.get(slot, {})
            b, a = pb.get("before"), pb.get("after")
            if b and a:
                d = [a["pos"][i] - b["pos"][i] for i in range(3)]
                dyaw = wrap(a.get("yaw", 0.0) - b.get("yaw", 0.0))
                dtilt = wrap(a.get("tilt", 0.0) - b.get("tilt", 0.0))
                droll = wrap(a.get("roll", 0.0) - b.get("roll", 0.0))
                sr = (a.get("scale", 1.0) + 1e-6) / (b.get("scale", 1.0) + 1e-6)
            else:
                d = [0.0, 0.0, 0.0]
                dyaw = dtilt = droll = 0.0
                sr = 1.0
            bbox = ent.get("bbox") or [0, 0, 0, 0]
            bw = max(1.0, bbox[2] - bbox[0])
            bh = max(1.0, bbox[3] - bbox[1])
            geoms.append([float(b is not None), float(a is not None),
                          d[0], d[1], d[2], math.sqrt(sum(v * v for v in d)),
                          dyaw, dtilt, droll, sr, bw / W, bh / H, 1.0 / (1.0 + bw * bh)])
            pair_idx.append(pi)
            slots.append(slot)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        slot_geom=np.asarray(geoms, dtype=np.float32),
                        slot_geom_pair=np.asarray(pair_idx, dtype=np.int32),
                        slot_geom_slot=np.asarray(slots, dtype=np.int32))
    print(json.dumps({"slots": len(geoms), "pairs": len(manifest), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
