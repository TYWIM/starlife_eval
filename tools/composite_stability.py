"""综合分的权重敏感性：换权重会不会换冠军？

综合分 = 7 维等权平均。等权是「不预设维度重要性」的选择，但**总分依赖权重**是客观事实。
这个脚本做两件事，把这件事摆到明面上：

1. **随机权重**：从 Dirichlet(1,...,1)（= 7 维均匀单纯形）抽 N 组权重，
   统计每个模型拿第 1 的概率、综合分的均值/标准差、以及两两比较的 P(A>B)。
2. **主题权重**：几组"人会真的这么选"的权重（变化派 / 关系派 / 事件派 / 全等权），
   看冠军是否改变、云端 4 个模型的名次如何互相换位。

用法::

    python tools/composite_stability.py --report reports/v1.2/report.json --n 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DIMS = ("change_top1", "change_balance", "relation_type", "rel_pos", "attrs", "event_f1",
        "state_consistency")

THEMES = {
    "等权（榜单默认）": [1, 1, 1, 1, 1, 1, 1],
    "变化派（只看改了什么）": [2, 2, 0, 0, 0, 0, 0],
    "关系派（拓扑/方位/交互）": [0, 0, 3, 2, 2, 0, 0],
    "事件派（写给下游吃）": [0, 0, 0, 0, 0, 2, 2],
    "感知派（变化+状态一致性）": [2, 1, 0, 0, 0, 0, 2],
}


def load(report_path: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    models, mat = [], []
    for mid, m in report["models"].items():
        dims = {d["key"]: d["value"] for d in (m.get("composite") or {}).get("dims") or []}
        models.append(mid)
        mat.append([float(dims.get(k)) if dims.get(k) is not None else np.nan for k in DIMS])
    return models, np.array(mat, dtype=float)


def weighted(mat: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    mask = ~np.isnan(mat)
    return (np.nan_to_num(mat) * w * mask).sum(axis=1) / (w * mask).sum(axis=1)


def ranks(scores: np.ndarray) -> np.ndarray:
    """1 = 最高分（并列时取最小名次）。"""
    order = (-scores).argsort()
    out = np.empty_like(order)
    out[order] = np.arange(1, len(scores) + 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/v1.2/report.json")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    models, mat = load(Path(args.report))
    rng = np.random.default_rng(args.seed)
    n = len(models)

    # 1) 随机权重（Dirichlet(1)）
    top1 = np.zeros(n)
    rank_hist = np.zeros((n, n + 1))
    samples = np.zeros((args.n, n))
    for i in range(args.n):
        w = rng.dirichlet(np.ones(len(DIMS)))
        s = weighted(mat, w)
        r = ranks(s)
        samples[i] = s
        top1[r == 1] += 1
        for j, rj in enumerate(r):
            rank_hist[j, rj] += 1
    top1 /= args.n
    rank_hist /= args.n

    print(f"== 随机权重（Dirichlet(1) 均匀抽 {args.n} 组，7 维）==")
    print(f"{'model':24s} {'综合分(等权)':>12s} {'随机权重 mean±sd':>18s} {'P(第1)':>8s} "
          f"{'P(前2)':>8s} {'最常名次':>8s}")
    eq = weighted(mat, np.ones(len(DIMS)))
    order = np.argsort(-eq)
    for j in order:
        best_rank = 1 + int(np.argmax(rank_hist[j, 1:]))
        print(f"{models[j]:24s} {eq[j]:12.1f} "
              f"{samples[:, j].mean():10.1f} ± {samples[:, j].std():4.1f} "
              f"{top1[j]:8.3f} {rank_hist[j, 1] + rank_hist[j, 2]:8.3f} "
              f"{best_rank:8d}")

    print("\n== 两两比较 P(前 > 后)（随机权重下）==")
    for ii in range(n):
        for jj in range(ii + 1, n):
            a, b = order[ii], order[jj]
            print(f"  {models[a]:22s} > {models[b]:22s}: P={(samples[:, a] > samples[:, b]).mean():.3f}")

    print("\n== 主题权重下的排名（同一份真实数据）==")
    print(f"{'权重方案':26s} " + " ".join(f"{m[:14]:>14s}" for m in models))
    for name, w in THEMES.items():
        s = weighted(mat, np.array(w, dtype=float))
        r = ranks(s)
        cells = " ".join(f"{'#' + str(r[j]) + ' ' + format(s[j], '.1f'):>14s}" for j in range(n))
        print(f"{name:26s} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
