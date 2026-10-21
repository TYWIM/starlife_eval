"""关系负样本采样 —— 逐行移植 ``train_heads.build_relation_data``。

这是 relation macro-F1（v2g 0.7745）可复现的关键：**评估时也必须**对每条真关系，
在同一 pair 内用 ``np.random.default_rng(20261020)`` 抽 ``sample_mult=2`` 个未标注槽对
作为 ``none`` 负样本，并且用同一个 RNG 顺序生成，否则采样结果与训练时不同、
relation/rel_pos/attrs 头的输入分布就对不上。

原函数：``starlife/starlife_scripts/train_heads.py:169-204 build_relation_data``。
本模块逐行对应，返回 dict 的键名与语义完全一致：
``frm/to/label/pos/contact/support/dist/split/is_none/n_types/types/rel_pos_keys``。

正样本在前（顺序 = npz ``rel_*`` 行序），负样本追加在后；
``label == n_types``（= len(rel_types)）表示 none；负样本 ``pos=-1``、
``contact/support/dist`` 全零。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

DEFAULT_SEED = 20261020
SAMPLE_MULT = 2


def _meta_of(z: Any, meta: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if meta is not None:
        return meta
    inner = getattr(z, "meta", None)
    if inner is None:
        raise TypeError("build_relation_data 需要 meta（或带 .meta 的 FeatureBundle）")
    return inner


def _arr(z: Any, key: str) -> np.ndarray:
    # FeatureBundle / dict / NpzFile 都支持 __getitem__，保持与原脚本一致
    return z[key]  # type: ignore[index]


def build_relation_data(z: Any, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """生成正样本 + 采样负样本（同帧未关联槽对）。

    ported from train_heads.py:169-204 build_relation_data（逐行一致）
    """
    rel_from, rel_to = _arr(z, "rel_from"), _arr(z, "rel_to")
    rel_type, rel_pos = _arr(z, "rel_type"), _arr(z, "rel_pos")
    pair_split = _arr(z, "pair_split")
    slot_pair = _arr(z, "slot_pair")
    rng = np.random.default_rng(DEFAULT_SEED)
    n_rel = len(rel_from)
    meta = _meta_of(z, meta)
    types = meta["rel_types"]
    n_types = len(types) + 1  # + none

    sample_mult = SAMPLE_MULT
    extra_from: list[int] = []
    extra_to: list[int] = []
    slot_pairs = set(zip(rel_from.tolist(), rel_to.tolist()))
    for i in range(n_rel):
        pid = int(slot_pair[rel_from[i]])
        same_pair_slots = np.where(slot_pair == pid)[0]
        if len(same_pair_slots) < 2:
            continue
        for _ in range(sample_mult):
            a = int(rng.choice(same_pair_slots))
            b = int(rng.choice(same_pair_slots))
            if a == b or (a, b) in slot_pairs:
                continue
            extra_from.append(a)
            extra_to.append(b)
    all_from = np.concatenate([rel_from, np.array(extra_from, dtype=np.int32)])
    all_to = np.concatenate([rel_to, np.array(extra_to, dtype=np.int32)])
    labels = np.concatenate([rel_type, np.full(len(extra_from), len(types), dtype=np.int32)])
    pos = np.concatenate([rel_pos, np.full(len(extra_from), -1, dtype=np.int32)])
    contact = np.concatenate([_arr(z, "rel_contact"), np.zeros(len(extra_from), dtype=np.int8)])
    support = np.concatenate([_arr(z, "rel_support"), np.zeros(len(extra_from), dtype=np.int8)])
    dist = np.concatenate([_arr(z, "rel_dist"), np.full(len(extra_from), 0.0, dtype=np.float32)])
    split = pair_split[slot_pair[all_from]]
    is_none = labels == len(types)
    return dict(
        frm=all_from,
        to=all_to,
        label=labels,
        pos=pos,
        contact=contact,
        support=support,
        dist=dist,
        split=split,
        is_none=is_none,
        n_types=n_types,
        types=types,
        rel_pos_keys=meta["rel_pos_keys"],
    )


def n_negative(rel: Mapping[str, Any]) -> int:
    return int(np.sum(rel["is_none"]))


def n_positive(rel: Mapping[str, Any]) -> int:
    return int(len(rel["frm"]) - n_negative(rel))
