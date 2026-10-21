"""60 对对照协议单测：手写 manifest + 预测，断言精确数值。

玩具数据（3 个 test pair）：
    pair0: slots [0,1,2]  changes={1:appeared}   relations=[(0,1,on,rel_pos=back,contact=F,support=T)]
    pair1: slots [0,1]    changes={}             relations=[(0,1,near,rel_pos=front,contact=T,support=F)]
    pair2: slots [0,1,2,3] changes={0:disappeared} relations=[(1,3,on,rel_pos=left,contact=F,support=F)]

预测故意错两类：pair1 关系类型错、pair2 rel_pos 错 + contact 错。
change 预测里 class2(displaced) 只作为误报出现、GT 从无 displaced —— 用来验证
change_macro_f1 用的是 macro_f1_compare（跳过 tp==0 的类）而不是 train 口径。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from starlife_eval.types import (
    DatasetManifest,
    Entity,
    HeadPrediction,
    PairPrediction,
    PairSample,
    Relation,
    RelationPrediction,
)
from starlife_eval.verifiers.metrics import macro_f1_compare, macro_f1_train
from starlife_eval.verifiers.pair_protocol import (
    evaluate_pair_protocol,
    local_pair_predictions,
    sample_pairs,
)


# --------------------------------------------------------------------- helpers


def _entity(slot: int) -> Entity:
    return Entity(slot=slot, bbox=(0.0, 0.0, 10.0, 10.0), type_key="table", present_after=True)


def _relation(
    frm: int, to: int, typ: str, rel_pos: str | None, contact: bool, support: bool
) -> Relation:
    return Relation(
        from_slot=frm, to_slot=to, type=typ, rel_pos=rel_pos, contact=contact, support=support, dist_m=None
    )


def _pair(
    index: int,
    split: str,
    n_slots: int,
    changes: dict[int, str] | None = None,
    relations: tuple[Relation, ...] = (),
    raw_relations: list[dict] | None = None,
) -> PairSample:
    entities = tuple(_entity(s) for s in range(n_slots))
    raw: dict = {}
    if raw_relations is not None:
        raw["relations"] = raw_relations
    return PairSample(
        index=index,
        pair_id=f"{index:06d}",
        split=split,
        before="",
        after="",
        entities=entities,
        changes=changes or {},
        relations=relations,
        raw=raw,
    )


def _manifest(pairs: list[PairSample]) -> DatasetManifest:
    offsets = [0]
    for p in pairs:
        offsets.append(offsets[-1] + len(p.entities))
    return DatasetManifest(root=Path("."), pairs=tuple(pairs), meta={}, slot_offsets=tuple(offsets))


def _toy_manifest() -> DatasetManifest:
    return _manifest(
        [
            _pair(
                0,
                "test",
                3,
                changes={1: "appeared"},
                relations=(_relation(0, 1, "on", "back", False, True),),
            ),
            _pair(
                1,
                "test",
                2,
                relations=(_relation(0, 1, "near", "front", True, False),),
            ),
            _pair(
                2,
                "test",
                4,
                changes={0: "disappeared"},
                relations=(_relation(1, 3, "on", "left", False, False),),
            ),
        ]
    )


def _toy_preds() -> dict[int, PairPrediction]:
    return {
        0: PairPrediction(
            pair_index=0,
            changes={0: "none", 1: "appeared", 2: "displaced"},
            relations=[RelationPrediction(0, 1, "on", "back", False, True)],
        ),
        1: PairPrediction(
            pair_index=1,
            changes={0: "none", 1: "appeared"},
            relations=[RelationPrediction(0, 1, "next_to", None, True, False)],  # 类型错
        ),
        2: PairPrediction(
            pair_index=2,
            changes={0: "disappeared", 1: "none", 2: "none", 3: "appeared"},
            relations=[RelationPrediction(1, 3, "on", "back", True, False)],  # rel_pos/contact 错
        ),
    }


# ---------------------------------------------------------------- sample_pairs


def test_sample_pairs_matches_original_algorithm_exactly():
    pairs = [_pair(i, "test", 2) for i in range(10)] + [_pair(10, "train", 2), _pair(11, "train", 2)]
    manifest = _manifest(pairs)

    got = sample_pairs(manifest, split="test", n=3, seed=20261020)
    # 与 eval_compare_cloud.main 完全相同的抽样表达式
    expected = sorted(random.Random(20261020).sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
    assert got == expected
    assert got == sorted(got)  # 返回排序
    assert len(got) == 3
    assert all(0 <= i < 10 for i in got)  # 不会抽到 train

    # 同 seed 两次一致
    assert sample_pairs(manifest, "test", 3, 20261020) == got
    # n 超过 test 对数时取全部 test（原脚本 min(n, len(test_idx))）
    assert sample_pairs(manifest, "test", 60, 20261020) == list(range(10))
    # n=0
    assert sample_pairs(manifest, "test", 0, 20261020) == []


def test_sample_pairs_uses_manifest_split_field():
    manifest = _manifest([_pair(0, "train", 2), _pair(1, "test", 2)])
    assert sample_pairs(manifest, split="test", n=60) == [1]
    assert sample_pairs(manifest, split="train", n=60) == [0]
    with pytest.raises(ValueError, match="没有任何 pair"):
        sample_pairs(manifest, split="val", n=1)


# --------------------------------------------------------- evaluate_pair_protocol


def test_pair_protocol_exact_values():
    result = evaluate_pair_protocol(_toy_manifest(), [0, 1, 2], _toy_preds())
    assert result == {
        "change_slots": 9,  # 3 + 2 + 4
        "change_top1": round(6 / 9, 4),  # 手算：6/9 命中
        "change_macro_f1": 0.7424,  # 手算见下
        "relation_pairs": 3,
        "relation_type_acc": round(2 / 3, 4),
        "rel_pos_acc": round(1 / 2, 4),
        "attrs_acc": round(3 / 4, 4),
    }


def test_change_macro_f1_uses_compare_semantics_not_train():
    """change_macro_f1 必须是 macro_f1_compare（跳过 tp==0 的类）。

    玩具例：类 displaced 在 GT 里从未出现（tp==0），只作为一次误报出现（fp=1）。
    compare 口径整类跳过 -> 0.7424；train 口径会把这个 0 分计入 -> 0.5568。
    """
    true_idx = [0, 1, 0, 0, 0, 3, 0, 0, 0]
    pred_idx = [0, 1, 2, 0, 1, 3, 0, 0, 1]
    compare = macro_f1_compare(true_idx, pred_idx, 4)
    train, _ = macro_f1_train(true_idx, pred_idx, 4)

    assert round(compare, 4) == 0.7424
    assert round(train, 4) == 0.5568
    assert compare > train

    result = evaluate_pair_protocol(_toy_manifest(), [0, 1, 2], _toy_preds())
    assert result["change_macro_f1"] == round(compare, 4)
    assert result["change_macro_f1"] != round(train, 4)


def test_missing_prediction_raises_by_default():
    with pytest.raises(KeyError, match="缺失 pair 预测"):
        evaluate_pair_protocol(_toy_manifest(), [0, 1, 2], {0: _toy_preds()[0]})


def test_missing_prediction_can_fall_back_to_empty_like_cloud_parser():
    manifest = _manifest(
        [_pair(0, "test", 2, changes={0: "appeared"}, relations=(_relation(0, 1, "on", "back", False, False),))]
    )
    result = evaluate_pair_protocol(manifest, [0], {}, strict=False)
    # 空预测：GT [appeared, none] vs pred [none, none] -> 1/2
    # compare: 类 none tp=1 fp=1 -> 2/3；类 appeared tp=0 跳过
    assert result["change_slots"] == 2
    assert result["change_top1"] == 0.5
    assert result["change_macro_f1"] == round(2 / 3, 4)
    assert result["relation_pairs"] == 1
    assert result["relation_type_acc"] == 0
    assert result["rel_pos_acc"] == 0
    assert result["attrs_acc"] == 0


def test_rel_pos_only_counted_when_gt_field_present():
    """GT 关系没有 rel_pos 字段时不计入 rel_pos 分母（原脚本 ``"rel_pos" in g``）。"""
    manifest = _manifest(
        [
            _pair(
                0,
                "test",
                2,
                changes={},
                relations=(_relation(0, 1, "on", None, False, False),),
                raw_relations=[{"from_slot": 0, "to_slot": 1, "type": "on", "contact": False, "support": False}],
            )
        ]
    )
    preds = {0: PairPrediction(0, changes={}, relations=[RelationPrediction(0, 1, "on", None, False, False)])}
    result = evaluate_pair_protocol(manifest, [0], preds)
    assert result["relation_pairs"] == 1
    assert result["relation_type_acc"] == 1.0
    assert result["rel_pos_acc"] == 0  # 分母为 0 -> 0
    assert result["attrs_acc"] == 1.0


def test_relation_match_last_duplicate_key_wins():
    """原脚本 dict 推导重复 (from,to) 后者覆盖，这里保持同样语义。"""
    manifest = _manifest(
        [
            _pair(
                0,
                "test",
                2,
                changes={},
                relations=(_relation(0, 1, "near", None, False, False),),
                raw_relations=[{"from_slot": 0, "to_slot": 1, "type": "near", "contact": False, "support": False}],
            )
        ]
    )
    preds = {
        0: PairPrediction(
            0,
            changes={},
            relations=[
                RelationPrediction(0, 1, "on", None, False, False),
                RelationPrediction(0, 1, "near", None, False, False),  # 后者生效
            ],
        )
    }
    result = evaluate_pair_protocol(manifest, [0], preds)
    assert result["relation_type_acc"] == 1.0


def test_unknown_change_label_counts_as_wrong_and_does_not_crash():
    """云端可能吐出非法 change 标签：top1 计错、macro 不参与，但不允许抛异常中断评测。"""
    manifest = _manifest([_pair(0, "test", 2, changes={0: "appeared"})])
    preds = {0: PairPrediction(0, changes={0: "moved", 1: "none"}, relations=[])}
    result = evaluate_pair_protocol(manifest, [0], preds)
    assert result["change_slots"] == 2
    assert result["change_top1"] == 0.5  # slot1 命中
    # 类 appeared 的 tp 仍为 0（非法预测不计入 appeared 的 fp/fn 之外的类）
    assert result["change_macro_f1"] == 1.0  # 只剩 none 类有 tp，其余跳过


def test_sample_then_evaluate_full_60_style_flow():
    manifest = _toy_manifest()
    indices = sample_pairs(manifest, "test", n=60, seed=20261020)
    assert indices == [0, 1, 2]
    preds = _toy_preds()
    assert evaluate_pair_protocol(manifest, indices, preds) == evaluate_pair_protocol(manifest, [0, 1, 2], preds)


def test_manifest_pairs_are_looked_up_by_index_not_position():
    """pair 行号与 tuple 位置不一致时按 ``PairSample.index`` 查（防 manifest 过滤后错位）。"""
    manifest = _manifest([_pair(0, "train", 2), _pair(1, "test", 2, changes={0: "appeared"})])
    preds = {1: PairPrediction(1, changes={0: "appeared", 1: "none"}, relations=[])}
    result = evaluate_pair_protocol(manifest, [1], preds)
    assert result["change_slots"] == 2
    assert result["change_top1"] == 1.0


# ------------------------------------------------- local_pair_predictions 映射


def _toy_head_for_mapping() -> HeadPrediction:
    """3 个 pair、每对 2 slot（全局 slot 0..5），2 条真关系 + 1 条负样本。"""
    return HeadPrediction(
        change=np.array([0, 1, 2, 0, 3, 0], dtype=np.int64),
        type=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
        relation=np.array([0, 1, 2], dtype=np.int64),  # near, on, none
        rel_pos=np.array([0, 1, 2], dtype=np.int64),
        attrs=np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int64),
        n_relation_positives=2,
    )


def _toy_relation_for_mapping() -> dict:
    return {
        "frm": np.array([0, 2, 4], dtype=np.int32),
        "to": np.array([1, 3, 5], dtype=np.int32),
        "label": np.array([0, 1, 2], dtype=np.int32),  # 2 == none
        "pos": np.array([0, 1, -1], dtype=np.int32),
        "contact": np.array([1, 0, 0], dtype=np.int8),
        "support": np.array([0, 1, 0], dtype=np.int8),
        "split": np.array(["test", "test", "test"]),
        "is_none": np.array([False, False, True]),
        "n_types": 3,
        "types": ["near", "on"],
        "rel_pos_keys": ["front", "back"],
    }


def test_local_pair_predictions_maps_global_slots_to_per_pair_slots():
    """HeadPrediction（全局行）-> PairPrediction（per-pair slot），负样本必须丢弃。"""
    manifest = _manifest([_pair(0, "test", 2), _pair(1, "test", 2), _pair(2, "test", 2)])
    meta = {"type_keys": ["a", "b"], "rel_types": ["near", "on"], "rel_pos_keys": ["front", "back"]}
    preds = local_pair_predictions(
        manifest, _toy_head_for_mapping(), _toy_relation_for_mapping(), [0, 1, 2], meta
    )
    assert set(preds) == {0, 1, 2}

    # pair0: 全局 slot 0/1 -> per-pair slot 0/1
    assert preds[0].changes == {0: "none", 1: "appeared"}
    assert preds[0].types == {0: "a", 1: "a"}
    assert preds[0].relations == [RelationPrediction(0, 1, "near", "front", True, False)]

    # pair1: 全局 slot 2/3 -> per-pair slot 0/1；类型/方位来自预测索引
    assert preds[1].changes == {0: "displaced", 1: "none"}
    assert preds[1].types == {0: "a", 1: "b"}
    assert preds[1].relations == [RelationPrediction(0, 1, "on", "back", False, True)]

    # pair2: 只有负样本关系 -> relations 为空
    assert preds[2].changes == {0: "disappeared", 1: "none"}
    assert preds[2].relations == []


def test_local_pair_predictions_only_returns_requested_pairs():
    manifest = _manifest([_pair(0, "test", 2), _pair(1, "test", 2), _pair(2, "test", 2)])
    meta = {"type_keys": ["a", "b"], "rel_types": ["near", "on"], "rel_pos_keys": ["front", "back"]}
    preds = local_pair_predictions(
        manifest, _toy_head_for_mapping(), _toy_relation_for_mapping(), [0, 2], meta
    )
    assert set(preds) == {0, 2}


def test_local_pair_predictions_requires_meta_or_taxonomy():
    manifest = _manifest([_pair(0, "test", 2), _pair(1, "test", 2), _pair(2, "test", 2)])
    rel = _toy_relation_for_mapping()
    rel.pop("types")
    rel.pop("rel_pos_keys")
    # 没有 meta 也没有词表 -> 明确报错
    with pytest.raises(ValueError, match="词表"):
        local_pair_predictions(manifest, _toy_head_for_mapping(), rel, [0])


def test_local_pair_predictions_validates_slot_rows():
    manifest = _manifest([_pair(0, "test", 2), _pair(1, "test", 2), _pair(2, "test", 2)])
    head = HeadPrediction(
        change=np.array([0, 1, 2, 0, 3], dtype=np.int64),  # 少一行
        type=np.array([0] * 5, dtype=np.int64),
        relation=np.array([0, 1, 2], dtype=np.int64),
        rel_pos=np.array([0, 1, 2], dtype=np.int64),
        attrs=np.zeros((3, 2), dtype=np.int64),
        n_relation_positives=2,
    )
    with pytest.raises(ValueError, match="总 slot 数"):
        local_pair_predictions(manifest, head, _toy_relation_for_mapping(), [0])
