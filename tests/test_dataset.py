"""dataset 层单测：manifest / features / geometry / 负采样 / 拼装。

原则：所有断言都写**具体数值**，不只断言“不报错”。真实产物 fixture 在场时
（tests/fixtures/）追加逐位 parity（``np.array_equal``，容差 0）；fixture 缺失时
``pytest.skip``，不影响合成数据单测。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from starlife_eval.dataset import (
    GEOMETRY_DIM,
    ManifestError,
    bbox_map_from_manifest,
    build_pair_base,
    build_pair_feat,
    build_relation_data,
    build_slot_feat,
    build_slot_inputs,
    build_type_feat,
    feature_dim,
    head_input_dims,
    load_features,
    load_manifest,
    pair_base_dim,
    pair_geometry,
    relation_onehot,
    relation_onehot_dim,
)
from starlife_eval.dataset.features import FeatureError
from starlife_eval.types import FeatureBundle

FIXTURES = Path(__file__).resolve().parent / "fixtures"


TYPE_KEYS = ["book", "bottle", "mug", "pen", "table"]
REL_TYPES = ["near", "on", "part_of"]
REL_POS_KEYS = ["front", "back", "left"]


# --------------------------------------------------------------------------- helpers


def _bbox(pair: int, slot: int) -> list[float]:
    x0 = 10.0 * slot + pair
    y0 = 20.0 + 5.0 * pair + 3.0 * slot
    return [x0, y0, x0 + 40.0, y0 + 30.0]


def make_dataset(
    root: Path,
    *,
    dim: int = 6,
    slots_per_pair: tuple[int, ...] = (3, 3, 2),
    splits: tuple[str, ...] = ("train", "train", "test"),
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """写一份形状可控的合成 dataset（manifest.jsonl + meta.json + 特征 npz）。

    数组布局逐位模仿 ``extract_features.run_spatial``：slot 行按 pair 顺序展开，
    ``slot_pair`` 是 manifest 行号，``rel_from/rel_to`` 是全局 slot 下标。
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_pair = len(slots_per_pair)
    assert len(splits) == n_pair

    rows: list[dict] = []
    slot_before: list[np.ndarray] = []
    slot_after: list[np.ndarray] = []
    slot_mask: list[int] = []
    slot_present: list[int] = []
    slot_change: list[int] = []
    slot_type: list[int] = []
    slot_pair: list[int] = []
    rel_from: list[int] = []
    rel_to: list[int] = []
    rel_type: list[int] = []
    rel_pos: list[int] = []
    rel_contact: list[int] = []
    rel_support: list[int] = []
    rel_dist: list[float] = []
    change_label = ["none", "appeared", "displaced", "disappeared"]

    slot_ptr = 0
    for i in range(n_pair):
        n = slots_per_pair[i]
        entities = [
            {"slot": k, "bbox": _bbox(i, k), "type_key": TYPE_KEYS[k % len(TYPE_KEYS)],
             "present_after": not (i == 1 and k == 2)}
            for k in range(n)
        ]
        changes: list[dict] = []
        if i == 0:
            changes = [{"slot": 1, "type": "appeared"}]
        elif i == 1:
            changes = [{"slot": 0, "type": "displaced"}, {"slot": 2, "type": "disappeared"}]
        # i == 2: 无变化
        rels = [
            {"from_slot": 0, "to_slot": 1, "type": "near", "rel_pos": "front",
             "contact": False, "support": False, "dist_m": 0.336},
            {"from_slot": 1, "to_slot": 0, "type": "on", "rel_pos": "back",
             "contact": True, "support": False, "dist_m": 0.068},
        ]
        rows.append({
            "pair_id": f"{i:06d}",
            "split": splits[i],
            "vocab_version": "1.0.0",
            "before": f"frames/{i:06d}_before.jpg",
            "after": f"frames/{i:06d}_after.jpg",
            "entities": entities,
            "changes": changes,
            "relations": rels,
            "negative": False,
            "meta": {"lighting": "warm"},
        })

        change_map = {c["slot"]: c["type"] for c in changes}
        for k, e in enumerate(entities):
            slot_before.append(rng.standard_normal(dim).astype(np.float16))
            slot_after.append((rng.standard_normal(dim) + 0.25).astype(np.float16))
            slot_mask.append(0 if not e["present_after"] else 1)
            slot_present.append(1 if e["present_after"] else 0)
            slot_change.append(change_label.index(change_map.get(k, "none")))
            slot_type.append(TYPE_KEYS.index(e["type_key"]))
            slot_pair.append(i)
        base = slot_ptr
        for rel in rels:
            rel_from.append(base + rel["from_slot"])
            rel_to.append(base + rel["to_slot"])
            rel_type.append(REL_TYPES.index(rel["type"]))
            rel_pos.append(REL_POS_KEYS.index(rel["rel_pos"]))
            rel_contact.append(int(rel["contact"]))
            rel_support.append(int(rel["support"]))
            rel_dist.append(rel["dist_m"])
        slot_ptr += n

    arrays = {
        "slot_before": np.stack(slot_before).astype(np.float16),
        "slot_after": np.stack(slot_after).astype(np.float16),
        "slot_mask": np.asarray(slot_mask, dtype=np.int8),
        "slot_present": np.asarray(slot_present, dtype=np.int8),
        "slot_change": np.asarray(slot_change, dtype=np.int8),
        "slot_type": np.asarray(slot_type, dtype=np.int32),
        "slot_pair": np.asarray(slot_pair, dtype=np.int32),
        "pair_split": np.asarray(splits),
        "rel_from": np.asarray(rel_from, dtype=np.int32),
        "rel_to": np.asarray(rel_to, dtype=np.int32),
        "rel_type": np.asarray(rel_type, dtype=np.int32),
        "rel_pos": np.asarray(rel_pos, dtype=np.int32),
        "rel_contact": np.asarray(rel_contact, dtype=np.int8),
        "rel_support": np.asarray(rel_support, dtype=np.int8),
        "rel_dist": np.asarray(rel_dist, dtype=np.float32),
    }
    meta = {
        "mode": "spatial",
        "pairs": n_pair,
        "slots": int(arrays["slot_pair"].shape[0]),
        "dim": dim,
        "type_keys": TYPE_KEYS,
        "rel_types": REL_TYPES,
        "rel_pos_keys": REL_POS_KEYS,
        "change_keys": change_label,
        "types_count": len(TYPE_KEYS),
        "relations": int(arrays["rel_from"].shape[0]),
    }
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    np.savez(root / "features.npz", **arrays)
    (root / "features.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return arrays


# ----------------------------------------------------------------------- manifest


def test_manifest_slot_offsets_and_split(tmp_path: Path) -> None:
    make_dataset(tmp_path, slots_per_pair=(3, 3, 2))
    m = load_manifest(tmp_path)

    assert len(m.pairs) == 3
    assert m.slot_offsets == (0, 3, 6, 8)
    assert [p.pair_id for p in m.pairs] == ["000000", "000001", "000002"]
    assert m.pairs[0].slots == (0, 1, 2)
    assert m.pairs[2].slots == (0, 1)

    # 全局 slot 下标映射 = slot_offsets[pair] + slot
    assert m.global_slot(0, 0) == 0
    assert m.global_slot(1, 2) == 5
    assert m.global_slot(2, 1) == 7
    for g in range(8):
        pair = m.pair_of_global_slot(g)
        expected = 0 if g < 3 else (1 if g < 6 else 2)
        assert pair == expected

    # split 过滤保留原 manifest 行号
    assert m.split_indices("train") == [0, 1]
    assert m.split_indices("test") == [2]
    assert m.split_indices("val") == []

    # changes / relations 解析
    assert m.pairs[0].changes == {1: "appeared"}
    assert m.pairs[1].changes == {0: "displaced", 2: "disappeared"}
    assert m.pairs[2].changes == {}
    rel = m.pairs[0].relations[0]
    assert (rel.from_slot, rel.to_slot, rel.type, rel.rel_pos) == (0, 1, "near", "front")
    assert rel.contact is False and rel.support is False
    assert rel.dist_m == pytest.approx(0.336)
    assert m.pairs[1].entities[2].present_after is False


def test_manifest_rejects_duplicate_pair_id(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    dup = dict(row)
    (tmp_path / "manifest.jsonl").write_text("\n".join([lines[0], json.dumps(dup)]) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="重复"):
        load_manifest(tmp_path)


def test_manifest_rejects_non_contiguous_slots(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["entities"][1]["slot"] = 5  # 不连续
    lines[0] = json.dumps(row)
    (tmp_path / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="连续"):
        load_manifest(tmp_path)


def test_manifest_rejects_bad_references(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()

    row = json.loads(lines[2])
    row["changes"] = [{"slot": 9, "type": "appeared"}]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join([lines[0], lines[1], json.dumps(row)]) + "\n", encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="不在 entities"):
        load_manifest(tmp_path)

    row2 = json.loads(lines[2])
    row2["changes"] = []
    row2["relations"] = [{"from_slot": 0, "to_slot": 9, "type": "near", "rel_pos": "front"}]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join([lines[0], lines[1], json.dumps(row2)]) + "\n", encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="不在 entities"):
        load_manifest(tmp_path)


def test_manifest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_manifest(tmp_path / "nope")


# ----------------------------------------------------------------------- features


def test_features_load_and_dim_from_data(tmp_path: Path) -> None:
    make_dataset(tmp_path, dim=7)
    bundle = load_features(tmp_path / "features.npz")
    assert feature_dim(bundle) == 7  # 不硬编码 5120
    assert bundle["slot_after"].shape == (8, 7)
    assert bundle["pair_split"].shape == (3,)
    assert bundle.type_keys == TYPE_KEYS
    assert bundle.rel_types == REL_TYPES
    assert bundle.rel_pos_keys == REL_POS_KEYS


def test_features_missing_key_raises(tmp_path: Path) -> None:
    arrays = make_dataset(tmp_path)
    arrays.pop("rel_dist")
    np.savez(tmp_path / "features.npz", **arrays)
    with pytest.raises(FeatureError, match="rel_dist"):
        load_features(tmp_path / "features.npz")


def test_features_meta_dim_mismatch_raises(tmp_path: Path) -> None:
    make_dataset(tmp_path, dim=6)
    meta = json.loads((tmp_path / "features.meta.json").read_text(encoding="utf-8"))
    meta["dim"] = 99
    (tmp_path / "features.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(FeatureError, match="meta.dim"):
        load_features(tmp_path / "features.npz")


def test_features_slot_type_out_of_range(tmp_path: Path) -> None:
    arrays = make_dataset(tmp_path)
    arrays["slot_type"][0] = 999
    np.savez(tmp_path / "features.npz", **arrays)
    with pytest.raises(FeatureError, match="slot_type"):
        load_features(tmp_path / "features.npz")


# ----------------------------------------------------------------------- geometry


def test_pair_geometry_nine_dims_exact_values() -> None:
    a = np.asarray([0.0, 0.0, 100.0, 50.0], dtype=np.float32)
    b = np.asarray([20.0, 10.0, 80.0, 40.0], dtype=np.float32)
    bbox = {0: a, 1: b}

    geo = pair_geometry({"frm": np.asarray([0]), "to": np.asarray([1])}, bbox)
    assert geo.shape == (1, GEOMETRY_DIM)
    # sax=say? a center=(50,25), b center=(50,25)
    expected = [
        0.0,                       # dxn = (50-50)/30
        0.0,                       # dyn
        100.0 / 60.0,              # wa/wb
        50.0 / 30.0,               # ha/hb
        (100.0 * 50.0) / (60.0 * 30.0),  # area_a/area_b
        1800.0 / 5000.0,           # iou: inter=60*30, union=5000+1800-1800
        0.0,                       # dist_norm: centers coincide
        50.0 / 1024.0,             # sax/img_w
        25.0 / 768.0,              # say/img_h
    ]
    np.testing.assert_allclose(geo[0], expected, rtol=0, atol=1e-6)
    assert geo.dtype == np.float32

    # 对称：交换 frm/to
    geo2 = pair_geometry({"frm": np.asarray([1]), "to": np.asarray([0])}, bbox)
    expected2 = [
        0.0, 0.0,
        60.0 / 100.0, 30.0 / 50.0,
        1800.0 / 5000.0,
        1800.0 / 5000.0,
        0.0,
        50.0 / 1024.0, 25.0 / 768.0,
    ]
    np.testing.assert_allclose(geo2[0], expected2, rtol=0, atol=1e-6)


def test_pair_geometry_offset_and_empty() -> None:
    a = np.asarray([0.0, 0.0, 100.0, 50.0], dtype=np.float32)
    b = np.asarray([20.0, 10.0, 80.0, 40.0], dtype=np.float32)
    bbox = {10: a, 11: b}
    # b 的中心相对 a 偏移: sax=50,say=25; hbx=50,hby=25 → 仍然 0；改用不平移后 b
    b2 = np.asarray([120.0, 60.0, 180.0, 90.0], dtype=np.float32)  # center (150,75), hw=30,hh=15
    bbox = {10: a, 11: b2}
    geo = pair_geometry({"frm": np.asarray([10]), "to": np.asarray([11])}, bbox)
    assert geo[0, 0] == pytest.approx((50.0 - 150.0) / 30.0)  # dxn
    assert geo[0, 1] == pytest.approx((25.0 - 75.0) / 15.0)  # dyn
    assert geo[0, 6] == pytest.approx(
        np.sqrt((50.0 - 150.0) ** 2 + (25.0 - 75.0) ** 2) / np.sqrt(1024.0 ** 2 + 768.0 ** 2)
    )
    empty = pair_geometry({"frm": np.zeros((0,), dtype=np.int32), "to": np.zeros((0,), dtype=np.int32)}, bbox)
    assert empty.shape == (0, GEOMETRY_DIM)


def test_bbox_map_from_manifest(tmp_path: Path) -> None:
    make_dataset(tmp_path, slots_per_pair=(3, 3, 2))
    m = load_manifest(tmp_path)
    bbox = bbox_map_from_manifest(m)
    assert sorted(bbox) == list(range(8))
    np.testing.assert_array_equal(bbox[0], np.asarray(_bbox(0, 0), dtype=np.float32))
    np.testing.assert_array_equal(bbox[7], np.asarray(_bbox(2, 1), dtype=np.float32))  # 全局 7 = pair2 slot1
    assert bbox[0].dtype == np.float32


# ------------------------------------------------------------ 负采样（核心 parity）


def _reference_build_relation_data(z: dict, meta: dict) -> dict:
    """train_heads.py:169-204 的独立抄本（只用于差分测试，不 import 原仓库）。"""
    rel_from, rel_to = z["rel_from"], z["rel_to"]
    rel_type, rel_pos = z["rel_type"], z["rel_pos"]
    pair_split = z["pair_split"]
    slot_pair = z["slot_pair"]
    rng = np.random.default_rng(20261020)
    n_rel = len(rel_from)
    types = meta["rel_types"]
    n_types = len(types) + 1
    sample_mult = 2
    extra_from, extra_to = [], []
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
    contact = np.concatenate([z["rel_contact"], np.zeros(len(extra_from), dtype=np.int8)])
    support = np.concatenate([z["rel_support"], np.zeros(len(extra_from), dtype=np.int8)])
    dist = np.concatenate([z["rel_dist"], np.full(len(extra_from), 0.0, dtype=np.float32)])
    split = pair_split[slot_pair[all_from]]
    is_none = labels == len(types)
    return dict(frm=all_from, to=all_to, label=labels, pos=pos, contact=contact,
                support=support, dist=dist, split=split, is_none=is_none, n_types=n_types)


def test_negative_sampling_matches_reference_and_is_deterministic(tmp_path: Path) -> None:
    arrays = make_dataset(tmp_path, slots_per_pair=(4, 4, 4, 3), splits=("train", "train", "val", "test"))
    meta = json.loads((tmp_path / "features.meta.json").read_text(encoding="utf-8"))
    rel = build_relation_data(arrays, meta)
    ref = _reference_build_relation_data(arrays, meta)

    for key in ("frm", "to", "label", "pos", "contact", "support", "dist", "split", "is_none"):
        np.testing.assert_array_equal(np.asarray(rel[key]), np.asarray(ref[key]), err_msg=key)
    assert rel["n_types"] == len(REL_TYPES) + 1 == 4

    # 确定性：同输入两次结果逐位相同（固定 rng(20261020)）
    rel2 = build_relation_data(arrays, meta)
    for key in ("frm", "to", "label", "pos"):
        np.testing.assert_array_equal(np.asarray(rel[key]), np.asarray(rel2[key]), err_msg=key)

    # 结构：正样本在前、负样本在后；负样本 label=n_types, pos=-1, 属性/距离全零
    n_pos = len(arrays["rel_from"])
    n_total = len(rel["frm"])
    assert n_pos == 8  # 4 对 × 每对 2 条真关系
    assert n_total > n_pos
    assert not rel["is_none"][:n_pos].any()
    assert rel["is_none"][n_pos:].all()
    assert np.all(rel["label"][n_pos:] == len(REL_TYPES))  # none 的 label = len(rel_types)
    assert rel["n_types"] == len(REL_TYPES) + 1 == 4
    assert np.all(rel["pos"][n_pos:] == -1)
    assert np.all(rel["contact"][n_pos:] == 0)
    assert np.all(rel["support"][n_pos:] == 0)
    assert np.all(rel["dist"][n_pos:] == 0.0)

    # 负样本必须是 same-pair 的未标注槽对、且 a != b
    pos_pairs = set(zip(arrays["rel_from"].tolist(), arrays["rel_to"].tolist()))
    slot_pair = arrays["slot_pair"]
    for k in range(n_pos, n_total):
        a, b = int(rel["frm"][k]), int(rel["to"][k])
        assert slot_pair[a] == slot_pair[b]
        assert a != b
        assert (a, b) not in pos_pairs


def test_negative_sampling_parity_with_real_fixture() -> None:
    """与服务器原始 train_heads.build_relation_data 输出逐位相等（6300 行）。"""
    ref_path = FIXTURES / "ref_relation_data.npz"
    gt_path = FIXTURES / "ref_gt.npz"
    meta_path = FIXTURES / "spatial_v2.meta.json"
    if not (ref_path.exists() and gt_path.exists() and meta_path.exists()):
        pytest.skip("缺少 ref_relation_data.npz / ref_gt.npz / spatial_v2.meta.json fixture")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ref = np.load(ref_path, allow_pickle=False)
    gt = np.load(gt_path, allow_pickle=False)
    is_none = ref["is_none"]
    n_pos = int(np.argmax(is_none))
    assert n_pos == 2575 and int(is_none.sum()) == 3725

    # 正样本前缀 = npz 原始 rel_* 输入；slot_pair/pair_split 来自 ref_gt（全量 GT）
    z = {
        "rel_from": ref["frm"][:n_pos],
        "rel_to": ref["to"][:n_pos],
        "rel_type": ref["label"][:n_pos],
        "rel_pos": ref["pos"][:n_pos],
        "rel_contact": ref["contact"][:n_pos],
        "rel_support": ref["support"][:n_pos],
        "rel_dist": ref["dist"][:n_pos],
        "pair_split": gt["pair_split"],
        "slot_pair": gt["slot_pair"],
    }
    out = build_relation_data(z, meta)
    for key in ("frm", "to", "label", "pos", "contact", "support", "dist", "split", "is_none"):
        np.testing.assert_array_equal(np.asarray(out[key]), np.asarray(ref[key]), err_msg=f"负采样 {key} 不逐位相等")
    assert int(out["n_types"]) == int(ref["n_types"]) == 12


# ---------------------------------------------------------------------- pair_data


def test_pair_data_shapes_and_onehot(tmp_path: Path) -> None:
    make_dataset(tmp_path, dim=6)
    m = load_manifest(tmp_path)
    bundle = load_features(tmp_path / "features.npz")
    rel = build_relation_data(bundle)

    slot = build_slot_inputs(bundle)
    assert slot["slot_feat"].shape == (8, 3 * 6 + 1)
    assert slot["type_feat"].shape == (8, 2 * 6)
    np.testing.assert_array_equal(build_slot_feat(bundle), slot["slot_feat"])
    np.testing.assert_array_equal(build_type_feat(bundle), slot["type_feat"])

    base = build_pair_base(bundle, rel)
    n_total = len(rel["frm"])
    assert base.shape == (n_total, 3 * 6 + 1)  # [after[frm], after[to], |diff|, dist]

    onehot = relation_onehot(rel["label"], n_classes=len(REL_TYPES) + 1)
    assert onehot.shape == (n_total, 4)
    assert np.all(onehot.sum(axis=1) == 1)
    assert np.all(onehot[rel["is_none"], 3] == 1)  # none → 最后一维（12 类时是 index 11）

    # d_in 决定是否拼 9 维几何
    no_geo = build_pair_feat(bundle, rel, d_in=base.shape[1])
    np.testing.assert_array_equal(no_geo, base)
    with_geo = build_pair_feat(bundle, rel, d_in=base.shape[1] + 9, manifest=m)
    assert with_geo.shape == (n_total, base.shape[1] + 9)
    np.testing.assert_allclose(with_geo[:, : base.shape[1]], base, rtol=0, atol=0)
    np.testing.assert_allclose(
        with_geo[:, base.shape[1]:], pair_geometry(rel, bbox_map_from_manifest(m)), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="不匹配"):
        build_pair_feat(bundle, rel, d_in=base.shape[1] + 3)
    with pytest.raises(ValueError, match="manifest/bbox_map"):
        build_pair_feat(bundle, rel, d_in=base.shape[1] + 9)


# ------------------------------------------- 真实特征切片逐位 parity（20 行 fixture）


def test_feature_assembly_parity_with_real_slice() -> None:
    """build_* 输出与服务器原始 slot_feat/type_feat/onehot/pair_base 逐位相等。"""
    slice_path = FIXTURES / "ref_features_slice.npz"
    meta_path = FIXTURES / "spatial_v2.meta.json"
    rel_path = FIXTURES / "ref_relation_data.npz"
    if not (slice_path.exists() and meta_path.exists() and rel_path.exists()):
        pytest.skip("缺少 ref_features_slice.npz / spatial_v2.meta.json / ref_relation_data.npz")

    sl = np.load(slice_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ref_rel = np.load(rel_path, allow_pickle=False)
    n_pos = int(np.argmax(ref_rel["is_none"]))
    assert len(sl["slot_after"]) == 20

    n_rows = len(sl["slot_after"])
    z = {
        "slot_before": sl["slot_before"],
        "slot_after": sl["slot_after"],
        "slot_mask": sl["slot_mask"],
        "slot_change": sl["slot_change"],
        "slot_type": sl["slot_type"],
        "slot_pair": np.zeros(n_rows, dtype=np.int32),
        "pair_split": np.asarray(["test"]),
    }
    slot = build_slot_inputs(z)
    np.testing.assert_array_equal(slot["slot_feat"], sl["slot_feat"])  # float32 逐位
    np.testing.assert_array_equal(slot["type_feat"], sl["type_feat"])

    onehot = relation_onehot(sl["rel_label"], n_classes=len(meta["rel_types"]) + 1)
    np.testing.assert_array_equal(onehot, sl["onehot"])

    # pair_base：只验证 frm/to 都落在前 20 个 slot 的行（切片未提供更多 slot 特征）
    frm, to = sl["rel_frm"], sl["rel_to"]
    ok = np.array([(int(f) < n_rows and int(t) < n_rows) for f, t in zip(frm, to)])
    assert int(ok.sum()) == 7  # fixture 里前 7 条关系可在切片内重建
    rel = {"frm": frm[ok], "to": to[ok], "dist": ref_rel["dist"][: len(frm)][ok]}
    base = build_pair_base(z, rel)
    np.testing.assert_array_equal(base, sl["pair_base"][ok])


# --------------------------------------------------- 真实维度回归（3D+1 / +9 / +12）


def test_head_input_dims_match_reference_oracle() -> None:
    """D=5120 时 pair 输入是 3D+1（v2 15361 / v2g 15370），rel_pos 再 +len(rel_types)+1=12。

    这直接对上 ``reference_oracle.json`` 里原始脚本记录的 ``_dims``，防止把 pair 拼装
    误写成 2D+1（那样 v2 会是 10241，与 checkpoint d_in 不符）。
    """
    slice_path = FIXTURES / "ref_features_slice.npz"
    meta_path = FIXTURES / "spatial_v2.meta.json"
    oracle_path = FIXTURES / "reference_oracle.json"
    if not (slice_path.exists() and meta_path.exists() and oracle_path.exists()):
        pytest.skip("缺少 ref_features_slice.npz / spatial_v2.meta.json / reference_oracle.json")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    with np.load(slice_path, allow_pickle=False) as sl:
        bundle = FeatureBundle(
            path=slice_path,
            arrays={"slot_after": sl["slot_after"], "slot_before": sl["slot_before"]},
            meta=meta,
        )
    dims = head_input_dims(bundle)
    assert feature_dim(bundle) == 5120
    assert pair_base_dim(bundle) == 15361 == oracle["pilot_v2"]["head_metrics"]["_dims"]["pair_feat"]
    assert dims["change"] == 15361 == oracle["pilot_v2"]["head_metrics"]["_dims"]["slot_feat"]
    assert dims["type"] == 10240
    assert dims["relation_base"] == 15361
    assert dims["relation_geo"] == 15370 == oracle["pilot_v2g"]["head_metrics"]["_dims"]["pair_feat"]
    assert dims["attrs_base"] == 15361 and dims["attrs_geo"] == 15370
    assert relation_onehot_dim(bundle) == len(meta["rel_types"]) + 1 == 12
    assert dims["rel_pos_base"] == 15361 + 12
    assert dims["rel_pos_geo"] == 15370 + 12
