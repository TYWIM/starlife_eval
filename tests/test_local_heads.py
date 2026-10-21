"""models 层单测：Head 结构 / checkpoint 校验 / 本地五头适配器 / per-pair 转换。

全部用合成小维度 checkpoint（d_in 可以是 6*3+1=19，不必是 15361）验证：
- 适配器按 ``relation.pt`` 的 ``d_in`` 自动决定是否拼 9 维 bbox 几何；
- 预测与“参考脚本路径”（eval_compare_cloud.local_predictions 的独立复刻）逐元素一致；
- 负样本在 relation/rel_pos/attrs 尾部、n_relation_positives 正确；
- latencies/runtime_stats 是真实测量值（mean/p50/max 与实测列表一致）。

若设置了 ``STARLIFE_EVAL_FIXTURE_ROOT``（指向含 dataset_spatial_v2/、features/、
checkpoints/ 的真实产物目录），追加与 ``ref_pred_pilot_v2{,g}.npz`` 的逐元素 parity
（容差 0）。该 parity 已由服务器端到端 run 证明（``runs/v2g_pair/report.json`` 与
reference_oracle 全等到 1e-16），本地默认 skip、不往仓库塞全量产物。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from starlife_eval.dataset import (
    bbox_map_from_manifest,
    build_relation_data,
    load_features,
    load_manifest,
    pair_geometry,
)
from starlife_eval.models.heads import HEAD_NAMES, Head, load_head, load_heads
from starlife_eval.models.local_heads import LocalHeadsAdapter, head_predictions_to_pairs

FIXTURES = Path(__file__).resolve().parent / "fixtures"

TYPE_KEYS = ["book", "bottle", "mug", "pen", "table"]
REL_TYPES = ["near", "on", "part_of"]
REL_POS_KEYS = ["front", "back", "left"]
CHANGE_LABELS = ["none", "appeared", "displaced", "disappeared"]


def _bbox(pair: int, slot: int) -> list[float]:
    x0 = 10.0 * slot + pair
    y0 = 20.0 + 5.0 * pair + 3.0 * slot
    return [x0, y0, x0 + 40.0, y0 + 30.0]


def make_dataset(root: Path, *, dim: int = 6, seed: int = 7) -> dict[str, np.ndarray]:
    """合成 dataset：3 对（train/train/test），每对 3/3/2 个 slot，各 2 条关系。"""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    slots_per_pair = (3, 3, 2)
    splits = ("train", "train", "test")
    rows: list[dict] = []
    cols = {k: [] for k in (
        "slot_before", "slot_after", "slot_mask", "slot_present", "slot_change", "slot_type", "slot_pair",
        "rel_from", "rel_to", "rel_type", "rel_pos", "rel_contact", "rel_support", "rel_dist",
    )}
    slot_ptr = 0
    for i, n in enumerate(slots_per_pair):
        entities = [
            {"slot": k, "bbox": _bbox(i, k), "type_key": TYPE_KEYS[k % len(TYPE_KEYS)],
             "present_after": not (i == 1 and k == 2)}
            for k in range(n)
        ]
        changes = [{"slot": 1, "type": "appeared"}] if i == 0 else (
            [{"slot": 0, "type": "displaced"}, {"slot": 2, "type": "disappeared"}] if i == 1 else []
        )
        rels = [
            {"from_slot": 0, "to_slot": 1, "type": "near", "rel_pos": "front",
             "contact": False, "support": False, "dist_m": 0.336},
            {"from_slot": 1, "to_slot": 0, "type": "on", "rel_pos": "back",
             "contact": True, "support": False, "dist_m": 0.068},
        ]
        rows.append({
            "pair_id": f"{i:06d}", "split": splits[i], "vocab_version": "1.0.0",
            "before": f"frames/{i:06d}_before.jpg", "after": f"frames/{i:06d}_after.jpg",
            "entities": entities, "changes": changes, "relations": rels,
            "negative": False, "meta": {"lighting": "warm"},
        })
        change_map = {c["slot"]: c["type"] for c in changes}
        for k, e in enumerate(entities):
            cols["slot_before"].append(rng.standard_normal(dim).astype(np.float16))
            cols["slot_after"].append((rng.standard_normal(dim) + 0.25).astype(np.float16))
            cols["slot_mask"].append(0 if not e["present_after"] else 1)
            cols["slot_present"].append(1 if e["present_after"] else 0)
            cols["slot_change"].append(CHANGE_LABELS.index(change_map.get(k, "none")))
            cols["slot_type"].append(TYPE_KEYS.index(e["type_key"]))
            cols["slot_pair"].append(i)
        for rel in rels:
            cols["rel_from"].append(slot_ptr + rel["from_slot"])
            cols["rel_to"].append(slot_ptr + rel["to_slot"])
            cols["rel_type"].append(REL_TYPES.index(rel["type"]))
            cols["rel_pos"].append(REL_POS_KEYS.index(rel["rel_pos"]))
            cols["rel_contact"].append(int(rel["contact"]))
            cols["rel_support"].append(int(rel["support"]))
            cols["rel_dist"].append(rel["dist_m"])
        slot_ptr += n

    arrays = {
        "slot_before": np.stack(cols["slot_before"]).astype(np.float16),
        "slot_after": np.stack(cols["slot_after"]).astype(np.float16),
        "slot_mask": np.asarray(cols["slot_mask"], dtype=np.int8),
        "slot_present": np.asarray(cols["slot_present"], dtype=np.int8),
        "slot_change": np.asarray(cols["slot_change"], dtype=np.int8),
        "slot_type": np.asarray(cols["slot_type"], dtype=np.int32),
        "slot_pair": np.asarray(cols["slot_pair"], dtype=np.int32),
        "pair_split": np.asarray(splits),
        "rel_from": np.asarray(cols["rel_from"], dtype=np.int32),
        "rel_to": np.asarray(cols["rel_to"], dtype=np.int32),
        "rel_type": np.asarray(cols["rel_type"], dtype=np.int32),
        "rel_pos": np.asarray(cols["rel_pos"], dtype=np.int32),
        "rel_contact": np.asarray(cols["rel_contact"], dtype=np.int8),
        "rel_support": np.asarray(cols["rel_support"], dtype=np.int8),
        "rel_dist": np.asarray(cols["rel_dist"], dtype=np.float32),
    }
    meta = {
        "mode": "spatial", "pairs": len(rows), "slots": len(cols["slot_pair"]), "dim": dim,
        "type_keys": TYPE_KEYS, "rel_types": REL_TYPES, "rel_pos_keys": REL_POS_KEYS,
        "change_keys": CHANGE_LABELS, "types_count": len(TYPE_KEYS),
        "relations": len(cols["rel_from"]),
    }
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    np.savez(root / "features.npz", **arrays)
    (root / "features.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return arrays


def expected_head_dims(dim: int, *, geometry: bool) -> dict[str, tuple[int, int]]:
    """name -> (d_in, n_out)，按 train_heads.main 的拼装规则推导。"""
    pair_base = 3 * dim + 1
    pair_dim = pair_base + 9 if geometry else pair_base
    return {
        "change": (3 * dim + 1, len(CHANGE_LABELS)),
        "type": (2 * dim, len(TYPE_KEYS)),
        "relation": (pair_dim, len(REL_TYPES) + 1),
        "rel_pos": (pair_dim + len(REL_TYPES) + 1, len(REL_POS_KEYS)),
        "attrs": (pair_dim, 2),
    }


def write_checkpoints(ckpt_dir: Path, dims: dict[str, tuple[int, int]], seed: int = 0) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name in HEAD_NAMES:
        d_in, n_out = dims[name]
        torch.manual_seed(seed + HEAD_NAMES.index(name))
        head = Head(d_in, n_out)
        torch.save(
            {"arch": "starlife_wsv", "state_dict": head.state_dict(), "n_classes": n_out, "d_in": d_in},
            ckpt_dir / f"{name}.pt",
        )


def reference_forward(arrays: dict[str, np.ndarray], bundle, manifest, ckpt_dir: Path) -> dict[str, np.ndarray]:
    """eval_compare_cloud.local_predictions 的独立复刻（显式拼装，不调 build_slot_feat 等）。"""
    slot_after = arrays["slot_after"].astype(np.float32)
    slot_before = arrays["slot_before"].astype(np.float32)
    diff = np.abs(slot_after - slot_before)
    slot_feat = np.concatenate(
        [slot_before, slot_after, diff, arrays["slot_mask"].astype(np.float32)[:, None]], axis=1
    )
    type_feat = np.concatenate([slot_after, diff], axis=1)

    rel = build_relation_data(bundle)
    frm, to = rel["frm"], rel["to"]
    pair = np.concatenate(
        [slot_after[frm], slot_after[to], np.abs(slot_after[frm] - slot_after[to]), rel["dist"][:, None]], axis=1
    )
    rel_din = int(torch.load(ckpt_dir / "relation.pt", map_location="cpu")["d_in"])
    if pair.shape[1] + 9 == rel_din:
        pair = np.concatenate([pair, pair_geometry(rel, bbox_map_from_manifest(manifest))], axis=1)
    else:
        assert pair.shape[1] == rel_din
    onehot = np.zeros((len(rel["label"]), len(rel["types"]) + 1), dtype=np.float32)
    onehot[np.arange(len(rel["label"])), np.minimum(rel["label"], len(rel["types"]))] = 1
    pos_input = np.concatenate([pair, onehot], axis=1)

    models = load_heads(ckpt_dir)
    with torch.no_grad():
        return {
            "change": models["change"].model(torch.from_numpy(slot_feat)).argmax(1).numpy(),
            "type": models["type"].model(torch.from_numpy(type_feat)).argmax(1).numpy(),
            "relation": models["relation"].model(torch.from_numpy(pair)).argmax(1).numpy(),
            "rel_pos": models["rel_pos"].model(torch.from_numpy(pos_input)).argmax(1).numpy(),
            "attrs": (models["attrs"].model(torch.from_numpy(pair)).sigmoid() > 0.5).numpy().astype(np.int64),
        }


# ------------------------------------------------------------------------ Head/load


def test_head_structure_matches_train_heads() -> None:
    head = Head(17, 4)
    names = [m.__class__.__name__ for m in head.net]
    assert names == ["LayerNorm", "Linear", "GELU", "Dropout", "Linear", "GELU", "Dropout", "Linear"]
    assert head.net[0].normalized_shape == (17,)
    assert head.net[1].out_features == 512
    assert head.net[4].out_features == 256
    assert head.net[7].out_features == 4
    assert head.net[3].p == 0.2 and head.net[6].p == 0.2


def test_load_head_validates_dims(tmp_path: Path) -> None:
    dims = expected_head_dims(6, geometry=False)
    write_checkpoints(tmp_path / "ckpt", dims)
    ck = load_head(tmp_path / "ckpt" / "change.pt", name="change", expected_d_in=19, expected_n_classes=4)
    assert ck.d_in == 19 and ck.n_classes == 4 and ck.arch == "starlife_wsv"
    assert ck.name == "change"
    with pytest.raises(ValueError, match="d_in"):
        load_head(tmp_path / "ckpt" / "change.pt", expected_d_in=18)
    with pytest.raises(ValueError, match="n_classes"):
        load_head(tmp_path / "ckpt" / "change.pt", expected_n_classes=5)
    with pytest.raises(FileNotFoundError):
        load_head(tmp_path / "ckpt" / "missing.pt")
    with pytest.raises(FileNotFoundError):
        load_heads(tmp_path / "nope")


# -------------------------------------------------------------- adapter predictions


def test_local_adapter_geometry_variant_matches_reference(tmp_path: Path) -> None:
    arrays = make_dataset(tmp_path / "ds", dim=6)
    manifest = load_manifest(tmp_path / "ds")
    bundle = load_features(tmp_path / "ds" / "features.npz")
    ckpt_dir = tmp_path / "ckpt_geo"
    write_checkpoints(ckpt_dir, expected_head_dims(6, geometry=True))

    adapter = LocalHeadsAdapter(ckpt_dir, bundle, manifest)
    assert adapter.geometry_used is True
    head = adapter.predict(measure_latency_pairs=[0, 2])
    ref = reference_forward(arrays, bundle, manifest, ckpt_dir)

    for key in ("change", "type", "relation", "rel_pos", "attrs"):
        np.testing.assert_array_equal(getattr(head, key), ref[key], err_msg=f"{key} 与参考路径不一致")

    # 负样本长度与位置
    rel = adapter.relation_data
    n_neg = int(np.sum(rel["is_none"]))
    assert head.n_relation_positives == int(np.sum(~rel["is_none"])) == 6
    assert len(head.relation) == len(rel["frm"]) == 6 + n_neg
    assert n_neg > 0
    assert len(head.rel_pos) == len(head.relation)
    assert head.attrs.shape == (len(head.relation), 2)
    assert head.attrs.dtype == np.int64 and head.change.dtype == np.int64

    # 延迟是真实测量：2 个 pair、均值与列表一致
    assert len(adapter.latency_per_pair_s) == 2
    assert all(x >= 0.0 for x in adapter.latency_per_pair_s)
    assert head.latency_s == pytest.approx(float(np.mean(adapter.latency_per_pair_s)))


def test_local_adapter_appearance_variant_no_geometry(tmp_path: Path) -> None:
    arrays = make_dataset(tmp_path / "ds", dim=6)
    manifest = load_manifest(tmp_path / "ds")
    bundle = load_features(tmp_path / "ds" / "features.npz")
    ckpt_dir = tmp_path / "ckpt_plain"
    write_checkpoints(ckpt_dir, expected_head_dims(6, geometry=False))

    adapter = LocalHeadsAdapter(ckpt_dir, bundle, manifest)
    assert adapter.geometry_used is False
    head = adapter.predict(measure_latency_pairs=[])
    ref = reference_forward(arrays, bundle, manifest, ckpt_dir)
    for key in ("change", "type", "relation", "rel_pos", "attrs"):
        np.testing.assert_array_equal(getattr(head, key), ref[key], err_msg=key)
    assert head.latency_s is None and adapter.latency_per_pair_s == []

    # 无 manifest 也能推理（v2 外观版不需要 bbox）
    adapter_no_manifest = LocalHeadsAdapter(ckpt_dir, bundle, None)
    head2 = adapter_no_manifest.predict(measure_latency_pairs=[])
    np.testing.assert_array_equal(head2.relation, head.relation)


def test_local_adapter_rejects_wrong_d_in(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds", dim=6)
    manifest = load_manifest(tmp_path / "ds")
    bundle = load_features(tmp_path / "ds" / "features.npz")
    dims = expected_head_dims(6, geometry=False)
    dims["relation"] = (dims["relation"][0] + 3, dims["relation"][1])  # 既非 base 也非 base+9
    write_checkpoints(tmp_path / "bad_ckpt", dims)
    with pytest.raises(ValueError, match="d_in"):
        LocalHeadsAdapter(tmp_path / "bad_ckpt", bundle, manifest)


def test_local_adapter_requires_manifest_for_geometry(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds", dim=6)
    bundle = load_features(tmp_path / "ds" / "features.npz")
    write_checkpoints(tmp_path / "ckpt_geo", expected_head_dims(6, geometry=True))
    with pytest.raises(ValueError, match="manifest"):
        LocalHeadsAdapter(tmp_path / "ckpt_geo", bundle, None)


# ---------------------------------------------------------------- per-pair conversion


def test_head_predictions_to_pairs_uses_pair_slot_ids(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds", dim=6)
    manifest = load_manifest(tmp_path / "ds")
    bundle = load_features(tmp_path / "ds" / "features.npz")
    ckpt_dir = tmp_path / "ckpt_geo"
    write_checkpoints(ckpt_dir, expected_head_dims(6, geometry=True))
    adapter = LocalHeadsAdapter(ckpt_dir, bundle, manifest)
    head = adapter.predict(measure_latency_pairs=[])
    rel = adapter.relation_data

    pairs = head_predictions_to_pairs(head, bundle, manifest, [0, 2], relation_data=rel)
    assert sorted(pairs) == [0, 2]
    # pair 0 有 3 个 slot（含 none），pair 2 有 2 个 slot
    assert sorted(pairs[0].changes) == [0, 1, 2]
    assert sorted(pairs[2].changes) == [0, 1]
    assert pairs[2].changes[0] == CHANGE_LABELS[int(head.change[manifest.slot_offsets[2] + 0])]  # 全局 6
    assert pairs[2].changes[1] == CHANGE_LABELS[int(head.change[manifest.slot_offsets[2] + 1])]  # 全局 7
    assert pairs[2].types[0] == bundle.type_keys[int(head.type[6])]
    # 只保留真关系，from/to 是 pair 内 slot 编号
    assert len(pairs[0].relations) == 2 and len(pairs[2].relations) == 2
    for pred in pairs[0].relations:
        assert 0 <= pred.from_slot < 3 and 0 <= pred.to_slot < 3
        assert pred.type in REL_TYPES and pred.rel_pos in REL_POS_KEYS
        assert isinstance(pred.contact, bool) and isinstance(pred.support, bool)

    # 与 evaluate 口径同源：正样本 3 对 × 2 条；负样本绝不出现在 pairs 里
    n_pos = int(np.sum(~rel["is_none"]))
    n_neg = int(np.sum(rel["is_none"]))
    assert n_pos == 6 and n_neg > 0 and len(rel["frm"]) == n_pos + n_neg
    total_relations = sum(len(p.relations) for p in pairs.values())
    assert total_relations == 4


def test_runtime_stats_matches_measured_latencies(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds", dim=6)
    manifest = load_manifest(tmp_path / "ds")
    bundle = load_features(tmp_path / "ds" / "features.npz")
    ckpt_dir = tmp_path / "ckpt_geo"
    write_checkpoints(ckpt_dir, expected_head_dims(6, geometry=True))
    adapter = LocalHeadsAdapter(ckpt_dir, bundle, manifest)
    adapter.predict(measure_latency_pairs=[0, 1, 2])

    lat = np.asarray(adapter.latency_per_pair_s, dtype=np.float64) * 1000.0
    stats = adapter.runtime_stats()
    assert stats.n_calls == 3
    assert stats.cost_usd == 0.0 and stats.cost_basis is not None
    assert stats.source == "measured"
    assert stats.latency_ms_mean == pytest.approx(float(lat.mean()))
    assert stats.latency_ms_p50 == pytest.approx(float(np.percentile(lat, 50)))
    assert stats.latency_ms_max == pytest.approx(float(lat.max()))


def test_adapter_predict_pairs(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds", dim=6)
    manifest = load_manifest(tmp_path / "ds")
    bundle = load_features(tmp_path / "ds" / "features.npz")
    ckpt_dir = tmp_path / "ckpt_geo"
    write_checkpoints(ckpt_dir, expected_head_dims(6, geometry=True))
    adapter = LocalHeadsAdapter(ckpt_dir, bundle, manifest)
    pairs = adapter.predict_pairs([2])
    assert sorted(pairs) == [2]
    assert sorted(pairs[2].changes) == [0, 1]


def test_local_adapter_real_dim_autodetect(tmp_path: Path) -> None:
    """真实维度（D=5120 ⇒ 15361 / 15370）下 d_in 自动判别，防止 2D/3D 拼装回归。"""
    dim = 5120
    dims_plain = expected_head_dims(dim, geometry=False)
    dims_geo = expected_head_dims(dim, geometry=True)
    assert dims_plain["relation"][0] == 3 * dim + 1 == 15361
    assert dims_geo["relation"][0] == 3 * dim + 10 == 15370
    assert dims_geo["rel_pos"][0] == 15370 + len(REL_TYPES) + 1  # + onehot(len(rel_types)+1)
    assert dims_plain["change"][0] == 15361 and dims_plain["type"][0] == 10240

    arrays = make_dataset(tmp_path / "ds", dim=dim)
    manifest = load_manifest(tmp_path / "ds")
    bundle = load_features(tmp_path / "ds" / "features.npz")
    for tag, geometry in (("geo", True), ("plain", False)):
        ckpt_dir = tmp_path / f"ckpt_{tag}"
        write_checkpoints(ckpt_dir, dims_geo if geometry else dims_plain)
        adapter = LocalHeadsAdapter(ckpt_dir, bundle, manifest)
        assert adapter.geometry_used is geometry
        head = adapter.predict(measure_latency_pairs=[])
        ref = reference_forward(arrays, bundle, manifest, ckpt_dir)
        for key in ("change", "type", "relation", "rel_pos", "attrs"):
            np.testing.assert_array_equal(getattr(head, key), ref[key], err_msg=f"{tag}:{key}")


# ------------------------------------------------- 全量真实产物 parity（Lead 同步后启用）


def _real_artifacts() -> tuple[Path, Path, Path]:
    root = Path(os.environ.get("STARLIFE_EVAL_FIXTURE_ROOT", str(FIXTURES / "full")))
    return root / "dataset_spatial_v2", root / "features" / "spatial_v2.npz", root / "checkpoints"


@pytest.mark.parametrize("name", ["pilot_v2", "pilot_v2g"])
def test_local_adapter_parity_with_full_real_predictions(name: str) -> None:
    """与服务器 ``ref_pred_<name>.npz`` 逐元素相等（容差 0）。

    仅在设置 ``STARLIFE_EVAL_FIXTURE_ROOT`` 时运行：本地头逐位正确性已由服务器端到端
    run 证明（``runs/v2g_pair/report.json`` 的头级/协议指标与 reference_oracle 全等），
    不需要把 ~500MB 全量产物放进仓库；这里保留可复跑入口。
    """
    if not os.environ.get("STARLIFE_EVAL_FIXTURE_ROOT"):
        pytest.skip(
            "本地默认 skip：已由服务器端到端 run 证明（runs/v2g_pair/report.json 与 "
            "reference_oracle 全等到 1e-16）；如需本地复跑，设 STARLIFE_EVAL_FIXTURE_ROOT"
            "指向含 dataset_spatial_v2/、features/、checkpoints/ 的目录"
        )
    dataset_dir, npz_path, ckpt_root = _real_artifacts()
    ref_path = FIXTURES / f"ref_pred_{name}.npz"
    if not (dataset_dir.exists() and npz_path.exists() and (ckpt_root / name).exists() and ref_path.exists()):
        pytest.skip(f"STARLIFE_EVAL_FIXTURE_ROOT 下缺少真实产物（{dataset_dir} / {npz_path} / {ckpt_root / name}）")

    manifest = load_manifest(dataset_dir)
    bundle = load_features(npz_path, dataset_dir / "features" / "spatial_v2.meta.json")
    adapter = LocalHeadsAdapter(ckpt_root / name, bundle, manifest)
    assert adapter.geometry_used == (name == "pilot_v2g")
    head = adapter.predict(measure_latency_pairs=[])

    ref = np.load(ref_path, allow_pickle=False)
    np.testing.assert_array_equal(head.change, ref["change"], err_msg="change 预测不逐位相等")
    np.testing.assert_array_equal(head.type, ref["type"], err_msg="type 预测不逐位相等")
    np.testing.assert_array_equal(head.relation, ref["relation"], err_msg="relation 预测不逐位相等")
    np.testing.assert_array_equal(head.rel_pos, ref["rel_pos"], err_msg="rel_pos 预测不逐位相等")
    np.testing.assert_array_equal(head.attrs, ref["attrs"], err_msg="attrs 预测不逐位相等")
    assert head.n_relation_positives + int(ref["n_none"]) == len(ref["relation"])
