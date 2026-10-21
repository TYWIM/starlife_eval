#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在缓存的 27B 特征上训练小分类头（方案 B：主干冻结，只训头）。

任务（空间帧对）：
    change   4 类（none/appeared/displaced/disappeared）
    type    22 类（闭集，试点头）
    relation 12 类（11 关系 + none，none 由同帧随机未关联槽对采样）
    rel_pos 10 类（关系类型 one-hot 作为条件输入）
    contact/support 二分类

另外可选：用物体数据训练 A 层 500 类的 type 头（检验容量扩展）。

用法：
    python train_heads.py --spatial ./features/spatial.npz \
        --out ./checkpoints/pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, list[float]]:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            f1s.append(float("nan"))
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    valid = [f for f in f1s if not np.isnan(f)]
    return (float(np.mean(valid)) if valid else 0.0), f1s


class Head(nn.Module):
    def __init__(self, d_in: int, n_out: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_out),
        )

    def forward(self, x):
        return self.net(x)


class PairHead(nn.Module):
    def __init__(self, d_in: int, n_out: int, extra: int = 0, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = Head(d_in + extra, n_out, hidden, dropout)


def train_classifier(name, Xtr, ytr, Xva, yva, Xte, yte, n_classes, device, epochs=40,
                     lr=1e-3, bs=256, out_dir: Path = None, extra_tr=None, extra_va=None, extra_te=None,
                     cw_power=1.0):
    def prep(X, E):
        X = torch.from_numpy(X.astype(np.float32))
        if E is not None:
            X = torch.cat([X, torch.from_numpy(E.astype(np.float32))], dim=1)
        return X.to(device)

    Xtr_t, Xva_t, Xte_t = prep(Xtr, extra_tr), prep(Xva, extra_va), prep(Xte, extra_te)
    ytr_t = torch.from_numpy(ytr.astype(np.int64)).to(device)
    yva_t = torch.from_numpy(yva.astype(np.int64)).to(device)
    yte_t = torch.from_numpy(yte.astype(np.int64)).to(device)

    freq = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    # v2: 逆频率加权（原来 1/sqrt），稀有类（appeared / back_left）不再被淹没
    weight = torch.from_numpy(1.0 / np.power(np.maximum(freq, 1.0), cw_power)).to(device)
    weight = weight / weight.mean()

    model = Head(Xtr_t.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    best_f1, best_state, best_epoch = -1.0, None, -1
    n = Xtr_t.size(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(1).cpu().numpy()
        f1, _ = macro_f1(yva, pred, n_classes)
        if f1 > best_f1:
            best_f1, best_epoch = f1, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    test_f1, per_class = macro_f1(yte, pred, n_classes)
    top1 = float((pred == yte).mean())
    if out_dir is not None:
        torch.save({"arch": "starlife_wsv", "state_dict": best_state, "n_classes": n_classes,
                    "d_in": int(Xtr_t.shape[1])}, out_dir / f"{name}.pt")
    print(f"[{name}] test top1={top1:.4f} macroF1={test_f1:.4f} (best epoch {best_epoch}, valF1={best_f1:.4f})", flush=True)
    return {"top1": top1, "macro_f1": test_f1, "per_class_f1": per_class,
            "best_epoch": best_epoch, "val_f1": best_f1, "n_test": int(len(yte))}


def load_spatial(path: Path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return z, meta


def load_manifest(dataset: str):
    """读取 manifest，返回 (rows, 全局 slot 索引 -> bbox)。"""
    rows = [json.loads(l) for l in Path(dataset, "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    bbox = {}
    offsets = [0]
    for r in rows:
        offsets.append(offsets[-1] + len(r["entities"]))
    for i, r in enumerate(rows):
        for k, e in enumerate(r["entities"]):
            bbox[offsets[i] + k] = np.asarray(e["bbox"], dtype=np.float32)
    return rows, bbox, offsets


def pair_geometry(rel, bbox, img_w: float = 1024.0, img_h: float = 768.0) -> np.ndarray:
    """bbox 几何特征（生产上 YOLO 框同样可得）：9 维。"""
    n = len(rel["frm"])
    if n == 0:
        return np.zeros((0, 9), dtype=np.float32)
    fa = np.stack([bbox[int(i)] for i in rel["frm"]])
    fb = np.stack([bbox[int(i)] for i in rel["to"]])
    sax, say = (fa[:, 0] + fa[:, 2]) * 0.5, (fa[:, 1] + fa[:, 3]) * 0.5
    hbx, hby = (fb[:, 0] + fb[:, 2]) * 0.5, (fb[:, 1] + fb[:, 3]) * 0.5
    haw = np.maximum((fa[:, 2] - fa[:, 0]) * 0.5, 1.0)
    hah = np.maximum((fa[:, 3] - fa[:, 1]) * 0.5, 1.0)
    hbw = np.maximum((fb[:, 2] - fb[:, 0]) * 0.5, 1.0)
    hbh = np.maximum((fb[:, 3] - fb[:, 1]) * 0.5, 1.0)
    dxn, dyn = (sax - hbx) / hbw, (say - hby) / hbh
    wa, ha = fa[:, 2] - fa[:, 0], fa[:, 3] - fa[:, 1]
    wb, hb = fb[:, 2] - fb[:, 0], fb[:, 3] - fb[:, 1]
    inter_w = np.clip(np.minimum(fa[:, 2], fb[:, 2]) - np.maximum(fa[:, 0], fb[:, 0]), 0, None)
    inter_h = np.clip(np.minimum(fa[:, 3], fb[:, 3]) - np.maximum(fa[:, 1], fb[:, 1]), 0, None)
    inter = inter_w * inter_h
    union = np.maximum(wa * ha + wb * hb - inter, 1.0)
    iou = inter / union
    dist = np.sqrt((sax - hbx) ** 2 + (say - hby) ** 2) / np.sqrt(img_w ** 2 + img_h ** 2)
    return np.stack([dxn, dyn, wa / wb, ha / hb, (wa * ha) / np.maximum(wb * hb, 1.0),
                     iou, dist, sax / img_w, say / img_h], axis=1).astype(np.float32)


def build_relation_data(z, meta):
    """生成正样本 + 采样负样本（同帧未关联槽对）。"""
    rel_from, rel_to = z["rel_from"], z["rel_to"]
    rel_type, rel_pos = z["rel_type"], z["rel_pos"]
    pair_split = z["pair_split"]
    slot_pair = z["slot_pair"]
    rng = np.random.default_rng(20261020)
    n_rel = len(rel_from)
    types = meta["rel_types"]
    n_types = len(types) + 1  # + none

    sample_mult = 2
    extra_from, extra_to = [], []
    slot_pairs = set(zip(rel_from.tolist(), rel_to.tolist()))
    for i in range(n_rel):
        pid = int(slot_pair[rel_from[i]])
        same_pair_slots = np.where(slot_pair == pid)[0]
        if len(same_pair_slots) < 2:
            continue
        for _ in range(sample_mult):
            a = int(rng.choice(same_pair_slots)); b = int(rng.choice(same_pair_slots))
            if a == b or (a, b) in slot_pairs:
                continue
            extra_from.append(a); extra_to.append(b)
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
                support=support, dist=dist, split=split, is_none=is_none, n_types=n_types,
                types=types, rel_pos_keys=meta["rel_pos_keys"])


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 StarLife 分类头（缓存特征）")
    ap.add_argument("--spatial", default="", help="空间数据集特征 npz")
    ap.add_argument("--dataset", default="", help="空间数据集目录（提供 manifest 以加入 bbox 几何特征）")
    ap.add_argument("--slot-geom", default="", help="slot 级几何特征 npz（build_slot_geometry.py 产物，给 change 头补位移）")
    ap.add_argument("--object", default="", help="物体数据集特征 npz（可选，训 A 层 type 头）")
    ap.add_argument("--object-catalog", default="", help="catalog.csv（过滤 A 层）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    results = {}

    if args.spatial:
        z, meta = load_spatial(Path(args.spatial))
        slot_after = z["slot_after"].astype(np.float32)
        slot_before = z["slot_before"].astype(np.float32)
        diff = np.abs(slot_after - slot_before)
        slot_feat = np.concatenate([slot_before, slot_after, diff,
                                    z["slot_mask"].astype(np.float32)[:, None]], axis=1)
        if args.slot_geom:
            g = np.load(args.slot_geom)
            geom = g["slot_geom"].astype(np.float32)
            if len(geom) != len(slot_after):
                raise SystemExit(f"slot 几何特征行数 {len(geom)} != slot 数 {len(slot_after)}")
            # z-score 标准化，避免欧氏位移量纲压过视觉特征
            mu, sd = geom.mean(0, keepdims=True), geom.std(0, keepdims=True) + 1e-6
            geom = (geom - mu) / sd
            slot_feat = np.concatenate([slot_feat, geom], axis=1)
            print(f"[geo] change 头加入 slot 几何特征 {geom.shape[1]} 维，slot_feat dim={slot_feat.shape[1]}", flush=True)
        slot_split = z["pair_split"][z["slot_pair"]]
        change_y = z["slot_change"].astype(np.int64)
        type_y = z["slot_type"].astype(np.int64)
        n_types = len(meta["type_keys"])

        mask = {s: slot_split == s for s in ("train", "val", "test")}
        results["change"] = train_classifier(
            "change", slot_feat[mask["train"]], change_y[mask["train"]],
            slot_feat[mask["val"]], change_y[mask["val"]],
            slot_feat[mask["test"]], change_y[mask["test"]], 4, device,
            epochs=args.epochs, out_dir=out)

        type_feat = np.concatenate([slot_after, diff], axis=1)
        results["type"] = train_classifier(
            "type", type_feat[mask["train"]], type_y[mask["train"]],
            type_feat[mask["val"]], type_y[mask["val"]],
            type_feat[mask["test"]], type_y[mask["test"]], n_types, device,
            epochs=args.epochs, out_dir=out)

        rel = build_relation_data(z, meta)
        rmask = {s: rel["split"] == s for s in ("train", "val", "test")}
        pair_feat = np.concatenate([slot_after[rel["frm"]], slot_after[rel["to"]],
                                    np.abs(slot_after[rel["frm"]] - slot_after[rel["to"]]),
                                    rel["dist"][:, None]], axis=1)
        if args.dataset:
            _, bbox_map, _ = load_manifest(args.dataset)
            pair_feat = np.concatenate([pair_feat, pair_geometry(rel, bbox_map)], axis=1)
            print(f"[geo] pair 几何特征已加入，pair_feat dim={pair_feat.shape[1]}", flush=True)
        onehot = np.zeros((len(rel["label"]), len(rel["types"]) + 1), dtype=np.float32)
        onehot[np.arange(len(rel["label"])), np.minimum(rel["label"], len(rel["types"]))] = 1
        pair_cond = np.concatenate([pair_feat, onehot], axis=1)

        results["relation"] = train_classifier(
            "relation", pair_feat[rmask["train"]], rel["label"][rmask["train"]],
            pair_feat[rmask["val"]], rel["label"][rmask["val"]],
            pair_feat[rmask["test"]], rel["label"][rmask["test"]], rel["n_types"], device,
            epochs=args.epochs, out_dir=out)

        pos_mask = {s: (rel["split"] == s) & (~rel["is_none"]) for s in ("train", "val", "test")}
        results["rel_pos"] = train_classifier(
            "rel_pos", pair_feat[pos_mask["train"]], rel["pos"][pos_mask["train"]],
            pair_feat[pos_mask["val"]], rel["pos"][pos_mask["val"]],
            pair_feat[pos_mask["test"]], rel["pos"][pos_mask["test"]],
            len(meta["rel_pos_keys"]), device, epochs=args.epochs, out_dir=out,
            extra_tr=onehot[pos_mask["train"]], extra_va=onehot[pos_mask["val"]],
            extra_te=onehot[pos_mask["test"]])

        # 属性：contact / support
        results["attrs"] = train_attr_head(rel, pair_feat, device, out, args.epochs)

        results["label_maps"] = {
            "change": ["none", "appeared", "displaced", "disappeared"],
            "type": meta["type_keys"], "relation": rel["types"] + ["none"],
            "rel_pos": meta["rel_pos_keys"],
        }

    if args.object and args.object_catalog:
        results["object_type"] = train_object_type(
            Path(args.object), Path(args.object_catalog), device, out, args.epochs)

    (out / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] metrics -> {out / 'metrics.json'}", flush=True)


def train_attr_head(rel, pair_feat, device, out: Path, epochs: int):
    model = Head(pair_feat.shape[1], 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    split = rel["split"]
    xtr = torch.from_numpy(pair_feat[split == "train"]).to(device)
    ytr = torch.from_numpy(np.stack([rel["contact"], rel["support"]], axis=1)[split == "train"]).float().to(device)
    xva = torch.from_numpy(pair_feat[split == "val"]).to(device)
    yva = np.stack([rel["contact"], rel["support"]], axis=1)[split == "val"]
    xte = torch.from_numpy(pair_feat[split == "test"]).to(device)
    yte = np.stack([rel["contact"], rel["support"]], axis=1)[split == "test"]
    best = (-1, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(xtr.size(0))
        for i in range(0, xtr.size(0), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = (model(xva).sigmoid() > 0.5).cpu().numpy().astype(int)
        f1 = float(np.mean([macro_f1(yva[:, k], pred[:, k], 2)[0] for k in range(2)]))
        if f1 > best[0]:
            best = (f1, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        pred = (model(xte).sigmoid() > 0.5).cpu().numpy().astype(int)
    f1s = [macro_f1(yte[:, k], pred[:, k], 2)[0] for k in range(2)]
    torch.save({"state_dict": best[1], "n_classes": 2, "d_in": pair_feat.shape[1]}, out / "attrs.pt")
    print(f"[attrs] contact F1={f1s[0]:.3f} support F1={f1s[1]:.3f}", flush=True)
    return {"contact_f1": f1s[0], "support_f1": f1s[1]}


def train_object_type(npz_path: Path, catalog_path: Path, device, out: Path, epochs: int):
    import csv

    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(npz_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    tier = {r["key"]: r["tier"] for r in csv.DictReader(catalog_path.open(encoding="utf-8-sig"))}
    keys = meta["type_keys"]
    keep = [i for i, k in enumerate(keys) if tier.get(k, "B") == "A"]
    keep_set = set(keep)
    remap = {old: new for new, old in enumerate(keep)}

    obj_type = z["obj_type"]
    sel = np.array([i for i in range(len(obj_type)) if int(obj_type[i]) in keep_set])
    y = np.array([remap[int(obj_type[i])] for i in sel], dtype=np.int64)
    X = z["crop"][sel].astype(np.float32)
    split = z["split"][z["obj_row"]][sel]
    print(f"[object-type] A 层类别 {len(keep)}，样本 {len(sel)}", flush=True)
    res = train_classifier("object_type_A", X[split == "train"], y[split == "train"],
                           X[split == "val"], y[split == "val"],
                           X[split == "test"], y[split == "test"], len(keep), device,
                           epochs=epochs, out_dir=out)
    res["n_classes"] = len(keep)
    return res


if __name__ == "__main__":
    main()
