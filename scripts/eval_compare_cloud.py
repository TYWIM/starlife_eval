"""对比评测：本地分类头 vs 云端 deepseek-flash（零样本）。

同一批 test 帧对、同一套标注：
- 本地：从缓存特征 npz 直接跑 change/type/relation/rel_pos/attrs 头
- 云端：把 before/after 两图 + 实体槽位表发给 deepseek-flash，要求输出同结构 JSON
输出：./eval_compare.json（含双方指标与云端延迟/用量）
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

import train_heads as T

API = "https://api.deepseek.com/v1/chat/completions"
CHANGE = ["none", "appeared", "displaced", "disappeared"]


def load_key() -> str:
    return Path(os.environ.get("DEEPSEEK_KEY_FILE", str(Path.home() / ".ds_key"))).read_text().strip()


def macro_f1(y_true, y_pred, n):
    f1s = []
    for c in range(n):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        if tp == 0:
            continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


def build_prompt(entities: list[dict]) -> str:
    slots = [{"slot": e["slot"], "bbox": [round(v, 1) for v in e["bbox"]]} for e in entities]
    return (
        "你是桌面场景的结构化标注器。给你同一场景的 before 和 after 两张图，以及图中实体的槽位与像素框。\n"
        f"实体：{json.dumps(slots, ensure_ascii=False)}\n"
        "只输出一个 JSON 对象，不要解释、不要代码块，格式：\n"
        '{"changes":[{"slot":0,"type":"none|appeared|displaced|disappeared"}],'
        '"relations":[{"from_slot":0,"to_slot":1,"type":"on|in|contains|held_by|part_of|belongs_to|near|next_to|facing|above|below",'
        '"rel_pos":"top|inside|front|back|left|right|front_left|front_right|back_left|back_right",'
        '"contact":true,"support":false}]}\n'
        "规则：changes 必须覆盖所有槽位（没变化写 none）；relations 只描述 after 帧里同时存在的实体，"
        "每个事实只写一次，from_slot 是主体、to_slot 是宿主。"
    )


def call_cloud(key: str, before: bytes, after: bytes, entities: list[dict], retries: int = 2):
    body = {
        "model": "deepseek-flash",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(entities)},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(before).decode()}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(after).decode()}},
            ],
        }],
        "temperature": 0,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body).encode()
    last = None
    for _ in range(retries + 1):
        req = urllib.request.Request(API, data=data, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read().decode())
            return payload, time.time() - t0
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2)
    raise RuntimeError(f"cloud call failed: {last}")


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i:j + 1])
        raise


def norm_pred(p) -> dict:
    if not isinstance(p, dict):
        return {"changes": {}, "relations": []}
    ch = p.get("changes", {})
    if isinstance(ch, list):
        nch = {}
        for x in ch:
            if isinstance(x, dict) and "slot" in x:
                try:
                    nch[int(x["slot"])] = str(x.get("type", "none"))
                except Exception:  # noqa: BLE001
                    pass
        ch = nch
    elif isinstance(ch, dict):
        ch = {int(k): v for k, v in ch.items()}
    else:
        ch = {}
    rel = p.get("relations", [])
    if isinstance(rel, dict):
        rel = list(rel.values())
    rel = [x for x in rel if isinstance(x, dict)]
    return {"changes": ch, "relations": rel}


def local_predictions(npz: Path, ckpt: Path, sample_pairs: set[int], dataset: Path | None = None):
    z = np.load(npz, allow_pickle=False)
    meta = json.loads(npz.with_suffix(".meta.json").read_text(encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def head(name, din, n):
        ck = torch.load(ckpt / f"{name}.pt", map_location=dev)
        m = T.Head(int(ck["d_in"]), int(ck["n_classes"])).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        return m

    slot_after = z["slot_after"].astype(np.float32)
    slot_before = z["slot_before"].astype(np.float32)
    diff = np.abs(slot_after - slot_before)
    slot_feat = np.concatenate([slot_before, slot_after, diff, z["slot_mask"].astype(np.float32)[:, None]], axis=1)
    type_feat = np.concatenate([slot_after, diff], axis=1)
    slot_pair = z["slot_pair"]

    rel = T.build_relation_data(z, meta)
    base = np.concatenate([slot_after[rel["frm"]], slot_after[rel["to"]],
                           np.abs(slot_after[rel["frm"]] - slot_after[rel["to"]]),
                           rel["dist"][:, None]], axis=1)
    # 依据 checkpoint 期望维度决定是否拼 bbox 几何特征（v2 外观版 / v2g 几何版通用）
    rel_din = int(torch.load(ckpt / "relation.pt", map_location="cpu")["d_in"])
    pair_feat = base
    if dataset is not None:
        _, bbox_map, _ = T.load_manifest(str(dataset))
        geo = T.pair_geometry(rel, bbox_map)
        if base.shape[1] + geo.shape[1] == rel_din:
            pair_feat = np.concatenate([base, geo], axis=1)
    onehot = np.zeros((len(rel["label"]), len(rel["types"]) + 1), dtype=np.float32)
    onehot[np.arange(len(rel["label"])), np.minimum(rel["label"], len(rel["types"]))] = 1

    h_change = head("change", slot_feat.shape[1], 4)
    h_type = head("type", type_feat.shape[1], len(meta["type_keys"]))
    h_rel = head("relation", pair_feat.shape[1], len(rel["types"]) + 1)
    h_pos = head("rel_pos", pair_feat.shape[1] + onehot.shape[1], len(meta["rel_pos_keys"]))
    h_attr = head("attrs", pair_feat.shape[1], 2)

    with torch.no_grad():
        change_p = h_change(torch.from_numpy(slot_feat).to(dev)).argmax(1).cpu().numpy()
        type_p = h_type(torch.from_numpy(type_feat).to(dev)).argmax(1).cpu().numpy()
        rel_logits = h_rel(torch.from_numpy(pair_feat).to(dev))
        rel_p = rel_logits.argmax(1).cpu().numpy()
        pos_in = np.concatenate([pair_feat, onehot], axis=1)
        pos_p = h_pos(torch.from_numpy(pos_in).to(dev)).argmax(1).cpu().numpy()
        attr_p = (h_attr(torch.from_numpy(pair_feat).to(dev)).sigmoid() > 0.5).cpu().numpy().astype(int)

    out = {}
    for k in np.where(np.isin(slot_pair, list(sample_pairs)))[0]:
        pid = int(slot_pair[k])
        out.setdefault(pid, {"changes": {}, "types": {}, "relations": []})
        out[pid]["changes"][int(k)] = CHANGE[int(change_p[k])]
        out[pid]["types"][int(k)] = meta["type_keys"][int(type_p[k])]
    for k in range(len(rel["frm"])):
        pid = int(slot_pair[rel["frm"][k]])
        if pid not in sample_pairs or rel["is_none"][k]:
            continue
        out[pid]["relations"].append({
            "frm": int(rel["frm"][k]), "to": int(rel["to"][k]),
            "type": rel["types"][int(rel_p[k])] if int(rel_p[k]) < len(rel["types"]) else "none",
            "rel_pos": meta["rel_pos_keys"][int(pos_p[k])],
            "contact": bool(attr_p[k, 0]), "support": bool(attr_p[k, 1]),
            "gt_type": rel["types"][int(rel["label"][k])],
            "gt_pos": meta["rel_pos_keys"][int(rel["pos"][k])],
            "gt_contact": bool(rel["contact"][k]), "gt_support": bool(rel["support"][k]),
        })
    return out, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./dataset_spatial_v2")
    ap.add_argument("--npz", default="./features/spatial_v2.npz")
    ap.add_argument("--ckpt", default="./checkpoints/pilot_v2")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", default="./eval_compare.json")
    args = ap.parse_args()

    ds = Path(args.dataset)
    rows = [json.loads(l) for l in (ds / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    test_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(test_idx, min(args.n, len(test_idx))))

    local, meta = local_predictions(Path(args.npz), Path(args.ckpt), set(sample), ds)
    raw_path = Path("./eval_cloud_raw.jsonl")
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[rec["pair_index"]] = rec
            except Exception:  # noqa: BLE001
                pass

    key = load_key()
    lat, usage = [], []
    for i in sample:
        if i in done:
            continue
        r = rows[i]
        before = (ds / r["before"]).read_bytes()
        after = (ds / r["after"]).read_bytes()
        payload, dt = call_cloud(key, before, after, r["entities"])
        text = payload["choices"][0]["message"]["content"]
        rec = {"pair_index": i, "pair_id": r["pair_id"], "raw": text, "latency_s": round(dt, 3),
               "usage": payload.get("usage", {})}
        with raw_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[i] = rec
        print(f"[cloud] {r['pair_id']} {dt:.2f}s", flush=True)

    # ---- 指标 ----
    def evaluate(get_pred):
        ch_true, ch_pred = [], []
        rel_total = rel_hit = 0
        pos_total = pos_hit = 0
        attr_total = attr_hit = 0
        for i in sample:
            r = rows[i]
            slots = [e["slot"] for e in r["entities"]]
            gt_change = {c["slot"]: c["type"] for c in r["changes"]}
            pred = get_pred(i)
            pch = pred.get("changes", {})
            for s in slots:
                ch_true.append(gt_change.get(s, "none"))
                ch_pred.append(pch.get(s, "none"))
            p_rel = {(x.get("from_slot"), x.get("to_slot")): x for x in pred.get("relations", [])}
            for g in r["relations"]:
                key_ = (g["from_slot"], g["to_slot"])
                rel_total += 1
                cand = p_rel.get(key_) or {}
                if cand.get("type") == g["type"]:
                    rel_hit += 1
                    if "rel_pos" in g:
                        pos_total += 1
                        if cand.get("rel_pos") == g["rel_pos"]:
                            pos_hit += 1
                    attr_total += 2
                    attr_hit += int(bool(cand.get("contact")) == bool(g["contact"]))
                    attr_hit += int(bool(cand.get("support")) == bool(g["support"]))
        return {
            "change_slots": len(ch_true),
            "change_top1": round(float(np.mean([a == b for a, b in zip(ch_true, ch_pred)])) if ch_true else 0, 4),
            "change_macro_f1": round(macro_f1([CHANGE.index(x) for x in ch_true],
                                              [CHANGE.index(x) for x in ch_pred], 4), 4),
            "relation_pairs": rel_total,
            "relation_type_acc": round(rel_hit / rel_total, 4) if rel_total else 0,
            "rel_pos_acc": round(pos_hit / pos_total, 4) if pos_total else 0,
            "attrs_acc": round(attr_hit / attr_total, 4) if attr_total else 0,
        }

    def cloud_pred(i):
        try:
            return norm_pred(parse_json(done[i]["raw"]))
        except Exception:  # noqa: BLE001
            return {"changes": {}, "relations": []}

    local_map = {}
    offsets = [0]
    for r in rows:
        offsets.append(offsets[-1] + len(r["entities"]))
    for i in sample:
        loc = local[i]
        ents = rows[i]["entities"]
        base = offsets[i]
        idx2slot = {base + k: e["slot"] for k, e in enumerate(ents)}
        local_map[i] = {
            "changes": {idx2slot.get(k, k): v for k, v in loc["changes"].items()},
            "relations": [{"from_slot": idx2slot.get(x["frm"], x["frm"]),
                           "to_slot": idx2slot.get(x["to"], x["to"]),
                           "type": x["type"], "rel_pos": x["rel_pos"],
                           "contact": x["contact"], "support": x["support"]}
                          for x in loc["relations"]],
        }

    cloud_lat = [done[i]["latency_s"] for i in sample if i in done]
    cloud_usage = [done[i]["usage"] for i in sample if i in done]
    report = {
        "n_pairs": len(sample),
        "local": evaluate(lambda i: local_map[i]),
        "cloud_deepseek_flash": evaluate(cloud_pred),
        "cloud_latency_s": {
            "mean": round(float(np.mean(cloud_lat)), 3) if cloud_lat else None,
            "median": round(float(np.median(cloud_lat)), 3) if cloud_lat else None,
            "max": round(float(np.max(cloud_lat)), 3) if cloud_lat else None,
        },
        "cloud_tokens": {
            "prompt": int(sum(u.get("prompt_tokens", 0) for u in cloud_usage)),
            "completion": int(sum(u.get("completion_tokens", 0) for u in cloud_usage)),
        },
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
