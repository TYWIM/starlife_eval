#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用冻结的 Qwen3.8-27B 把两个试点数据集抽成特征缓存，供分类头训练。

思路（方案 B：主干冻结，只训小头）：
- 整帧、物体区域（crop）都送进主干，取最后一层 hidden state 的最后一个 token 作为特征；
- 特征存成 npz（f16），训练脚本只读缓存，不重复跑 27B；
- 支持 spatial（帧对）与 object（单图）两种模式。

用法：
    python extract_features.py --mode spatial --model /root/models/Qwen3.8-27B \
        --dataset ./dataset_spatial_pilot --out ./features/spatial.npz
    python extract_features.py --mode object --model ... \
        --dataset ./dataset_object_pilot --out ./features/object.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def patch_vision_patch_embed(model) -> bool:
    """把视觉塔 patch_embed 的 Conv3d 换成等价 matmul。

    本机实测（A800 80G / torch 2.9.0+cu128 / cudnn 9.8）：bf16 Conv3d 一次
    patch_embed 要 2027 ms，而数学上等价的 matmul 只要 0.03 ms，慢约 6 万倍。
    这是 cuDNN 3D 卷积的算子回退问题，不是算力不足。替换后单帧编码从
    25 s 降到 0.05 s，输出最大误差 0.03（bf16 舍入级别）。
    """
    visual = getattr(getattr(model, "model", None), "visual", None)
    pe = getattr(visual, "patch_embed", None)
    proj = getattr(pe, "proj", None)
    if pe is None or proj is None:
        return False

    def fast_forward(hidden_states):  # 等同 kernel=stride 的 Conv3d，无 padding
        target = proj.weight.dtype
        n = hidden_states.shape[0]
        x = hidden_states.view(
            n, pe.in_channels, pe.temporal_patch_size, pe.patch_size, pe.patch_size
        ).to(dtype=target)
        x = x.reshape(n, -1)
        w = proj.weight.reshape(pe.embed_dim, -1)
        return (x @ w.t() + proj.bias).view(-1, pe.embed_dim)

    pe.forward = fast_forward
    print("[patch] vision patch_embed: Conv3d -> matmul", flush=True)
    return True


def load_model(model_dir: str):
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as MM
    except ImportError:
        MM = None

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=False)
    candidates = [c for c in (MM, AutoModelForCausalLM, AutoModel) if c is not None]
    last = None
    for cls in candidates:
        for trust in (False, True):
            try:
                try:
                    model = cls.from_pretrained(
                        model_dir, dtype=torch.bfloat16, device_map="auto",
                        low_cpu_mem_usage=True, trust_remote_code=trust,
                    )
                except TypeError:
                    model = cls.from_pretrained(
                        model_dir, torch_dtype=torch.bfloat16, device_map="auto",
                        low_cpu_mem_usage=True, trust_remote_code=trust,
                    )
                model.eval()
                print(f"[model] {cls.__name__} trust={trust}", flush=True)
                patch_vision_patch_embed(model)
                return model, processor
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"[model] {cls.__name__} trust={trust} failed: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
    raise RuntimeError(f"模型加载失败: {last}")


@torch.inference_mode()
def encode_images(model, processor, images: list[Image.Image], prompt: str,
                  batch_size: int, prefetch: int = 0) -> np.ndarray:
    """返回 (N, D) float32，最后一层最后 token。prefetch>0 时用独立线程并行做 CPU 预处理。"""
    import queue
    import threading

    device = next(model.parameters()).device
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def prep(chunk):
        inputs = processor(text=[text] * len(chunk), images=chunk, return_tensors="pt", padding=True)
        return {k: v for k, v in inputs.items()}

    def run_batch(inputs):
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        out = model(**inputs, output_hidden_states=True, use_cache=False)
        hs = getattr(out, "hidden_states", None)
        if hs is None:
            hs = out[1]
        last = hs[-1]
        mask = inputs.get("attention_mask")
        if mask is None:
            idx = torch.full((last.size(0),), last.size(1) - 1, device=last.device)
        else:
            idx = mask.sum(dim=1) - 1
        pooled = last[torch.arange(last.size(0), device=last.device), idx]
        return pooled.float().cpu().numpy()

    feats: list[np.ndarray] = []
    n = len(images)
    t0 = time.time()
    if prefetch <= 0:
        for start in range(0, n, batch_size):
            feats.append(run_batch(prep(images[start:start + batch_size])))
            done = min(start + batch_size, n)
            if done % 256 == 0 or done == n:
                print(f"  encoded {done}/{n} ({done / max(time.time() - t0, 1e-6):.1f}/s)", flush=True)
        return np.concatenate(feats, axis=0)

    q: queue.Queue = queue.Queue(maxsize=prefetch * 2)

    def producer():
        try:
            for start in range(0, n, batch_size):
                q.put(prep(images[start:start + batch_size]))
        finally:
            q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    done = 0
    while True:
        item = q.get()
        if item is None:
            break
        feats.append(run_batch(item))
        done += next(iter(item.values())).shape[0] if isinstance(next(iter(item.values())), torch.Tensor) else batch_size
        if done % 256 < batch_size or done >= n:
            print(f"  encoded {done}/{n} ({done / max(time.time() - t0, 1e-6):.1f}/s)", flush=True)
    th.join(timeout=1.0)
    return np.concatenate(feats, axis=0)


def crop(img: Image.Image, bbox: list[float], margin: float = 0.12,
         size: int = 224) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1 = max(0, int(x1 - mx)); y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx)); y2 = min(h, int(y2 + my))
    if x2 - x1 < 4 or y2 - y1 < 4:
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, x1 + 8), min(h, y1 + 8)
    return img.crop((x1, y1, x2, y2)).resize((size, size))


SPATIAL_PROMPT = "这是桌面上的一帧画面。请描述画面中央这个物体的类别与外观特征。"
OBJECT_PROMPT = "请描述画面中央这个物体的类别与外观特征。"


def run_spatial(args, model, processor) -> None:
    ds = Path(args.dataset)
    rows = [json.loads(l) for l in (ds / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[spatial] {len(rows)} pairs", flush=True)

    frame_before, frame_after = [], []
    slot_before, slot_after, slot_mask = [], [], []
    pair_of_slot, slot_present = [], []
    for i, r in enumerate(rows):
        b = Image.open(ds / r["before"]).convert("RGB")
        a = Image.open(ds / r["after"]).convert("RGB")
        if not args.skip_frames:
            frame_before.append(b); frame_after.append(a)
        for e in r["entities"]:
            box = e["bbox"]
            slot_before.append(crop(b, box, size=args.crop_size))
            slot_after.append(crop(a, box, size=args.crop_size))
            slot_mask.append(0 if not e.get("present_after", True) else 1)
            pair_of_slot.append(i)
            slot_present.append(int(bool(e.get("present_after", True))))
        if (i + 1) % 200 == 0:
            print(f"  loaded {i + 1}/{len(rows)}", flush=True)

    if args.skip_frames:
        print("[spatial] skip frame encoding", flush=True)
        fb = np.zeros((len(rows), 0), dtype=np.float16)
        fa = fb
        dim = 5120
    else:
        print("[spatial] encode frames", flush=True)
        fb = encode_images(model, processor, frame_before, SPATIAL_PROMPT, args.batch_size, args.prefetch)
        fa = encode_images(model, processor, frame_after, SPATIAL_PROMPT, args.batch_size, args.prefetch)
        dim = fb.shape[1]
    print("[spatial] encode before crops (absent slots -> zero)", flush=True)
    present_before = [k for k, m in enumerate(slot_mask) if m]
    sb = np.zeros((len(slot_before), dim), dtype=np.float32)
    if present_before:
        sb[present_before] = encode_images(
            model, processor, [slot_before[k] for k in present_before], SPATIAL_PROMPT, args.batch_size,
        args.prefetch)
    print("[spatial] encode after crops", flush=True)
    sa = encode_images(model, processor, slot_after, SPATIAL_PROMPT, args.batch_size, args.prefetch)

    # 标签
    change_map = {"none": 0, "appeared": 1, "displaced": 2, "disappeared": 3}
    type_keys = sorted({e["type_key"] for r in rows for e in r["entities"]})
    type_idx = {k: i for i, k in enumerate(type_keys)}
    rel_types = sorted({rel["type"] for r in rows for rel in r.get("relations", [])})
    rel_pos_keys = sorted({rel["rel_pos"] for r in rows for rel in r.get("relations", [])})

    slot_change, slot_type = [], []
    rel_from, rel_to, rel_type, rel_pos, rel_contact, rel_support, rel_dist = [], [], [], [], [], [], []
    slot_ptr = 0
    for r in rows:
        changes = {c["slot"]: c["type"] for c in r.get("changes", [])}
        for e in r["entities"]:
            slot_change.append(change_map.get(changes.get(e["slot"], "none"), 0))
            slot_type.append(type_idx[e["type_key"]])
        base = slot_ptr
        for rel in r.get("relations", []):
            rel_from.append(base + rel["from_slot"])
            rel_to.append(base + rel["to_slot"])
            rel_type.append(rel_types.index(rel["type"]))
            rel_pos.append(rel_pos_keys.index(rel["rel_pos"]))
            rel_contact.append(int(bool(rel.get("contact"))))
            rel_support.append(int(bool(rel.get("support"))))
            rel_dist.append(float(rel.get("dist_m", 0.0)))
        slot_ptr += len(r["entities"])

    splits = [r["split"] for r in rows]
    pair_of_slot_arr = np.array(pair_of_slot, dtype=np.int32)
    np.savez_compressed(
        args.out,
        img_before=fb.astype(np.float16), img_after=fa.astype(np.float16),
        slot_before=sb.astype(np.float16), slot_after=sa.astype(np.float16),
        slot_mask=np.array(slot_mask, dtype=np.int8),
        slot_present=np.array(slot_present, dtype=np.int8),
        slot_change=np.array(slot_change, dtype=np.int8),
        slot_type=np.array(slot_type, dtype=np.int32),
        slot_pair=pair_of_slot_arr,
        pair_split=np.array(splits),
        rel_from=np.array(rel_from, dtype=np.int32),
        rel_to=np.array(rel_to, dtype=np.int32),
        rel_type=np.array(rel_type, dtype=np.int32),
        rel_pos=np.array(rel_pos, dtype=np.int32),
        rel_contact=np.array(rel_contact, dtype=np.int8),
        rel_support=np.array(rel_support, dtype=np.int8),
        rel_dist=np.array(rel_dist, dtype=np.float32),
    )
    meta = {
        "mode": "spatial", "pairs": len(rows), "slots": len(slot_change),
        "dim": int(fb.shape[1]), "type_keys": type_keys, "rel_types": rel_types,
        "rel_pos_keys": rel_pos_keys, "change_keys": list(change_map),
        "types_count": len(type_keys), "relations": len(rel_type),
    }
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[spatial] done {args.out}  dim={meta['dim']} slots={meta['slots']} rels={meta['relations']}", flush=True)


def run_object(args, model, processor) -> None:
    ds = Path(args.dataset)
    rows = [json.loads(l) for l in (ds / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[object] {len(rows)} images", flush=True)

    images, crops_img, obj_row, obj_type = [], [], [], []
    type_keys = sorted({o["type_key"] for r in rows for o in r["objects"]})
    type_idx = {k: i for i, k in enumerate(type_keys)}
    for i, r in enumerate(rows):
        img = Image.open(ds / r["image"]).convert("RGB")
        if not args.skip_frames:
            images.append(img)
        for o in r["objects"]:
            crops_img.append(crop(img, o["bbox"], size=args.crop_size))
            obj_row.append(i)
            obj_type.append(type_idx[o["type_key"]])
        if (i + 1) % 200 == 0:
            print(f"  loaded {i + 1}/{len(rows)}", flush=True)

    if args.skip_frames:
        print("[object] skip image encoding", flush=True)
        fi = np.zeros((len(rows), 0), dtype=np.float16)
    else:
        print("[object] encode images", flush=True)
        fi = encode_images(model, processor, images, OBJECT_PROMPT, args.batch_size, args.prefetch)
    print("[object] encode crops", flush=True)
    fc = encode_images(model, processor, crops_img, OBJECT_PROMPT, args.batch_size, args.prefetch)

    np.savez_compressed(
        args.out,
        img=fi.astype(np.float16), crop=fc.astype(np.float16),
        obj_row=np.array(obj_row, dtype=np.int32),
        obj_type=np.array(obj_type, dtype=np.int32),
        split=np.array([r["split"] for r in rows]),
        negative=np.array([int(bool(r.get("negative"))) for r in rows], dtype=np.int8),
        negative_type=np.array([r.get("negative_type") or "" for r in rows]),
    )
    meta = {
        "mode": "object", "images": len(rows), "objects": len(obj_row),
        "dim": int(fc.shape[1]), "type_keys": type_keys, "types_count": len(type_keys),
    }
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[object] done {args.out}  dim={meta['dim']} objects={meta['objects']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3.8-27B 特征抽取（冻结主干）")
    ap.add_argument("--mode", required=True, choices=["spatial", "object"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试/子集）")
    ap.add_argument("--crop-size", type=int, default=224, help="实体 crop 边长（像素）")
    ap.add_argument("--skip-frames", action="store_true", help="跳过整帧编码（只抽实体 crop，快 10 倍以上）")
    ap.add_argument("--prefetch", type=int, default=0, help="CPU 预处理并行线程数（>0 提速，建议 4~6）")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model)
    if torch.cuda.is_available():
        print(f"[model] cuda peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB", flush=True)
    if args.mode == "spatial":
        run_spatial(args, model, processor)
    else:
        run_object(args, model, processor)


if __name__ == "__main__":
    main()
