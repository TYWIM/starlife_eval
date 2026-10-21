"""端到端性能测试：特征抽取 + 分类头推理的延迟与吞吐。"""
import glob
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_features as E
import train_heads as T
from PIL import Image

M = "/root/models/models/Qwen--Qwen3.8-27B/snapshots/master"
CKPT = "./checkpoints/pilot_v2"
DATA = "./dataset_spatial_v2/frames"


def sync():
    torch.cuda.synchronize()


def t(fn, n=1, warm=0):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.time()
    for _ in range(n):
        fn()
    sync()
    return (time.time() - t0) / n


model, proc = E.load_model(M)
print("[bench] model loaded", flush=True)

frames = sorted(glob.glob(DATA + "/*_before.jpg"))[:16]
frame_imgs = [Image.open(p).convert("RGB") for p in frames]
crop_imgs = [img.crop((200, 150, 500, 400)).resize((224, 224)) for img in frame_imgs]

res = {}
for name, imgs, prompt in [
    ("frame", frame_imgs[:1], E.SPATIAL_PROMPT),
    ("frame_bs4", frame_imgs[:4], E.SPATIAL_PROMPT),
    ("crop", crop_imgs[:1], E.SPATIAL_PROMPT),
    ("crop_bs8", crop_imgs[:8], E.SPATIAL_PROMPT),
    ("crop_bs8_object", crop_imgs[:8], E.OBJECT_PROMPT),
]:
    E.encode_images(model, proc, imgs, prompt, len(imgs))
    per = t(lambda: E.encode_images(model, proc, imgs, prompt, len(imgs)), n=3, warm=1)
    res[name] = {"batch": len(imgs), "total_s": round(per, 3),
                 "per_image_ms": round(1000 * per / len(imgs), 1)}
    print(name, res[name], flush=True)

heads, dins = {}, {}
for pt in sorted(glob.glob(CKPT + "/*.pt")):
    ck = torch.load(pt, map_location="cuda")
    head = T.Head(int(ck["d_in"]), int(ck["n_classes"])).cuda().eval()
    head.load_state_dict(ck["state_dict"])
    heads[Path(pt).stem] = head
    dins[Path(pt).stem] = int(ck["d_in"])
print("[bench] heads:", list(heads.keys()), flush=True)

event_frames = frame_imgs[:2]
event_crops = crop_imgs[:12]
t0 = time.time()
E.encode_images(model, proc, event_frames, E.SPATIAL_PROMPT, 2)
E.encode_images(model, proc, event_crops, E.SPATIAL_PROMPT, 8)
sync()
event_s = time.time() - t0
res["event_latency_s"] = round(event_s, 3)
res["event_frames"] = 2
res["event_crops"] = 12
print(f"[bench] change event (2 frames + 12 crops) = {event_s:.2f} s", flush=True)

if "change" in heads:
    X = torch.randn(12, dins["change"], device="cuda")
    per = t(lambda: heads["change"](X), n=200, warm=20)
    res["change_head_12slots_ms"] = round(1000 * per, 3)
    print("[bench] change head 12 slots:", res["change_head_12slots_ms"], "ms", flush=True)

Path("./bench_pipeline.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print("[bench] done", json.dumps(res, ensure_ascii=False), flush=True)
