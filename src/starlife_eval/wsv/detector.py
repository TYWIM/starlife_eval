"""Frozen grounding front-end for OpenWSV.

The detector lives inside the evaluation framework: every model that needs object boxes
receives exactly the same boxes produced here (same weights, same thresholds, same
pre-processing). It is deliberately *not* part of any model's submission, because a
per-model detector would make "accuracy" a product of detector x state model and kill
reproducibility.

Example
-------
    det = FrozenDetector()
    boxes = det.detect("scene.jpg", ["mirror", "fireplace"])
    dets = det.detections_for_phrases(image_path, "mirror", "fireplace")
    # dets is ready for the StarLife-1-WSV /predict API
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import threading
from typing import Iterable

from .config import DETECTOR


def _ensure_libs() -> None:
    """Allow running against the local `pylibs` target when torch is not site-installed."""
    libs = pathlib.Path(os.environ.get("OPENWSV_PYLIBS", "")) if os.environ.get("OPENWSV_PYLIBS") else None
    if libs and libs.exists() and str(libs) not in sys.path:
        sys.path.insert(0, str(libs))


def _norm(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").replace("-", " ").split()).strip(" .")


class FrozenDetector:
    """Deterministic GroundingDINO wrapper with the framework's pinned settings."""

    def __init__(self, device: str | None = None, cache_dir: str | pathlib.Path | None = None,
                 load_model: bool = True):
        _ensure_libs()
        self.spec = dict(DETECTOR)
        self.cache_dir = str(cache_dir or os.environ.get("OPENWSV_HF_CACHE", "")) or None
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._meta: dict | None = None
        self.device = device or self._pick_device()
        if load_model:
            self._load()

    # -- setup ------------------------------------------------------------
    @staticmethod
    def _pick_device() -> str:
        try:
            import torch  # noqa: PLC0415

            for candidate in DETECTOR["device_preference"]:
                if candidate == "cuda" and torch.cuda.is_available():
                    return "cuda"
                if candidate == "cpu":
                    return "cpu"
        except Exception:  # noqa: BLE001 - torch missing -> the caller sees a clear error on load
            pass
        return "cpu"

    def _load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        torch.manual_seed(0)
        self._processor = AutoProcessor.from_pretrained(self.spec["model"], cache_dir=self.cache_dir)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.spec["model"], cache_dir=self.cache_dir)
            .to(self.device)
            .eval()
        )
        self._meta = self._build_meta()

    def _build_meta(self) -> dict:
        import transformers  # noqa: PLC0415

        weights_sha = self._weights_sha256()
        pinned = self.spec.get("weights_sha256")
        if pinned and weights_sha not in (pinned, "unavailable"):
            raise RuntimeError(
                "GroundingDINO weights changed: local snapshot sha256 is "
                f"{weights_sha} but the framework pins {pinned}; bump FRAMEWORK_VERSION to re-baseline."
            )
        return {
            "model": self.spec["model"],
            "revision": self.spec["revision"],
            "weights_sha256": pinned or weights_sha,
            "weights_bytes": self.spec.get("weights_bytes"),
            "box_threshold": self.spec["box_threshold"],
            "text_threshold": self.spec["text_threshold"],
            "shortest_edge": self.spec["shortest_edge"],
            "longest_edge": self.spec["longest_edge"],
            "device": self.device,
            "transformers": transformers.__version__,
        }

    def _weights_sha256(self) -> str:
        """sha256 of the local `model.safetensors` (HF LFS oid); streamed, size-guarded."""
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            snapshot = pathlib.Path(
                snapshot_download(self.spec["model"], cache_dir=self.cache_dir, local_files_only=True)
            )
        except Exception:  # noqa: BLE001
            return "unavailable"
        weights = snapshot / "model.safetensors"
        if not weights.exists():
            return "unavailable"
        return hashlib.sha256(weights.read_bytes()).hexdigest() if weights.stat().st_size < 64 * 1024 * 1024 else self._stream_sha(weights)

    @staticmethod
    def _stream_sha(path: pathlib.Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        return h.hexdigest()

    # -- inference --------------------------------------------------------
    def detect(self, image, phrases: Iterable[str]) -> dict:
        """Return ``{phrase: [x1, y1, x2, y2]}`` plus ``_scores`` / ``_meta``."""
        phrases = [p for p in (str(x).strip() for x in phrases) if p]
        if not phrases:
            return {"_scores": {}, "_meta": dict(self.meta())}
        if self._model is None:
            raise RuntimeError("FrozenDetector was constructed with load_model=False; call .load() first")

        from PIL import Image  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if isinstance(image, (str, pathlib.Path)):
            image = Image.open(image)
        image = image.convert("RGB")
        prompt = " . ".join(phrases) + " ."

        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad(), self._lock:
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.spec["box_threshold"],
            text_threshold=self.spec["text_threshold"],
            target_sizes=[image.size[::-1]],
        )
        boxes: list[tuple[list[float], float, str]] = []
        for res in results:
            for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
                boxes.append(([float(v) for v in box.tolist()], float(score), _norm(label)))
        out: dict = {"_scores": {}, "_meta": self.meta(), "_image_size": [int(image.size[0]), int(image.size[1])]}
        for phrase in phrases:
            want = _norm(phrase)
            best = None
            for box, score, label in boxes:
                if want and (want in label or label in want):
                    if best is None or score > best[1]:
                        best = (box, score)
            if best is not None:
                out[phrase] = best[0]
                out["_scores"][phrase] = best[1]
        return out

    def detections_for_phrases(self, image_path, x_phrase: str, y_phrase: str,
                               before=None, after=None) -> dict | None:
        """Build the WSV ``detections`` payload from a single image (before == after)."""
        from PIL import Image  # noqa: PLC0415

        img = Image.open(image_path).convert("RGB")
        boxes = self.detect(img, [x_phrase, y_phrase])
        if x_phrase not in boxes or y_phrase not in boxes:
            return None
        size = boxes.get("_image_size") or [img.size[0], img.size[1]]
        payload = {"width": size[0], "height": size[1]}
        for key, frame in (("before", before), ("after", after)):
            frame_img = Image.open(frame).convert("RGB") if frame else img
            frame_boxes = boxes if frame is None else self.detect(frame_img, [x_phrase, y_phrase])
            fsize = frame_boxes.get("_image_size") or list(frame_img.size)
            payload[f"{key}_size"] = fsize
            payload[key] = [
                {"id": "x", "bbox": frame_boxes[x_phrase]},
                {"id": "y", "bbox": frame_boxes[y_phrase]},
            ]
        # WSV expects a single canvas size; frames are the same size in every suite
        payload["width"], payload["height"] = payload["after_size"]
        payload.pop("before_size", None)
        payload.pop("after_size", None)
        payload["_scores"] = boxes.get("_scores", {})
        payload["_meta"] = self.meta()
        return payload

    def meta(self) -> dict:
        if self._meta is None:
            self._meta = {
                "model": self.spec["model"],
                "revision": self.spec["revision"],
                "weights_sha256": "pending-load",
                "box_threshold": self.spec["box_threshold"],
                "text_threshold": self.spec["text_threshold"],
                "device": self.device,
            }
        return dict(self._meta)


def load_metadata(cache_dir: str | pathlib.Path | None = None) -> dict:
    """Detector metadata without loading weights (for offline report generation)."""
    return FrozenDetector(load_model=False, cache_dir=cache_dir).meta()


if __name__ == "__main__":  # tiny CLI smoke test: python -m starlife_eval.wsv.detector img.jpg "mirror" "fireplace"
    det = FrozenDetector()
    boxes = det.detect(sys.argv[1], sys.argv[2:])
    print(json.dumps({"meta": boxes["_meta"], "scores": boxes["_scores"]}, ensure_ascii=False, indent=1))
    for phrase in sys.argv[2:]:
        print(phrase, boxes.get(phrase))
