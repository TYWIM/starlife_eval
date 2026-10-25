"""Frozen framework constants for OpenWSV (WSV).

Everything the leaderboard depends on must be pinned here: dataset fingerprints, the
frozen detection front-end, prompt version, seeds, and the scoring contract. Bumping any
value starts a new framework version and requires re-running every row.
"""
from __future__ import annotations

import hashlib
import pathlib

FRAMEWORK_VERSION = "openwsv-1"

# --- detection front-end (part of the framework, not of any model) ----------
DETECTOR = {
    "model": "IDEA-Research/grounding-dino-tiny",
    # pinned huggingface commit + safetensors sha256 (HF LFS oid), recorded 2026-10-23
    "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "weights_sha256": "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3",
    "weights_bytes": 689359096,
    "box_threshold": 0.20,
    "text_threshold": 0.20,
    "device_preference": ("cuda", "cpu"),
    # processor defaults (transformers GroundingDinoImageProcessor)
    "shortest_edge": 800,
    "longest_edge": 1333,
}

PROMPT_VERSION = "openwsv-1"

SEEDS = {
    "sample": 20261020,
    "bootstrap": 20261022,
    "bootstrap_iters": 1000,
}

# --- suites ---------------------------------------------------------------
SUITES = {
    "spatialmqa": {
        "file": "suites/spatialmqa.jsonl",
        "kind": "text",
        "capability": "T1",
        "answer_type": "choice",
    },
    "spatialmqa_relation": {
        "file": "suites/spatialmqa.jsonl",
        "kind": "text",
        "capability": "T1",
        "answer_type": "choice",
        "filter": r"^Where (?:is|are) the (?P<x>.+?) located relative to the (?P<y>.+?)\?$",
    },
    "pairs": {
        "file": "suites/pairs_test.jsonl",
        "kind": "structured",
        "capability": "T4/T5/T6",
        "answer_type": "state",
        "split": "test",
        "n": 1000,
    },
    "change": {
        "file": "suites/change_test.jsonl",
        "kind": "text",
        "capability": "T4",
        "answer_type": "choice",
        "split": "test",
        "n": 1000,
        "source": "derived from pairs_test.jsonl GT",
    },
    "relation_change": {
        "file": "suites/relation_change_test.jsonl",
        "kind": "text",
        "capability": "T5",
        "answer_type": "choice",
        "split": "test",
        "n": 600,
        "source": "synthetic openwsv-data-1.1",
    },
    "state_replay": {
        "file": "suites/state_replay_test.jsonl",
        "kind": "text",
        "capability": "T6",
        "answer_type": "choice",
        "split": "test",
        "n": 1000,
        "source": "derived from pairs_test.jsonl after-state GT",
    },
    "cross_frame": {
        "file": "suites/cross_frame_test.jsonl",
        "kind": "text",
        "capability": "T7",
        "answer_type": "choice",
        "split": "test",
        "n": 400,
        "source": "synthetic openwsv-data-1.1",
    },
}

SUITE_DIR = pathlib.Path(__file__).resolve().parent


def _sha256_file(path: str | pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dataset_fingerprint(root: str | pathlib.Path | None = None) -> dict:
    """sha256 of every suite jsonl that ships with the framework."""
    root = pathlib.Path(root) if root else SUITE_DIR
    out = {}
    for name, spec in SUITES.items():
        path = root / spec["file"]
        if path.exists():
            key = str(path.relative_to(root)).replace("\\", "/")
            out[key] = _sha256_file(path)
    return out


def framework_meta(detector_meta: dict | None = None) -> dict:
    """Metadata block embedded in every result JSON."""
    meta = {
        "version": FRAMEWORK_VERSION,
        "prompt_version": PROMPT_VERSION,
        "seed": SEEDS["sample"],
        "bootstrap_seed": SEEDS["bootstrap"],
        "dataset_sha256": dataset_fingerprint(),
    }
    if detector_meta is not None:
        meta["detector"] = detector_meta
    return meta
