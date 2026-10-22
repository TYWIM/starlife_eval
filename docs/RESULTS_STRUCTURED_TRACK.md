# v2 structured-input track — StarLife-1-WSV-Lite (served service)

Status: **provisional**, separate track. Not comparable with the text-only rows in
[`RESULTS_SPATIALMQA.md`](RESULTS_SPATIALMQA.md).

## Why a separate track

The v2 text track asks a model to read one image, a question and a list of options and to answer
with a letter. StarLife-1-WSV does not emit free-form text at all: it takes **two frames plus
detector boxes** and returns **structured events** (`slots`, `pairs`, `events`). Feeding it the
text task would require inventing a mapping from events to letters, and, because the task ships
no detector, would also mean handing it ground-truth boxes as input.

So the in-house model is scored on the structured protocol it was built for
(`dataset_spatial_v3`, 60-pair test sample) and reported on its own table, with the input mode
stated on every row.

## Protocol

| item | value |
| --- | --- |
| dataset | `dataset_spatial_v3` (GitHub release `v1.5`), split `test`, 60 pairs |
| sampling | `sample_pairs(split="test", n=60, seed=20261020)` — sorted, 1,000 test pairs available |
| input | `before`/`after` JPEG, 1024×768, base64; `detections` = ground-truth boxes from the manifest |
| service | `StarLife-1-WSV-Lite` v1.0.0, `POST /predict` |
| calls | 2 per pair: detections in manifest order and reversed (see below), 120 calls total |
| metrics | same definitions as `starlife_eval.verifiers.pair_protocol.evaluate_pair_protocol` |
| CI | 95% bootstrap over the 60 pairs, 1,000 draws, seed 20261022 (percentile method; sampling seed 20261020) |

**Direction convention.** The service enumerates relation pairs as `combinations(slots, 2)`, i.e.
only in slot order, so a ground-truth relation whose `from_slot > to_slot` cannot be expressed in
a single run. Every pair is therefore sent twice (forward and reversed detections); for each GT
relation the run whose orientation matches the GT direction is used. The GT direction only
selects an input ordering — labels never enter the model.

## Result

| mode | change top-1 | change macro-F1 | relation type | spatial position | contact & support |
| --- | --- | --- | --- | --- | --- |
| **SERVED-API** (end-to-end, boxes-only geometry) | **91.45%** [88.70–94.02] | **67.95%** [63.86–88.34] | **75.90%** [69.94–82.07] | **88.10%** [81.60–94.12] (n=126) | **93.65%** [89.17–97.13] (n=126) |
| LOCAL-HEADS reference (published v1.5, GT-pose features) | 99.70% | 99.40% | 92.20% | 83.70% | 99.00% |

* 351 change slots, 166 GT relations, 126 relations with `rel_pos` and attributes.
* ~1.1 s per call server-side (p50 over 120 calls), transport via an SSH tunnel adds ~0.5 s.

The gap to the local-heads reference is expected and documented in the service itself: the served
pipeline runs in *degraded geometry* mode — geometry is approximated from detector boxes, depth is
a scale proxy and orientation deltas are zero (`e2e.py` docstring) — while the v1.5 reference row
consumes feature bundles extracted with ground-truth 3D pose. Change balance and relation type are
the dimensions most sensitive to that difference; spatial position is actually higher here because
box geometry is what the served mode optimises for.

## Caveats

1. **n = 60 pairs.** Confidence intervals are wide by construction; treat the numbers as a first
   structured-track readout, not as a leaderboard position.
2. **Ground-truth detections.** The service performs no detection; the boxes come from the dataset
   manifest. This measures the model's heads, not its perception.
3. **Event F1 and state consistency are N/A** in this run. The service returns an event stream, but
   the 60-pair verifier path for events/state replay is not wired in yet, so the composite is not
   computed and no composite is displayed.
4. **Two runs per pair.** The direction-matched protocol needs 120 calls instead of 60; this is a
   property of the service's pair enumeration, not of the model.

## Reproduce

```bash
# 1. dataset (1.1 GB, GitHub release v1.5)
#    extract manifest.jsonl + the frames of the sampled pairs; the script reports missing frames
# 2. service reachable (here: ssh -L 127.0.0.1:8890:127.0.0.1:8000 root@<box>)
export WSV_API_URL=http://127.0.0.1:8890
export WSV_API_KEY=<key>
export DATASET_DIR=/path/to/dataset_spatial_v3
export OUT_DIR=./structured_track_out
python scripts/structured_track/run_structured_track.py
```

Outputs: `raw_forward.json`, `raw_reversed.json`, `structured_result.json`.

## Provenance

* service: `StarLife-1-WSV-Lite` v1.0.0 (`Qwen3.8-27B` frozen backbone + 5 heads, A800-80GB),
  `POST /predict`, health `ready: true` at run time.
* run: 2026-10-22, 120 calls, no retries needed, 0 failed pairs.
* the scoring script is in this repository (`scripts/structured_track/`); the raw service responses
  for this run are attached to release `v2.0-data-sources` as `structured_track_raw.zip` (14 KB).
