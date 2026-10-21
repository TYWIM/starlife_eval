# Data card — dataset_spatial_v3 (synthetic)

## Summary

| field | value |
|---|---|
| name | `dataset_spatial_v3` |
| type | **synthetic** tabletop scenes (procedurally rendered) |
| generator | `spatial3d-1.0.0`, vocab `1.0.0` |
| seed | `20261020` |
| pairs | 5,000 (10,000 frames) |
| resolution | 1024 × 768, JPEG |
| splits | train 3,000 / val 1,000 / test 1,000 (by scene group; no frame crosses splits) |
| per pair | 2 frames (`before`, `after`), slot table with bbox / change class / relations / contact / support |
| release format | `dataset_spatial_v3.tar.gz` (manifest.jsonl, meta.json, frames/, stats.csv, README) |
| license | CC BY 4.0 |

## Provenance

The scenes are rendered from a fixed 3D room layout with a small pool of asset categories
(`monitor`, `keyboard`, `mug`, `book`, `lamp`, …), randomized object placement, and a scripted
before→after edit (appear / disappear / displace). Camera intrinsics, poses and the resulting slot
geometry are stored with the manifest, so the dataset is fully regenerable from the generator
version and seed recorded in `meta.json`.

**This is not real-world data.** A real-world capture set is being collected; it will be released
under a separate dataset version, not mixed into v1.5.

## Annotation semantics

- change classes: `none`, `appeared`, `displaced`, `disappeared` (per slot);
- relation triples: `from_slot`, `to_slot`, one of 11 relation classes, one of 10 relative positions,
  plus `contact`/`support` booleans, for relations present in the `after` frame;
- the GT comes from the generator (programmatic), not from model annotation;
- scenes deliberately include empty-change pairs and rare `displaced` cases so macro-F1 is meaningful.

## Splits

Splits are by scene group, so the same scene never appears in both train and test. The 60-pair
evaluation sample is drawn from the test split with seed `20261020` — see `docs/METHOD.md`.

## What is released

| artifact | size (≈) | purpose |
|---|---|---|
| `dataset_spatial_v3.tar` | 1.2 GB | full dataset (manifest + frames + GT) — this is all a participant needs |

Frozen backbone features and the organizers' model weights are **not** part of the benchmark
release: participants only need the task data; every entry, including ours, is ranked from its
published model outputs through the unified schema.

Cloud providers' raw outputs used for the public board are **not** redistributed; every row can be
reproduced by re-running the model through the published `chat_vlm` adapter.

## Integrity

`meta.json` records generator/vocab versions and the seed; `stats.csv` summarizes per-split class
counts. `checksums.sha256` inside the dataset archive lists frame hashes; `report.json` embeds
16-hex sha256 prefixes of manifest and features so a scored run can be traced to exact inputs.

## Ethical / usage notes

- No human subjects, no personal data, no real cameras in v1.5.
- Synthetic scenes may contain artefacts; evaluations on this set are about pipeline and protocol,
  not about a real deployment distribution.
- Please cite the dataset as `dataset_spatial_v3` (synthetic) with the generator version above and do
  not describe it as real-world data.
