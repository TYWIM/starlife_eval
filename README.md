# starlife_eval — Spatial & Relational Understanding Benchmark (v1.5)

Evaluation toolkit for the **openwsv** benchmark: given before/after RGB frames of a tabletop
scene plus detected object slots, a model has to report what changed, how slots relate spatially
and physically, and the resulting world state.

This repository is the reference implementation of the benchmark's scoring pipeline. It can score
**any multimodal model** that speaks the OpenAI-compatible chat-completions API (zero-shot), and it
ships the frozen **local heads** baseline (Qwen3.8-27B slot features + 5 trained heads) used on the
public leaderboard.

> **Data note — synthetic.** The v1.5 dataset (`dataset_spatial_v3`) is **synthetic**:
> procedurally rendered 1024×768 tabletop scenes from generator `spatial3d-1.0.0` (fixed seed
> 20261020, 5,000 pairs / 10,000 frames). It is not real-world capture. Real-world data collection
> is in progress and will be released as a separate version.

## Why this exists

Most "spatial reasoning" comparisons are prompt-only and unreproducible. This benchmark fixes the
pipeline instead:

- **fixed sample** — a deterministic 60-pair sample from the frozen test split (seed `20261020`),
  the same pairs for every model;
- **fixed GT** — per-slot change classes, relation triples, and contact/support labels come from the
  scene generator, not from another model;
- **fixed scoring** — 7 equal-weight dimensions (0–100) and a published composite formula,
  implemented in `src/starlife_eval/verifiers/`;
- **fixed outputs** — `report.json`, `report.md`, `leaderboard.generated.json`, plus 95% pair-level
  bootstrap confidence intervals (`tools/confidence_intervals.py`).

## Repo layout

```
src/starlife_eval/     scoring pipeline (dataset loading, adapters, verifiers, report writer)
tests/                 unit + parity tests (pytest)
tools/                 confidence intervals, reliability, difficulty curve, data-root preparation
configs/               v1.5 leaderboard configs + template for your own model
scripts/               method pipeline: feature extraction, head training, slot geometry, vocab
reports/v1.5/          results archived for the public leaderboard (v3 heads + completed cloud rows)
docs/                  method, data card, reproduction guide, test-suite design
```

## Quick start — score your own multimodal model (zero-shot)

You need: Python ≥ 3.11, and an OpenAI-compatible endpoint (any VLM).

1. **Get the dataset** (synthetic, ~1.2 GB) from the
   [v1.5 release](https://github.com/TYWIM/starlife_eval/releases/tag/v1.5): download
   `dataset_spatial_v3.tar.gz` and unpack it so that you have a data root like
   ```
   <data_root>/dataset_spatial_v3/{manifest.jsonl, meta.json, frames/, stats.csv}
   ```
   `configs/template_multimodal_model.json` in that release is the config to edit.
2. **Install**
   ```bash
   pip install -e .
   ```
3. **Describe your model** in a config (copy `configs/template_multimodal_model.json`):
   ```json
   {
     "dataset": "dataset_spatial_v3",
     "features": "features/spatial_v3.npz",
     "split": "test",
     "n_pairs": 60,
     "sample_seed": 20261020,
     "models": [{
       "id": "my-vlm",
       "kind": "chat_vlm",
       "mode": "ZERO-SHOT",
       "family": "MyModel",
       "provider": "MyLab",
       "base_url": "https://your-endpoint/v1",
       "model": "your-model-name",
       "api_key_env": "MY_VLM_API_KEY",
       "max_tokens": 4000
     }]
   }
   ```
4. **Run**
   ```bash
   export MY_VLM_API_KEY=...            # never commit keys
   python -m starlife_eval run --config my_vlm.json --data-root <data_root> --out runs/my-vlm --live
   ```
   `runs/my-vlm/report.json` and `leaderboard.generated.json` contain the 7 dimensions and the
   composite score; the same 60 pairs and the same GT are used for every model.

Notes:

- `--live` enables network calls; without it the runner replays the cached JSONL in `source`.
- The prompt contract (two images + slot table, JSON answer schema) is in
  `src/starlife_eval/models/openai_compat.py`.
- Raw calls are cached as JSONL, so interrupted runs resume and stay auditable.

## Quick start — reproduce the local-heads baseline

1. Download `dataset_spatial_v3.tar.gz` **and** `spatial_v3_features.tar.gz` + `v3_heads.tar.gz`
   from the release, and place them as
   ```
   <data_root>/features/spatial_v3.npz
   <data_root>/features/spatial_v3_geom.npz
   <data_root>/features/spatial_v3.meta.json
   <data_root>/checkpoints/v3_heads/{change,type,relation,rel_pos,attrs}.pt
   ```
2. Run
   ```bash
   python -m starlife_eval run --config configs/v15.json --data-root <data_root> \
     --only starlife-v1-wsv --out runs/local-heads
   ```
   (GPU is required for the heads; the trainer used a 27B backbone embedding, not included.)

## What is scored

| dimension | metric | definition |
|---|---|---|
| change_top1 | per-slot 4-class top-1 | none / appeared / displaced / disappeared |
| change_balance | macro-F1 over the 4 change classes | equal weight for rare classes |
| relation_type | relation-class accuracy | per `(from_slot, to_slot)` triple, 11 classes |
| rel_pos | relative-position accuracy | 10 classes, matched on the same triples |
| attrs | contact / support F1 | two binary physical attributes |
| event_f1 | event-level macro-F1 | predictions translated into the four `WorldEvent` kinds |
| state_consistency | world-state replay agreement | after-scene objects / changes / relations / attributes |

Composite = equal-weight mean of the available dimensions (missing dimensions are removed from the
denominator, never scored as 0). See `docs/METHOD.md` for metric definitions and the 60-pair
sampling protocol.

## Reproducibility rules

- `sample_pairs(manifest, split, 60, seed=20261020)` = `sorted(random.Random(20261020).sample(idx, 60))`;
  the sample indices are part of every `report.json`.
- Bootstrap intervals: 1,000 pair-level resamples, seed `20261021`.
- Every `report.json` embeds config, dataset/feature/checkpoint sha256 (16 hex chars), timestamps and
  per-metric N/A reasons; nothing is filled in manually.
- API keys live only in environment variables; the runner never writes them to disk.

## Status of the public leaderboard data

The published board currently contains four rows scored on this v1.5 sample
(`starlife-v1-wsv` local heads, `deepseek-flash`, `gpt-6.1-sol`, `gemini-3.8-flash`); the other rows
are the previous v1.4 sample until their v1.5 rerun completes. See
[`reports/v1.5/`](reports/v1.5/) and the site's mixed-data note.

## License

Code: MIT (`LICENSE`). Synthetic dataset and trained heads: CC BY 4.0. See `docs/DATA_CARD.md`.
