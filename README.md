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
   `dataset_spatial_v3.tar` and unpack it so that you have:
   ```
   <data_root>/dataset_spatial_v3/{manifest.jsonl, meta.json, frames/, stats.csv}
   ```
   `configs/template_multimodal_model.json` is the config to edit. Frozen slot features /
   trained heads are only needed to reproduce the organizers' reference baseline and are not part
   of the benchmark release.
2. **Install**
   ```bash
   pip install -e .
   ```
3. **Describe your model** in a config (copy `configs/template_multimodal_model.json`):
   ```json
   {
     "dataset": "dataset_spatial_v3",
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

## Organizer model (participates under the same protocol)

The organizers' model (frozen Qwen3.8-27B slot features + five trained heads, see
[`baselines/`](baselines/)) takes part in the ranking like any other entry: same data, same prompt,
same answer schema, same scorer. Its head-level metrics are auxiliary diagnostics only — the main
score comes from the unified task outputs, so models without heads are fully comparable.

Model weights/features are not part of the benchmark release; what is published for every entry
(including ours) is the model output cache, so every row can be re-scored and inspected.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the plan to grow the protocol into a task-matrix
benchmark aligned with SpatialMQA / Spatial-DISE / Spatial457 / MindEdit-Bench.

## v2 task-matrix benchmark (integration in progress)

The pilot board below scores structured change/relation outputs on a synthetic dataset. The **v2**
protocol generalises the benchmark to any multimodal or world model: authoritative public task sets
(SpatialMQA, Spatial457, MindEdit-Bench, Spatial-DISE where licenses allow) plus an own diagnostic
supplement, a unified task/prediction JSONL schema, three prompt variants, deterministic scoring,
bootstrap CIs and paired significance tests.

```bash
python tools/fetch_benchmarks.py --list
python -m starlife_eval.v2 run   --tasks data/tasks/spatialmqa.jsonl \
  --model configs/template_multimodal_model.json --out runs/v2/my-model
python -m starlife_eval.v2 score --tasks data/tasks/spatialmqa.jsonl \
  --pred runs/v2/my-model/predictions.jsonl --out runs/v2/my-model --model-name my-model
python -m starlife_eval.v2 compare --tasks data/tasks/spatialmqa.jsonl \
  --pred-a runs/a/predictions.jsonl --pred-b runs/b/predictions.jsonl
```

The organizers' model participates under the same protocol; head-level metrics are auxiliary only.
Full contract: [`docs/BENCHMARK_V2.md`](docs/BENCHMARK_V2.md).

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

## Roadmap

The current 60-pair change/relation board is a **v1.5 pilot**, not an authoritative benchmark.
`docs/ROADMAP.md` describes the plan for a task-matrix benchmark (multi-hop spatial QA, 6D pose,
counterfactual reasoning, world-state consistency, real-world human-verified subset) with
statistical significance and external-benchmark cross-validation. The organizers' model is an ordinary entry; the docs contain numbers, confidence intervals and
significance markers only — no SOTA/top-1 wording.

## Status of the public leaderboard data

### v2 task matrix (current campaign)

[`docs/RESULTS_SPATIALMQA.md`](docs/RESULTS_SPATIALMQA.md) and [`reports/v2/`](reports/v2/): five
general-purpose multimodal models on one cell of the v2 matrix — **T1 SpatialMQA**, 1,612 single-image
multiple-choice tasks (test 1,076 + validation 536), deterministic scoring with 95% bootstrap CI and
per-model coverage. MindEdit (T3) is withdrawn from this report (truncated replies had been scored as
wrong answers and a prompt revision invalidated the remaining rows; see the doc) and will be published
only after a clean, fully covered re-run. The organizers' model has not been scored on v2 yet.

### v1.5 pilot (previous board)

The published board currently contains four rows scored on this v1.5 sample
(`starlife-v1-wsv` local heads, `deepseek-flash`, `gpt-6.1-sol`, `gemini-3.8-flash`); the other rows
are the previous v1.4 sample until their v1.5 rerun completes. See
[`reports/v1.5/`](reports/v1.5/) and the site's mixed-data note.

## License

Code: MIT (`LICENSE`). Synthetic dataset and trained heads: CC BY 4.0. See `docs/DATA_CARD.md`.
