# v2 task-matrix benchmark (protocol)

Status: harness implemented; data integration in progress. This document defines the v2 protocol
that replaces the pilot "60-pair change/relation" board for **general** multimodal and world models.

## Contract

Every participant — closed API VLM, open-weights VLM, specialist model, world model — answers the
**same tasks** and returns the **same answer schema**. Only then are the numbers comparable.

A **task** is one JSONL record:

```json
{"task_id": "spatialmqa-0001", "dataset": "spatialmqa", "capability": "T1",
 "answer_type": "choice", "question": "...", "images": ["data/raw/spatialmqa/img/0001.jpg"],
 "choices": ["left", "right", "in front", "behind"], "answer": 2,
 "answers": [], "tolerance": null, "split": "test", "meta": {}}
```

- `answer_type`: `choice` (index), `short` (accepted strings or numeric tolerance), `json`
  (checked field-by-field against the task's answer object).
- `capability`: T1 spatial relations (multi-hop), T2 6D location/orientation, T3 counterfactual
  object reasoning, T4 state change / world-state consistency, T5 object & attribute understanding,
  T6 compositional reasoning.

A **prediction** is:

```json
{"task_id": "spatialmqa-0001", "choice": 2, "text": null, "value": null, "raw": "...", "error": null}
```

Predictions are archived per model; any row can be re-scored without re-querying the model.

## Running

```bash
python -m starlife_eval.v2 run --tasks data/tasks/spatialmqa.jsonl \
  --model configs/template_multimodal_model.json --out runs/v2/my-model
python -m starlife_eval.v2 score --tasks data/tasks/spatialmqa.jsonl \
  --pred runs/v2/my-model/predictions.jsonl --out runs/v2/my-model --model-name my-model
python -m starlife_eval.v2 compare --tasks data/tasks/spatialmqa.jsonl \
  --pred-a runs/a/predictions.jsonl --pred-b runs/b/predictions.jsonl
```

Three prompt variants (`direct`, `reasoned`, `minimal`) are round-robined over tasks; the raw replies
are cached so template sensitivity can be measured. `tools/fetch_benchmarks.py --list` prints the
registered datasets and their licenses.

## Data sources (authoritative first)

| dataset | capability | official source | status |
|---|---|---|---|
| SpatialMQA (ACL 2025) | T1 | https://hf-mirror.com/datasets/liuziyan/SpatialMQA (CC BY 4.0) | downloaded, 536 dev tasks converted |
| Spatial457 (CVPR 2025) | T2 | https://hf-mirror.com/datasets/RyanWW/Spatial457 (Apache 2.0) | download in progress |
| MindEdit-Bench | T3 | https://hf-mirror.com/datasets/ZODAOfficial/MindEdit-Bench (CC BY 4.0) | downloaded, 1,003 tasks converted |
| Spatial-DISE (ICLR 2026) | T1–T6 | https://hf-mirror.com/datasets/TACPS-liv/Spatial-DISE (CC BY-NC-SA 4.0) | download-only, not redistributed |
| spatial3d (own) | T4/T5 | `dataset_spatial_v3.tar` (CC BY 4.0) | diagnostic supplement only |

Rules: official test splits only; no repackaging of third-party data into this repository (scripts +
citations only); own synthetic set is supplementary, never the sole source.

## Scoring and statistics

- Deterministic correctness: choice index; accepted short answers or numeric tolerance; JSON field
  match. **No LLM judge** for the primary score.
- Report per capability and per dataset; composite is a published, user-selectable weighting.
- 95% bootstrap CI over tasks (1,000 draws, seed 20261021); abstentions and API errors are reported
  separately and never converted into "correct".
- Model comparisons: paired McNemar on aligned correctness vectors; "significant" only at p < 0.05.
- Every score must be produced by code and carry the run manifest (model, endpoint snapshot time,
  temperature, prompt template, data hash, code commit).

## Governance

- The ranking contains **all** entries, including the organizers' model; no special-cased rows.
- Head-level metrics from the pilot protocol are auxiliary diagnostics only; the v2 main score comes
  from the unified task outputs.
- Docs use numbers, CIs and significance only — no SOTA/top-1/marketing wording.
- Every entry (including ours) is reproducible from its published output cache; weights are not
  part of the benchmark release.
- Versioning: a data or protocol change bumps the version; **every model is re-tested** on the new
  version before its row is published — rows are never carried over across versions.

## Relation to the pilot board (v1.5)

The pilot 60-pair board (structured change/relation scoring on `dataset_spatial_v3`) remains usable
as a **structured diagnostic** (T4/T5-style outputs). It is not the general spatial-reasoning
benchmark; v2 supersedes it for leaderboard purposes.
