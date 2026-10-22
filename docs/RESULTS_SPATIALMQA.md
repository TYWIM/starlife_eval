# SpatialMQA results — v2 task matrix (T1)

Protocol: `docs/BENCHMARK_V2.md`. Dataset: SpatialMQA, 1,612 single-image multiple-choice tasks
(test 1,076 + validation 536), deterministic letter matching, accuracy reported over covered tasks,
95% bootstrap CI (1,000 draws, seed 20261022). Raw per-task predictions are in the run artifacts
shipped with each release.

## Leaderboard (generated 2026-10-22)

See [`reports/v2/leaderboard_spatialmqa.md`](../reports/v2/leaderboard_spatialmqa.md) for the current
table, per-split breakdown and coverage. The generated JSON is
[`reports/v2/leaderboard_spatialmqa.json`](../reports/v2/leaderboard_spatialmqa.json).

## What is measured

One capability only — **T1 spatial relations** — asked as a single image with 2/4/6 text options.
Scores are therefore *not* a general spatial-intelligence ranking; they are one cell of the v2 matrix.

## Coverage and comparability

- A model is only comparable when it answered every task (coverage 100%). Partial rows are shown with
  their coverage and must not be read as final scores.
- `minimax-m3.1-flash` covers 1,611/1,612 (99.94%): one task returns a persistent provider-side
  `HTTP 404 {"code":"NOT_FOUND"}` and is excluded from its denominator; every other model completed
  all 1,612 tasks.
- Empty model output (reasoning models exhausting `max_tokens`) is recorded as a retryable error and
  excluded from the denominator, never scored as a wrong answer.
- Not evaluated: `claude-opus-5-5` (gateway quota exhausted), `seed-2.1-turbo` (run not started).

## MindEdit (T3) is withdrawn from this report

MindEdit results were discarded during this campaign: an earlier run scored truncated (empty) replies
as wrong, and a prompt revision (labelling which image is view1/2/3) invalidated the remaining rows.
A/B testing showed the labelling has no measurable effect (mimo-v2.6-pro, 20 tasks: 6/20 without labels
vs 5/20 with); MindEdit is simply hard for this class of models (15–24 options, ~7% random baseline).
MindEdit will be reported only after a clean, fully covered re-run.

## Provider provenance

| model | endpoint used |
|---|---|
| deepseek-flash | api.apigoto.com |
| qwen3.8-flash | api.apigoto.com |
| glm-5.3-flash | api.apigoto.com |
| mimo-v2.6-pro | tokenrhythm.studio |
| minimax-m3.1-flash | tokenrhythm.studio |

Provider-side model versions and any hidden system prompts are outside our control; the runner passes
`temperature: 0` and identical prompt templates to every model, and prompt templates are public in
`src/starlife_eval/v2/prompts.py`.
