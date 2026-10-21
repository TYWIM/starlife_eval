# Roadmap: from a pilot board to a general spatial-reasoning benchmark (v2)

Status: plan. This repository currently publishes a **pilot** protocol (60-pair before/after
change + relation scoring on a synthetic dataset). That is not an authoritative spatial benchmark,
and the organizers' own baseline must not be presented as a leaderboard winner. This document is the
plan to fix both.

## Why the current setup is weak

1. **Single-source data** — everything is rendered by our own generator; no external validity.
2. **Small sample** — 60 pairs gives roughly ±1.5–3 points on the composite: 1–2 point gaps are
   noise, so ranking is unstable.
3. **Head-centric metrics** — only models that expose our five classifier heads can be measured on
   head-level metrics; general VLMs report N/A on those dimensions, which makes the board
   non-comparable.
4. **Self-serving framing** — organizers' model at the top, organizers' weights in the benchmark
   release: this must be separated (see `baselines/`).
5. **Narrow output space** — no multi-hop QA, no 6D pose, no counterfactual questions, no open-ended
   answers.

## Target: task matrix aligned with established benchmarks

| id | capability | question style | reference |
|---|---|---|---|
| T1 | spatial relations, incl. multi-hop ("A relative to B, via C?") | multiple choice / short answer | [SpatialMQA](https://aclanthology.org/2025.acl-long.31/) |
| T2 | 2D/3D location and orientation (6D) | numeric with tolerance / MC | [Spatial457](https://openaccess.thecvf.com//content/CVPR2025/html/Wang_Spatial457_A_Diagnostic_Benchmark_for_6D_Spatial_Reasoning_of_Large_CVPR_2025_paper.html) |
| T3 | counterfactual object edits ("if moved/rotated, what happens?") | multiple choice | [MindEdit-Bench](https://huggingface.co/datasets/ZODAOfficial/MindEdit-Bench) |
| T4 | state change / world-state consistency | structured JSON (events + state snapshot) | our event/replay verifiers, made model-agnostic |
| T5 | object & attribute understanding (contact, support, material) | multiple choice / structured | synthetic + real subset |
| T6 | compositional reasoning under constraints | multiple choice | external sets + synthetic |

Every capability is reported separately plus a public, user-selectable weighting. Missing dimensions
are reported as N/A with a reason — never scored as zero.

## Data strategy (three sources)

1. **External authoritative subsets (primary)**: evaluate on the official test subsets of
   SpatialMQA / Spatial-DISE / Spatial457 / MindEdit-Bench where licenses allow; redistribute only
   scripts + citations, never their data. These numbers serve as external validation.
2. **Own procedural data (diagnostic only)**: keep the `spatial3d` generator for scalable,
   difficulty-controlled diagnostics; public generator version + seed, hidden-seed private holdout
   to resist contamination.
3. **Real-world subset (validity)**: 300–500 real tabletop scenes, 3 independent annotators +
   adjudication, published **human ceiling** and inter-annotator κ. Required before calling the
   benchmark authoritative.

## Protocol requirements

- **Reproducibility**: frozen data release with hashes; published prompts/parsers/scorers; every
  number produced by code; `run_manifest.json` per run (model, endpoint snapshot time, temperature,
  prompt template, data hash, code commit).
- **Low variance**: test sets of ≥ 1,000 items (CI half-width ≤ ±2 pp); ≥ 3 prompt-template variants
  averaged with reported spread; temperature 0; self-consistency across repeated runs reported;
  paired McNemar tests for model comparisons — "significantly different" only with p < 0.05.
- **Model-agnostic outputs**: multiple choice (deterministic scoring), structured JSON (schema +
  field-level F1), open-ended answers (dual judges calibrated against a human-verified set; judges
  never replace the primary deterministic score).

## World-model track

Continuous-frame input; output = structured world state (objects, attributes, relations, positions)
+ change events. Metrics: state-snapshot agreement per component, event P/R/F1, temporal id-stability,
and counterfactual consistency. VLM and world-model boards are separate, not mixed.

## Board governance

- The leaderboard contains **all** entries, including the organizers' model, under the same data,
  prompt, answer schema and scorer. No special-cased rows; head-level metrics are auxiliary.
- Published rows carry human ceilings, per-task breakdowns, CIs and significance markers; the docs
  use numbers and intervals only — no SOTA/top-1 wording.
- Every entry (including ours) is reproducible from its published output cache; model weights are
  not required and are not part of the benchmark release.
- Versioned releases with changelogs and data hashes; external-benchmark columns reported alongside.

## Milestones

| phase | timeline | deliverable |
|---|---|---|
| P0 | 1 week | benchmark repo ships task data only (no organizer weights/features); unified answer schema for every participant; neutral wording in docs (done in this repo) |
| P1 | 3–4 weeks | T1/T2/T3 external-subset evaluation scripts (~1,000 items); T4/T5 diagnostic sets; protocol v2.0 |
| P2 | 4–6 weeks | real-world human-verified subset + human ceiling; external cross-benchmark validation; public v2 board |
| P3 | ongoing | CI smoke set, canary strings, seasonal data versions, endpoint snapshots |

## Risks

- External benchmark licenses may forbid redistribution → ship scripts + citations only.
- Real subset may be too small for "authoritative" → release as research preview with κ and n.
- Gaps smaller than noise → the board must show intervals and significance, not ranks alone.
