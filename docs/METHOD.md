# Method — how starlife_eval scores spatial & relational understanding

This document is the normative description of the v1.5 benchmark. If code and prose disagree, the
code in `src/starlife_eval/verifiers/` is authoritative.

## 1. Input contract

Each **pair** is two frames of the same tabletop scene (`before`, `after`) plus the slot table: every
object detected in the `after` frame has a slot index, a class label, a bbox, and the generator's
ground-truth change class / relations / contact & support flags. The pair is the unit of scoring —
models are never asked to track video, only to compare two frames.

For a zero-shot multimodal model the prompt contains:

1. both RGB frames (JPEG, in order),
2. the slot table (`slot`, `bbox`, plus class where the model is allowed to see it — the reference
   cloud prompt does **not** reveal the GT change class),
3. an answer schema: `{"changes":[{"slot":int,"type":...}], "relations":[{"from_slot":int,
   "to_slot":int,"type":..., "rel_pos":..., "contact":bool, "support":bool}]}`.

The exact prompt and parsing/normalisation live in `src/starlife_eval/models/openai_compat.py`.

## 2. Sampling protocol

```
test split = 1,000 pairs (dataset_spatial_v3)
sample     = sorted(random.Random(20261020).sample(test_indices, 60))
```

The sample is fixed for every model and every score on the public board. Reproduction requires
`dataset_spatial_v3` exactly as released (manifest + frames). `sample_pairs()` is implemented in
`src/starlife_eval/verifiers/pair_protocol.py`.

## 3. The seven dimensions

All dimensions are reported on the 0–100 scale. The composite is their equal-weight mean; a
dimension that cannot be measured for a model is omitted from the denominator and reported with an
`na` reason — it is never replaced by 0 or by a placeholder.

### 3.1 change_top1 (变化识别 top1)

Per slot, four classes: `none`, `appeared`, `displaced`, `disappeared`. Accuracy over all slots of
the 60 pairs.

### 3.2 change_balance (变化平衡性, macro-F1)

Macro-F1 over the same four classes, equal weight per class (the rare `displaced` class is not
down-weighted).

### 3.3 relation_type (关系类型准确率)

Relations are matched per pair on `(from_slot, to_slot)`. A prediction is correct when the relation
class matches GT (11 classes: `on`, `in`, `contains`, `held_by`, `part_of`, `belongs_to`, `near`,
`next_to`, `facing`, `above`, `below`). Unmatched GT relations count as misses; predicted relations
without a GT counterpart count as false positives in the event/F1 views.

### 3.4 rel_pos (空间方位准确率)

Same triple matching as relation_type, scored on the relative position label
(`top`, `inside`, `front`, `back`, `left`, `right`, `front_left`, `front_right`, `back_left`,
`back_right`). The public board reports the 10-class accuracy.

### 3.5 attrs (接触/支撑准确率)

`contact` and `support` are binary attributes in the relation table; the dimension is the mean F1 of
the two binaries.

### 3.6 event_f1 (事件级 F1)

Predictions and GT are both translated into the runtime's `WorldEvent` contract
(`EntityCreated`, `StateChanged`, `RelationAdded`, `RelationRemoved`) with Unix-ms timestamps and
stable per-pair entity ids (`src/starlife_eval/translator.py`). Events are matched on
`(kind, subject, object, class)`, P/R/F1 is computed per kind, then macro-averaged over the kinds
that are measurable for the model (`StateChanged`, `RelationAdded` for zero-shot VLMs; the local
heads can additionally emit `EntityCreated`). `RelationRemoved` and before-relation deltas are
reported as N/A where the dataset cannot support them — not scored as zero.

### 3.7 state_consistency (世界状态一致性)

Events are replayed into a minimal `WorldStore` (`src/starlife_eval/verifiers/state_replay.py`) and
the after-scene snapshot is compared with GT on: object set, change classes, relation set, and
contact/support attributes, weighted equally over the measurable components.

## 4. Confidence intervals

`tools/confidence_intervals.py` runs 1,000 pair-level bootstrap resamples (seed `20261021`) over the
60 sample pairs for every dimension and the composite, producing
`confidence_intervals.json`. Intervals describe sampling noise of the 60-pair sample, not model
uncertainty.

## 5. Thresholds (v1.5 targets, from the internal test-suite design)

| metric | target |
|---|---|
| change top-1 | ≥ 0.97 |
| change macro-F1 | ≥ 0.85 |
| type closed-set top-1 | ≥ 0.96 |
| relation macro-F1 | ≥ 0.84 |
| rel_pos accuracy (10 classes) | ≥ 0.85 |
| attrs F1 | ≥ 0.93 |
| event F1 | ≥ 0.80 |
| state consistency | ≥ 0.90 |

These are reported for information; the leaderboard uses measured values, not thresholds.

## 6. Known limitations (honest list)

- **Synthetic data.** v1.5 is procedurally rendered; real-world capture is being collected and will
  be a separate dataset version.
- **Pair-level, not video.** Temporal continuity (id switches, occlusion through time) is not
  measured in v1.5; a continuous-sequence protocol is designed but not shipped.
- **Zero-shot schema.** Cloud VLMs do not emit entity types, so `type` and the `types` part of state
  replay are N/A for them by construction.
- **Prompt sensitivity.** The prompt is fixed and published; a different prompt is a different
  benchmark entry, which is why the prompt lives in the config/code and is recorded in every report.
