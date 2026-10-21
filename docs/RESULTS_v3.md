# Results — organizers' model (v3 heads)

> This row belongs to the organizers' own model and is an ordinary entry in the ranking: same
> data, same prompt, same answer schema, same scorer. Head-level metrics below are auxiliary
> diagnostics; they are not the main score and are N/A for models that only answer through the
> unified task schema. No top-1/SOTA wording is used.

Dataset: `dataset_spatial_v3` (synthetic, 5,000 pairs / 10,000 frames, seed 20261020).
Backbone: Qwen3.8-27B vision tower, **frozen**; heads: LayerNorm → Linear(d,512) → GELU →
Dropout(0.2) → Linear(512,256) → GELU → Dropout → Linear(256,out). Space heads take
`[slot_before, slot_after, |diff|, mask]` (3 × 5120 + 1) + 13-dim slot geometry + pair geometry
(15,370-d pair input); five heads change(4) / type(22 closed-set) / relation(12) / rel_pos(10) /
attrs(contact, support). Trained with `scripts/train_heads.py --epochs 40`, best epoch by val
macro-F1 (36–39), single A800 80 GB, ~6 min for all five heads. Test split is scored once.

## Head-level metrics (test split)

| head | top-1 | macro-F1 | best epoch | n |
|---|---|---|---|---|
| change (4 classes) | **0.9973** | **0.9948** | 36 | 5,839 |
| type (22 closed-set classes) | 0.9606 | 0.9568 | 38 | 5,839 |
| relation (12 classes) | 0.9070 | 0.8735 | 39 | 6,610 |
| rel_pos (10 classes) | 0.8632 | 0.8122 | 37 | 2,690 |
| attrs (contact / support) | — | 0.9548 / 0.9665 | — | — |

Weak classes (macro-F1):

- relation: `next_to` 0.429 is the only collapsed class; the rest are ≥ 0.76 (`near` 0.763,
  `below` 0.860, `above` 0.919);
- rel_pos: `back_left` 0.678 / `left` 0.685 / `back_right` 0.718, the rest ≥ 0.76 (`top` 0.941);
- type: `notebook` 0.826 / `mouse` 0.872 / `phone` 0.885 are the weakest, the rest ≥ 0.89;
- change: none 0.998 / disappeared 0.998 / appeared 1.000 / displaced 0.983.

## 60-pair benchmark row (public board protocol)

Deterministic 60-pair sample from the test split (`seed=20261020`), 7 equal-weight dimensions
(0–100), 95% pair-level bootstrap (1,000 draws, seed 20261021):

| dimension | value | 95% CI |
|---|---|---|
| change_top1 | 99.7 | 99.1 – 100.0 |
| change_balance (macro-F1) | 99.4 | 98.3 – 100.0 |
| relation_type | 92.2 | 88.2 – 96.0 |
| rel_pos | 83.7 | 77.3 – 89.8 |
| attrs | 99.0 | 97.5 – 100.0 |
| event_f1 | 96.6 | 94.8 – 98.2 |
| state_consistency | 95.9 | 94.5 – 97.2 |
| **composite (equal weight)** | **95.2** | **93.7 – 96.5** |

Reproduce with the released `spatial_v3_features.tar.gz` + `v3_heads.tar.gz`:

```bash
python -m starlife_eval run --config configs/local_heads_v15.json --data-root "$DATA_ROOT" \
  --only starlife-v1-wsv --out runs/local-heads
python tools/confidence_intervals.py --config configs/local_heads_v15.json --data-root "$DATA_ROOT" \
  --out runs/local-heads/confidence_intervals.json --report runs/local-heads/report.json
```

## Notes

- The open-vocabulary type head (mapping crop features into a Qwen3-Embedding text space) was
  prototyped: r2 converged on train (top-1 0.89) but did not generalize (val 0.046 / test 0.044 /
  test top-5 0.126), so **OOV type retrieval is not part of the v1.5 board**. `type` is reported as
  closed-set on the 22 classes only; zero-shot cloud VLMs report type as N/A because their answer
  schema has no entity types.
- v2 → v3 differences and the full data/repro details are in `docs/METHOD.md` and
  `docs/DATA_CARD.md`; the internal test-suite plan is in `docs/TEST_SUITE_DESIGN.md`.
