# Reproduce the benchmark end to end

Three independent things can be reproduced:

| goal | what you need | time |
|---|---|---|
| A. score **your** multimodal model on the 60-pair sample | this repo + dataset release + API endpoint | minutes |
| B. reproduce the **local-heads** row | + features + trained heads + GPU | minutes |
| C. retrain the local heads from scratch | + GPU + backbone for feature extraction (not included) | hours |

## A. Score any multimodal model (zero-shot)

```bash
# 1) dataset (synthetic, 1.2 GB) from the v1.5 release
tar -xzf dataset_spatial_v3.tar.gz          # -> dataset_spatial_v3/
export DATA_ROOT=$PWD

# 2) this repo
pip install -e .

# 3) your model as an OpenAI-compatible chat endpoint
cp configs/template_multimodal_model.json my_vlm.json
#   edit: base_url, model, api_key_env, id/name/family/provider

export MY_VLM_API_KEY=...                   # never commit this
python -m starlife_eval run --config my_vlm.json --data-root "$DATA_ROOT" \
  --out runs/my-vlm --live

# 4) results
cat runs/my-vlm/report.md
python tools/confidence_intervals.py --config my_vlm.json --data-root "$DATA_ROOT" \
  --out runs/my-vlm/confidence_intervals.json --report runs/my-vlm/report.json
```

Interrupted runs resume from the JSONL cache (`runs/<name>/...` or the `source` you configure);
delete the cache to force fresh calls.

Outputs:

- `report.json` — per-model metrics + per-dimension N/A reasons + provenance (config, input sha256,
  timestamps, latencies);
- `report.md` — human-readable summary;
- `leaderboard.generated.json` — row in the site's data contract;
- `confidence_intervals.json` — 95% pair-level bootstrap (1,000 draws, seed 20261021).

## B. Local-heads baseline

Additional assets: `spatial_v3_features.tar.gz` and `v3_heads.tar.gz`.

```bash
tar -xzf spatial_v3_features.tar.gz   # features/spatial_v3.npz + spatial_v3_geom.npz + .meta.json
tar -xzf v3_heads.tar.gz              # checkpoints/v3_heads/*.pt
python -m starlife_eval run --config configs/local_heads_v15.json \
  --data-root "$DATA_ROOT" --only starlife-v1-wsv --out runs/local-heads
```

The heads are tiny MLPs; the expensive part (27B backbone slot-feature extraction) is what produced
`spatial_v3.npz` and is not shipped as weights.

## C. Retrain the heads (method reproduction)

`scripts/` contains the method pipeline used to build the benchmark:

| script | step |
|---|---|
| `scripts/extract_features.py` | backbone slot/relation features (needs the backbone and GPU) |
| `scripts/build_slot_geometry.py` | 13-dim per-slot geometry from camera poses / bboxes |
| `scripts/build_entity_vocab.py` | entity vocabulary + frozen type vocabulary |
| `scripts/train_heads.py` | trains the five heads (`--slot-geom` for v3 geometry) |
| `scripts/validate_dataset.py` | dataset/meta sanity checks |
| `scripts/bench_pipeline.py` | single-frame / batch / crop latency measurement |

Training on the released features reproduces the local-heads row; feature extraction is the only
step that needs the original 27B backbone.

## Determinism checklist

- manifest sha256 prefix and features sha256 prefix are printed in every `report.json`;
- the 60-pair sample indices are in `report.json` (seed 20261020);
- JSON answers are parsed deterministically (no sampling: `temperature=0` in the reference configs);
- a rerun with the same model and same cache yields identical numbers.

## Common pitfalls

- `--live` missing → the runner replays cache; with no cache it errors instead of calling the API.
- A model that returns prose instead of the JSON schema gets normalised where possible; if the answer
  is unusable the pair is recorded as an empty prediction (visible in the JSONL cache) — check the
  cache, don't silently average.
- `api_key_env` must name an environment variable; the runner never stores keys.
