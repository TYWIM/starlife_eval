# OpenWSV (`starlife_eval`) — World-State Vision Benchmark framework

**OpenWSV** (short: **WSV**) is a horizontal benchmark that evaluates, under **one frozen
framework**:

- **text-only multimodal models (VLMs)**: multiple-choice / short-answer / JSON answers;
- **structured world-state models (not video generation)**: consume a before/after frame pair
  plus the **framework-provided object boxes**, and emit structured slots, object-pair
  relations and events (via StarLife-1-WSV `POST /predict`).

Only if every framework component is byte-identical can results from different model classes
and different runs be compared.

The T4/T6 text suites are **text projections of the structured pairs GT**: they ask the same
atomic facts (what changed, what the after state is) with the same images, so text VLMs and
structured world-state models can be scored on the same underlying tasks. T5/T7 add
relation-change and cross-frame suites from the additive `openwsv-data-1.1` bundle.

## The frozen five

| Item | Value | Where it lives |
| --- | --- | --- |
| Dataset | sha256 of the suite files (shipped in the package) | `src/starlife_eval/wsv/suites/*.jsonl` + `framework.dataset_sha256` in result JSON |
| Detector | GroundingDINO-tiny, weights sha256 pinned, `box_threshold=0.2`, `text_threshold=0.2` | `wsv/config.py::DETECTOR`, `wsv/detector.py::FrozenDetector` (all box-consuming models share the exact same boxes) |
| Prompt | `openwsv-1` | `wsv/prompts.py::PROMPT_VERSION` |
| Scorer | `starlife_eval@<git commit>`, deterministic, no LLM judge | `wsv/scoring.py`, `wsv/state.py` |
| Seeds | sampling `20261020`, bootstrap `20261022` | `wsv/config.py::SEEDS` |

> The detector belongs to the **framework**, not to any model: otherwise a score degenerates
> into "detector × state model" and can be neither attributed nor reproduced.

## Suites

| suite | n | description | applies to |
| --- | ---: | --- | --- |
| `spatialmqa` | 1,612 | single-image multiple-choice spatial relation questions (T1) | text VLMs |
| `spatialmqa_relation` | 305 | the "Where is/are the X located relative to the Y?" subset of the above | text VLMs |
| `pairs` | 1,000 | before/after frame pairs from the `dataset_spatial_v3` test split (T4–T6) | structured world-state models |
| `change` | 1,000 | T4: one "what happened to the X?" question per test pair, derived from the pairs GT | text VLMs (paired with `pairs`) |
| `relation_change` | 600 | T5: before/after relation-change questions on synthetic pairs (`openwsv-data-1.1`) | text VLMs |
| `state_replay` | 1,000 | T6: after-state fact probes on the same test pairs (existence / relation / rel_pos / contact) | text VLMs (paired with `pairs`) |
| `cross_frame` | 400 | T7: disjoint cross-frame questions ("X only in image 1, Y only in image 2") | text VLMs |

`pairs` ground truth is produced by the synthetic scene generator (`spatial3d-1.0.0`, fixed
seeds) — **not real footage**; relation/change/contact labels come from the generator.
`change` and `state_replay` are deterministic projections of exactly that GT
(`scripts/wsv/derive_text_suites.py`), so every text question has a structured counterpart.
`relation_change` and `cross_frame` come from the additive `openwsv-scene-1.0.0` generator
(`scripts/wsv/gen/`), also seeded and regenerable byte-for-byte.

## Data bundle (clone and reproduce)

Images are not downloaded separately: the bundle ships **as 13 parts** inside the repo under
`data/openwsv-data-1.0/` (494.6 MB total; the joined zip must hash to
`971090ee15347b016f5c4afd92b6a711481d81de169c9d45d55e64a6fef2ad63`). The join script checks
the full zip hash and then unpacks:

```bash
python scripts/wsv/join_data.py --dest ./openwsv-data-1.0.zip --extract-to ./data
export OPENWSV_DATA_ROOT=$PWD/data/openwsv-data-1.0
```

The bundle contains 1,583 single-image question images, 2,000 frames (1,000 test pairs), both
suite jsonl files and a per-file `SHA256SUMS`. 29 tasks reference images missing from the
public release; they are scored as **not covered** and excluded from the accuracy denominator.

### Additive suites: `openwsv-data-1.1`

T5/T7 images are **not** part of the 1.0 bundle; they are generated deterministically (seed
`20261020`, version `openwsv-scene-1.0.0`) with the bundled generator:

```bash
python -m scripts.wsv.gen            # writes data/openwsv-data-1.1/ (2,000 JPEGs + SHA256SUMS)
python scripts/wsv/derive_text_suites.py   # T4/T6 jsonl from the frozen pairs GT
```

Run T4–T7 with a data root that contains both packages and keep the image paths prefixed:

```bash
python -m starlife_eval.wsv run --suite change --model <id> --adapter openai \
    --base-url "$OPENAI_BASE_URL" --data-root data --out reports/openwsv-v2/
```

(`data/` contains the extracted `openwsv-data-1.0/` plus the generated
`openwsv-data-1.1/`; the suite files point at `openwsv-data-1.0/...` / `openwsv-data-1.1/...`.)

## Quick start

```bash
pip install -e ".[dev,detector]"

# list suites with their sha256
python -m starlife_eval.wsv --list-suites

# text VLM through an OpenAI-compatible gateway
python -m starlife_eval.wsv run --suite spatialmqa --model qwen3.8-flash \
  --adapter openai --base-url "$OPENAI_BASE_URL" --api-key "$OPENAI_API_KEY" \
  --out reports/openwsv/spatialmqa__qwen3.8-flash.json

# structured world-state model (framework detector + StarLife-1-WSV API)
python -m starlife_eval.wsv run --suite pairs --model starlife-1-wsv \
  --adapter wsv --api-url "$WSV_API_URL" --api-key "$WSV_API_KEY" \
  --out reports/openwsv/pairs__starlife-1-wsv.json
```

## Result JSON contract

```json
{"framework": {"version": "openwsv-1", "dataset_sha256": {},
               "detector": {"model": "IDEA-Research/grounding-dino-tiny",
                            "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
                            "weights_sha256": "1a2412ef...", "box_threshold": 0.2,
                            "text_threshold": 0.2},
               "prompt_version": "openwsv-1", "scorer": "starlife_eval@<commit>",
               "seed": 20261020, "bootstrap_seed": 20261022},
 "suite": "spatialmqa", "n": 1612,
 "model": {"id": "...", "kind": "vlm", "uses_harness_boxes": false, "adapter": "openai"},
 "metrics": {"accuracy": 0.0, "coverage": 1.0, "ci95": [0.0, 0.0]},
 "per_task": [{"task_id": "...", "covered": true, "pred": 0, "gold": 1, "error": null}]}
```

**Not covered ≠ wrong**: empty output, timeout and unparseable answers are recorded as
`covered=false` and removed from the accuracy denominator; coverage is reported separately.
Every accuracy carries a 95% task-level bootstrap CI (1,000 iterations, `seed=20261022`).

## Layout

```
src/starlife_eval/wsv/         # framework: config / detector / tasks / prompts / adapters / scoring / state / runner / cli
src/starlife_eval/wsv/suites/  # frozen task files (7 suites, jsonl, shipped in the package)
src/starlife_eval/verifiers/   # structured scoring (pair_protocol etc.)
data/openwsv-data-1.0/         # 13 data-bundle parts + hashes
data/openwsv-data-1.1/         # generated T5/T7 frames (local, reproducible; gitignored)
scripts/wsv/                   # join/derive/generate/summarize helpers
scripts/wsv/gen/               # openwsv-scene-1.0.0 T5/T7 generator (numpy + Pillow)
scripts/wsv/derive_text_suites.py  # T4/T6 projections of the pairs GT
scripts/wsv/build_site_data.py     # reports -> site/src/data/openwsv.generated.json
scripts/wsv/summarize_results.py   # markdown tables + exact McNemar tests
tests/                         # offline unit tests (pytest, no network/GPU)
```

## Site

The leaderboard lives in the nested `site/` repo (Vue 3 + Vite, static build) and is data-driven
by `site/src/data/openwsv.generated.json`, which is regenerated from result JSONs with
`python scripts/wsv/build_site_data.py` after a campaign. It reports coverage and CIs next to
every score and links the upstream project repository.

## License and data notes

- Code: MIT (see `LICENSE`).
- Data: synthetic scenes generated by `spatial3d-1.0.0` (pairs) and `openwsv-scene-1.0.0`
  (T5/T7); the spatial-relation question set is derived from its public dataset under the
  original license. T4/T6 suites are deterministic text projections of the pairs GT and add
  no images.
- The benchmark reports metrics, coverage and CIs only; it makes no composite ranking claim
  beyond the published composite definition and uses no SOTA/#1 wording.
