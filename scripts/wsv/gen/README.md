# `scripts/wsv/gen` — OpenWSV scene generator

Deterministic, dependency-free generator for the synthetic frames and task files of
the OpenWSV suites **T5 `relation_change`** (600 tasks) and **T7 `cross_frame`**
(400 tasks).  Pure Python + numpy + Pillow: analytic ray casting of shaded
boxes / cylinders / spheres on a fixed tabletop set, no external assets, no
network access, no fonts or text drawn into the images.

- **generator version**: `openwsv-scene-1.0.0`
- **seed**: `20261020`
- **image format**: 1024x768 JPEG, quality 90, 4:4:4 (`subsampling=0`)
- **camera**: one fixed pinhole camera shared by both frames of every task
  (`fx = fy = 800`, `cx = 512`, `cy = 384`)

All randomness is derived per task from
`SeedSequence([SEED, suite_code, task_index])` (suite code 5 for T5, 7 for T7),
so task *i* is identical no matter which other tasks are generated (pilot runs
are byte-identical to the first *i* tasks of a full run).

## Layout

```
scripts/wsv/gen/
├── __init__.py       # GENERATOR_VERSION, SEED, reproduce command
├── __main__.py       # python -m scripts.wsv.gen
├── build.py          # CLI / orchestration / data package / sheets / report / determinism
├── camera.py         # pinhole camera, projection helpers
├── objects.py        # object construction, descriptions, projected bbox helpers
├── scenes.py         # task planning: T5/T7 scene layouts and 3D ground truth
├── render.py         # numpy ray caster (shading, culling, JPEG output)
├── suites.py         # row schema builder for the two jsonl suites
├── contacts.py       # visual QA contact sheets
└── validation.py     # end-to-end checks (rows, images, sha256, GT re-derivation)
```

## Reproduce (from the repository root)

Full canonical regeneration — writes `data/openwsv-data-1.1/` (2,000 JPEGs +
`SHA256SUMS` + `meta.json` + `README.md`), both suite jsonl files, the contact
sheets in `_local/`, and the generation report:

```bash
C:\Users\MECHREVO\miniconda3\python.exe -X utf8 -m scripts.wsv.gen.build --force \
    --determinism-check --report _local/REPORT_DATA_GEN.md
```

Pilot / smoke run (first 10 tasks per suite, useful to eyeball new scenes):

```bash
C:\Users\MECHREVO\miniconda3\python.exe -X utf8 -m scripts.wsv.gen.build \
    --limit 10 --out-root _local/pilot --force --no-data-files
```

Determinism re-check (renders the first 5 tasks twice in separate fresh
processes and compares every byte; exits non-zero on mismatch):

```bash
C:\Users\MECHREVO\miniconda3\python.exe -X utf8 -m scripts.wsv.gen.build \
    --limit 5 --determinism-check --force
```

Useful flags: `--limit N`, `--out-root DIR`, `--workers N`, `--seed N`,
`--no-images`, `--no-data-files`, `--no-sheets`, `--report PATH`,
`--determinism-check`, `--force`, `--quiet`.

## Ground truth

- **T5 `relation_change`** — question asks how the relation between object A and
  object B changed between the two frames; fixed 4 choices
  `[It appeared. / It disappeared. / It changed. / It stayed the same.]`
  (answer index = class).  Classes are 150 each (`appeared`, `disappeared`,
  `changed`, `stayed_same`).  A relation signature is the image-plane direction
  of A's projected bbox centre relative to B's (`left/right/above/below`) plus a
  projected centre-distance bucket (`close/mid/far`); the class is derived by
  diffing the before/after structured state, never from filenames.
  `appeared` / `disappeared` scenes differ between the frames only by the
  transient object; `changed` moves/redraws A enough to cross a signature edge;
  `stayed_same` keeps the A–B signature while some distractor moves.
- **T7 `cross_frame`** — frame1 and frame2 have completely disjoint object sets
  (no object appears in both), 3 objects per frame, same camera.
  Question: "The {X} appears only in the first image, and the {Y} appears only in
  the second image. In the shared camera view, where is the {X} relative to the
  {Y}?" Choices `[Left. / Right. / Above. / Below.]`, 100 tasks per direction.
  The direction is measured between projected bbox centres and is required to be
  unambiguous: the dominant axis must dominate by >= 1.5x and the main-axis gap
  must be >= 8% of the image width/height.

Every row's `meta` stores the full structured state (entities with 2D bbox,
type, colour, 3D centre, depth; relations for T5; GT direction/gaps for T7).
`meta` exists for audit and future structured scoring; the runner never sends it
to a model — only `question` / `images` / `choices` reach the prompt.

## Determinism

The renderer is pure numpy floating-point math with fixed operation order; JPEG
saving uses fixed parameters (`quality=90, subsampling=0, optimize=False,
progressive=False`) and no timestamps are written.  `--determinism-check`
renders the first 5 tasks of each suite twice in fresh processes and compares
file-by-file sha256; the generation report records `identical=True` and the
combined digest.  Changing `SUPERSAMPLE`, the JPEG parameters or the room/camera
constants changes the images and therefore the canonical hashes.

## Cost

About 0.7–1.0 s per frame on one core (screen-space culling keeps small objects
cheap) and a worker RSS of a few tens of MB; the full 2,000-frame generation
takes a couple of minutes on a few cores.
