# v1.5 public results archive

These JSON files are derived artifacts copied from the evaluation runs behind the public board
(openwsv). They are kept here so the published numbers can be inspected without re-running models.

| file | contents |
|---|---|
| `v3_only_leaderboard.json` | leaderboard row for the local v3 heads, 60-pair sample from `dataset_spatial_v3` (seed 20261020) |
| `v3_only_confidence_intervals.json` | 95% pair-level bootstrap intervals for that run (1,000 draws, seed 20261021) |
| `v15_partial_leaderboard.json` | the four rows re-scored on the v1.5 sample at the time of the release: `starlife-v1-wsv` 95.2, `gpt-6.1-sol` 73.3, `gemini-3.8-flash` 72.1, `deepseek-flash` 69.1 |
| `v15_partial_confidence_intervals.json` | bootstrap intervals for the same four rows |

Fields of interest in each row: `compositeScore` (equal-weight 7-dim mean), `composite.dims`
(per-dimension values 0–100 and sources), `metrics.pair_protocol` (raw per-dimension metrics),
`na` (why a dimension is N/A), `version`, `latency`.

Rerun (server-style reproduction):

```bash
python -m starlife_eval run --config configs/local_heads_v15.json --data-root "$DATA_ROOT" \
  --only starlife-v1-wsv --out runs/local-heads
python tools/confidence_intervals.py --config configs/local_heads_v15.json --data-root "$DATA_ROOT" \
  --out runs/local-heads/confidence_intervals.json --report runs/local-heads/report.json
```

Other rows on the board (cloud VLMs) are reproducible by running the same 60 pairs with your own
OpenAI-compatible endpoint — see `docs/REPRODUCE.md`. Their raw provider outputs are not
redistributed.
