# Organizer reference baseline (not part of the benchmark)

The five heads in `local_heads_v15.json` belong to the organizers' reference model
(frozen Qwen3.8-27B slot features + five MLP heads trained on the synthetic training split).

They are **not** required to run the benchmark and are **not** part of the benchmark data:
any OpenAI-compatible multimodal model is scored through the same prompt/schema/verifier path.

The baseline is reported separately so the board does not present the organizers' own model as a
candidate in the general ranking. If you want to reproduce it, the (optional) baseline artifacts
are published outside the benchmark release; the benchmark itself only needs the dataset archive.
