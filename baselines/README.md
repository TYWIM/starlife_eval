# Organizer model (ordinary benchmark entry)

The five heads in `local_heads_v15.json` belong to the organizers' model
(frozen Qwen3.8-27B slot features + five MLP heads trained on the synthetic training split).

This model participates in the ranking under exactly the same protocol as every other model:
same task data, same prompt, same answer schema, same scorer. The heads are an internal
implementation detail; head-level metrics are auxiliary diagnostics and never the main score,
so a model without heads is scored on the same footing.

Weights are not distributed with the benchmark release. The row's model outputs are published
as a normal output cache, so it can be re-scored and inspected like any other entry.
