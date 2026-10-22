# SpatialMQA (T1) — v2 task matrix

1612 tasks (test + validation), deterministic scoring, 95% bootstrap CI over covered tasks.

| model | coverage | n | accuracy | 95% CI | complete |
|---|---|---|---|---|---|
| qwen3.8-flash | 100.0% | 1612 | **0.7655** | 0.7444–0.7854 | yes |
| deepseek-flash | 100.0% | 1612 | **0.7488** | 0.7289–0.7674 | yes |
| glm-5.3-flash | 100.0% | 1612 | **0.7388** | 0.7177–0.7587 | yes |
| minimax-m3.1-flash | 99.9% | 1611 | **0.6232** | 0.6002–0.6468 | no |
| mimo-v2.6-pro | 100.0% | 1612 | **0.5459** | 0.5223–0.5707 | yes |

Split breakdown:

| model | test | validation |
|---|---|---|
| qwen3.8-flash | 0.7649 (1076) | 0.7668 (536) |
| deepseek-flash | 0.7426 (1076) | 0.7612 (536) |
| glm-5.3-flash | 0.7351 (1076) | 0.7463 (536) |
| minimax-m3.1-flash | 0.6372 (1075) | 0.5951 (536) |
| mimo-v2.6-pro | 0.5437 (1076) | 0.5504 (536) |

Not evaluated: `claude-opus-5-5` (gateway quota), `seed-2.1-turbo` (not started).
