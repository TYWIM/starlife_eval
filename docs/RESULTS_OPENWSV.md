# OpenWSV 榜单结果

评测协议：[`docs/OPENWSV.md`](OPENWSV.md)（冻结框架 `openwsv-1`）。
生成日期：2026-10-23，仓库 `starlife_eval`。

分数口径：确定性判分、无 LLM judge；**未覆盖 ≠ 错**，准确率只在覆盖题上计算并给出
覆盖率；95% CI 为任务级 bootstrap（1,000 draws，seed=20261022）。**T1 是单项能力，
不是综合排名**；coverage 低于 100% 的行不可当作最终分。

---

## 已出分：T1 spatial relations

### T1 · SpatialMQA 全量 1,612 题

数据：`spatialmqa` 1,612 题（test 1,076 + validation 536），确定性字母匹配。
来源：v2 战役产物（同一批 1,612 题、同一确定性判分口径），OpenWSV `openwsv-1`
框架下复跑前沿用该读数。

| 排名 | 模型 | 覆盖率 | n | 准确率 | 95% CI |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | qwen3.8-flash | 100.0% | 1612 | **0.7655** | [0.7444, 0.7854] |
| 2 | deepseek-flash | 100.0% | 1612 | **0.7488** | [0.7289, 0.7674] |
| 3 | glm-5.3-flash | 100.0% | 1612 | **0.7388** | [0.7177, 0.7587] |
| 4 | minimax-m3.1-flash | 99.94% | 1611 | **0.6232** | [0.6002, 0.6468] |
| 5 | mimo-v2.6-pro | 100.0% | 1612 | **0.5459** | [0.5223, 0.5707] |

* `minimax-m3.1-flash` 有 1 题持续返回 provider 侧 `HTTP 404 {"code":"NOT_FOUND"}`，
  按未覆盖剔除分母（1,611/1,612），其余模型全部覆盖。
* 空输出（reasoning 模型耗尽 `max_tokens`）记为可重试错误、按未覆盖处理，不判错。

分 split 读数（同一 v2 战役）：

| 模型 | test (n=1076) | validation (n=536) |
| --- | ---: | ---: |
| qwen3.8-flash | 0.7649 | 0.7668 |
| deepseek-flash | 0.7426 | 0.7612 |
| glm-5.3-flash | 0.7351 | 0.7463 |
| minimax-m3.1-flash | 0.6372 (n=1075) | 0.5951 |
| mimo-v2.6-pro | 0.5437 | 0.5504 |

### T1-relation · 关系题子集 305 题

数据：`spatialmqa_relation` = 全量中形如
`Where is/are the X located relative to the Y?` 的 305 题（正则过滤，见
`wsv/tasks.py::RELATION_PATTERN`）。

| 排名 | 模型 | 覆盖率 | n | 准确率 |
| ---: | --- | ---: | ---: | ---: |
| 1 | glm-5.3-flash | 100.0% | 305 | **0.9279** |
| 2 | qwen3.8-flash | 100.0% | 305 | **0.9213** |
| 3 | deepseek-flash | 100.0% | 305 | **0.9082** |
| 4 | minimax-m3.1-flash | 99.67% | 304 | **0.8816** |
| 5 | mimo-v2.6-pro | 99.67% | 304 | **0.8684** |

* 子集的 95% CI 未在本轮记录（subset 读数来自 v2 战役的逐题预测重算）；需要时用
  `scoring.bootstrap_ci` 在 305 题覆盖向量上重算（seed=20261022，1,000 draws）。
* minimax-m3.1-flash / mimo-v2.6-pro 各有 1 题未覆盖（provider 侧 404），分母 304。

---

## 待跑：T4–T6（结构化世界模型）

`pairs` 套件（`dataset_spatial_v3` split=test，1,000 对）的 OpenWSV 框架读数**待跑**，
需要两件东西到位：框架检测器 `wsv/detector.py`（Lead 冻结）与可达的
StarLife-1-WSV-Lite `POST /predict` 服务。

| 能力 | suite | n | 状态 |
| --- | --- | ---: | --- |
| T4 state change / world-state consistency | `pairs` | 1000 | **待跑** |
| T5 object & attribute understanding | `pairs` | 1000 | **待跑** |
| T6 compositional reasoning | `pairs` | 1000 | **待跑** |

跑法（服务就绪后）：

```bash
python -m starlife_eval.wsv run --suite pairs --model starlife-1-wsv-lite \
    --adapter wsv --api-url http://127.0.0.1:8890 --out reports/openwsv/
```

`pairs` 会报 `change_top1 / change_macro_f1 / relation_type_acc / rel_pos_acc /
attrs_acc`（各带 pair 级 bootstrap CI），单个模型 2 次调用/对（forward + reversed，
方向匹配协议，见 `docs/OPENWSV.md` §4.2）。

> 兼容性说明：仓库里曾有一个 **provisional** 的 60 对结构化读数
> （`docs/RESULTS_STRUCTURED_TRACK.md`，change top-1 91.45% 等）。那一轮用的是
> **GT 框**、旧采样（60 对）与旧判分脚本，和 OpenWSV 的"框架检测器 + 1,000 对 +
> `openwsv-1`"口径不可比，**不进本榜**；T4–T6 以 `pairs` 套件重跑结果为准。

---

## 未参与本轮

* `claude-opus-5-5`（网关配额耗尽）、`seed-2.1-turbo`（未启动）。
* MindEdit（T3）在 v2 战役中撤回（曾把截断空回复误判为错），不参与本榜。

---

## 复现与产物

* 逐题预测/原始响应随各轮运行产物发布；判分脚本在本仓库
  `src/starlife_eval/wsv/`（scoring/state/runner）。
* 数据版本：`spatialmqa` sha256 `7be678f02a21b94696aef9aa31d6f3a27fed278d3a8bc66ae06d6e987d0c7e50`；
  `pairs` sha256 `a29af6e171d55767622ba073af59ebb6d13a65cdbb2c69cb112632cb7500558b`。
* 榜单 JSON 契约见 `docs/OPENWSV.md` §5；站点数据生成读取
  `framework / suite / n / model / metrics / per_task` 字段。

---

## 口径提醒

1. 这些都是**单项能力**读数，不构成"综合智能"或"SOTA"结论。
2. 存在 provider 侧不可控因素（版本、隐藏 system prompt、限流）：runner 统一传
   `temperature=0`、同一提示词模板；模板公开在 `wsv/prompts.py`。
3. 覆盖率 < 100% 的行（minimax / mimo 各 1 题）不能当作最终分直接比较；差异在 1 题量级时
   也与 CI 宽度同阶，不应过度解读名次。
