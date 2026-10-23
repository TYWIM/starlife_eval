# OpenWSV (`starlife_eval`) — 世界状态视觉评测框架

**OpenWSV**（简称 **WSV**）是一个横测榜单：在**同一个冻结框架**下，同时评测

- **文本多模态模型（VLM）**：单图/多图多选、短答、JSON 输出；
- **结构化世界模型（非视频生成）**：输入 before/after 两帧 + **框架检测器给出的物体框**，输出结构化变化、物体对关系与状态（StarLife-1-WSV 的 `POST /predict`）。

只有框架逐位一致，两类模型、不同轮次的结果才可比。

## 冻结框架五件套

| 冻结项 | 取值 | 落地位置 |
| --- | --- | --- |
| 数据版本 | suite 文件 sha256（随包分发） | `src/starlife_eval/wsv/suites/*.jsonl` + 结果 JSON `framework.dataset_sha256` |
| 检测器 | GroundingDINO-tiny，权重 sha256 固定、`box_threshold=0.2`、`text_threshold=0.2` | `wsv/config.py::DETECTOR` + `wsv/detector.py::FrozenDetector`（所有需要框的模型共用同一份输出） |
| 提示词 | `openwsv-1` | `wsv/prompts.py::PROMPT_VERSION` |
| 判分器 | `starlife_eval@<git commit>`，确定性判分，无 LLM judge | `wsv/scoring.py`、`wsv/state.py` |
| 随机种子 | 采样 `20261020`，bootstrap `20261022` | `wsv/config.py::SEEDS` |

> 检测器属于**框架**，不属于任何模型：否则"分数"退化成"检测器 × 状态模型"，无法归因、无法复现。

## 数据集（三套 suite）

| suite | n | 说明 | 适用 |
| --- | ---: | --- | --- |
| `spatialmqa` | 1,612 | SpatialMQA 单图多选（T1 空间关系） | 文本 VLM |
| `spatialmqa_relation` | 305 | 上述中 "Where is/are the X located relative to the Y?" 关系子集 | 文本 VLM + 结构化模型同表 |
| `pairs` | 1,000 | `dataset_spatial_v3` test 划分的 before/after 帧对（T4–T6） | 结构化世界模型 |

`pairs` 的 GT 来自合成场景生成器（`spatial3d-1.0.0`，固定种子），**不是真实拍摄**；关系/变化/接触支撑标签由生成器给出。

## 快速开始

```bash
pip install -e ".[dev,detector]"

# 列出 suite（含 sha256）
python -m starlife_eval.wsv --list-suites

# 文本 VLM（OpenAI 兼容网关）
python -m starlife_eval.wsv run --suite spatialmqa --model qwen3.8-flash \
  --adapter openai --base-url "$OPENAI_BASE_URL" --api-key "$OPENAI_API_KEY" \
  --out reports/openwsv/spatialmqa__qwen3.8-flash.json --limit 50

# 结构化世界模型（框架检测器 + StarLife-1-WSV API）
python -m starlife_eval.wsv run --suite pairs --model starlife-1-wsv \
  --adapter wsv --api-url "$WSV_API_URL" --api-key "$WSV_API_KEY" \
  --out reports/openwsv/pairs__starlife-1-wsv.json --limit 10
```

结果 JSON 契约（稳定字段）：

```json
{"framework": {"version": "openwsv-1", "dataset_sha256": {...},
               "detector": {"model": "...", "revision": "...", "weights_sha256": "...",
                            "box_threshold": 0.2, "text_threshold": 0.2},
               "prompt_version": "openwsv-1", "scorer": "starlife_eval@<commit>",
               "seed": 20261020, "bootstrap_seed": 20261022},
 "suite": "spatialmqa", "n": 1612,
 "model": {"id": "...", "kind": "vlm", "uses_harness_boxes": false, "adapter": "openai"},
 "metrics": {"accuracy": 0.0, "coverage": 1.0, "ci95": [0.0, 0.0]},
 "per_task": [{"task_id": "...", "covered": true, "pred": 0, "gold": 1, "error": null}]}
```

**未覆盖 ≠ 错**：空输出、超时、无法解析一律记为 `covered=false`，从准确率分母中剔除，覆盖率单独报告。所有准确率带 95% 任务级 bootstrap CI（1,000 次，`seed=20261022`）。

## 数据包（克隆即可复现）

图像不单独下载：数据包以 **13 个分片**随仓库提交在 `data/openwsv-data-1.0/`
（合计 494.6 MB，拼回后的 zip sha256 = `971090ee15347b016f5c4afd92b6a711481d81de169c9d45d55e64a6fef2ad63`）。
分片是为了绕开单文件 100 MB 限制，join 脚本会校验完整 sha256：

```bash
python scripts/wsv/join_data.py --dest ./openwsv-data-1.0.zip \
  --extract-to ./data
export OPENWSV_DATA_ROOT=$PWD/data/openwsv-data-1.0
```

包内包含：1,583 张单图题面图像、2,000 张前后帧（1,000 对 test）、两份 suite jsonl
与逐文件 `SHA256SUMS`；29 题缺图按未覆盖处理，不计入准确率分母。

## 目录

```
src/starlife_eval/wsv/         # 框架：config / detector / tasks / prompts / adapters / scoring / state / runner / cli
src/starlife_eval/wsv/suites/  # 冻结任务文件（jsonl，随包分发）
src/starlife_eval/verifiers/   # 结构化判分实现（pair_protocol 等）
tests/                         # 离线单测（pytest，无需网络/GPU）
scripts/wsv/run_suite.py       # 命令行薄封装示例
```

## 许可与数据说明

- 代码：MIT（沿用仓库 LICENSE）。
- 数据：合成数据由 `spatial3d-1.0.0` 生成；SpatialMQA 任务与真值来自其公开数据集，按原许可引用。
- 榜单只报指标、覆盖率与 CI；不报综合分排名，不使用 SOTA/榜首表述。
