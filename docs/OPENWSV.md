# OpenWSV 冻结评测框架

OpenWSV（简称 **WSV**）是 StarLife-1-WSV 观测模型横测榜单的评测框架。榜单在**同一个冻结
框架**下同时评两类模型：

* **文本 VLM**（`kind="vlm"`）：单图/多图多选，输出一个选项字母；
* **结构化世界模型**（`kind="structured"`）：输入 before/after 两帧 + 框架检测器给出的框，
  输出结构化变化/关系（StarLife-1-WSV-Lite 的 `POST /predict`）。

只有框架完全一致，两类模型、以及不同轮次的结果才可比。所有评测入口都在
`src/starlife_eval/wsv/`，任务文件冻结在 `src/starlife_eval/wsv/suites/`。

---

## 1. 五件冻结物

| 冻结项 | 取值 | 落地位置 |
| --- | --- | --- |
| 数据版本 | suite 文件 sha256（见 §2），随包分发 | `wsv/suites/*.jsonl` + 结果 JSON `framework.dataset_sha256` |
| 检测器 | GroundingDINO-tiny `IDEA-Research/grounding-dino-tiny`，权重 sha256、`box_threshold=0.2`、`text_threshold=0.2` | `wsv/config.py::DETECTOR`（参数）+ `wsv/detector.py::FrozenDetector`（推理；所有需要框的模型共用同一份输出） |
| 提示词版本 | `openwsv-1` | `wsv/prompts.py::PROMPT_VERSION`（与 `config.py` 同步） |
| 判分器 | `starlife_eval@<git short commit>` | 结果 JSON `framework.scorer`（运行时取 `git rev-parse --short HEAD`，可用 `OPENWSV_SCORER` 覆盖） |
| 随机种子 | 任务/采样 `20261020`；bootstrap `20261022` | `config.py::SEEDS`，写进结果 JSON |

检测器属于框架：**模型不自带检测器**，否则"感知"和"推理"混在一起无法横测。

* 入口：`FrozenDetector().detect(image, phrases)`，返回
  `{phrase: [x1, y1, x2, y2], "_scores": {...}, "_meta": {...}, "_image_size": [w, h]}`
  （同时兼容模块级 `detect(image, phrases)` 的旧接口）；
* runner 把它转成 `{width, height, before: [{id, bbox}], after: [{id, bbox}]}`，同一轮评测
  只加载一次检测器、所有结构化模型共用；`_meta`（含真实 `weights_sha256`）写进结果 JSON；
* 文本 VLM 不需要框（`uses_harness_boxes=false`），其 `framework.detector.weights_sha256`
  在权重未加载时为 `pending-load`（用 `OPENWSV_DETECTOR_WEIGHTS_SHA256` 可显式覆盖）。

---

## 2. 数据集三套 suite

| suite | n | 说明 | 冻结文件 | sha256 |
| --- | ---: | --- | --- | --- |
| `spatialmqa` | 1612 | SpatialMQA 单图多选，T1 spatial relations（test 1076 + validation 536） | `suites/spatialmqa.jsonl` | `7be678f02a21b94696aef9aa31d6f3a27fed278d3a8bc66ae06d6e987d0c7e50` |
| `spatialmqa_relation` | 305 | 上者中正则 `^Where (?:is\|are) the (?P<x>.+?) located relative to the (?P<y>.+?)\?$` 筛出的关系题子集（T1-relation） | 同 `suites/spatialmqa.jsonl`（加载时过滤） | 同上 |
| `pairs` | 1000 | `dataset_spatial_v3` split=test 的 before/after 帧对，结构化世界模型 T4–T6 | `suites/pairs_test.jsonl` | `a29af6e171d55767622ba073af59ebb6d13a65cdbb2c69cb112632cb7500558b` |

`load_suite(name)` 返回 `list[dict]`，每项固定字段：

```json
{"task_id": "...", "suite": "...", "question": "...", "images": ["..."],
 "choices": ["..."], "answer": 0, "answers": [], "meta": {}}
```

* 文本多选：`answer` 是 0-based 选项下标；`meta` 含 `capability/dataset/split/answer_type`，
  关系子集额外含 `relation_x/relation_y`；
* `pairs`：`answer=None`、`choices=[]`，真值在 `meta`：`before/after`（相对数据集根）、
  `entities[{slot,type_key,bbox,present_after}]`、`changes{slot: label}`、
  `relations[{from_slot,to_slot,type,rel_pos,rel_pos_present,contact,support,dist_m}]`、
  `pair_id/split/negative`。GT 框只用于重建判分 manifest，**不进入模型输入**；模型输入
  的框一律来自框架检测器。

样本量在加载时与 `SUITE_SIZES` 校验：数据被改动会直接报错（不允许静默漂移）。
`--list-suites` 打印各 suite 的 n、sha256 和文件路径。

> 发布提示：仓库 `.gitignore` 有 `*.jsonl` 规则（原本用于忽略本地数据），`wsv/suites/`
> 下的冻结任务文件需要 `git add -f src/starlife_eval/wsv/suites/*.jsonl` 才会进版本库，
> 否则 clone 后 `load_suite` 会报缺文件。sha256 以上表为准。

---

## 3. 提示词（`openwsv-1`）

`build_prompt(task, *, protocol)` 支持三种 protocol，文本对同一 `(task, protocol)` **逐字稳定**
（不依赖路径、时间、随机数、环境变量）：

| protocol | 用途 | 关键约束 |
| --- | --- | --- |
| `choice` | 文本多选（默认） | 列出 `A./B./...` 选项，要求**只输出一个选项字母** |
| `short` | 短答 | 要求一行 `Answer: <短答>` |
| `json` | 结构化输出 | 只输出一个 JSON 对象；schema 来自 `task.meta.schema` 或默认 schema |

文本 suite 固定用 `choice`。改模板必须同时升级 `PROMPT_VERSION`，否则与历史结果不可比。

---

## 4. 判分（确定性，无 LLM judge）

### 4.1 文本多选

* `parse_choice(text, n_choices)` **严格解析**，只接受 `"A"`、`"(C)"`、`"Answer: D"`、
  `"选 A"`、`"选项 B"`、`` `B` `` 等显式形式；自由文本不猜字母；
* 解析失败 / 调用报错 / 空输出 = **未覆盖（uncovered）**，从准确率分母剔除，
  **不按错题计**（`coverage = n_covered / n`）；
* `accuracy` 只在覆盖题上算；`ci95` 是覆盖题正确性向量上的任务级 bootstrap
  （1,000 draws，分位法，seed=20261022）；
* 两模型对比用精确 McNemar 检验（`scoring.mcnemar`，不一致对上的双侧二项精确 p 值）。

### 4.2 结构化 pairs

WSV 服务按 `combinations(slots, 2)` **只输出升序对**，所以每个 pair 跑两次：
`forward`（检测列表按 manifest slot 升序）与 `reversed`（逆序）。每条 GT 关系取**与 GT
方向一致**的那次响应：`from_slot < to_slot` 用 forward，否则用 reversed。GT 方向只决定
输入顺序，标签不进入模型。

指标定义直接复用
[`verifiers/pair_protocol.py`](../src/starlife_eval/verifiers/pair_protocol.py)
（`evaluate_pair_protocol`，只读复用，不重复实现）：

| 指标 | 定义 |
| --- | --- |
| `change_top1` | 所有 entity slot 的 change 逐槽 top-1（GT 缺省 `none`） |
| `change_macro_f1` | `macro_f1_compare` 口径（与 60 对云端协议一致） |
| `relation_type_acc` | GT 关系上的关系类型命中率 |
| `rel_pos_acc` | 类型命中且 GT 声明了 `rel_pos` 时的方位命中率 |
| `attrs_acc` | 类型命中时 contact/support 各 1 项 |

CI 用 **pair 级 bootstrap**（重采样 pair，1,000 draws，seed=20261022）；两次响应缺一的
pair 记为未覆盖，从分母剔除。

---

## 5. 结果 JSON 契约（字段稳定）

每个模型写一个 `{out_dir}/{suite}__{model}.json`；单模型运行时另写 `result.json`。

```json
{
  "framework": {
    "version": "openwsv-1",
    "dataset_sha256": "<suite 文件 sha256>",
    "detector": {
      "model": "IDEA-Research/grounding-dino-tiny",
      "weights_sha256": "<权重 sha256；未加载时为 pending-load>",
      "box_threshold": 0.2,
      "text_threshold": 0.2
    },
    "prompt_version": "openwsv-1",
    "scorer": "starlife_eval@<commit>",
    "seed": 20261020,
    "bootstrap_seed": 20261022
  },
  "suite": "spatialmqa",
  "n": 1612,
  "model": {
    "id": "qwen3.8-flash",
    "kind": "vlm",
    "uses_harness_boxes": false,
    "adapter": "openai-chat:qwen3.8-flash"
  },
  "metrics": {"accuracy": 0.7655, "coverage": 1.0, "ci95": [0.7444, 0.7854]},
  "per_task": [
    {"task_id": "spatialmqa-000000024001", "covered": true, "pred": 1, "gold": 1, "error": null}
  ]
}
```

`kind` ∈ `vlm | structured`；`per_task` 的 `pred/gold` 在文本 suite 是选项下标，在 `pairs`
是"关系类型命中数 / GT 关系数"。`pairs` 的 `metrics` 额外带
`change_top1 / change_macro_f1 / relation_type_acc / rel_pos_acc / attrs_acc /
ci95_detail / change_slots / relation_pairs`，其中 `metrics.accuracy` 等于
`change_top1`（主指标）。未覆盖题的 `error` 记录原因（`unparsable choice`、
`missing prediction`、`adapter: ...`），覆盖题为 `null`。

契约样例（fixture 假模型，仅验证字段，**不是真实分数**）：
`.pytest_tmp/wsv_contract_samples/`。

---

## 6. CLI 与复现

```bash
# 列表
python -m starlife_eval.wsv --list-suites

# T1 文本 VLM（OpenAI 兼容端点）
python -m starlife_eval.wsv run --suite spatialmqa \
    --model qwen3.8-flash --adapter openai \
    --base-url https://api.example.com/v1 --out reports/openwsv/

# 冒烟 / 子集
python -m starlife_eval.wsv run --suite spatialmqa_relation \
    --model qwen3.8-flash --adapter openai \
    --base-url https://api.example.com/v1 --out reports/openwsv/ --limit 20

# 结构化世界模型（需 detector.py + 可达的 WSV 服务）
python -m starlife_eval.wsv run --suite pairs --model starlife-1-wsv-lite \
    --adapter wsv --api-url http://127.0.0.1:8890 --out reports/openwsv/ --limit 2

# 薄封装示例（读环境变量）
python scripts/wsv/run_suite.py
```

环境变量（密钥只从环境变量/一次性参数读入内存，**绝不写盘、不写日志、不写结果 JSON**）：

| 变量 | 说明 |
| --- | --- |
| `OPENWSV_API_KEY`（回退 `WSV_API_KEY` / `V2_API_KEY`） | 密钥；也可 `--api-key-env NAME` 指定变量名 |
| `OPENWSV_DATA_ROOT` | 图像相对路径根目录（`spatialmqa.jsonl` 里是 `v2data/...`） |
| `OPENWSV_DATASET_DIR` | `dataset_spatial_v3` 根目录（含 `frames/` 与 `manifest.jsonl`） |
| `OPENWSV_DETECTOR_WEIGHTS_SHA256` 等 | 覆盖 detector 冻结参数（通常由 `detector.py` 提供） |
| `OPENWSV_SCORER` | 覆盖判分器 commit 标识 |

约束：本地仓库只做离线判分与编排；**不联网跑模型、不下载权重**。跑真模型需要你自己
提供端点/服务并显式执行 CLI。

---

## 7. 目录结构

```
src/starlife_eval/wsv/
├── __init__.py       # 导出 load_suite / run_suite
├── __main__.py       # python -m starlife_eval.wsv
├── suites/           # 冻结任务文件（spatialmqa.jsonl / pairs_test.jsonl）
├── tasks.py          # load_suite / dataset_sha256 / describe_suites
├── prompts.py        # PROMPT_VERSION / build_prompt
├── adapters.py       # HttpVlmAdapter / WsvApiAdapter（异常不抛）
├── scoring.py        # parse_choice / score_choices / bootstrap_ci / mcnemar
├── state.py          # WSV 响应 -> PairPrediction、方向匹配、指标 + CI
├── runner.py         # run_suite 编排 + 结果 JSON 契约
├── cli.py            # 命令行
├── config.py         # 框架配置（Lead 维护）
└── detector.py       # 框架检测器（Lead 维护）
tests/test_wsv_tasks.py
tests/test_wsv_scoring.py
tests/test_wsv_state.py
scripts/wsv/run_suite.py
```

单测（不联网，使用包内冻结任务与手写 fixture；未 pip install 时测试文件会把 `src/`
自动加入 `sys.path`）：

```bash
python -m pytest tests/test_wsv_tasks.py tests/test_wsv_scoring.py \
    tests/test_wsv_state.py -q
```

---

## 8. 已知边界

1. `detector.py` 由 Lead 维护（`FrozenDetector`）；未提供时可执行文件或依赖（torch /
   transformers / 权重缓存）缺失时，`pairs` 套件会以明确错误退出，文本 suite 不受影响。
2. 当前只实现 T1（文本）与 pairs 结构化评测；事件级 P/R/F1 与世界状态回放指标不在本包内
   （见 `verifiers/event_metrics.py` / `verifiers/state_replay.py`），pairs 只报 change/relation
   五个维度。
3. `spatialmqa` 的 1,612 题是 T1 单项能力，不是综合排名；coverage < 100% 的行不可当作最终分。
