# starlife_eval — 架构与口径契约

本目录是 StarLife **唯一真后端评测系统**。它替代第三方 demo 仓库 `openwsv/openwsv_eval`
（该 demo 的后端全是 mock：假模型、编造分数、无 3D 数据跑 affordance/gating 判题），
只保留其前端（`third_party/openwsv/src`）。

> 本文件是**唯一口径来源**。所有 teammate 实现必须严格对齐本文；
> 指标实现以 `starlife/starlife_scripts/{train_heads,eval_compare_cloud,bench_pipeline}.py`
> 与 `starlife/starlife_doc/决策判别模型/{测试套件设计,训练结果_pilot_v2}.md` 为准，禁止自创指标。

---

## 1. 数据根目录布局（`--data-root`）

`--data-root` 指向一个镜像服务器工作区的目录；**不硬编码任何机器路径**。
在服务器上直接 `--data-root <data_root>`；在本地把产物同步到任意目录后指过去。

```
<data-root>/
  dataset_spatial_v2/          # 数据集目录（--dataset 指定相对名）
    manifest.jsonl             # 每行一个 pair（见 §2）
    meta.json                  # 数据集元信息
    frames/{pair_id}_before.jpg, {pair_id}_after.jpg
  features/
    spatial_v2.npz             # 缓存特征（见 §3）
    spatial_v2.meta.json       # 特征元信息（type_keys / rel_types / rel_pos_keys）
  checkpoints/
    pilot_v2g/{change,type,relation,rel_pos,attrs}.pt + metrics.json
  cloud/
    eval_cloud_raw.jsonl       # deepseek-flash 离线回放缓存
    eval_compare_v2g.json      # 历史对照组结果（只作交叉校验，不参与计算）
  bench/
    bench_pipeline.json        # 真实特征抽取 / 头推理延迟基准
```

`--data-root` + `--dataset/--features/--checkpoints`（均为相对路径）共同定位产物。
`EvalConfig`（§7）可序列化为 json，便于复现。

## 2. 数据集 manifest 契约（真实产物，勿改）

```jsonc
{"pair_id":"000001","split":"train","vocab_version":"1.0.0",
 "before":"frames/000001_before.jpg","after":"frames/000001_after.jpg",
 "entities":[{"slot":0,"bbox":[x0,y0,x1,y1],"type_key":"table","present_after":true}, ...],
 "changes":[{"slot":3,"type":"appeared"}, ...],          // 只列非 none
 "relations":[{"from_slot":2,"to_slot":3,"type":"near","rel_pos":"back",
               "contact":false,"support":false,"dist_m":0.336}, ...],
 "negative":false, "meta":{"lighting":"warm", ...}}
```

- `bbox` 是像素框（1024×768）；`entities` 是 before 帧的槽位表；
- `relations` 只描述 **after** 帧；`changes.type ∈ {none,appeared,displaced,disappeared}`；
- 切分按场景组，test 200 对；`split` 字段即权威切分。

## 3. 特征 npz 契约（真实产物，勿改）

`np.load(path, allow_pickle=False)`，键与 `extract_features.py` 一致：

| 键 | dtype | shape | 含义 |
|---|---|---|---|
| `slot_before`, `slot_after` | float16 | (N_slot, 5120) | 冻结主干池化特征 |
| `slot_mask` | int8 | (N_slot,) | after 帧是否存在 |
| `slot_present` | int8 | (N_slot,) | 同上（原始位） |
| `slot_change` | int8 | (N_slot,) | 0 none / 1 appeared / 2 displaced / 3 disappeared |
| `slot_type` | int32 | (N_slot,) | 索引 `meta.type_keys` |
| `slot_pair` | int32 | (N_slot,) | 所属 pair 在 manifest 中的行号 |
| `pair_split` | <U5 | (1000,) | 每个 pair 的 split |
| `rel_from`, `rel_to` | int32 | (N_rel,) | **全局 slot 下标** |
| `rel_type` | int32 | (N_rel,) | 索引 `meta.rel_types` |
| `rel_pos` | int32 | (N_rel,) | 索引 `meta.rel_pos_keys` |
| `rel_contact`, `rel_support` | int8 | (N_rel,) | 二值属性 |
| `rel_dist` | float32 | (N_rel,) | 米 |
| `img_before`, `img_after` | float16 | (N_pair, 5120) | 帧级特征（当前未用于头） |

`spatial_v2.meta.json` 关键字段：`type_keys`(22)、`rel_types`(11)、`rel_pos_keys`(10)、
`change_keys`(4)、`dim`(5120)。

## 4. 头结构与特征拼装（必须逐位对齐 `train_heads.py` / `eval_compare_cloud.py`）

`Head`：`LayerNorm(d_in) → Linear(d_in,512) → GELU → Dropout(0.2) → Linear(512,256) → GELU
→ Dropout(0.2) → Linear(256,n_out)`。推理 `.eval()`，dropout 失效。

权重文件 `{name}.pt` 是 `{"state_dict","n_classes","d_in"}`，`d_in` 是**唯一权威**的输入维度来源；
按 `d_in` 反推是否拼接 bbox 几何特征（v2 外观版 / v2g 几何版通用）。

特征拼装（`slot_after`/`slot_before` 取 float32）：

```
diff      = |slot_after - slot_before|
slot_feat = [slot_before, slot_after, diff, slot_mask[:,None]]          # d_in 15361
type_feat = [slot_after, diff]                                          # d_in 10240
pair_base = [after[frm], after[to], |after[frm]-after[to]|, rel_dist]   # d_in 15361
pair_feat = pair_base (+ 9 维 bbox 几何) 当且仅当 rel_din == base+9     # d_in 15370
onehot    = onehot(GT rel label, 12 类)   # label==none 时 index=11
rel_pos   = [pair_feat, onehot]                                         # d_in 15382
attrs     = pair_feat
change    = slot_feat
```

**bbox 几何 9 维**（`train_heads.pair_geometry`，img 1024×768）：
`[dxn, dyn, wa/wb, ha/hb, area_a/area_b, iou, dist_norm, sax/1024, say/768]`。
manifest 行 → 全局 slot 下标的偏移量：`offsets[i+1] = offsets[i] + len(rows[i].entities)`。

**关系负样本采样**（`train_heads.build_relation_data`，评估也必须照做，否则 macro-F1 无法复现）：
对每条真关系，用 `np.random.default_rng(20261020)` 在同 pair 内采 2 个未关联槽对作 `none` 负样本；
`rel_pos` / `attrs` / 事件级只统计真关系（非 none）。

**rel_pos 输入用 GT 关系类型 one-hot**（既有协议如此，属条件评测；报告须注明）。

## 5. 指标定义（写死，测试与文档同源）

### 5.1 头级（L2，全 test split，镜像 `train_heads.py`）
- `macro_f1_train`：逐类 F1，**跳过 `tp+fp+fn==0` 的类**，再等权平均（`nanmean` 语义）；
- `change`: top1 / macro-F1 / per-class F1（4 类）；
- `type`: top1 / macro-F1（22 类，仅统计 GT 类型在词表内）；
- `relation`: top1 / macro-F1（12 类：11 关系 + none，**含负样本**）；
- `rel_pos`: top1 / macro-F1（10 类，**只统计非 none 的真关系**）；
- `attrs`: `contact_f1` = macro-F1(二分类)，`support_f1` 同，`attrs` 汇总 = 两者均值。

### 5.2 60 对对照协议（镜像 `eval_compare_cloud.py`，云端可比）
- 抽样：`test_idx` 上 `random.Random(20261020).sample(...,n=60)` 后排序；
- 关系按 `(from_slot,to_slot)` 匹配；`change`: top1 / `macro_f1_compare`；
- `macro_f1_compare`：逐类 F1，**跳过 `tp==0` 的类**（与 5.1 不同，云端 change macro-F1 0.7781 依赖此口径）；
- `relation_type_acc`：只在 GT 关系对上的类型命中率（无负样本）；
- `rel_pos_acc`：GT 关系对上且 GT 有 rel_pos 的命中率；
- `attrs_acc`：GT 关系对上 contact/support 各算 1 项的准确率。

### 5.3 事件级（L3，pair 级投影）
把预测翻译成 `starlife_store/Core/WorldEvent.cs` 四类事件的子集：
- `StateChanged{entity=slot, change_type}`：`change != none` 时；
- `RelationAdded{from,to,relation_type}`：每条预测关系；
- `EntityCreated{entity=slot,type}`：`appeared` 且预测/GT 均有类型时（type 维度仅本地头有）。
匹配单位 `(event_kind, from, to, label)`，逐类 P/R/F1 + macro。
**v2 数据无 before 关系标注 → `RelationRemoved` 与实体 ID 连续性（`id_switch_rate`）不可测，标 N/A。**
L1 契约（4 类事件 schema / 时间戳单调 / ID 分配复用）用构造张量单测，不需要真数据。

### 5.4 世界状态回放（L4）
预测事件回放进 `WorldStore`（内存最小实现，`starlife_eval/verifiers/state_replay.py`），
快照与 GT after 场景比较，五个维度：`objects`（存在集合）、`types`（仅本地头）、
`changes`、`relations`（集合 + 类型）、`attrs`（rel_pos/contact/support）。
`state_acc` = 可用维度等权平均（`测试套件设计.md §3`）；不可用维度进 `na` 并写明原因。

### 5.5 运行时
- 延迟：本地头推理实测（per pair，`runtime.latency_ms_mean`），**但前端展示必须加上特征抽取**：
  `pipeline_latency_ms_mean = 1000 × bench.event_latency_s + 头前向 ms`（否则 2ms 不可比）；
  特征抽取用 `bench/bench_pipeline.json` 的实测值**引用**，不重跑渲染；
- 云端延迟：回放缓存里的 `latency_s`（真实测量值）；
- 成本：本地 `cost_usd = 0` + `cost_basis`（自有硬件）；云端无公开计费口径 → `null`。
  成本与 token 只记在 `report.json` 的 `runtime`（运维参考），**不进榜单契约、不计分**。

### 5.6 前端 `score` 的映射（唯一允许的聚合，须在报告注明出处）
- `score` = `100 × event.macro_f1_common`，`macro_f1_common` = **StateChanged 与 RelationAdded
  两类 F1 的等权平均**。这两类所有模型都能产出，横向可比；`EntityCreated` 需要实体类型标注，
  云端零样本不返回类型（N/A），若并入 macro 会让本地模型因多测一个更难维度而被系统性压低，
  故 `EntityCreated` 单独展示、不计入 score。全类 macro（本地含 EntityCreated）另存 `event.macro_f1` 备查；
- `topologyF1` = `100 × pair_protocol.relation_type_acc`；
- `affordanceAcc` = `100 × pair_protocol.attrs_acc`；
- `stateTransitionAcc` = `100 × pair_protocol.change_top1`；
- `latency` = 横向可比延迟 ms：本地头 = `bench.event_latency_s`（特征抽取实测）+ 头前向实测；
  云端 = 缓存 `latency_s` 实测均值；
- 行数据只带 `latency`；`cost`/`token` 不入契约（schema v4 起删除）。
测不了的维度一律 `null` + `na` 说明，**禁止填假值/占位值**。

### 5.7 综合能力分（`composite`，v1.2 起为榜单主分）

`score` 是"可比事件分"，只覆盖两类事件；用户需要一个总分，但**总分必然依赖权重**。
因此 `report/composite.py` 定死如下口径，产物里同时给出逐维分数与权重，谁都能重算：

- **取维度**：只取 6 个模型 **100% 可测**的 7 个维度（避免用 N/A 维度给模型白送/白扣分）：
  | key | 来源 | 说明 |
  |---|---|---|
  | `change_top1` | `pair_protocol.change_top1` | 变化识别 |
  | `change_balance` | `pair_protocol.change_macro_f1` | 变化平衡性（compare 口径） |
  | `relation_type` | `pair_protocol.relation_type_acc` | 关系/拓扑 |
  | `rel_pos` | `pair_protocol.rel_pos_acc` | 空间方位 |
  | `attrs` | `pair_protocol.attrs_acc` | 接触/支撑 |
  | `event_f1` | `event_metrics.macro_f1_common` | 事件级 F1 |
  | `state_consistency` | `state_replay` 的 objects/changes/relations/attrs 四维均值 | 世界状态一致性 |
- **权重**：等权 `1/n`（`n` = 可用维度数），写进 `composite.weights`。刻意的选择：
  不预设"关系比变化更重要"，避免拍脑袋权重制造排名；需要其他权重的人拿 `dims` 自己加权即可。
- **缺值**：某维 `None` 时**从分母剔除**并记入 `composite.missing`，**不当 0 分**。
- **`state_consistency` 不含 `types`**：类型只有本地模型可测，混入会让云端模型被扣一个它没有的维度。
- `compositeScore` = `100 × mean(可用维度)`；榜单排序与 `isBest` 均以它为准，
  旧 `score/topologyF1/affordanceAcc/stateTransitionAcc` 全部保留用于对照（v1.0/v1.1 口径不丢）。

### 5.8 本评测在世界模型能力栈中的位置（对外叙事必须与之一致）

世界模型六层：① 感知编码 → ② 状态估计 → ③ 变化识别 → ④ 关系与拓扑 → ⑤ 事件化 → ⑥ 世界状态一致性。
本评测：**①作为给定条件（冻结 Qwen3.8-27B 编码器，不评分）**，测 **②～⑥**，
输出的不是像素/视频，而是**可被机器人或仿真消费的结构化世界状态更新**（事件流 + 回放一致性）。

## 6. 预测数据模型（`starlife_eval/types.py`）

```python
@dataclass(frozen=True)
class HeadPrediction:                 # 与 npz 行对齐，仅供本地头
    change: np.ndarray   # (N_slot,) int64
    type: np.ndarray     # (N_slot,) int64
    relation: np.ndarray # (N_rel_plus_neg,) int64  含负样本
    rel_pos: np.ndarray  # (N_rel_plus_neg,) int64
    attrs: np.ndarray    # (N_rel_plus_neg, 2) int64

@dataclass
class PairPrediction:                 # 逐 pair、逐 slot_id，云端/事件/状态回放用
    pair_index: int
    changes: dict[int, str]
    types: dict[int, str]
    relations: list[RelationPrediction]   # from_slot/to_slot 用 pair 内 slot 编号

@dataclass
class RelationPrediction:
    from_slot: int; to_slot: int
    type: str; rel_pos: str | None
    contact: bool | None; support: bool | None
```

适配器只负责 `predict(...)`：
- `local_heads.LocalHeadsAdapter(ckpt_dir, features, dataset)` → `HeadPrediction`；
- `replay_cloud.ReplayCloudAdapter(raw_jsonl, ...)` → `PairPrediction`；
- `openai_compat.ChatVLMAdapter(base_url, model, api_key_env)` → `PairPrediction`（`--live` 才调用，key 只读环境变量）；
- 接不了的基线 → 注册表里标 `unavailable`，报告 `na: "无可用 key/缓存"`，**不许编数**。

## 7. `EvalConfig` 与 CLI

```
python -m starlife_eval run --data-root <dir> --config configs/full_v14.json --out runs/v14_full
python -m starlife_eval leaderboard --run runs/v14_full --out <openwsv>/src/data/leaderboard.generated.json
```

`configs/full_v14.json`（v1.4，11 个模型）要点：
```json
{"dataset":"dataset_spatial_v2","features":"features/spatial_v2.npz",
 "checkpoints":"checkpoints/pilot_v2","bench":"bench/bench_pipeline.json",
 "cloud_cache":"cloud/eval_cloud_raw.jsonl","split":"test","n_pairs":60,"sample_seed":20261020,
 "models":[
  {"id":"starlife-v1-wsv","kind":"local_heads","mode":"LOCAL-HEADS","source":"checkpoints/pilot_v2"},
  {"id":"deepseek-flash","kind":"replay_cloud","mode":"ZERO-SHOT"},
  {"id":"gpt-6-astra","kind":"chat_vlm","base_url":"https://api.apigoto.com/v1",
   "model":"gpt-6-astra","api_key_env":"STARLIFE_EVAL_APIGOTO_KEY",
   "max_tokens":6000,"tolerate_call_errors":true,"retries":3,"timeout_s":180},
  {"id":"claude-opus-5-5","kind":"chat_vlm","base_url":"https://www.yydsapi.uno/v1",
   "api_key_env":"STARLIFE_EVAL_YYDS_KEY",
   "headers":{"User-Agent":"Mozilla/5.0 ..."}},
  {"id":"glm-5.3-flash","kind":"chat_vlm","base_url":"https://api.apigoto.com/v1",
   "model":"glm-5.3-flash","max_tokens":6000,"tolerate_call_errors":true}
 ]}
```

- `chat_vlm` 的可选 extra：`max_tokens`、`headers`（覆盖/追加请求头）、
  `tolerate_call_errors`（HTTP/网络彻底失败的 pair 记空预测并继续，不中断整场）、
  `retries` / `retry_sleep_s` / `timeout_s`；
- 缓存与解析相关 extra：
  - `cache_path`：覆盖 live 缓存位置（默认 `<data-root>/cloud/live_<id>.jsonl`）；
    由 `registry.resolve_cache_path()` 统一解析（`extra.cache_path` > `spec.source` > 默认），
    `runner/executor.py` 也走同一函数，保证「建适配器」和「执行」用同一份缓存；
  - `refetch_empty`（该 pair 缓存正文为空 → 重新请求）、`refetch_unparsable`（缓存 JSON 解析失败 → 重新请求）、
    `refetch_invalid`（缓存 JSON 能解析但标签是 prompt 回显/非法值 → 重新请求）。
    这三个开关只影响 live 命中路径，不会改写已有缓存行，新行按 `pair_index` 覆盖旧行；
  - 推理型模型（输出 `<think>…</think>`）在解析前统一 `_strip_think()`：只取 `</think>` 之后的正文，
    无闭合标签视为思考被截断（空预测，计入分母、不剔除）；
- `kind:"unavailable"` 用于"这次没跑成但要在榜上说明原因"的模型：报告原样输出 `N/A` + `note` 文本；

## 8. 报告

`report/report.json`：`{"meta","models":{id:{head_metrics,pair_protocol,event,state,runtime,na}},"leaderboard":[...]}`
`report/report.md`：人读版，含 `na` 说明、口径出处、产物指纹（npz/ckpt 的 sha256 前缀）。

辅助产物（不进榜单契约）：`reports/v1.4/confidence_intervals.json` —— 每个指标的 95% bootstrap 区间
（60 对 pair 有放回重采 1000 次，逐 pair 充分统计量按重数累加后重算 7 维与综合分），
生成脚本 `tools/confidence_intervals.py`；全样本对拍官方 report（容差 0.2，覆盖 0.1 舍入）。

辅助产物（不进榜单契约）：`reports/v1.4/difficulty_curve.json` —— 按「场景复杂度」分档后，
在每档子集上重算 7 维综合分，供前端折线图使用；生成脚本 `tools/difficulty_curve.py`
（离线、只读缓存）。它的 `composite` 与榜单同定义但**限定子集**，文档里必须标注为诊断视图。

前端契约（`third_party/openwsv/src/data/benchmarkData.ts` 的 `ModelResult`）：
由 `report/leaderboard.py` 生成 `leaderboard.generated.json`，字段见 §5.6；`null` 允许，
前端必须把 `null` 渲染为 `N/A` 而不是 `0`。

## 9. 目录职责

| 目录 | 负责 |
|---|---|
| `src/starlife_eval/types.py`, `config.py` | Lead（共享契约，勿改签名） |
| `dataset/` | 数据/特征加载、几何特征、关系负采样 |
| `models/` | 头结构、本地 checkpoint 适配器、云端回放/实时适配器 |
| `verifiers/` | §5 全部指标 + 单测 |
| `translator.py` | 预测 → 事件流（L1/L3） |
| `runner/`, `report/`, `cli.py` | 编排、报告、榜单生成 |
| `tests/` | pytest，**必须**用已知输入断言数值 |

## 10. 验收数字（服务器真实产物，容差 ±0.001）

| 模型 | change top1/macroF1 | type top1 | relation top1 | rel_pos top1 | attrs |
|---|---|---|---|---|---|
| v2g test(200 对) | 0.9270 / 0.6983 | 0.9365 | 0.7745 | 0.7042 | 0.8895 |
| v2 test(200 对) | 0.9365 / 0.7232 | 0.9451 | 0.7934 | 0.6878 | — |
| deepseek-flash 60 对 | 0.9251 / 0.7781 | — | 0.2784 | 0.4898 | 0.9286 |

v1.4 全部 11 个模型的综合分/逐维数字见 `README.md` §10 与 `reports/v1.4/report.json`；
本轮 `acceptance_v14.json` 只固定 `starlife-v1-wsv`（沿用 v2 的 60 对数字）与 `deepseek-flash` 两组，
**16/16 通过**（`reports/v1.4/acceptance.log`，16/16 通过）。云端 10 个模型没有写死期望值的原因：
它们是真实 API 回放，缓存一旦补齐数字即固定，但补缓存顺序受外部网关影响，不适合做回归基线。

本地头在 60 对协议上应得：change 0.9135/0.6409、relation_type_acc 0.9034、rel_pos_acc 0.7736、attrs_acc 0.9717；
云端延迟均值 6.41 s（真实记录值）。
