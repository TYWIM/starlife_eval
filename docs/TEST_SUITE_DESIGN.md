# 观测模块测试套件设计（只设计，先不实现）

## 0. 定位

这个模块是世界模型的**视觉观测前端**：帧 → 结构化事件流（`EntityCreated / StateChanged / RelationAdded / RelationRemoved`），世界状态由下游 `WorldStore` 维护。所以测试不能只看"分类准确率"，要回答四个问题：

1. **观测准不准**：每个判别头在留出集上的头级指标；
2. **事件对不对**：把预测翻译成事件后，与 GT 事件流的 P/R/F1；
3. **世界状态一致不一致**：事件回放后的世界状态是否等于 GT 场景状态（这是世界模型真正关心的）；
4. **稳不稳、快不快**：跨帧 ID 连续、鲁棒性、延迟。

## 1. 分层测试（L1–L7）

| 层 | 内容 | 输入 | 通过标准 |
|---|---|---|---|
| L1 契约/单元 | `ObservationTranslator` 输出 schema：4 种事件类型、字段完整、时间戳 Unix ms 单调、`entity_id` 分配与复用、同帧去重、`RelationRemoved` 只在断开时出现 | 构造的 slot/relation 张量 | 100% 用例通过（纯离线，无模型） |
| L2 头级指标 | change(4 类) / type / relation / rel_pos / attrs 的 top-1、macro-F1、混淆矩阵；OOV 类型走文本检索时看 top-k | 冻结的 test split 特征 | 见 §4 门槛 |
| L3 事件级 | 预测事件流 vs GT 事件流：按 `(pair_id, slot)` 匹配算 P/R/F1；每类事件单独报 | 合成序列（连续帧） | event F1 ≥ 0.80 |
| L4 世界状态 | 事件回放后 `WorldStore` 快照 vs GT 场景：对象集合、类型、属性、关系、位置关系的一致率 | 场景 GT（生成器自带） | 状态一致率 ≥ 0.90 |
| L5 鲁棒性 | 光照/色温/曝光/模糊/遮挡率/JPEG 质量/分辨率(448 vs 768 vs 1024) 单因素扫描，画退化曲线 | 同 test 集重渲染 | 任一单因素下 top-1 降幅 ≤ 5pp |
| L6 性能 | 单帧、4 帧批、事件（2 帧 + 12 crop）、端到端（含 head）延迟与吞吐；显存峰值 | A800 80G + 目标部署卡 | 见 §4 |
| L7 云端对照 | 同 test 集上 `deepseek-flash` vs 本地：指标 + 延迟 + token 成本 | 复用 `eval_compare_cloud.py` | 不对齐也接受，只做记录 |

## 2. 数据切分与防泄漏

- **按场景组切分**（现有 20 组 → 12/4/4），同一场景的所有帧只出现在一个 split；
- 合成器随机种子固定，test 集**冻结版本号**（如 `test_spatial_v3`），后续改动只能新增版本；
- 黄金集：从 test 里抽 100–200 个场景，人工/bug 修复后不再变动，作为回归基线；
- OOV 测试：从词表里留出 50 个 A/B 类不参与训练，只用于 type 检索评测（零样本）。

## 3. 指标定义（避免各说各话）

- `change_top1` / `change_macroF1`：4 类（none/appeared/displaced/disappeared）逐 slot 分类；macro-F1 对 4 类等权平均，**少数类（displaced）权重不因样本少而变小**；
- `type_acc`：仅统计 GT 类型在词表内的 slot；OOV 场景另报 `type_top5_recall`；
- `relation_P/R/F1`：以 `(pair, relation_type)` 为匹配单位，不在 GT 里的 pair 不参与；11 类分别报 + macro；
- `rel_pos_acc`：GT 有 rel_pos 且关系存在的 pair；
- `attrs_F1`：contact/support 两个二分类；
- `id_switch_rate`：相邻帧同一物体被分配不同 entity_id 的比例；
- `event_F1`：把 GT 与预测都翻译成事件后匹配（type + 实体 + 关系），三类事件分别 P/R/F1；
- `state_acc`：回放后世界状态与 GT 的逐项一致率（对象/类型/属性/关系/位置关系，加权平均）。

## 4. 门槛（v3 目标）

| 指标 | v2 现状 | v3 目标 |
|---|---|---|
| change top-1 | 0.936 | ≥ 0.97 |
| change macro-F1 | 0.641（云 0.778） | ≥ 0.85 |
| type A 闭集 top-1 | 0.945 | ≥ 0.96 |
| type 开放词表 top-5 | 未测（OOV 检索 0.68 左右） | ≥ 0.85（A/B 留出 50 类） |
| relation macro-F1 | 0.77 | ≥ 0.84 |
| rel_pos acc（10 类） | 0.70 | ≥ 0.85（合并 back_left/right 为 back 后） |
| attrs F1 | 0.87 | ≥ 0.93 |
| event F1 / state_acc | 未测 | ≥ 0.80 / ≥ 0.90 |
| 事件延迟（2 帧 + 12 crop，A800） | 1.10–1.72 s | ≤ 0.8 s（CUDA Graph/编译后） |
| 单帧吞吐 | 2.4 fps（0.41 s/帧） | ≥ 4 fps |

## 5. 工程形态（后续实现时）

- 入口 `starlife_scripts/eval_suite.py`：`--dataset`（特征 npz + 图像）、`--checkpoints`、`--out`；输出 `report.json` + `report.md`；
- 每次训练自动跑 L2，nightly 跑 L3/L4/L5（渲染成本高）；
- CI gate：只对黄金集跑 L2，指标低于门槛则失败；
- 所有指标定义写死在 `metrics.py` 里，测试和文档同源。

## 6. 现在先不做的原因

渲染和特征抽取要占满当前 8 核配额（CPU 受限），套件实现依赖 v3 数据定型；先把数据与模型短板补上，套件按本文档一次实现到位，避免边写边改指标定义。
