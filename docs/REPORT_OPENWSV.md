# OpenWSV（WSV）——世界状态视觉评测榜单 新版测试报告

> 版本：v3.2（2026-10-25）。相对 v3.1 的修改：数据版本扩展到 `openwsv-data-1.0`（冻结）+ `openwsv-data-1.1`（增量合成），实际发布 7 套 suite（T1/T4/T5/T6/T7）；框架版本仍为 `openwsv-1`，检测器/提示词/判分/种子不变；T2/T3 未实现、T8 未构建；§10 结果待最终跑分回填。v3.1 起只保留"固定框架"一种评测设置，删除 L0/L1/L2 分档与"平台检测"设想，删除 GT 框诊断档。
> 定位：按 SpatialMQA / MindEdit / Spatial457 / Spatial-DISE 等方法论思路，自建能同时评测「文本多模态模型」与「结构化世界模型（非视频生成）」的空间理解横测榜单。
> 结论先行：现有公开空间基准只收文本答案、只给单帧、没有状态协议，无法公平评测结构化世界模型；OpenWSV 用一套**冻结的测试框架**（固定数据、固定检测器、固定提示词、固定判分）让两类模型在同一批题、同一份 GT 上出分。

---

## 1. 为什么要另做一套

过去两周的目标原是"复跑 SpatialMQA / MindEdit / Spatial457"，跑下来的经验是：

1. **这些基准只认文本答案**。我们的模型（StarLife-1-WSV）输出结构化世界状态（slots / pairs / events），不生成自然语言；硬接只能写"事件→选项"适配器，测出来的是适配器，不是模型。
2. **它们各自只覆盖一个切面**：SpatialMQA 单图关系（5,392 题，COCO2017 人工标注）、MindEdit 三视角相机运动/跨视角关系（1,003 题，6 类任务）、Spatial457 6D 位姿诊断（7 题型 × 5 难度，合成）、Spatial-DISE 559 题评测集 + 12K 训练集、PureSpace 抽象几何。**没有一个同时覆盖"空间理解 + 物品关系 + 空间变化 + 关系变化"**。
3. **没有统一框架**：数据集、提示词、解析、判分各写各的，同一模型换个脚本分数就变；空输出、超时、解析失败被当答错。
4. **没有统计纪律**：只报点值，不报覆盖率、无置信区间、不分"答错"与"没答"。

v3 的目标：**同一批题、同一份 GT、同一个冻结框架、可复现的统计**。

---

## 2. 现有基准调研

### 2.1 空间理解 / 关系类

| 基准 | 规模与来源 | 答题形式 | 覆盖能力 | 不够用的地方 |
|---|---|---|---|---|
| [SpatialMQA](https://github.com/ziyan-xiaoyu/spatialmqa) | 5,392 题，COCO2017 人工标注 | 2/4/6 选项 | 单图空间关系（left/right/front/behind/on-above/below） | 单帧、仅关系、无变化；结构化模型无从作答 |
| [Spatial457](https://openaccess.thecvf.com//content/CVPR2025/html/Wang_Spatial457_A_Diagnostic_Benchmark_for_6D_Spatial_Reasoning_of_Large_CVPR_2025_paper.html) | 合成，7 题型 × 5 难度，级联评测 | 多选 | 多物体识别、2D/3D 位置、3D 朝向 | 分布窄；答案文本化，状态模型难对齐 |
| [Spatial-DISE](https://ar5iv.labs.arxiv.org/html/2510.13394) | 559 题评测集 + 12K 训练集 | VQA | 内在/动态空间推理，10 类任务 | 文本答案；训练集与评测集同源，污染风险 |
| [PureSpace](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_PureSpace_A_Benchmark_for_Abstract_Spatial_Reasoning_in_Vision-Language_Models_CVPRF_2026_paper.html) | 抽象几何物体 | VQA | 旋转、投影、补全 | 与真实世界状态脱节；不可用于结构化模型 |
| [OmniSpatial](https://ar5iv.labs.arxiv.org/html/2506.03135) | 4 大类 50 子类，人工标注 | 多选/开放 | 动态推理、复杂空间逻辑、空间交互、视角转换 | 全文本；无状态协议；无变化成对数据 |
| [SpinBench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) | 认知科学驱动诊断集 | 多选 | 视角转换、旋转下的关系变化 | 只问答，不产出状态 |

### 2.2 跨视角 / 时序变化类

| 基准 | 形态 | 覆盖能力 | 缺口 |
|---|---|---|---|
| [MindEdit-Bench](https://huggingface.co/datasets/ZODAOfficial/MindEdit-Bench) | 1,003 题室内三视角多选，6 类任务（相机运动、跨视角物体关系、透视变换、空间编辑、跨视角可见性） | 视角间关系与"编辑后"关系 | 不产出结构化变化描述；无真值状态 |
| [VSI-Bench](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.html) | 5,000+ 视频问答 | 视频空间记忆与推理 | 视频输入；无逐帧状态；答案文本 |
| [Disjoint-3DQA](https://ar5iv.labs.arxiv.org/html/2505.24257) | 5,399 题 / 1,668 场景 / 856 物体对 | 不共帧的跨帧物体关系整合 | 只问答；无关系变化类型标注 |
| [REM](https://github.com/jbt-cmu/REM) | 可控 3D 环境长时序具身推理 | 多帧轨迹推理 | 偏具身决策；不评关系变化本身 |

### 2.3 变化检测 / 世界模型评测

| 方向 | 代表 | 可借用之处 |
|---|---|---|
| 图像变化描述 | CLEVR-Change、Spot-the-Diff | 提供"前后帧 + 变化描述"构造方式；但评测是 caption 相似度，不可确定性判分 |
| 关系变化检测 | SSPRCD（街景场景图差分，关系级 F1 0.89） | 证明"关系级变化"可被显式评测，即 T5 的形态 |
| 世界模型闭环评测 | [World-In-World](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html)、[ScratchWorld](https://arxiv-org.ezproxy.obspm.fr/html/2606.31689v1) | 强调"可控性 > 画质"，与本榜"非视频生成、评状态"一致；但它们是行动条件化闭环，不评空间关系变化 |
| 3D 场景问答 | ScanQA / SQA3D | 3D 理解成熟，但需点云/网格输入，与"图像到世界状态"的接口不同 |

### 2.4 调研结论（缺口清单）

1. **没有"同一 GT、两种答案形态"的协议**：文本模型答选项，状态模型出结构化状态，两者要能对上同一份真值。
2. **没有"空间变化 + 关系变化"的联合评测**：变化类基准评像素/描述，关系类基准评静态关系，交界处空白。
3. **没有冻结的测试框架**：数据、检测、提示词、判分随脚本漂移，分数不可复现。
4. **没有覆盖率与置信区间纪律**：空输出、超时、解析失败被当作答错。
5. **污染物无标注**：训练集与评测集同源、同模板改写。

---

## 3. 评测对象：两类模型

### 3.1 A 类：文本多模态模型（VLM）
输入图像（单帧/多帧）+ 问题 + 选项/指令，输出自然语言或选项字母。已测：qwen3.8-flash、deepseek-flash、glm-5.3-flash、minimax-m3.1-flash、mimo-v2.6-pro。

### 3.2 B 类：结构化世界模型（非视频生成）
输入一帧或前后帧 + **检测框（由测试框架统一提供，见 §7）**，输出结构化世界状态：物体槽位、类型、存在性、变化类别、物体对关系、方位、接触/支撑、事件流。代表：StarLife-1-WSV-Lite（冻结 Qwen3.8-27B 主干 + 5 个判别头）及同类的状态预测/场景图/世界模型工作。

> 明确排除：视频生成类世界模型（Sora 类）。其评测是视觉保真度与可控性（见 World-In-World），另立子榜；本榜只评"从观测到世界状态"的能力。

---

## 4. 设计原则

1. **单一设置**：所有模型在一个**冻结的测试框架**里评测——同一份数据版本、同一套提示词、同一个检测器（供需要框的模型使用）、同一份判分器、同一组随机种子。不做多档排名，不做"某模型自选输入"。
2. **双协议、同 GT**：每题有文本答案（choice / short / json）与结构化答案（state / relation set）两种合法形态，判分器归一到同一真值，准确率定义一致。
3. **覆盖率优先**：准确率只在"有有效输出"的题上算，覆盖率单独报告；空响应/超时/解析失败**不是答错**，而是未覆盖。
4. **确定性判分**：选择/短答规范化精确匹配；状态用集合/图匹配；事件用带类型 P/R/F1；判分器开源。
5. **统计纪律**：95% bootstrap CI（按题重采样，1,000 次，固定种子）；成对比较用 McNemar。
6. **抗污染**：合成题公开生成器 + 私有 holdout + canary；真实题保留 15% 私有集。
7. **表述克制**：只用"高/低、显著/不显著"，不写 SOTA、榜首、领先 X 个点。

---

## 5. 能力矩阵（T1–T8，openwsv-1 实际发布）

| 编号 | 能力 | 任务形态 | 判分指标 | 数据来源 / 当前状态 |
|---|---|---|---|---|
| T1 | 单图空间关系 | 单图 "X 在 Y 的哪一侧" 多选 | 准确率 + 覆盖率 + bootstrap CI | `spatialmqa` 1,612 题（test 1,076 + validation 536）；`spatialmqa_relation` 305 题为同一冻结文件的关系题子集 |
| T2 | 6D 位姿/朝向 | — | — | **未实现**（不在 `openwsv-1` 包内，unreleased） |
| T3 | 多视角/视角变换 | — | — | **未实现**（不在 `openwsv-1` 包内，unreleased） |
| T4 | 空间变化 | 前后帧单物体变化 4 选（appeared / disappeared / displaced / stayed same） | top-1 准确率、4 类 macro-F1 | `change` 1,000 题，由冻结 `pairs` GT 确定性投影，**不新增图像** |
| T5 | 关系变化 | 前后帧 A–B 关系变化 4 选（appeared / disappeared / changed / stayed same） | 准确率（4 类均衡，各 150） | `relation_change` 600 题，合成数据包 `openwsv-data-1.1` |
| T6 | 状态回放 | 后帧状态事实 4 选（existence / relation_type / rel_pos / contact，各 250） | 准确率（按探针分项） | `state_replay` 1,000 题，由冻结 `pairs` 的 after-state GT 确定性投影，**不新增图像** |
| T7 | 跨帧不相交推理 | 两帧物体集合不交，问同一相机下 A 相对 B 的方位 4 选（left / right / above / below） | 准确率（4 类均衡，各 100） | `cross_frame` 400 题，合成数据包 `openwsv-data-1.1` |
| T8 | 抽象几何 | 旋转 / 投影 / 补全 | 准确率 | **未构建**（本轮不做） |

说明：

- 实际发布 7 套 suite：`spatialmqa`、`spatialmqa_relation`、`pairs`、`change`、`relation_change`、`state_replay`、`cross_frame`。其中 `pairs`（1,000 对 before/after 帧 + 结构化 GT）是 T4–T6 的共同底座。
- `spatialmqa_relation` 与 `spatialmqa` 共用同一冻结文件，加载时按固定正则筛选（`Where is/are the X located relative to the Y?`），不重复计题。
- T2/T3 仅保留能力定义，本版本未实现，标注为 unreleased；T8 尚无数据集与套件。
- 文本模型统一按 `choice` 协议作答；B 类结构化模型在 T1（关系）、T4（变化）、T5（关系变化）、T6（回放）为主项参与，T7 可用"前后帧 + 框架框"近似参与并在结果行标注；T2/T3/T8 本版本无对应套件。

---

## 6. 数据构建（增量数据模型）

### 6.1 只增不改：1.0 冻结 + 1.1 增量

- `openwsv-data-1.0`（冻结）：已发布数据包原样保留，不修改、不重算；内含 SpatialMQA 题面图像、`pairs` 的 2,000 张 before/after 帧（1,000 对 test 划分）、1.0 版 suites 与 `SHA256SUMS`。
- 新增内容全部为**追加**：
  - T4 `change` 与 T6 `state_replay` 是同一份冻结 `pairs` GT 的确定性文本投影，**不新增图像**（复用 1.0 包内帧）；
  - T5 `relation_change` 与 T7 `cross_frame` 使用独立的合成数据包 `openwsv-data-1.1`，不改动 1.0 的任何文件。
- 评测数据根同时包含两个包（`--data-root data`）；新增套件的图像路径带包前缀 `openwsv-data-1.0/...`、`openwsv-data-1.1/...`，相对数据根稳定；旧套件 `spatialmqa`/`pairs` 保持原有相对路径，可单独用 `OPENWSV_DATA_ROOT=./openwsv-data-1.0` 跑。
- 每行 `meta` 保存结构化真值（实体 / 2D bbox / 变化类别 / 关系 / 方位 / 接触支撑 / 合成 GT），只用于审计与结构化判分，**绝不进入提示词**；模型输入只有 `question` / `images` / `choices`。

### 6.2 T4/T6：冻结 pairs GT 的确定性投影

- 脚本 `scripts/wsv/derive_text_suites.py`，输入只有冻结的 `pairs_test.jsonl`（1,000 对），输出 `change_test.jsonl`（1,000 题）与 `state_replay_test.jsonl`（1,000 题），一题一对、与 `pairs` 严格 1:1。
- T4 每题问某物体前后帧的变化（appeared / disappeared / displaced / stayed same），选项按 seed 20261020 确定性轮换，答案下标均衡。
- T6 每题问一个后帧状态事实，四类探针各 250 题：存在性（existence）、关系类型（relation_type）、相对位置（rel_pos）、接触（contact）；答案是 after-state 真值。
- 产物 sha256：`change` = `7012ac77153c7321209c882b451a8a0f1771240e3979f5c18077a4eecbc3e302`，`state_replay` = `3bd06b52e7c71d0756fedd962c1d895c2516fbf1cae7f29b4e263e4e51962bb2`；`python scripts/wsv/derive_text_suites.py --check` 重算并与现有文件逐字节比对，必须零漂移。

### 6.3 T5/T7：确定性合成数据包 `openwsv-data-1.1`

- 生成器 `openwsv-scene-1.0.0`（seed `20261020`）：纯 Python + numpy + Pillow 的解析光线投射，固定桌上场景与固定针孔相机，无外部素材、无网络、无文本绘制。
- 共 2,000 帧 1,024×768 JPEG（质量 90、4:4:4）：T5 600 对 = 1,200 帧（before/after），T7 400 对 = 800 帧（frame1/frame2）；每个任务的两帧共享同一相机与同一房间。
- T5 `relation_change`：4 类均衡各 150（appeared / disappeared / changed / stayed same）。A–B 关系签名 = 图像平面方位（left/right/above/below）+ 投影中心距桶（close/mid/far），类别由前后帧结构化状态差分产生，不依赖文件名。
- T7 `cross_frame`：两帧物体集合完全不交（没有物体同时出现在两帧），每帧恰好 3 个物体；4 个方向各 100（left/right/above/below）。GT 由两物体投影 bbox 中心的**主方向**判定，要求主方向 ≥1.5× 次方向、主轴间距 ≥8% 画幅（宽或高）。
- 确定性：生成器逐字节可复现（固定 JPEG 参数、无时间戳/绝对路径）；`python -m scripts.wsv.gen.build --force --determinism-check` 会用两个独立进程各渲染前 5 个任务、逐文件比对 sha256（套件 jsonl 也参与对比），随后写出 `SHA256SUMS`/`meta.json`/`README.md`。数据包约 79 MB，**不进 git**，按上述命令在本地再生成；套件 sha256：`relation_change` = `0166b005bae55a5473a30a65c1398e84bae2a7c34631f1e2b62ec194475bba7f`，`cross_frame` = `d11e0ff5d72fea7fbe948edb8f567c470b4d567bdc3e6fd2ff102792ec9dfb91`（2026-10-25 只把 T7 题面措辞明确为 "primary image-plane direction"，未重渲染任何图像，2,000 帧与 T5 套件字节不变）。

### 6.4 当前发布规模（openwsv-1，7 套 suite）

| suite | 能力 | 题量 | 来源 | 新增图像 |
|---|---|---:|---|---|
| `spatialmqa` | T1 | 1,612 | SpatialMQA（test 1,076 + validation 536） | 否（1.0 包内） |
| `spatialmqa_relation` | T1-relation | 305 | 同一文件按固定正则筛出 | 否 |
| `pairs` | T4/T5/T6 底座 | 1,000 对 | 冻结结构化 GT（`dataset_spatial_v3` test） | 否（1.0 包内） |
| `change` | T4 | 1,000 | `pairs` GT 确定性投影 | 否 |
| `relation_change` | T5 | 600 | `openwsv-scene-1.0.0` 合成（1.1） | 1,200 帧 |
| `state_replay` | T6 | 1,000 | `pairs` after-state GT 确定性投影 | 否 |
| `cross_frame` | T7 | 400 | `openwsv-scene-1.0.0` 合成（1.1） | 800 帧 |

> 受控变异（视角/遮挡/光照/干扰物/镜像对称陷阱）与真实采集、私有 holdout 属于后续稳健性评测内容，本版本不包含。

---

## 7. 冻结的测试框架（唯一评测设置）

**框架由五件冻结物组成，全部版本化并写进规范；换任何一件即新框架版本，全榜重跑。**

| 组件 | 冻结内容 | 说明 |
|---|---|---|
| 数据 | 数据集版本号 + 文件清单 sha256 | 同一批题、同一批 GT |
| 检测器 | 权重文件 sha256 + 版本 + 预处理 + 置信度/NMS 阈值 | **框架内置，所有需要框的模型共用同一份输出** |
| 提示词 | 提示模板文本 + 选择/短答/JSON 的输出规范 | 所有模型同一提示 |
| 判分器 | 判分代码 commit + 规范化规则 + 事件匹配规则 | 确定性、可复现 |
| 随机性 | 采样种子 + bootstrap 种子 | 固定 |

> 数据版本：本版本评测数据由 `openwsv-data-1.0`（冻结，原样保留）与 `openwsv-data-1.1`（增量，T5/T7 合成）共同组成；框架版本仍为 `openwsv-1`——只新增套件，检测器、提示词、判分规则与随机种子均未改动，因此 v3.1 已发布的 T1 分数可与本版本同表比较。数据指纹由 `config.dataset_fingerprint()` 逐个 suite 文件 sha256 固定，样本量由 `SUITE_SIZES` 在加载时校验（`--list-suites` 可打印全部 n/sha256）。

### 7.1 检测器为什么放进框架
我们的模型（以及所有结构化世界模型）需要物体框才能推理，而现实中既没有上游检测管线，也没有"平台检测"服务。若让每个参赛方自带检测器，"分数"就变成"检测器 × 状态模型"的乘积，无法归因、无法复现；若给 GT 框，输入里就含答案线索，不能算模型能力。**唯一可复现的做法：检测器属于测试框架**——一次实现、开源、冻结版本，所有需要框的模型拿到完全相同的框。

### 7.2 检测器规格（第一版）
GroundingDINO（tiny 权重，HF commit `a2bb814dd30d`，`model.safetensors` sha256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`，689,359,096 字节），固定输入尺寸与阈值，开放词表提示由框架按题面物体短语自动生成；输出 bbox 与分数一并写入框架日志，供任何人复算。**检测器错误由全体 B 类模型共同承担**，这是框架属性，不针对某家。

### 7.3 每一行必须披露
- 是否使用了框架提供的框（VLM 行为"否"，B 类模型为"是"）；
- 模型/系统组成（B 类行注明"框架框 + 状态模型"）；
- 覆盖率、题量、CI。

### 7.4 为什么不做分档排名
分档（纯像素 / 框架框 / GT 框）会让榜单解释成本变高，且容易给"自选输入"留后门。用户决策：**只做固定框架一种设置**。GT 框只允许作为内部诊断实验，不进榜、不发布分数。

---

## 8. 判分协议（统一到同一真值）

1. **答案规范**：文本模型输出 `choice`（选项下标）或 `short`（规范化匹配）或 `json`（字段级比较）；结构化模型输出状态对象（slots / pairs / events）。
2. **状态对齐**：槽位 id 与 GT 对齐；关系按 (from, to) 有向匹配；方向约定写进协议（我们已踩坑：服务端只输出升序对，需正/逆序两次推理后按 GT 方向取）。
3. **指标**：
   - 选择/短答：覆盖率内准确率 + bootstrap CI；
   - 变化：top-1 + 4 类 macro-F1（none/appeared/displaced/disappeared）；
   - 关系：关系级 accuracy / macro-F1；关系变化用 F1（新增/消失/改类型）；
   - 方位：只在关系正确的对上计（避免"关系错但方位蒙对"）；
   - 事件：四类事件 P/R/F1；
   - 回放：状态一致率（存在性/类型/关系/属性分项）。
4. **禁止事项**：缺输出不记 0；N/A 不当错；不做单一综合分主排序（综合分只作辅助，权重公开）。

---

## 9. 榜单呈现

1. **主表**：能力 × 模型，单元格 = 指标 + CI + 覆盖率；用"uses harness boxes"列区分 VLM 行与结构化模型行。
2. **能力雷达**：单模型在 T1–T8 的相对位置（可视化，不作排名依据）。
3. **分项表**：MQA 关系子集、变化子集、关系变化子集各一张，题量/难度/来源标注。
4. **稳健性曲线**：随视角偏移/遮挡比例/干扰物数量的性能衰减。
5. **不做**：单一综合分大排行榜（会诱导为权重取巧）。

---

## 10. 第一版可用结果

### 10.1 T1 单图关系（1,612 题，5 家文本模型，框架直出）

| 模型 | 准确率 | 95% CI | 覆盖率 |
|---|---|---|---|
| qwen3.8-flash | 76.55% | 74.44–78.54 | 100% |
| deepseek-flash | 74.88% | 72.89–76.74 | 100% |
| glm-5.3-flash | 73.88% | 71.77–75.87 | 100% |
| minimax-m3.1-flash | 62.32% | 60.02–64.68 | 99.94% |
| mimo-v2.6-pro | 54.59% | 52.23–57.07 | 100% |

### 10.2 T1 关系子集（305 题，"X 相对 Y" 问法，5 家文本模型）

glm-5.3-flash 92.79% [89.84–95.41] / qwen3.8-flash 92.13% [89.18–94.76] / deepseek-flash 90.82% [87.21–93.77] / minimax-m3.1-flash 88.16% [84.54–91.78] / mimo-v2.6-pro 86.84% [82.57–90.46]（305 题任务级 bootstrap CI）。

> 305 题是"同题同 GT"的锚点：StarLife-1-WSV 在**框架检测框**下跑同一批题后，才能与上表并列。

### 10.3 StarLife-1-WSV（结构化世界状态协议，部署服务口径）

- 回放/状态类读数：T6 `state_replay` 1,000 题已完成（确定性映射，映射不出记未覆盖）：
  **覆盖率 63.80%，覆盖题准确率 95.30%，95% CI [93.57, 96.87]**；
- 变化/关系类（T4/T5）与跨帧（T7）：部署服务的 change 头当前存在
  appeared/disappeared 语义与 presence 标记矛盾、displaced 概率饱和的问题（服务侧
  自述为部署候选口径、位置变化判定待 T14），确定性映射暂不可靠，待服务侧修复后
  再评；T7 为合成域，检测器为 22 类真实物体闭集，需要域适配后再评。
- 映射规则与分探针读数见 `docs/RESULTS_OPENWSV.md` §三；该模型自备检测器，
  **不使用**框架检测框（行标注 `uses harness boxes = 否`）。

### 10.4 旧分数能否直接搬上 OpenWSV 榜

**不能，需要按 openwsv-1 重跑。** 理由：
1. 旧 v2 数字用的是各自的脚本与提示词，解析规则不同（宽容/严格），失败重试策略不同；
2. OpenWSV 把"提示词版本 + 解析规则 + 覆盖率口径"冻结进框架，换框架必须重算，否则同一模型两套分数并存；
3. 旧 raw 响应可离线重判部分字段，但提示词不同意味着输入不同，重判不能替代重跑。

**重跑代价**：5 家 VLM 的 T1（1,612 题/模型），提示词与解析换成 openwsv-1；单模型约 1–2 小时（并发可压缩），API 费用可忽略。旧数字保留在 v2 报告中作为历史版本，不再进入新榜。

### 10.5 我们的模型（StarLife-1-WSV）在 OpenWSV 中的位置

- 能力：T1 关系、T4 空间变化、T5 关系变化、T6 状态回放（主项）；
- 输入：部署服务自备检测器（不消费框架检测框），结构化世界状态经确定性映射投影到
  同一答案空间；行标注 `uses harness boxes = 否`；
- 已出分：T6 `state_replay` 1,000 题，覆盖率 63.80% / 覆盖题准确率 95.30%
  （见 §10.3 与 `docs/RESULTS_OPENWSV.md`）；
- 待出分：T1-relation 305 题（与 5 家 VLM 同表）、T4/T5/T7（服务侧 change 头与域差
  问题定调后）。
- 与 VLM 同表时，自备检测器误差计入系统分数，这是端到端评测的应有之义；覆盖率不足
  的部分按未覆盖报告，不折算成错。

### 10.6 T4–T7 文本套件首轮完整结果（4 家文本模型）

`change` 1,000 / `relation_change` 600 / `state_replay` 1,000 / `cross_frame` 400，
四选一确定性判分、temperature=0、冻结提示词。综合分为四套件覆盖题准确率的等权平均，
仅四套件全部完成的模型进入综合分：

| 模型 | T4 | T5 | T6 | T7 | 综合分 [95% CI] | 覆盖率 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| deepseek-v4.1-flash | 94.93% | 95.75% | 80.96% | 99.50% | **92.79%** [91.93–93.60] | 97.67% |
| minimax-m3.1-flash | 93.50% | 97.00% | 79.70% | 94.25% | **91.11%** [90.07–92.13] | 100.00% |
| glm-5.3-flash | 95.80% | 98.00% | 81.00% | 84.96% | **89.94%** [88.80–91.03] | 99.97% |
| mimo-v2.6-pro | 92.19% | 92.71% | 75.32% | 61.50% | **80.43%** [78.53–82.19] | 90.30% |

* T4/T5/T7 已接近天花板（≥92%），真正区分来自 T6 与覆盖率；综合分上 deepseek 与
  glm 的 CI 不重叠，与 minimax 部分重叠。
* 361 条未解析回复已冻结提示词重放归档（`reports/openwsv-v2/diagnostics/`），
  解析器不放宽、覆盖率如实报告；mimo 的 T7 未覆盖主要是输出解释性长句而非字母。
* 逐套件 CI、精确 McNemar 与逐题预测见 `reports/openwsv-v2/TABLES.md` 与各报告 JSON。

---

## 11. 与现有榜单的差异

| 维度 | 公开榜单 | OpenWSV v3 |
|---|---|---|
| 答题形态 | 仅文本 | 文本 + 结构化状态双协议 |
| 能力覆盖 | 单点 | 关系 + 变化 + 关系变化 + 跨帧不相交 + 回放（已发布）；位姿/视角/抽象几何未构建 |
| 测试框架 | 各家脚本自定 | 冻结框架（数据/检测器/提示词/判分/种子） |
| 检测器 | 无或自备 | 框架内置冻结检测器，全体共用 |
| 覆盖率 | 不报或当 0 分 | 单独报告 |
| 统计 | 点值 | bootstrap CI + McNemar |
| 变化评测 | 无或 caption 相似度 | 确定性变化类别 + 关系级变化 F1 |
| 抗污染 | 一般 | 私有 holdout + canary + 生成器公开 |

---

## 12. 路线图

**阶段 0（已完成）**
- T1 全量 1,612 题（5 家文本模型）+ 关系子集 305 题；
- 双协议判分器、bootstrap CI、覆盖率报告。

**阶段 1（数据与框架已完成，`openwsv-1`）**
- T4–T7 数据与框架落地：`change` 1,000、`state_replay` 1,000（`pairs` GT 确定性投影，`--check` 零漂移）；`relation_change` 600、`cross_frame` 400（合成 `openwsv-data-1.1`，生成器 `openwsv-scene-1.0.0` 逐字节可复现）；
- 7 套 suite 全部接入框架：`load_suite` 校验样本量、`--list-suites` 输出 n 与 sha256；
- 已完成：4 家文本模型在 T4/T5/T6/T7 文本套件上的首轮完整跑分（结果见 §10.6）；
- 结构化 StarLife-1-WSV 轨道：T6 `state_replay` 1,000 题已出分（§10.3）；T4/T5 与
  T1-relation 305 题待服务侧 change 头修复后补跑，T7 需域适配后补跑。

**阶段 2（待排期）**
- T8 抽象几何（数据集与套件均未构建）；
- T2（位姿）、T3（跨视角）尚未实现，不在 `openwsv-1` 包内；
- 真实采集 300–500 对前后帧；
- 稳健性曲线（视角/遮挡/光照/干扰物）；
- 私有 holdout + canary。

**阶段 3（待排期）**
- 邀请 3–5 家外部模型/系统提交；
- 发布技术报告与数据卡；
- 建立复赛机制（私有集评测）。

人力：1 名工程 + 0.5 名标注/质检。算力：合成渲染与 CPU 判分为主，推理 A800 / RTX4070 各一台足够。

---

## 13. 风险与限制

1. **检测器误差是框架误差**：框架框错，所有 B 类模型一起错；公开权重与日志可让读者自行评估该误差，但无法归零。
2. **合成分布窄**：合成场景与真实的差距需真实采集数据平衡。
3. **文本模型"猜"策略**：二选一有 50% 下限，需对称陷阱与四选一题控制。
4. **样本量与显著性**：305 题在 ±5 点差异上不显著，结论要谨慎。
5. **标注成本**：真实前后帧的关系/变化标注是主要成本项。
6. **伦理与版权**：公开基准只取题目与真值并注明来源；自采数据需肖像/场所授权。

---

## 14. 附录

### 14.1 术语
- **槽位（slot）**：场景中可跟踪物体的表示；**关系（pair）**：两槽位的类型/方位/接触支撑；**变化（change）**：none / appeared / displaced / disappeared；**事件（event）**：带类型的时序片段。
- **冻结框架**：数据、检测器、提示词、判分器、种子全部版本化固定的评测环境。

### 14.2 历史数据（不进榜）
早期 60 对结构化评测使用了数据集 GT 框（输入含真值线索），按 v3.1 协议属于**诊断实验**，已从榜单元数据中移除，仅保留于仓库附录供追溯。

### 14.3 已发布的支撑材料
- v2 T1 结果：`publish/starlife_eval/docs/RESULTS_SPATIALMQA.md`
- 结构化轨道（历史诊断）：`publish/starlife_eval/docs/RESULTS_STRUCTURED_TRACK.md`
- 复现脚本：`publish/starlife_eval/scripts/structured_track/run_structured_track.py`
- 关系子集基准：`.scratch/v2run/leaderboard_relation_subset.json`

### 14.4 引用
- SpatialMQA: https://github.com/ziyan-xiaoyu/spatialmqa
- MindEdit-Bench: https://huggingface.co/datasets/ZODAOfficial/MindEdit-Bench
- Spatial457: https://openaccess.thecvf.com//content/CVPR2025/html/Wang_Spatial457_A_Diagnostic_Benchmark_for_6D_Spatial_Reasoning_of_Large_CVPR_2025_paper.html
- Spatial-DISE: https://ar5iv.labs.arxiv.org/html/2510.13394
- PureSpace: https://openaccess.thecvf.com/content/CVPR2026F/html/Li_PureSpace_A_Benchmark_for_Abstract_Spatial_Reasoning_in_Vision-Language_Models_CVPRF_2026_paper.html
- OmniSpatial: https://ar5iv.labs.arxiv.org/html/2506.03135
- SpinBench: https://proceedings.iclr.cc/paper_files/paper/2026/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html
- VSI-Bench: https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.html
- Disjoint-3DQA: https://ar5iv.labs.arxiv.org/html/2505.24257
- REM: https://github.com/jbt-cmu/REM
- World-In-World: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html
- ScratchWorld: https://arxiv-org.ezproxy.obspm.fr/html/2606.31689v1
- SSPRCD（关系变化检测）: https://doaj.org/article/05860e3ddaef464089d074ea6b75f8fa
