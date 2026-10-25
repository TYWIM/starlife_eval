# OpenWSV 数据包与复现说明

> 版本 `openwsv-data-1.0`（冻结）+ `openwsv-data-1.1`（增量，T5/T7 合成数据），
> 框架版本 `openwsv-1`，冻结日期 2026-10-25。数据包只包含**图像 + 任务文件**；
> 检测器权重不随包分发（按 sha256 从 Hugging Face 拉取）。

## 1. 内容

```
openwsv-data-1.0/                       # 冻结数据包（13 个分片随仓库提交）
├── v2data/mqa/images/                  # 1,583 张单图空间关系题面图像
├── data_v15/dataset_spatial_v3/frames/ # 2,000 张前后帧（1,000 对 test 划分）
├── suites/spatialmqa.jsonl             # 1,612 题（题面/选项/答案）
├── suites/pairs_test.jsonl             # 1,000 对结构化 GT（变化/关系/方位/接触支撑）
├── SHA256SUMS                          # 全部文件的 sha256
└── README.md

openwsv-data-1.1/                       # 增量数据包（本地生成，不入 git；确定性可复现）
├── t5_frames/                          # 1,200 张 T5 关系变化前后帧
├── t7_frames/                          # 800 张 T7 跨帧不相交帧
├── SHA256SUMS
├── meta.json                           # 生成器版本 / seed / 计数 / 两类分布
└── README.md
```

说明：
- `spatialmqa.jsonl` 中 29 题引用的图像不在公开图像包内（原数据集缺图），这些题在评测时记为**未覆盖**（`covered=false`），不计入准确率分母；
- `pairs_test.jsonl` 的 GT 由合成场景生成器 `spatial3d-1.0.0` 给出，非真实拍摄；
- `openwsv-data-1.1` 的 T5/T7 由 `openwsv-scene-1.0.0` 确定性生成（seed=20261020），
  与 1.0 相同相机模型；T4/T6 **不新增图像**，直接复用 1.0 的 pairs 帧；
- 评测时的数据根应同时包含两个包（`--data-root data`），suite 内的图像路径带包前缀
  `openwsv-data-1.0/...` / `openwsv-data-1.1/...`；spatialmqa/pairs 仍可用
  `OPENWSV_DATA_ROOT=./openwsv-data-1.0` 跑。

## 2. 下载、生成与校验

```bash
# 1.0 数据包随仓库提交，无需下载：data/openwsv-data-1.0/ 下有 13 个分片
# 一次 join + 校验 + 解压：
python scripts/wsv/join_data.py --dest ./openwsv-data-1.0.zip --extract-to ./data

# 手工做法：
cat data/openwsv-data-1.0/openwsv-data-1.0.zip.part* > openwsv-data-1.0.zip
sha256sum openwsv-data-1.0.zip   # 期望 971090ee15347b016f5c4afd92b6a711481d81de169c9d45d55e64a6fef2ad63
unzip openwsv-data-1.0.zip
cd openwsv-data-1.0 && sha256sum -c SHA256SUMS

# 1.1 数据包：确定性生成（约 2,000 张 JPEG，默认写入 data/openwsv-data-1.1）
python -m scripts.wsv.gen.build --force --determinism-check
# T4/T6 文本套件：从冻结 pairs GT 派生（--check 验证零漂移）
python scripts/wsv/derive_text_suites.py --check
```

> `join_data.py` 会校验拼回后的 zip sha256 与包内 `SHA256SUMS`，任一不符即失败；
> 注意它会把解压结果放在 `data/openwsv-data-1.0/`，会把该目录下的分片移出工作区
> （`git checkout -- data/openwsv-data-1.0/` 可恢复被跟踪的分片）。

## 3. 跑评测

```bash
# T1：图像路径在 1.0 包内，data root 指向解压后的 1.0 包
export OPENWSV_DATA_ROOT=./openwsv-data-1.0
python -m starlife_eval.wsv run --suite spatialmqa --model <id> \
  --adapter openai --base-url "$OPENAI_BASE_URL" --api-key "$OPENAI_API_KEY" \
  --out reports/openwsv/spatialmqa__<id>.json

# T4/T5/T6/T7：数据根需同时含 1.0 与 1.1，路径带包前缀
python -m starlife_eval.wsv run --suite change --model <id> \
  --adapter openai --base-url "$OPENAI_BASE_URL" --api-key "$OPENAI_API_KEY" \
  --data-root data --out reports/openwsv-v2/ --workers 3
```

结构化模型（T4–T6）：

```bash
python -m starlife_eval.wsv run --suite pairs --model <id> \
  --adapter wsv --api-url "$WSV_API_URL" --api-key "$WSV_API_KEY" \
  --out reports/openwsv/pairs__<id>.json
```

运行时会自动加载**框架冻结检测器**（GroundingDINO-tiny，权重 sha256 固定），
首次运行按配置下载权重；权重 sha256 与 `wsv/config.py` 不一致会直接报错。

## 4. 复现口径

- 未覆盖 ≠ 答错：空输出、超时、无法解析、图像缺失均记 `covered=false`，从准确率分母剔除；
- 所有准确率附 95% 任务级 bootstrap CI（1,000 次，`seed=20261022`）；
- 结果 JSON 的 `framework` 段记录数据 sha256、检测器 sha256、提示词版本、判分器 commit 与种子；
  任何人可用同一 JSON 校验分数是否由同一框架产出；
- T4/T6 由 `scripts/wsv/derive_text_suites.py` 从 pairs GT 确定性派生，`--check` 必须零漂移；
  T5/T7 由 `openwsv-scene-1.0.0` 确定性生成（seed=20261020），重跑生成器应得相同字节。

## 5. 数据来源与许可

- `spatialmqa` 题面与真值来自其公开数据集（按其原许可引用），仅重排路径与判分口径；
- `pairs` 为自建合成数据（生成器 `spatial3d-1.0.0`），随本包公开，私有 holdout 15% 不发布；
- T5/T7 为自建合成数据（生成器 `openwsv-scene-1.0.0`），生成器随仓库公开，图像可确定性重建。
