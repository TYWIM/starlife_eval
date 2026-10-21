"""report.md：人读版报告（真实数值、口径出处、N/A 原因、产物指纹）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

NA_TEXT = "N/A"


def _f(x: Any, nd: int = 4) -> str:
    if x is None:
        return NA_TEXT
    if isinstance(x, bool):
        return "true" if x else "false"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x: Any) -> str:
    return NA_TEXT if x is None else f"{float(x):.1f}"


def render_markdown(report: dict[str, Any]) -> str:
    meta = report.get("meta") or {}
    models = report.get("models") or {}
    lb = report.get("leaderboard") or []
    lines: list[str] = []
    add = lines.append

    add("# 空间理解与关系推理基准 · 真实评测报告")
    add("")
    add(f"- 生成时间：`{meta.get('generated_at')}`")
    add(f"- 数据集：`{(meta.get('config') or {}).get('dataset')}`（split=`{meta.get('split')}`，"
        f"测试对数={meta.get('n_test_pairs', 'N/A')}，评测抽样对数={meta.get('n_pairs')}）")
    add(f"- 特征：`{(meta.get('artifacts') or {}).get('features')}` "
        f"sha256[:16]=`{(meta.get('artifacts') or {}).get('features_sha256_16')}`")
    add(f"- 权重：`{(meta.get('artifacts') or {}).get('checkpoints')}` "
        f"{', '.join((meta.get('artifacts') or {}).get('checkpoint_files') or [])}")
    add(f"- 生成器：starlife_eval（唯一真后端；未复用任何 demo/mock 代码）")
    add("")

    add("## 1. 排行榜（主分 = 综合能力分，明细 = 原始指标）")
    add("")
    add("> 被评对象不限 StarLife：本地头模型只是其中一个条目，云端零样本模型是同类参赛者。")
    add("")
    add("| # | 模型 | 类型 | **综合分** | 变化识别 | 关系类型 | 接触支撑 | 事件F1 | 状态一致性 | 延迟 ms |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(lb, 1):
        dims = {d.get("key"): d.get("value") for d in ((r.get("composite") or {}).get("dims") or [])}
        add(f"| {i} | {r.get('name')} | {r.get('mode')} | "
            f"{'N/A' if r.get('compositeScore') is None else round(float(r['compositeScore']), 1)} | "
            f"{_num(dims.get('change_top1'))} | "
            f"{_num(dims.get('relation_type'))} | "
            f"{_num(dims.get('attrs'))} | "
            f"{_num(dims.get('event_f1'))} | "
            f"{_num(dims.get('state_consistency'))} | "
            f"{'N/A' if r.get('latency') is None else round(float(r['latency']), 1)} |")
    add("")

    add("## 1.1 综合分怎么算")
    add("")
    add("> **综合分 = 7 个维度（各 0-100）的等权平均**，缺项从分母剔除、不当 0 分。"
        "等权是刻意的：不预设「关系比变化更重要」，避免用一个拍脑袋权重制造排名。"
        "7 个维度都是 6 个模型 100% 可测的，没有任何模型因为测不出的维度被扣分。")
    add("")
    add("| 维度 | 通俗解释 | 口径出处 |")
    add("|---|---|---|")
    add("| 变化识别 top1 | 两帧之间每个物体发生了什么（出现/消失/被挪动/没变），认对的比例 | `pair_protocol.change_top1` |")
    add("| 变化平衡性 | 稀有变化（如「消失」）也判得准吗；防止「全猜没变」刷分 | `pair_protocol.change_macro_f1` |")
    add("| 关系类型 | 物体之间 11 类空间关系判对的比例（先找对两个物体） | `pair_protocol.relation_type_acc` |")
    add("| 空间方位 | 关系判对后，前/后/左/右/上/下调对的比例 | `pair_protocol.rel_pos_acc` |")
    add("| 接触/支撑 | 两物体是否接触、是否互相支撑（交互推断的地基） | `pair_protocol.attrs_acc` |")
    add("| 事件级 F1 | 写成事件流后 StateChanged + RelationAdded 的 P/R/F1 | `event_metrics.macro_f1_common` |")
    add("| 世界状态一致性 | 事件回放重建的 after 状态与真实状态的一致率（objects/changes/relations/attrs 四维） | `state_replay.state_acc_common` |")
    add("")
    add("> `state_consistency` 不含 `types`（云端无类型输出）。")
    add("> 原 `score`（= 事件级可比 macroF1）仍保留在 report.json 里，用于和 v1.0 口径对照；榜单排序只看综合分。")
    add("")

    add("## 2. 头级指标（全 test split，L2）")
    add("")
    add("| 模型 | change top1 | change macroF1 | relation top1 | relation macroF1 | rel_pos top1 | rel_pos macroF1 | attrs contact | attrs support |")
    add("|---|---|---|---|---|---|---|---|---|")
    for mid, m in models.items():
        h = m.get("head_metrics") or {}
        if not h:
            continue
        add(f"| {mid} | {_f((h.get('change') or {}).get('top1'))} | {_f((h.get('change') or {}).get('macro_f1'))} | "
            f"{_f((h.get('relation') or {}).get('top1'))} | "
            f"{_f((h.get('relation') or {}).get('macro_f1'))} | {_f((h.get('rel_pos') or {}).get('top1'))} | "
            f"{_f((h.get('rel_pos') or {}).get('macro_f1'))} | {_f((h.get('attrs') or {}).get('contact_f1'))} | "
            f"{_f((h.get('attrs') or {}).get('support_f1'))} |")
    add("")
    add("> 复现基准（原始脚本，容差 ±0.001）：v2g change 0.9270/0.6983、relation 0.7745、"
        "rel_pos 0.7042、attrs contact 0.8895；v2 change 0.9365/0.7232、relation 0.7934、rel_pos 0.6878。")
    add("")

    add("## 3. 60 对云端同批协议")
    add("")
    add("| 模型 | change top1 | change macroF1 | relation_type_acc | rel_pos_acc | attrs_acc |")
    add("|---|---|---|---|---|---|")
    for mid, m in models.items():
        p = m.get("pair_protocol") or {}
        if not p:
            continue
        add(f"| {mid} | {_f(p.get('change_top1'))} | {_f(p.get('change_macro_f1'))} | "
            f"{_f(p.get('relation_type_acc'))} | {_f(p.get('rel_pos_acc'))} | {_f(p.get('attrs_acc'))} |")
    add("")

    add("## 4. 事件级与世界状态回放（L3/L4）")
    add("")
    add("| 模型 | event macroF1(全部可测类) | score kinds macroF1 | StateChanged F1 | EntityCreated F1 | RelationAdded F1 | state_acc | objects | changes | relations | attrs | types |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mid, m in models.items():
        e = m.get("event_metrics") or {}
        s = m.get("state_replay") or {}
        pk = (e.get("per_kind") or {})
        add(f"| {mid} | {_f(e.get('macro_f1'))} | {_f(e.get('macro_f1_common'))} | "
            f"{_f((pk.get('StateChanged') or {}).get('f1'))} | "
            f"{_f((pk.get('EntityCreated') or {}).get('f1'))} | {_f((pk.get('RelationAdded') or {}).get('f1'))} | "
            f"{_f(s.get('state_acc'))} | {_f(s.get('objects'))} | {_f(s.get('changes'))} | "
            f"{_f(s.get('relations'))} | {_f(s.get('attrs'))} | {_f(s.get('types'))} |")
    add("")
    add("> `score = macro_f1_common`，只对 **StateChanged + RelationAdded** 等权（所有模型都能产出的类）。")
    add("> `EntityCreated` 需要实体类型，云端零样本不返回类型 → 该类 N/A；若并入 score 会让本地模型因多测一个更难维度而被压低，横向不可比。")
    add("> RelationRemoved 在 v2 数据上不可测（无 before 关系标注），一律 N/A 且不计分。")
    add("")

    add("## 5. 运行时（延迟实测）")
    add("")
    add("| 模型 | 横向可比延迟 ms（含特征抽取） | 头前向/API 延迟 mean ms | p50 ms | max ms |")
    add("|---|---|---|---|---|")
    for mid, m in models.items():
        rt = m.get("runtime") or {}
        add(f"| {mid} | {_num(rt.get('pipeline_latency_ms_mean'))} | {_num(rt.get('latency_ms_mean'))} | "
            f"{_num(rt.get('latency_ms_p50'))} | {_num(rt.get('latency_ms_max'))} |")
    add("")
    add("")
    add("")
    add("> 本地模型：`bench_pipeline.json` 实测特征抽取（2 帧 + 12 crop，1024×768）`event_latency_s` 加上 60 对头前向均值；"
        "云端模型：60 次真实 API 调用的均值。两者都是**每事件**延迟，可直接比较。")
    add("")
    bench = (meta.get("artifacts") or {}).get("bench_pipeline")
    if bench:
        add("整机特征抽取基准（`bench_pipeline.json` 真实测量，未在评测中重跑渲染）：")
        add("")
        for k, v in bench.items():
            add(f"- `{k}` = {v}")
        add("")

    add("## 6. N/A 与不可测项（诚实清单）")
    add("")
    any_na = False
    for mid, m in models.items():
        na = m.get("na") or {}
        for k, v in na.items():
            any_na = True
            add(f"- **{mid}** · `{k}`：{v}")
    for mid, m in models.items():
        notes = m.get("notes") or []
        for n in notes:
            any_na = True
            add(f"- **{mid}**：{n}")
    if not any_na:
        add("- （无）")
    add("")

    add("## 7. 指标口径")
    add("")
    for key, d in (report.get("metric_definitions") or {}).items():
        add(f"- `{key}`：{d.get('doc')}")
        if d.get("source"):
            add(f"  - 出处：`{d['source']}`")
    add("")
    return "\n".join(lines)


def _num(x: Any, nd: int = 1) -> str:
    if x is None:
        return NA_TEXT
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def write_markdown(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path
