"""OpenWSV runner：在冻结框架下跑一套 suite × 一个模型，写出稳定契约的结果 JSON。

冻结框架五件套（结果 JSON ``framework`` 字段）：

=============  ==========================================================
数据版本        suite 冻结文件 sha256（``framework.dataset_sha256``）
检测器          GroundingDINO-tiny（权重 sha256 + 阈值），``detector.py`` 提供
提示词版本      ``openwsv-1``（``prompts.PROMPT_VERSION``）
判分器          ``starlife_eval@<git commit>``
随机种子        采样/评测 seed=20261020，bootstrap seed=20261022
=============  ==========================================================

检测器属于框架：所有"需要框"的模型共用同一份 ``detector.detect(image_path,
phrases) -> {phrase: bbox, "scores": ..., "meta": ...}`` 输出，不让模型自带
检测器（否则感知与推理混在一起，无法横测）。

结果 JSON 契约（字段稳定，站点数据生成依赖它）::

    {
      "framework": {"version", "dataset_sha256", "detector": {"model",
                     "weights_sha256", "box_threshold", "text_threshold"},
                     "prompt_version", "scorer", "seed", "bootstrap_seed"},
      "suite": "...", "n": <int>,
      "model": {"id", "kind": "vlm"|"structured", "uses_harness_boxes",
                "adapter"},
      "metrics": {"accuracy", "coverage", "ci95", ...suite 补充指标},
      "per_task": [{"task_id", "covered", "pred", "gold", "error"}]
    }

文本 suite（``spatialmqa`` / ``spatialmqa_relation``）：``accuracy`` = 覆盖题准确率，
``coverage`` = 可解析且无错误的题占比，``pred``/``gold`` = 选项下标。

结构化 suite（``pairs``）：``accuracy`` = change_top1（主指标），并附带
``change_macro_f1`` / ``relation_type_acc`` / ``rel_pos_acc`` / ``attrs_acc``
与 ``ci95_detail``；``pred``/``gold`` = 关系类型命中数 / GT 关系数。

约定：模型调用失败或输出无法解析记为**未覆盖**（``covered=false``），
不按错题计；覆盖率低于 100% 的行不可当作最终分。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters import HttpVlmAdapter, WsvApiAdapter
from .prompts import PROMPT_VERSION, build_prompt
from .scoring import DEFAULT_BOOTSTRAP_SEED, DEFAULT_BOOT, score_choices
from .state import evaluate_pairs
from .tasks import SUITES, dataset_sha256, load_suite

try:  # 冻结常量以 config.py 为唯一真源（Lead 维护）
    from .config import DETECTOR as _CONFIG_DETECTOR
    from .config import FRAMEWORK_VERSION as _CONFIG_FRAMEWORK_VERSION
    from .config import SEEDS as _CONFIG_SEEDS
except Exception:  # noqa: BLE001 - config.py 缺失时退化为内置默认值
    _CONFIG_DETECTOR = {}
    _CONFIG_FRAMEWORK_VERSION = None
    _CONFIG_SEEDS = {}

__all__ = [
    "FRAMEWORK_VERSION",
    "SAMPLE_SEED",
    "BOOTSTRAP_SEED",
    "DETECTOR_DEFAULTS",
    "framework_block",
    "run_suite",
]

#: 冻结框架版本（与 PROMPT_VERSION 同步升级；值来自 config.py）
FRAMEWORK_VERSION = _CONFIG_FRAMEWORK_VERSION or "openwsv-1"

#: 任务/采样随机种子（冻结）
SAMPLE_SEED = int(_CONFIG_SEEDS.get("sample", 20261020))

#: bootstrap CI 种子（冻结）
BOOTSTRAP_SEED = int(_CONFIG_SEEDS.get("bootstrap", DEFAULT_BOOTSTRAP_SEED))

#: 检测器冻结参数的默认值（``detector.py`` 提供真实值后覆盖）
DETECTOR_DEFAULTS: dict[str, Any] = {
    "model": str(_CONFIG_DETECTOR.get("model", "IDEA-Research/grounding-dino-tiny")),
    "weights_sha256": "",
    "box_threshold": float(_CONFIG_DETECTOR.get("box_threshold", 0.2)),
    "text_threshold": float(_CONFIG_DETECTOR.get("text_threshold", 0.2)),
}

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SUITE_PROTOCOL = {
    "spatialmqa": "choice",
    "spatialmqa_relation": "choice",
    "pairs": None,  # 结构化，不走文本提示词
    "change": "choice",
    "relation_change": "choice",
    "state_replay": "choice",
    "cross_frame": "choice",
}

_STRUCTURED_ADAPTERS = {"wsv", "wsv_api", "wsv-api", "structured"}
_VLM_ADAPTERS = {"openai", "http", "vlm", "chat"}


# ----------------------------------------------------------------- framework


def _normalize_detector_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """把 detector 元信息裁剪成结果 JSON 契约里的 4 个字段。"""
    out = dict(DETECTOR_DEFAULTS)
    for key in DETECTOR_DEFAULTS:
        value = meta.get(key)
        if value in (None, ""):
            continue
        out[key] = float(value) if key in ("box_threshold", "text_threshold") else value
    return {
        "model": str(out["model"]),
        "weights_sha256": str(out["weights_sha256"]),
        "box_threshold": float(out["box_threshold"]),
        "text_threshold": float(out["text_threshold"]),
    }


def _detector_meta() -> dict[str, Any]:
    """检测器冻结参数：config.py < detector.py（含离线 load_metadata）< 环境变量。"""
    meta = dict(DETECTOR_DEFAULTS)
    try:
        from . import detector as det  # noqa: PLC0415 - 由 Lead 提供该模块
    except Exception:  # noqa: BLE001 - 未提供 detector.py 时用默认值
        det = None
    if det is not None:
        spec = getattr(det, "DETECTOR", None)
        if isinstance(spec, Mapping):
            meta.update({k: spec[k] for k in DETECTOR_DEFAULTS if k in spec})
        loader = getattr(det, "load_metadata", None)
        if callable(loader):
            try:  # 不加载权重，离线可用；权重未落盘时为 "pending-load"
                loaded = loader()
                if isinstance(loaded, Mapping):
                    meta.update(
                        {k: loaded[k] for k in DETECTOR_DEFAULTS
                         if k in loaded and loaded[k] not in (None, "")}
                    )
            except Exception:  # noqa: BLE001
                pass

    env_map = {
        "OPENWSV_DETECTOR_MODEL": "model",
        "OPENWSV_DETECTOR_WEIGHTS_SHA256": "weights_sha256",
        "OPENWSV_DETECTOR_BOX_THRESHOLD": "box_threshold",
        "OPENWSV_DETECTOR_TEXT_THRESHOLD": "text_threshold",
    }
    for env, key in env_map.items():
        value = os.environ.get(env)
        if value not in (None, ""):
            if key in ("box_threshold", "text_threshold"):
                meta[key] = float(value)
            else:
                meta[key] = value
    return _normalize_detector_meta(meta)


def _scorer_commit() -> str:
    """判分器身份：``starlife_eval@<short commit>``（不在 git 仓库时 unknown）。"""
    override = os.environ.get("OPENWSV_SCORER", "").strip()
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        commit = out.stdout.strip()
        if out.returncode == 0 and commit:
            return f"starlife_eval@{commit}"
    except Exception:  # noqa: BLE001 - 非 git 环境（sdist/zip）不阻塞评测
        pass
    return "starlife_eval@unknown"


def framework_block(suite: str) -> dict[str, Any]:
    """组装结果 JSON 的 ``framework`` 冻结信息。"""
    return {
        "version": FRAMEWORK_VERSION,
        "dataset_sha256": dataset_sha256(suite),
        "detector": _detector_meta(),
        "prompt_version": PROMPT_VERSION,
        "scorer": _scorer_commit(),
        "seed": SAMPLE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


# ------------------------------------------------------------------- helpers


def _as_specs(models: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(models, Mapping):
        return [dict(models)]
    specs = [dict(m) for m in models]
    if not specs:
        raise ValueError("models 不能为空")
    return specs


def _adapter_name(spec: Mapping[str, Any], suite: str) -> tuple[str, str]:
    """返回 ``(adapter 名, kind)``；kind ∈ {vlm, structured}。"""
    raw = str(spec.get("adapter") or ("wsv" if suite == "pairs" else "openai")).lower()
    if raw in _STRUCTURED_ADAPTERS:
        if suite != "pairs":
            raise ValueError(f"suite {suite!r} 是文本多选，不能使用结构化 adapter {raw!r}")
        return raw, "structured"
    if raw in _VLM_ADAPTERS:
        if suite == "pairs":
            raise ValueError(
                "suite 'pairs' 需要结构化 adapter（WSV /predict），不能用文本 VLM adapter"
            )
        return raw, "vlm"
    raise ValueError(f"未知 adapter {raw!r}；可用 openai/http/vlm 或 wsv/api/structured")


def _build_adapter(spec: Mapping[str, Any], adapter: str, kind: str) -> Any:
    if kind == "vlm":
        base_url = spec.get("base_url")
        if not base_url:
            raise ValueError(f"model {spec.get('id')!r} 缺 base_url（OpenAI 兼容端点根 URL）")
        return HttpVlmAdapter(
            base_url=str(base_url),
            api_key=spec.get("api_key"),
            model=str(spec.get("model") or spec["id"]),
            temperature=float(spec.get("temperature", 0.0)),
            max_tokens=int(spec.get("max_tokens", 1024)),
            timeout_s=float(spec.get("timeout_s", 180.0)),
        )
    api_url = spec.get("api_url") or spec.get("base_url")
    if not api_url:
        raise ValueError(f"model {spec.get('id')!r} 缺 api_url（WSV 服务根 URL）")
    return WsvApiAdapter(
        api_url=str(api_url),
        api_key=spec.get("api_key"),
        timeout_s=float(spec.get("timeout_s", 180.0)),
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def _round4(value: Any) -> Any:
    if value is None:
        return None
    return round(float(value), 4)


# --------------------------------------------------------------- data roots


def _data_root(explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("OPENWSV_DATA_ROOT") or os.environ.get("V2_DATA_ROOT")
    return Path(env) if env else Path(".")


def _resolve_image(path: str | Path, data_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return data_root / candidate


def _dataset_dir(data_root: Path) -> Path:
    """结构化 pairs 的数据集根目录（``frames/`` + ``manifest.jsonl``）。"""
    env = os.environ.get("OPENWSV_DATASET_DIR")
    if env:
        return Path(env)
    candidates = [
        data_root / "data_v15" / "dataset_spatial_v3",
        data_root / "dataset_spatial_v3",
        Path("data_v15") / "dataset_spatial_v3",
    ]
    for candidate in candidates:
        if (candidate / "manifest.jsonl").exists():
            return candidate
    return candidates[0]


def _load_detector() -> tuple[Any, dict[str, Any] | None]:
    """加载框架检测器，返回 ``(detect 可调用对象, 冻结元信息)``。

    兼容两种 ``detector.py`` 形态：

    * 模块级 ``detect(image, phrases)`` 函数；
    * （当前实现）``FrozenDetector`` 类：实例化一次、``.detect(image, phrases)``，
      元信息取实例的 ``.meta()``（含真实 weights sha256）。
    """
    try:
        from . import detector as det  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "pairs 套件需要框架检测器 detector.py（由 Lead 提供）："
            "FrozenDetector().detect(image, phrases) 或模块级 detect(image, phrases)"
        ) from exc

    module_detect = getattr(det, "detect", None)
    if callable(module_detect):
        return module_detect, None

    detector_cls = getattr(det, "FrozenDetector", None)
    if detector_cls is None:
        raise RuntimeError("detector.py 既没有 detect() 函数，也没有 FrozenDetector 类")
    try:
        instance = detector_cls()
    except Exception as exc:  # noqa: BLE001 - torch/transformers/权重缓存缺失时给出可操作提示
        raise RuntimeError(
            "框架检测器加载失败（需要 torch + transformers + GroundingDINO-tiny 权重缓存/离线目录）："
            f"{type(exc).__name__}: {exc}"
        ) from exc
    meta: dict[str, Any] | None = None
    try:
        value = instance.meta()
        if isinstance(value, Mapping):
            meta = _normalize_detector_meta(value)
    except Exception:  # noqa: BLE001 - 元信息失败不阻塞评测
        meta = None
    return instance.detect, meta


def _lookup_detection(detections: Mapping[str, Any], phrase: str) -> Any:
    if phrase in detections:
        return detections[phrase]
    lowered = {str(k).lower(): v for k, v in detections.items()}
    for candidate in (phrase.lower(), phrase.lower().replace("_", " ")):
        if candidate in lowered:
            return lowered[candidate]
    return None


def _detection_box(value: Any) -> list[float] | None:
    """支持 ``bbox`` 直接是 4 数或 ``{bbox|box: [...]}``。"""
    if isinstance(value, Mapping):
        value = value.get("bbox") or value.get("box")
    if value is None:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return box if len(box) == 4 else None


def _pair_detections(
    task: Mapping[str, Any],
    *,
    dataset_dir: Path,
    detect: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """跑一次框架检测器，返回 forward / reversed 两种检测排序。"""
    meta = task.get("meta") or {}
    entities = list(meta.get("entities") or ())
    before_path = dataset_dir / str(meta["before"])
    after_path = dataset_dir / str(meta["after"])
    before_phrases = [str(e["type_key"]) for e in entities]
    after_entities = [e for e in entities if bool(e.get("present_after", True))]
    after_phrases = [str(e["type_key"]) for e in after_entities]

    before_det = detect(before_path, before_phrases) or {}
    after_det = detect(after_path, after_phrases) or {}
    if not isinstance(before_det, Mapping) or not isinstance(after_det, Mapping):
        raise RuntimeError(f"detector.detect 返回类型非法（pair {meta.get('pair_id')}）")

    # ``FrozenDetector.detect`` 用 "_meta"/"_image_size"；兼容旧接口的 "meta"/width/height
    frame_meta = before_det.get("_meta") or before_det.get("meta") or {}
    if not isinstance(frame_meta, Mapping):
        frame_meta = {}
    image_size = before_det.get("_image_size") or after_det.get("_image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        width, height = int(image_size[0]), int(image_size[1])
    else:
        width = int(frame_meta.get("width") or frame_meta.get("image_width") or 1024)
        height = int(frame_meta.get("height") or frame_meta.get("image_height") or 768)

    def boxes(ordered: Sequence[Mapping[str, Any]], frame_det: Mapping[str, Any]) -> list[dict[str, Any]]:
        out = []
        for entity in ordered:
            phrase = str(entity["type_key"])
            value = _lookup_detection(frame_det, phrase)
            box = _detection_box(value)
            if box is None:
                continue  # 未检测到 -> 该 slot 不进入 WSV 输入（不伪造框）
            out.append({"id": f"slot{int(entity['slot'])}", "bbox": box})
        return out

    def layout(ordered: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        present_after = {
            int(e["slot"]) for e in entities if bool(e.get("present_after", True))
        }
        return {
            "width": width,
            "height": height,
            "before": boxes(ordered, before_det),
            "after": boxes([e for e in ordered if int(e["slot"]) in present_after], after_det),
        }

    forward = layout(entities)
    reversed_layout = layout(list(reversed(entities)))
    return forward, reversed_layout


# -------------------------------------------------------------------- runners


def _run_text_suite(
    suite: str,
    tasks: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    adapter_name, kind = _adapter_name(spec, suite)
    adapter = _build_adapter(spec, adapter_name, kind)
    protocol = str(spec.get("protocol") or _SUITE_PROTOCOL[suite])
    data_root = _data_root(spec.get("data_root"))
    workers = max(1, int(spec.get("workers", 1)))
    preds: dict[str, dict[str, Any]] = {}

    def run_one(task: Mapping[str, Any]) -> None:
        prompt = build_prompt(task, protocol=protocol)
        images = [_resolve_image(p, data_root) for p in task.get("images") or ()]
        result = adapter.complete(prompt, images)
        preds[str(task["task_id"])] = {
            "text": result.get("text"),
            "error": result.get("error"),
            "latency_s": result.get("latency_s"),
        }

    total = len(tasks)
    done = [0]
    if workers == 1:
        for task in tasks:
            run_one(task)
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  {done[0]}/{total} tasks", flush=True)
    else:
        lock = threading.Lock()

        def wrapped(task: Mapping[str, Any]) -> None:
            run_one(task)
            with lock:
                done[0] += 1
                if done[0] % 25 == 0:
                    print(f"  {done[0]}/{total} tasks", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(wrapped, tasks))

    scored = score_choices(tasks, preds)
    metrics = {
        "accuracy": _round4(scored["accuracy"]),
        "coverage": _round4(scored["coverage"]),
        "ci95": [_round4(v) for v in scored["ci95"]],
    }
    return {
        "model": {
            "id": str(spec["id"]),
            "kind": "vlm",
            "uses_harness_boxes": False,
            "adapter": adapter.adapter_id,
        },
        "metrics": metrics,
        "per_task": [
            {
                "task_id": row["task_id"],
                "covered": bool(row["covered"]),
                "pred": row["pred"],
                "gold": row["gold"],
                "error": row["error"],
            }
            for row in scored["per_task"]
        ],
    }


def _run_pairs_suite(
    suite: str,
    tasks: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    *,
    detect: Any,
) -> dict[str, Any]:
    adapter_name, kind = _adapter_name(spec, suite)
    adapter = _build_adapter(spec, adapter_name, kind)
    data_root = _data_root(spec.get("data_root"))
    dataset_dir = _dataset_dir(data_root)

    responses: dict[str, dict[str, Any]] = {}
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        task_id = str(task["task_id"])
        try:
            forward_dets, reversed_dets = _pair_detections(
                task, dataset_dir=dataset_dir, detect=detect
            )
        except Exception as exc:  # noqa: BLE001 - 单对检测失败 -> 未覆盖，不中断整轮
            responses[task_id] = {"error": f"detector: {type(exc).__name__}: {exc}"}
            continue
        meta = task.get("meta") or {}
        before = dataset_dir / str(meta["before"])
        after = dataset_dir / str(meta["after"])
        forward = adapter.predict(before, after, forward_dets)
        reversed_ = adapter.predict(before, after, reversed_dets)
        error = forward.get("error") or reversed_.get("error")
        responses[task_id] = {
            "forward": forward if not forward.get("error") else None,
            "reversed": reversed_ if not reversed_.get("error") else None,
            "error": error,
        }
        if index % 10 == 0:
            print(f"  {index}/{total} pairs", flush=True)

    evaluation = evaluate_pairs(tasks, responses, n_boot=DEFAULT_BOOT, seed=BOOTSTRAP_SEED)
    point = evaluation["metrics"]
    ci = evaluation["ci95"]
    metrics = {
        "accuracy": _round4(point["change_top1"]),
        "coverage": _round4(evaluation["coverage"]),
        "ci95": list(ci.get("change_top1") or [0.0, 0.0]),
        "change_top1": _round4(point["change_top1"]),
        "change_macro_f1": _round4(point["change_macro_f1"]),
        "relation_type_acc": _round4(point["relation_type_acc"]),
        "rel_pos_acc": _round4(point["rel_pos_acc"]),
        "attrs_acc": _round4(point["attrs_acc"]),
        "ci95_detail": {k: (list(v) if v is not None else None) for k, v in ci.items()},
        "change_slots": int(point["change_slots"]),
        "relation_pairs": int(point["relation_pairs"]),
    }
    per_task = [
        {
            "task_id": row["task_id"],
            "covered": bool(row["covered"]),
            "pred": int(row["relation_type_correct"]),
            "gold": int(row["relation_pairs"]),
            "error": row["error"],
        }
        for row in evaluation["per_pair"]
    ]
    return {
        "model": {
            "id": str(spec["id"]),
            "kind": "structured",
            "uses_harness_boxes": True,
            "adapter": adapter.adapter_id,
        },
        "metrics": metrics,
        "per_task": per_task,
    }


# --------------------------------------------------------------------- public


def run_suite(
    suite: str,
    models: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    out_dir: str | Path,
    limit: int | None = None,
    data_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """跑一套 suite（全部任务或 ``limit`` 冒烟）并写出结果 JSON。

    Parameters
    ----------
    suite:
        ``spatialmqa`` | ``spatialmqa_relation`` | ``pairs``。
    models:
        单个或一组模型 spec dict；公共字段 ``id`` / ``adapter``。文本模型另需
        ``base_url`` / ``model``，可选 ``api_key`` / ``temperature`` /
        ``max_tokens`` / ``workers``；结构化模型需 ``api_url``（或 ``base_url``）
        / ``api_key``。``api_key`` 仅内存使用，绝不落盘。
    out_dir:
        输出目录；每个模型写 ``{suite}__{model}.json``，单模型时另写 ``result.json``。
    limit:
        只跑前 ``limit`` 条任务（冒烟用；``None`` 跑全量）。
    data_root:
        图像相对路径的根目录；缺省用环境变量 ``OPENWSV_DATA_ROOT``，再缺省
        ``.``。pairs 数据集根目录可用 ``OPENWSV_DATASET_DIR`` 覆盖。

    Returns
    -------
    list[dict]
        每个模型一份完整结果 JSON（内容与落盘文件相同）。
    """
    if suite not in SUITES:
        raise ValueError(f"未知 suite {suite!r}；可用: {', '.join(SUITES)}")
    specs = _as_specs(models)
    for spec in specs:
        if "id" not in spec:
            raise ValueError(f"model spec 缺 id: {spec!r}")
        spec.setdefault("data_root", data_root)

    tasks = load_suite(suite)
    if limit is not None:
        if int(limit) < 0:
            raise ValueError(f"limit 必须 >= 0，实际 {limit}")
        tasks = tasks[: int(limit)]

    framework = framework_block(suite)
    detect = None
    if suite == "pairs":
        # 检测器属于框架：整轮只加载一次，所有结构化模型共用同一份输出
        detect, detector_meta = _load_detector()
        if detector_meta is not None:
            framework["detector"] = detector_meta
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for spec in specs:
        if suite == "pairs":
            payload = _run_pairs_suite(suite, tasks, spec, detect=detect)
        else:
            payload = _run_text_suite(suite, tasks, spec)
        result = {
            "framework": framework,
            "suite": suite,
            "n": len(tasks),
            "model": payload["model"],
            "metrics": payload["metrics"],
            "per_task": payload["per_task"],
        }
        path = out / f"{suite}__{_safe_name(str(spec['id']))}.json"
        _write_json(path, result)
        results.append(result)
        print(
            json.dumps(
                {
                    "ok": True,
                    "suite": suite,
                    "model": result["model"]["id"],
                    "n": len(tasks),
                    "metrics": result["metrics"],
                    "out": str(path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if len(results) == 1:
        _write_json(out / "result.json", results[0])
    return results


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")