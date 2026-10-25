"""OpenWSV gateway grid runner：串行、可续跑的（suite × model）批量跑分。

这是 T4–T7「其他模型」跑分（SPEC_T4_T7 §3）的编排层，**只包 CLI、不改框架**：

* 单个 ``python -m starlife_eval.wsv run`` 子进程、``--workers 3``（硬上限 3）；
* 所有 (suite, model) 组合**严格串行**，同一时刻最多一个子进程、最多 3 个在途请求；
* 结果已落盘且 ``n`` 与预期一致时跳过（可续跑，中断后重跑同一条命令即可）；
* 密钥只从环境变量读入子进程 env（``OPENWSV_API_KEY``），不进命令行/日志/结果 JSON；
* 每个组合一份日志 ``{out}/logs/{suite}__{model}.log``，墙钟耗时记在
  ``{out}/logs/timings.json``；
* 每完成一次全量运行重写 ``{out}/SUMMARY.md``（accuracy / coverage / 95% CI / n /
  墙钟 / Top 错误类别），便于 Lead 边跑边读。

全新 suite 尚未在 ``tasks.py`` 注册、或数据根下图片缺失时，**先整体 preflight 再启动**，
以退出码 2 优雅失败并打印原因（避免跑到一半才炸）。

用法::

    # 计划（不跑）：检查 suite 是否已注册、图片是否可解析
    python scripts/wsv/run_gateway.py --dry-run

    # 冒烟：每个 suite × model 只跑 10 题，结果写到 smoke/ 子目录（不覆盖全量）
    python scripts/wsv/run_gateway.py --limit 10 --models cn:glm-5.3-flash \
        --out-dir reports/openwsv-v2/smoke --no-summary

    # 全量网格（默认 model 外层 → suite 内层：一个模型跑完 4 套再下一个）
    python scripts/wsv/run_gateway.py --data-root data

    # 只重算汇总（不跑模型、不需要密钥）
    python scripts/wsv/run_gateway.py --summary-only

默认 roster（Lead 最终决定，4 个 fast 模型）:
``cn:glm-5.3-flash`` → ``cn:deepseek-v4.1-flash`` → ``cn:minimax-m3.1-flash`` →
``cn:mimo-v2.6-pro``；每个模型内 suite 顺序:
``change`` → ``state_replay`` → ``relation_change`` → ``cross_frame``
（合计 4 × (1000+1000+600+400) = 12,000 次请求）。

结果有效性：resume 跳过时会校验结果 JSON 的 ``framework.dataset_sha256`` 是否等于
当前冻结 suite 文件 sha256；suite 重新生成后旧结果不会被当作已完成（SUMMARY 标
``stale-sha`` 并重跑）。

环境变量
--------
``OPENWSV_API_KEY``（回退 ``WSV_API_KEY`` / ``V2_API_KEY``）：网关密钥；仅注入子进程 env。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from starlife_eval.wsv.tasks import (  # noqa: E402
    SUITE_SIZES,
    SUITES,
    dataset_sha256,
    load_suite,
)

#: 本阶段 T4–T7 的 suite 顺序（Lead 决定：单个模型内 change → state_replay →
#: relation_change → cross_frame；配合 --order model 即「一个模型跑完 4 套再下一个」）
TARGET_SUITES: tuple[str, ...] = ("change", "state_replay", "relation_change", "cross_frame")

#: 模型顺序（Lead 最终 roster：4 个 fast 模型；cn:step-5-preview / cn:kimi-k3.1-pro
#: 已剔除；解析上等价于服务端 model 名，文件名里 ':' 会被清洗）
DEFAULT_MODELS: tuple[str, ...] = (
    "cn:glm-5.3-flash",
    "cn:deepseek-v4.1-flash",
    "cn:minimax-m3.1-flash",
    "cn:mimo-v2.6-pro",
)

DEFAULT_BASE_URL = "http://127.0.0.1:7864/v1"
DEFAULT_OUT_DIR = _REPO / "reports" / "openwsv-v2"
DEFAULT_DATA_ROOT = "data"

#: 网关并发硬上限（SPEC §3）：单子进程 --workers <= 3，且不同模型不得并行
HARD_WORKER_CAP = 3

_KEY_ENVS: tuple[str, ...] = ("OPENWSV_API_KEY", "WSV_API_KEY", "V2_API_KEY")

#: 结果 JSON 契约里的文件名清洗规则，必须与 runner._safe_name 完全一致
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_ADAPTER_TIMEOUT_NOTE = "adapter timeout_s=180s (CLI 未暴露 --timeout), retries=2"


# --------------------------------------------------------------------- helpers


def safe_name(value: str) -> str:
    """与 ``runner._safe_name`` 一致：``cn:glm-5.3-flash`` -> ``cn_glm-5.3-flash``。"""
    return _SAFE_RE.sub("_", str(value)).strip("_") or "model"


def result_path(out_dir: Path, suite: str, model: str) -> Path:
    """结果 JSON 路径（与 runner 落盘位置逐字一致）。"""
    return out_dir / f"{suite}__{safe_name(model)}.json"


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +08:00")


def _expected_n(suite: str, limit: int | None) -> int:
    size = int(SUITE_SIZES[suite])
    return size if limit is None else min(int(limit), size)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(v) for v in values))


def _split_csv(values: Iterable[str]) -> list[str]:
    """``--suites change,state_replay`` 与 ``--suites change state_replay`` 都接受。"""
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return _dedupe(out)


class _Logger:
    """控制台 + 追加写到 ``{out}/logs/run_gateway.log``（不接触密钥）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"{_now()} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ------------------------------------------------------------------ validation


def _check_workers(workers: int) -> str | None:
    if workers < 1:
        return f"--workers 必须 >= 1，实际 {workers}"
    if workers > HARD_WORKER_CAP:
        return (
            f"--workers={workers} 超过网关硬并发上限 {HARD_WORKER_CAP}；"
            "请用 --workers 3（单进程）并保持模型串行"
        )
    return None


def _resolve_api_key(api_key_env: str | None) -> tuple[str | None, str]:
    names = _dedupe(([api_key_env] if api_key_env else []) + list(_KEY_ENVS))
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value, name
    return None, names[0]


def preflight_suite(
    suite: str,
    data_root: Path,
    limit: int | None,
) -> tuple[list[str], list[str]]:
    """加载 suite 并检查图片路径可解析；返回 ``(errors, missing_images)``。"""
    try:
        tasks = load_suite(suite)
    except (FileNotFoundError, ValueError) as exc:
        return [f"suite 数据不可用：{type(exc).__name__}: {exc}"], []
    if limit is not None:
        tasks = tasks[: int(limit)]
    missing: list[str] = []
    for task in tasks:
        for image in task.get("images") or ():
            candidate = Path(str(image))
            if not candidate.is_absolute():
                candidate = data_root / candidate
            if not candidate.exists():
                missing.append(str(image))
    return [], missing


#: suite 当前冻结文件 sha256 缓存（单进程内文件不会变）
_SHA_CACHE: dict[str, str] = {}


def current_dataset_sha(suite: str) -> str | None:
    """返回 suite 当前冻结文件 sha256；文件缺失返回 ``None``。"""
    if suite not in _SHA_CACHE:
        try:
            _SHA_CACHE[suite] = dataset_sha256(suite)
        except (FileNotFoundError, ValueError):
            return None
    return _SHA_CACHE[suite]


def _is_complete(path: Path, suite: str, expected_n: int) -> tuple[bool, str]:
    """结果可解析、n/per_task 等于预期、**且 framework.dataset_sha256 与当前冻结文件一致**。

    最后一条是防呆：suite 被重新生成（sha 变化）后，旧结果不能被 resume 逻辑当成
    "已完成"跳过，必须重跑；旧 sha 的结果在 SUMMARY.md 里也会标成 ``stale-sha``。
    """
    if not path.exists():
        return False, "missing result"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable result: {type(exc).__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return False, "result is not a JSON object"
    n = payload.get("n")
    per_task = payload.get("per_task")
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        return False, f"invalid n={n!r}"
    if n_int != expected_n:
        return False, f"n={n_int} != expected {expected_n}"
    if not isinstance(per_task, list) or len(per_task) != expected_n:
        got = len(per_task) if isinstance(per_task, list) else "non-list"
        return False, f"per_task length {got} != expected {expected_n}"
    expected_sha = current_dataset_sha(suite)
    if expected_sha is not None:
        framework = payload.get("framework") or {}
        got = framework.get("dataset_sha256") if isinstance(framework, Mapping) else None
        if got != expected_sha:
            got_text = (str(got)[:8] + "…") if got else "missing"
            return False, f"stale dataset_sha256 {got_text} != current {expected_sha[:8]}…"
    return True, "complete (sha ok)"


# ---------------------------------------------------------------------- runner


def _tail(path: Path, lines: int = 15) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(text[-lines:])


def _run_one(
    *,
    suite: str,
    model: str,
    out_dir: Path,
    log_path: Path,
    data_root: str,
    base_url: str,
    workers: int,
    max_tokens: int,
    temperature: float,
    limit: int | None,
    timeout_s: float,
    api_key: str,
) -> dict[str, Any]:
    """跑一次 CLI 子进程；返回 ``{rc, wall_s, error, tail}``（不抛）。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-X", "utf8", "-m", "starlife_eval.wsv", "run",
        "--suite", suite,
        "--model", model,
        "--adapter", "openai",
        "--base-url", base_url,
        "--max-tokens", str(int(max_tokens)),
        "--temperature", str(float(temperature)),
        "--workers", str(int(workers)),
        "--data-root", data_root,
        "--out", str(out_dir),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(int(limit))])

    env = dict(os.environ)
    env["OPENWSV_API_KEY"] = api_key  # 只进 env；命令行/日志/结果 JSON 均不出现
    env["PYTHONIOENCODING"] = "utf-8"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_REPO / "src") + (os.pathsep + existing if existing else "")

    started = time.perf_counter()
    error: str | None = None
    rc: int | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"# {_now()} suite={suite} model={model}\n")
            log.write(f"# cwd={_REPO}\n# cmd={' '.join(cmd)}\n")
            log.write(f"# data_root={data_root} base_url={base_url} workers={workers}\n")
            log.write(f"# {_ADAPTER_TIMEOUT_NOTE}\n\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            rc = int(proc.returncode)
            log.write(f"\n# exit={rc} wall_s={time.perf_counter() - started:.2f}\n")
    except subprocess.TimeoutExpired:
        rc = -9
        error = f"subprocess timeout after {timeout_s:.0f}s"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n# TIMEOUT after {timeout_s:.0f}s\n")
    except OSError as exc:
        rc = -1
        error = f"spawn failed: {type(exc).__name__}: {exc}"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n# SPAWN FAILED {error}\n")

    wall_s = time.perf_counter() - started
    if rc != 0 and error is None:
        error = f"exit code {rc}"
    return {
        "rc": rc,
        "wall_s": wall_s,
        "error": error,
        "tail": _tail(log_path),
    }


def _record_timing(out_dir: Path, key: str, record: Mapping[str, Any]) -> None:
    path = out_dir / "logs" / "timings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload[key] = {**dict(record), "updated_at": _now()}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


# --------------------------------------------------------------------- summary


def _error_class(message: Any) -> str:
    text = str(message or "unknown error").strip()
    if text.startswith("HTTP "):
        parts = text.split()
        return "HTTP " + parts[1].rstrip(":") if len(parts) > 1 else "HTTP error"
    if text.startswith("unparsable choice"):
        return "unparsable choice"
    if text.startswith("empty response"):
        return "empty response"
    if text.startswith("missing prediction"):
        return "missing prediction"
    head = text.split(":")[0].strip()
    return head[:48] or "error"


def _error_counts(per_task: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in per_task or ():
        if row.get("covered"):
            continue
        cls = _error_class(row.get("error"))
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _fmt_errors(counts: Mapping[str, int], top: int = 3) -> str:
    if not counts:
        return "—"
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return "; ".join(f"{name}×{count}" for name, count in ordered)


def _fmt_ci(ci: Any) -> str:
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except (TypeError, ValueError, IndexError):
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def _fnum(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _collect_rows(
    out_dir: Path,
    suites: Sequence[str],
    models: Sequence[str],
) -> list[dict[str, Any]]:
    timings: dict[str, Any] = {}
    timing_path = out_dir / "logs" / "timings.json"
    if timing_path.exists():
        try:
            loaded = json.loads(timing_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                timings = loaded
        except (OSError, json.JSONDecodeError):
            timings = {}

    rows: list[dict[str, Any]] = []
    for model in models:
        for suite in suites:
            key = f"{suite}__{safe_name(model)}"
            path = result_path(out_dir, suite, model)
            row: dict[str, Any] = {
                "model": model,
                "suite": suite,
                "key": key,
                "path": path,
                "state": "pending",
                "n": None,
                "n_covered": None,
                "accuracy": None,
                "coverage": None,
                "ci95": None,
                "errors": {},
                "wall_s": None,
            }
            timing = timings.get(key) or {}
            if isinstance(timing, Mapping) and timing.get("wall_s") is not None:
                row["wall_s"] = float(timing["wall_s"])
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, Mapping):
                    metrics = payload.get("metrics") or {}
                    per_task = payload.get("per_task") or []
                    covered = sum(1 for item in per_task
                                  if isinstance(item, Mapping) and item.get("covered"))
                    framework = payload.get("framework") or {}
                    got_sha = framework.get("dataset_sha256") if isinstance(framework, Mapping) else None
                    row.update(
                        state="done",
                        n=int(payload.get("n") or 0),
                        n_covered=covered,
                        accuracy=metrics.get("accuracy"),
                        coverage=metrics.get("coverage"),
                        ci95=metrics.get("ci95"),
                        errors=_error_counts(per_task),
                        dataset_sha256=got_sha,
                    )
                    expected_sha = current_dataset_sha(suite)
                    if expected_sha is not None and got_sha != expected_sha:
                        row["state"] = "stale-sha"
                        row["error"] = (
                            f"stale dataset_sha256 {str(got_sha or 'missing')[:8]} != "
                            f"current {expected_sha[:8]}"
                        )
                else:
                    row["state"] = "unreadable"
            elif isinstance(timing, Mapping) and timing.get("status") == "failed":
                row["state"] = "failed"
                row["error"] = timing.get("error")
            rows.append(row)
    return rows


def write_summary(
    out_dir: Path,
    suites: Sequence[str],
    models: Sequence[str],
    *,
    args: argparse.Namespace | None = None,
    log: _Logger | None = None,
) -> Path:
    """根据已落盘结果重写 ``{out}/SUMMARY.md``（原子替换，可随时读）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect_rows(out_dir, suites, models)
    workers = int(getattr(args, "workers", HARD_WORKER_CAP) or HARD_WORKER_CAP)
    limit = getattr(args, "limit", None)
    data_root = getattr(args, "data_root", DEFAULT_DATA_ROOT)
    base_url = getattr(args, "base_url", DEFAULT_BASE_URL)

    done = [row for row in rows if row["state"] == "done"]
    failed = [row for row in rows if row["state"] == "failed"]
    stale = [row for row in rows if row["state"] == "stale-sha"]
    pending = [row for row in rows if row["state"] == "pending"]
    lookup = {(row["model"], row["suite"]): row for row in rows}

    lines: list[str] = ["# OpenWSV v2 — other-model grid (T4–T7)", ""]
    lines.append(f"- Generated: {_now()}")
    lines.append(f"- Endpoint: `{base_url}` (model id = served model name)")
    lines.append(f"- Data root: `{data_root}`")
    mode = "full suite" if limit is None else f"smoke limit={int(limit)}"
    lines.append(
        f"- Grid: {len(models)} models × {len(suites)} suites, {mode}; "
        f"single process, `--workers {workers}` (hard cap {HARD_WORKER_CAP}), models sequential"
    )
    lines.append(
        "- Params: temperature=0, max_tokens=4096; adapter retries=2, "
        "adapter timeout_s=180 (CLI 未暴露 --timeout)"
    )
    lines.append(
        "- Scoring: accuracy 只在覆盖题上算；coverage = 可解析且无调用错误的占比；"
        "95% CI = 任务级 bootstrap (seed 20261022)。coverage < 100% 的行不是最终分。"
    )
    lines.append(
        f"- Progress: {len(done)}/{len(rows)} runs done, {len(pending)} pending, "
        f"{len(failed)} failed, {len(stale)} stale-sha (must rerun)"
    )
    lines.append("")

    # 1) 准确率矩阵（accuracy，括号内 coverage / n）
    lines.append("## Accuracy matrix (accuracy; coverage; n)")
    lines.append("")
    header = "| model | " + " | ".join(suites) + " | macro |"
    divider = "| --- | " + " | ".join("---" for _ in suites) + " | --- |"
    lines.append(header)
    lines.append(divider)
    for model in models:
        cells: list[str] = []
        accs: list[float] = []
        for suite in suites:
            row = lookup.get((model, suite))
            if row is None or row["state"] != "done":
                cells.append(row["state"] if row and row["state"] in ("failed", "stale-sha") else "pending")
                continue
            cells.append(
                f"{_fnum(row['accuracy'])} ({_fnum(row['coverage'], 3)}; n={row['n']})"
            )
            try:
                accs.append(float(row["accuracy"]))
            except (TypeError, ValueError):
                pass
        macro = f"{sum(accs) / len(accs):.4f} ({len(accs)} suite)" if accs else "—"
        lines.append(f"| `{model}` | " + " | ".join(cells) + f" | {macro} |")
    lines.append("")

    # 2) 每行明细
    lines.append("## Per-run detail")
    lines.append("")
    lines.append(
        "| model | suite | n | covered | accuracy | 95% CI | coverage | wall (s) | ≈req (s) | top errors |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |")
    for row in rows:
        wall = f"{row['wall_s']:.1f}" if row["wall_s"] is not None else "—"
        per_req = "—"
        if row["wall_s"] is not None and row["n"]:
            per_req = f"{row['wall_s'] * workers / row['n']:.2f}"
        if row["state"] == "done":
            lines.append(
                f"| `{row['model']}` | {row['suite']} | {row['n']} | {row['n_covered']} | "
                f"{_fnum(row['accuracy'])} | {_fmt_ci(row['ci95'])} | "
                f"{_fnum(row['coverage'], 3)} | {wall} | {per_req} | {_fmt_errors(row['errors'])} |"
            )
        else:
            note = row["state"]
            if row.get("error"):
                note = f"{note}: {str(row['error'])[:80]}"
            lines.append(
                f"| `{row['model']}` | {row['suite']} | — | — | — | — | — | {wall} | {per_req} | {note} |"
            )
    lines.append("")

    # 3) 吞吐投影：支持 Lead 的 8h / 2.5 s/task 阈值判断（只看全量行 wall/n）
    lines.append("## Throughput projection (effective s/task = wall_s / n, workers=3)")
    lines.append("")
    suite_sizes = {suite: int(SUITE_SIZES[suite]) for suite in suites if suite in SUITE_SIZES}
    per_suite_rates: dict[str, list[float]] = {}
    for row in done:
        if not row["n"] or row["wall_s"] is None:
            continue
        size = suite_sizes.get(row["suite"])
        if size is not None and int(row["n"]) != size:
            continue  # 冒烟行不参与投影
        per_suite_rates.setdefault(row["suite"], []).append(float(row["wall_s"]) / int(row["n"]))
    all_rates = [rate for rates in per_suite_rates.values() for rate in rates]
    if suite_sizes and (all_rates or per_suite_rates):
        lines.append("| suite | n | full runs measured | effective s/task | projected total (min) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        projected_total_s = 0.0
        for suite, size in suite_sizes.items():
            rates = per_suite_rates.get(suite) or []
            rate = (sum(rates) / len(rates)) if rates else (
                sum(all_rates) / len(all_rates) if all_rates else None
            )
            projected = (rate * size * len(models)) if rate is not None else None
            if projected is not None:
                projected_total_s += projected
            rate_text = _fnum(rate, 2) if rate is not None else "—"
            projected_text = f"{projected / 60.0:.1f}" if projected is not None else "—"
            lines.append(
                f"| {suite} | {size} | {len(rates)} | {rate_text} | {projected_text} |"
            )
        if projected_total_s:
            hours = projected_total_s / 3600.0
            lines.append("")
            flag = (
                f"WARNING: projected {hours:.1f} h > 8 h threshold — message Lead before continuing"
                if hours > 8.0 else
                f"projected {hours:.1f} h <= 8 h threshold"
            )
            lines.append(f"- Grid projection ({len(models)} models × {sum(suite_sizes.values())} tasks): "
                         f"**{hours:.1f} h** — {flag}")
    else:
        lines.append("- waiting for the first full run to measure effective s/task")
    lines.append("")

    # 4) 异常
    anomalies: list[str] = []
    for row in done:
        try:
            coverage = float(row["coverage"])
        except (TypeError, ValueError):
            coverage = None
        if coverage is not None and coverage < 0.95:
            anomalies.append(
                f"coverage<95%: `{row['model']}` × {row['suite']} = {coverage:.3f} "
                f"(errors: {_fmt_errors(row['errors'])})"
            )
        if row["errors"].get("unparsable choice") and row["coverage"]:
            if row["errors"]["unparsable choice"] > 0.05 * row["n"]:
                anomalies.append(
                    f"parse rate low: `{row['model']}` × {row['suite']} "
                    f"unparsable={row['errors']['unparsable choice']}/{row['n']}"
                )
    for row in failed:
        anomalies.append(
            f"failed: `{row['model']}` × {row['suite']} — {str(row.get('error') or 'see log')[:120]}"
        )
    for row in stale:
        anomalies.append(
            f"STALE (old suite sha, must rerun): `{row['model']}` × {row['suite']} — "
            f"{str(row.get('error') or '')[:120]}"
        )
    lines.append("## Anomalies")
    lines.append("")
    if anomalies:
        lines.extend(f"- {item}" for item in anomalies)
    else:
        lines.append("- none")
    lines.append("")
    lines.append(
        "> Errors/parse failures count as **uncovered**, not wrong; do not use rows with "
        "coverage < 1 as final scores (AGENTS.md §4). See `logs/` for per-run stdout."
    )
    lines.append("")

    text = "\n".join(lines)
    tmp = out_dir / "SUMMARY.md.tmp"
    tmp.write_text(text, encoding="utf-8")
    final = out_dir / "SUMMARY.md"
    os.replace(tmp, final)
    if log is not None:
        log.log(f"[summary] wrote {final} ({len(done)}/{len(rows)} runs done)")
    return final


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/wsv/run_gateway.py",
        description=(
            "OpenWSV T4–T7 gateway grid runner: sequential (suite × model) CLI runs, "
            "resumable, workers<=3, API key from env only."
        ),
    )
    parser.add_argument("--suites", nargs="+", default=list(TARGET_SUITES),
                        help=f"suite 顺序，空格或逗号分隔（默认: {' '.join(TARGET_SUITES)}）;"
                             " 例如 --suites change,state_replay")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        help="模型顺序，空格或逗号分隔（默认: Lead 最终 roster 的 5 个）")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"OpenAI 兼容端点根 URL（默认 {DEFAULT_BASE_URL}）")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help=f"图像相对路径根目录（默认 {DEFAULT_DATA_ROOT}）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"结果/日志/汇总输出目录（默认 {DEFAULT_OUT_DIR}）")
    parser.add_argument("--limit", type=int, default=None,
                        help="每个 suite 只跑前 N 题（冒烟；跳过判定用 n==N）")
    parser.add_argument("--workers", type=int, default=HARD_WORKER_CAP,
                        help=f"单个子进程并发（默认且最大 {HARD_WORKER_CAP}，硬上限）")
    parser.add_argument("--max-tokens", type=int, default=4096, help="最大输出 token（默认 4096）")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度（默认 0）")
    parser.add_argument("--api-key-env", default=None,
                        help="密钥环境变量名（默认 OPENWSV_API_KEY，回退 WSV_API_KEY/V2_API_KEY）")
    parser.add_argument("--order", choices=("model", "suite"), default="model",
                        help="外层循环：model（默认，一个模型跑完 4 套再下一个）或 suite")
    parser.add_argument("--force", action="store_true", help="忽略已完成的文件，强制重跑")
    parser.add_argument("--allow-missing", action="store_true",
                        help="跳过尚未注册的 suite（默认整体优雅失败）")
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="preflight 发现缺图时仍继续（默认优雅失败）")
    parser.add_argument("--timeout", type=float, default=21600.0,
                        help="单个 (suite, model) 子进程超时秒数（默认 21600=6h）")
    parser.add_argument("--max-consecutive-failures", type=int, default=3,
                        help="连续失败达到该数则中止网格（默认 3）")
    parser.add_argument("--no-summary", action="store_true", help="不生成 SUMMARY.md")
    parser.add_argument("--summary-only", action="store_true",
                        help="只根据已落盘结果重写 SUMMARY.md 后退出（不跑模型、不需要密钥）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划与 preflight 结果，不跑模型")
    return parser


def _build_grid(suites: Sequence[str], models: Sequence[str], order: str) -> list[tuple[str, str]]:
    if order == "model":
        return [(suite, model) for model in models for suite in suites]
    return [(suite, model) for suite in suites for model in models]


def run_grid(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    log = _Logger(out_dir / "logs" / "run_gateway.log")
    suites = _split_csv(args.suites)
    models = _split_csv(args.models)

    if args.limit is not None and args.limit < 1:
        log.log(f"[fatal] --limit 必须 >= 1，实际 {args.limit}")
        return 2
    worker_error = _check_workers(args.workers)
    if worker_error:
        log.log(f"[fatal] {worker_error}")
        return 2
    if not suites or not models:
        log.log("[fatal] --suites / --models 不能为空")
        return 2

    if args.summary_only:
        path = write_summary(out_dir, suites, models, args=args, log=log)
        print(str(path))
        return 0

    missing_suites = [suite for suite in suites if suite not in SUITES]
    if missing_suites:
        log.log(
            f"[fatal] suite 尚未在冻结框架注册：{missing_suites}；"
            f"当前可用: {list(SUITES)}"
        )
        if not args.allow_missing:
            return 2
        suites = [suite for suite in suites if suite in SUITES]
        if not suites:
            return 2

    # preflight：注册表 + 数据文件 + 图片路径解析，全部通过才启动
    data_root = Path(args.data_root)
    preflight_ok = True
    for suite in suites:
        errors, missing_images = preflight_suite(suite, data_root, args.limit)
        if errors:
            preflight_ok = False
            log.log(f"[preflight] {suite}: FAIL — {'; '.join(errors)}")
            continue
        if missing_images:
            sample = ", ".join(_dedupe(missing_images)[:5])
            log.log(
                f"[preflight] {suite}: {len(_dedupe(missing_images))} 张图片无法在 "
                f"data_root={data_root} 下解析（例: {sample}）"
            )
            if not args.allow_missing_images:
                preflight_ok = False
        else:
            log.log(f"[preflight] {suite}: ok（n={_expected_n(suite, args.limit)}，图片可解析）")
    if not preflight_ok:
        log.log("[fatal] preflight 未通过，未启动任何模型调用。先修数据根/注册表再重跑。")
        return 2

    grid = _build_grid(suites, models, args.order)
    log.log(
        f"[plan] {len(grid)} runs = {len(suites)} suites × {len(models)} models, "
        f"order={args.order}, workers={args.workers}, data_root={data_root}, "
        f"out_dir={out_dir}, limit={args.limit if args.limit is not None else 'none'}"
    )
    if args.dry_run:
        for index, (suite, model) in enumerate(grid, start=1):
            path = result_path(out_dir, suite, model)
            complete, reason = _is_complete(path, suite, _expected_n(suite, args.limit))
            state = f"SKIP ({reason})" if complete and not args.force else "RUN"
            log.log(f"[dry-run {index}/{len(grid)}] {suite} × {model} -> {state}")
        return 0

    api_key, key_env = _resolve_api_key(args.api_key_env)
    if not api_key:
        log.log(
            f"[fatal] 未找到网关密钥：请设置环境变量 {key_env}"
            "（只从环境读，不写盘、不打印）"
        )
        return 3

    done = skipped = failed = 0
    consecutive = 0
    aborted = False
    failures: list[str] = []
    for index, (suite, model) in enumerate(grid, start=1):
        expected = _expected_n(suite, args.limit)
        path = result_path(out_dir, suite, model)
        log_path = out_dir / "logs" / f"{suite}__{safe_name(model)}.log"
        tag = f"[{index}/{len(grid)}] {suite} × {model}"

        if not args.force:
            complete, reason = _is_complete(path, suite, expected)
            if complete:
                skipped += 1
                log.log(f"{tag} SKIP (n={expected} {reason})")
                continue

        log.log(f"{tag} START expected_n={expected}")
        outcome = _run_one(
            suite=suite,
            model=model,
            out_dir=out_dir,
            log_path=log_path,
            data_root=args.data_root,
            base_url=args.base_url,
            workers=args.workers,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            limit=args.limit,
            timeout_s=args.timeout,
            api_key=api_key,
        )
        complete, reason = _is_complete(path, suite, expected)
        ok = outcome["rc"] == 0 and complete
        record: dict[str, Any] = {
            "suite": suite,
            "model": model,
            "wall_s": round(float(outcome["wall_s"]), 3),
            "rc": outcome["rc"],
            "status": "ok" if ok else "failed",
        }
        if ok:
            try:
                metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics") or {}
            except (OSError, json.JSONDecodeError):
                metrics = {}
            done += 1
            consecutive = 0
            record.update(n=expected, metrics=metrics)
            _record_timing(out_dir, f"{suite}__{safe_name(model)}", record)
            log.log(
                f"{tag} OK wall={outcome['wall_s']:.1f}s "
                f"acc={metrics.get('accuracy')} cov={metrics.get('coverage')} "
                f"ci95={metrics.get('ci95')} -> {path}"
            )
        else:
            failed += 1
            consecutive += 1
            detail = outcome["error"] or f"result not complete ({reason})"
            record.update(error=str(detail))
            _record_timing(out_dir, f"{suite}__{safe_name(model)}", record)
            failures.append(f"{suite} × {model}: {detail}")
            log.log(f"{tag} FAIL rc={outcome['rc']} {detail}")
            if outcome["tail"]:
                log.log("[tail]\n" + outcome["tail"])
            if consecutive >= max(1, int(args.max_consecutive_failures)):
                log.log(
                    f"[abort] 连续失败 {consecutive} 次（阈值 {args.max_consecutive_failures}），"
                    "停止网格（已完成的文件会保留，修好后重跑同一条命令即可续跑）"
                )
                aborted = True
                break

        if not args.no_summary and args.limit is None:
            write_summary(out_dir, suites, models, args=args, log=log)

    if not args.no_summary and args.limit is None:
        write_summary(out_dir, suites, models, args=args, log=log)

    log.log(
        f"[done] runs={len(grid)} ok={done} skipped={skipped} failed={failed} "
        f"aborted={aborted}"
    )
    if failures:
        for item in failures:
            log.log(f"[failed] {item}")
    if aborted:
        return 4
    return 0 if failed == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return run_grid(args)


if __name__ == "__main__":
    raise SystemExit(main())
