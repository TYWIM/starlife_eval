"""v2 runner: query any OpenAI-compatible multimodal model over v2 tasks, cache raw replies,
write predictions JSONL and a scored report. Resumable: completed task ids are skipped.

Usage:
    python -m starlife_eval.v2 run --tasks data/tasks.jsonl --model model.json --out runs/v2
    python -m starlife_eval.v2 score --tasks data/tasks.jsonl --pred runs/v2/predictions.jsonl --out runs/v2 --model my-model
    python -m starlife_eval.v2 compare --tasks data/tasks.jsonl --pred-a a.jsonl --pred-b b.jsonl
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from .prompts import PROMPT_TEMPLATES, build_prompt, parse_choice, parse_json_answer, parse_short
from .schema import Prediction, Task, load_predictions, load_tasks, write_predictions
from .scoring import mcnemar, score_run, write_report_md


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing API key: set environment variable {name}")
    return value


def _image_part(path_or_url: str) -> dict[str, Any]:
    if path_or_url.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": path_or_url}}
    data = Path(path_or_url).read_bytes()
    b64 = base64.b64encode(data).decode()
    suffix = Path(path_or_url).suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}


def _call(model: Mapping[str, Any], task: Task, template: str, timeout: float = 180.0) -> tuple[str, float]:
    content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(task, template)}]
    for img in task.images:
        content.append(_image_part(img))
    body = {
        "model": model["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": float(model.get("temperature", 0)),
        "max_tokens": int(model.get("max_tokens", 2000)),
    }
    req = urllib.request.Request(
        model["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + _env(model.get("api_key_env", "V2_API_KEY")),
                 "Content-Type": "application/json", "User-Agent": model.get("user_agent", "starlife-eval-v2")},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
    latency = time.time() - t0
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    return str(text), latency


def _parse(task: Task, raw: str, template: str) -> Prediction:
    if task.answer_type == "choice":
        return Prediction(task_id=task.task_id, choice=parse_choice(raw, len(task.choices)),
                          raw=raw, template=template)
    if task.answer_type == "short":
        return Prediction(task_id=task.task_id, text=parse_short(raw), raw=raw, template=template)
    return Prediction(task_id=task.task_id, value=parse_json_answer(raw), raw=raw, template=template)


def cmd_run(args: Any) -> int:
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    spec = json.loads(Path(args.model).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    cache_path = out / "raw_responses.jsonl"
    done = load_predictions(pred_path) if pred_path.exists() else {}
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["task_id"]] = rec

    preds: list[Prediction] = list(done.values())
    with cache_path.open("a", encoding="utf-8") as cf:
        for i, task in enumerate(tasks):
            if task.task_id in done:
                continue
            template = PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)]
            try:
                raw, latency = _call(spec, task, template, timeout=float(spec.get("timeout_s", 180)))
                pred = _parse(task, raw, template)
                pred = Prediction(**{**pred.to_dict(), "latency_s": latency})
                cf.write(json.dumps({"task_id": task.task_id, "template": template, "raw": raw,
                                     "latency_s": latency}, ensure_ascii=False) + "\n")
                cf.flush()
            except Exception as exc:  # noqa: BLE001 —— 单题失败不拖垮整轮
                pred = Prediction(task_id=task.task_id, error=f"{type(exc).__name__}: {exc}",
                                  template=template)
            preds.append(pred)
            write_predictions(preds, pred_path)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(tasks)} tasks", flush=True)

    preds_map = {p.task_id: p for p in preds}
    report = score_run(args.model_name or spec.get("name", spec.get("model", "model")), tasks, preds_map,
                       meta={"dataset": spec.get("dataset", "v2"), "protocol_version": "v2"}).report()
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_md(report, out / "report.md")
    print(json.dumps({"ok": True, "out": str(out), "accuracy": report["accuracy"],
                      "ci95": report["ci95"], "n": report["n_tasks"]}, ensure_ascii=False))
    return 0


def cmd_score(args: Any) -> int:
    tasks = load_tasks(args.tasks)
    preds = load_predictions(args.pred)
    report = score_run(args.model_name or "model", tasks, preds,
                       meta={"pred_file": str(args.pred), "protocol_version": "v2"}).report()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_md(report, out / "report.md")
    print(json.dumps({"ok": True, "accuracy": report["accuracy"], "ci95": report["ci95"]}, ensure_ascii=False))
    return 0


def cmd_compare(args: Any) -> int:
    tasks = load_tasks(args.tasks)
    a = score_run("a", tasks, load_predictions(args.pred_a))
    b = score_run("b", tasks, load_predictions(args.pred_b))
    test = mcnemar(a.flags, b.flags)
    print(json.dumps({"a_accuracy": round(float(sum(a.flags)) / len(a.flags), 4),
                      "b_accuracy": round(float(sum(b.flags)) / len(b.flags), 4),
                      "mcnemar": test}, ensure_ascii=False, indent=2))
    return 0
