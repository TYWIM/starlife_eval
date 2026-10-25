"""Derive the T4 ``change`` and T6 ``state_replay`` text suites from the frozen pairs GT.

The pairs suite (``pairs_test.jsonl``) ships a structured before/after state for
each of the 1,000 test pairs.  Both new suites are *text projections* of exactly
that GT, so a text VLM and a structured world-state model can be compared on the
same underlying pairs (same images, same after-state facts).

Design decisions
----------------
* One text task per pair, so the new suites stay paired with ``pairs`` 1:1.
* T4 asks about the change of one object (changed slot if any, otherwise an
  unchanged object -> "It stayed the same.").
* T6 asks one after-state fact per pair, cycling through four probe kinds:
  existence, relation type, relative position, contact.  The after-state answers
  are the "replayed" state facts, i.e. what the world looks like after the edit.
* Only objects whose ``type_key`` is unique inside the pair are used as referents:
  object descriptions are type names only, so a duplicated type would make the
  question ambiguous.
* Options are rotated deterministically (seed 20261020) so that the answer
  index is balanced and not correlated with label order.

This script is deterministic and read-only w.r.t. the frozen pairs file; rerunning
it reproduces byte-identical outputs (sorted, ``\\n`` line endings, UTF-8).

Usage::

    python scripts/wsv/derive_text_suites.py
    python scripts/wsv/derive_text_suites.py --check   # fail if outputs drift
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = REPO_ROOT / "src" / "starlife_eval" / "wsv" / "suites" / "pairs_test.jsonl"
DEFAULT_OUT = REPO_ROOT / "src" / "starlife_eval" / "wsv" / "suites"
DEFAULT_IMAGE_PREFIX = "openwsv-data-1.0/data_v15/dataset_spatial_v3"

SEED = 20261020
N_PAIRS = 1000
N_T6 = 1000
T6_PER_KIND = N_T6 // 4

RELATION_LABELS = {
    "on": "on",
    "contains": "contains",
    "held_by": "held by",
    "in": "in",
    "near": "near",
    "next_to": "next to",
    "part_of": "part of",
    "belongs_to": "belongs to",
    "facing": "facing",
    "above": "above",
    "below": "below",
}

REL_POS_LABELS = {
    "top": "top",
    "inside": "inside",
    "front": "front",
    "back": "back",
    "left": "left",
    "right": "right",
    "front_left": "front left",
    "front_right": "front right",
    "back_left": "back left",
    "back_right": "back right",
}

CHANGE_OPTIONS = ["It appeared.", "It disappeared.", "It was displaced.", "It stayed the same."]
CHANGE_INDEX = {"appeared": 0, "disappeared": 1, "displaced": 2, "none": 3}

T4_DATASET = "dataset_spatial_v3"
T4_CAPABILITY = "T4"
T6_CAPABILITY = "T6"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt input
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{lineno}: expected JSON object")
        rows.append(row)
    return rows


def _phrase(type_key: str) -> str:
    return str(type_key).replace("_", " ").strip()


def _unique_slots(row: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    """slot -> entity for entities whose type occurs exactly once in the pair."""
    counts = collections.Counter(str(e["type_key"]) for e in row.get("entities") or ())
    return {
        int(e["slot"]): e
        for e in row.get("entities") or ()
        if counts[str(e["type_key"])] == 1
    }


def _image_paths(row: Mapping[str, Any], prefix: str) -> list[str]:
    return [
        f"{prefix}/{row['before']}",
        f"{prefix}/{row['after']}",
    ]


def _rotate(choices: Sequence[str], answer: int, offset: int) -> tuple[list[str], int]:
    offset %= len(choices)
    rotated = list(choices[offset:]) + list(choices[:offset])
    return rotated, (int(answer) - offset) % len(choices)


def _task(
    *,
    task_id: str,
    suite: str,
    dataset: str,
    capability: str,
    question: str,
    images: list[str],
    choices: list[str],
    answer: int,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "suite": suite,
        "dataset": dataset,
        "capability": capability,
        "answer_type": "choice",
        "split": "test",
        "question": question,
        "images": images,
        "choices": list(choices),
        "answer": int(answer),
        "meta": dict(meta),
    }


# --------------------------------------------------------------------- T4


def derive_change(rows: Sequence[Mapping[str, Any]], prefix: str, seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row["pair_id"])
        unique = _unique_slots(row)
        changes = [c for c in row.get("changes") or () if int(c["slot"]) in unique]
        if changes:
            change = rng.choice(changes)
            slot = int(change["slot"])
            change_type = str(change["type"])
            target = unique[slot]
        else:
            present = [e for e in unique.values() if bool(e.get("present_after", True))]
            target = rng.choice(present)
            slot = int(target["slot"])
            change_type = "none"
        entity_type = _phrase(str(target["type_key"]))
        question = f"Between the two frames, what happened to the {entity_type}?"
        offset = len(tasks) % len(CHANGE_OPTIONS)
        choices, answer = _rotate(CHANGE_OPTIONS, CHANGE_INDEX[change_type], offset)
        tasks.append(
            _task(
                task_id=f"change-{pair_id}",
                suite="change",
                dataset=T4_DATASET,
                capability=T4_CAPABILITY,
                question=question,
                images=_image_paths(row, prefix),
                choices=choices,
                answer=answer,
                meta={
                    "pair_id": pair_id,
                    "probe": "change",
                    "target_slot": slot,
                    "target_type": str(target["type_key"]),
                    "gt_change": change_type,
                    "present_after": bool(target.get("present_after", True)),
                    "negative": bool(row.get("negative", False)),
                },
            )
        )
    return tasks


# --------------------------------------------------------------------- T6


def _candidate_relations(
    row: Mapping[str, Any], unique: Mapping[int, Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    return [
        rel
        for rel in row.get("relations") or ()
        if int(rel["from_slot"]) in unique and int(rel["to_slot"]) in unique
    ]


def _relation_question(rel: Mapping[str, Any], unique: Mapping[int, Mapping[str, Any]]) -> tuple[str, str, str]:
    a = _phrase(str(unique[int(rel["from_slot"])]["type_key"]))
    b = _phrase(str(unique[int(rel["to_slot"])]["type_key"]))
    # direction matters for asymmetric relation types (part_of / belongs_to / contains),
    # so the question must read "A related to B" (A = from_slot, B = to_slot).
    return a, b, f"In the second frame, how is the {a} related to the {b}?"


def _relpos_question(rel: Mapping[str, Any], unique: Mapping[int, Mapping[str, Any]]) -> tuple[str, str, str]:
    a = _phrase(str(unique[int(rel["from_slot"])]["type_key"]))
    b = _phrase(str(unique[int(rel["to_slot"])]["type_key"]))
    return a, b, f"In the second frame, where is the {a} relative to the {b}?"


def _contact_question(rel: Mapping[str, Any], unique: Mapping[int, Mapping[str, Any]]) -> tuple[str, str, str]:
    a = _phrase(str(unique[int(rel["from_slot"])]["type_key"]))
    b = _phrase(str(unique[int(rel["to_slot"])]["type_key"]))
    return a, b, f"In the second frame, does the {a} touch the {b}?"


def derive_state_replay(rows: Sequence[Mapping[str, Any]], prefix: str, seed: int = SEED) -> list[dict[str, Any]]:
    """Build exactly 1,000 T6 tasks, 250 per probe kind, one task per pair.

    Pair allocation is deterministic: existing objects first serve existence (125
    present + 125 absent), then relations serve type / rel_pos / contact probes.
    """
    rng = random.Random(seed + 1)
    assert len(rows) == N_PAIRS
    unique_by_pair = {i: _unique_slots(row) for i, row in enumerate(rows)}

    absent_pairs = [
        i
        for i, row in enumerate(rows)
        if any(
            (not bool(e.get("present_after", True))) and int(e["slot"]) in unique_by_pair[i]
            for e in row.get("entities") or ()
        )
    ]
    rng.shuffle(absent_pairs)
    used: set[int] = set()
    kinds: dict[int, str] = {}
    existence_bucket: dict[int, bool] = {}  # True -> ask about a present object, False -> absent
    for i in absent_pairs[: T6_PER_KIND // 2]:
        kinds[i] = "existence"
        existence_bucket[i] = False
        used.add(i)
    remaining = [i for i in range(N_PAIRS) if i not in used]
    rng.shuffle(remaining)
    present_pairs = [
        i
        for i in remaining
        if any(
            bool(e.get("present_after", True)) and int(e["slot"]) in unique_by_pair[i]
            for e in rows[i].get("entities") or ()
        )
    ]
    for i in present_pairs[: T6_PER_KIND // 2]:
        kinds[i] = "existence"
        existence_bucket[i] = True
        used.add(i)

    remaining = [i for i in range(N_PAIRS) if i not in used]
    rng.shuffle(remaining)
    relation_pairs = [i for i in remaining if _candidate_relations(rows[i], unique_by_pair[i])]
    assert len(relation_pairs) >= 3 * T6_PER_KIND, (len(relation_pairs), 3 * T6_PER_KIND)
    for i in relation_pairs[:T6_PER_KIND]:
        kinds[i] = "relation_type"
    for i in relation_pairs[T6_PER_KIND : 2 * T6_PER_KIND]:
        kinds[i] = "rel_pos"
    for i in relation_pairs[2 * T6_PER_KIND : 3 * T6_PER_KIND]:
        kinds[i] = "contact"
    used.update(relation_pairs[: 3 * T6_PER_KIND])
    assert len(kinds) == N_T6 and used == set(range(N_PAIRS))

    rel_type_vocab = sorted({str(rel["type"]) for row in rows for rel in row.get("relations") or ()})
    rel_pos_vocab = sorted(
        {str(rel["rel_pos"]) for row in rows for rel in row.get("relations") or () if rel.get("rel_pos") is not None}
    )
    assert set(rel_type_vocab) <= set(RELATION_LABELS)
    assert set(rel_pos_vocab) <= set(REL_POS_LABELS)

    tasks: list[dict[str, Any]] = []
    per_kind = collections.Counter()
    for i, row in enumerate(rows):
        kind = kinds[i]
        pair_id = str(row["pair_id"])
        unique = unique_by_pair[i]
        images = _image_paths(row, prefix)
        meta: dict[str, Any] = {"pair_id": pair_id, "probe": kind}
        if kind == "existence":
            # choose the unique entity matching the allocated bucket (absent / present)
            bucket = existence_bucket[i]
            candidates = [
                e
                for e in unique.values()
                if bool(e.get("present_after", True)) is bucket
            ]
            entity = rng.choice(candidates)
            etype = _phrase(str(entity["type_key"]))
            question = f"In the second frame, is the {etype} present?"
            present = bool(entity.get("present_after", True))
            offset = len(tasks) % 2
            choices, answer = _rotate(["Yes.", "No."], 0 if present else 1, offset)
            meta.update(
                {
                    "target_slot": int(entity["slot"]),
                    "target_type": str(entity["type_key"]),
                    "gt_present_after": present,
                }
            )
        else:
            candidates = _candidate_relations(row, unique)
            rel = rng.choice(candidates)
            a, b, question = (
                _relation_question(rel, unique) if kind == "relation_type"
                else _relpos_question(rel, unique) if kind == "rel_pos"
                else _contact_question(rel, unique)
            )
            meta.update(
                {
                    "from_slot": int(rel["from_slot"]),
                    "to_slot": int(rel["to_slot"]),
                    "from_type": str(unique[int(rel["from_slot"])]["type_key"]),
                    "to_type": str(unique[int(rel["to_slot"])]["type_key"]),
                    "relation_type": str(rel["type"]),
                    "rel_pos": rel.get("rel_pos"),
                    "contact": bool(rel.get("contact", False)),
                }
            )
            if kind == "relation_type":
                gt_label = RELATION_LABELS[str(rel["type"])]
                distractors = rng.sample(
                    [RELATION_LABELS[t] for t in rel_type_vocab if RELATION_LABELS[t] != gt_label], 3
                )
                choices = [gt_label] + distractors
                answer = 0
                meta["gt_relation_label"] = gt_label
            elif kind == "rel_pos":
                gt_label = REL_POS_LABELS[str(rel["rel_pos"])]
                distractors = rng.sample(
                    [REL_POS_LABELS[t] for t in rel_pos_vocab if REL_POS_LABELS[t] != gt_label], 3
                )
                choices = [gt_label] + distractors
                answer = 0
                meta["gt_rel_pos_label"] = gt_label
            else:
                choices = ["Yes.", "No."]
                answer = 0 if bool(rel.get("contact", False)) else 1
                meta["gt_contact"] = bool(rel.get("contact", False))
            offset = len(tasks) % len(choices)
            choices, answer = _rotate(choices, answer, offset)
        per_kind[kind] += 1
        tasks.append(
            _task(
                task_id=f"state-{pair_id}",
                suite="state_replay",
                dataset=T4_DATASET,
                capability=T6_CAPABILITY,
                question=question,
                images=images,
                choices=choices,
                answer=answer,
                meta=meta,
            )
        )
    assert per_kind["existence"] == per_kind["relation_type"] == per_kind["rel_pos"] == per_kind["contact"] == T6_PER_KIND
    return tasks


# ------------------------------------------------------------------ output


def _render(tasks: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(task, ensure_ascii=False, sort_keys=False) + "\n" for task in tasks)


def _dump(tasks: Iterable[Mapping[str, Any]], path: Path) -> tuple[str, int]:
    text = _render(tasks)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text.splitlines())


def _summary(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answers = collections.Counter(int(t["answer"]) for t in tasks)
    meta_keys = collections.Counter(t["meta"].get("probe", "change") for t in tasks)
    return {
        "n": len(tasks),
        "answer_index": dict(sorted(answers.items())),
        "probe_kind": dict(sorted(meta_keys.items())),
        "images": len({p for t in tasks for p in t["images"]}),
        "choices": dict(sorted(collections.Counter(len(t["choices"]) for t in tasks).items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive T4/T6 text suites from the frozen pairs GT")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS), help="frozen pairs_test.jsonl")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="output suites directory")
    parser.add_argument("--image-prefix", default=DEFAULT_IMAGE_PREFIX,
                        help="image path prefix relative to the data root")
    parser.add_argument("--check", action="store_true", help="verify existing outputs match (no write)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = _read_jsonl(Path(args.pairs))
    if len(rows) != N_PAIRS:
        raise SystemExit(f"expected {N_PAIRS} pairs, got {len(rows)}")

    change = derive_change(rows, args.image_prefix)
    replay = derive_state_replay(rows, args.image_prefix)
    outputs = [("change_test.jsonl", change), ("state_replay_test.jsonl", replay)]

    out_dir = Path(args.out_dir)
    drift = False
    for name, tasks in outputs:
        path = out_dir / name
        if args.check:
            expected = _render(tasks)
            actual = path.read_text(encoding="utf-8") if path.exists() else ""
            digest = hashlib.sha256(actual.encode("utf-8")).hexdigest()
            n = len(actual.splitlines())
            if actual != expected:
                drift = True
        else:
            digest, n = _dump(tasks, path)
        if n != len(tasks):
            drift = True
        print(json.dumps(
            {"file": name, "n": n, "sha256": digest, "summary": _summary(tasks), "check": bool(args.check)},
            ensure_ascii=False,
        ))
    if args.check:
        print(json.dumps({"drift": drift}, ensure_ascii=False))
    return 1 if (args.check and drift) else 0


if __name__ == "__main__":
    raise SystemExit(main())
