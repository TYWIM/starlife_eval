"""Prepare the OpenWSV data bundle: verify the release zip and unpack it.

Usage:
    python scripts/wsv/prepare_data.py --zip openwsv-data-1.0.zip --dest ./data
    python scripts/wsv/prepare_data.py --dest ./data          # skip zip, verify unpacked tree

The release zip sha256 is printed in the release notes; pass --expect-sha256 to fail
closed when the download does not match.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys
import tempfile
import zipfile

BUNDLE = "openwsv-data-1.0"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sums(root: pathlib.Path) -> tuple[int, list[str]]:
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise SystemExit(f"missing {sums}")
    bad: list[str] = []
    count = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        target = root / name
        count += 1
        if not target.is_file():
            bad.append(f"missing {name}")
        elif sha256_file(target) != expected:
            bad.append(f"hash mismatch {name}")
    return count, bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify and unpack the OpenWSV data bundle")
    ap.add_argument("--zip", type=pathlib.Path, help="openwsv-data-1.0.zip")
    ap.add_argument("--dest", type=pathlib.Path, default=pathlib.Path("data"))
    ap.add_argument("--expect-sha256", help="expected sha256 of the zip itself")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    if args.zip:
        if not args.zip.is_file():
            raise SystemExit(f"zip not found: {args.zip}")
        actual = sha256_file(args.zip)
        if args.expect_sha256 and actual != args.expect_sha256:
            raise SystemExit(f"zip sha256 mismatch: {actual} != {args.expect_sha256}")
        print(f"zip sha256: {actual}")
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(args.zip) as zf:
                zf.extractall(tmp)
            src = pathlib.Path(tmp) / BUNDLE
            if not src.is_dir():
                raise SystemExit(f"zip does not contain {BUNDLE}/")
            if (args.dest / BUNDLE).exists():
                shutil.rmtree(args.dest / BUNDLE)
            shutil.move(str(src), str(args.dest / BUNDLE))

    root = args.dest / BUNDLE
    if not root.is_dir():
        raise SystemExit(f"bundle not found at {root}; pass --zip to unpack it")
    count, bad = verify_sums(root)
    for problem in bad[:20]:
        print("BAD:", problem)
    print(f"verified {count - len(bad)}/{count} files under {root}")
    print(f"export OPENWSV_DATA_ROOT={root.resolve()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())