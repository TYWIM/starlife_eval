"""Join the split data-bundle parts committed under data/openwsv-data-1.0/.

The 494.6 MB bundle is split into 13 parts (<100 MB each) so it can live in git;
the joined zip must hash to the pinned sha256 below.

Usage:
    python scripts/wsv/join_data.py --dest ./data/openwsv-data-1.0.zip
    python scripts/wsv/join_data.py --extract-to ./data   # join + verify + unpack
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

ZIP_SHA256 = "971090ee15347b016f5c4afd92b6a711481d81de169c9d45d55e64a6fef2ad63"
BUNDLE = "openwsv-data-1.0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Join and verify the OpenWSV data bundle")
    ap.add_argument("--parts-dir", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2] / "data" / "openwsv-data-1.0")
    ap.add_argument("--dest", type=pathlib.Path, default=pathlib.Path("openwsv-data-1.0.zip"))
    ap.add_argument("--extract-to", type=pathlib.Path)
    args = ap.parse_args()

    parts = sorted(args.parts_dir.glob("openwsv-data-1.0.zip.part*"))
    if not parts:
        raise SystemExit(f"no parts under {args.parts_dir}")
    blob = b"".join(p.read_bytes() for p in parts)
    actual = sha256_bytes(blob)
    print(f"joined {len(parts)} parts -> {len(blob)/1024/1024:.1f} MB, sha256 {actual}")
    if actual != ZIP_SHA256:
        raise SystemExit(f"sha256 mismatch: {actual} != {ZIP_SHA256}")
    args.dest.write_bytes(blob)
    print(f"wrote {args.dest}")

    if args.extract_to:
        from prepare_data import main as prepare_main  # noqa: PLC0415
        sys.argv = ["prepare_data.py", "--zip", str(args.dest), "--dest", str(args.extract_to),
                    "--expect-sha256", ZIP_SHA256]
        return prepare_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())