"""Deterministic synthetic scene generator for the OpenWSV T5/T7 suites.

Version string ``openwsv-scene-1.0.0``; fixed seed :data:`SEED` (20261020).
Pure Python + numpy + Pillow; no network access and no external assets.

The generator produces a fixed tabletop ``room`` (table, backdrop, floor) that is
shared by every frame, so two frames of one task always show the same camera /
room. Two suites are built from that room:

* ``relation_change`` (T5, n=600): before/after pairs of the same camera, with a
  controlled relation change between two described objects;
* ``cross_frame`` (T7, n=400): frame1/frame2 pairs whose object sets are
  disjoint; the question asks for the image-plane direction of an object that
  appears only in frame 1 relative to an object that appears only in frame 2.

Reproduce (from the repository root)::

    C:\\Users\\MECHREVO\\miniconda3\\python.exe -X utf8 -m scripts.wsv.gen.build

See ``scripts/wsv/gen/README.md`` for the full CLI.
"""

from __future__ import annotations

GENERATOR_VERSION = "openwsv-scene-1.0.0"
SEED = 20261020
__version__ = GENERATOR_VERSION

__all__ = ["GENERATOR_VERSION", "SEED", "__version__"]
