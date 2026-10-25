"""Allow ``python -m scripts.wsv.gen`` to run the generator CLI."""

from __future__ import annotations

from .build import main

if __name__ == "__main__":
    raise SystemExit(main())
