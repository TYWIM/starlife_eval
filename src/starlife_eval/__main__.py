"""Allow ``python -m starlife_eval <args>`` as a shortcut for the WSV CLI."""

from .wsv.cli import main

raise SystemExit(main())
