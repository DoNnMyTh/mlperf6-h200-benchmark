#!/usr/bin/env python3
"""Entry point: python3 powermon.py [start|status|stop|report|probe] (no args = wizard)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from powermon.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
