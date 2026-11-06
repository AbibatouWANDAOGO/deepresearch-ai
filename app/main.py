"""Command-line entry point.

    python -m app.main     # check the configuration and prepare local storage
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from app import __version__
from app.config import get_settings


def main() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "settings"
            print(f"  - {location}: {error['msg']}", file=sys.stderr)
        return 1

    settings.ensure_storage()
    print(f"DeepResearch AI {__version__}: configuration OK")
    width = max(len(key) for key in settings.summary())
    for key, value in settings.summary().items():
        print(f"  {key:<{width}}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
