"""Run all local quality gates with the active Python interpreter."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

COMMANDS: tuple[tuple[str, ...], ...] = (
    ("ruff", "format", "--check", "."),
    ("ruff", "check", "."),
    ("mypy",),
    ("pytest",),
)


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        print("smart-cv-check does not accept arguments.", file=sys.stderr)
        return 2
    for arguments in COMMANDS:
        command = [sys.executable, "-m", *arguments]
        print(f"> {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
