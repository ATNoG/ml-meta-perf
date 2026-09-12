"""Run one repository quality check portably from a pre-commit environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "ruff": ("ruff", "check", "src", "tests", "scripts"),
    "basedpyright": ("basedpyright", "--pythonpath", sys.executable),
    "vulture": ("vulture",),
    "unittest": (sys.executable, "-m", "unittest", "discover", "-s", "tests"),
}


def main() -> int:
    """Run the requested check against the current working tree."""
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(COMMANDS)
        print(f"usage: {Path(sys.argv[0]).name} {{{choices}}}", file=sys.stderr)
        return 2
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(COMMANDS[sys.argv[1]], cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
