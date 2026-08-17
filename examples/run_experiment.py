"""Reproduce the full study: equations, tables and report into ``results/``.

Identical to ``python -m metafit --no-figures``, spelled out so the entry point is
visible from a file rather than only from a command line.

Run from the repository root:

    PYTHONPATH=src venv/bin/python examples/run_experiment.py
"""

from pathlib import Path

from metafit.cli import main

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "results"
    raise SystemExit(main(["--output", str(destination), "--no-figures"]))
