"""Reproduce the full study and write the fitted equations to ``results/``.

Run from the repository root:

    PYTHONPATH=src venv/bin/python examples/run_experiment.py
"""

from pathlib import Path

from metafit.cli import main

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "results"
    raise SystemExit(main(["--save", str(destination)]))
