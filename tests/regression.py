"""Does the MCC experiment still produce the results it produced at the start of the work?

``tests/expected/mcc/`` holds the equations (E1, E2, E3-Valid, E3-MAX), their curves and the scores that
``ml-meta-perf fit --config config/mcc.json`` wrote at the start of the migration, when the equations
were taken from the paper pipeline (E1, E2 and E3 byte-identical to it). A fit takes about thirty
seconds, so it is not a unit test: CI runs it after ``fit``, and this is how to run it by hand::

    ml-meta-perf fit --config config/mcc.json --output /tmp/mcc --quiet
    python -m tests.regression /tmp/mcc/meta_dataset/MCC

The terms must be identical and every number equal to ``rtol`` (a different BLAS may move the last
digits, not the first six). The exit status is the number of differences, capped at 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

EXPECTED = Path(__file__).resolve().parent / "expected" / "mcc"
VARIANTS = ("e1", "e2", "e3", "e3-max")
NUMERIC_KEYS = ("intercept", "weights", "standardized_weights")


def _equation(actual: dict[str, object], expected: dict[str, object], rtol: float) -> list[str]:
    problems: list[str] = []
    if actual["terms"] != expected["terms"]:
        problems.append("the terms differ")
    for key in NUMERIC_KEYS:
        if not np.allclose(
            np.asarray(actual[key], dtype=float), np.asarray(expected[key], dtype=float), rtol=rtol, atol=1e-9
        ):
            problems.append(f"{key} differ beyond rtol {rtol}")
    for key in sorted((set(actual) | set(expected)) - set(NUMERIC_KEYS) - {"terms"}):
        if actual.get(key) != expected.get(key):
            problems.append(f"{key}: {actual.get(key)!r} != {expected.get(key)!r}")
    return problems


def _table(actual: pl.DataFrame, expected: pl.DataFrame, rtol: float) -> list[str]:
    if actual.columns != expected.columns or actual.height != expected.height:
        return [f"shape {actual.shape} {actual.columns} != {expected.shape} {expected.columns}"]
    problems = []
    for name in expected.columns:
        if expected[name].dtype.is_numeric():
            if not np.allclose(
                actual[name].to_numpy(), expected[name].to_numpy(), rtol=rtol, atol=1e-9, equal_nan=True
            ):
                problems.append(f"column {name} differs beyond rtol {rtol}")
        elif actual[name].to_list() != expected[name].to_list():
            problems.append(f"column {name} differs")
    return problems


def compare(actual: Path, expected: Path = EXPECTED, rtol: float = 1e-6) -> list[str]:
    """Every difference between a result folder and the expected one, one line each; empty when none."""
    problems: list[str] = []
    for variant in VARIANTS:
        left, right = actual / variant, expected / variant
        for problem in _equation(
            json.loads((left / "equation.json").read_text()), json.loads((right / "equation.json").read_text()), rtol
        ):
            problems.append(f"{variant}/equation.json: {problem}")
        for problem in _table(pl.read_csv(left / "curve.csv"), pl.read_csv(right / "curve.csv"), rtol):
            problems.append(f"{variant}/curve.csv: {problem}")
    for problem in _table(pl.read_csv(actual / "scores.csv"), pl.read_csv(expected / "scores.csv"), rtol):
        problems.append(f"scores.csv: {problem}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m tests.regression RESULT_FOLDER", file=sys.stderr)
        return 2
    problems = compare(Path(argv[0]))
    for problem in problems:
        print(problem)
    print("the MCC experiment reproduces the expected results" if not problems else f"{len(problems)} difference(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
