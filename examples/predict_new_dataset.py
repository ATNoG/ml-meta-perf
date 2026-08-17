"""Apply a saved equation to a new (dataset, model) pair.

The point of publishing an equation rather than a fitted object is that it can be
evaluated by hand. This script does the same thing programmatically, and prints the
per-term contributions so the prediction can be read as a breakdown rather than a number.

    PYTHONPATH=src venv/bin/python examples/run_experiment.py       # writes results/e3.json
    PYTHONPATH=src venv/bin/python examples/predict_new_dataset.py
"""

from pathlib import Path

import numpy as np

from metafit.data import ALL_FEATURES, DATASET_COLUMN, MODEL_COLUMN, columns_as_arrays, load
from metafit.model import Equation

EQUATION_PATH = Path(__file__).resolve().parents[1] / "results" / "e3.json"


def main() -> int:
    if not EQUATION_PATH.is_file():
        print(f"no equation at {EQUATION_PATH}; run examples/run_experiment.py first")
        return 1

    equation = Equation.load(EQUATION_PATH)

    # Stand in for a genuinely new row by taking one from the meta-dataset.
    frame = load()
    row = frame.head(1)
    columns = columns_as_arrays(row, ALL_FEATURES)

    print(f"dataset : {row[DATASET_COLUMN].item()}")
    print(f"model   : {row[MODEL_COLUMN].item()}")
    print(f"actual  : {row['MCC'].item():+.4f}")
    print(f"predicted: {float(equation.predict(columns)[0]):+.4f}\n")

    print("contribution breakdown:")
    print(f"  {equation.intercept:+10.4f}   intercept")
    contributions: list[tuple[float, str]] = []
    with np.errstate(all="ignore"):
        for term, weight in zip(equation.terms, equation.weights, strict=True):
            contributions.append((weight * float(term.evaluate(columns)[0]), term.name))
    for value, name in sorted(contributions, key=lambda item: abs(item[0]), reverse=True):
        print(f"  {value:+10.4f}   {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
