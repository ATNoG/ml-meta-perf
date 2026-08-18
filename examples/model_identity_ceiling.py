"""How much would better model descriptors be worth? An upper bound, measured.

Five model meta-features recover 63% of what model identity explains
([chapter 6](../assets/docs/06-results.md)). Replacing them with identity itself --
a free level and slope per model, fitted inside each training fold -- measures what the
missing 37% is worth, and therefore bounds what any richer descriptor set could buy.

This is a **ceiling**, not a model. It is reproduced here rather than wired into
``python -m metafit`` because a table of 50 fitted numbers is not an equation and has no
place among the study's results; see
[chapter 9](../assets/docs/09-model-effects.md) for why it was withdrawn.

    PYTHONPATH=src venv/bin/python examples/model_identity_ceiling.py
"""

from metafit.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    MODEL_FEATURES,
    columns_as_arrays,
    groups,
    load,
    target,
)
from metafit.experiment import DEFAULT_E3, run_e3
from metafit.identity import carrier_stability, correct_out_of_fold
from metafit.validate import score


def main() -> int:
    frame = load()
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    e3 = run_e3(frame)
    size = DEFAULT_E3.headline_terms
    path = e3.paths["loo_dataset"][size]

    rows = {
        "E3 alone": path.predictions,
        "+ per-model levels": correct_out_of_fold(path, columns, truth, datasets, models),
        "+ levels and one slope": correct_out_of_fold(
            path, columns, truth, datasets, models, DATASET_FEATURES
        ),
    }

    print(f"{'':26s} {'R2':>8s} {'MAE':>8s}   leave-one-dataset-out, {size} terms")
    baseline = None
    for label, prediction in rows.items():
        scores = score(truth, prediction)
        gain = "" if baseline is None else f"   ({scores.r2 - baseline:+.4f})"
        baseline = baseline if baseline is not None else scores.r2
        print(f"{label:26s} {scores.r2:8.4f} {scores.mae:8.4f}{gain}")

    print("\nwhich dataset feature carried the slope, across folds:")
    print(carrier_stability(columns, truth, datasets, models, path, DATASET_FEATURES))

    # Under leave-one-model-out the held-out model has no training row, so there is no
    # effect to apply and the correction is exactly the identity. That is the boundary.
    held_out = e3.paths["loo_model"][size]
    corrected = correct_out_of_fold(
        held_out, columns, truth, models, models, DATASET_FEATURES
    )
    print(
        "\nleave-one-model-out: R2 "
        f"{score(truth, held_out.predictions).r2:.4f} -> {score(truth, corrected).r2:.4f} "
        "(no gain, by construction)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
