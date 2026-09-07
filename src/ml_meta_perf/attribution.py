"""What each term is worth, and whether dataset or model features carry the equation.

An equation is only useful as guidance if its terms can be compared, and raw weights
cannot be: they are in the units of whatever the term happens to compute. Two comparable
quantities are derived here instead.

*Effect* is the swing in predicted MCC produced by moving a term from its 10th to its
90th percentile, holding everything else fixed. It is stated in MCC units, which is what
makes one term's importance comparable with another's and with the target itself.

*Share* attributes the variance of the equation's output to groups of terms. Terms are
labelled by the features they use -- dataset-only, model-only, or mixed -- so the split
between "how hard is this data" and "how capable is this model" can be read off the
fitted equation rather than assumed.

Study chapter: [7. From equation to practice](../../assets/docs/07-practices.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ml_meta_perf.model import Equation, direction

DATASET_ONLY = "dataset"
MODEL_ONLY = "model"
MIXED = "mixed"


def classify(
    features: tuple[str, ...],
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
) -> str:
    """Label a term by which feature groups it draws on."""
    uses_dataset = any(feature in dataset_features for feature in features)
    uses_model = any(feature in model_features for feature in features)
    if uses_dataset and uses_model:
        return MIXED
    return MODEL_ONLY if uses_model else DATASET_ONLY


def contributions(equation: Equation, columns: dict[str, np.ndarray]) -> np.ndarray:
    """The per-row contribution of each term, as a ``(n_rows, n_terms)`` matrix."""
    if not equation.terms:
        return np.zeros((len(next(iter(columns.values()))), 0))
    with np.errstate(all="ignore"):
        return np.column_stack(
            [weight * term.evaluate(columns) for term, weight in zip(equation.terms, equation.weights, strict=True)]
        )


def term_effects(
    equation: Equation,
    columns: dict[str, np.ndarray],
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
) -> pl.DataFrame:
    """Per-term effect sizes, in MCC units, strongest first.

    ``effect`` is the 10th-to-90th-percentile swing of the term's contribution: how much
    predicted MCC moves across the bulk of the observed range of that term. Percentiles
    rather than min-to-max so that a single extreme row cannot define the headline number.
    """
    matrix = contributions(equation, columns)
    rows: list[dict[str, object]] = []
    for index, term in enumerate(equation.terms):
        column = matrix[:, index]
        low, high = np.percentile(column, [10.0, 90.0])
        rows.append(
            {
                "term": term.name,
                "group": classify(term.features, dataset_features, model_features),
                "weight": equation.weights[index],
                "beta": equation.standardized_weights[index],
                "effect": float(high - low),
                "direction": direction(equation.standardized_weights[index]),
            }
        )
    return pl.DataFrame(rows).sort(pl.col("effect").abs(), descending=True)


def group_shares(
    equation: Equation,
    columns: dict[str, np.ndarray],
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
) -> pl.DataFrame:
    """How much of the equation's output variance each feature group drives.

    Shares are computed as ``cov(group total, prediction) / var(prediction)``. That
    decomposition sums to exactly 1 even when the groups are correlated, which a naive
    ``var(group) / var(total)`` does not -- and the groups here are correlated, because
    three of the model features are functions of dataset size.
    """
    matrix = contributions(equation, columns)
    total = matrix.sum(axis=1)
    variance = float(np.var(total))
    labels = [classify(term.features, dataset_features, model_features) for term in equation.terms]

    rows: list[dict[str, object]] = []
    for group in (DATASET_ONLY, MODEL_ONLY, MIXED):
        selected = [index for index, label in enumerate(labels) if label == group]
        if not selected:
            rows.append({"group": group, "n_terms": 0, "share": 0.0, "effect_sum": 0.0})
            continue
        subtotal = matrix[:, selected].sum(axis=1)
        # Population covariance (ddof=0) to match np.var. np.cov defaults to ddof=1, and
        # mixing the two inflates every share by n/(n-1) so the decomposition stops
        # summing to 1.
        covariance = float(((subtotal - subtotal.mean()) * (total - total.mean())).mean())
        share = covariance / variance if variance > 1e-15 else 0.0
        low, high = np.percentile(subtotal, [10.0, 90.0])
        rows.append(
            {
                "group": group,
                "n_terms": len(selected),
                "share": share,
                "effect_sum": float(high - low),
            }
        )
    return pl.DataFrame(rows)


def variance_decomposition(
    target: np.ndarray,
    datasets: np.ndarray,
    models: np.ndarray,
) -> pl.DataFrame:
    """How much of MCC is explained by dataset identity alone, and by model identity alone.

    This is measured on the target itself, independent of any equation, and is the honest
    answer to "does the choice of model matter more than the choice of data". Each row is
    the variance explained by knowing *only* that group label -- an upper bound on what
    any equation over that group's features could reach.
    """
    total = float(((target - target.mean()) ** 2).sum())
    rows: list[dict[str, object]] = []
    for name, labels in (("dataset identity", datasets), ("model identity", models)):
        prediction = np.zeros_like(target)
        for label in np.unique(labels):
            mask = labels == label
            prediction[mask] = target[mask].mean()
        residual = float(((target - prediction) ** 2).sum())
        rows.append(
            {
                "knowing only": name,
                "n_groups": int(np.unique(labels).shape[0]),
                "variance_explained": 1.0 - residual / total if total > 1e-15 else 0.0,
            }
        )
    return pl.DataFrame(rows)
