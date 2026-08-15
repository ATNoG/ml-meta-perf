"""Turning a fitted equation into written guidance.

This is the payoff the whole project trades accuracy for, so it is worth being exact
about what it is and is not. A practice here is a statement of the form "across this
meta-dataset, higher X went with higher/lower MCC, worth about D MCC points". That is an
**association measured on 20 datasets**, not a causal claim and not a guarantee; the
`confidence` column carries the evidence needed to discount it.

The derivation is deliberately empirical rather than symbolic. Reading a sign off a
weight is wrong as soon as a feature appears in more than one term, or appears inside a
ratio's denominator, or appears under a transform that flips its monotonicity. Instead
every term's contribution is evaluated over the real data, contributions are summed per
feature, and the direction is the rank correlation between the feature and the total it
drives. That answers "what does this equation actually do as this feature rises", which
is the question a practitioner is asking.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from metafit.attribution import contributions
from metafit.data import FEATURE_GLOSSARY
from metafit.model import Equation
from metafit.stats import spearman

#: A feature has to move predicted MCC by at least this much, between the bottom and top
#: decile of its own range, before it is worth writing down as guidance.
MIN_EFFECT = 0.02

#: And it has to have been selected in at least this fraction of cross-validation folds.
#: A term picked in 4 of 20 folds is an artefact of which datasets landed in the training
#: split; publishing it as advice would be the most misleading thing this module could do.
MIN_STABILITY = 0.5

#: And its relationship has to be monotone enough for "higher X" to mean anything. A
#: feature can carry a large effect through a non-monotone term -- a product with another
#: feature that changes sign, say -- and no directional sentence would be true of it.
MIN_DIRECTION = 0.15

#: Fraction of rows taken as the low and high end when measuring an effect.
DECILE = 0.1


def feature_practices(
    equation: Equation,
    columns: dict[str, np.ndarray],
    stability: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per raw feature the equation uses, strongest effect first.

    ``effect`` is signed: the mean MCC this feature's terms contribute in its top decile
    minus the mean they contribute in its bottom decile. Ordering by the feature is the
    whole point -- a quantile of the contributions alone would be order-invariant and
    would describe the term's spread rather than its response to the feature.

    ``direction`` is the rank correlation between the feature and the MCC it drives, so
    it survives transforms and multiple appearances. ``stability`` is the mean selection
    frequency of the terms involved, when a cross-validated stability table is supplied.

    A feature appearing in a term shared with another feature is credited the whole of
    that term, so two features sharing one dominant term will show the same effect. That
    is honest -- the term cannot be split between them -- but it means ``effect`` reads
    as "how much MCC moves across this feature's range", not "how much this feature owns".
    """
    matrix = contributions(equation, columns)
    frequency = _frequency_lookup(stability)

    features = sorted({feature for term in equation.terms for feature in term.features})
    if not features:
        # An empty frame built from an empty row list carries no schema, so every later
        # column reference fails. Declare it instead.
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "meaning": pl.String,
                "n_terms": pl.Int64,
                "direction": pl.Float64,
                "effect": pl.Float64,
                "stability": pl.Float64,
            }
        )
    rows: list[dict[str, object]] = []
    for feature in features:
        indices = [index for index, term in enumerate(equation.terms) if feature in term.features]
        total = matrix[:, indices].sum(axis=1)
        values = columns[feature]
        order = np.argsort(values)
        edge = max(1, round(DECILE * values.shape[0]))
        effect = float(total[order[-edge:]].mean() - total[order[:edge]].mean())
        rows.append(
            {
                "feature": feature,
                "meaning": FEATURE_GLOSSARY.get(feature, feature),
                "n_terms": len(indices),
                "direction": spearman(values, total),
                "effect": effect,
                "stability": float(
                    np.mean([frequency.get(equation.terms[index].name, float("nan")) for index in indices])
                )
                if frequency
                else float("nan"),
            }
        )
    return pl.DataFrame(rows).sort(pl.col("effect").abs(), descending=True)


def _frequency_lookup(stability: pl.DataFrame | None) -> dict[str, float]:
    if stability is None or stability.height == 0:
        return {}
    return dict(zip(stability["term"].to_list(), stability["frequency"].to_list(), strict=True))


def _confidence(effect: float, stability: float) -> str:
    if np.isnan(stability):
        return "unrated"
    if stability >= 0.85 and abs(effect) >= 0.10:
        return "strong"
    if stability >= MIN_STABILITY and abs(effect) >= 0.05:
        return "moderate"
    return "weak"


def best_practices(
    equation: Equation,
    columns: dict[str, np.ndarray],
    stability: pl.DataFrame | None = None,
    *,
    min_effect: float = MIN_EFFECT,
    min_stability: float = MIN_STABILITY,
    min_direction: float = MIN_DIRECTION,
) -> pl.DataFrame:
    """Written practices, filtered to those the evidence actually supports.

    A feature is reported when it moves predicted MCC by at least ``min_effect``, its
    relationship is monotone enough to describe in a sentence (``min_direction``), and
    its terms survived at least ``min_stability`` of the cross-validation folds. Features
    failing any test are dropped rather than reported with a caveat -- a practice nobody
    should act on is better left unwritten.
    """
    table = feature_practices(equation, columns, stability)
    if table.height == 0:
        return table.with_columns(pl.lit("").alias("practice"), pl.lit("").alias("confidence"))

    kept = table.filter(
        (pl.col("effect").abs() >= min_effect) & (pl.col("direction").abs() >= min_direction)
    )
    if stability is not None:
        kept = kept.filter(pl.col("stability").is_nan() | (pl.col("stability") >= min_stability))

    statements: list[str] = []
    confidences: list[str] = []
    for row in kept.iter_rows(named=True):
        rises = float(row["direction"]) > 0.0
        statements.append(
            f"Higher {row['meaning']} went with {'higher' if rises else 'lower'} MCC "
            f"(about {abs(float(row['effect'])):.2f} MCC between its lowest and highest decile)."
        )
        confidences.append(_confidence(float(row["effect"]), float(row["stability"])))

    return kept.with_columns(
        pl.Series("practice", statements),
        pl.Series("confidence", confidences),
    )


def render(practices: pl.DataFrame) -> str:
    """The practices as a numbered, readable list."""
    if practices.height == 0:
        return "No practice met the effect and stability thresholds."
    lines: list[str] = []
    for index, row in enumerate(practices.iter_rows(named=True), start=1):
        lines.append(f"{index:2d}. [{row['confidence']:8s}] {row['practice']}")
    return "\n".join(lines)
