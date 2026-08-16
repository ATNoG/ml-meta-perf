"""Choosing how many terms the published equation should have.

The accuracy-versus-length curve is a classic performance curve: it climbs steeply, then
flattens, and every term past the bend costs readability for almost no accuracy. Picking
the bend by eye is exactly the kind of judgement this project tries to remove from the
results, so it is delegated to a detector.

Two answers are offered because they ask different questions:

* the **knee** of the in-sample curve -- where additional terms stop paying, which is the
  accuracy-versus-explainability trade the study is about;
* the **Pareto front** over (length, cross-validated accuracy) -- every length that is
  not beaten by a shorter equation on the metric that matters for transfer.

``kneeliverse`` supplies the detector. Its ``autoelbow`` takes no threshold, sensitivity
or smoothing window, so the chosen length is a property of the curve rather than of a
parameter someone picked to get the answer they wanted.

Study chapter: [3. Search and fitting](../../assets/docs/03-search-and-fitting.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def knee_index(sizes: np.ndarray, scores: np.ndarray) -> int:
    """Index of the knee in a performance curve, or the last point if detection fails."""
    from kneeliverse.autoelbow import knee

    if sizes.shape[0] < 3:
        return int(sizes.shape[0] - 1)
    points = np.column_stack([sizes.astype(np.float64), scores.astype(np.float64)])
    try:
        return int(knee(points))
    except (ValueError, IndexError):
        return int(sizes.shape[0] - 1)


def knee_terms(curve: pl.DataFrame, column: str = "r2_in_sample") -> int:
    """The number of terms at the knee of ``column``.

    Defaults to the in-sample curve: it is monotone by construction, which is what a knee
    detector expects. The cross-validated curves are not monotone -- they wander with fold
    noise on 20 groups -- so a knee found on them would be describing the noise.
    """
    sizes = curve["n_terms"].to_numpy()
    return int(sizes[knee_index(sizes, curve[column].to_numpy())])


def pareto_front(curve: pl.DataFrame, column: str = "r2_loo_dataset") -> pl.DataFrame:
    """Lengths that no shorter equation beats on ``column``.

    A length is on the front when nothing shorter scores at least as well. Reading down
    the front gives the shortest equation achieving each level of accuracy, which is the
    decision the term budget actually poses.
    """
    table = curve.sort("n_terms")
    scores = table[column].to_numpy()
    keep = [index for index in range(len(scores)) if not np.any(scores[:index] >= scores[index])]
    return table[keep]


def pareto_table(curve: pl.DataFrame) -> pl.DataFrame:
    """Per-length flags for whether a length is on the in-sample or transfer front.

    Both fronts are reported because they answer the two halves of the trade. The
    in-sample front asks "what is the shortest equation achieving this quality of fit",
    which is the explainability question; the cross-validated front asks the same about
    transfer. A length on both is defensible under either priority, and on this data the
    in-sample front is every length -- fit is monotone in terms, so nothing is ever
    dominated -- which is itself the reason the in-sample curve alone cannot choose a
    length and the knee is needed.
    """
    table = curve.sort("n_terms")
    columns = [column for column in ("r2_in_sample", "r2_loo_dataset") if column in table.columns]
    flags: dict[str, list[bool]] = {}
    for column in columns:
        scores = table[column].to_numpy()
        front = {index for index in range(len(scores)) if not np.any(scores[:index] >= scores[index])}
        flags[f"front_{column.removeprefix('r2_')}"] = [index in front for index in range(len(scores))]
    return table.select("n_terms", *columns).with_columns(
        **{name: pl.Series(name, values) for name, values in flags.items()}
    )


def recommend(curve: pl.DataFrame) -> pl.DataFrame:
    """Both recommendations plus the evidence, as one small table."""
    rows: list[dict[str, object]] = []
    sizes = curve["n_terms"].to_numpy()

    for column, label in (("r2_in_sample", "knee (in-sample)"), ("r2_loo_dataset", "knee (loo-dataset)")):
        if column not in curve.columns:
            continue
        index = knee_index(sizes, curve[column].to_numpy())
        rows.append(
            {
                "rule": label,
                "n_terms": int(sizes[index]),
                "r2_in_sample": float(curve["r2_in_sample"].to_numpy()[index]),
                "r2_loo_dataset": (
                    float(curve["r2_loo_dataset"].to_numpy()[index])
                    if "r2_loo_dataset" in curve.columns
                    else float("nan")
                ),
            }
        )

    if "r2_loo_dataset" in curve.columns:
        best = curve.sort("r2_loo_dataset", descending=True).head(1)
        rows.append(
            {
                "rule": "best loo-dataset",
                "n_terms": int(best["n_terms"][0]),
                "r2_in_sample": float(best["r2_in_sample"][0]),
                "r2_loo_dataset": float(best["r2_loo_dataset"][0]),
            }
        )
    return pl.DataFrame(rows)
