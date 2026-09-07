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

Study chapter: [3. Term generation and selection](../../assets/docs/03-term-selection.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: The detectors ``knee_index`` can use. ``autoelbow`` is the default because it takes no
#: threshold, sensitivity or smoothing window; the other two are offered so a reported knee
#: can be checked against methods that make different assumptions.
DETECTORS = ("autoelbow", "kneedle", "lmethod")


def _detect(points: np.ndarray, detector: str) -> int | None:
    if detector == "kneedle":
        from kneeliverse.kneedle import knee
    elif detector == "lmethod":
        from kneeliverse.lmethod import knee
    else:
        from kneeliverse.autoelbow import knee
    return knee(points)


def simplify_curve(sizes: np.ndarray, scores: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices surviving global RDP simplification of the curve.

    Cross-validated curves on 20 groups wander by more than the differences being looked
    for, and a knee detector run on that wander reports the wander. Global RDP reduces the
    curve to the points that carry its shape -- *global* because the reconstruction error
    is evaluated over the whole trace rather than per segment, so a locally tempting split
    that does not help the curve as a whole is not taken.

    On this data it is what makes the detectors agree: run raw, the three report 12, 13 and
    5 terms for the in-sample curve; run on the simplified curve they all report 12.
    """
    from kneeliverse.rdp import grdp

    points = np.column_stack([sizes.astype(np.float64), scores.astype(np.float64)])
    return np.asarray(grdp(points, t=tolerance)[0], dtype=int)


def knee_index(
    sizes: np.ndarray,
    scores: np.ndarray,
    *,
    detector: str = "autoelbow",
    tolerance: float | None = None,
) -> int:
    """Index of the knee in a performance curve, or the last point if detection fails.

    With ``tolerance`` set the curve is simplified first and the knee is located on the
    simplified trace, then mapped back to an index into the original.
    """
    if sizes.shape[0] < 3:
        return int(sizes.shape[0] - 1)

    order = np.arange(sizes.shape[0])
    if tolerance is not None:
        try:
            order = simplify_curve(sizes, scores, tolerance)
        except (ValueError, IndexError):
            order = np.arange(sizes.shape[0])
        if order.shape[0] < 3:
            order = np.arange(sizes.shape[0])

    points = np.column_stack([sizes[order].astype(np.float64), scores[order].astype(np.float64)])
    try:
        found = _detect(points, detector)
    except (ValueError, IndexError):
        return int(sizes.shape[0] - 1)
    if found is None:
        return int(sizes.shape[0] - 1)
    return int(order[int(found)])


#: The three protocols a length can be judged on, in the order `consensus_curve` combines them.
PROTOCOLS = ("r2_in_sample", "r2_loo_dataset", "r2_loo_model")


def consensus_curve(curve: pl.DataFrame, how: str = "median") -> np.ndarray:
    """One score per length, combining every protocol present.

    **The published length must not be chosen on in-sample R2 alone.** In-sample is monotone
    in the number of terms, so it can only ever say "more", and a length picked on it is
    picked on the one curve that cannot express the trade the choice is about. But the
    cross-validated curves cannot be used alone either: on twenty groups they wander, and on
    this corpus leave-one-dataset-out has genuine craters -- a held-out dataset outside the
    convex hull of the other nineteen is extrapolated to far outside MCC's range and clipped,
    which at one length drops the pooled figure from 0.63 to 0.39.

    So the detector runs on a consensus of all three. ``median`` is the default and is what
    makes it robust: a crater in one protocol moves the median to the middle value rather
    than dragging an average down with it. At the length above, the three protocols read
    0.659 / 0.393 / 0.616 and the median is 0.616 -- the crater is ignored, which is correct,
    because a single fold's extrapolation is a property of that fold and not of the length.

    ``mean`` is offered for comparison and is *not* robust to that. ``min`` is the
    conservative reading -- a length is only as good as its worst protocol.
    """
    columns = [name for name in PROTOCOLS if name in curve.columns]
    if not columns:
        raise ValueError("curve carries none of the protocol columns")
    stacked = np.column_stack([curve[name].to_numpy() for name in columns])
    if how == "mean":
        return stacked.mean(axis=1)
    if how == "min":
        return stacked.min(axis=1)
    return np.median(stacked, axis=1)


def knee_terms(curve: pl.DataFrame, column: str = "consensus") -> int:
    """The number of terms at the knee of ``column``.

    ``consensus`` -- the default -- runs the detector on `consensus_curve`, the per-length
    median over every protocol available. Passing an explicit column name runs it on that
    curve alone, which is what the per-protocol rows of `recommend` do.

    The default used to be ``r2_in_sample``, on the argument that it is monotone by
    construction and so is the shape a knee detector expects. Monotone is exactly the
    problem: a curve that only rises says nothing about where the extra terms stop being
    worth their readability, which is the question being asked.
    """
    sizes = curve["n_terms"].to_numpy()
    scores = consensus_curve(curve) if column == "consensus" else curve[column].to_numpy()
    return int(sizes[knee_index(sizes, scores)])


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
    """Every rule that could choose a length, plus the evidence, as one small table.

    The consensus knee is the one to read. The per-protocol rows are reported beside it so
    that a reader can see whether the protocols agree -- when they do not, the choice is
    being made by whichever curve was picked, and that has to be visible rather than buried
    in a default argument.
    """
    rows: list[dict[str, object]] = []
    sizes = curve["n_terms"].to_numpy()

    labelled = [("consensus", "knee (consensus)")] + [
        (column, f"knee ({column.removeprefix('r2_').replace('_', '-')})") for column in PROTOCOLS
    ]
    for column, label in labelled:
        if column != "consensus" and column not in curve.columns:
            continue
        scores = consensus_curve(curve) if column == "consensus" else curve[column].to_numpy()
        index = knee_index(sizes, scores)
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
