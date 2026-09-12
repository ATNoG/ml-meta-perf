"""The retained E3 equation-selection rules.

E3-Valid follows the first sustained plateau in the median of in-sample,
leave-one-dataset-out and leave-one-model-out R2. E3-MAX independently maximises the worst of
those protocols and the doubly-held-out cell protocol.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: The three protocols a length can be judged on, in the order `consensus_curve` combines them.
PROTOCOLS = ("r2_in_sample", "r2_loo_dataset", "r2_loo_model")
PLATEAU_TOLERANCE = 0.001
PLATEAU_WINDOW = 3

#: Every protocol a *configuration* is judged on, which is `PROTOCOLS` plus the doubly-held-out
#: one. `floor_curve` takes the minimum over these and E3-MAX selects on it.
#:
#: The cell protocol is in this set and not in `PROTOCOLS` deliberately. `consensus_curve` is
#: the per-length reading used by E3-Valid and stays on the three single-group protocols. The E3-MAX rule
#: is the one that publishes a headline about an unseen (dataset, model) cell, so it is the one
#: that has to select on that protocol rather than on three looser ones.
JUDGED_PROTOCOLS = (*PROTOCOLS, "r2_loo_cell")


def consensus_curve(curve: pl.DataFrame, how: str = "median") -> np.ndarray:
    """One score per length, combining every protocol present.

    **A length must not be chosen on in-sample R2 alone.** In-sample is monotone in the number
    of terms, so it can only ever say "more", and a length picked on it is picked on the one
    curve that cannot express the trade the choice is about. But the cross-validated curves
    cannot be used alone either: on twenty groups they wander, and leave-one-dataset-out has
    genuine craters -- a held-out dataset lying outside the convex hull of the other nineteen
    is extrapolated far outside MCC's range and then clipped, which at one length drops the
    pooled figure from 0.63 to 0.39.

    ``median`` is the default and is what makes this robust: a crater in one protocol moves the
    median to the middle value rather than dragging an average down with it. At the length
    above the three protocols read 0.667 / 0.399 / 0.616 and the median is 0.616 -- the crater
    is ignored, which is correct, because one fold's extrapolation is a property of that fold
    and not of the length.

    ``mean`` is offered for comparison and is *not* robust to that. ``min`` is the conservative
    reading: a length is only as good as its worst protocol.
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


def plateau_index(
    scores: np.ndarray,
    *,
    tolerance: float = PLATEAU_TOLERANCE,
    window: int = PLATEAU_WINDOW,
) -> int:
    """Return the best point immediately before a sustained performance plateau.

    The input order is the increasing equation length. The rule follows the best score seen
    so far and selects the first point whose best-so-far gain over the next ``window`` points
    does not exceed ``tolerance``. If no plateau is found, it returns the first global maximum.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("plateau selection needs a non-empty one-dimensional score curve")
    if tolerance < 0.0:
        raise ValueError("plateau tolerance must be non-negative")
    if window < 1:
        raise ValueError("plateau window must be positive")
    envelope = np.maximum.accumulate(values)
    for index in range(max(len(values) - window, 0)):
        if float(envelope[index + window] - envelope[index]) <= tolerance:
            return index
    return int(np.argmax(values))


def plateau_configuration(
    curves: dict[int, pl.DataFrame],
    *,
    tolerance: float = PLATEAU_TOLERANCE,
    window: int = PLATEAU_WINDOW,
) -> tuple[int, int]:
    """Select E3-Valid across arities with the retained plateau rule."""
    candidates: dict[int, tuple[float, int]] = {}
    for arity, curve in sorted(curves.items()):
        for size, score in zip(curve["n_terms"], consensus_curve(curve), strict=True):
            key = int(size)
            value = float(score)
            incumbent = candidates.get(key)
            if incumbent is None or (-value, complexity(arity, key), arity) < (
                -incumbent[0],
                complexity(incumbent[1], key),
                incumbent[1],
            ):
                candidates[key] = (value, arity)
    if not candidates:
        raise ValueError("no E3 curves are available for plateau selection")
    sizes = sorted(candidates)
    scores = np.asarray([candidates[size][0] for size in sizes], dtype=np.float64)
    index = plateau_index(scores, tolerance=tolerance, window=window)
    best_index = int(np.argmax(scores[: index + 1]))
    size = sizes[best_index]
    return candidates[size][1], size


def complexity(arity: int, n_terms: int) -> int:
    """Feature slots an equation spends: a term of arity ``a`` names ``a`` raw features.

    **Not a count of fitted coefficients** -- those number ``n_terms + 1``. This charges for how
    much of the *grammar* an equation uses, which is what lets it tell the same term count at
    arity 2 apart from arity 3. A coefficient count cannot make that distinction.
    """
    return arity * n_terms


def floor_curve(curve: pl.DataFrame) -> np.ndarray:
    """The **worst** of every protocol present, per length. This is what a length is judged on.

    A length is only as good as the protocol it does worst on. That is the conservative
    reading `consensus_curve` offers as ``how="min"``, taken here as the criterion rather than
    as an option, and taken over **four** protocols rather than three: in-sample,
    leave-one-dataset-out, leave-one-model-out and the doubly-held-out cell protocol.

    Two reasons the minimum rather than the median, and both are about what the study claims.
    The median lets a length hide its weakest protocol behind its other two, and the weakest
    protocol here is always `r2_loo_cell` -- the one the study's headline is actually about,
    the cell where neither the dataset nor the model has been seen. A rule that reports the
    strictest protocol and then selects on a median of looser ones is selecting on a different
    quantity from the one it publishes. And the median of four is an average of the middle two,
    which is neither a protocol nor a bound; the minimum is always some protocol's own number.

    The craters `consensus_curve` was made robust against are still handled, because they are
    *shared*: a held-out dataset outside the convex hull of the other nineteen is extrapolated
    under leave-one-dataset-out and under the cell protocol alike, so at those lengths the
    minimum drops with the median rather than instead of it. On this corpus the two agree on
    where the arity-2 curve peaks to within the lengths that crater.
    """
    columns = [name for name in JUDGED_PROTOCOLS if name in curve.columns]
    if not columns:
        raise ValueError("curve carries none of the protocol columns")
    return np.column_stack([curve[name].to_numpy() for name in columns]).min(axis=1)


def protocol_spread(curve: pl.DataFrame) -> np.ndarray:
    """How far a length falls from its fit to its worst protocol: ``in-sample - floor``.

    **Reported, not selected on** -- E3-Valid uses `plateau_configuration` -- and reported because the
    claim it measures would otherwise be asserted. An equation that fits well and transfers
    badly is a different object from one that does both moderately, and the floor alone cannot
    tell them apart: two lengths reaching the same worst protocol from a different fit are the
    same number to `floor_curve` and are not the same equation.

    On the corrected corpus the selected arity-3 equation has a spread of 0.0640. The arity-2
    candidate is slightly tighter at 0.0629, but its worst-protocol R2 is lower by 0.0444.
    This is the same quantity the beam negatives record: a policy that fits well and then
    collapses under validation has a large spread.

    This is deliberately *not* folded into the selection score. Combining a level and a spread
    needs a weight between them, a weight is a free parameter, and a free parameter is what
    this rule was revised to remove. The two are reported side by side and the argument is made
    in the chapter instead.
    """
    columns = [name for name in JUDGED_PROTOCOLS if name in curve.columns]
    if "r2_in_sample" not in columns:
        raise ValueError("a spread needs the in-sample column to measure the drop from")
    return curve["r2_in_sample"].to_numpy() - floor_curve(curve)


def floor_argmax_index(curve: pl.DataFrame) -> int:
    """The *row position* of `floor_argmax`'s length, for callers indexing the curve."""
    return int(np.argmax(floor_curve(curve)))


def floor_argmax(curve: pl.DataFrame) -> int:
    """The length maximising `floor_curve`. **This is the length rule, for one grammar.**

    `experiment.run_equation` uses it for equations without the E3 plateau selection, and
    `arity_candidates` uses it to build the E3-MAX candidate set.

    An argmax, so there is no threshold, no smoothing window, no sensitivity parameter and no
    corpus size in it.
    """
    return int(curve["n_terms"].to_numpy()[floor_argmax_index(curve)])


def arity_candidates(curves: dict[int, pl.DataFrame]) -> dict[int, int]:
    """One length per arity: the argmax of that arity's `floor_curve`.

    **The candidate set is one length per grammar, and that is what keeps the rule honest.**
    This is the capability candidate for each grammar. It remains separate from E3-Valid,
    whose plateau rule reads the three-protocol Combined R2 curve.
    """
    return {arity: floor_argmax(curve) for arity, curve in curves.items()}


def most_capable(curves: dict[int, pl.DataFrame]) -> tuple[int, int]:
    """``(arity, n_terms)`` with the best floor: how far the additive form reaches.

    No complexity penalty, because this one is not put forward as an equation to read -- it
    exists to bound what the form can do, and a bound should not be discounted for being long.
    Ties go to the shorter equation.
    """
    ranked = [
        (float(floor_curve(curves[arity]).max()), -size, arity, size)
        for arity, size in arity_candidates(curves).items()
    ]
    _, _, arity, size = max(ranked)
    return arity, size
