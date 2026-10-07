"""How long an equation should be: one rule for E1, E2 and E3-Valid, and a bound for E3-MAX.

Every equation is searched once up to a horizon and cross-validated at every length, so the
choice of length is a reading of a complete curve. The curve read is the **floor**: the worst
R2 over in-sample (IS), leave-one-dataset-out (LODO), leave-one-model-out (LOMO) and doubly
held-out (DHO) validation, so a length is only as good as the protocol it does worst on.

Adjacent lengths on that curve differ by 0.03 to 0.05 for reasons that belong to one fold, not
to the length, so any argmax over it -- or any threshold on its raw steps -- chooses noise.
`plateau_knee` therefore reads a smoothed curve and asks where the returns stop:

1. **smooth** the floor with a running median, so no single length can choose or block;
2. **propose** lengths with multi-Kneedle (the `kneeliverse` library) on the smoothed curve's
   Pareto front -- the lengths where the rate of improvement bends -- plus the smoothed
   maximum itself, after which by definition nothing improves;
3. **choose** the first proposed length after which the smoothed floor gains at most ``delta``
   over the next ``window`` lengths.

A knee alone lands where the *rate* of improvement first bends, five to nine terms in, well
before the curve levels off; a threshold alone reads noise. Together they choose the start of
the first sustained plateau, which is the shortest equation that has stopped improving. When
every knee is still followed by real gains, that start is the smoothed maximum.

E3-MAX is not put forward as an equation to read: it is the same configuration under the wider
grammar, at the raw maximum of its floor (`floor_argmax`), bounding what the additive form can
reach.

Study chapter: [3. Term generation and selection][study-chapter] -- the rationale, in prose.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/03-term-selection.md
"""

from __future__ import annotations

import kneeliverse.kneedle as kneedle
import kneeliverse.multi_knee as multi_knee
import numpy as np
import polars as pl

#: Every protocol a length is judged on; `floor_curve` takes the minimum over those present.
JUDGED_PROTOCOLS = ("r2_in_sample", "r2_loo_dataset", "r2_loo_model", "r2_loo_cell")

#: multi-Kneedle's recursion settings: stop splitting a segment once a straight line explains
#: it to this coefficient of determination ...
KNEE_FIT_THRESHOLD = 0.001
#: ... or once it has fewer points than this.
KNEE_MIN_POINTS = 3


def complexity(arity: int, n_terms: int) -> int:
    """Feature slots an equation spends: a term of arity ``a`` names ``a`` raw features.

    **Not a count of fitted coefficients** -- those number ``n_terms + 1``. This charges for how
    much of the *grammar* an equation uses, which is what tells the same term count at arity 2
    apart from arity 3.
    """
    return arity * n_terms


def floor_curve(curve: pl.DataFrame) -> np.ndarray:
    """The **worst** of every protocol present, per length. This is what a length is judged on.

    The minimum rather than a median, because the weakest protocol is almost always DHO -- the
    cell where neither the dataset nor the model has been seen, which is what the study's
    headline is about -- and a rule that reports the strictest protocol but selects on a looser
    summary selects on a different quantity from the one it publishes. The minimum is always
    some protocol's own number.
    """
    columns = [name for name in JUDGED_PROTOCOLS if name in curve.columns]
    if not columns:
        raise ValueError("curve carries none of the protocol columns")
    return np.column_stack([curve[name].to_numpy() for name in columns]).min(axis=1)


def smoothed(values: np.ndarray, width: int) -> np.ndarray:
    """A centred running median of ``width`` points, shrinking at the ends.

    A median rather than a mean because the noise here is spiky: one held-out dataset outside
    the convex hull of the other nineteen craters a single length, and a mean would spread that
    crater onto its neighbours where a median ignores it. The same holds for a single lucky
    length, which is what a raw argmax picks.
    """
    if width < 1:
        raise ValueError("the smoothing width must be positive")
    half = width // 2
    return np.array([np.median(values[max(index - half, 0) : index + half + 1]) for index in range(len(values))])


def knee_lengths(curve: pl.DataFrame, smoothing: int) -> list[int]:
    """The lengths multi-Kneedle proposes on the smoothed floor's Pareto front, ascending.

    The front keeps only lengths that improve on every shorter one, so a knee is always a
    length worth its terms; the knees are where the front's rate of improvement bends.
    """
    lengths = curve["n_terms"].to_numpy()
    floor = smoothed(floor_curve(curve), smoothing)
    front = [index for index in range(len(floor)) if index == 0 or floor[index] > floor[:index].max()]
    if len(front) < KNEE_MIN_POINTS + 1:
        return []
    points = np.column_stack([lengths[front], floor[front]]).astype(np.float64)
    found = multi_knee.multi_knee(kneedle.knee, points, t1=KNEE_FIT_THRESHOLD, t2=KNEE_MIN_POINTS)
    return sorted({int(lengths[front[int(index)]]) for index in found})


def plateau_start(curve: pl.DataFrame, *, delta: float, window: int, smoothing: int) -> tuple[int, str]:
    """The equation length and how it was reached: ``"knee"`` or ``"maximum"``.

    Candidates are the lengths `knee_lengths` proposes and the smoothed maximum. The chosen one
    is the first, in increasing length, whose smoothed floor gains at most ``delta`` over the
    next ``window`` lengths. The maximum always qualifies -- nothing after it gains -- so the
    rule always returns; a maximum closer to the horizon than ``window`` is still returned,
    and `experiment.selection_summary` flags a length that sits at the horizon.
    """
    if delta < 0.0 or window < 1:
        raise ValueError("delta must be non-negative and the window positive")
    lengths = curve["n_terms"].to_numpy()
    floor = smoothed(floor_curve(curve), smoothing)
    peak = int(lengths[int(np.argmax(floor))])
    position = {int(length): index for index, length in enumerate(lengths)}
    for candidate in sorted({*knee_lengths(curve, smoothing), peak}):
        if candidate == peak:
            return peak, "maximum"
        index = position[candidate]
        ahead = floor[index + 1 : index + 1 + window]
        if len(ahead) == window and float(ahead.max() - floor[index]) <= delta:
            return candidate, "knee"
    return peak, "maximum"  # unreachable: the peak is always a candidate


def plateau_knee(curve: pl.DataFrame, *, delta: float, window: int, smoothing: int) -> int:
    """The equation length: the start of the first sustained plateau. **The length rule.**

    See `plateau_start`, which also says whether the length is a knee or the smoothed maximum.
    """
    return plateau_start(curve, delta=delta, window=window, smoothing=smoothing)[0]


def protocol_spread(curve: pl.DataFrame) -> np.ndarray:
    """How far a length falls from its fit to its worst protocol: ``IS - floor``.

    **Reported, not selected on.** An equation that fits well and transfers badly is a
    different object from one that does both moderately, and the floor alone cannot tell them
    apart. Folding the spread into the selection would need a weight between level and spread,
    which is a free parameter; the two are reported side by side instead.
    """
    if "r2_in_sample" not in curve.columns:
        raise ValueError("a spread needs the IS column to measure the drop from")
    return curve["r2_in_sample"].to_numpy() - floor_curve(curve)


def floor_argmax_index(curve: pl.DataFrame) -> int:
    """The *row position* of `floor_argmax`'s length, for callers indexing the curve."""
    return int(np.argmax(floor_curve(curve)))


def floor_argmax(curve: pl.DataFrame) -> int:
    """The length maximising the raw floor: **E3-MAX's rule**, a capability bound.

    No smoothing and no complexity penalty, because this equation is not put forward to be
    read -- it bounds what the additive form can reach, and a bound should not be discounted
    for being long. Ties go to the shorter equation. A result at the horizon means the bound
    is the horizon's, not the data's; `experiment` reports that.
    """
    return int(curve["n_terms"].to_numpy()[floor_argmax_index(curve)])
