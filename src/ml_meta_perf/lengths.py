"""Choosing the arity **and** the number of terms: rules that read one table of candidates.

`ml_meta_perf.selection` reads one curve, at the arity the study was configured with. The
question E3-Valid answers is wider: *the smallest explainable equation that does nearly as well
as the best*, and how small is a joint statement about the grammar (how many raw features a term
may name, ``arity``) and the length (``n_terms``). Both spend the reader's attention, so a
candidate costs its ``complexity = arity * n_terms`` feature slots (`selection.complexity`) and
what it earns is its worst-protocol R2, the floor (`selection.floor_curve`).

A rule reads the table of ``(arity, n_terms)`` candidates and returns one of them. Every rule here
is scale-free -- R2 is a fraction of variance, the complexity axis is rescaled to the unit square
where a distance is taken, and no constant is tuned to a corpus -- and none selects on a raw
argmax, because adjacent lengths differ by 0.03 to 0.05 for reasons that belong to one fold.

* ``floor_argmax`` -- the reference, E3-MAX's rule: the best floor, shortest on ties. Not a
  candidate for E3-Valid: it is the ceiling the others are scored against.
* ``plateau_knee`` -- today's rule (`selection.plateau_start`) at one fixed arity.
* ``pareto_knee`` -- the Pareto front of (complexity, smoothed floor) over every arity, and the
  knee of that front by Kneedle (the user's `kneeliverse` library).
* ``pareto_plateau`` -- the same front, multi-Kneedle proposing knees, and the first knee after
  which the front gains at most ``delta`` over the next ``window`` front points. Measured and
  rejected (unstable on short fronts), so a study file cannot name it.
* ``pareto_within_delta`` -- the cheapest front point within ``delta`` of the front's best floor.
* ``plateau_pareto`` -- the plateau rule at every arity, then the cheapest pick within ``delta`` of the best.
* ``pareto_elbow`` -- the front's point farthest from the chord joining its ends, in the unit
  square (the maximum-distance elbow).

Chapter 7, section 7.6 has the definitions; the evidence is `.agents/notes/length-rules.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import kneeliverse.kneedle as kneedle
import kneeliverse.multi_knee as multi_knee
import numpy as np
import polars as pl

from ml_meta_perf.selection import (
    KNEE_FIT_THRESHOLD,
    KNEE_MIN_POINTS,
    complexity,
    floor_curve,
    plateau_start,
    smoothed,
)


@dataclass(frozen=True, order=True)
class Pick:
    """The candidate a rule chose."""

    arity: int
    n_terms: int

    @property
    def complexity(self) -> int:
        return complexity(self.arity, self.n_terms)


def candidate_table(curves: Mapping[int, pl.DataFrame], smoothing: int) -> pl.DataFrame:
    """One row per ``(arity, n_terms)``: its complexity, its raw floor, and its floor smoothed along the length.

    ``curves`` maps each arity to the cross-validated curve of that grammar (`experiment.fit_path`).
    Smoothing is a running median along the length *within* an arity, the only direction in which
    neighbours are comparable equations.
    """
    if not curves:
        raise ValueError("a table needs at least one arity")
    rows: list[pl.DataFrame] = []
    for arity in sorted(curves):
        floor = floor_curve(curves[arity])
        rows.append(
            pl.DataFrame(
                {
                    "arity": arity,
                    "n_terms": curves[arity]["n_terms"].to_numpy(),
                    "complexity": [complexity(arity, int(n)) for n in curves[arity]["n_terms"]],
                    "floor": floor,
                    "smooth": smoothed(floor, smoothing),
                }
            )
        )
    return pl.concat(rows)


def pareto_front(table: pl.DataFrame) -> pl.DataFrame:
    """The candidates no cheaper candidate matches: strictly better smoothed floor than every cheaper one.

    Sorted by complexity. Candidates of equal complexity keep the better smoothed floor, then the
    one with fewer terms (the lower arity), so the front is a function of the table alone.
    """
    ordered = table.sort(["complexity", "smooth", "n_terms"], descending=[False, True, False])
    keep: list[int] = []
    best = -np.inf
    last = None
    for index, (cost, value) in enumerate(zip(ordered["complexity"], ordered["smooth"], strict=True)):
        if cost == last:
            continue
        last = cost
        if value > best:
            keep.append(index)
            best = value
    return ordered[keep]


def _pick(row: Mapping[str, object]) -> Pick:
    return Pick(int(row["arity"]), int(row["n_terms"]))  # pyright: ignore[reportArgumentType]


def floor_argmax(table: pl.DataFrame, **_extra: float) -> Pick:
    """The best raw floor; ties go to the lower complexity. The reference every rule is scored against."""
    best = table.filter(pl.col("floor") == pl.col("floor").max()).sort(["complexity", "n_terms"])
    return _pick(best.row(0, named=True))


def plateau_knee(
    table: pl.DataFrame, *, delta: float, window: int, smoothing: int, fixed_arity: int, **_extra: float
) -> Pick:
    """`selection.plateau_start` at one arity: today's rule, the baseline."""
    column = table.filter(pl.col("arity") == fixed_arity).sort("n_terms")
    if column.is_empty():
        raise ValueError(f"the table has no candidates at arity {fixed_arity}")
    curve = pl.DataFrame({"n_terms": column["n_terms"], "r2_in_sample": column["floor"]})
    length, _how = plateau_start(curve, delta=delta, window=window, smoothing=smoothing)
    return Pick(fixed_arity, length)


def _front_points(front: pl.DataFrame) -> np.ndarray:
    return np.column_stack([front["complexity"].to_numpy(), front["smooth"].to_numpy()]).astype(np.float64)


def pareto_knee(table: pl.DataFrame, **_extra: float) -> Pick:
    """The Kneedle knee of the Pareto front; the front's end when it has no knee."""
    front = pareto_front(table)
    if front.height > KNEE_MIN_POINTS:
        index = kneedle.knee(_front_points(front))
        if index is not None:
            return _pick(front.row(int(index), named=True))
    return _pick(front.row(front.height - 1, named=True))


def pareto_plateau(table: pl.DataFrame, *, delta: float, window: int, **_extra: float) -> Pick:
    """The first multi-Kneedle knee of the front after which the front gains at most ``delta`` over ``window`` points.

    The end of the front always qualifies (nothing follows it), so the rule always returns, like
    `selection.plateau_start`.
    """
    front = pareto_front(table)
    last = front.height - 1
    candidates = {last}
    if front.height > KNEE_MIN_POINTS:
        found = multi_knee.multi_knee(kneedle.knee, _front_points(front), t1=KNEE_FIT_THRESHOLD, t2=KNEE_MIN_POINTS)
        candidates |= {int(index) for index in found}
    smooth = front["smooth"].to_numpy()
    for index in sorted(candidates):
        ahead = smooth[index + 1 : index + 1 + window]
        if index == last or (len(ahead) == window and float(ahead.max() - smooth[index]) <= delta):
            return _pick(front.row(index, named=True))
    return _pick(front.row(last, named=True))  # unreachable: the last point is always a candidate


def pareto_within_delta(table: pl.DataFrame, *, delta: float, **_extra: float) -> Pick:
    """The cheapest front point whose smoothed floor is within ``delta`` of the front's best."""
    front = pareto_front(table)
    near = front.filter(pl.col("smooth") >= pl.col("smooth").max() - delta)
    return _pick(near.row(0, named=True))


def plateau_pareto(table: pl.DataFrame, *, delta: float, window: int, smoothing: int, **_extra: float) -> Pick:
    """The study's plateau rule at every arity, then the cheapest of those picks within ``delta`` of the best.

    Each arity's length is the validated `selection.plateau_start`; the arities are then compared on
    the Pareto front of their picks, so a grammar that buys the same floor with fewer feature slots wins.
    """
    picks = [
        plateau_knee(table, delta=delta, window=window, smoothing=smoothing, fixed_arity=int(a))
        for a in sorted(table["arity"].unique())
    ]
    chosen = pl.DataFrame(
        [
            {**table.filter((pl.col("arity") == p.arity) & (pl.col("n_terms") == p.n_terms)).row(0, named=True)}
            for p in picks
        ]
    )
    near = chosen.filter(pl.col("smooth") >= pl.col("smooth").max() - delta).sort(["complexity", "n_terms"])
    return _pick(near.row(0, named=True))


def pareto_elbow(table: pl.DataFrame, **_extra: float) -> Pick:
    """The front's point farthest from the chord joining its two ends, both axes rescaled to ``[0, 1]``."""
    front = pareto_front(table)
    if front.height < 3:
        return _pick(front.row(front.height - 1, named=True))
    points = _front_points(front)
    span = points.max(axis=0) - points.min(axis=0)
    unit = (points - points.min(axis=0)) / np.where(span > 0.0, span, 1.0)
    chord = unit[-1] - unit[0]
    length = float(np.hypot(*chord))
    if length == 0.0:
        return _pick(front.row(0, named=True))
    distance = np.abs(chord[0] * (unit[:, 1] - unit[0, 1]) - chord[1] * (unit[:, 0] - unit[0, 0])) / length
    return _pick(front.row(int(np.argmax(distance)), named=True))


#: The rules a study file may name (`config.LENGTH_METHODS` lists the same names).
LENGTH_RULES: dict[str, Callable[..., Pick]] = {
    "plateau_knee": plateau_knee,
    "pareto_knee": pareto_knee,
    "pareto_elbow": pareto_elbow,
    "pareto_within_delta": pareto_within_delta,
    "plateau_pareto": plateau_pareto,
}

#: Every rule that was measured, including ``pareto_plateau``, which is unstable on short fronts and not selectable.
MEASURED_RULES: dict[str, Callable[..., Pick]] = {**LENGTH_RULES, "pareto_plateau": pareto_plateau}
