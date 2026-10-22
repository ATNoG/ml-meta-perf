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

import math
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
        """Feature slots $c(a, n) = a \\, n$ spent by this candidate."""
        return complexity(self.arity, self.n_terms)


def candidate_table(
    curves: Mapping[int, pl.DataFrame], smoothing: int, details: Mapping[int, pl.DataFrame] | None = None
) -> pl.DataFrame:
    """One row per $(a, n)$: its complexity $c(a, n) = a \\, n$, its raw floor, and its floor smoothed along $n$.

    ``curves`` maps each arity $a$ to the cross-validated curve of that grammar (`experiment.fit_path`).
    Smoothing is a running median along the length *within* an arity, the only direction in which
    neighbours are comparable equations. ``details`` maps each arity to the per-length columns the
    resampling and information rules read (`experiment.candidate_details`); a rule that needs one
    that is absent says so.
    """
    if not curves:
        raise ValueError("a table needs at least one arity")
    rows: list[pl.DataFrame] = []
    for arity in sorted(curves):
        floor = floor_curve(curves[arity])
        frame = pl.DataFrame(
            {
                "arity": arity,
                "n_terms": curves[arity]["n_terms"].to_numpy(),
                "complexity": [complexity(arity, int(n)) for n in curves[arity]["n_terms"]],
                "floor": floor,
                "smooth": smoothed(floor, smoothing),
            }
        )
        if details is not None:
            frame = frame.join(details[arity], on="n_terms", how="left")
        rows.append(frame)
    return pl.concat(rows)


def pareto_front(table: pl.DataFrame) -> pl.DataFrame:
    """The Pareto front $\\mathcal{P} = \\{(a, n) : s(a, n) > s(a', n')\\ \\forall\\, c(a', n') < c(a, n)\\}$.

    Sorted by complexity $c(a, n) = a \\, n$. Candidates of equal complexity keep the better smoothed
    floor $s(a, n)$, then the one with fewer terms (the lower arity), so the front is a function of
    the table alone.
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
    """Convert a candidate table row into a typed `Pick`."""
    return Pick(int(row["arity"]), int(row["n_terms"]))  # pyright: ignore[reportArgumentType]


def floor_argmax(table: pl.DataFrame, **_extra: float) -> Pick:
    """The best raw floor $\\arg\\max_{(a,n)} \\operatorname{floor}(a, n)$; ties go to the lower complexity."""
    best = table.filter(pl.col("floor") == pl.col("floor").max()).sort(["complexity", "n_terms"])
    return _pick(best.row(0, named=True))


def plateau_knee(
    table: pl.DataFrame, *, delta: float, window: int, smoothing: int, fixed_arity: int, **_extra: float
) -> Pick:
    """`selection.plateau_start` at one fixed arity $a = \\text{fixed\\_arity}$: today's rule, the baseline."""
    column = table.filter(pl.col("arity") == fixed_arity).sort("n_terms")
    if column.is_empty():
        raise ValueError(f"the table has no candidates at arity {fixed_arity}")
    curve = pl.DataFrame({"n_terms": column["n_terms"], "r2_in_sample": column["floor"]})
    length, _how = plateau_start(curve, delta=delta, window=window, smoothing=smoothing)
    return Pick(fixed_arity, length)


def _front_points(front: pl.DataFrame) -> np.ndarray:
    """Extract the 2D coordinate matrix $(c_i, s_i) \\in \\mathbb{R}^{|\\mathcal{P}| \\times 2}$ of the Pareto front."""
    return np.column_stack([front["complexity"].to_numpy(), front["smooth"].to_numpy()]).astype(np.float64)


def pareto_knee(table: pl.DataFrame, **_extra: float) -> Pick:
    """The Kneedle knee of the Pareto front $\\mathcal{P}$; the front's end when it has no knee."""
    front = pareto_front(table)
    if front.height > KNEE_MIN_POINTS:
        index = kneedle.knee(_front_points(front))
        if index is not None:
            return _pick(front.row(int(index), named=True))
    return _pick(front.row(front.height - 1, named=True))


def pareto_plateau(table: pl.DataFrame, *, delta: float, window: int, **_extra: float) -> Pick:
    """The first multi-Kneedle knee of $\\mathcal{P}$ after which the front gains at most $\\delta$ over $w$ points.

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
    """The cheapest front point $(a, n) \\in \\mathcal{P}$ with $s(a, n) \\ge \\max_{\\mathcal{P}} s - \\delta$."""
    front = pareto_front(table)
    near = front.filter(pl.col("smooth") >= pl.col("smooth").max() - delta)
    return _pick(near.row(0, named=True))


def plateau_pareto(table: pl.DataFrame, *, delta: float, window: int, smoothing: int, **_extra: float) -> Pick:
    """The study's plateau rule at every arity $a$, then the cheapest of those picks within $\\delta$ of the best.

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
    """The front point farthest from the chord joining its two ends in the unit square $[0, 1]^2$:

    $$\\arg\\max_{i} \\frac{|(x_e - x_0)(y_i - y_0) - (y_e - y_0)(x_i - x_0)|}{\\sqrt{(x_e - x_0)^2 + (y_e - y_0)^2}}.$$
    """
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


# --- rules that read the shape of the front beyond the knee and the elbow -----------------------


def l_method(table: pl.DataFrame, **_extra: float) -> Pick:
    """The L-method (Salvador and Chan 2004): two straight lines fitted to the front, split where they fit best.

    For every split $s \\in \\{1,\\dots,m-3\\}$ of the $m$-point unit-square front, the left and right segments
    are each fitted by ordinary least squares, and the split minimising the point-weighted RMSE

    $$\\mathrm{RMSE}_{\\text{total}}(s) = \\frac{s+1}{m}\\,\\mathrm{RMSE}(0\\dots s)
    + \\frac{m-s-1}{m}\\,\\mathrm{RMSE}(s+1\\dots m-1)$$

    is the knee. Both axes are rescaled to $[0, 1]^2$ first, so the choice does not depend on the units.
    """
    front = pareto_front(table)
    if front.height < 4:
        return _pick(front.row(front.height - 1, named=True))
    points = _front_points(front)
    span = points.max(axis=0) - points.min(axis=0)
    unit = (points - points.min(axis=0)) / np.where(span > 0.0, span, 1.0)

    def rmse(part: np.ndarray) -> float:
        """Root mean squared error of an ordinary least-squares line fitted to ``part``."""
        x, y = part[:, 0], part[:, 1]  # at least two points, at distinct costs: the front is strictly increasing
        slope, intercept = np.polyfit(x, y, 1)
        return float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))

    count = len(unit)
    errors = {
        split: (split + 1) / count * rmse(unit[: split + 1]) + (count - split - 1) / count * rmse(unit[split + 1 :])
        for split in range(1, count - 2)
    }
    return _pick(front.row(min(errors, key=lambda split: errors[split]), named=True))


def bend_angle(table: pl.DataFrame, **_extra: float) -> Pick:
    """The front point where the front turns the most: largest turning angle on $[0, 1]^2$.

    For adjacent unit-square segment vectors $u_{i-1}, u_i$, the turning angle is

    $$\\theta_i = \\left|\\operatorname{atan2}(u_{i-1} \\times u_i,\\; u_{i-1} \\cdot u_i)\\right|,$$

    the discrete curvature of a bi-objective front: a point where moving either way along the front
    costs a lot in the other objective. The two ends have one neighbour only and are never chosen
    unless the front has fewer than three points.
    """
    front = pareto_front(table)
    if front.height < 3:
        return _pick(front.row(front.height - 1, named=True))
    points = _front_points(front)
    span = points.max(axis=0) - points.min(axis=0)
    unit = (points - points.min(axis=0)) / np.where(span > 0.0, span, 1.0)
    before, after = unit[1:-1] - unit[:-2], unit[2:] - unit[1:-1]
    turn = np.abs(np.arctan2(before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0], (before * after).sum(axis=1)))
    return _pick(front.row(int(np.argmax(turn)) + 1, named=True))


def log_slope(table: pl.DataFrame, **_extra: float) -> Pick:
    """The front point that buys the largest relative drop in error per feature slot spent.

    With error $e_i = \\max(1 - s_i, 10^{-12})$ and cost $c_i$, the score of front point $i \\ge 1$ is

    $$\\operatorname{score}_i = -\\frac{\\ln e_i - \\ln e_{i-1}}{c_i - c_{i-1}},$$

    the rule symbolic-regression libraries use to rank the members of a Pareto front by default. The cheapest
    point has no predecessor and is chosen only when the front has one point.
    """
    front = pareto_front(table)
    if front.height < 2:
        return _pick(front.row(0, named=True))
    error = np.maximum(1.0 - front["smooth"].to_numpy(), 1e-12)
    cost = front["complexity"].to_numpy().astype(np.float64)
    score = -np.diff(np.log(error)) / np.diff(cost)
    return _pick(front.row(int(np.argmax(score)) + 1, named=True))


# --- rules that use the noise of the held-out datasets ---------------------------------------


class MissingColumns(ValueError):
    """The table lacks columns a rule reads: it was built without the details."""


def _require(table: pl.DataFrame, rule: str, *columns: str) -> None:
    """Raise `MissingColumns` if ``table`` is missing any of ``columns`` required by ``rule``."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise MissingColumns(
            f"{rule} needs the columns {missing}: build the table with `candidate_table(..., details)`"
        )


def _losses(table: pl.DataFrame, rule: str) -> np.ndarray:
    """The $(G, M)$ matrix of normalised dataset squared errors $u_{g,c} = L_{c,g} / \\mathrm{SST}$.

    Column $c$ sums to $\\sum_{g=1}^{G} u_{g,c} = 1 - \\operatorname{floor}_c$, so the datasets are the units
    whose resampling says how far a floor could move with a different draw of datasets.
    """
    _require(table, rule, "loss", "sst")
    matrix = np.array(table["loss"].to_list(), dtype=np.float64).T
    return matrix / table["sst"].to_numpy()[None, :]


def _cheapest(table: pl.DataFrame, allowed: np.ndarray) -> Pick:
    """The cheapest candidate among those flagged, ties to fewer terms and then to the better floor."""
    kept = table.filter(pl.Series(allowed)).sort(["complexity", "n_terms", "smooth"], descending=[False, False, True])
    return _pick(kept.row(0, named=True))


def one_se(table: pl.DataFrame, *, se_multiple: float = 1.0, **_extra: float) -> Pick:
    """The one-standard-error rule: the cheapest candidate $c$ within $m$ standard errors of the best $c^\\ast$:

    $$\\sum_{g=1}^{G} u_{g,c} \\le \\sum_{g=1}^{G} u_{g,c^\\ast} + m \\sqrt{G}\\,\\operatorname{sd}_g(u_{g,c^\\ast}),
    \\qquad c^\\ast = \\arg\\min_c \\sum_{g=1}^{G} u_{g,c}.$$

    Candidates of equal cost keep the better floor.
    """
    losses = _losses(table, "one_se")
    totals = losses.sum(axis=0)
    best = int(np.argmin(totals))
    error = float(np.sqrt(len(losses)) * losses[:, best].std(ddof=1)) if len(losses) > 1 else 0.0
    return _cheapest(table, totals <= totals[best] + se_multiple * error)


def projection(table: pl.DataFrame, *, se_multiple: float = 1.0, **_extra: float) -> Pick:
    """The cheapest candidate $c$ not worse than the reference $c^\\ast$ by more than $m$ paired standard errors:

    $$\\sum_{g=1}^{G} (u_{g,c} - u_{g,c^\\ast}) \\le m \\sqrt{G}\\,\\operatorname{sd}_g(u_{g,c} - u_{g,c^\\ast}).$$

    The projection-predictive recipe (Piironen and Vehtari): the reference $c^\\ast = \\arg\\min_c \\sum_g u_{g,c}$
    is the candidate with the lowest total held-out error (E3-MAX's choice when the grammars agree). Pairing
    removes the dataset-to-dataset variation the two share.
    """
    losses = _losses(table, "projection")
    totals = losses.sum(axis=0)
    reference = int(np.argmin(totals))
    difference = losses - losses[:, [reference]]
    if len(losses) < 2:
        return _cheapest(table, np.ones(len(totals), dtype=bool))
    error = np.sqrt(len(losses)) * difference.std(axis=0, ddof=1)
    return _cheapest(table, difference.sum(axis=0) <= se_multiple * error)


def mcs_members(losses: np.ndarray, alpha: float, rounds: int, seed: int) -> np.ndarray:
    """The model confidence set $\\widehat{\\mathcal{M}}_{1-\\alpha}$ of Hansen, Lunde and Nason (2011).

    Starting from every candidate, the equal-predictive-ability hypothesis is tested with the range-of-means
    statistic $T_{\\max} = \\max_{i \\in \\mathcal{M}} t_i$, where
    $t_i = \\bar{d}_i / \\widehat{\\mathrm{sd}}(\\bar{d}_i)$
    is the bootstrap-studentised mean difference
    $\\bar{d}_i = \\frac{1}{G}\\sum_g (u_{g,i} - \\bar{u}_{g,\\mathcal{M}})$
    of candidate $i$ from the average of those left; while it is rejected at level $\\alpha$ the candidate with the
    largest $t_i$ is eliminated. The survivors cannot be told apart from the best.
    """
    groups, count = losses.shape
    draws = np.random.default_rng(seed).integers(0, groups, size=(rounds, groups))
    active = np.arange(count)
    while len(active) > 1:
        part = losses[:, active]
        relative = part - part.mean(axis=1, keepdims=True)
        observed = relative.mean(axis=0)
        boot = relative[draws].mean(axis=1)
        spread = np.sqrt(boot.var(axis=0, ddof=1)) + 1e-300
        statistic = (observed / spread).max()
        null = ((boot - observed) / spread).max(axis=1)
        if float((null >= statistic).mean()) >= alpha:
            break
        active = np.delete(active, int(np.argmax(observed / spread)))
    return active


def mcs(
    table: pl.DataFrame, *, alpha: float = 0.1, bootstrap_rounds: int = 500, seed: int = 0, **_extra: float
) -> Pick:
    """The cheapest member of the model confidence set $\\widehat{\\mathcal{M}}_{1-\\alpha}$ at level $\\alpha$."""
    losses = _losses(table, "mcs")
    members = mcs_members(losses, alpha, bootstrap_rounds, seed)
    allowed = np.zeros(losses.shape[1], dtype=bool)
    allowed[members] = True
    return _cheapest(table, allowed)


def stable_within_delta(table: pl.DataFrame, *, delta: float, stability_min: float = 0.9, **_extra: float) -> Pick:
    """The cheapest candidate within $\\delta$ of the best floor whose terms keep their sign across the datasets.

    ``sign_stability`` is the share of held-out datasets under which the weakest term of the equation keeps
    the sign it has in the full fit. Candidates that cannot be assessed (no LODO) pass; when none passes the
    stability filter the rule falls back to the cheapest within $\\delta$.
    """
    _require(table, "stable_within_delta", "sign_stability")
    smooth = table["smooth"].to_numpy()
    near = smooth >= smooth.max() - delta
    stable = (table["sign_stability"].fill_nan(1.0).fill_null(1.0) >= stability_min).to_numpy()
    return _cheapest(table, near & stable if bool((near & stable).any()) else near)


# --- information criteria and description length --------------------------------------------


def _fit_term(table: pl.DataFrame, rule: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return $(n,\\, n \\ln(\\mathrm{RSS} / n),\\, k)$ with $k = j + 1$ weights per candidate."""
    _require(table, rule, "rss", "n_rows", "sst")
    n = table["n_rows"].to_numpy().astype(np.float64)
    rss = np.maximum(table["rss"].to_numpy(), 1e-12 * table["sst"].to_numpy())
    return n, n * np.log(rss / n), table["n_terms"].to_numpy().astype(np.float64) + 1.0


def _lowest(table: pl.DataFrame, criterion: np.ndarray) -> Pick:
    """The candidate with the lowest criterion; within rounding of the lowest, the cheaper."""
    return _cheapest(table, criterion <= np.nanmin(criterion) + 1e-9 * (1.0 + abs(np.nanmin(criterion))))


def aic(table: pl.DataFrame, **_extra: float) -> Pick:
    """Akaike's criterion $\\mathrm{AIC} = n \\ln(\\mathrm{RSS}/n) + 2k$, with $k = j + 1$ weights (in-sample fit)."""
    _, fit, k = _fit_term(table, "aic")
    return _lowest(table, fit + 2.0 * k)


def aicc(table: pl.DataFrame, **_extra: float) -> Pick:
    """Corrected AIC $\\mathrm{AICc} = \\mathrm{AIC} + \\frac{2k(k+1)}{n - k - 1}$ ($+\\infty$ where $n \\le k+1$)."""
    n, fit, k = _fit_term(table, "aicc")
    room = n - k - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        penalty = np.where(room > 0.0, 2.0 * k * (k + 1.0) / room, np.inf)
    return _lowest(table, fit + 2.0 * k + penalty)


def bic(table: pl.DataFrame, **_extra: float) -> Pick:
    """Schwarz's Bayesian criterion $\\mathrm{BIC} = n \\ln(\\mathrm{RSS}/n) + k \\ln n$."""
    n, fit, k = _fit_term(table, "bic")
    return _lowest(table, fit + k * np.log(n))


def ebic(table: pl.DataFrame, **_extra: float) -> Pick:
    """Extended BIC (Chen and Chen 2008): $\\mathrm{EBIC} = \\mathrm{BIC} + 2 \\gamma \\ln \\binom{P}{j}$.

    $\\gamma = \\operatorname{clip}_{[0,1]}(1 - \\ln n / (2 \\ln P))$ charges for searching $j$ of $P$ terms.
    """
    _require(table, "ebic", "library")
    n, fit, k = _fit_term(table, "ebic")
    library = table["library"].to_numpy().astype(np.float64)
    j = table["n_terms"].to_numpy().astype(np.float64)
    gamma = np.clip(1.0 - np.log(n) / (2.0 * np.log(np.maximum(library, 2.0))), 0.0, 1.0)
    log_choices = _log_choices(library, j)
    return _lowest(table, fit + k * np.log(n) + 2.0 * gamma * log_choices)


def mdl(table: pl.DataFrame, **_extra: float) -> Pick:
    """Two-part minimum description length in nats:

    $$\\mathrm{MDL} = \\frac{n}{2} \\ln\\!\\left(\\frac{\\mathrm{RSS}}{n}\\right)
    + \\frac{k}{2} \\ln n + \\sum_{i=1}^{j} \\ell(t_i),$$

    where $\\ell(t_i)$ is `terms.description_length`.
    """
    _require(table, "mdl", "code")
    n, fit, k = _fit_term(table, "mdl")
    return _lowest(table, 0.5 * fit + 0.5 * k * np.log(n) + table["code"].to_numpy())


def _log_choices(library: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Compute $\\ln \\binom{P}{j} = \\ln \\Gamma(P+1) - \\ln \\Gamma(j+1) - \\ln \\Gamma(P-j+1)$ elementwise."""
    return np.array(
        [
            math.lgamma(p + 1.0) - math.lgamma(k + 1.0) - math.lgamma(max(p - k, 0.0) + 1.0)
            for p, k in zip(library, j, strict=True)
        ]
    )


#: The rules a study file may name, bar ``auto`` (`config.LENGTH_METHODS` lists the same names plus it).
SELECTABLE_RULES: dict[str, Callable[..., Pick]] = {
    "plateau_knee": plateau_knee,
    "pareto_knee": pareto_knee,
    "pareto_elbow": pareto_elbow,
    "pareto_within_delta": pareto_within_delta,
    "plateau_pareto": plateau_pareto,
    "l_method": l_method,
    "bend_angle": bend_angle,
    "log_slope": log_slope,
    "one_se": one_se,
    "projection": projection,
    "mcs": mcs,
    "stable_within_delta": stable_within_delta,
    "aic": aic,
    "aicc": aicc,
    "bic": bic,
    "ebic": ebic,
    "mdl": mdl,
}


def all_picks(table: pl.DataFrame, **params: float) -> dict[str, Pick]:
    """What every selectable rule picks from ``table``; a rule whose columns the table lacks is left out."""
    picks: dict[str, Pick] = {}
    for name, rule in SELECTABLE_RULES.items():
        try:
            picks[name] = rule(table, **params)
        except MissingColumns:
            continue
    return picks


def _smooth_of(table: pl.DataFrame, pick: Pick) -> float:
    """Look up the smoothed floor $s(a, n)$ of ``pick`` in ``table``."""
    row = table.filter((pl.col("arity") == pick.arity) & (pl.col("n_terms") == pick.n_terms))
    return float(row["smooth"][0])


def auto_choice(table: pl.DataFrame, *, tolerance: float = 0.0, **params: float) -> tuple[str, Pick, dict[str, Pick]]:
    """Run every rule and take the pick closest to the best score, then with fewer terms, then the lower arity.

    The score of a pick is its smoothed floor, so the pick closest to E3-MAX's is the one with the highest
    value; picks whose value is within ``tolerance`` of the highest count as equally close. Among those the
    fewest terms win, then the lowest arity, then the rule that comes first in `SELECTABLE_RULES`. With
    ``tolerance = 0`` only exact ties are broken by parsimony.
    Returns the winning rule's name, its pick and every rule's pick.
    """
    picks = all_picks(table, **params)
    value = {name: _smooth_of(table, pick) for name, pick in picks.items()}
    best = max(value.values())
    order = list(SELECTABLE_RULES)
    close = [name for name in picks if value[name] >= best - tolerance]
    winner = min(close, key=lambda name: (picks[name].n_terms, picks[name].arity, order.index(name)))
    return winner, picks[winner], picks


def auto(table: pl.DataFrame, *, tolerance: float = 0.0, **params: float) -> Pick:
    """`auto_choice`'s pick: the rule-agnostic choice."""
    return auto_choice(table, tolerance=tolerance, **params)[1]


def rule_table(table: pl.DataFrame, picks: Mapping[str, Pick], chosen: str) -> pl.DataFrame:
    """One row per rule that ran: what it picked, that pick's smoothed floor, and whether it was the one used."""
    return pl.DataFrame(
        [
            {
                "rule": name,
                "arity": pick.arity,
                "n_terms": pick.n_terms,
                "complexity": pick.complexity,
                "floor": _smooth_of(table, pick),
                "chosen": name == chosen,
            }
            for name, pick in picks.items()
        ]
    )


#: Every rule a study file may name.
LENGTH_RULES: dict[str, Callable[..., Pick]] = {**SELECTABLE_RULES, "auto": auto}

#: Every rule that was measured, including ``pareto_plateau``, which is unstable on short fronts and not selectable.
MEASURED_RULES: dict[str, Callable[..., Pick]] = {**SELECTABLE_RULES, "pareto_plateau": pareto_plateau}
