"""Deciding whether a beam policy actually beats the published one.

The sweep in `ml_meta_perf.equation_search` ranks configurations by a weighted objective, and
that objective is a *shortlisting device*. It has already, once, put a nine-term equation on
top that a paired test called significantly worse -- the weight on `stability` outvoted an
accuracy gap. So the sweep proposes and this module decides, using the same rule the rest of
the study uses for two-configuration comparisons: `validate.paired_comparison`, an exact sign
test plus a bootstrap over the twenty held-out datasets, on **per-dataset mean absolute
error** under leave-one-dataset-out.

Per-dataset MAE rather than pooled R2, for two reasons the project learned the hard way. A
per-fold R2 divides by that fold's own target variance, and three datasets here have almost
none -- `5G_Slicing`'s twenty-five models all score about 0.986 -- so an ordinary error there
returns a large negative number and would dominate any paired test. And a difference of two
pooled numbers over twenty folds is not a measurement at all; it produced three wrong
conclusions in a single session.

The verdict a policy can earn is one of three, and it is a verdict about **accuracy only**:

* **better** -- the paired interval lies entirely below zero, so the policy's per-dataset
  errors are smaller on this corpus by more than resampling explains;
* **tie** -- the interval spans zero;
* **worse** -- the interval lies entirely above zero.

**Accuracy is no longer the only axis.** The corpus is going to grow, so a policy that matches
the incumbent's accuracy while finishing sooner is worth having, and "a tie is more machinery
for no gain" is true only at equal cost. So every row also carries **`standing`**, which is the
column to read: one word for how the policy compares with the incumbent on accuracy, cost and
the four protocols together.

* **better** -- wins the paired test and gives up nothing on the strictest protocol;
* **cheaper** -- ties on accuracy and is faster by more than `SPEED_TOLERANCE`;
* **equal** -- indistinguishable on both axes. Most rows land here, and it means the policy is
  more machinery for nothing;
* **worse** -- loses the paired test, or buys its speed with the worst protocol.

`frontier` is the second, cross-row column: the rows no *other* option beats on both axes,
**with the incumbent itself included as an option** at (0.0 gain, 1.0x). Leaving it out was the
first version's mistake -- it made the least-bad losing policy look like a winner, because
nothing in the table beat it except the thing the table was not showing.

**A verdict is a statement about one operating point.** Measured on 2026-09-08: at six terms
`prune-relative-0.02` beat the incumbent on 16 of 20 folds, and at the published fifteen it
returned the identical fifteen terms -- while `cap-2`, the largest positive dR2 at seven terms,
lost 0.203 of leave-one-dataset-out R2 at fifteen. What a pruner removes at step 6 of a beam is
not what it removes at step 15. Score both operating points or claim neither.

**Four protocols, because that is the study's claim.** The equation is put forward as holding
its R2 *and its ranking* whether nothing is held out, the dataset is, the model is, or both are
-- which is exactly what the opaque regressors fail to do. A beam policy has to be judged the
same way, so every run is scored under all four and the table carries the spread between them.
Leave-one-cell is 476 ridge solves against leave-one-dataset-out's 20: too expensive for the
728,640-point sweep, negligible here, where a handful of configurations are refitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ml_meta_perf.beam import VANILLA, BeamPolicy
from ml_meta_perf.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    columns_as_arrays,
    groups,
    target,
)
from ml_meta_perf.experiment import Configuration
from ml_meta_perf.fit import fit, prune
from ml_meta_perf.terms import build_library
from ml_meta_perf.validate import (
    cross_validate_doubly_held_out,
    cross_validate_fixed_form,
    paired_comparison,
)

#: How the paired interval is read. Both bounds on one side of zero is a verdict; anything
#: spanning it is a tie, and a tie means the incumbent stays *on accuracy*. Cost is read from
#: `speedup` and `frontier`, which is a separate question with a separate answer.
BETTER, TIE, WORSE = "better", "tie", "worse"

#: The four protocols every run is scored under, weakest first. The order is the order the
#: leakage argument runs in: nothing held out, the dataset held out, the model held out, both
#: held out. `PROTOCOLS[-1]` is the only one under which the equation and an opaque regressor
#: are denied the same things.
PROTOCOLS = ("in_sample", "loo_dataset", "loo_model", "loo_cell")

#: How many times the fit is timed. One measurement of a sub-second fit is not a measurement:
#: the first table produced here reported speedups from 0.86x to 1.04x off single 0.35-0.41 s
#: runs, a spread the same policy shows against itself. `seconds` reports the **best** of these
#: -- the minimum is the run least interrupted by everything else on the machine, which is the
#: usual choice for a timing under a scheduler -- and `seconds_median` is carried beside it so
#: a policy whose timing is merely erratic cannot hide behind one lucky run.
TIMING_REPEATS = 5

#: How much faster a policy has to measure before the speed counts as real. With
#: `TIMING_REPEATS` best-of timings, three policies that produce the *identical* equation at
#: `DEFAULT_E3` still spread from 0.978x to 1.002x, so anything inside a few percent is the
#: clock rather than the beam. Five percent is the floor a `candidate` has to clear; it is a
#: convention, and the honest way to lower it is more repeats, not a smaller number.
SPEED_TOLERANCE = 0.05


@dataclass(frozen=True)
class PolicyRun:
    """One policy fitted at one configuration, with the per-dataset errors it produced."""

    policy: str
    configuration: Configuration
    features: tuple[str, ...]
    n_terms: int
    terms: tuple[str, ...]
    r2_in_sample: float
    r2_loo_dataset: float
    r2_loo_model: float
    #: Mean absolute error per held-out dataset, in corpus order of `np.unique`. This is what
    #: the paired test consumes; a pooled number cannot be paired.
    fold_mae: np.ndarray
    #: Wall time for the fit alone, so a policy that buys accuracy by spending time is not
    #: mistaken for one that is simply better. The **best** of `repeats` timings; see
    #: `TIMING_REPEATS`.
    seconds: float
    #: R2 under every protocol in `PROTOCOLS`. `r2_in_sample`, `r2_loo_dataset` and
    #: `r2_loo_model` are the same numbers, kept as named fields because the paired test and
    #: the published-path test read them directly.
    r2_loo_cell: float = float("nan")
    #: Per-dataset mean average precision under each protocol -- the study's ranking metric,
    #: since Spearman saturates here and NDCG@3 saturates harder. Ranking has to hold up across
    #: the protocols too, so a policy cannot trade it away for R2 unnoticed.
    ranking_map: tuple[float, ...] = ()
    #: Median of the timing repeats, beside the best in `seconds`.
    seconds_median: float = float("nan")
    #: How many times the fit was timed. 1 means the timing is a single sample and no speed
    #: claim should be made from it.
    repeats: int = 1

    @property
    def r2_by_protocol(self) -> dict[str, float]:
        """R2 under each of `PROTOCOLS`, in that order."""
        return {
            "in_sample": self.r2_in_sample,
            "loo_dataset": self.r2_loo_dataset,
            "loo_model": self.r2_loo_model,
            "loo_cell": self.r2_loo_cell,
        }

    @property
    def r2_spread(self) -> float:
        """Max minus min R2 across the protocols -- **the study's own headline claim, as one
        number.** An equation that means what it says loses little when the dataset, the model
        or both are withheld; a forest loses everything by the fourth column. Smaller is
        better, and a policy that buys leave-one-dataset-out R2 by widening this has not
        improved the equation.
        """
        values = [value for value in self.r2_by_protocol.values() if np.isfinite(value)]
        return float(max(values) - min(values)) if values else float("nan")

    @property
    def r2_worst(self) -> float:
        """The weakest protocol's R2. Read beside `r2_spread`: a policy can be flat and bad."""
        values = [value for value in self.r2_by_protocol.values() if np.isfinite(value)]
        return float(min(values)) if values else float("nan")

    @property
    def ranking_worst(self) -> float:
        """The weakest protocol's ranking MAP, on the same reasoning as `r2_worst`."""
        values = [value for value in self.ranking_map if np.isfinite(value)]
        return float(min(values)) if values else float("nan")


def run_policy(
    frame: pl.DataFrame,
    config: Configuration,
    model_features: tuple[str, ...],
    policy: BeamPolicy = VANILLA,
    *,
    repeats: int = TIMING_REPEATS,
) -> PolicyRun | None:
    """Fit one policy at one configuration and score it under all four protocols.

    The equation is pruned and refitted exactly as the published path does, then
    cross-validated with its **form fixed** -- the study's reported protocol. Re-running
    selection inside the folds would ask whether the *policy* generalises, which is a
    different and much noisier question than whether the equation it found does.

    The fit is run `repeats` times and the best time kept. The search is deterministic, so
    every repeat produces the same equation and the only thing varying is the clock; `repeats`
    is a timing budget, not a seed sweep. Pass ``repeats=1`` where the timing does not matter
    and the fit is expensive -- and then do not quote a speedup.
    """
    import time

    columns = columns_as_arrays(frame, DATASET_FEATURES + model_features)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    timings: list[float] = []
    library = result = None
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        try:
            library = build_library(
                DATASET_FEATURES,
                model_features,
                columns,
                max_arity=config.max_arity,
                max_abs_zscore=config.max_abs_zscore,
            )
            result = fit(
                library,
                truth,
                max_terms=config.max_terms,
                penalty=config.penalty,
                pool_size=config.pool_size,
                beam_width=config.beam_width,
                policy=policy,
            )
        except ValueError:
            return None
        timings.append(time.perf_counter() - started)
    assert library is not None and result is not None
    if config.headline_terms not in result.equations:
        return None

    equation = prune(result.equations[config.headline_terms], columns, truth, penalty=config.penalty)
    if not equation.terms:
        return None

    from ml_meta_perf.equation_search import ranking_scores
    from ml_meta_perf.stats import r2_score

    equations = {config.headline_terms: equation}
    predictions: dict[str, np.ndarray] = {"in_sample": equation.predict(columns)}
    folded = {
        label: cross_validate_fixed_form(library, columns, truth, labels, equations, penalty=config.penalty)
        for label, labels in (("loo_dataset", datasets), ("loo_model", models))
    }
    held = folded["loo_dataset"][config.headline_terms]
    predictions["loo_dataset"] = held.predictions
    predictions["loo_model"] = folded["loo_model"][config.headline_terms].predictions
    # Both groups of every cell withheld: 476 solves against the 20 above, and the only
    # protocol under which the equation is denied what an opaque regressor is denied.
    doubly = cross_validate_doubly_held_out(
        library, columns, truth, datasets, models, equations, penalty=config.penalty
    )
    predictions["loo_cell"] = doubly[config.headline_terms].predictions

    order = list(np.unique(datasets))
    return PolicyRun(
        policy=policy.name,
        configuration=config,
        features=model_features,
        n_terms=len(equation.terms),
        terms=tuple(term.name for term in equation.terms),
        r2_in_sample=r2_score(truth, predictions["in_sample"]),
        r2_loo_dataset=r2_score(truth, predictions["loo_dataset"]),
        r2_loo_model=r2_score(truth, predictions["loo_model"]),
        r2_loo_cell=r2_score(truth, predictions["loo_cell"]),
        # Ranking is scored per held-out dataset under every protocol, so the claim "R2 *and*
        # ranking hold up when both halves are withheld" is a column rather than an assertion.
        ranking_map=tuple(ranking_scores(truth, predictions[name], datasets)[0] for name in PROTOCOLS),
        fold_mae=np.array([held.per_fold[label].mae for label in order]),
        seconds=float(min(timings)),
        seconds_median=float(np.median(timings)),
        repeats=len(timings),
    )


def compare(incumbent: PolicyRun, challenger: PolicyRun) -> dict[str, object]:
    """One paired verdict, challenger against incumbent, on per-dataset MAE.

    ``mae_gain`` is **positive when the challenger wins** -- it is the mean reduction in
    per-dataset absolute error, with the sign handled by ``lower_is_better`` rather than by
    the caller. The verdict names the challenger throughout.
    """
    # `lower_is_better` puts the sign in one place: the result's `mean` and interval are then
    # positive when the *first* argument wins, whatever the metric's own polarity. Handling it
    # here instead is how a comparison table gets read backwards.
    result = paired_comparison(challenger.fold_mae, incumbent.fold_mae, lower_is_better=True)
    verdict = BETTER if result.low > 0.0 else WORSE if result.high < 0.0 else TIE
    return {
        "policy": challenger.policy,
        "verdict": verdict,
        "mae_gain": result.mean,
        "ci_low": result.low,
        "ci_high": result.high,
        "p_value": result.p_value,
        "wins": result.wins,
        "losses": result.losses,
        "folds": int(challenger.fold_mae.shape[0]),
        "r2_loo_dataset": challenger.r2_loo_dataset,
        "r2_loo_dataset_delta": challenger.r2_loo_dataset - incumbent.r2_loo_dataset,
        "r2_loo_model_delta": challenger.r2_loo_model - incumbent.r2_loo_model,
        "r2_loo_cell": challenger.r2_loo_cell,
        "r2_loo_cell_delta": challenger.r2_loo_cell - incumbent.r2_loo_cell,
        "r2_in_sample": challenger.r2_in_sample,
        "r2_in_sample_delta": challenger.r2_in_sample - incumbent.r2_in_sample,
        # The spread across the four protocols, and its change. A policy that lifts one
        # protocol by widening the spread has moved the equation away from what the study
        # claims for it, whatever the paired test on leave-one-dataset-out says.
        "r2_spread": challenger.r2_spread,
        "r2_spread_delta": challenger.r2_spread - incumbent.r2_spread,
        "r2_worst": challenger.r2_worst,
        "r2_worst_delta": challenger.r2_worst - incumbent.r2_worst,
        **{
            f"rank_map_{name}": value
            for name, value in zip(PROTOCOLS, challenger.ranking_map, strict=False)
        },
        "rank_map_worst": challenger.ranking_worst,
        "rank_map_worst_delta": challenger.ranking_worst - incumbent.ranking_worst,
        "n_terms": challenger.n_terms,
        "seconds": challenger.seconds,
        "seconds_median": challenger.seconds_median,
        "repeats": challenger.repeats,
        "speedup": incumbent.seconds / challenger.seconds if challenger.seconds > 0 else float("nan"),
        "shared_terms": len(set(challenger.terms) & set(incumbent.terms)),
        "terms": " | ".join(challenger.terms),
    }


def _standing(verdict: str, speedup: float, worst_delta: float) -> str:
    """One word for how a policy compares with the incumbent, over both axes and all four
    protocols. This is the column to read; `verdict` alone answers only the accuracy half.

    Order matters. A policy that buys its speed by giving up the strictest protocol is
    **worse** whatever the paired test on leave-one-dataset-out says -- that test is on one
    protocol, and the study's claim is about four.
    """
    if verdict == WORSE or (np.isfinite(worst_delta) and worst_delta < 0.0):
        return "worse"
    if verdict == BETTER:
        return "better"
    if np.isfinite(speedup) and speedup > 1.0 + SPEED_TOLERANCE:
        return "cheaper"
    return "equal"


def _frontier(gains: list[float], speedups: list[float]) -> list[bool]:
    """Which rows no other option beats on **both** accuracy and speed.

    Pareto, maximising both, with two corrections the first version needed.

    **The incumbent is one of the options.** Callers pass it in as (0.0, 1.0) -- doing nothing
    is always available, so a policy it dominates is not on any frontier worth the name.
    Without that point the least-bad of a set of losing policies came out marked `true`, which
    reads as a win and is the opposite of the truth.

    **Speed is compared in units of `SPEED_TOLERANCE`**, because a 1.008x that the clock
    produced must not dominate a 1.000x. Quantising is cruder than a proper interval and it is
    the same convention `_standing` uses, so the two columns cannot disagree.

    Ties on both axes stay on the frontier together: two options that measure identically are
    equally good, and dropping one would be an arbitrary choice dressed up as a result.
    """
    steps = [value / SPEED_TOLERANCE if np.isfinite(value) else value for value in speedups]
    keep = []
    for index, (gain, speed) in enumerate(zip(gains, steps, strict=True)):
        if not np.isfinite(gain) or not np.isfinite(speed):
            keep.append(False)
            continue
        dominated = any(
            other != index
            and np.isfinite(gains[other])
            and np.isfinite(steps[other])
            and gains[other] >= gain
            and round(steps[other]) >= round(speed)
            and (gains[other] > gain or round(steps[other]) > round(speed))
            for other in range(len(gains))
        )
        keep.append(not dominated)
    return keep


def compare_all(runs: list[PolicyRun], incumbent_name: str = "vanilla") -> pl.DataFrame:
    """Every policy paired against the incumbent, most improved first.

    Adds the two columns the verdict alone cannot carry. **`standing`** is the one to read --
    `better`, `cheaper`, `equal` or `worse`, over both axes and all four protocols; see
    `_standing`. **`frontier`** is the cross-row view, with the incumbent included among the
    options so a policy that doing nothing already beats cannot be marked `true`.

    Rows sort by standing, then by accuracy: anything actionable is at the top.

    Returns an empty frame rather than raising when the incumbent is missing: a sweep that
    lost its baseline row to a failed fit should report that it has no comparison, not crash
    after hours of cluster time.
    """
    incumbent = next((run for run in runs if run.policy == incumbent_name), None)
    if incumbent is None:
        return pl.DataFrame()
    rows = [compare(incumbent, run) for run in runs if run.policy != incumbent_name]
    if not rows:
        return pl.DataFrame()
    table = pl.DataFrame(rows)
    # The incumbent goes in last and comes back out: it is an option every policy competes
    # with, not a row of the report.
    gains = [float(value) for value in table["mae_gain"]] + [0.0]
    speedups = [float(value) for value in table["speedup"]] + [1.0]
    table = table.with_columns(
        pl.Series("frontier", _frontier(gains, speedups)[:-1]),
        pl.Series(
            "standing",
            [
                _standing(str(verdict), float(speedup), float(worst))
                for verdict, speedup, worst in zip(
                    table["verdict"], table["speedup"], table["r2_worst_delta"], strict=True
                )
            ],
        ),
    )
    rank = {"better": 0, "cheaper": 1, "equal": 2, "worse": 3}
    return table.with_columns(pl.col("standing").replace_strict(rank).alias("_rank")).sort(
        ["_rank", "mae_gain"], descending=[False, True]
    ).drop("_rank")
