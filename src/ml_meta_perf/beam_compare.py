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

The verdict a policy can earn is therefore one of three, and only the first is a reason to
change anything:

* **better** -- the paired interval lies entirely below zero, so the policy's per-dataset
  errors are smaller on this corpus by more than resampling explains;
* **tie** -- the interval spans zero. A tie is not a win. If a variant ties the incumbent it
  is more machinery for no measured gain, and the incumbent stays;
* **worse** -- the interval lies entirely above zero.
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
from ml_meta_perf.validate import cross_validate_fixed_form, paired_comparison

#: How the paired interval is read. Both bounds on one side of zero is a verdict; anything
#: spanning it is a tie, and a tie means the incumbent stays.
BETTER, TIE, WORSE = "better", "tie", "worse"


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
    #: mistaken for one that is simply better.
    seconds: float


def run_policy(
    frame: pl.DataFrame,
    config: Configuration,
    model_features: tuple[str, ...],
    policy: BeamPolicy = VANILLA,
) -> PolicyRun | None:
    """Fit one policy at one configuration and keep its per-dataset errors.

    The equation is pruned and refitted exactly as the published path does, then
    cross-validated with its **form fixed** -- the study's reported protocol. Re-running
    selection inside the folds would ask whether the *policy* generalises, which is a
    different and much noisier question than whether the equation it found does.
    """
    import time

    columns = columns_as_arrays(frame, DATASET_FEATURES + model_features)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)

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
    elapsed = time.perf_counter() - started
    if config.headline_terms not in result.equations:
        return None

    equation = prune(result.equations[config.headline_terms], columns, truth, penalty=config.penalty)
    if not equation.terms:
        return None

    from ml_meta_perf.stats import r2_score

    paths = {}
    for label, labels in (("loo_dataset", datasets), ("loo_model", groups(frame, MODEL_COLUMN))):
        paths[label] = cross_validate_fixed_form(
            library, columns, truth, labels, {config.headline_terms: equation}, penalty=config.penalty
        )
    held = paths["loo_dataset"][config.headline_terms]
    order = list(np.unique(datasets))
    return PolicyRun(
        policy=policy.name,
        configuration=config,
        features=model_features,
        n_terms=len(equation.terms),
        terms=tuple(term.name for term in equation.terms),
        r2_in_sample=r2_score(truth, equation.predict(columns)),
        r2_loo_dataset=r2_score(truth, held.predictions),
        r2_loo_model=r2_score(truth, paths["loo_model"][config.headline_terms].predictions),
        fold_mae=np.array([held.per_fold[label].mae for label in order]),
        seconds=elapsed,
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
        "n_terms": challenger.n_terms,
        "seconds": challenger.seconds,
        "speedup": incumbent.seconds / challenger.seconds if challenger.seconds > 0 else float("nan"),
        "shared_terms": len(set(challenger.terms) & set(incumbent.terms)),
        "terms": " | ".join(challenger.terms),
    }


def compare_all(runs: list[PolicyRun], incumbent_name: str = "vanilla") -> pl.DataFrame:
    """Every policy paired against the incumbent, most improved first.

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
    return pl.DataFrame(rows).sort("mae_gain", descending=True)
