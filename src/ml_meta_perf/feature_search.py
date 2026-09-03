"""Searching `descriptors.CANDIDATE_FEATURES` down to a clean `MODEL_FEATURES` set.

The unit being chosen here is a **feature set**, not a fitted equation. `fit.fit` and
`selection.Selector` already choose which *terms* enter one equation, honestly, inside
each cross-validation fold (`validate.cross_validate_path`'s docstring is explicit that
term selection has to happen inside the fold or the score is leaked). This module adds
one layer above that: which *raw columns* are even offered to the term builder in the
first place, which is a decision term selection never makes for you -- a beam search
happily builds terms over 61 candidate columns and reports whichever ones its equation
happened to use, but "whichever ones ended up in one equation" is not the same claim as
"this is the feature set worth defending in front of a committee".

Four ideas, in order:

**The objective scores in the E3 frame** (dataset features always present, mirrors
`experiment.run_e3`), not model-features-alone. An earlier version of this module scored
model-features-alone (mirroring `run_e2`) to keep trials cheap, and that was a real
mistake, not just a simplification: with no dataset features in the library, no
dataset x model cross term (ratio, product, sum_ratio) is ever built, so a model feature
that is weak *by itself* but valuable as `dataset_feature / model_feature` scores as
worthless and gets dropped -- backwards, given `02-equation-form.md`'s own finding that
the dataset x model interaction is "the exact combination carrying E3's entire lift over
E1". Scoring with the dataset features fixed in the library (varying only which model
features join them) lets the search actually see that value. `max_arity=2` at search time
(no three-feature `sum_ratio`) keeps the now-larger library affordable per trial; the
final recommended set still gets one full, honest E1/E2/E3 comparison at `max_arity=3`
through the unmodified `experiment` module before it is trusted.

**The search space is multi-objective** (R2, feature count), not a single subset scored
against a hand-picked parsimony penalty. `descriptors.Registry.resolve_conflicts` keeps
every candidate subset free of same-concept duplicates *before* it is ever scored, so the
search is choosing among already-non-redundant sets rather than hoping a beam search's
collinearity guard cleans up after it.

**The final recommendation is a stability selection**, not one search's report. Running
the search once and reporting its own winner is exactly the pattern
`cross_validate_path` refuses to allow at the term level, moved up one layer: the subset
that looks best on the data used to choose it is optimistic about that same data.
Following Meinshausen & Bühlmann (2010), the search is repeated over many random
subsamples of the 20 datasets, and the recommended set is whatever survives a strong
majority of repeats.

Study chapter: none yet -- this module supports a recommendation for `data.MODEL_FEATURES`,
not a result. See the repository's plan file (this session) for the two-phase process:
Phase A produces the report this module writes; Phase B (separate) wires the winner into
`data.py`, the shipped corpus and `experiment.py`'s tuned defaults.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from ml_meta_perf.data import DATASET_COLUMN, DATASET_FEATURES, columns_as_arrays, groups, target
from ml_meta_perf.descriptors import Registry
from ml_meta_perf.experiment import Configuration, run_equation
from ml_meta_perf.terms import build_library
from ml_meta_perf.validate import baseline_group_mean, cross_validate_path, score

#: Deliberately smaller than `experiment.DEFAULT_E3` -- this runs thousands of times
#: during a search, `DEFAULT_E3` runs once at the end through the untouched pipeline.
#: `SEARCH_MAX_ARITY` is 2, not `DEFAULT_E3`'s 3: with the 12 dataset features always in
#: the library, a three-feature `sum_ratio` grammar over dataset+model is already the
#: production-scale library at every trial, which is affordable once, not thousands of
#: times. The two-feature grammar still reaches every dataset x model *pairwise* ratio
#: and product, which is where the interaction value actually lives.
SEARCH_PENALTY = 1.0
SEARCH_POOL_SIZE = 80
SEARCH_MAX_TERMS_BASE = 8
SEARCH_MAX_TERMS_PER_FEATURE = 2
SEARCH_MAX_TERMS_CAP = 24
SEARCH_BEAM_WIDTH = 4
SEARCH_MAX_ARITY = 2


@dataclass(frozen=True)
class SubsetScore:
    """One candidate feature set's leave-one-dataset-out score."""

    features: tuple[str, ...]
    r2_loo_dataset: float
    n_features: int


def evaluate_subset(
    features: frozenset[str] | Sequence[str],
    frame: pl.DataFrame,
    *,
    dataset_features: tuple[str, ...] = DATASET_FEATURES,
    penalty: float = SEARCH_PENALTY,
    pool_size: int = SEARCH_POOL_SIZE,
    beam_width: int = SEARCH_BEAM_WIDTH,
    max_arity: int = SEARCH_MAX_ARITY,
) -> SubsetScore:
    """Score one model-feature subset by leave-one-dataset-out R2, **with the dataset
    features always in the library** -- this mirrors `experiment.run_e3`, not `run_e2`.

    Scoring model-features-alone (no dataset features in the library) would never build a
    single dataset x model cross term, so a model feature that is weak by itself but
    valuable as e.g. ``dataset_feature / model_feature`` would score as worthless and get
    dropped -- exactly backwards, since that interaction is documented
    (`02-equation-form.md`) as carrying most of E3's lift over E1. Pass
    ``dataset_features=()`` to recover the cheaper model-alone score for diagnostics.

    An empty ``features`` is still a legal point in this space -- "no model feature is
    worth keeping" is a claim the search should be able to reach -- and with
    ``dataset_features`` also empty this scores against the per-dataset-mean baseline
    `validate.baseline_group_mean` gives E1's ceiling; with ``dataset_features`` non-empty
    it still fits and scores the resulting dataset-only equation rather than falling back,
    since that is E1's own score and a perfectly legal (if unlikely to win) point on the
    front.

    The term budget is `SEARCH_MAX_TERMS_BASE` (roughly what E1 alone needs) plus
    `SEARCH_MAX_TERMS_PER_FEATURE` per model feature, capped: dataset terms need their own
    budget regardless of subset size, and each added model feature buys a little more
    room for the cross terms it can now form.
    """
    ordered = tuple(sorted(features))
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)

    if not ordered and not dataset_features:
        baseline = baseline_group_mean(truth, datasets)
        return SubsetScore((), r2_loo_dataset=score(truth, baseline).r2, n_features=0)

    columns = columns_as_arrays(frame, dataset_features + ordered)
    try:
        library = build_library(dataset_features, ordered, columns, max_arity=max(1, max_arity))
    except ValueError:
        # `Library.__init__` raises rather than returning empty (`terms.py:511`) when
        # every candidate term fails admissibility or the stability filter -- a real,
        # reachable outcome here, not just there: several of the AI-proposed descriptors
        # are near-zero for every model but one, and a lone sparse feature with no other
        # column to share the library with can fail the z-score spike check outright.
        # That is "this subset is unusable", scored the same as any other unusable one.
        baseline = baseline_group_mean(truth, datasets)
        return SubsetScore(ordered, r2_loo_dataset=score(truth, baseline).r2, n_features=len(ordered))

    max_terms = min(
        SEARCH_MAX_TERMS_CAP,
        SEARCH_MAX_TERMS_BASE + len(ordered) * SEARCH_MAX_TERMS_PER_FEATURE,
        len(library),
    )
    path = cross_validate_path(
        library, columns, truth, datasets,
        max_terms=max_terms, penalty=penalty, pool_size=pool_size, beam_width=beam_width,
    )
    if not path:
        baseline = baseline_group_mean(truth, datasets)
        return SubsetScore(ordered, r2_loo_dataset=score(truth, baseline).r2, n_features=len(ordered))

    # The best size this subset reaches, not a fixed one -- picking "how many terms this
    # subset's equation should have" is exactly what the knee/Pareto machinery in
    # `selection.py` does for the final equation; inside the search loop the simpler
    # "best of the sizes actually built" answers the same question without the extra cost
    # of curve-shape detection on every one of thousands of trials.
    best_r2 = max(path[size].scores(truth).r2 for size in path)
    return SubsetScore(ordered, r2_loo_dataset=best_r2, n_features=len(ordered))


def _selected(booleans: dict[str, int], registry: Registry) -> frozenset[str]:
    """Raw boolean draws, repaired to respect the exclusion groups.

    Both search backends draw one boolean per admissible candidate and pass it through
    this same repair step, so a subset's exclusion-group cleanliness never depends on
    which optimizer proposed it.
    """
    raw = frozenset(name for name, keep in booleans.items() if keep)
    return registry.resolve_conflicts(raw)


def optuna_objective(
    trial: object, registry: Registry, frame: pl.DataFrame, *, dataset_features: tuple[str, ...]
) -> tuple[float, int]:
    """One Optuna trial: draw a boolean per admissible candidate, repair, score.

    ``trial`` is typed ``object`` rather than ``optuna.Trial`` so this module does not
    require optuna at import time -- only `optuna_search` does, imported locally, so
    `evaluate_subset` and the stability-selection machinery around it stay usable
    (and testable) without the `search` extra installed.
    """
    booleans = {
        name: trial.suggest_categorical(name, [0, 1])  # pyright: ignore[reportAttributeAccessIssue]
        for name in registry.admissible
    }
    resolved = _selected(booleans, registry)
    result = evaluate_subset(resolved, frame, dataset_features=dataset_features)
    return result.r2_loo_dataset, result.n_features


def optuna_search(
    registry: Registry,
    frame: pl.DataFrame,
    *,
    dataset_features: tuple[str, ...] = DATASET_FEATURES,
    n_trials: int = 150,
    seed: int = 0,
) -> pl.DataFrame:
    """Multi-objective search (maximise R2, minimise feature count) via Optuna's NSGA-II.

    Returns the Pareto front as a table: one row per non-dominated trial, its feature
    tuple, R2 and count. `NSGAIISampler` is used because the space is genuinely discrete
    and multi-objective -- there is no continuous relaxation to fall back on here, unlike
    `pyblindopt_search`. ``dataset_features`` defaults to the real 12 (scoring in the E3
    frame, see `evaluate_subset`); pass a smaller tuple (or ``()``) to cut per-trial cost
    for a smoke test.
    """
    import optuna  # pyright: ignore[reportMissingImports]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, registry, frame, dataset_features=dataset_features),
        n_trials=n_trials, show_progress_bar=False,
    )

    rows: list[dict[str, object]] = []
    for trial in study.best_trials:
        resolved = _selected(trial.params, registry)
        rows.append(
            {
                "features": ", ".join(sorted(resolved)),
                "n_features": trial.values[1],
                "r2_loo_dataset": trial.values[0],
            }
        )
    return pl.DataFrame(rows).sort("n_features") if rows else pl.DataFrame(
        schema={"features": pl.Utf8, "n_features": pl.Int64, "r2_loo_dataset": pl.Float64}
    )


def pyblindopt_search(
    registry: Registry,
    frame: pl.DataFrame,
    *,
    dataset_features: tuple[str, ...] = DATASET_FEATURES,
    lambdas: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1),
    n_pop: int = 20,
    n_iter: int = 25,
    seed: int = 0,
) -> pl.DataFrame:
    """A scalarised sweep via pyBlindOpt's Differential Evolution, OBLESA-seeded.

    pyBlindOpt optimizes a single continuous objective, so the discrete choice is encoded
    as one gene per admissible candidate in [0, 1] (thresholded at 0.5, then repaired
    through the same `Registry.resolve_conflicts` `optuna_search` uses) and the
    R2-vs-count trade is scalarised as ``-R2 + lambda * n_features``. Sweeping ``lambdas``
    traces out a front comparable to Optuna's, from an independently implemented
    optimizer -- agreement between the two is the point, not either one's raw score.
    ``dataset_features`` is forwarded to `evaluate_subset` exactly as in `optuna_search`.
    """
    import pyBlindOpt.de as de  # pyright: ignore[reportMissingImports]
    import pyBlindOpt.init as pbo_init  # pyright: ignore[reportMissingImports]

    admissible = registry.admissible
    bounds = np.array([[0.0, 1.0]] * len(admissible))
    rng = np.random.default_rng(seed)

    def decode(vector: np.ndarray) -> frozenset[str]:
        booleans = {name: int(value >= 0.5) for name, value in zip(admissible, vector, strict=True)}
        return _selected(booleans, registry)

    rows: list[dict[str, object]] = []
    for lam in lambdas:
        def objective(vector: np.ndarray, lam: float = lam) -> float:
            # One individual in, one score out -- deliberately not vectorised over a
            # whole population. `utils.compute_objective` tries calling this with the
            # full (n_pop, d) matrix first ("optimistic vectorization") and only falls
            # back to one call per row if that raises; each call here does a real
            # cross-validated fit, which cannot be vectorised across candidate subsets
            # anyway, so writing it any other way just makes the optimistic path an
            # expensive, silently-caught failure on every generation instead of a cheap
            # one-time miss.
            result = evaluate_subset(decode(vector), frame, dataset_features=dataset_features)
            return -result.r2_loo_dataset + lam * result.n_features

        initial = pbo_init.oblesa(objective, bounds, n_pop=n_pop, seed=rng.integers(0, 2**31 - 1))
        optimizer = de.DifferentialEvolution(
            objective, bounds, population=initial, n_pop=n_pop, n_iter=n_iter,
            seed=rng.integers(0, 2**31 - 1),
        )
        best_vector, _ = optimizer.optimize()
        resolved = decode(best_vector)
        result = evaluate_subset(resolved, frame, dataset_features=dataset_features)
        rows.append(
            {
                "lambda": lam,
                "features": ", ".join(sorted(resolved)),
                "n_features": result.n_features,
                "r2_loo_dataset": result.r2_loo_dataset,
            }
        )
    return pl.DataFrame(rows)


def stability_selection(
    registry: Registry,
    frame: pl.DataFrame,
    *,
    dataset_features: tuple[str, ...] = DATASET_FEATURES,
    repeats: int = 30,
    hold_out_fraction: float = 0.25,
    trials_per_repeat: int = 40,
    seed: int = 0,
) -> pl.DataFrame:
    """Selection frequency of every admissible candidate across resampled repeats.

    Each repeat holds out a random slice of datasets *before* searching, so a repeat's
    winning subset is never chosen using rows that repeat could not otherwise score on --
    the same leakage `cross_validate_path` already refuses at the term level, one layer
    up. The winner within a repeat is the knee of that repeat's Pareto front: the smallest
    feature count past which R2 stops paying for itself, exactly the trade
    `selection.knee_index` already makes for equation length.

    Returns one row per candidate, sorted by selection frequency descending.
    """
    from ml_meta_perf.selection import knee_index

    rng = np.random.default_rng(seed)
    all_datasets = np.unique(groups(frame, DATASET_COLUMN))
    tally: dict[str, int] = dict.fromkeys(registry.admissible, 0)

    for repeat in range(repeats):
        held_out = rng.choice(
            all_datasets, size=max(1, round(len(all_datasets) * hold_out_fraction)), replace=False
        )
        kept_frame = frame.filter(~pl.col(DATASET_COLUMN).is_in(held_out.tolist()))

        front = optuna_search(
            registry, kept_frame, dataset_features=dataset_features,
            n_trials=trials_per_repeat, seed=int(rng.integers(0, 2**31 - 1)),
        )
        if front.height == 0:
            continue
        sizes = front["n_features"].to_numpy().astype(float)
        scores = front["r2_loo_dataset"].to_numpy()
        winner_index = knee_index(sizes, scores) if front.height >= 3 else int(np.argmax(scores))
        winning_features = front["features"][winner_index]
        for name in (f.strip() for f in winning_features.split(",") if f.strip()):
            tally[name] = tally.get(name, 0) + 1

        logging.getLogger(__name__).debug("stability repeat %d/%d done", repeat + 1, repeats)

    rows = [
        {"feature": name, "selected": count, "frequency": count / repeats}
        for name, count in sorted(tally.items(), key=lambda item: item[1], reverse=True)
    ]
    return pl.DataFrame(rows)


def recommend_features(frequency: pl.DataFrame, *, threshold: float = 0.7) -> tuple[str, ...]:
    """Candidates selected in at least ``threshold`` of the stability-selection repeats.

    Already free of exclusion-group duplicates: every repeat's winner passed through
    `Registry.resolve_conflicts` before being tallied, so nothing here needs to re-check
    that two recommended features never share a concept.
    """
    if frequency.height == 0:
        return ()
    survivors = frequency.filter(pl.col("frequency") >= threshold).sort("frequency", descending=True)
    return tuple(survivors["feature"].to_list())


#: A reduced-budget *arity-3* configuration -- unlike `SEARCH_MAX_ARITY=2` above, dropping
#: to arity 2 was measured (this session) to rank candidates differently from
#: `DEFAULT_E3` itself, which defeats the point of searching. Cutting `pool_size` and
#: `max_terms` instead costs much less fidelity for the same speedup: at six features this
#: configuration reaches 0.455/0.446 (LOO-dataset/LOO-model) against `DEFAULT_E3`'s
#: 0.478/0.428 in roughly a third of the time.
GREEDY_SEARCH_CONFIG = Configuration(
    max_abs_zscore=3.0, penalty=20.0, pool_size=250, max_terms=16, headline_terms=16, beam_width=4, max_arity=3
)


@dataclass(frozen=True)
class GreedyStep:
    """One step of `greedy_forward_search`: the feature just added and the score reached."""

    added: str | None
    features: tuple[str, ...]
    loo_dataset_r2: float
    loo_model_r2: float
    combined: float


def greedy_forward_search(
    registry: Registry,
    frame: pl.DataFrame,
    *,
    dataset_features: tuple[str, ...] = DATASET_FEATURES,
    config: Configuration = GREEDY_SEARCH_CONFIG,
    max_features: int = 15,
    min_improvement: float = 0.002,
) -> list[GreedyStep]:
    """Stepwise-forward feature selection against the real fitting pipeline, not a proxy.

    `optuna_search`/`pyblindopt_search` score a cheap arity-2 library to keep a
    combinatorial search over dozens of candidates affordable. That was measured (this
    session) to rank candidates differently from `experiment.run_equation` itself --
    several of its favoured subsets scored *worse* than the shipped `MODEL_FEATURES` once
    checked against the real arity-3 pipeline. A proxy that disagrees with the thing it is
    a proxy for is worse than no proxy, so this scores every candidate with
    `run_equation` directly, at `GREEDY_SEARCH_CONFIG`'s reduced-but-still-arity-3 budget.
    That trade only works because the search itself is greedy and small: with a
    correlation-and-group-pruned candidate pool (`Registry.reduce_to_representatives`,
    `Registry.with_correlation_conflicts`) there are few enough candidates that
    ``O(remaining)`` real fits per step is affordable, where it would not be for a
    combinatorial search over the same pool.

    The climbed score is the **mean of leave-one-dataset-out and leave-one-model-out R2**,
    not leave-one-dataset-out alone. `06-results.md` documents leave-one-model-out as the
    weaker of the two for the shipped features (0.428 against 0.478); a search that only
    rewards the axis already doing well has no reason to find features that close that
    specific gap.

    At each step every not-yet-selected admissible candidate is tried once, alone, added
    to the current selection; the one giving the largest improvement to the combined score
    is kept. Stops when no remaining candidate improves it by at least ``min_improvement``
    or ``max_features`` is reached -- greedy, not exhaustive, which is the trade that makes
    real-pipeline scoring affordable at all. Returns the full step history, starting from
    the dataset-only (empty model-feature) equation, so the caller can see the marginal
    contribution of every feature the search kept, not only the final set.
    """

    def score(features: tuple[str, ...]) -> tuple[float, float]:
        result = run_equation(frame, dataset_features, features, config, "greedy")
        return float(result.cross_validated["loo_dataset"]["r2"]), float(result.cross_validated["loo_model"]["r2"])

    selected: list[str] = []
    remaining = list(registry.admissible)
    loo_d, loo_m = score(())
    history = [
        GreedyStep(added=None, features=(), loo_dataset_r2=loo_d, loo_model_r2=loo_m, combined=(loo_d + loo_m) / 2)
    ]

    while remaining and len(selected) < max_features:
        best_candidate: str | None = None
        best_scores: tuple[float, float, float] | None = None
        for candidate in remaining:
            trial = tuple(sorted((*selected, candidate)))
            loo_d, loo_m = score(trial)
            combined = (loo_d + loo_m) / 2
            if best_scores is None or combined > best_scores[2]:
                best_candidate, best_scores = candidate, (loo_d, loo_m, combined)
        assert best_candidate is not None and best_scores is not None  # remaining is non-empty here

        if best_scores[2] - history[-1].combined < min_improvement:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        history.append(
            GreedyStep(
                added=best_candidate, features=tuple(sorted(selected)),
                loo_dataset_r2=best_scores[0], loo_model_r2=best_scores[1], combined=best_scores[2],
            )
        )
    return history
