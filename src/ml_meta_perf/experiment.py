"""Fit one equation and validate it: the search, the cross-validated curve, the chosen length.

An equation is fitted on all observed rows by `fit_path` (search once up to a horizon, then
cross-validate every length with the form fixed), and one length is published by `finalise`.
E1, E2 and E3 differ **only** in which features they may use, so they share one function,
`run_equation`, one configuration (`config.Configuration`) and one length rule
(``selection.method``); E3-MAX is the same configuration at the capability arity at the raw
maximum of its worst-protocol curve (`run_capability`): a capability bound on the additive form,
not an equation put forward to be read. `ml_meta_perf.study` runs the equations a study lists.

Study chapter: [4. The equation][study-chapter] -- the rationale, in prose.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/07-generic-targets.md
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ml_meta_perf.config import Configuration, SelectionConfig, default_schema, load_config
from ml_meta_perf.data import Schema, columns_as_arrays, groups, target
from ml_meta_perf.lengths import LENGTH_RULES, candidate_table
from ml_meta_perf.model import Equation
from ml_meta_perf.search import prune, pruning_threshold, search
from ml_meta_perf.selection import floor_argmax, plateau_knee
from ml_meta_perf.terms import Library, build_library
from ml_meta_perf.validate import (
    CrossValidation,
    Scores,
    cross_validate_doubly_held_out,
    cross_validate_fixed_form,
    fold_selections,
    score,
    term_stability,
)

#: The protocol a `FittedPath.paths` key stands for, as the user names it.
PROTOCOL_NAMES = {"loo_dataset": "LODO", "loo_model": "LOMO", "loo_cell": "DHO"}


@dataclass
class EquationReport:
    """One equation together with everything said about it."""

    equation: Equation
    #: The selected length: `selection.plateau_knee` for E1, E2 and E3-Valid, the raw floor
    #: maximum for E3-MAX. Every output reads the result from here.
    n_terms: int
    #: The grammar this equation was searched under -- the configuration's for E1, E2 and
    #: E3-Valid, ``selection.capability_arity`` for E3-MAX.
    arity: int
    in_sample: dict[str, float | int]
    curve: pl.DataFrame
    cross_validated: dict[str, dict[str, float | int]] = field(default_factory=dict)
    stability: pl.DataFrame | None = None
    #: The full cross-validated path per protocol, kept so later tables can reuse it.
    #: ``leakage_demonstration`` and ``decision_quality`` ask for the same folds at the
    #: same settings, and recomputing them was a third of the study's runtime.
    paths: dict[str, dict[int, CrossValidation]] = field(default_factory=dict)


def _curve(
    sizes: tuple[int, ...],
    in_sample: dict[int, Scores],
    paths: dict[str, dict[int, CrossValidation]],
    truth: np.ndarray,
) -> pl.DataFrame:
    """One row per equation length, carrying every metric under every protocol.

    IS gets the same three metrics as the cross-validated columns, not just R²:
    an error curve that omits the IS line cannot show how far apart fit and
    transfer run, which is the whole point of plotting them together.
    """
    rows: list[dict[str, object]] = []
    for size in sizes:
        if size not in in_sample:
            continue
        row: dict[str, object] = {
            "n_terms": size,
            "r2_in_sample": in_sample[size].r2,
            "mae_in_sample": in_sample[size].mae,
            "smape_in_sample": in_sample[size].smape,
        }
        for label, path in paths.items():
            scores = path[size].scores(truth)
            row[f"r2_{label}"] = scores.r2
            row[f"mae_{label}"] = scores.mae
            row[f"smape_{label}"] = scores.smape
        rows.append(row)
    return pl.DataFrame(rows)


@dataclass(frozen=True)
class FittedPath:
    """Everything `run_equation` computes before a length is chosen.

    Searching and cross-validating every length is the expensive half of an equation; choosing
    one length and finalising it is cheap. Keeping the two apart means a length rule only ever
    reads a curve that was computed once, and lets `ml_meta_perf.sweep` score hundreds of
    configurations without finalising any of them.
    """

    library: Library
    columns: dict[str, np.ndarray]
    truth: np.ndarray
    datasets: np.ndarray
    equations: dict[int, Equation]
    paths: dict[str, dict[int, CrossValidation]]
    curve: pl.DataFrame
    config: Configuration
    schema: Schema


def fit_path(
    frame: pl.DataFrame,
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    config: Configuration,
    name: str,
    *,
    schema: Schema | None = None,
) -> FittedPath:
    """Search one grammar and cross-validate every length on it, choosing none of them."""
    schema = schema or default_schema()
    columns = columns_as_arrays(frame, dataset_features + model_features)
    truth = target(frame, schema)
    datasets = groups(frame, schema.dataset_column)

    library = build_library(
        dataset_features,
        model_features,
        columns,
        max_arity=config.max_arity,
        max_abs_zscore=config.max_abs_zscore,
        exclude=schema.excluded,
    )
    result = search(
        library,
        truth,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
        name=name,
        bounds=schema.fit_bounds,
        label=schema.label,
    )
    # Fixed form: the terms are chosen once, here, and only the weights are refit in each
    # fold. See `validate.cross_validate_fixed_form` for why that is the reported protocol.
    paths: dict[str, dict[int, CrossValidation]] = {}
    models = groups(frame, schema.model_column) if schema.model_column else datasets
    if "LODO" in schema.protocols:
        paths["loo_dataset"] = cross_validate_fixed_form(
            library, columns, truth, datasets, result.equations, penalty=config.penalty
        )
    if "LOMO" in schema.protocols:
        paths["loo_model"] = cross_validate_fixed_form(
            library, columns, truth, models, result.equations, penalty=config.penalty
        )
    # DHO at **every** length, not just the selected one: every selection rule reads the
    # complete curve, and E3-MAX selects on this protocol through `selection.floor_curve`.
    if "DHO" in schema.protocols:
        paths["loo_cell"] = cross_validate_doubly_held_out(
            library, columns, truth, datasets, models, result.equations, penalty=config.penalty
        )
    in_sample = {k: score(truth, eq.predict(columns)) for k, eq in result.equations.items()}
    curve = _curve(tuple(sorted(result.equations)), in_sample, paths, truth)
    return FittedPath(library, columns, truth, datasets, result.equations, paths, curve, config, schema)


def finalise(fitted: FittedPath, size: int) -> EquationReport:
    """Publish one length of a fitted path: prune it, and measure how stable its form is."""
    if size not in fitted.equations:
        raise ValueError(f"selected equation length {size} is unavailable")
    config, truth = fitted.config, fitted.truth
    # The same folds with selection re-run inside them, kept only for `stability`: the share
    # of the equation's terms that survive when one dataset is removed. That is what
    # licenses fixing the form, and it is not part of any reported score.
    reselected = fold_selections(
        fitted.library,
        truth,
        fitted.datasets,
        n_terms=size,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
    )
    equation = prune(
        fitted.equations[size],
        fitted.columns,
        truth,
        penalty=config.penalty,
        min_contribution=pruning_threshold(fitted.schema.fit_bounds, truth),
    )

    return EquationReport(
        equation=equation,
        n_terms=size,
        arity=config.max_arity,
        in_sample=score(truth, equation.predict(fitted.columns)).as_dict(),
        curve=fitted.curve,
        cross_validated={
            label: path[size].scores(truth).as_dict() | path[size].dispersion() for label, path in fitted.paths.items()
        },
        stability=term_stability(reselected, tuple(term.name for term in equation.terms)),
        paths=fitted.paths,
    )


def _resolve(config: Configuration | None, selection: SelectionConfig | None) -> tuple[Configuration, SelectionConfig]:
    """Fill whichever of the two was not given from ``config/study.json``."""
    if config is None or selection is None:
        study = load_config()
        return config or study.search, selection or study.selection
    return config, selection


def select_length(curve: pl.DataFrame, selection: SelectionConfig) -> int:
    """The length rule every published equation shares: `selection.plateau_knee`."""
    return plateau_knee(curve, delta=selection.delta, window=selection.window, smoothing=selection.smoothing)


def run_equation(
    frame: pl.DataFrame,
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    config: Configuration,
    selection: SelectionConfig,
    name: str,
    *,
    schema: Schema | None = None,
) -> EquationReport:
    """Fit one equation and validate it. The three equations differ **only** in the
    features they may draw on, and this is the single code path that says so.

    Every one of them is fitted on all 484 rows and scored on all 484 rows, under all four
    protocols, and has its length chosen by the same rule. That uniformity is the point: the
    gaps between E1, E2 and E3 are only evidence about what each half of the meta-data is
    worth if nothing else differs between them -- not the fitting scale, not the protocol,
    not the denominator of the R2, not the way the length was picked.

    **The length is derived, not asserted**: `select_length` reads it off the complete
    cross-validated curve.
    """
    fitted = fit_path(frame, dataset_features, model_features, config, name, schema=schema)
    if selection.method == "plateau_knee":
        return finalise(fitted, select_length(fitted.curve, selection))
    # The Pareto rules choose the arity too: every grammar up to the configured one is fitted, and
    # the rule reads their curves as one table of (arity, n_terms) candidates (`ml_meta_perf.lengths`).
    by_arity = {
        arity: fitted
        if arity == config.max_arity
        else fit_path(
            frame, dataset_features, model_features, dataclasses.replace(config, max_arity=arity), name, schema=schema
        )
        for arity in range(1, config.max_arity + 1)
    }
    table = candidate_table({arity: path.curve for arity, path in by_arity.items()}, selection.smoothing)
    pick = LENGTH_RULES[selection.method](
        table,
        delta=selection.delta,
        window=selection.window,
        smoothing=selection.smoothing,
        fixed_arity=config.max_arity,
    )
    return finalise(by_arity[pick.arity], pick.n_terms)


def run_capability(
    frame: pl.DataFrame,
    config: Configuration | None = None,
    selection: SelectionConfig | None = None,
    *,
    schema: Schema | None = None,
) -> EquationReport:
    """E3-MAX: E3's configuration at the configured capability arity, at the raw maximum of its floor.

    **A capability measurement, not a recommendation.** It answers the question E3-Valid cannot
    answer about itself -- whether the additive form is out of reach or merely out of the
    readable range -- so it takes `selection.floor_argmax` rather than the length rule, with
    no smoothing and no complexity penalty. Only the arity differs from E3-Valid.
    """
    config, selection = _resolve(config, selection)
    wider = dataclasses.replace(config, max_arity=selection.capability_arity)
    schema = schema or default_schema()
    fitted = fit_path(frame, schema.dataset_features, schema.model_features, wider, schema.tag("E3-MAX"), schema=schema)
    return finalise(fitted, floor_argmax(fitted.curve))
