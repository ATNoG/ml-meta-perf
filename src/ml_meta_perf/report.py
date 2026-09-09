"""Turning a finished study into a written report, without a human in the loop.

The whole argument of this project is that the fitted equation can be *read*. That claim
would be worth very little if reading it still required an expert to sit down with the
weights and narrate what they mean -- the narration would then be the product, and the
equation merely its raw material. So every statement this module makes is derived from
the fitted object by arithmetic that is written down here, and the same equation always
produces the same report.

Two ideas carry the analysis.

*Standardised weights rank the terms.* The target is centred but never scaled during
fitting, so a standardised weight is already in MCC units: it is how far predicted MCC
moves when that term moves by one standard deviation of itself. That makes ``beta``
directly comparable across terms whose raw units have nothing to do with each other, and
it is the number used to decide which terms are major.

*Effects state what the terms are worth on this data.* A large weight on a term that
barely varies is not important, so ``effect`` -- the swing in predicted MCC across the
middle 80% of a term's observed range -- is reported next to every weight. The two
disagree often enough to be worth printing together.

Feature-level guidance is not re-derived here. It comes from `ml_meta_perf.practices`, which
measures direction empirically rather than reading signs off weights, because a feature
appearing in two terms or inside a denominator has no single sign to read.

Study chapter: [6. Best practices against the equation](../../assets/docs/06-practices.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl

from ml_meta_perf.attribution import classify, contributions
from ml_meta_perf.data import (
    ALL_FEATURES,
    DATASET_COLUMN,
    FEATURE_GLOSSARY,
    MODEL_COLUMN,
    corpus_summary,
    missing_cells,
    target_summary,
)
from ml_meta_perf.experiment import Configuration, Report
from ml_meta_perf.guidance import as_table as as_guidance_table
from ml_meta_perf.guidance import assess, equation_coverage, equation_evidence
from ml_meta_perf.guidance import render as render_guidance
from ml_meta_perf.model import Equation, direction
from ml_meta_perf.practices import render as render_practices
from ml_meta_perf.stats import spearman
from ml_meta_perf.terms import TRANSFORMS, Atom, Term

#: A term is *major* when it falls inside the smallest group of terms whose standardised
#: weights account for this share of the total weight mass. Set at four fifths because
#: that is where the tail of near-zero weights begins on this data, not because 80% has
#: any special status -- ``term_importance`` reports the cumulative share for every term
#: so a reader can draw the line somewhere else.
MAJOR_MASS = 0.8

#: How many terms get written out as sentences. A flat equation can flag most of its
#: terms as major, and twenty near-identical paragraphs is not a readable analysis --
#: the table carries the rest.
SENTENCE_LIMIT = 8

#: The grammar's operations, in the order ``operation_usage`` reports them.
OPERATIONS = ("atom", "ratio", "product", "sum_ratio", "ratio_of_sums")

#: How many raw features each operation names. Used to tell an operation the search
#: *declined* from one the arity cap never offered it.
OPERATION_ARITY = {"atom": 1, "ratio": 2, "product": 2, "sum_ratio": 3, "ratio_of_sums": 4}


def term_importance(
    equation: Equation,
    columns: dict[str, np.ndarray],
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    stability: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per term, ordered by standardised weight, with the major ones flagged.

    ``beta`` is the standardised weight: the MCC the term contributes per standard
    deviation of itself. ``effect`` is its 10th-to-90th-percentile contribution swing on
    the real data. ``share`` is ``|beta|`` as a fraction of the total, and ``cumulative``
    its running sum down the ranking, so ``major`` marks the leading terms that together
    reach `MAJOR_MASS`.

    Ranking on ``beta`` rather than on the raw weight is not cosmetic. Raw weights carry
    the units of whatever the term computes, so a term over instance counts and a term
    over class entropy have weights that cannot be compared at all; sorting on them would
    rank the terms by the size of their units.
    """
    if not equation.terms:
        return pl.DataFrame(
            schema={
                "rank": pl.Int64,
                "term": pl.String,
                "group": pl.String,
                "features": pl.String,
                "weight": pl.Float64,
                "beta": pl.Float64,
                "effect": pl.Float64,
                "share": pl.Float64,
                "cumulative": pl.Float64,
                "major": pl.Boolean,
                "stability": pl.Float64,
                "direction": pl.String,
            }
        )

    matrix = contributions(equation, columns)
    frequency = (
        dict(zip(stability["term"].to_list(), stability["frequency"].to_list(), strict=True))
        if stability is not None and stability.height
        else {}
    )

    betas = np.abs(np.asarray(equation.standardized_weights, dtype=np.float64))
    total = float(betas.sum())
    order = np.argsort(-betas)

    rows: list[dict[str, object]] = []
    running = 0.0
    for position, index in enumerate(order, start=1):
        beta = float(equation.standardized_weights[index])
        share = float(betas[index] / total) if total > 0.0 else 0.0
        running += share
        low, high = np.percentile(matrix[:, index], [10.0, 90.0])
        term = equation.terms[index]
        rows.append(
            {
                "rank": position,
                "term": term.name,
                "group": classify(term.features, dataset_features, model_features),
                "features": ", ".join(term.features),
                "weight": float(equation.weights[index]),
                "beta": beta,
                "effect": float(high - low),
                "share": share,
                "cumulative": running,
                # The term that crosses the threshold is kept, so the flagged group always
                # accounts for at least MAJOR_MASS rather than just short of it.
                "major": bool(running - share < MAJOR_MASS),
                "stability": float(frequency.get(term.name, float("nan"))),
                "direction": direction(beta),
            }
        )
    return pl.DataFrame(rows)


def term_sentences(importance: pl.DataFrame, limit: int | None = None) -> list[str]:
    """A written line per major term, generated from its row of `term_importance`.

    The sentence says three things, in the order a reader needs them: what the term does
    to MCC, how much of the equation's weight mass it carries, and how often the folds
    agreed it belonged. Nothing is inferred beyond the numbers in the row -- in
    particular no claim is made about the *features* inside the term, because a feature's
    direction depends on the other terms it appears in and is `ml_meta_perf.practices`' job.
    """
    if importance.height == 0:
        return []
    table = importance.filter(pl.col("major")) if "major" in importance.columns else importance
    if limit is not None:
        table = table.head(limit)

    lines: list[str] = []
    for row in table.iter_rows(named=True):
        beta = float(row["beta"])
        stability = float(row["stability"])
        agreement = (
            "selected in every fold"
            if stability >= 0.999
            else f"selected in {stability:.0%} of folds"
            if not np.isnan(stability)
            else "fold agreement not measured"
        )
        lines.append(
            f"`{row['term']}` ({row['group']}) {row['direction']}: one standard deviation of "
            f"this term is worth {beta:+.3f} MCC, it moves predicted MCC by {float(row['effect']):.3f} "
            f"across the middle 80% of its observed range, it carries {float(row['share']):.1%} of the "
            f"equation's weight mass, and it was {agreement}."
        )
    return lines


def group_sentences(shares: pl.DataFrame) -> list[str]:
    """A written line per feature group, generated from the share table."""
    readings = {
        "dataset": "terms over dataset meta-features alone (how hard is this data)",
        "model": "terms over model meta-features alone (how capable is this model)",
        "mixed": "terms mixing dataset and model features (which model suits which data)",
    }
    lines: list[str] = []
    for row in shares.iter_rows(named=True):
        group = str(row["group"])
        count = int(row["n_terms"])
        lines.append(
            f"{readings.get(group, group)}: {count} term{'' if count == 1 else 's'}, "
            f"{float(row['share']):.1%} of the equation's output variance."
        )
    return lines


def coverage(importance: pl.DataFrame) -> dict[str, float | int]:
    """How concentrated the equation is: how few terms carry how much of it.

    ``effective_terms`` is the inverse Simpson index of the weight shares,
    $1 / \\sum_i s_i^2$. It answers "how many equally-weighted terms would behave like
    this equation": it equals the term count when every term carries the same weight, and
    falls toward 1 as one term takes over. It is reported because the headline count and
    the major count both depend on where a threshold is drawn, and this does not.
    """
    if importance.height == 0:
        return {
            "n_terms": 0,
            "n_major": 0,
            "major_share": 0.0,
            "top_share": 0.0,
            "effective_terms": 0.0,
        }
    major = importance.filter(pl.col("major"))
    shares = importance["share"].to_numpy()
    return {
        "n_terms": importance.height,
        "n_major": major.height,
        "major_share": float(major["share"].sum()),
        "top_share": float(importance["share"][0]),
        "effective_terms": float(1.0 / np.sum(shares**2)),
    }


def unstable_majors(importance: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
    """Major terms the folds did not agree on, which are the ones not to build advice on.

    A large standardised weight and a low selection frequency together mean the term is
    doing a lot of work for *this* training set and would be replaced by something else
    on another. That combination is invisible in the equation as printed -- it looks like
    any other large coefficient -- so it is extracted rather than left to be noticed.
    """
    if importance.height == 0 or "stability" not in importance.columns:
        return importance
    return importance.filter(
        pl.col("major") & pl.col("stability").is_not_nan() & (pl.col("stability") < threshold)
    ).select("rank", "term", "group", "beta", "share", "stability")


def _atoms(term: Term | Atom) -> list[Atom]:
    """Every `Atom` inside a term, however deeply nested."""
    if isinstance(term, Atom):
        return [term]
    collected: list[Atom] = []
    for operand in term.operands:
        collected.extend(_atoms(operand))
    return collected


def feature_usage(
    equation: Equation,
    importance: pl.DataFrame,
    available: tuple[str, ...],
) -> pl.DataFrame:
    """Which raw features the equation reached for, how often, and under what transforms.

    A second way to read a flat equation. When no single term dominates, the question
    "which term matters" has no useful answer, but "which of the features on offer did the
    search actually use, and how did it have to bend them" still does -- and it is asked of
    the vocabulary rather than of the weights, so a spread of weights does not blunt it.

    ``share`` sums the standardised-weight mass of every term a feature appears in. A
    feature in two terms is credited both, so the column does not sum to 1: it answers
    "how much of the equation touches this feature", not "how much does this feature own".
    ``unused`` features are reported too, since a feature the search declined to use after
    seeing every transform of it is a finding about the meta-data.
    """
    mass = dict(zip(importance["term"].to_list(), importance["share"].to_list(), strict=True))
    rows: list[dict[str, object]] = []
    for feature in available:
        indices = [index for index, term in enumerate(equation.terms) if feature in term.features]
        transforms = sorted(
            {atom.transform for index in indices for atom in _atoms(equation.terms[index]) if atom.feature == feature}
        )
        operations = sorted({equation.terms[index].operation for index in indices})
        rows.append(
            {
                "feature": feature,
                "meaning": FEATURE_GLOSSARY.get(feature, feature),
                "n_terms": len(indices),
                "share": float(sum(mass.get(equation.terms[index].name, 0.0) for index in indices)),
                "transforms": ", ".join(transforms),
                "operations": ", ".join(operations),
            }
        )
    return pl.DataFrame(rows).sort("share", descending=True)


def operation_usage(
    equation: Equation,
    importance: pl.DataFrame,
    max_arity: int | None = None,
) -> pl.DataFrame:
    """Which grammar operations and transforms earned their place in the equation.

    The vocabulary offers five operations and five transforms; the search is free to
    ignore any of them, and what it declined is as much a result as what it chose --
    every unused entry is a shape the data turned out not to need.

    That reading only holds for entries the search could actually have used, so
    ``max_arity`` marks the rest ``offered = no``. Reporting ``ratio_of_sums`` as
    "declined" when a three-feature cap kept it out of the library would be reading a
    configuration choice as a finding about the data.
    """
    mass = dict(zip(importance["term"].to_list(), importance["share"].to_list(), strict=True))
    rows: list[dict[str, object]] = []
    for kind, names in (("operation", OPERATIONS), ("transform", TRANSFORMS)):
        for name in names:
            if kind == "operation":
                indices = [index for index, term in enumerate(equation.terms) if term.operation == name]
                offered = max_arity is None or OPERATION_ARITY[name] <= max_arity
            else:
                indices = [
                    index
                    for index, term in enumerate(equation.terms)
                    if any(atom.transform == name for atom in _atoms(term))
                ]
                offered = True
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "offered": offered,
                    "n_terms": len(indices),
                    "share": float(sum(mass.get(equation.terms[index].name, 0.0) for index in indices)),
                }
            )
    return pl.DataFrame(rows)


def _shared_features(equation: Equation, members: list[int]) -> list[str]:
    """Features appearing in at least half of a block's terms, commonest first.

    What names a block. The terms were grouped on how their contributions move, not on
    what they contain, so this is a genuine question rather than a restatement of the
    grouping -- and when a block has no common feature, the empty answer is worth seeing
    too.
    """
    counts: dict[str, int] = {}
    for index in members:
        for feature in set(equation.terms[index].features):
            counts[feature] = counts.get(feature, 0) + 1
    # Strict majority. Half would let a two-term block claim a feature only one of its
    # terms names, which is the opposite of shared.
    threshold = len(members) // 2 + 1
    return [
        feature for feature, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if count >= threshold
    ]


def term_groups(
    equation: Equation,
    columns: dict[str, np.ndarray],
    importance: pl.DataFrame,
    *,
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Terms whose contributions move together, read as one block.

    An additive equation invites reading term by term, and that works when one or two
    weights dominate. When they do not -- and here they do not -- the honest unit of
    explanation is larger than a term and smaller than the whole equation. Terms are
    grouped by the correlation of their per-row contributions, so a group is a set of terms
    that rise and fall together across the meta-dataset and can be described in one
    sentence.

    Grouping on contributions rather than on shared features is deliberate. Two terms can
    share no feature and still track each other, and two terms over the same feature can
    move independently once their transforms differ; what a reader needs to know is whether
    they say the same thing about a row.

    Correlation is signed and taken on the contribution, so a term entering with a negative
    weight joins the group it actually agrees with rather than the one it merely shares
    features with.
    """
    if not equation.terms:
        return pl.DataFrame(
            schema={
                "group": pl.Int64,
                "n_terms": pl.Int64,
                "share": pl.Float64,
                "effect": pl.Float64,
                "direction": pl.String,
                "shared": pl.String,
                "terms": pl.String,
            }
        )

    matrix = contributions(equation, columns)
    centered = matrix - matrix.mean(axis=0)
    scale = np.sqrt((centered**2).sum(axis=0))
    scale = np.where(scale < 1e-12, 1.0, scale)
    correlation = (centered / scale).T @ (centered / scale)

    mass = dict(zip(importance["term"].to_list(), importance["share"].to_list(), strict=True))
    # Single-link agglomeration: a term joins a group when it tracks *any* member, which is
    # the right rule here because a chain of terms that each track the next is one story
    # told in several pieces, not several stories.
    assignment = list(range(len(equation.terms)))
    for i in range(len(equation.terms)):
        for j in range(i + 1, len(equation.terms)):
            if correlation[i, j] >= threshold:
                merged, absorbed = sorted((assignment[i], assignment[j]))
                assignment = [merged if label == absorbed else label for label in assignment]

    rows: list[dict[str, object]] = []
    for label in sorted(set(assignment)):
        members = [index for index, value in enumerate(assignment) if value == label]
        total = matrix[:, members].sum(axis=1)
        low, high = np.percentile(total, [10.0, 90.0])
        swing = float(high - low)
        # The group's own sign: whether its combined contribution rises with itself, which
        # is what a one-sentence description of the block has to state.
        signed = float(np.sum([equation.standardized_weights[index] for index in members]))
        rows.append(
            {
                "group": len(rows) + 1,
                "n_terms": len(members),
                "share": float(sum(mass.get(equation.terms[index].name, 0.0) for index in members)),
                "effect": swing,
                "direction": direction(signed),
                "shared": ", ".join(_shared_features(equation, members)),
                "terms": " ; ".join(equation.terms[index].name for index in members),
            }
        )
    return pl.DataFrame(rows).sort("share", descending=True).with_columns(pl.int_range(1, pl.len() + 1).alias("group"))


def marginal_versus_conditional(
    practices: pl.DataFrame,
    columns: dict[str, np.ndarray],
    truth: np.ndarray,
) -> pl.DataFrame:
    """Each practice's direction against the feature's own correlation with MCC.

    A practice states what the *equation* does as a feature rises, with every other term
    present. A marginal correlation states what the feature does alone. The two are
    different quantities and on this data they mostly disagree, which is what conditioning
    does rather than a defect: a marginal correlation mixes a feature's effect with
    everything it travels with, while inside the equation the terms carrying those
    companions are already there.

    It is worth printing because the marginal view is the one a reader will accidentally
    check against -- plot the feature against MCC and be puzzled -- so the disagreement is
    better stated than discovered.
    """
    if practices.height == 0 or "direction" not in practices.columns:
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "meaning": pl.String,
                "marginal": pl.Float64,
                "conditional": pl.Float64,
                "practice_says": pl.String,
                "agrees": pl.Boolean,
            }
        )
    rows: list[dict[str, object]] = []
    for row in practices.iter_rows(named=True):
        feature = str(row["feature"])
        marginal = spearman(columns[feature], truth)
        conditional = float(row["direction"])
        rows.append(
            {
                "feature": feature,
                "meaning": str(row.get("meaning", feature)),
                "marginal": marginal,
                "conditional": conditional,
                "practice_says": "higher" if conditional > 0.0 else "lower",
                "agrees": bool(np.sign(marginal) == np.sign(conditional)),
            }
        )
    return pl.DataFrame(rows)


def _table(frame: pl.DataFrame, float_format: str = "{:.4f}") -> str:
    """A polars frame as a GitHub-flavoured markdown table."""
    if frame.height == 0:
        return "_(empty)_"

    def cell(value: object) -> str:
        # "no" rather than an empty cell: a blank reads as a missing measurement, which
        # is a different claim from a measured disagreement.
        if isinstance(value, bool):
            return "yes" if value else "no"
        if value is None:
            return ""
        if isinstance(value, float):
            return "" if np.isnan(value) else float_format.format(value)
        return str(value).replace("|", r"\|")

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.iter_rows()]
    return "\n".join([header, rule, *body])


def single_prediction(
    equation: Equation,
    columns: dict[str, np.ndarray],
    truth: np.ndarray,
    frame: pl.DataFrame,
    *,
    limit: int = 6,
) -> str:
    """One row's prediction broken into its terms, as a printed block.

    The interpretability payoff in its most direct form: a reader can add the column up and
    get the prediction back, which is a thing no opaque regressor lets anyone do.

    **The row is chosen by a rule, not picked.** The equation's median absolute error is a
    typical prediction rather than its best case, and choosing it arithmetically is what keeps
    this example from being a flattering one somebody selected once. The chapter this lands in
    is the chapter arguing that the analysis is generated rather than authored; it carried a
    hand-written breakdown until 2026-09-07, and every figure in it had gone stale -- the
    intercept it printed was 1.4787 against the equation's actual 1.3586.

    Ties are broken by taking the first such row in corpus order, so the choice is a function
    of the data and the equation alone.
    """
    predicted = equation.predict(columns)
    error = np.abs(truth - predicted)
    # The median error itself, not the row at the middle of the sorted order: with an even
    # row count those differ, and `argmin` against the median value picks an actual row.
    row = int(np.argmin(np.abs(error - float(np.median(error)))))

    shares = contributions(equation, columns)[row]
    ordered = sorted(zip(shares, equation.terms, strict=True), key=lambda pair: -abs(pair[0]))
    total = equation.intercept + float(shares.sum())

    lines = [
        f"dataset  : {frame[DATASET_COLUMN][row]}",
        f"model    : {frame[MODEL_COLUMN][row]}",
        f"actual   : {truth[row]:+.4f}",
        f"predicted: {predicted[row]:+.4f}",
        "",
        "contribution breakdown:",
        f"    {equation.intercept:+.4f}   intercept",
    ]
    for value, term in ordered[:limit]:
        lines.append(f"    {value:+.4f}   {term.name}")
    if len(ordered) > limit:
        remainder = sum(value for value, _ in ordered[limit:])
        lines.append(f"    {remainder:+.4f}   the remaining {len(ordered) - limit} terms")
    lines.append(f"  = {total:+.4f}   sum")
    # `Equation.predict` clips to the range MCC can take at all, so on a row where the linear
    # form runs past the end of that range the column does not add up to the number printed
    # above it. Saying so is the honest form: a breakdown a reader cannot add up, with no
    # explanation of why, is worse than no breakdown.
    if abs(total - predicted[row]) > 5e-5:
        lines.append(f"    clipped to {predicted[row]:+.4f}, the range MCC can take")
    return "```\n" + "\n".join(lines) + "\n```"


#: How each equation's own ceiling is labelled in `Report.comparison`. E1 and E2 are bounded
#: by what their group identity can explain at all; E3 is bounded by the additive oracle only
#: in the sense that passing it demonstrates interaction, so it is not given a ratio.
CEILING_ROWS = {"E1": "E1 reference: true dataset means", "E2": "E2 reference: true model means"}


def _headline(report: Report) -> pl.DataFrame:
    """The three equations and the capability variant, each against its own ceiling.

    The one table the study can be summarised by. It exists because the summary page kept a
    hand-written copy of these numbers and had no way of noticing when they moved -- the same
    failure the generated chapter sections were introduced to remove, reappearing one level
    up. **The ``reached`` column is the point of the table**, not the R2 column: E1's
    structural maximum is a per-dataset constant, so its 0.35 and E3's 0.66 are not
    comparable as achievements, and printing the fraction of each equation's own ceiling
    beside them is what stops a reader making that comparison anyway.
    """
    ceilings = dict(zip(report.comparison["equation"].to_list(), report.comparison["r2"].to_list(), strict=True))
    rows: list[dict[str, object]] = []
    listed = (("E1", report.e1), ("E2", report.e2), ("E3", report.e3), ("E3 capability", report.e3_capability))
    for name, equation in listed:
        ceiling = ceilings.get(CEILING_ROWS.get(name, ""))
        in_sample = float(equation.in_sample["r2"])
        rows.append(
            {
                "equation": name,
                "features": {"E1": "dataset", "E2": "model"}.get(name, "both"),
                "terms": equation.equation.n_terms,
                "in-sample R2": in_sample,
                "LOO-dataset R2": float(equation.cross_validated["loo_dataset"]["r2"]),
                "LOO-model R2": float(equation.cross_validated["loo_model"]["r2"]),
                "own ceiling": float(ceiling) if ceiling is not None else float("nan"),
                "reached": in_sample / float(ceiling) if ceiling else float("nan"),
            }
        )
    return pl.DataFrame(rows)


def _scores(label: str, scores: dict[str, float | int]) -> str:
    return (
        f"| {label} | {float(scores['r2']):.4f} | {float(scores['mae']):.4f} | "
        f"{float(scores['rmse']):.4f} | {int(scores['n'])} |"
    )


def _crater_note(report: Report) -> str:
    """The deepest leave-one-dataset-out crater on the length curve, and what it costs a mean.

    The consensus curve takes a **median** across the three protocols rather than a mean, and
    the argument for that is only convincing next to a length where the two disagree. Which
    length that is moves whenever the configuration does -- chapter 3 named one in prose and
    it had drifted onto the published length by the time anyone reread it, which turned the
    illustration into a claim that the equation the study ships sits in a crater.

    Deepest is measured against the neighbouring lengths, not against the curve's own mean: a
    crater is a local collapse, and a length at the end of a declining run is not one.
    """
    curve = report.e3.curve
    if curve.height < 3 or "r2_loo_dataset" not in curve.columns:
        return ""
    protocols = ["r2_in_sample", "r2_loo_dataset", "r2_loo_model"]
    if any(name not in curve.columns for name in protocols):
        return ""
    values = np.column_stack([curve[name].to_numpy() for name in protocols])
    held = curve["r2_loo_dataset"].to_numpy()
    lengths = curve["n_terms"].to_numpy()
    neighbours = (held[:-2] + held[2:]) / 2.0
    index = int(np.argmax(neighbours - held[1:-1])) + 1
    depth = float(neighbours[index - 1] - held[index])
    if depth <= 0.0:
        return ""
    row = values[index]
    return (
        f"**Why the consensus is a median and not a mean.** The deepest crater on this curve "
        f"is at **{int(lengths[index])} terms**, where the three protocols read "
        f"{row[0]:.3f} / {row[1]:.3f} / {row[2]:.3f}. The median takes {float(np.median(row)):.3f} "
        f"and ignores it; a mean would be dragged to {float(row.mean()):.3f}. The crater is "
        f"{depth:.3f} below the neighbouring lengths and is not a property of the length at "
        "all -- it is one held-out dataset sitting outside the convex hull of the other "
        "nineteen in term space, where a linear equation extrapolates without limit and "
        "`validate._clip_to_training` pins the fold to its training floor. One fold's "
        "extrapolation should not choose the published length.\n"
    )


def _saturated_note(report: Report) -> str:
    """Read the saturated fit against the published equation, in one generated sentence.

    The comparison is the point, and stating it in prose beside the table is how it stops
    being two numbers a reader has to subtract. Both sides come from this run, so the
    sentence cannot drift from the table above it.
    """
    if report.saturated.height == 0:
        return ""
    row = report.saturated.to_dicts()[0]
    published = float(report.e3.cross_validated["loo_dataset"]["r2"])
    return (
        f"**The solver is not the hard part; the sample size is.** All "
        f"{int(row['terms'])} terms at once fit better in-sample than the published equation "
        f"({float(row['r2_in_sample']):.4f} against "
        f"{float(report.e3.in_sample['r2']):.4f}) and transfer at "
        f"{float(row['r2_loo_dataset_clipped']):.4f} leave-one-dataset-out, against the "
        f"published equation's {published:.4f}. The unclipped figure — "
        f"{float(row['r2_loo_dataset_unclipped']):.1f} — is what the fit does when a held-out "
        "dataset falls outside the convex hull of the other nineteen and nothing bounds the "
        "extrapolation. A design this much wider than 20 held-out groups can support has "
        "nothing to constrain it, which is what selection is for.\n"
    )


def _coverage_note(report: Report, columns: dict[str, np.ndarray]) -> str:
    """How far the equation reaches into the catalogue, and how to read a disagreement.

    Two counts a reader will otherwise conflate: the verdict tally above is drawn from corpus
    statistics that any study with this data could compute, while this is about the terms. And
    one reading that has to be spelled out, because the table looks self-contradictory without
    it -- a feature in a denominator enters inverted, so the same feature can carry opposite
    signs in two terms and mean one thing.
    """
    counts = equation_coverage(report, columns)
    if not counts["pairs"]:
        return ""
    lines = [
        f"**Read this against the verdict tally above, not as part of it.** Of the "
        f"{counts['practices']} practices, {counts['with_feature_claims']} make a claim about a "
        "quantity the equation contains; the rest are about a protocol, a metric, or a family "
        "of learners, and pairing one of those with a coefficient would be inventing a "
        f"connection. Those are carried by **{counts['carrying_terms']} of the equation's "
        f"{counts['terms']} terms**, giving {counts['pairs']} directed (practice, term) "
        f"pairings: **{counts['agree']} come out the way the practice predicts and "
        f"{counts['disagree']} do not**"
        + (
            f", with {counts['undirected']} pairing too weak to state a direction for and "
            f"{counts['unselected']} claim resting on a feature the search never took.\n"
            if counts["undirected"] or counts["unselected"]
            else ".\n"
        ),
    ]
    if counts["disagree"] and counts["disagree_in_denominator"] == counts["disagree"]:
        lines.append(
            f"\n**Every one of the {counts['disagree']} disagreements is the same feature "
            "entering as a *denominator*, and that is arithmetic rather than conflict.** A "
            "negatively-weighted ratio contributes more as its denominator grows, so a term "
            "of the form `dataset property / capacity` must rise with capacity. Read the "
            "`position` column across those rows and the equation is saying one coherent "
            "thing: it has **no marginal statement about capacity at all**, only statements "
            "about capacity *relative to* something the dataset demands. Where capacity is a "
            "numerator instead — measured against the class count rather than against a "
            "difficulty — it lowers predicted MCC, which is the practice's own claim.\n"
            "\n**So the mismatch is in how the expectation was written down, not in what the "
            "equation says.** A practice recommending that capacity be *matched to the "
            "problem* is a conditional claim; encoding it as `capacity lowers MCC` is a "
            "marginal one, and a marginal expectation cannot match a term that only ever "
            "speaks conditionally. This is the clearest case in the study of why the equation "
            "is worth reading term by term: no per-feature summary, and no opaque model, can "
            "distinguish 'more capacity is better' from 'more capacity per unit of difficulty "
            "is better'.\n"
        )
    else:
        lines.append(
            "\n**Read the `position` column before calling a row a disagreement.** A feature "
            "in a denominator enters the term inverted, so a negatively-weighted ratio "
            "contributes more as that feature rises. The same feature carrying opposite signs "
            "in two terms is usually one statement about a ratio rather than two conflicting "
            "ones about a quantity.\n"
        )
    lines.append(
        "\n``beta`` is the strength and ``effect`` is what the term is worth on this data; "
        "**agreement in sign with a negligible effect is agreement without evidence**, which "
        "is why the two are printed together. ``rho`` is the measured association the "
        "direction comes from, and a pairing below the same floor the per-feature statements "
        "use is reported as having no direction rather than being given a sign it cannot "
        "support.\n"
    )
    return "\n".join(lines)


def _identity_note(report: Report) -> str:
    """What the identity ceiling is worth, and whether the study's own test calls it real.

    A difference of two pooled R2 values over twenty folds is not a measurement -- that has
    produced three wrong conclusions on this project -- so the rungs are paired against the
    uncorrected equation and the verdict comes from the test rather than from the size of the
    gap. The two halves of the paired test are reported separately where they disagree,
    because they answer different questions: the sign test asks whether the correction wins
    *consistently*, the bootstrap whether the mean gain survives a different draw of datasets.
    """
    if report.identity.height < 3:
        return ""
    rows = report.identity.to_dicts()
    base, level, slope = rows[0], rows[1], rows[2]
    total = float(slope["r2_loo_dataset"]) - float(base["r2_loo_dataset"])
    step = float(level["r2_loo_dataset"]) - float(base["r2_loo_dataset"])

    lines = [
        f"**The gap is {total:.3f} of leave-one-dataset-out R2**, of which a per-model *level* "
        f"recovers {step:.3f} and the level-plus-slope form the remaining {total - step:.3f}.\n",
        "**Whether that is real is a paired question**, so each rung is compared with the "
        "uncorrected equation dataset by dataset, on absolute error, over the twenty held-out "
        "folds. The two rungs come back differently, and the difference is the finding:\n",
    ]
    if level["verdict"] == "tie":
        lines.append(
            f"* a per-model **level** is a **tie** -- it wins on {int(level['wins'])} of the "
            "twenty folds and its interval spans zero. A constant shift per model, which is "
            "what a level is, adds nothing the equation does not already have.\n"
        )
    if slope["verdict"] == "real":
        lines.append(
            f"* a per-model **slope** is **not** a tie: the bootstrap interval "
            f"[{float(slope['ci_low']):+.4f}, {float(slope['ci_high']):+.4f}] lies entirely "
            f"above zero, on {int(slope['wins'])} winning folds of twenty. Read it with the "
            f"sign test beside it, which at p = {float(slope['sign_p']):.3f} does **not** reach "
            "significance -- so the gain is carried by its size on the folds it wins rather "
            "than by winning nearly all of them. That is weaker evidence than the interval "
            "alone suggests, and stronger than a tie.\n"
        )
        lines.append(
            "**So the question this chapter was written to close is not closed.** The half of "
            "the correction that survives is the one that lets a model's advantage depend on "
            "the data -- exactly what a *capability* descriptor would have to do, and exactly "
            "what none of the descriptors this corpus records does. The mixed terms were "
            "supposed to absorb that interaction and have absorbed only part of it.\n"
        )
    else:
        lines.append(
            "* neither rung survives pairing, so on this corpus perfect model identity adds "
            "nothing measurable to the published equation and the model side is as well "
            "described as free per-model numbers could make it.\n"
        )
    lines.append(
        "Every model-side encoding the study tried and rejected was rejected for failing to "
        "recover this gap, so it is a property of the corpus rather than of the search -- and "
        "the one route to closing it that survives on the merits is measuring what a model is "
        "good at rather than asserting it.\n"
    )
    return "\n".join(lines)


def _opaque_note(report: Report) -> str:
    """Read the opaque table across its protocols, and against the equation on each.

    The columns are ordered by how much the estimator was allowed to see, and reading them
    left to right is the argument. The last one is the comparison that matters: under
    leave-one-cell-out neither side has the dataset or the model, so it is the only protocol on
    which an opaque regressor and a fifteen-term equation are denied the same things.
    """
    if report.opaque.height == 0 or "r2_loo_cell" not in report.opaque.columns:
        return ""
    rows = report.opaque.to_dicts()
    best_fit = max(rows, key=lambda row: float(row["r2_in_sample"]))
    best_cell = max(rows, key=lambda row: float(row["r2_loo_cell"]))
    return (
        f"**Read the {best_fit['model']} row across.** It fits this meta-data at R2 "
        f"{float(best_fit['r2_in_sample']):.4f}; holding out a whole model leaves it at "
        f"{float(best_fit['r2_loo_model']):.4f}; holding out a whole dataset drops it to "
        f"{float(best_fit['r2_loo_dataset']):.4f}; and with **both** held out it reaches "
        f"{float(best_fit['r2_loo_cell']):.4f}. The published equation is at "
        f"{float(report.e3.in_sample['r2']):.4f} and "
        f"{float(report.e3.cross_validated['loo_dataset']['r2']):.4f} on the first and third of "
        "those.\n"
        "\nThe ordering of those four columns is the whole finding. A flexible model on twenty "
        "dataset groups, with dataset features constant inside a group, does not learn a "
        "relationship -- it learns which dataset a row came from and looks the answer up. Every "
        "column that removes an identity removes some of that, and the column that removes "
        "both leaves almost nothing.\n"
        f"\n**Under full leakage prevention the best opaque estimator reaches "
        f"{float(best_cell['r2_loo_cell']):.4f}** ({best_cell['model']}), which is at or below "
        "what predicting the corpus mean would score. This is the like-for-like comparison in "
        "the study: leave-one-dataset-out still hands a forest the held-out learner on nineteen "
        "other problems, and leave-one-model-out still hands it the held-out dataset. Only here "
        "is it denied what the equation is denied -- and it is also the protocol on which the "
        "trivial per-model baselines cannot be computed at all, since a model held out of every "
        "fold has no rows to average. A feature-based predictor still predicts.\n"
        "\nThis is the likely provenance of the R2 near 0.9 figures reported for opaque "
        "meta-models: an in-sample or randomly-split forest reproduces them exactly. **None of "
        "these is tuned**, and tuning them would answer a different objection -- the failure is "
        "that the sample has twenty dataset groups, which no amount of tuning changes. What the "
        "table licenses is that the accuracy this study traded away was not there to be had "
        "under a protocol where the dataset is genuinely unseen.\n"
    )


def _length_note(report: Report) -> str:
    """How the length was chosen, and what the alternatives would have chosen.

    Every rule the study computed is on the page. A selection rule is only defensible if what
    it beats is visible beside it, and the geometric rules here disagree with the one adopted
    by a wide margin.
    """
    if report.length_choice.height == 0:
        return ""
    rows = report.length_choice.to_dicts()
    selected = next((row for row in rows if row["verdict"] == "selected"), None)
    if selected is None:
        return ""
    ties = [row for row in rows if row["verdict"] in ("tie", "selected")]
    shortest = min(ties, key=lambda row: int(row["n_terms"])) if ties else selected
    worse = [row for row in rows if row["verdict"] == "worse"]

    lines = [
        f"The length is chosen by one rule with no threshold and no smoothing: **the argmax of "
        f"the consensus curve** (`selection.best_length`), which here selects "
        f"**{int(selected['n_terms'])} terms**. Nothing about that number is written down — it "
        "falls out of the curve, and it re-derives itself if the corpus changes.\n",
        _crater_note(report),
        "Every alternative rule is reported beside it, because a selection rule is only "
        "defensible if what it beats is on the page:\n",
        _table(report.term_choice) + "\n",
        f"The geometric rules — the Pareto-front knee by its three standard forms — choose far "
        f"shorter equations, and **{len(worse)} of the {len(rows)} lengths searched are "
        "significantly worse** than the selected one when paired fold by fold over the "
        "held-out datasets. A knee finds where the *marginal* return per term collapses, which "
        "on a saturating curve is early; it does not ask whether the accuracy still being "
        "added is real.\n",
        f"The parsimony alternative is **{int(shortest['n_terms'])} terms** — the shortest "
        "length whose paired interval against the selected one spans zero. It is reported and "
        "not adopted: the accuracy it gives up is measurable "
        f"({float(shortest['r2_loo_dataset']):.4f} against "
        f"{float(selected['r2_loo_dataset']):.4f} leave-one-dataset-out) even where it is not "
        "significant.\n",
    ]
    return "\n".join(lines)


def _capability_note(report: Report) -> str:
    """The same features under the full grammar: how far the additive form reaches."""
    capability = report.e3_capability
    if not capability.equation.terms:
        return ""
    published = float(report.e3.in_sample["r2"])
    reached = float(capability.in_sample["r2"])
    lines = [
        f"The published equation is the **parsimonious** grammar (arity 2). The same features "
        f"under the **full** grammar (arity 3), with the length chosen by the same rule, reach "
        f"{len(capability.equation.terms)} terms at R² {reached:.4f} in-sample:\n",
        "| | terms | in-sample | LOO-dataset | LOO-model |",
        "|---|---|---|---|---|",
        f"| published (arity 2) | {len(report.e3.equation.terms)} | {published:.4f} | "
        f"{float(report.e3.cross_validated['loo_dataset']['r2']):.4f} | "
        f"{float(report.e3.cross_validated['loo_model']['r2']):.4f} |",
        f"| capability (arity 3) | {len(capability.equation.terms)} | {reached:.4f} | "
        f"{float(capability.cross_validated['loo_dataset']['r2']):.4f} | "
        f"{float(capability.cross_validated['loo_model']['r2']):.4f} |",
        "",
        "This is a **capability measurement, not a recommendation**. It answers the question "
        "the published equation cannot answer about itself — whether the additive form is out "
        f"of room or whether this equation is short of it — and the answer is that {reached - published:+.4f} "
        "of in-sample R² is still available to a longer equation over a wider grammar. What "
        "that costs is what the published equation is buying: more terms, an operation more, "
        "and a form that reselects far less often across folds.\n",
    ]
    return "\n".join(lines)


def _reach_note(report: Report) -> str:
    """The grammar's implied ceiling ladder, and where the equation lands on it.

    Everything here is arithmetic over `analysis.grammar_ceiling` and the fitted R2. The
    point of the section is that the three rungs are computable *before* any search runs, so
    the equation's score can be read against an expectation rather than against nothing.
    """
    if report.ceiling.height == 0:
        return ""
    ladder = report.ceiling.to_dicts()[0]
    raw = float(ladder["r2_raw_additive"])
    best = float(ladder["r2_best_per_feature"])
    single = float(ladder["r2_all_single_feature"])
    fitted = float(report.e3.in_sample["r2"])
    n_single = int(float(ladder["n_single_feature_terms"]))
    n_terms = len(report.e3.equation.terms)

    strongest = report.reach.head(1).to_dicts()[0] if report.reach.height else None
    lines = [
        "Three levels of what the vocabulary can explain, each a least-squares fit over the "
        "library and each computable before the search runs. They bound a *sum of "
        "per-feature functions*, which is a different question from the additive oracle "
        "above: that one bounds a per-dataset value plus a per-model value.\n",
        "| level | terms | R² |",
        "|---|---|---|",
        f"| every raw feature, untransformed | {len(ALL_FEATURES)} | {raw:.4f} |",
        f"| the best single-feature term per feature | {len(ALL_FEATURES)} | {best:.4f} |",
        f"| every single-feature term at once | {n_single} | {single:.4f} |",
        f"| **the fitted equation (E3)** | **{n_terms}** | **{fitted:.4f}** |",
        "",
    ]
    if strongest is not None:
        lines.append(
            f"No individual feature carries much: the strongest is `{strongest['feature']}` at "
            f"R² {float(strongest['r2_best']):.3f}, so any accuracy beyond that is combination "
            "rather than a single dominant driver. Transforming the features is worth "
            f"{best - raw:+.3f} over entering them raw.\n"
        )
    verdict = (
        f"E3 reaches {fitted:.4f} with {n_terms} terms, **above** the {single:.4f} that all "
        f"{n_single} single-feature terms reach together. An equation cannot pass that level "
        "by describing features one at a time, so the excess is what the cross-feature terms "
        "buy — the same conclusion the additive oracle reaches, by an independent route."
        if fitted > single
        else f"E3 reaches {fitted:.4f} with {n_terms} terms against the {single:.4f} available "
        f"from all {n_single} single-feature terms, so on this configuration its accuracy is "
        "still within what per-feature description alone could explain."
    )
    lines.append(verdict + "\n")
    return "\n".join(lines)


def _protocol_note(report: Report) -> str:
    """What the equation is worth as more of the problem becomes unfamiliar.

    Built from the two decision tables, which now carry the protocol of every row. The point
    of the section is that the strictest protocol -- both the dataset and the model out of the
    training set -- is the one the practitioner's question actually needs, and that the trivial
    baselines cannot be computed under it at all.
    """
    ranking = report.ranking_baselines
    decision = report.decision_baselines
    if ranking.height == 0 or decision.height == 0:
        return ""

    order = [
        ("in-sample", "in-sample", "nothing held out"),
        ("loo-dataset", "leave-one-dataset-out", "the dataset unseen, the model known"),
        ("loo-model", "leave-one-model-out", "the model unseen, the dataset known"),
        ("loo-cell", "leave-one-cell-out", "**both unseen**"),
    ]
    rank_rows = {str(row["predictor"]): row for row in ranking.to_dicts()}
    mid = sorted({float(value) for value in decision["threshold"]})
    threshold = mid[len(mid) // 2]
    dec_rows = {str(row["predictor"]): row for row in decision.filter(pl.col("threshold") == threshold).to_dicts()}

    lines = [
        "The ranking and the go/no-go decision are reported with **both the dataset and the "
        "model of every cell held out of the fit**. Neither single-group protocol answers the "
        "question those tasks pose: leave-one-dataset-out has seen the learner on the other "
        "nineteen problems, and leave-one-model-out has seen the dataset. A recommendation is "
        "asked about a pair that has not been run.\n",
        f"| what the equation was shown | AP | MRR | hit@1 | regret | F1 @ {threshold:g} | MCC @ {threshold:g} |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label, shown in order:
        name = next((n for n in rank_rows if key in n), None)
        dname = next((n for n in dec_rows if key in n), None)
        if name is None or dname is None:
            continue
        r, d = rank_rows[name], dec_rows[dname]
        emphasis = "**" if key == "loo-cell" else ""
        lines.append(
            f"| {emphasis}{label}{emphasis} — {shown} | {float(r['ap']):.3f} | {float(r['mrr']):.3f} | "
            f"{float(r['hit_at_1']):.2f} | {float(r['regret']):.3f} | "
            f"{float(d['f1']):.3f} | {float(d['mcc']):.3f} |"
        )
    lines.append("")

    baselines = [str(name) for name in rank_rows if "per-model" in name]
    if baselines:
        best = max(baselines, key=lambda n: float(rank_rows[n]["ap"]))
        lines.append(
            "The trivial predictors are in the tables below at leave-one-dataset-out, which is "
            "the only protocol under which they exist. **Under the strictest one they cannot be "
            "computed at all**: a model held out of every fold has no rows to average, so "
            f'"how well does this model usually do" has no value. The best of them reaches AP '
            f"{float(rank_rows[best]['ap']):.3f} and F1 {float(dec_rows[best]['f1']):.3f} while "
            "being shown the model identity the strictest row of the equation is denied.\n"
        )
    return "\n".join(lines)


def _ranking_verdict(baselines: pl.DataFrame) -> str:
    """State the ranking comparison from the paired test, never from the means.

    The means over twenty datasets are the trap this study has fallen into three times.
    hit@1 moves only in steps of 0.05 on twenty folds, so one dataset flipping its top pick
    shifts it further than the gaps in the table; and average precision differences of this
    size sit well inside the bootstrap interval. The sentence is therefore built from
    `ap_vs_e3_significant`, not from which row has the larger mean.
    """
    if "ap_vs_e3_significant" not in baselines.columns:
        return ""
    rivals = [row for row in baselines.to_dicts() if row.get("ap_vs_e3_significant") is not None]
    if not rivals:
        return ""
    beaten = [row for row in rivals if row["ap_vs_e3_significant"]]
    names = ", ".join(str(row["predictor"]) for row in rivals)
    if beaten:
        return (
            "Paired over the datasets, the equation differs significantly from: "
            + ", ".join(str(row["predictor"]) for row in beaten)
            + ". The remaining comparisons are ties.\n"
        )
    return (
        f"**None of these differences survives a paired test.** Against {names} the sign test "
        "and the bootstrap interval over datasets both include zero, so on ranking the equation "
        "is indistinguishable from ordering the models by how well they usually do. Read the "
        "means in the table above as ties, not as a ranking of the predictors — including where "
        "a baseline's mean is the larger one.\n"
    )


def _baseline_centre_note(baselines: pl.DataFrame) -> str:
    """Which centre is the harder baseline, per metric, derived rather than asserted.

    The mean minimises squared error and the median minimises absolute error, so the fair
    opponent differs by metric. Rather than state that as a claim, the sentence is built from
    the table: for each metric, the best mean row is compared with the best median row.
    """
    rows = {str(row["baseline"]): row for row in baselines.to_dicts()}
    pairs = [(name, name.replace(" mean ", " median ")) for name in rows if " mean " in name]
    if not pairs:
        return ""

    def best(names: list[str], metric: str, lower_is_better: bool) -> tuple[str, float]:
        scored = [(name, float(rows[name][metric])) for name in names]
        return min(scored, key=lambda item: item[1]) if lower_is_better else max(scored, key=lambda item: item[1])

    means = [mean for mean, _ in pairs]
    medians = [median for _, median in pairs if median in rows]
    if not medians:
        return ""

    lines = [
        "Every trivial predictor appears at its mean and at its median, because the metrics "
        "disagree about which is the honest opponent: the mean minimises squared error and "
        "the median minimises absolute error, so an MAE quoted against a mean baseline is "
        "quoted against a predictor that is not minimising the metric it is judged on. "
        "Reading the strongest baseline of each kind, per metric:\n",
        "| metric | strongest mean baseline | strongest median baseline | harder |",
        "|---|---|---|---|",
    ]
    for metric, lower_is_better in (("r2", False), ("mae", True), ("smape", True)):
        mean_name, mean_value = best(means, metric, lower_is_better)
        median_name, median_value = best(medians, metric, lower_is_better)
        median_wins = median_value < mean_value if lower_is_better else median_value > mean_value
        lines.append(
            f"| {metric.upper()} | {mean_name} ({mean_value:.4f}) | "
            f"{median_name} ({median_value:.4f}) | **{'median' if median_wins else 'mean'}** |"
        )
    return "\n".join(lines) + "\n"


def _configuration(config: Configuration) -> str:
    return _table(
        pl.DataFrame([{"setting": field.name, "value": str(getattr(config, field.name))} for field in fields(config)])
    )


def render(
    report: Report,
    columns: dict[str, np.ndarray],
    truth: np.ndarray,
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    *,
    frame: pl.DataFrame | None = None,
    config: Configuration | None = None,
    source: str | None = None,
) -> str:
    """The full study as a markdown report.

    Every table and every sentence is computed from ``report``. The prose that is fixed
    -- section headings and the explanations of what each table means -- is fixed because
    it describes the *method*, which does not change between runs; anything describing
    the *result* is generated.
    """
    importance = term_importance(report.e3.equation, columns, dataset_features, model_features, report.e3.stability)
    concentration = coverage(importance)
    equation = report.e3.equation
    # Counted from the corpus rather than written into the prose below it. A generated
    # sentence carrying a literal "20 datasets" is the same staleness the generated sections
    # exist to prevent, one level further in.
    datasets = int(frame[DATASET_COLUMN].n_unique()) if frame is not None else 0

    parts: list[str] = []
    parts.append("# ml-meta-perf — fitted equation and extracted practices\n")
    parts.append(
        "Generated by `ml_meta_perf.report.render`. Every number, table and sentence below is "
        "computed from the fitted equation and the validation run; nothing is written by "
        "hand at report time.\n"
    )

    parts.append("## 1. Run\n")
    if source:
        parts.append(f"Meta-dataset: `{source}`\n")
    if config is not None:
        parts.append(_configuration(config) + "\n")

    parts.append("## 1b. The corpus\n")
    parts.append(
        "What the meta-dataset is, computed from the file the run was fitted on rather than "
        "transcribed into prose beside it.\n"
    )
    if frame is not None:
        parts.append(_table(corpus_summary(frame)) + "\n")
        parts.append(
            "**The absent cells are not missing at random.** Every dataset short of models is "
            "one of the smallest in the corpus, which is what the instance counts show:\n"
        )
        parts.append(_table(missing_cells(frame), float_format="{:.0f}") + "\n")
        parts.append("And how MCC is distributed over those rows:\n")
        parts.append(_table(target_summary(frame)) + "\n")
        parts.append(
            "**A third of the corpus is pinned at one end of the range or the other.** That is "
            "what makes MAE rather than SMAPE the reported error: SMAPE divides by "
            "`|truth| + |prediction|`, so every row at exactly zero contributes the full 200% "
            "unless the prediction is exactly zero too, and the metric ends up dominated by "
            "the rows the equation is already known to handle worst.\n"
        )
        parts.append(
            "Two things these tables cannot say, both of which bound every number in the study. "
            "Each row is the **best of three seeds**, not their mean, so the target is "
            "optimistic and has a noise floor no predictor can go below; and every model was "
            "trained on a stratified sample **capped at 100,000 rows**, so `nr_inst` is the "
            "source dataset's size rather than the training set's. Both are properties of the "
            "corpus builder upstream, and are audited against it in the prose above.\n"
        )
    else:
        parts.append("_(not summarised: the meta-dataset was not supplied to the renderer)_\n")

    parts.append("## 1c. The headline\n")
    parts.append(
        "The one table the study is summarised by, so that the summary cannot drift from "
        f"the chapters. Every row is scored on the same {int(report.e3.in_sample['n'])} rows "
        "under the same protocols; the equations differ **only** in which features they may "
        "draw on.\n"
    )
    parts.append(_table(_headline(report)) + "\n")
    parts.append(
        "**Do not read these R² values as achievements against each other.** They share a "
        "scale but not a ceiling: E1 sees only dataset features, every row of a dataset "
        "shares one feature vector, and so E1 can predict nothing but a per-dataset "
        "constant. Its structural maximum is the `true dataset means` row, and reaching it "
        "means E1 is *done* rather than weak. The comparable quantity is the fraction of "
        "each equation's own ceiling, which the last column gives.\n"
    )
    parts.append(
        "**E3's ceiling cells are blank because it has no structural one.** Nothing in the "
        "feature set stops an equation over both halves of the meta-data from predicting "
        "every cell, so there is no group-identity bound to divide by. The reference it is "
        "read against instead is the additive oracle, in the comparison table of chapter 5 -- "
        "and E3 is *expected* to pass that, because the oracle bounds only an equation "
        "additive in dataset effect plus model effect, which E3's mixed terms are not.\n"
    )

    parts.append("## 2. The equation\n")
    parts.append(
        f"E3 uses **{equation.n_terms} terms** over dataset and model meta-features, "
        "simplified and refitted after pruning, so it evaluates exactly as printed.\n"
    )
    parts.append("```\n" + str(equation) + "\n```\n")
    parts.append("LaTeX:\n")
    parts.append("```latex\n" + equation.to_latex() + "\n```\n")

    parts.append("## 3. How well it does\n")
    parts.append("| protocol | R² | MAE | RMSE | n |")
    parts.append("|---|---|---|---|---|")
    parts.append(_scores("in-sample", report.e3.in_sample))
    for label, scores in report.e3.cross_validated.items():
        parts.append(_scores(label.replace("_", "-"), scores))
    parts.append("")
    parts.append(
        "Cross-validated rows hold out a whole dataset or a whole model, so the equation "
        "is scored on a group it has never seen. That is the number that matters, and it "
        "is well below the in-sample one at this sample size.\n"
    )
    parts.append("Against the baselines and the ceiling that bounds any additive equation:\n")
    parts.append(_table(report.comparison) + "\n")

    parts.append("### The trivial predictors, at both centres\n")
    parts.append(_table(report.baselines) + "\n")
    parts.append(_baseline_centre_note(report.baselines))

    parts.append("## 3c. How far the form could reach\n")
    parts.append(
        "Two ceilings, both computed from the library alone and so available *before* an "
        "equation exists. Each is an expectation the fitted equation is then held against, "
        "rather than a number read off it.\n"
    )
    parts.append("### How far the additive form reaches\n")
    parts.append(_capability_note(report))

    parts.append("### What the vocabulary could reach, before any search\n")
    parts.append(_reach_note(report))

    parts.append("### Where the variance is, before any equation\n")
    parts.append("Variance of MCC explained by group identity alone, with no equation involved:\n")
    parts.append(_table(report.decomposition) + "\n")

    parts.append("### How fast interaction pays\n")
    parts.append(
        "The additive form cannot represent dataset-by-model interaction beyond what its mixed "
        "terms reach. The ladder below adds interaction components to an oracle that is "
        "handed the true group means, so it measures the ceiling rather than any equation:\n"
    )
    parts.append(_table(report.oracles) + "\n")
    parts.append(
        "That is a ceiling, not a score. Whether the equation reaches any of it is a separate "
        "question, and the answer is that it reaches some: below, `alignment` is the squared "
        "correlation between the equation's own interaction residual and the leading components "
        "of the oracle's, over observed cells. `leading_share` is how much of the interaction "
        "variance those components carry, and `interaction_share` how much of MCC's variance is "
        "interaction at all.\n"
    )
    parts.append(_table(report.interaction) + "\n")

    parts.append("## 3b. Why a subset rather than every term\n")
    parts.append(
        "The control for the whole selection stage. If handing every candidate term to "
        "unpenalised least squares in one go transferred well, the beam search and the length "
        "rule would be machinery in search of a problem.\n"
    )
    parts.append(_table(report.saturated) + "\n")
    parts.append(_saturated_note(report))

    parts.append("## 4. Equation analysis\n")
    parts.append(
        f"The equation has {concentration['n_terms']} terms, of which "
        f"**{concentration['n_major']}** carry {float(concentration['major_share']):.0%} of the "
        f"standardised weight mass; the single largest carries "
        f"{float(concentration['top_share']):.1%}, and the weights behave like "
        f"**{float(concentration['effective_terms']):.1f} equally-weighted terms** "
        "(inverse Simpson index of the shares).\n"
    )
    flatness = float(concentration["effective_terms"]) / max(int(concentration["n_terms"]), 1)
    parts.append(
        "That last number is the one to read for concentration, because it does not depend "
        "on where a threshold is drawn. At "
        f"{flatness:.0%} of the term count the equation is "
        + (
            "**flat**: no single term dominates. That is a statement about the *unit of "
            "explanation*, not about the quality of the equation — MCC here is inferred by "
            "a set of terms acting together rather than by one or two that could be quoted "
            "on their own. Three readings follow, and the sections below give each one: "
            "read the terms in the blocks that move together, read which features the "
            "search reached for, and read which operations it needed to apply to them.\n"
            if flatness > 0.6
            else "**concentrated**: a minority of terms does most of the work, and reading "
            "those few is close to reading the whole equation.\n"
        )
    )
    parts.append(
        "`beta` is the standardised weight — the MCC contributed per standard deviation of "
        "the term, which is what makes terms in unrelated units comparable. `effect` is the "
        "swing in predicted MCC across the middle 80% of the term's observed range. "
        "`stability` is the fraction of leave-one-dataset-out folds that selected the term.\n"
    )
    parts.append(_table(importance) + "\n")

    shown = min(SENTENCE_LIMIT, int(concentration["n_major"]))
    parts.append(f"### The {shown} largest terms, in words\n")
    if int(concentration["n_major"]) > shown:
        parts.append(
            f"({int(concentration['n_major'])} terms are flagged major; the leading {shown} are "
            "written out, and the table above carries the rest.)\n"
        )
    for index, sentence in enumerate(term_sentences(importance, limit=SENTENCE_LIMIT), start=1):
        parts.append(f"{index}. {sentence}")
    parts.append("")

    unstable = unstable_majors(importance)
    parts.append("### Large terms the folds disagreed on\n")
    if unstable.height == 0:
        parts.append("None: every major term was selected by at least half of the leave-one-dataset-out folds.\n")
    else:
        parts.append(
            f"{unstable.height} of the major terms were selected by fewer than half of the "
            "folds. A large weight and a low selection frequency together mean the term is "
            "doing its work for *this* training set and would be replaced by something else "
            "on another, which the equation as printed does not show. **Do not build "
            "guidance on these.**\n"
        )
        parts.append(_table(unstable) + "\n")

    blocks = term_groups(report.e3.equation, columns, importance)
    parts.append("### Reading the terms in blocks\n")
    parts.append(
        "An additive form invites reading one term at a time, and that works when one or "
        "two weights dominate. When they do not, the honest unit is larger than a term and "
        "smaller than the equation: terms whose per-row contributions move together say the "
        "same thing about a row and can be read as one block. Grouping is on the "
        "contributions rather than on shared features, because two terms can share no "
        "feature and still track each other.\n"
    )
    parts.append(
        f"{blocks.height} blocks over {equation.n_terms} terms"
        + (
            f", the largest holding {int(blocks['n_terms'][0])} terms and "
            f"{float(blocks['share'][0]):.0%} of the weight mass.\n"
            if blocks.height
            else ".\n"
        )
    )
    parts.append(_table(blocks) + "\n")

    parts.append("### Which features the search reached for\n")
    usage = feature_usage(report.e3.equation, importance, dataset_features + model_features)
    used = usage.filter(pl.col("n_terms") > 0)
    parts.append(
        f"**{used.height} of {usage.height}** available meta-features appear in the "
        "equation. `share` sums the weight mass of every term a feature appears in, so a "
        "feature in two terms is credited both and the column does not sum to 1 — it "
        "answers how much of the equation touches this feature, not how much it owns. A "
        "feature the search declined to use after seeing every transform of it is itself a "
        "result.\n"
    )
    parts.append(_table(usage) + "\n")

    parts.append("### Which operations the equation needed\n")
    operations = operation_usage(report.e3.equation, importance, config.max_arity if config is not None else None)
    parts.append(
        "The vocabulary offers five operations and five transforms and the search is free "
        "to ignore any of them, so a row that was offered and went unused is a shape this "
        "data turned out not to need. Rows marked `offered = no` were kept out of the "
        "library by the arity cap and say nothing about the data:\n"
    )
    parts.append(_table(operations) + "\n")

    parts.append("### Where the equation's variance comes from\n")
    for sentence in group_sentences(report.shares):
        parts.append(f"- {sentence}")
    parts.append("")
    parts.append(_table(report.shares) + "\n")

    parts.append("## 4b. The ceiling on model descriptors\n")
    parts.append(
        "Under leave-one-dataset-out every model appears in every training fold, so the "
        "equation's residual can be averaged per model on the training rows and applied to "
        "the held-out dataset with no leak. That replaces the model descriptors with the best "
        "possible substitute -- the model's **identity**, fitted freely -- so what it adds is "
        "a ceiling on what any descriptor set could reach by telling these classifiers "
        "apart.\n"
    )
    parts.append(_table(report.identity) + "\n")
    parts.append(_identity_note(report))

    parts.append("## 5. Best practices\n")
    parts.append(
        "A best practice is general, transferable advice that already circulates in the "
        "field — not a property of this equation. So the practices below are taken from the "
        "literature and this study is used to *weigh* them: each verdict, and the numbers "
        "inside it, are computed from this run by `ml_meta_perf.guidance`, against a stated "
        "threshold, so other data can overturn any of them.\n"
    )
    if frame is not None:
        verdicts = assess(frame, report)
        parts.append(render_guidance(verdicts) + "\n")
        parts.append("At a glance:\n")
        parts.append(_table(as_guidance_table(verdicts).select("practice", "verdict", "magnitude")) + "\n")
    else:
        parts.append("_(not assessed: the meta-dataset was not supplied to the renderer)_\n")

    parts.append("### Each practice against the equation's own terms\n")
    parts.append(
        "The verdicts above are drawn from corpus averages -- family means, variance shares, "
        "paired tests -- which any study with this corpus could compute. This table asks the "
        "stronger question, and the one an interpretability-first study is uniquely able to "
        "ask: **which of the fifteen terms carries this practice, with what strength and "
        "which sign?**\n"
    )
    if frame is not None:
        parts.append(_table(equation_evidence(report, columns)) + "\n")
        parts.append(
            "**One row per (practice, term) pair**, because the equation is a sum of terms and "
            "a term is the unit a practice can be held against. `expected` is what the "
            "practice predicts as the feature rises; `direction` is what *that term's own "
            "contribution* does, measured on the data rather than read off the weight sign -- "
            "which would be wrong the moment the feature sits in a denominator, and several "
            "here do. `not selected` marks a claim resting on a feature the search never took, "
            "so the equation is silent on it rather than supporting it.\n"
        )
        parts.append(_coverage_note(report, columns))
    else:
        parts.append("_(not checked: the meta-dataset was not supplied to the renderer)_\n")

    parts.append("## 5b. The measurements underneath\n")
    parts.append(
        "What the fitted equation says about each raw feature it uses, kept only when the "
        "feature moves predicted MCC enough to matter, does so monotonically enough for a "
        "sentence to be true of it, and does so through terms that survived most folds. "
        "Directions are measured on the data rather than read off weight signs, because a "
        "feature can appear in several terms and inside denominators.\n"
    )
    parts.append(
        f"**These are associations across {datasets} datasets, not causal claims, and not practices "
        "on their own** — a statement about a meta-feature column is a measurement. Section "
        "5 is where they become advice, by supporting or failing to support something a "
        "practitioner could already have been told.\n"
    )
    parts.append(render_practices(report.practices) + "\n")
    parts.append("Evidence:\n")
    parts.append(
        _table(
            report.practices.select("feature", "meaning", "n_terms", "direction", "effect", "stability", "confidence")
            if report.practices.height
            else report.practices
        )
        + "\n"
    )

    parts.append("### These are conditional statements, not marginal ones\n")
    conditional = marginal_versus_conditional(report.practices, columns, truth)
    agree = int(conditional["agrees"].sum()) if conditional.height else 0
    parts.append(
        "A practice states what the *equation* does as a feature rises, with every other "
        "term present. A marginal correlation states what the feature does alone. They are "
        f"different quantities, and here **{agree} of {conditional.height} agree** on the "
        "sign:\n"
    )
    parts.append(_table(conditional) + "\n")
    parts.append(
        "Disagreement is what conditioning does, not a defect. A marginal correlation mixes "
        "a feature's effect with everything it travels with; inside the equation the terms "
        "carrying those companions are already present, so what is left for this feature is "
        "what it adds beyond them. The practical consequence: **these statements describe "
        "what to expect once the other factors are accounted for, not what a scatter plot of "
        "that one feature will show** — and the scatter plot is what a reader will "
        "accidentally check against.\n"
    )

    parts.append("## 5c. Reading a single prediction\n")
    parts.append(
        "The same equation on one row, term by term. The column adds up, and a reader can "
        "check that it does, which is the interpretability payoff in its most direct form "
        "-- and the row is "
        "chosen arithmetically, at the equation's **median absolute error**, so the example "
        "is a typical prediction rather than a flattering one somebody picked.\n"
    )
    if frame is not None:
        parts.append(single_prediction(equation, columns, truth, frame) + "\n")
    else:
        parts.append("_(not shown: the meta-dataset was not supplied to the renderer)_\n")

    parts.append("## 6. Acting on it\n")
    parts.append(
        "R² is the wrong question for a practitioner, who asks whether a model will work on "
        "some data rather than what its MCC will be to three decimals. Thresholding both "
        "the prediction and the truth turns the equation into a go/no-go rule, scored here "
        "on held-out datasets:\n"
    )
    parts.append(_table(report.decision) + "\n")
    parts.append(
        "`majority` is the floor any such rule has to clear. The harder comparison is a "
        'predictor that answers "how well does this model usually do", thresholded the same '
        "way — at both centres, for the reason the error metrics report both:\n"
    )
    parts.append(_table(report.decision_baselines) + "\n")

    parts.append(_protocol_note(report))

    parts.append("Ranking models within a held-out dataset:\n")
    regret = float(np.mean(report.selection["regret"].to_numpy()))
    parts.append(
        f"- mean top-1 regret **{regret:.3f}** MCC — what you give up by taking the model the equation ranks first\n"
    )
    parts.append(_table(report.ranking_baselines) + "\n")
    parts.append(_ranking_verdict(report.ranking_baselines))

    parts.append("## 6b. What an opaque model reaches, and does not\n")
    parts.append(
        "The other side of the trade, priced. Three standard regressors on the same "
        "eighteen raw columns, under the same protocols, with the same clip to the training "
        "fold's range that every reported number uses.\n"
    )
    parts.append(_table(report.opaque) + "\n")
    parts.append(_opaque_note(report))

    parts.append("## 7. Why a random split is not a protocol\n")
    parts.append(
        "The same equation under three splits. A random k-fold puts rows of one dataset on "
        "both sides of the fold, and since the dataset features are constant within a dataset "
        "the equation can memorise dataset identity rather than predict from features. The "
        "gap between the first row and the other two is what that memorisation is worth:\n"
    )
    parts.append(_table(report.leakage) + "\n")

    parts.append("## 8. Equation length\n")
    parts.append(_length_note(report))
    parts.append("The full curve the rule reads, at every length under all three protocols:\n")
    parts.append(_table(report.e3.curve) + "\n")

    parts.append("## 9. The dataset-only and model-only controls\n")
    parts.append(
        "E1 sees dataset meta-features only, so it can predict just one value per dataset; "
        "E2 sees model meta-features only. Together they show how much of MCC each half of "
        "the meta-data explains on its own.\n"
    )
    parts.append(f"**E1** ({report.e1.equation.n_terms} terms):\n")
    parts.append("```\n" + str(report.e1.equation) + "\n```\n")
    parts.append(f"**E2** ({report.e2.equation.n_terms} terms):\n")
    parts.append("```\n" + str(report.e2.equation) + "\n```\n")

    parts.append("### The capability variant, in full\n")
    parts.append(
        "The same feature sets under the looser arity-3 grammar. It is **not** the study's "
        "recommendation and not what the term-by-term analysis above is about; it exists so "
        "that the published equation's accuracy can be read against what the additive *form* "
        "can do, rather than only against oracles and baselines. It is printed here in full "
        "because a ceiling quoted as a number and never shown is a ceiling a reader has to "
        "take on trust -- and because the reason it is not recommended is visible only in the "
        f"reading: {report.e3_capability.equation.n_terms} terms over three-feature "
        "expressions is past the point where the equation can be reasoned about a term at a "
        "time, which is the whole thing this study is trading accuracy for.\n"
    )
    capability_scores = report.e3_capability.cross_validated
    parts.append(
        f"It reaches **{float(report.e3_capability.in_sample['r2']):.4f}** in-sample against "
        f"the published equation's {float(report.e3.in_sample['r2']):.4f}, and "
        f"**{float(capability_scores['loo_dataset']['r2']):.4f}** leave-one-dataset-out "
        f"against {float(report.e3.cross_validated['loo_dataset']['r2']):.4f}.\n"
    )
    parts.append(f"**E3 capability** ({report.e3_capability.equation.n_terms} terms):\n")
    parts.append("```\n" + str(report.e3_capability.equation) + "\n```\n")
    parts.append("LaTeX:\n")
    parts.append("```latex\n" + report.e3_capability.equation.to_latex() + "\n```\n")

    return "\n".join(parts)


#: Which generated section belongs in which chapter, by the ``## N. Title`` heading `render`
#: gives it. The study has six chapters and its results belong *in* them: an earlier layout
#: put every generated table in a separate chapter 10 and had the written chapters point at
#: it, which kept the numbers from going stale and left the results dispersed across two
#: places a reader had to hold at once.
CHAPTER_SECTIONS: dict[str, tuple[str, ...]] = {
    "index.md": ("1c. The headline",),
    "01-dataset.md": ("1b. The corpus",),
    # Ownership, one topic to one chapter. The length rule lives with the selection procedure
    # that applies it, the ceilings with the equation they bound, the protocols with the
    # evaluation. `term_choice` and the length note were rendered into two chapters at once
    # until 2026-09-07, which is most of what made chapters 4 and 5 read as repetitive.
    "03-term-selection.md": ("3b. Why a subset rather than every term", "8. Equation length"),
    "04-equation.md": (
        "2. The equation",
        "3c. How far the form could reach",
        "4. Equation analysis",
        "4b. The ceiling on model descriptors",
        "9. The dataset-only and model-only controls",
    ),
    "05-evaluation.md": (
        "3. How well it does",
        "6. Acting on it",
        "6b. What an opaque model reaches, and does not",
        "7. Why a random split is not a protocol",
    ),
    "06-practices.md": ("5. Best practices", "5b. The measurements underneath", "5c. Reading a single prediction"),
}

#: The markers a generated block sits between. Everything between them is replaced on every
#: run; everything outside them is written by hand and never touched.
BEGIN, END = "<!-- generated: do not edit below -->", "<!-- end generated -->"


def sections(text: str) -> dict[str, str]:
    """Split a rendered report into its ``## N. Title`` sections, keyed by that title."""
    found: dict[str, str] = {}
    title, body = None, []
    for line in text.split("\n"):
        if line.startswith("## "):
            if title is not None:
                found[title] = "\n".join(body).strip()
            title, body = line[3:].strip(), []
        elif title is not None:
            body.append(line)
    if title is not None:
        found[title] = "\n".join(body).strip()
    return found


def _demote(body: str) -> str:
    """Push a section's own subheadings down one level, so it nests under a chapter heading."""
    return "\n".join(("#" + line) if line.startswith("###") else line for line in body.split("\n"))


def splice(page: str, block: str) -> str:
    """Replace the generated block in ``page``, or insert one if it has none.

    A new block goes *before* the chapter's closing limitations section, so a chapter reads
    method, then results, then caveats -- the order a paper is read in. Replacing an existing
    block leaves it wherever it already is, so the position survives a hand edit.
    """
    stamped = f"{BEGIN}\n\n{block.strip()}\n\n{END}"
    if BEGIN in page and END in page:
        head, rest = page.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        return head + stamped + tail
    for line in page.split("\n"):
        if line.startswith("## Limitations"):
            head, tail = page.split(line, 1)
            return head.rstrip() + "\n\n" + stamped + "\n\n" + line + tail
    return page.rstrip() + "\n\n" + stamped + "\n"


def write_into_chapters(
    report: Report,
    columns: dict[str, np.ndarray],
    truth: np.ndarray,
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    docs: str | Path,
    *,
    frame: pl.DataFrame | None = None,
    config: Configuration | None = None,
    source: str | None = None,
) -> list[Path]:
    """Write each generated section into the chapter it belongs to.

    The results live in the chapters that discuss them rather than in a separate generated
    chapter, and they are still generated rather than narrated: everything between `BEGIN` and
    `END` is replaced on every run, and everything outside is hand-written prose that the
    pipeline never touches. A chapter therefore cannot carry a stale table, which is the
    property the separate report chapter existed to guarantee.
    """
    text = render(
        report,
        columns,
        truth,
        dataset_features,
        model_features,
        frame=frame,
        config=config,
        source=source,
    )
    available = sections(text)
    folder = Path(docs)
    written: list[Path] = []
    for filename, titles in CHAPTER_SECTIONS.items():
        page = folder / filename
        if not page.is_file():
            continue
        blocks: list[str] = []
        for title in titles:
            body = available.get(title)
            if body:
                # Drop the "N. " ordinal: the chapter supplies the position, and a generated
                # heading numbered against the report would contradict the chapter it sits in.
                heading = title.split(". ", 1)[-1]
                blocks.append(f"## {heading}\n\n{_demote(body)}")
        if blocks:
            page.write_text(splice(page.read_text(), "\n\n".join(blocks)))
            written.append(page)
    return written


def write(
    report: Report,
    columns: dict[str, np.ndarray],
    truth: np.ndarray,
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    path: str | Path,
    *,
    frame: pl.DataFrame | None = None,
    config: Configuration | None = None,
    source: str | None = None,
) -> Path:
    """Render the report and write it to ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render(
            report,
            columns,
            truth,
            dataset_features,
            model_features,
            frame=frame,
            config=config,
            source=source,
        ),
        encoding="utf-8",
    )
    return destination


def glossary() -> pl.DataFrame:
    """The feature glossary as a table, for appending to a report or a paper.

    Restricted to `ALL_FEATURES`, which is narrower than `FEATURE_GLOSSARY`: the glossary also
    explains the four model columns retired on 2026-09-05, because the corpus still carries
    them, and a reader of *this* report should see only what the equations may draw on.
    """
    return pl.DataFrame([{"feature": name, "meaning": FEATURE_GLOSSARY[name]} for name in ALL_FEATURES])
