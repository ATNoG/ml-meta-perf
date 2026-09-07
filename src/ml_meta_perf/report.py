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

Study chapter: [7. From equation to practice](../../assets/docs/07-practices.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl

from ml_meta_perf.attribution import classify, contributions
from ml_meta_perf.data import ALL_FEATURES, FEATURE_GLOSSARY
from ml_meta_perf.experiment import Configuration, Report
from ml_meta_perf.guidance import as_table as as_guidance_table
from ml_meta_perf.guidance import assess
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


def _scores(label: str, scores: dict[str, float | int]) -> str:
    return (
        f"| {label} | {float(scores['r2']):.4f} | {float(scores['mae']):.4f} | "
        f"{float(scores['rmse']):.4f} | {int(scores['n'])} |"
    )


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

    parts.append("## 5b. The measurements underneath\n")
    parts.append(
        "What the fitted equation says about each raw feature it uses, kept only when the "
        "feature moves predicted MCC enough to matter, does so monotonically enough for a "
        "sentence to be true of it, and does so through terms that survived most folds. "
        "Directions are measured on the data rather than read off weight signs, because a "
        "feature can appear in several terms and inside denominators.\n"
    )
    parts.append(
        "**These are associations across 20 datasets, not causal claims, and not practices "
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

    parts.append("## 6. Acting on it\n")
    parts.append(
        "R² is the wrong question for a practitioner, who asks whether a model will work on "
        "some data rather than what its MCC will be to three decimals. Thresholding both "
        "the prediction and the truth turns the equation into a go/no-go rule, scored here "
        "on held-out datasets:\n"
    )
    parts.append(_table(report.decision) + "\n")
    parts.append("Ranking models within a held-out dataset:\n")
    correlation = float(np.mean(report.selection["spearman"].to_numpy()))
    regret = float(np.mean(report.selection["regret"].to_numpy()))
    parts.append(
        f"- mean rank correlation **{correlation:.3f}**\n"
        f"- mean top-1 regret **{regret:.3f}** MCC — what you "
        "give up by taking the model the equation ranks first\n"
    )

    parts.append("## 7. What bounds the result\n")
    parts.append(
        "The additive form cannot represent dataset-by-model interaction beyond what its mixed "
        "terms reach. The ladder below adds interaction components to an oracle that is "
        "handed the true group means, so it measures the ceiling rather than any equation:\n"
    )
    parts.append(_table(report.oracles) + "\n")
    parts.append("Variance of MCC explained by group identity alone, with no equation involved:\n")
    parts.append(_table(report.decomposition) + "\n")
    parts.append("Validation protocol, same equation, different splits:\n")
    parts.append(_table(report.leakage) + "\n")

    parts.append("## 8. Equation length\n")
    parts.append(
        "Where additional terms stop paying, by knee detection on the accuracy-versus-length "
        "curve and by Pareto dominance:\n"
    )
    parts.append(_table(report.term_choice) + "\n")
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

    return "\n".join(parts)


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
