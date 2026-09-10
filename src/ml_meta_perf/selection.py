"""Choosing how many terms the published equation should have.

More terms fit better and read worse, so the length is a trade rather than an optimum, and
the study wants that trade decided by a stated rule rather than by eye.

**Knee detection used to be that rule and has been removed.** It was tried thoroughly: three
detectors (``autoelbow``, ``kneedle``, ``lmethod``) on each protocol's curve and on a
consensus of them, raw and after global RDP simplification at seven tolerances. The
simplification worked as intended -- on the raw curve the detectors split 6/8/4 and after
gRDP they agree at 8 -- and a Pareto-front knee, by all three of the standard rules (closest
to the utopia point, furthest from the nadir, maximum distance from the chord between the
front's extremes), lands in the same place. Every geometric reading of this curve says four
to eight terms.

And every one of those lengths is **significantly worse** than the published equation when
the two are paired over the twenty held-out datasets. That is not a failure of the detectors;
it is the two questions being different. A knee finds where the *marginal* return per term
collapses, which on this curve is early because R2 saturates. It does not ask whether the
accuracy still being added is real -- and here it is, for several terms past the bend.
Reporting a knee that never chose the published length was a standing contradiction.

**The configuration rule is `best_configuration`, and it is in two stages.** Given one curve
per grammar:

* `floor_curve` scores each length by its **worst** protocol of the four -- in-sample,
  leave-one-dataset-out, leave-one-model-out, and the doubly-held-out cell -- and
  `arity_candidates` takes that curve's argmax within each grammar. An argmax, so no
  threshold, no smoothing window and no sensitivity parameter.
* `best_configuration` then chooses the **grammar**: the smallest `complexity` among the
  candidates that `most_capable` does not beat by more than the spread of that beating.
  `grammar_margin` is the number it compares -- the accuracy the larger grammar buys over the
  paired bootstrap spread of that same accuracy -- and the bar is a ratio of one, where signal
  equals noise. On the corrected corpus that ratio is 3.09, so the extra grammar capacity is
  retained.

On the corrected corpus both rules select **arity 3 with 25 terms**. Neither number is written
down anywhere; both fall out of the validation curves.

**It takes no row count, and that is the point of the revision.** The rule it replaced was
the consensus discounted by adjusted R2's degrees-of-freedom factor against a complexity
budget, which priced a feature slot by the corpus size: holding the curve and the folds fixed
and changing only the row count could flip its answer while the plateau it was supposed to
be reading never moved. There is now nothing for a
corpus size to enter through. What decides an extra term is whether the accuracy it adds
survives being paired over the held-out groups, which is a property of the data rather than
of how much of it there is. The full account is in `best_configuration`, including what is
still owed: the rule was written knowing the answer it had to reproduce.

`best_length` -- the argmax of the three-protocol `consensus_curve` -- remains the per-length
reading the chapters plot and the figures caption, and is unchanged.

Three further readings are computed and reported rather than used, because a selection rule
is only defensible if the alternatives it beats are on the page:

* the **Pareto front** over (length, accuracy), and `pareto_knee` -- the honest geometric
  summary, needing no detector, no threshold and no smoothing;
* the **consensus curve** itself, so that no single protocol can decide a length alone;
* the **parsimony alternative** in `ml_meta_perf.experiment.length_comparison` -- the shortest
  length whose paired interval against the chosen one spans zero. On this corpus that is 23
  terms under the published arity-3 grammar. It is reported and not adopted: the accuracy it
  gives up is measurable even where it is not significant.

Study chapter: [3. Term generation and selection](../../assets/docs/03-term-selection.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ml_meta_perf.validate import BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED

#: The three protocols a length can be judged on, in the order `consensus_curve` combines them.
PROTOCOLS = ("r2_in_sample", "r2_loo_dataset", "r2_loo_model")

#: Every protocol a *configuration* is judged on, which is `PROTOCOLS` plus the doubly-held-out
#: one. `floor_curve` takes the minimum over these and `best_configuration` selects on it.
#:
#: The cell protocol is in this set and not in `PROTOCOLS` deliberately. `consensus_curve` and
#: `best_length` are the per-length reading the chapters plot and the figures caption, and they
#: stay on the three single-group protocols where they have always been. The configuration rule
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


def pareto_front(curve: pl.DataFrame, column: str = "r2_loo_dataset") -> pl.DataFrame:
    """Lengths that no shorter equation beats on ``column``.

    A length is on the front when nothing shorter scores at least as well. Reading down the
    front gives the shortest equation achieving each level of accuracy, which is the decision
    the term budget actually poses.
    """
    table = curve.sort("n_terms")
    scores = table[column].to_numpy()
    keep = [index for index in range(len(scores)) if not np.any(scores[:index] >= scores[index])]
    return table[keep]


def pareto_table(curve: pl.DataFrame) -> pl.DataFrame:
    """Per-length flags for whether a length is on the in-sample or transfer front.

    Both fronts are reported because they answer the two halves of the trade. The in-sample
    front asks "what is the shortest equation achieving this quality of fit", which is the
    explainability question; the cross-validated front asks the same about transfer. A length
    on both is defensible under either priority, and on this data the in-sample front is every
    length -- fit is monotone in terms, so nothing is ever dominated -- which is itself the
    reason the in-sample curve alone cannot choose a length.
    """
    table = curve.sort("n_terms")
    columns = [column for column in ("r2_in_sample", "r2_loo_dataset") if column in table.columns]
    flags: dict[str, list[bool]] = {}
    for column in columns:
        scores = table[column].to_numpy()
        front = {index for index in range(len(scores)) if not np.any(scores[:index] >= scores[index])}
        flags[f"front_{column.removeprefix('r2_')}"] = [index in front for index in range(len(scores))]
    return table.select("n_terms", *columns).with_columns(
        **{name: pl.Series(name, values) for name, values in flags.items()}
    )


def pareto_knee(curve: pl.DataFrame, column: str = "consensus") -> dict[str, int]:
    """Where the Pareto front bends, by the three standard multi-criteria rules.

    Reported as a diagnostic and **not** as the selection rule. Each normalises the front to
    the unit square and then picks the point closest to the utopia corner (fewest terms, best
    accuracy), the point furthest from the nadir corner, or the point furthest from the chord
    joining the front's two extremes. On this data all three agree with each other and with
    the knee detectors, and all three choose a length the paired test rejects -- which is the
    finding, and the reason `experiment.length_comparison` decides instead.
    """
    table = curve.sort("n_terms")
    sizes = table["n_terms"].to_numpy().astype(np.float64)
    scores = consensus_curve(table) if column == "consensus" else table[column].to_numpy()

    keep = [index for index in range(len(scores)) if not np.any(scores[:index] >= scores[index])]
    if len(keep) < 3:
        return {}
    front_sizes, front_scores = sizes[keep], scores[keep]

    def unit(values: np.ndarray) -> np.ndarray:
        span = float(values.max() - values.min())
        return (values - values.min()) / (span if span > 1e-12 else 1.0)

    x, y = unit(front_sizes), unit(front_scores)
    to_ideal = np.hypot(x, y - 1.0)
    to_nadir = np.hypot(x - 1.0, y)
    numerator = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    chord = float(np.hypot(y[-1] - y[0], x[-1] - x[0])) or 1.0
    return {
        "closest_to_ideal": int(front_sizes[int(to_ideal.argmin())]),
        "furthest_from_nadir": int(front_sizes[int(to_nadir.argmax())]),
        "furthest_from_chord": int(front_sizes[int((numerator / chord).argmax())]),
    }


def best_length(curve: pl.DataFrame, column: str = "consensus") -> int:
    """The length that maximises ``column``: the three-protocol reading of the same question.

    **`floor_argmax` is what publishes a length**, and this is what is reported beside it.
    The two differ in what they combine: this takes the median of the three single-group
    protocols, that takes the minimum over all four including the doubly-held-out cell. On the
    current corpus they agree -- both select 25 terms under the retained arity-3 grammar --
    and they are kept apart because agreement is a result rather than a
    guarantee, and because the chapters plot and caption this reading.

    Until 2026-09-09 this docstring called itself the rule and `run_equation` had already
    stopped calling it. Nothing caught that, because the two agreed.

    It has no threshold, no smoothing window and no sensitivity parameter, so it is a property
    of the curve rather than of a value someone picked to get a preferred answer -- and it
    re-derives itself if the corpus changes, which a hardcoded length cannot.

    Run on `consensus_curve` by default, never on in-sample alone: in-sample is monotone in
    the number of terms, so its argmax is always the longest equation searched and the rule
    would be vacuous.

    Neither length is written down anywhere: both fall out of a curve.

    `experiment.length_comparison` reports the alternative that trades accuracy for brevity:
    the shortest length whose paired interval against this one spans zero. On the current
    corpus that is 23 terms under arity 3. It is reported rather than
    adopted, because the accuracy it gives up is measurable even though it is not significant.
    """
    scores = consensus_curve(curve) if column == "consensus" else curve[column].to_numpy()
    return int(curve["n_terms"].to_numpy()[int(np.argmax(scores))])


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

    **Reported, not selected on** -- `best_configuration` decides -- and reported because the
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

    `experiment.run_equation` calls it to decide how long the equation it just fitted should
    be, and `arity_candidates` calls it once per grammar to build the candidate set
    `best_configuration` chooses among. One function so the two cannot drift: the length a
    published equation has and the length its grammar is represented by in the configuration
    rule are the same number by construction.

    An argmax, so there is no threshold, no smoothing window, no sensitivity parameter and no
    corpus size in it -- see `best_configuration` for why that last one matters.
    """
    return int(curve["n_terms"].to_numpy()[floor_argmax_index(curve)])


def arity_candidates(curves: dict[int, pl.DataFrame]) -> dict[int, int]:
    """One length per arity: the argmax of that arity's `floor_curve`.

    **The candidate set is one length per grammar, and that is what keeps the rule honest.**
    Running the paired test of `best_configuration` over *every* length instead is the rule
    this replaces on the other side -- it was measured, and it admits nine terms at arity 4
    (0.057 below the floor the study will not go under) because a paired test over twenty
    groups cannot separate a nine-term equation from a twenty-three-term one. The test is not
    strong enough to choose a length and is never asked to. Length is chosen here, by an
    argmax with no parameter in it; the test only ever chooses a *grammar*.
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


def grammar_margin(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    """What a larger grammar buys, the spread of that same quantity, and their ratio.

    ``candidate`` and ``reference`` are per-held-out-group errors on the same folds, in the
    same group order. Returns ``(gain, scale, ratio)``: the mean per-group error the reference
    saves over the candidate, the paired bootstrap standard error of that mean, and
    ``gain / scale``. **The ratio is the quantity `best_configuration` compares against one**,
    so the decision is a number to overcome rather than a verdict to fail to reach.

    **Why this and not `paired_comparison(...).significant`.** Significance is a
    failure-to-reject: "we could not tell these apart" is absence of evidence, it is decided
    by the test's power rather than by anything about the equations, and on twenty groups at
    a fold spread of 0.059 pooled R2 that power is low enough to wave almost anything through.
    Measured on this corpus, the significance form let *every* candidate past and returned the
    first one it visited, so the smallest complexity won by default and the statistic decided
    nothing -- which is exactly the criticism that retired the rule before it ("the
    discrimination came entirely from preferring the simplest grammar, so the statistics were
    decoration"). At a ratio of one the same comparison rejects the six-, eight-, nine- and
    twelve-term arity-2 equations where significance rejected only six and eight.

    **The one in ``ratio > 1`` is where signal equals noise, not a tuned constant**, and the
    verdict is not knife-edge on it: the arity-3 candidate's advantage over arity 2 comes out
    at a ratio of **3.09**. Report the ratio and a reader can apply their own bar.

    **Where the bar sits relative to a sign test, measured, because it is not obvious.** The
    ratio is a signal-to-noise reading on the *mean*, so concentration lowers it but does not
    veto: on twenty folds the same mean gain scores 1.02 when one group carries all of it,
    1.50 when two do, 2.23 when four do, and unbounded when every group carries an equal
    share. A gain one group could have produced therefore sits right on the bar rather than
    below it, which is looser than the sign test -- that would call it a tie. Looser is the
    intended direction here: the standing warning in this project is that the paired test
    *under*-calls, having once scored a 0.203 collapse of leave-one-dataset-out R2 as a tie.
    What a magnitude reading owes in return is that it is never read alone, and `floor_curve`
    and `protocol_spread` are what it is read beside.

    Both numbers come from the same twenty folds and neither is a function of the corpus size,
    so the invariance the revision was for is unaffected: hold the curve and the folds and
    nothing here moves. Growing the corpus moves the gain and the spread together, which is
    learning from more data rather than a price changing underneath a fixed measurement.
    """
    if candidate.shape != reference.shape:
        raise ValueError("a grammar margin needs one error per group on both sides")
    differences = candidate - reference
    gain = float(differences.mean())
    # A gain of zero or less is no gain: the larger grammar is not better and the ratio must
    # not be able to take it, whatever the spread happens to be.
    if gain <= 0.0:
        return gain, 0.0, 0.0
    # No fold-to-fold variation at all is a gain that is perfectly consistent, not one that is
    # unmeasurable. Tested exactly rather than against a tolerance, because a tolerance here
    # would be the free parameter this rule exists to avoid.
    if float(np.ptp(differences)) == 0.0:
        return gain, 0.0, float("inf")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, differences.size, size=(BOOTSTRAP_ROUNDS, differences.size))
    scale = float(differences[draws].mean(axis=1).std())
    return gain, scale, gain / scale


def best_configuration(
    curves: dict[int, pl.DataFrame],
    errors: dict[int, dict[int, np.ndarray]],
) -> tuple[int, int]:
    """``(arity, n_terms)`` for **the study's equation**: the simplest grammar that is not
    measurably worse than the most capable one.

    Two stages, and each is answered by the tool that can answer it:

    * **the length**, per arity, is `arity_candidates` -- the argmax of `floor_curve`. No
      threshold, no smoothing, no free parameter, and no corpus size anywhere in it.
    * **the grammar** is the smallest `complexity` among the candidates that `most_capable`
      does not beat by more than the spread of that beating -- `grammar_margin` returns the
      ratio and this compares it against one. ``errors`` is one array of per-held-out-group
      errors per candidate, keyed ``arity -> n_terms``, under the protocol the floor lands
      on: the doubly-held-out one.

    **The decision is a value to overcome, and that is a correction made after the first
    version of this rule.** That version asked whether `paired_comparison` found the
    difference *significant*, which is a failure-to-reject and decides nothing when the test
    has no power: measured, it waved every candidate through and returned the first one
    visited, so complexity decided alone and the statistic was decoration. `grammar_margin`
    replaces it with a ratio of two measured quantities -- the accuracy the larger grammar
    buys, over the paired spread of that same accuracy -- and a larger grammar is taken when
    that ratio exceeds one. On the corrected corpus the ratio is **3.09**, so arity 3 is
    retained.

    **What this fixes.** The rule it replaces was the consensus discounted by adjusted R2's
    degrees-of-freedom factor against a complexity budget. It gave the right answer here and
    it was not a rule about the curve at all: it charged a price per feature slot, and the
    price is set by the number of rows. Holding the curve, the folds and the plateau fixed and
    changing only the corpus size could move its answer even though the curve did not change,
    because slots get cheaper as the corpus grows until the penalty vanishes and the rule
    converges on its own unpenalised argmax. This corpus is going to grow, so that was a
    verdict scheduled to change with nothing about the equations changing.

    **This rule takes no row count.** It is a function of the curves and of the folds those
    curves were measured on, which is what the required invariance check amounts to: hold the
    curve fixed, vary the hypothetical corpus size, and the answer cannot move because there
    is nothing for the size to enter through. What decides an extra term is whether the
    accuracy it adds is larger than the spread between held-out groups, which is a property of
    the data and not of how much of it there is.

    **What the complexity tie-break is standing in for**, and `protocol_spread` measures it:
    the candidate this returns does not merely cost fewer feature slots than `most_capable`'s,
    it also *falls less far from its own fit*. On this corpus (2, 12) drops 0.039 from
    in-sample to its worst protocol and (3, 14) drops 0.050, for a floor 0.003 higher. The
    larger grammar buys a level inside the fold-to-fold spread and pays for it in consistency.
    That is reported beside the decision rather than folded into it: combining a level and a
    spread needs a weight, and a weight is the free parameter this revision removed.

    **Both halves are still measured rather than assumed**, and the disclosure the old
    docstring carried still applies to this one: it was written knowing the answer it had to
    reproduce, and it has not been tested against a corpus whose right answer is unknown.
    What is now defensible is the *shape*: no free parameter, no corpus size, a length chosen
    by an argmax and a grammar chosen by a test that demonstrably fires -- it rejects the
    six- and eight-term arity-2 equations against the same reference.
    """
    candidates = arity_candidates(curves)
    top_arity, top_size = most_capable(curves)
    reference = errors[top_arity][top_size]
    # `most_capable`'s own candidate is in this loop and compares against itself: every
    # difference is zero, so its ratio is zero and the loop always returns by the time it
    # reaches that entry. The line after it is for an empty ``curves``.
    for _, arity, size in sorted((complexity(arity, size), arity, size) for arity, size in candidates.items()):
        _, _, ratio = grammar_margin(errors[arity][size], reference)
        if ratio <= 1.0:
            return arity, size
    return top_arity, top_size


def recommend(curve: pl.DataFrame, published: int | None = None) -> pl.DataFrame:
    """Every length a stated rule could pick, with the evidence, as one small table.

    None of these rows *is* the decision -- `experiment.length_comparison` is, because it is
    the only one that asks whether a difference is larger than the spread between folds. These
    are the geometric readings, reported so that a reader can see they disagree with it and
    by how much.
    """
    rows: list[dict[str, object]] = []
    sizes = curve["n_terms"].to_numpy()

    def row(rule: str, index: int) -> dict[str, object]:
        return {
            "rule": rule,
            "n_terms": int(sizes[index]),
            "r2_in_sample": float(curve["r2_in_sample"].to_numpy()[index]),
            "r2_loo_dataset": (
                float(curve["r2_loo_dataset"].to_numpy()[index]) if "r2_loo_dataset" in curve.columns else float("nan")
            ),
        }

    for rule, length in pareto_knee(curve).items():
        matches = np.flatnonzero(sizes == length)
        if matches.size:
            rows.append(row(f"pareto front, {rule.replace('_', ' ')}", int(matches[0])))

    if "r2_loo_dataset" in curve.columns:
        rows.append(row("best loo-dataset", int(np.argmax(curve["r2_loo_dataset"].to_numpy()))))

    consensus = consensus_curve(curve)
    rows.append(row("best consensus (median of three)", int(np.argmax(consensus))))
    # The rule that actually publishes a length. Reported beside the consensus reading rather
    # than instead of it: the two agree on this corpus and the table exists to show what the
    # rule beats, which includes showing that a second defensible reading lands in the same
    # place. The label said "the rule" on the consensus row until 2026-09-09, by which time
    # `run_equation` had been calling this one for a day.
    rows.append(row("best floor over four protocols (the rule)", floor_argmax_index(curve)))

    if published is not None:
        matches = np.flatnonzero(sizes == published)
        if matches.size:
            rows.append(row("published", int(matches[0])))
    return pl.DataFrame(rows)
