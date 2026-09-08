"""Beam-search policies: how the beam is pruned, kept diverse, and started.

`ml_meta_perf.fit.Selector` runs a textbook beam: expand every parent by a fixed number of
candidates, score them all, keep the best ``beam_width``. That is the 1970s formulation, and
the decoding literature has since accumulated a family of variants that change what a beam
step costs and what it keeps. This module makes those variants selectable so they can be
*measured* against the incumbent rather than argued about.

The taxonomy follows the beam-search literature; the mapping onto this problem is the part
that needed thought, because subset selection is not sequence decoding.

**Adaptive pruning** (Freitag & Al-Onaizan, WMT 2017). A fixed beam width either keeps
candidates that are hopeless or discards ones that narrowly missed. Score-based pruning drops
a child whose objective is too far from the step's best, so the beam *narrows itself* where
the landscape is peaked and stays wide where it is flat. Their relative form prunes at a
*ratio* of the best score, which does not transfer here: `fit.Subset.rss` is a penalised
objective that ridge shrinkage can drive slightly negative, and a ratio test on a quantity
that changes sign is meaningless. Both forms here are therefore differences, with the
"relative" one taken as a fraction of the target's total sum of squares -- the natural scale
of the problem, and the same denominator R2 uses.

**Max candidates per history** (same paper). Cap how many children any one parent may
contribute to the surviving beam. Without it a single strong parent can fill the whole beam
with variations of itself, which is the failure a beam is supposed to prevent: width spent on
one idea rather than several. This is the diversity mechanism most likely to matter here,
because the term library is deliberately redundant and near-duplicate terms produce
near-identical children.

**Determinantal selection** (Meister et al., EMNLP 2021). Choose the beam to maximise
``log det(D + w K)`` over a similarity kernel rather than to maximise score alone -- a
principled continuum from pure score (``w = 0``) to pure diversity. The kernel here is
feature overlap between subsets, which is the similarity that matters for an equation meant
to be read: two subsets over the same features say the same thing twice.

**Space-filling initialisation** (ESS/OBL). A beam that starts from the empty set has one
first move: take the highest-scoring terms. Those look alike -- the strongest terms in a
redundant library are near-duplicates of each other -- so the beam's ``k`` starting points
can all be the same idea, and every later step inherits that. Seeding the first step with
terms chosen to be *spread out* in term space instead gives the beam genuinely different
starting points. See `ml_meta_perf.seeding`.

**Nothing here changes the published equation unless it is asked to.** `VANILLA` is the
incumbent policy and is bit-identical to the code before this module existed; the tests pin
that, because a refactor that quietly moved the published result would be worse than no
refactor at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

#: A subset's objective, as `fit.Subset.rss` reports it: penalised residual sum of squares,
#: lower is better, and **it may be negative**. Every comparison here is a difference for
#: that reason.
Objective = float


@dataclass(frozen=True)
class BeamPolicy:
    """One configuration of the beam's pruning, diversity and initialisation.

    Every field is off by default, so `BeamPolicy()` is the textbook beam. That is deliberate:
    a variant has to be switched on explicitly and measured, and the default path stays the
    one the study publishes.
    """

    name: str = "vanilla"

    #: Absolute pruning: drop a child whose objective exceeds the step's best by more than
    #: this. In MCC-squared units, so it is comparable across equation lengths but not across
    #: corpora -- use ``relative_prune`` for something scale-free.
    absolute_prune: float | None = None

    #: Relative pruning, as a fraction of the target's total sum of squares. ``0.01`` keeps
    #: children within 1% of the total variance of the step's best. Scale-free, and the same
    #: denominator R2 divides by, so a value here reads as "worth this much R2".
    relative_prune: float | None = None

    #: Max candidates per history: how many children one parent may place in the surviving
    #: beam. ``None`` is unlimited, which is the textbook behaviour and the one that lets a
    #: single parent monopolise the width.
    per_parent_cap: int | None = None

    #: Weight on the determinantal diversity term when selecting the beam. ``0.0`` selects on
    #: score alone. Larger values buy spread at the cost of score; the kernel is feature
    #: overlap, so what it spreads is *what the equation talks about*.
    diversity_weight: float = 0.0

    #: How many of the first step's beam slots are filled by space-filling selection over
    #: term space rather than by score. ``0`` disables it. Capped at the beam width by the
    #: caller, since seeding more starting points than the beam can hold does nothing.
    seed_terms: int = 0

    #: How many extra children **every** step proposes from the guided space-filling field,
    #: on top of the ones the residual ranking generates. ``0`` disables it.
    #:
    #: This is the difference between using ESS as a design and using it as a sampler.
    #: ``seed_terms`` calls it once, before the search knows anything, and never speaks to it
    #: again -- which is what the 2026-09-08 focused run measured and rejected. ``probe_terms``
    #: runs it every step against the objectives the beam has actually computed, so the field
    #: it places into is refitted as the search learns. See `ml_meta_perf.seeding.ProbeField`.
    probe_terms: int = 0

    #: How hard the probe field pulls towards the regions that scored well, in ESS's own
    #: units. ``0.0`` is pure repulsion -- space-filling with no idea where the good terms are,
    #: which is the null this variant has to beat. Larger values trade coverage for
    #: exploitation, and the beam already exploits, so the useful range is small.
    probe_attraction: float = 0.5

    def is_vanilla(self) -> bool:
        """Whether this policy leaves every beam decision to the incumbent code path.

        Checked rather than assumed at the point of use: the fast path skips machinery that
        would otherwise have to be proven a no-op, and "proven" here means a test that fits
        the real corpus both ways and compares the equations.
        """
        return (
            self.absolute_prune is None
            and self.relative_prune is None
            and self.per_parent_cap is None
            and self.diversity_weight == 0.0
            and self.seed_terms == 0
            and self.probe_terms == 0
        )

    def threshold(self, best: Objective, total: float) -> Objective | None:
        """The worst objective a child may have and still survive pruning.

        ``None`` when no pruning is configured. When both forms are set the *tighter* one
        wins, so combining them can only narrow the beam -- which keeps a swept grid over the
        two from containing points that are looser than either alone.
        """
        limits = [
            best + self.absolute_prune if self.absolute_prune is not None else None,
            best + self.relative_prune * total if self.relative_prune is not None else None,
        ]
        present = [limit for limit in limits if limit is not None]
        return min(present) if present else None

    def with_name(self, name: str) -> BeamPolicy:
        return replace(self, name=name)


#: The incumbent. Every reported number in the study comes from this policy.
VANILLA = BeamPolicy()


def prune(
    scores: np.ndarray,
    policy: BeamPolicy,
    total: float,
) -> np.ndarray:
    """A boolean mask over children: which survive this step's pruning.

    ``scores`` must already be the children's objectives; the caller sorts. Returns an
    all-true mask when the policy configures no pruning, so the caller can apply it
    unconditionally without branching.

    **At least one child always survives.** A threshold that excluded everything would end
    the search a step early and silently return a shorter equation than asked for, which is a
    far worse failure than a beam that is occasionally wider than intended.
    """
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    limit = policy.threshold(float(scores.min()), total)
    if limit is None:
        return np.ones(scores.shape[0], dtype=bool)
    keep = scores <= limit
    if not keep.any():
        keep[int(np.argmin(scores))] = True
    return keep


def cap_per_parent(parents: list[int], width: int, cap: int | None) -> list[int]:
    """Positions to keep, in the order given, with at most ``cap`` from any one parent.

    ``parents`` names the parent each child came from, already ordered best-first. Returns
    positions rather than a mask because the caller needs the order preserved: the beam is
    filled in score order and the cap only decides who is skipped.

    With ``cap`` unset this is ``range(width)`` -- the textbook behaviour, in which one parent
    may take every slot.
    """
    if cap is None:
        return list(range(min(width, len(parents))))
    taken: dict[int, int] = {}
    kept: list[int] = []
    for position, parent in enumerate(parents):
        if len(kept) >= width:
            break
        if taken.get(parent, 0) >= cap:
            continue
        taken[parent] = taken.get(parent, 0) + 1
        kept.append(position)
    # A cap tight enough to starve the beam is worse than no cap: the search would proceed on
    # fewer parents than the width asks for and the comparison with the incumbent would be
    # confounded by that rather than by the diversity the cap was meant to buy. Backfill in
    # score order with whatever the cap excluded.
    if len(kept) < min(width, len(parents)):
        chosen = set(kept)
        for position in range(len(parents)):
            if len(kept) >= width:
                break
            if position not in chosen:
                kept.append(position)
        kept.sort()
    return kept


def determinantal_select(
    scores: np.ndarray,
    features: list[frozenset[str]],
    width: int,
    weight: float,
) -> list[int]:
    """Choose ``width`` children maximising score and feature spread together.

    A greedy maximisation of ``score_gain + weight * novelty``, where novelty is how much of a
    candidate's feature set is not already represented by what has been chosen. That is the
    determinantal objective's practical shape -- the log-determinant of a kernel is large when
    the chosen items cover different directions -- without building and factorising a kernel
    per step, which at several hundred thousand steps would dominate the run.

    Scores are objectives, so they are *minimised*; they are negated and rescaled to the unit
    interval before combining, so ``weight`` reads the same regardless of the corpus scale.
    ``weight = 0`` reduces to taking the best ``width`` in order, and is asserted to.
    """
    order = list(range(len(scores)))
    if weight <= 0.0 or not features:
        return order[: min(width, len(order))]

    span = float(scores.max() - scores.min())
    quality = np.zeros_like(scores) if span <= 0.0 else (float(scores.max()) - scores) / span

    chosen: list[int] = []
    covered: set[str] = set()
    remaining = set(order)
    while remaining and len(chosen) < width:
        best_position, best_value = None, -np.inf
        for position in sorted(remaining):
            names = features[position]
            novelty = len(names - covered) / len(names) if names else 0.0
            value = float(quality[position]) + weight * novelty
            if value > best_value:
                best_position, best_value = position, value
        assert best_position is not None
        chosen.append(best_position)
        covered |= features[best_position]
        remaining.discard(best_position)
    return chosen


#: The variants the sweep compares, each a named point rather than a range. Named so a result
#: can be reported as "history cap 2 beats the incumbent" rather than as a tuple of knobs.
CATALOGUE: tuple[BeamPolicy, ...] = (
    VANILLA,
    BeamPolicy(name="prune-relative-0.005", relative_prune=0.005),
    BeamPolicy(name="prune-relative-0.02", relative_prune=0.02),
    BeamPolicy(name="prune-relative-0.05", relative_prune=0.05),
    BeamPolicy(name="cap-1", per_parent_cap=1),
    BeamPolicy(name="cap-2", per_parent_cap=2),
    BeamPolicy(name="cap-3", per_parent_cap=3),
    BeamPolicy(name="diverse-0.1", diversity_weight=0.1),
    BeamPolicy(name="diverse-0.3", diversity_weight=0.3),
    BeamPolicy(name="diverse-1.0", diversity_weight=1.0),
    BeamPolicy(name="ess-seed-6", seed_terms=6),
    BeamPolicy(name="ess-seed-12", seed_terms=12),
    BeamPolicy(name="cap-2+prune-0.02", per_parent_cap=2, relative_prune=0.02),
    BeamPolicy(name="cap-2+ess-6", per_parent_cap=2, seed_terms=6),
    BeamPolicy(name="cap-2+diverse-0.3", per_parent_cap=2, diversity_weight=0.3),
    # ESS used as a sampler rather than as a design. `probe-0` is the null the guided ones have
    # to beat: the same probes, placed by repulsion alone, with the measured objectives ignored.
    # Without it a win could be the extra candidates rather than the guidance.
    BeamPolicy(name="probe-4-attract-0.0", probe_terms=4, probe_attraction=0.0),
    BeamPolicy(name="probe-4-attract-0.5", probe_terms=4, probe_attraction=0.5),
    BeamPolicy(name="probe-4-attract-1.0", probe_terms=4, probe_attraction=1.0),
    BeamPolicy(name="probe-8-attract-0.5", probe_terms=8, probe_attraction=0.5),
)

BY_NAME: dict[str, BeamPolicy] = {policy.name: policy for policy in CATALOGUE}
