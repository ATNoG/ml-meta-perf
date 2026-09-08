"""Choosing a beam's starting terms to be spread out rather than merely strong.

A beam search that starts from the empty subset has exactly one first move: take the
highest-scoring single terms. On this library that is a weakness rather than a neutral choice.
The vocabulary is deliberately redundant -- `nr_attr` and `nr_outliers` correlate at 0.9995,
and every ratio has a near-twin -- so the *strongest* terms are largely near-duplicates of
each other. A width-6 beam can therefore spend all six slots on one idea at step one, and
every later step inherits that: the beam is nominally six wide and effectively one.

This module chooses those starting terms to cover term space instead. Two strategies, both
restricted to the screened pool, so what is being spread is a set of terms already known to
carry signal -- diversity *within* a good region, not diversity instead of it.

**`maximin`** is the classical greedy farthest-point rule: start from the strongest term, then
repeatedly take whichever pool term is furthest from everything chosen. Deterministic, no
parameters, and the obvious baseline any fancier method has to beat.

**`empty_space`** uses the Empty Space Algorithm (`ess.esa`) to *place* points in the sparse
regions of the embedded pool and snaps each to the nearest unused term. The difference from
maximin is that ESA relaxes all points together under a repulsive force rather than fixing
them one at a time, which avoids the greedy failure where an early pick settles into an
arrangement optimal for itself and the rest squeeze into its gaps.

The embedding is the part to be sceptical about, so it is stated plainly. Each term is a
column of the standardised design, and two terms are similar when their columns are
correlated -- that is the similarity that matters, because two correlated columns cannot both
earn a place in one equation. The columns are projected onto their leading principal
components and rank-mapped to the unit cube, which is the domain ESA and TORANN work on. Rank
rather than min-max: a single extreme term would otherwise compress every other term into a
corner, and the pool contains exactly that kind of term by construction.

**`guided_space` is the same algorithm used the way ESS is meant to be used.** One `ess.esa`
call with no scores is a *design*, not a search: it places points once, against anchors that
are only the pool's own geometry, and then the search never tells it anything. Measured that
way, `empty_space` seeding lost -- worse equations and 1.5-2.8x slower -- and the negative was
recorded as "space-filling initialisation does not pay", which was the wrong conclusion from
the right measurement. What was tested was a one-shot sampler.

ESS is a *continuous* sampler. `ess.esa` takes an `attractiveness` value per anchor and a
`attraction_weight`, and relaxes new points under repulsion from the anchors **and** attraction
towards the regions that scored well; the working implementations in `pyBlindOpt.init.oblesa`
and the Optuna `ESSSampler` both run it in rounds, feeding each round's *measured* probes back
in as anchors so the next field fit sees them at the same standing as the sampler's own points.
`ProbeField` is that loop, with the beam supplying the measurements:

* anchors are the terms whose objective the search has actually computed;
* attractiveness is the negated objective, because `Subset.rss` is lower-is-better and ESS's
  contract is higher-is-more-attractive -- ESS is not told which way the caller optimises and
  cannot guess;
* `tell` folds each round's results back in, so the field sharpens as the beam progresses
  instead of being fixed before it starts.

**None of this touches the published equation.** `beam.VANILLA` seeds nothing and probes
nothing, and every path here is reached only through a policy that asks for it.
"""

from __future__ import annotations

import numpy as np

#: How many dimensions the term embedding uses. Four is enough to separate the library's
#: redundancy groups and low enough that a space-filling design is meaningful: empty space is
#: a weaker notion in every dimension added, and at the library sizes here (45 to 838 terms) a
#: much higher dimension would make every term equidistant from every other.
EMBEDDING_DIMENSIONS = 4

#: ESA relaxation length. The package default is 1024; the placements here converge far
#: earlier because the point counts are small (a handful of seeds against a few hundred pool
#: terms), and the sweep runs this thousands of times.
ESA_EPOCHS = 256


def embed_terms(design: np.ndarray, pool: list[int], dimensions: int = EMBEDDING_DIMENSIONS) -> np.ndarray:
    """Pool terms as points in the unit cube, close when their columns are correlated.

    ``design`` is the standardised design, so a column's inner product with another is its
    correlation up to the row count, and the leading principal components of the pool's
    columns are the directions along which the library actually varies.

    Returned coordinates are **ranks** rescaled to ``[0, 1)``, not min-maxed values. The pool
    reliably contains a term or two whose column is dominated by one extreme row -- that is
    what `terms.MAX_ABS_ZSCORE` bounds rather than forbids -- and under min-max those terms
    push everything else into a corner, leaving a "space-filling" design that fills the space
    between two outliers.
    """
    columns = design[:, pool]
    centred = columns - columns.mean(axis=0)
    # Right singular vectors scaled by their singular values: the pool's coordinates in the
    # space its own variation spans. `full_matrices=False` keeps this a thin decomposition,
    # which at 476 rows by a few hundred terms is milliseconds.
    _, values, vectors = np.linalg.svd(centred, full_matrices=False)
    taken = min(dimensions, vectors.shape[0])
    coordinates = (vectors[:taken].T * values[:taken])[:, :taken]

    ranked = np.empty_like(coordinates)
    for axis in range(coordinates.shape[1]):
        order = np.argsort(coordinates[:, axis], kind="stable")
        ranks = np.empty(order.shape[0], dtype=np.float64)
        ranks[order] = np.arange(order.shape[0], dtype=np.float64)
        ranked[:, axis] = ranks / max(order.shape[0], 1)
    return ranked


def _toroidal_distance(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """L1 distance on the unit torus from every point to every reference point.

    Toroidal because that is the geometry ESA places into and TORANN indexes: on a periodic
    domain there is no boundary, so a space-filling design is not pushed against walls. Mixing
    a toroidal placement with a Euclidean snap would put each seed at the nearest term under a
    different metric from the one that chose where to look.
    """
    delta = np.abs(points[:, None, :] - reference[None, :, :])
    return np.minimum(delta, 1.0 - delta).sum(axis=2)


def maximin(embedded: np.ndarray, strength: np.ndarray, count: int) -> list[int]:
    """Greedy farthest-point selection over the embedded pool, started at the strongest term.

    The classical space-filling heuristic and the baseline for `empty_space`. Deterministic
    and parameter-free: ties are broken by ascending pool position, so the result is a
    property of the library rather than of the order a caller assembled it in -- the same
    property `fit._descending` exists to preserve.
    """
    if count <= 0 or embedded.shape[0] == 0:
        return []
    chosen = [int(np.argmax(strength))]
    while len(chosen) < min(count, embedded.shape[0]):
        distances = _toroidal_distance(embedded, embedded[chosen]).min(axis=1)
        distances[chosen] = -np.inf
        chosen.append(int(np.argmax(distances)))
    return chosen


def empty_space(embedded: np.ndarray, strength: np.ndarray, count: int, *, seed: int = 0) -> list[int]:
    """ESA-placed seeds, snapped to the nearest unused pool term.

    The first seed is the strongest term -- there is no argument for starting a beam anywhere
    else, and it makes the comparison with `maximin` a comparison of what the *rest* of the
    seeds do. `ess.esa` then places the remaining points in the sparse regions of the pool's
    own embedding, relaxing them together rather than fixing them one at a time, and each is
    snapped to the closest term not already taken.

    Snapping is what makes this usable: ESA returns coordinates, and a beam needs terms. A
    placed point that lands in genuinely empty space snaps to whichever real term is nearest
    to that emptiness, which is the term the greedy rule would have reached only after several
    rounds, if at all.

    Falls back to `maximin` if ESA is unavailable, so a missing optional dependency degrades to
    the classical rule rather than to an error in the middle of a sweep.
    """
    if count <= 0 or embedded.shape[0] == 0:
        return []
    limit = min(count, embedded.shape[0])
    chosen = [int(np.argmax(strength))]
    if limit == 1:
        return chosen

    try:
        import ess
    except ImportError:
        return maximin(embedded, strength, count)

    bounds = np.array([[0.0, 1.0]] * embedded.shape[1])
    placed = ess.esa(embedded, bounds, n=limit - 1, epochs=ESA_EPOCHS, seed=seed)

    for point in np.atleast_2d(placed):
        distances = _toroidal_distance(point[None, :], embedded)[0]
        distances[chosen] = np.inf
        nearest = int(np.argmin(distances))
        if np.isfinite(distances[nearest]):
            chosen.append(nearest)
    return chosen


#: Attraction law for the guided field. `cauchy` rather than ESS's default of "same as the
#: repulsion metric": a gaussian attraction dies off over the same short range the repulsion
#: acts on, so it only weakens the push instead of pulling a point across the space. This is
#: the choice `pyBlindOpt.init._ess_engine` documents and it is copied deliberately.
ATTRACTION_METRIC = "cauchy"

#: Neighbours the attractiveness of an unmeasured position is averaged over. Small, because
#: the anchor count here starts at a handful of screened terms.
ATTRACTION_NEIGHBOURS = 8


class ProbeField:
    """ESS run continuously: anchors accumulate and carry the objectives the search measured.

    The unit is a **pool position**, not a coordinate, because a beam needs terms. `propose`
    places points in the embedded pool under repulsion from everything already measured and
    attraction towards what measured well, then snaps each to the nearest term not already
    taken. `tell` folds the results back in, so the round after next is fitted against them.

    Deliberately not a `dataclass`: it is mutable state threaded through a search, and the
    frozen dataclasses elsewhere in this package are values.

    Degrades rather than fails. Without `ess` installed, or before anything has been measured,
    `propose` falls back to the maximin rule over the same embedding -- so a policy that asks
    for probes still runs, and what it loses is the guidance rather than the search.
    """

    def __init__(
        self,
        embedded: np.ndarray,
        strength: np.ndarray,
        *,
        attraction_weight: float = 0.5,
        seed: int = 0,
        epochs: int = ESA_EPOCHS,
    ) -> None:
        self.embedded = embedded
        #: The screening score per pool position. Used only by the cold-start fallback, to
        #: pick a first anchor that is at least strong -- the guided path never reads it,
        #: because once anything has been measured the objectives are better evidence.
        self.strength = strength
        self.attraction_weight = attraction_weight
        self.seed = seed
        self.epochs = epochs
        self.bounds = np.array([[0.0, 1.0]] * embedded.shape[1]) if embedded.size else np.zeros((0, 2))
        #: The best objective seen for each pool position. Lower is better throughout this
        #: package, and the sign is flipped only at the ESS boundary.
        #:
        #: **One anchor per term, not one per measurement.** A beam tries the same term against
        #: many parents -- on the published configuration that is four thousand measurements
        #: over a pool of six hundred -- and feeding all of them in would pile seven copies of
        #: one coordinate into the anchor set. ESS reads anchors as occupied *space*, so
        #: duplicates are not extra evidence: they are extra repulsion, and a term the beam
        #: keeps trying because it is useful would end up pushing probes away hardest. Keeping
        #: the best objective per position also makes the anchor set bounded by the pool, which
        #: is what stops the field's cost growing with the length of the search.
        self.best: dict[int, float] = {}

    @property
    def measured(self) -> list[int]:
        """Pool positions with a measurement, in insertion order."""
        return list(self.best)

    @property
    def objectives(self) -> list[float]:
        """Their best objectives, aligned with `measured`."""
        return list(self.best.values())

    def tell(self, positions: list[int], objectives: list[float]) -> None:
        """Record what the search learned about these pool positions. Lower objective is better."""
        for position, objective in zip(positions, objectives, strict=True):
            if not np.isfinite(objective):
                continue
            key = int(position)
            current = self.best.get(key)
            if current is None or objective < current:
                self.best[key] = float(objective)

    def _attractiveness(self) -> np.ndarray:
        """Anchor scores on ESS's scale: higher is more attractive, and bounded.

        Negated because the caller minimises, then scaled to unit range. Unscaled objectives
        would make `attraction_weight` mean something different at every step, since `rss`
        shrinks as the equation grows -- the weight has to be a property of the policy, not of
        how far through the search it happens to be.
        """
        values = -np.asarray(self.objectives, dtype=float)
        spread = float(values.max() - values.min()) if values.size else 0.0
        return (values - values.min()) / spread if spread > 0.0 else np.zeros_like(values)

    def propose(self, count: int, taken: list[int]) -> list[int]:
        """`count` pool positions to try next, avoiding everything in ``taken``."""
        if count <= 0 or self.embedded.shape[0] == 0:
            return []
        anchors = self.embedded[self.measured] if self.measured else None
        if anchors is None or anchors.shape[0] < 2:
            return self._fallback(count, taken)
        try:
            import ess
        except ImportError:
            return self._fallback(count, taken)

        placed = ess.esa(
            anchors,
            self.bounds,
            n=count,
            epochs=self.epochs,
            seed=self.seed,
            attractiveness=self._attractiveness(),
            attraction_weight=self.attraction_weight,
            attraction_metric=ATTRACTION_METRIC,
            attraction_kwargs={"power": 1.0},
            k_att=min(ATTRACTION_NEIGHBOURS, anchors.shape[0]),
        )
        return self._snap(np.atleast_2d(placed), taken)

    def _snap(self, points: np.ndarray, taken: list[int]) -> list[int]:
        """Each placed coordinate to the nearest pool term nothing has claimed.

        Toroidal, matching the geometry ESA placed into. A Euclidean snap after a toroidal
        placement puts every probe at the nearest term under a different metric from the one
        that chose where to look.
        """
        claimed = list(taken)
        chosen: list[int] = []
        for point in points:
            distances = _toroidal_distance(point[None, :], self.embedded)[0]
            distances[claimed] = np.inf
            nearest = int(np.argmin(distances))
            if np.isfinite(distances[nearest]):
                chosen.append(nearest)
                claimed.append(nearest)
        return chosen

    def _fallback(self, count: int, taken: list[int]) -> list[int]:
        """Maximin over the unclaimed pool: the classical rule, when ESS cannot be asked."""
        masked = self.strength.copy()
        masked[taken] = -np.inf
        if not np.isfinite(masked).any():
            return []
        chosen = [int(np.argmax(masked))]
        while len(chosen) < min(count, self.embedded.shape[0] - len(taken)):
            distances = _toroidal_distance(self.embedded, self.embedded[chosen + list(taken)]).min(axis=1)
            distances[chosen] = -np.inf
            distances[list(taken)] = -np.inf
            if not np.isfinite(distances).any():
                break
            chosen.append(int(np.argmax(distances)))
        return chosen


#: The seeding rules the sweep compares, by the name a policy names them with.
STRATEGIES = {"maximin": maximin, "empty_space": empty_space}


def seed_positions(
    design: np.ndarray,
    pool: list[int],
    strength: np.ndarray,
    count: int,
    *,
    strategy: str = "empty_space",
    seed: int = 0,
) -> list[int]:
    """Library indices for the beam's starting terms, spread across term space.

    ``strength`` is the screening score `fit.guided_screen` already computed for the pool, so
    seeding costs one SVD and one relaxation rather than a second pass over the library.
    Returns **library** indices, not pool positions, because that is what the beam takes.
    """
    if count <= 0 or not pool:
        return []
    embedded = embed_terms(design, pool)
    rule = STRATEGIES.get(strategy, empty_space)
    positions = rule(embedded, strength, count) if strategy != "empty_space" else empty_space(
        embedded, strength, count, seed=seed
    )
    return [pool[position] for position in positions]
