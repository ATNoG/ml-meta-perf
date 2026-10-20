"""The study's hyperparameters, read from ``config/study.json``.

**Every tuned value lives in that one file**, and nowhere in the code: the search settings all
three equations share, the rule that chooses an equation's length, the opaque regressors'
sizes, and the grid the configuration sweep explores. The dataclasses below are its schema and
carry no defaults for anything that was tuned, so a value can only come from the file (or from
a command-line override of it) and cannot drift between a constant and a chapter.

Besides the hyperparameters the file **describes the problem**: which CSV holds the corpus, which
columns name the datasets and the models, which column is the target and what range it can take,
and which columns are features of the dataset and which of the model. Those are the ``corpus``,
``response`` and ``features`` sections, and they are what makes the tool usable on a problem other
than MCC: nothing about a problem lives in the code.

The ``features`` section says which columns *are* features, not which an equation may use: E1
searches every dataset feature, E2 every model feature, E3 both, and the search decides what
each equation keeps. Restricting the pool would be a second, hidden search over feature
subsets.

`ml-meta-perf-search` (`ml_meta_perf.sweep`) proposes a replacement file; adopting it is a
reviewed change to ``config/study.json`` rather than something a run does to itself.

Study chapter: [3. Term generation and selection][study-chapter] -- the rationale, in prose.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/03-term-selection.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

#: The configuration file, relative to the repository root. Resolved like the corpus
#: (`data.resolve_path`): the source checkout this module runs from, else the working directory.
CONFIG_RELATIVE_PATH = Path("config") / "study.json"


def _default_config_path() -> Path:
    checkout = Path(__file__).resolve().parents[2] / CONFIG_RELATIVE_PATH
    working = Path.cwd() / CONFIG_RELATIVE_PATH
    return next((path for path in (checkout, working) if path.is_file()), checkout)


DEFAULT_CONFIG_PATH = _default_config_path()


@dataclass(frozen=True)
class Schema:
    """Which columns a meta-dataset's rows, groups, features and target live in.

    The runtime form of a study's data description: `StudyConfig.schema` builds it from the
    ``corpus``, ``response`` and ``features`` sections of the study file.
    """

    dataset_column: str
    model_column: str
    dataset_features: tuple[str, ...]
    model_features: tuple[str, ...]
    target_column: str
    bounds: tuple[float, float] | None
    task_type: str | None = None
    log_target: bool = False

    def __post_init__(self) -> None:
        if self.log_target and self.bounds is not None:
            raise ValueError("log1p requires bounds=None; only the unbounded FlexFL cost targets support it")

    @property
    def features(self) -> tuple[str, ...]:
        return self.dataset_features + self.model_features

    @property
    def slug(self) -> str:
        base = self.target_column if self.task_type is None else f"{self.target_column}-{self.task_type}"
        return f"{base}-log1p" if self.log_target else base

    @property
    def label(self) -> str:
        return f"log1p({self.target_column})" if self.log_target else self.target_column

    def tag(self, name: str) -> str:
        """``name`` with a ``_log1p`` suffix when this schema fits a log-transformed target."""
        return f"{name}_log1p" if self.log_target else name


@dataclass(frozen=True)
class CorpusConfig:
    """Where the corpus is and which columns name its rows' groups."""

    #: The CSV, relative to the repository root (else the working directory) unless absolute.
    path: str
    #: The column naming the dataset, the first grouping factor (leave-one-dataset-out).
    dataset_column: str
    #: The column naming the model, the second grouping factor (leave-one-model-out).
    model_column: str


@dataclass(frozen=True)
class ResponseConfig:
    """The target: its column and the range it can take."""

    column: str
    #: ``[low, high]`` when the target cannot leave a range (MCC, a bounded error); ``None`` for
    #: a cost, a time or a count, which is then never clipped (chapter 7, section 7.1).
    bounds: tuple[float, float] | None


@dataclass(frozen=True)
class FeaturesConfig:
    """The feature columns, by what they describe."""

    dataset: tuple[str, ...]
    model: tuple[str, ...]


class ConfigError(ValueError):
    """The study file is well formed JSON but does not describe a usable study."""


@dataclass(frozen=True)
class Configuration:
    """The search hyperparameters, **shared by E1, E2 and E3**.

    One configuration for all three equations is what makes the gaps between them evidence
    about the meta-data rather than about three tuning runs.
    """

    #: The largest standard score a term may reach before it is rejected as a spike.
    max_abs_zscore: float
    #: The ridge penalty on standardised terms.
    penalty: float
    #: Terms surviving the correlation screen into the beam.
    pool_size: int
    #: The search horizon: the longest equation the search explores and the range every curve
    #: covers. **Not the published length** -- `selection.plateau_knee` derives that.
    max_terms: int
    #: Beam width. Eight times wider converges to the fourth decimal, so this is a cost
    #: control rather than a tuned knob.
    beam_width: int
    #: Raw features per term: 2 allows products and ratios of two features, 3 adds
    #: ratio-of-sums ``(a + b) / c``.
    max_arity: int


@dataclass(frozen=True)
class SelectionConfig:
    """How an equation's length is chosen from its cross-validated curve.

    See `selection.plateau_knee`: the curve is the worst R2 over the four protocols, smoothed
    by a running median of ``smoothing`` lengths; the length is the first knee of its Pareto
    front after which the smoothed curve gains at most ``delta`` over the next ``window``
    lengths.
    """

    delta: float
    window: int
    smoothing: int
    #: The capability bound's grammar: E3-MAX is fitted at this arity under the same
    #: configuration and takes the raw maximum of its curve.
    capability_arity: int


@dataclass(frozen=True)
class OpaqueConfig:
    """The opaque regressors the equation is compared against (`ml_meta_perf.opaque`).

    Chosen on a grid scored under all four protocols: tuning on the held-out protocols can only
    favour the opaque side, which is the conservative direction for this study's claim.
    """

    forest_trees: int
    #: Share of the features considered at each split.
    forest_max_features: float
    boosting_stages: int
    boosting_learning_rate: float
    boosting_max_depth: int


@dataclass(frozen=True)
class SweepConfig:
    """The grid `ml_meta_perf.sweep` explores, and how it chooses among close candidates."""

    zscores: tuple[float, ...]
    penalties: tuple[float, ...]
    arities: tuple[int, ...]
    #: Only configurations whose chosen equation has at most this many terms are proposed:
    #: the study's readability limit, stated rather than implied.
    readable_terms: int
    #: Among those, candidates whose smoothed floor is within this of the best are treated as
    #: indistinguishable on R2, and the one with the best doubly-held-out ranking average
    #: precision is proposed.
    band: float


@dataclass(frozen=True)
class StudyConfig:
    """The whole of ``config/study.json``."""

    corpus: CorpusConfig
    response: ResponseConfig
    features: FeaturesConfig
    search: Configuration
    selection: SelectionConfig
    opaque: OpaqueConfig
    sweep: SweepConfig

    def schema(self) -> Schema:
        """The data description the loader and the fit consume."""
        return Schema(
            dataset_column=self.corpus.dataset_column,
            model_column=self.corpus.model_column,
            dataset_features=self.features.dataset,
            model_features=self.features.model,
            target_column=self.response.column,
            bounds=self.response.bounds,
        )


def _validated(corpus: CorpusConfig, response: ResponseConfig, features: FeaturesConfig) -> None:
    """Reject a study description that cannot describe a corpus, before any file is read."""
    if not corpus.path:
        raise ConfigError("corpus.path is empty")
    identifiers = (corpus.dataset_column, corpus.model_column)
    if not all(identifiers) or corpus.dataset_column == corpus.model_column:
        raise ConfigError(f"corpus needs two different, non-empty group columns, got {identifiers}")
    if response.bounds is not None and not response.bounds[0] < response.bounds[1]:
        raise ConfigError(f"response.bounds must satisfy low < high, got {response.bounds}")
    for kind, names in (("dataset", features.dataset), ("model", features.model)):
        if not names:
            raise ConfigError(f"features.{kind} is empty: an equation needs something to describe")
    every = features.dataset + features.model
    repeated = sorted({name for name in every if every.count(name) > 1})
    if repeated:
        raise ConfigError(f"features listed more than once (or as both dataset and model): {repeated}")
    reserved = sorted({*identifiers, response.column} & set(every))
    if reserved:
        raise ConfigError(f"a feature cannot also be a group or the response: {reserved}")
    if response.column in identifiers:
        raise ConfigError(f"the response column {response.column!r} is also a group column")


def parse_config(payload: dict[str, Any]) -> StudyConfig:
    """Build the typed configuration from the file's JSON object; unknown keys are an error."""
    sweep = payload["sweep"]
    bounds = payload["response"]["bounds"]
    corpus = CorpusConfig(**payload["corpus"])
    response = ResponseConfig(
        column=payload["response"]["column"],
        bounds=None if bounds is None else (float(bounds[0]), float(bounds[1])),
    )
    features = FeaturesConfig(
        dataset=tuple(str(name) for name in payload["features"]["dataset"]),
        model=tuple(str(name) for name in payload["features"]["model"]),
    )
    _validated(corpus, response, features)
    return StudyConfig(
        corpus=corpus,
        response=response,
        features=features,
        search=Configuration(**payload["search"]),
        selection=SelectionConfig(**payload["selection"]),
        opaque=OpaqueConfig(**payload["opaque"]),
        sweep=SweepConfig(
            zscores=tuple(float(value) for value in sweep["zscores"]),
            penalties=tuple(float(value) for value in sweep["penalties"]),
            arities=tuple(int(value) for value in sweep["arities"]),
            readable_terms=int(sweep["readable_terms"]),
            band=float(sweep["band"]),
        ),
    )


@cache
def load_config(path: str | Path | None = None) -> StudyConfig:
    """Read and validate ``config/study.json`` (or ``path``). Cached: the file is read once."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.is_file():
        raise FileNotFoundError(f"study configuration not found: {resolved} (pass --config)")
    return parse_config(json.loads(resolved.read_text(encoding="utf-8")))


def default_schema() -> Schema:
    """The data description of the study in ``config/study.json``: what a bare call analyses."""
    return load_config().schema()


def default_configuration() -> Configuration:
    """The shared search configuration from the study file -- what a bare run fits under."""
    return load_config().search
