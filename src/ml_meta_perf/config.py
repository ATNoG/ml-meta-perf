"""The study's hyperparameters, read from ``config/mcc.json``.

**Every tuned value lives in that one file**, and nowhere in the code: the search settings all
equations share, the rule that chooses an equation's length, and the grid the configuration
sweep explores. The dataclasses below are its schema and
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
reviewed change to ``config/mcc.json`` rather than something a run does to itself.

Study chapter: [3. Term generation and selection][study-chapter] -- the rationale, in prose.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/03-term-selection.md
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

#: The configuration file, relative to the repository root. Resolved like the corpus
#: (`data.resolve_path`): the source checkout this module runs from, else the working directory.
CONFIG_RELATIVE_PATH = Path("config") / "mcc.json"


def _default_config_path() -> Path:
    """Resolve ``config/mcc.json`` from the repository checkout or the working directory."""
    checkout = Path(__file__).resolve().parents[2] / CONFIG_RELATIVE_PATH
    working = Path.cwd() / CONFIG_RELATIVE_PATH
    return next((path for path in (checkout, working) if path.is_file()), checkout)


DEFAULT_CONFIG_PATH = _default_config_path()


#: ``(low, high)``, either end ``None`` when that side is open: MCC is ``(-1, 1)``, a time is ``(0, None)``.
Bounds = tuple[float | None, float | None]

#: The transforms a target can be fitted on. ``log`` fits ``log(y)`` for a strictly positive target and
#: ``log1p`` fits ``log(1 + y)`` for a non-negative one (zeros allowed): a cost spanning orders
#: of magnitude becomes a relative error, and the equation's terms multiply (chapter 7).
TRANSFORMS = ("none", "log1p", "log")

#: The equations a study can fit: E1 the dataset features, E2 the model features, E3 both.
EQUATIONS = ("E1", "E2", "E3")

#: E3-MAX, the best score the configuration can reach: a variant of E3, fitted beside it by a study of one equation.
E3_MAX = "E3-MAX"

#: The equation lists a study of ONE all-features equation may ask for: E3-Valid alone, or with E3-MAX.
SINGLE_EQUATIONS = (("E3",), ("E3", E3_MAX))

#: Every equation list a study file may carry: one equation (E3-Valid, with E3-MAX or not), or the three of
#: the dataset/model comparison (E1 dataset features, E2 model features, E3 both), with E3-MAX or not.
ALLOWED_EQUATIONS = (*SINGLE_EQUATIONS, EQUATIONS, (*EQUATIONS, E3_MAX))

#: The rules choosing an equation's arity and length (`lengths.LENGTH_RULES`), repeated so the file is checked on read.
LENGTH_METHODS = (
    "plateau_knee",
    "pareto_knee",
    "pareto_elbow",
    "pareto_within_delta",
    "plateau_pareto",
    "l_method",
    "bend_angle",
    "log_slope",
    "one_se",
    "projection",
    "mcs",
    "stable_within_delta",
    "aic",
    "aicc",
    "bic",
    "ebic",
    "mdl",
    "auto",
)

#: What a study may forbid for a feature: `terms.EXCLUDABLE`, repeated so the file is checked before any term is built.
EXCLUDABLE = ("log", "sqrt", "inv", "sq")

#: The protocols an equation can be scored under: in-sample, then leave-one-dataset-out, leave-one-model-out
#: and doubly held-out. The last two hold out a model, so they need a model group (chapter 7, section 7.2).
PROTOCOLS = ("IS", "LODO", "LOMO", "DHO")

#: The protocols that need a model group: a corpus without one cannot hold a model out.
MODEL_PROTOCOLS = ("LOMO", "DHO")


def _transform_bounds_problem(transform: str, low: float | None) -> str:
    """Why a target bounded below by ``low`` cannot be fitted on ``transform``'s scale, or an empty string."""
    if transform == "log1p" and (low is None or low < 0.0):
        return "log1p needs a target that cannot be negative: bounds=None, or a low bound of 0 or more"
    if transform == "log" and (low is None or low <= 0.0):
        return "log needs a target that is always positive: bounds=None, or a low bound above 0"
    return ""


@dataclass(frozen=True)
class RowFilter:
    """Keep the rows whose ``column`` equals ``equals``: one task type of a mixed corpus."""

    column: str
    equals: bool | int | float | str
    #: Names the subset in output folders (``classification``); optional.
    label: str | None = None


@dataclass(frozen=True)
class RequiredColumn:
    """A column the corpus must have, with what to do when it is missing (a stale corpus)."""

    column: str
    hint: str = ""


@dataclass(frozen=True)
class Schema:
    """Which columns a meta-dataset's rows, groups, features and target live in, and how to read it.

    The runtime form of a study's data description: `StudyConfig.schema` builds it from the
    ``corpus``, ``response`` and ``features`` sections of the study file.
    """

    dataset_column: str
    #: ``None`` when the corpus has one grouping factor: no model can be held out.
    model_column: str | None
    dataset_features: tuple[str, ...]
    model_features: tuple[str, ...]
    target_column: str
    bounds: Bounds | None
    transform: str = "none"
    row_filter: RowFilter | None = None
    required: tuple[RequiredColumn, ...] = ()
    single_valued: tuple[str, ...] = ()
    drop_missing_response: bool = False
    drop_constant_features: bool = False
    #: The protocols the equations are scored under; `selection.floor_curve` judges a length on the worst.
    protocols: tuple[str, ...] = PROTOCOLS
    #: ``(feature, operations)`` pairs the term grammar may not use, sorted by feature (`GrammarConfig`).
    exclusions: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}; expected one of {TRANSFORMS}")
        if self.transform != "none" and self.bounds is not None:
            problem = _transform_bounds_problem(self.transform, self.bounds[0])
            if problem:
                raise ValueError(problem)

    @property
    def excluded(self) -> dict[str, tuple[str, ...]]:
        """The grammar exclusions as the map `terms.build_library` takes."""
        return dict(self.exclusions)

    @property
    def group_columns(self) -> tuple[str, ...]:
        """The columns naming the groups a row belongs to: the dataset, then the model when there is one."""
        return (self.dataset_column,) if self.model_column is None else (self.dataset_column, self.model_column)

    @property
    def model_group(self) -> str:
        """The model column, for code that cannot run without one (the model baselines, the model protocols)."""
        if self.model_column is None:
            raise ValueError("this study has no model column: nothing here can hold a model out")
        return self.model_column

    @property
    def features(self) -> tuple[str, ...]:
        """All feature columns $F_d \\cup F_m$, dataset features followed by model features."""
        return self.dataset_features + self.model_features

    @property
    def fit_bounds(self) -> Bounds | None:
        """The bounds on the fitted scale: $g(B) = [g(a), g(b)]$ for $g \\in \\{\\log(1+\\cdot), \\log\\}$."""
        if self.bounds is None or self.transform == "none":
            return self.bounds
        low, high = self.bounds
        forward = math.log1p if self.transform == "log1p" else math.log
        return (None if low is None else forward(low), None if high is None else forward(high))

    @property
    def log_target(self) -> bool:
        """Whether the equation is fitted on a logarithm of the target (``log1p`` or ``log``)."""
        return self.transform != "none"

    @property
    def slug(self) -> str:
        """Filesystem-safe folder name ``<target>[-<row_filter.label>][-<transform>]`` for this study."""
        parts = [self.target_column]
        if self.row_filter is not None and self.row_filter.label:
            parts.append(self.row_filter.label)
        if self.log_target:
            parts.append(self.transform)
        return "-".join(parts)

    @property
    def label(self) -> str:
        """Left-hand side of the equation: ``transform(target)`` when log-transformed, else ``target``."""
        return f"{self.transform}({self.target_column})" if self.log_target else self.target_column

    def tag(self, name: str) -> str:
        """``name`` with the transform as a suffix when this schema fits a log-transformed target."""
        return f"{name}_{self.transform}" if self.log_target else name


@dataclass(frozen=True)
class CorpusConfig:
    """Where the corpus is and which columns name its rows' groups."""

    #: The CSV, relative to the repository root (else the working directory) unless absolute.
    path: str
    #: The column naming the dataset, the first grouping factor (leave-one-dataset-out).
    dataset_column: str
    #: The column naming the model, the second grouping factor (leave-one-model-out), or ``None``
    #: when the corpus has one grouping factor only.
    model_column: str | None
    #: Keep only the rows where a column has a value: one task type of a mixed corpus.
    row_filter: RowFilter | None = None
    #: Columns the corpus must have; a missing one is reported with its hint (a stale corpus).
    require: tuple[RequiredColumn, ...] = ()
    #: Columns that must hold a single value over the rows fitted (one early-stopping rule).
    single_valued: tuple[str, ...] = ()
    #: Drop (and report) rows with an empty response instead of rejecting the corpus.
    drop_missing_response: bool = False
    #: Remove the features that take one value in the rows fitted before the library is built.
    drop_constant_features: bool = False


@dataclass(frozen=True)
class ResponseConfig:
    """The target: its column and the range it can take."""

    column: str
    #: ``[low, high]`` when the target cannot leave a range (MCC, a bounded error); ``None`` for
    #: a cost, a time or a count, which is then never clipped (chapter 7, section 7.1).
    bounds: Bounds | None
    #: ``none``, ``log1p`` to fit ``log(1 + y)`` (non-negative target) or ``log`` to fit ``log(y)`` (positive target).
    transform: str = "none"


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
    #: The rule choosing E3-Valid's arity and length, from `LENGTH_METHODS`. ``plateau_knee`` fits the
    #: configured arity only; every other rule fits each arity up to it and chooses among them, and
    #: ``auto`` runs them all (chapter 7, 7.6).
    method: str
    #: The capability bound's grammar: E3-MAX is fitted at this arity under the same
    #: configuration and takes the raw maximum of its curve.
    capability_arity: int
    #: The conventional constants of the rules that have one. They are not tuned to a corpus.
    #: Confidence level of the model confidence set (``mcs``).
    alpha: float = 0.1
    #: Standard errors ``one_se`` and ``projection`` allow beyond the best candidate.
    se_multiple: float = 1.0
    #: Bootstrap draws of ``mcs``, and the seed that fixes them.
    bootstrap_rounds: int = 500
    seed: int = 0
    #: Share of held-out datasets under which every term keeps its sign (``stable_within_delta``).
    stability_min: float = 0.9
    #: ``auto``: picks whose smoothed floor is within this of the highest count as equally close to E3-MAX.
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in LENGTH_METHODS:
            raise ConfigError(f"selection.method must be one of {LENGTH_METHODS}, got {self.method!r}")
        problems = {
            "alpha must lie in (0, 1)": not 0.0 < self.alpha < 1.0,
            "se_multiple must be positive": self.se_multiple <= 0.0,
            "bootstrap_rounds must be at least 50": self.bootstrap_rounds < 50,
            "stability_min must lie in [0, 1]": not 0.0 <= self.stability_min <= 1.0,
            "tolerance must be non-negative": self.tolerance < 0.0,
        }
        for message, failed in problems.items():
            if failed:
                raise ConfigError(f"selection.{message}")


@dataclass(frozen=True)
class GrammarConfig:
    """Which operations a feature may not take part in, so the search never builds an unacceptable term."""

    #: Feature -> operations from ``EXCLUDABLE``: ``log``, ``sqrt``, ``sq`` remove that transform of the
    #: feature everywhere, ``inv`` removes it from every denominator. A feature that can reach zero in
    #: future data should not be inverted or logged even if this corpus never shows it.
    exclude: dict[str, tuple[str, ...]]


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
    """The whole of ``config/mcc.json``."""

    corpus: CorpusConfig
    response: ResponseConfig
    features: FeaturesConfig
    #: Which equations to fit: E1 (dataset features), E2 (model features), E3 (both).
    equations: tuple[str, ...]
    #: The protocols every equation is scored under, drawn from `PROTOCOLS`; a length is judged on the worst.
    protocols: tuple[str, ...]
    grammar: GrammarConfig
    search: Configuration
    selection: SelectionConfig
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
            transform=self.response.transform,
            row_filter=self.corpus.row_filter,
            required=self.corpus.require,
            single_valued=self.corpus.single_valued,
            drop_missing_response=self.corpus.drop_missing_response,
            drop_constant_features=self.corpus.drop_constant_features,
            protocols=self.protocols,
            exclusions=tuple(sorted((feature, tuple(ops)) for feature, ops in self.grammar.exclude.items())),
        )


def _validated(
    corpus: CorpusConfig,
    response: ResponseConfig,
    features: FeaturesConfig,
    equations: tuple[str, ...],
    protocols: tuple[str, ...],
    grammar: GrammarConfig,
) -> None:
    """Reject a study description that cannot describe a corpus, before any file is read."""
    if not corpus.path:
        raise ConfigError("corpus.path is empty")
    if equations not in ALLOWED_EQUATIONS:
        raise ConfigError(
            f"equations must be one of {[list(names) for names in ALLOWED_EQUATIONS]}, got {list(equations)}"
        )
    if not corpus.dataset_column or corpus.model_column == "" or corpus.dataset_column == corpus.model_column:
        raise ConfigError(
            "corpus needs a non-empty dataset column and a model column that differs from it "
            f"(or null), got {(corpus.dataset_column, corpus.model_column)}"
        )
    identifiers = tuple(name for name in (corpus.dataset_column, corpus.model_column) if name)
    _validated_protocols(protocols, has_model_group=corpus.model_column is not None)
    known = set(features.dataset) | set(features.model)
    for feature, operations in grammar.exclude.items():
        unknown = sorted(set(operations) - set(EXCLUDABLE))
        if feature not in known or unknown or not operations or len(set(operations)) != len(operations):
            raise ConfigError(
                f"grammar.exclude[{feature!r}] must name a listed feature and distinct operations drawn from "
                f"{EXCLUDABLE}, got {list(operations)}"
            )
    low, high = response.bounds or (None, None)
    if low is not None and high is not None and not low < high:
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
    if response.transform not in TRANSFORMS:
        raise ConfigError(f"response.transform must be one of {TRANSFORMS}, got {response.transform!r}")
    if response.transform != "none" and response.bounds is not None:
        problem = _transform_bounds_problem(response.transform, low)
        if problem:
            raise ConfigError(f"response.transform {response.transform}: {problem}")
    if corpus.row_filter is not None and not corpus.row_filter.column:
        raise ConfigError("corpus.row_filter.column is empty")
    for requirement in corpus.require:
        if not requirement.column:
            raise ConfigError("corpus.require holds an empty column name")
    for rule in corpus.single_valued:
        if not rule:
            raise ConfigError("corpus.single_valued holds an empty column name")


def _validated_protocols(protocols: tuple[str, ...], *, has_model_group: bool) -> None:
    """The protocols must be known, distinct, include the fit, hold something out, and have their groups."""
    if any(name not in PROTOCOLS for name in protocols) or len(set(protocols)) != len(protocols):
        raise ConfigError(f"protocols must be distinct names drawn from {PROTOCOLS}, got {protocols}")
    if "IS" not in protocols or len(protocols) < 2:
        raise ConfigError(f"protocols must include IS and at least one held-out protocol, got {protocols}")
    needing = [name for name in protocols if name in MODEL_PROTOCOLS]
    if needing and not has_model_group:
        raise ConfigError(f"protocols {needing} hold a model out, but corpus.model_column is null")


def _bounds(raw: Any) -> Bounds | None:
    """``[low, high]`` from the file; an end may be null (open), and both open is no bounds at all."""
    if raw is None:
        return None
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise ConfigError(f"response.bounds must be null or a [low, high] pair, got {raw!r}")
    low, high = (None if end is None else float(end) for end in raw)
    return None if low is None and high is None else (low, high)


def _corpus(payload: dict[str, Any]) -> CorpusConfig:
    """The ``corpus`` section, with its nested filter and requirements; unknown keys are an error."""
    options = dict(payload)
    row_filter = options.pop("row_filter", None)
    require = options.pop("require", ())
    single_valued = options.pop("single_valued", ())
    return CorpusConfig(
        row_filter=None if row_filter is None else RowFilter(**row_filter),
        require=tuple(RequiredColumn(**item) for item in require),
        single_valued=tuple(str(name) for name in single_valued),
        **options,
    )


def parse_config(payload: dict[str, Any]) -> StudyConfig:
    """Build the typed configuration from the file's JSON object; unknown keys are an error."""
    sweep = payload["sweep"]
    bounds = _bounds(payload["response"]["bounds"])
    corpus = _corpus(payload["corpus"])
    response = ResponseConfig(
        column=payload["response"]["column"],
        bounds=bounds,
        **{key: value for key, value in payload["response"].items() if key == "transform"},
    )
    unknown = set(payload["response"]) - {"column", "bounds", "transform"}
    if unknown:
        raise TypeError(f"unexpected response keys: {sorted(unknown)}")
    features = FeaturesConfig(
        dataset=tuple(str(name) for name in payload["features"]["dataset"]),
        model=tuple(str(name) for name in payload["features"]["model"]),
    )
    equations = tuple(str(name) for name in payload["equations"])
    protocols = tuple(str(name) for name in payload["protocols"])
    grammar = GrammarConfig(
        exclude={str(k): tuple(str(o) for o in v) for k, v in payload["grammar"]["exclude"].items()}
    )
    _validated(corpus, response, features, equations, protocols, grammar)
    return StudyConfig(
        corpus=corpus,
        response=response,
        features=features,
        equations=equations,
        protocols=protocols,
        grammar=grammar,
        search=Configuration(**payload["search"]),
        selection=SelectionConfig(**payload["selection"]),
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
    """Read and validate ``config/mcc.json`` (or ``path``). Cached: the file is read once."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.is_file():
        raise FileNotFoundError(f"study configuration not found: {resolved} (pass --config)")
    return parse_config(json.loads(resolved.read_text(encoding="utf-8")))


def study_of(arguments: Any) -> StudyConfig:
    """The study a command line describes: the study file's sections, each possibly overridden by a flag.

    A flag can make the description unusable (a log scale on MCC, say), and a file's own checks
    ran only on the file; they run again here, so the command line gets the same `ConfigError`.
    """
    equations, protocols = tuple(arguments.equations), tuple(arguments.protocols)
    _validated(
        arguments.corpus,
        arguments.response,
        arguments.features,
        equations,
        protocols,
        arguments.grammar,
    )
    return StudyConfig(
        corpus=arguments.corpus,
        response=arguments.response,
        features=arguments.features,
        equations=equations,
        protocols=protocols,
        grammar=arguments.grammar,
        search=arguments.search,
        selection=arguments.selection,
        sweep=arguments.sweep,
    )


def default_schema() -> Schema:
    """The data description of the study in ``config/mcc.json``: what a bare call analyses."""
    return load_config().schema()


def default_configuration() -> Configuration:
    """The shared search configuration from the study file -- what a bare run fits under."""
    return load_config().search
