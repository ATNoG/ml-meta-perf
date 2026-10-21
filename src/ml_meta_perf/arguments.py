"""The command-line flags every command that reads a study shares.

A study is the file ``config/mcc.json`` (or any file of the same shape); each of its sections is
also a flag, so ``--search.penalty 3`` overrides one field of the file for one run, ``--config
other.json`` replaces the file, and ``--print_config`` shows the effective study. The values live
in the file and nowhere in the code: the flags carry no default of their own.
"""

from __future__ import annotations

from jsonargparse import ActionConfigFile, ArgumentParser  # pyright: ignore[reportPrivateImportUsage]

from ml_meta_perf.config import (
    DEFAULT_CONFIG_PATH,
    Configuration,
    CorpusConfig,
    FeaturesConfig,
    GrammarConfig,
    ResponseConfig,
    SelectionConfig,
    SweepConfig,
)


def study_parser(prog: str, description: str) -> ArgumentParser:
    """A parser for a command that reads a study: the study file and every section of it as a flag."""
    parser = ArgumentParser(prog=prog, description=description, default_config_files=[str(DEFAULT_CONFIG_PATH)])
    parser.add_argument(
        "--config", action=ActionConfigFile, help=f"study file (default: {DEFAULT_CONFIG_PATH.name} in config/)"
    )
    parser.add_argument("--corpus", type=CorpusConfig, help="the corpus CSV and its group columns")
    parser.add_argument("--response", type=ResponseConfig, help="the target column, its bounds and its scale")
    parser.add_argument("--features", type=FeaturesConfig, help="the dataset and model feature columns")
    parser.add_argument("--equations", type=tuple[str, ...], help="which equations to fit: E1, E2, E3, E3-MAX")
    parser.add_argument("--protocols", type=tuple[str, ...], help="the protocols to score under: IS, LODO, LOMO, DHO")
    parser.add_argument("--grammar", type=GrammarConfig, help="operations a feature may not take part in")
    parser.add_argument("--search", type=Configuration, help="the search hyperparameters every equation shares")
    parser.add_argument("--selection", type=SelectionConfig, help="the rule choosing E3-Valid's arity and length")
    parser.add_argument("--sweep", type=SweepConfig, help="the grid `sweep` explores")
    parser.add_argument(
        "--data", type=str | None, default=None, help="corpus CSV, used as written (overrides corpus.path)"
    )
    return parser
