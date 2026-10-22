"""Where a study's results go, and the report generated from them.

One study writes one folder, ``<output>/<csv stem>/<target>[-tags]/``: the CSV's name first, so
two corpora never share a folder, then the target with its row filter and scale. In it:

* ``<variant>/equation.json``, ``equation.txt``, ``curve.csv`` for each equation fitted (``e1``,
  ``e2``, ``e3`` for E3-Valid, ``e3-max``);
* ``scores.csv`` (every variant, protocol and scale), ``term_effects.csv`` and ``group_shares.csv``;
* ``report.md``, generated from those numbers by arithmetic only (nothing in it is narrated);
* ``manifest.json``: the corpus (absolute path, SHA-256, rows), the study (SHA-256 of its effective
  description) and the package version, so a folder says exactly what produced it.

A folder whose manifest names a different corpus file content is not overwritten without
``--force``: a result must not silently change meaning.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import polars as pl

from ml_meta_perf import __version__
from ml_meta_perf.config import E3_MAX, StudyConfig
from ml_meta_perf.experiment import EquationReport
from ml_meta_perf.study import VARIANT_NAMES, StudyReport

#: The folder a variant is written to.
VARIANT_FOLDERS = {"E1": "e1", "E2": "e2", "E3": "e3", E3_MAX: "e3-max"}


class OverwriteError(RuntimeError):
    """The destination already holds the result of a different corpus."""


def file_sha256(path: Path) -> str:
    """Compute the hexadecimal SHA-256 digest of ``path`` in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def study_sha256(study: StudyConfig) -> str:
    """The SHA-256 of the study's effective description, canonically serialised (sorted keys)."""
    text = json.dumps(dataclasses.asdict(study), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def study_directory(root: str | Path, corpus: Path, report: StudyReport) -> Path:
    """Result folder path ``<root>/<csv stem>/<target>[-tags]``."""
    return Path(root) / corpus.stem / report.schema.slug


def manifest_of(report: StudyReport, study: StudyConfig, corpus: Path) -> dict[str, object]:
    """Build the provenance manifest dictionary for ``report`` and ``study`` on ``corpus``."""
    return {
        "version": __version__,
        "corpus": {"path": str(corpus.resolve()), "sha256": file_sha256(corpus), "rows": report.facts["rows"]},
        "study_sha256": study_sha256(study),
        "target": report.schema.label,
        "equations": list(study.equations),
        "protocols": list(study.protocols),
        "selection_method": study.selection.method,
    }


def check_overwrite(destination: Path, manifest: dict[str, object], *, force: bool) -> None:
    """Refuse to overwrite the result of a different corpus content unless ``force``."""
    existing = destination / "manifest.json"
    if force or not existing.is_file():
        return
    previous = json.loads(existing.read_text(encoding="utf-8"))
    before, now = previous["corpus"]["sha256"], manifest["corpus"]["sha256"]  # pyright: ignore[reportIndexIssue]
    if before != now:
        raise OverwriteError(
            f"{destination} holds the result of a different corpus (sha256 {before[:12]}, now {now[:12]}); "
            "pass --force to replace it"
        )


def _equation_files(destination: Path, report: EquationReport) -> list[Path]:
    """Write ``equation.json``, ``equation.txt``, ``curve.csv`` and optional ``rules.csv`` into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    report.equation.save(destination / "equation.json")
    (destination / "equation.txt").write_text(f"{report.equation}\n", encoding="utf-8")
    report.curve.write_csv(destination / "curve.csv")
    files = [destination / name for name in ("equation.json", "equation.txt", "curve.csv")]
    (destination / "rules.csv").unlink(missing_ok=True)  # never leave the table of an earlier method
    if report.rules is not None:
        report.rules.write_csv(destination / "rules.csv")
        files.append(destination / "rules.csv")
    return files


def write_study(
    report: StudyReport, study: StudyConfig, corpus: Path, destination: Path, *, force: bool = False
) -> list[Path]:
    """Write every output of a study into ``destination`` and return the files written."""
    manifest = manifest_of(report, study, corpus)
    check_overwrite(destination, manifest, force=force)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, equation in report.equations.items():
        written += _equation_files(destination / VARIANT_FOLDERS[name], equation)
    for name, frame in (
        ("scores.csv", report.scores),
        ("term_effects.csv", report.effects),
        ("group_shares.csv", report.shares),
    ):
        if not frame.is_empty():
            frame.write_csv(destination / name)
            written.append(destination / name)
    (destination / "report.md").write_text(render_report(report, manifest), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [*written, destination / "report.md", destination / "manifest.json"]


def _cell(value: object) -> str:
    """Format a table cell: floats to 4 significant digits, ``NaN`` and ``None`` as empty strings."""
    if isinstance(value, float):
        return "" if value != value else f"{value:.4g}"
    return "" if value is None else str(value)


def markdown_table(frame: pl.DataFrame) -> str:
    """A GitHub-flavoured table of ``frame``, floats to four significant digits."""
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(_cell(value) for value in row) + " |" for row in frame.iter_rows()]
    return "\n".join([header, rule, *rows])


def render_report(report: StudyReport, manifest: dict[str, object]) -> str:
    """The report of a study, from its numbers: facts, the study, the scores, the equations, the terms."""
    schema, facts = report.schema, report.facts
    corpus = manifest["corpus"]
    lines = [
        f"# {schema.label}",
        "",
        f"Corpus `{Path(str(corpus['path'])).name}` (sha256 `{str(corpus['sha256'])[:12]}`): {facts['rows']} rows, "  # pyright: ignore[reportIndexIssue]
        f"{facts['datasets']} datasets"
        + (f", {facts['models']} models" if "models" in facts else "")
        + f"; the target runs from {facts['target_min']:.4g} to {facts['target_max']:.4g}.",
        "",
        f"Protocols: {', '.join(schema.protocols)}. Length rule: `{manifest['selection_method']}`. "
        f"Scale: {schema.transform}.",
        "",
        "## Equations",
        "",
    ]
    names = list(report.equations)
    summary = pl.DataFrame(
        [
            {
                "variant": VARIANT_NAMES[name],
                "terms": report.equations[name].n_terms,
                "arity": report.equations[name].arity,
                "features used": len({f for t in report.equations[name].equation.terms for f in t.features}),
            }
            for name in names
        ]
    )
    lines += [markdown_table(summary), ""]
    for name in names:
        lines += [f"### {VARIANT_NAMES[name]}", "", "```", str(report.equations[name].equation), "```", ""]
        rules = report.equations[name].rules
        if rules is not None and rules.height > 1:
            lines += ["Length rules (`chosen` is the one used):", "", markdown_table(rules), ""]
    scale = "original" if schema.log_target else "fit"
    lines += [f"## Scores ({'on the target itself' if schema.log_target else 'on the target'})", ""]
    lines += [markdown_table(report.scores.filter(pl.col("scale") == scale).drop("scale")), ""]
    if schema.log_target:
        lines += [
            "## Scores on the fitted scale",
            "",
            markdown_table(report.scores.filter(pl.col("scale") == "fit").drop("scale")),
            "",
        ]
    if not report.effects.is_empty():
        lines += ["## What the terms of E3-Valid are worth", "", markdown_table(report.effects), ""]
        lines += ["## Which features carry E3-Valid", "", markdown_table(report.shares), ""]
    return "\n".join(lines)
