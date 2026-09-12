"""Generate the English E3-Valid and E3-MAX configuration-search report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import polars as pl


def _metric(value: object) -> str:
    return f"{float(cast(float, value)):.4f}"


def _table_row(label: str, values: list[str]) -> str:
    return "| " + " | ".join([label, *values]) + " |"


def _candidate_table(payload: dict[str, Any]) -> str:
    candidates = [payload["e3_max"], payload["e3_valid"]]
    rows = [
        _table_row("E3 result", ["E3-MAX", "E3-Valid"]),
        _table_row("---", ["---:", "---:"]),
        _table_row("Selected terms", [str(item["n_terms"]) for item in candidates]),
        _table_row("Maximum arity", [str(item["max_arity"]) for item in candidates]),
        _table_row("Complexity", [str(item["complexity"]) for item in candidates]),
        _table_row("Composite objective $J$", [_metric(item["objective"]) for item in candidates]),
        _table_row("In-sample R²", [_metric(item["in_sample_r2"]) for item in candidates]),
        _table_row("LODO R²", [_metric(item["loo_dataset_r2"]) for item in candidates]),
        _table_row("LOMO R²", [_metric(item["loo_model_r2"]) for item in candidates]),
        _table_row("Doubly held-out R²", [_metric(item["loo_cell_r2"]) for item in candidates]),
        _table_row("Combined R²", [_metric(item["combined_r2"]) for item in candidates]),
        _table_row("Four-protocol R² floor", [_metric(item["four_protocol_floor"]) for item in candidates]),
        _table_row("Term stability", [_metric(item["stability"]) for item in candidates]),
        _table_row("Ridge penalty", [str(item["penalty"]) for item in candidates]),
        _table_row("Maximum absolute z-score", [str(item["max_abs_zscore"]) for item in candidates]),
    ]
    return "\n".join(rows)


def _equation(search_directory: Path, stem: str) -> str:
    return (search_directory / f"{stem}.txt").read_text(encoding="utf-8").strip()


def generate(search_directory: Path) -> tuple[Path, Path]:
    selected_path = search_directory / "selected_configurations.json"
    curve_path = search_directory / "e3_valid_term_count_curve.csv"
    if not selected_path.is_file() or not curve_path.is_file():
        raise FileNotFoundError("selection outputs and the term-count curve must exist")
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    curve = pl.read_csv(curve_path)
    minimum_terms = cast(int, curve["n_terms"].min())
    maximum_terms = cast(int, curve["n_terms"].max())
    valid = payload["e3_valid"]
    maximum = payload["e3_max"]
    diagnostics = payload["e3_valid_diagnostics"]
    features = ", ".join(f"`{feature}`" for feature in json.loads(maximum["features"]))

    report = f"""# E3 Selection: Search from {minimum_terms} to {maximum_terms} Terms

## Scope

E3-Valid and E3-MAX share the base configuration selected by the exhaustive search:

- Model descriptors: {features}
- Ridge penalty: {maximum["penalty"]}
- Maximum absolute z-score: {maximum["max_abs_zscore"]}

For every term count, the curve retains the arity-2 or arity-3 candidate with the highest
Combined R². Combined R² is the median of in-sample, LODO, and LOMO R².

## Results

{_candidate_table(payload)}

## Selection rules

E3-MAX maximises the minimum R² over in-sample, LODO, LOMO, and doubly held-out
evaluation. In the last protocol, every row sharing the test cell's dataset or model is
removed from training. E3-Valid selects the highest best-so-far Combined R² immediately before a
sustained plateau. The plateau tolerance is `{diagnostics["tolerance"]}`, and the forward
window covers `{diagnostics["window"]}` evaluated term counts. This rule selects
{valid["n_terms"]} terms at maximum arity {valid["max_arity"]}.

## Term-count curve

![E3 term-count curve](e3_valid_term_count_curve.png)

## Equations

### E3-MAX

```text
{_equation(search_directory, "e3_max")}
```

### E3-Valid

```text
{_equation(search_directory, "e3_valid")}
```
"""
    report_path = search_directory / "e3_selection_report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")

    summary = f"""# E3 Selection Summary: {minimum_terms}-{maximum_terms} Terms

The search selected E3-MAX with {maximum["n_terms"]} terms at maximum arity
{maximum["max_arity"]}, reaching a four-protocol R² floor of
{_metric(maximum["four_protocol_floor"])}. The retained plateau rule selected E3-Valid with
{valid["n_terms"]} terms at maximum arity {valid["max_arity"]} and a Combined R² of
{_metric(valid["combined_r2"])}.

See [the full E3 selection report](e3_selection_report.md) and
[the term-count curve](e3_valid_term_count_curve.png).
"""
    summary_path = search_directory / "summary.md"
    summary_path.write_text(summary, encoding="utf-8", newline="\n")
    return report_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search_directory", type=Path)
    arguments = parser.parse_args()
    for path in generate(arguments.search_directory.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
