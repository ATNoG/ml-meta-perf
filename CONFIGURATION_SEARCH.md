# E3 (dataset-and-model equation) configuration recalibration

The ordinary `ml-meta-perf` command reproduces the study with its retained configuration and
selection rules. The separate `ml-meta-perf-search` command performs the exhaustive, resumable
configuration search intended for a Slurm node.

The abbreviations used below are coefficient of determination (**R²**), in-sample (**IS**),
leave-one-dataset-out (**LODO**), leave-one-model-out (**LOMO**), doubly held out (**DHO**),
comma-separated values (**CSV**), Secure File Transfer Protocol (**SFTP**), line feed (**LF**),
carriage return plus line feed (**CRLF**), and central processing unit (**CPU**).

## Retained 1–25-term search

The canonical output directory for the completed corrected-corpus sweep is
`results/configuration_search`. Search outputs are generated locally and excluded from Git
because the merged CSV files are large; the retained settings and reported results are preserved
in the source, chapters, and figures. Its shared base configuration is:

- model descriptors: `Model Capability`, `Processing Units Number`, `Fitting Regime`, and
  `Loss Margin Behaviour`;
- ridge penalty: `1.0`;
- maximum term z-score: `5.0`;
- candidate pool: `600`;
- beam width: `6`;
- maximum arities: `2` and `3`;
- equation lengths: `1` through `25`.

E3-MAX selects the arity-3 equation with 25 terms. Its R² values are 0.7194 under IS, 0.6911
under LODO, 0.6554 under LOMO, and 0.6554 under DHO.

E3-Valid selects the arity-2 equation with 18 terms. Its corresponding R² values are 0.6787,
0.6517, 0.6149, and 0.6103.

## Search and selection

The default search evaluates every subset containing two to six of the six model descriptors.
It crosses those 57 subsets with:

- ridge penalties `0.1, 0.3, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 60, 80`;
- term z-score caps `3.0, 3.5, 4.0, 4.25, 4.5, 5.0`;
- maximum arities `2` and `3`;
- every equation length from 1 through 25, obtained from one reused beam path.

The cheap stage ranks candidates with the historical composite objective `J`. It combines
IS, LODO and LOMO R², threshold decisions, ranking,
term stability and brevity. `J` produces a diverse shortlist and does not make the final
selection.

The expensive stage computes DHO validation for the shortlisted base settings:
for each observed cell, it removes every row sharing that cell's dataset or model.
E3-MAX maximises the minimum R² over all four protocols. E3-Valid shares E3-MAX's descriptor
subset, ridge penalty and z-score, and uses:

$$
R^2_{\mathrm{combined}} = \operatorname{median}\left(
R^2_{\mathrm{IS}},
R^2_{\mathrm{LODO}},
R^2_{\mathrm{LOMO}}
\right).
$$

At every term count, the arity with the highest Combined R² is retained. The rule follows the
best Combined R² observed so far and selects the first point whose gain over the next three
evaluated term counts is at most `0.001`. These values are exposed as `--plateau-window` and
`--plateau-tolerance` and are recorded in `selected_configurations.json`.

## Copy with SFTP

Upload `pyproject.toml`, `requirements-reproducibility.txt`, `src/`, and `scripts/` to the
same project directory on the cluster. Do not upload the local `venv/` or `results/`
directories. SFTP transfers the files as bytes; both tracked `.sbatch` files use Unix LF line
endings, enforced by `.gitattributes` for Git checkouts. After a direct Windows upload, verify
them on the cluster with:

```bash
file scripts/*.sbatch
```

If that command reports CRLF line endings, convert them before submission:

```bash
sed -i 's/\r$//' scripts/*.sbatch
```

After both jobs finish, download the complete `results/configuration_search/` and
`results/configuration_search_max100/` directories. The Slurm logs are useful for checking
completion but are excluded from Git by `slurm-*.out`.

## Run on Slurm

Create the cluster environment from the repository root:

```bash
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements-reproducibility.txt
```

The cluster may provide Python directly rather than through the `module` command. Verify the
planned grid without fitting:

```bash
venv/bin/python -m ml_meta_perf.configuration_search plan
```

Before requesting a full node, run a small end-to-end check:

```bash
venv/bin/python -m ml_meta_perf.configuration_search all \
  --output results/configuration_search_smoke --jobs 1 \
  --penalty 20 --zscore 3 --arity 2 \
  --feature-set "Model Capability,Processing Units Number" \
  --min-terms 2 --max-terms 3 --pool 40 --beam 2 --shortlist-top 1
```

Submit the retained 25-term search with:

```bash
sbatch scripts/equation_search.sbatch
```

Submit the separate exploratory 100-term search with:

```bash
sbatch scripts/equation_search_100.sbatch
```

The jobs request one 60-core node, assign one independent configuration path to each worker,
and limit numerical libraries to one thread per worker to avoid CPU oversubscription. Both
commands are resumable: complete shards are skipped when the same job is submitted again.

## Run stages manually

The `all` stage is equivalent to:

```bash
venv/bin/python -m ml_meta_perf.configuration_search sweep --jobs 60
venv/bin/python -m ml_meta_perf.configuration_search merge
venv/bin/python -m ml_meta_perf.configuration_search shortlist
venv/bin/python -m ml_meta_perf.configuration_search validate --jobs 60
venv/bin/python -m ml_meta_perf.configuration_search select \
  --plateau-tolerance 0.001 --plateau-window 3
```

Every invocation must use the same scientific flags and output directory as the original run.
The manifest rejects an incompatible search.

## Result files

```text
manifest.json                 exact search identity
run_summary.json              requested stage and elapsed time
grid.csv                     every requested configuration
equation_search.csv          all cheap-stage lengths and components of J
shortlist.csv                base settings sent to DHO validation
finalists.csv                shortlist with the fourth protocol
finalist_fold_errors.csv     per-dataset DHO errors
selected_configurations.json selected settings, metrics and plateau diagnostics
e3_valid.json / .txt         retained E3-Valid equation
e3_max.json / .txt           E3-MAX capability equation
```

Generate the selection curves and English reports with:

```bash
venv/bin/python scripts/plot_e3_valid_selection.py results/configuration_search
venv/bin/python scripts/report_e3_valid_selection.py results/configuration_search
venv/bin/python scripts/plot_e3_valid_selection.py results/configuration_search_max100
venv/bin/python scripts/report_e3_valid_selection.py results/configuration_search_max100
```

Generate the full term-count curves for the primary and 100-term searches with:

```bash
venv/bin/python scripts/plot_e3_validation_curves.py results/configuration_search
venv/bin/python scripts/plot_e3_validation_curves.py results/configuration_search_max100
```

Explore the retained plateau rule locally over the completed primary 1–25-term search with:

```bash
venv/bin/python scripts/analyze_plateau_sensitivity.py results/configuration_search
```

This writes `plateau_sensitivity.csv`, `.png`, `.pdf`, and `.md` beside the search outputs. It
does not refit equations or modify the selected configuration.

The ordinary experiment contains the same shared base configuration and the same E3-Valid
plateau rule. It derives the equations from the packaged meta-dataset rather than treating the
search result directory as an unversioned runtime input.
