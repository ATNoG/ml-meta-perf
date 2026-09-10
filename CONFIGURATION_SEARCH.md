# E3 configuration recalibration

The ordinary `ml-meta-perf` command reproduces the study with its retained configuration. It
does not recalibrate that configuration. The separate `ml-meta-perf-search` command performs
the exhaustive, resumable search intended for a Slurm node.

## Retained result

The corrected-corpus sweep completed as Slurm job 15343 on 2026-09-10. It evaluated 10,944
configuration-and-arity paths, covering 218,880 equation lengths, and validated 158
shortlisted finalists. The retained shared configuration is:

- model descriptors: `Model Capability`, `Processing Units Number`, `Fitting Regime`, and
  `Loss Margin Behaviour`;
- ridge penalty: `1.0`;
- maximum term z-score: `5.0`;
- arity: `3`;
- selected length: `25` terms.

E3-Valid and E3-MAX coincide at this configuration. Their R² values are 0.7194 in-sample,
0.6911 leave-one-dataset-out, 0.6554 leave-one-model-out, and 0.6554 with both the dataset and
model held out. The competing arity-2 grammar peaks at 17 terms with a four-protocol floor of
0.6109. Its error disadvantage is 3.09 times its paired bootstrap spread, so it does not pass
the grammar rule. The ordinary experiment defaults now contain this retained result.

## What is searched

The default search evaluates every subset containing two to six of the six model descriptors.
No descriptor is mandatory. It crosses those 57 subsets with:

- ridge penalties `0.1, 0.3, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 60, 80`;
- term z-score caps `3.0, 3.5, 4.0, 4.25, 4.5, 5.0`;
- maximum arities `2` and `3`;
- every equation length from 6 through 25, obtained from one reused beam path.

This produces 10,944 configuration-and-arity paths. The candidate pool remains 600 and the
beam width remains 6 because the earlier beam audit found the same equation to four decimal
places at up to eight times the search effort.

The cheap stage ranks candidates with the historical composite objective `J`. It uses
in-sample, leave-one-dataset-out and leave-one-model-out R2, threshold decisions, ranking,
term stability and brevity. `J` creates a shortlist; it does not make the final decision.

The expensive stage computes the doubly-held-out cell protocol only for shortlisted base
settings. E3-MAX maximises the worst R2 over all four protocols. E3-Valid shares E3-MAX's
feature subset, penalty and z-score, and is the simplest grammar whose loss relative to E3-MAX
does not exceed the paired bootstrap spread of that loss.

## Copy the project to the cluster

The safest route is to commit these files on a branch, push it, and check out that exact commit
on the cluster. From the cluster:

```bash
git clone <repository-url> ml-meta-perf
cd ml-meta-perf
git checkout <commit-or-branch>
```

If the cluster already has a checkout:

```bash
cd <cluster-project-directory>/ml-meta-perf
git fetch --all --prune
git checkout <commit-or-branch>
git pull --ff-only
```

For a direct copy without Git, run this from the parent directory on the local machine:

```bash
rsync -az --delete \
  --exclude .git --exclude venv --exclude results --exclude .idea \
  ml-meta-perf/ <cluster-user>@<cluster-host>:<cluster-project-directory>/ml-meta-perf/
```

Do not copy the Windows virtual environment. Python environments are platform-specific.

## Create the cluster environment

Run once from the repository root on the cluster:

```bash
module load python/3.12  # only if the cluster requires a Python module
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -e .
```

Verify the planned grid without fitting anything:

```bash
venv/bin/python -m ml_meta_perf.configuration_search plan
```

It must report 10,944 configurations and create:

```text
results/configuration_search/manifest.json
results/configuration_search/grid.csv
```

Inspect `manifest.json` before submission. It records the meta-dataset and package-source
SHA-256 hashes, Git revision, dirty-worktree state, grid, objective weights, Python version
and dependency versions.

Before requesting a full node, run this small end-to-end check from the login node or a short
interactive allocation:

```bash
venv/bin/python -m ml_meta_perf.configuration_search all \
  --output results/configuration_search_smoke --jobs 1 \
  --penalty 20 --zscore 3 --arity 2 \
  --feature-set "Model Capability,Processing Units Number" \
  --min-terms 2 --max-terms 3 --pool 40 --beam 2 --shortlist-top 1
```

It should finish in seconds and write `e3_valid.json` and `e3_max.json` under the separate
smoke directory. It does not affect the full result directory.

## Submit and monitor the Slurm job

From the repository root:

```bash
sbatch scripts/equation_search.sbatch
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f slurm-e3-config-search-<job-id>.out
```

The script requests one 60-core node, 1 GiB per core and 48 hours. Adjust only the Slurm header
if the cluster uses another partition or CPU count. The Python grid remains recorded in the
manifest.

If the job reaches its time limit or the node fails, submit the same script again:

```bash
sbatch scripts/equation_search.sbatch
```

Completed sweep and finalist shards are detected and skipped. Do not change the grid while
reusing an output directory; the manifest rejects a different experiment.

## Run stages manually

The `all` stage used by Slurm is equivalent to:

```bash
venv/bin/python -m ml_meta_perf.configuration_search sweep --jobs 60
venv/bin/python -m ml_meta_perf.configuration_search merge
venv/bin/python -m ml_meta_perf.configuration_search shortlist
venv/bin/python -m ml_meta_perf.configuration_search validate --jobs 60
venv/bin/python -m ml_meta_perf.configuration_search select
```

This is useful when only the final stage needs to be repeated. Every invocation must use the
same scientific flags and output directory as the original run. The default Slurm script avoids
that source of error by running `all` with one manifest.

## Results to copy back

When the job exits successfully, archive the entire result directory and its Slurm log:

```bash
tar -czf configuration-search-<job-id>.tar.gz \
  results/configuration_search \
  slurm-e3-config-search-<job-id>.out
```

Download it from the local machine:

```bash
scp <cluster-user>@<cluster-host>:<cluster-project-directory>/ml-meta-perf/configuration-search-<job-id>.tar.gz .
```

The principal files are:

```text
manifest.json                 exact experiment identity
run_summary.json              requested stage, timestamps and elapsed time
grid.csv                     every requested configuration
equation_search.csv          all cheap-stage lengths and components of J
shortlist.csv                base settings sent to leave-one-cell-out
finalists.csv                shortlist with the fourth protocol
finalist_fold_errors.csv     per-dataset errors used by the paired comparison
shared_grammar_margins.csv   E3-Valid versus E3-MAX decision
selected_configurations.json selected settings and headline metrics
e3_valid.json / .txt         selected readable equation
e3_max.json / .txt           selected maximum-capability equation
```

The cluster job never edits `experiment.py` or the paper. For job 15343, the selected values
were reviewed locally, copied into the defaults, and used to regenerate the results, figures,
and documentation. A future recalibration should follow the same review and incorporation
step rather than treating search output as an automatic source-code change.
