#!/bin/bash
# Does a better beam find a better equation? Submit with:
#   sbatch --job-name=beam --cpus-per-task=62 scripts/beam_search.sbatch
#
# Crosses the 15 beam policies of `ml_meta_perf.beam` with the same configuration grid the
# equation search uses -- 16 feature subsets x 11 penalties x 23 lengths x 6 z-caps x 2
# arities = 48,576 points each, 728,640 in total. At the measured 2.2 s per point that is
# about 445 core-hours, so roughly 7 hours of wall-clock on 62 cores. The time limit is set
# well above that because a job killed at hour seven of eight leaves nothing.
#
# **Every policy gets its own configuration sweep, and that is the whole reason the job is
# this large.** `DEFAULT_E3` was itself chosen by a sweep under the vanilla beam, so scoring a
# different beam at that configuration compares a tuned setup against an untuned one and will
# favour the incumbent whatever the beam is worth. Anything cheaper than a full cross answers
# a question nobody asked.
#
# Two stages, run back to back. The sweep ranks by `equation_search.OBJECTIVE_WEIGHTS`, which
# is a **shortlisting device** -- it has already put a nine-term equation on top that a paired
# test called significantly worse, because the weight on `stability` outvoted an accuracy gap.
# The compare stage is the decision: each policy's best configuration, refit and paired
# against the incumbent on per-dataset absolute error over the twenty held-out datasets.
#
# A tie is not a win. If a variant matches the published beam it is more machinery for no
# measured gain, and `DEFAULT_E3` with `beam.VANILLA` stays.
#
# Re-sync before submitting. The cluster copy at ~/aiml-model is an rsync of the tree, not a
# clone, and it went three weeks stale once and searched the wrong grammar as a result:
#
#   rsync -av --delete --exclude='__pycache__' --exclude='*.egg-info' \
#     --relative ./src ./scripts ./pyproject.toml ./requirements.txt playstation:~/aiml-model/

#SBATCH --partition=cpuPartition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
# 1G per worker. The measured peak on the largest grid point is 216 MB, and the beam policies
# add only an SVD over a few hundred columns on top of that. This has to be set explicitly:
# the cluster default is 4G per CPU, which at --cpus-per-task=62 asks for 248G against
# `atari`'s 246G. That made an earlier job unschedulable on the one node that could have run
# it, and Slurm parked it five months out.
#SBATCH --mem-per-cpu=1G
#SBATCH --output=results/cluster/slurm-%x-%j.out
#SBATCH --error=results/cluster/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/cluster

# One BLAS thread per worker: each joblib worker runs an independent fit, and letting every
# one spawn a 62-thread pool oversubscribes the node by three orders of magnitude.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JOBLIB_START_METHOD=loky
export PYTHONUNBUFFERED=1

NAME="${BEAM_NAME:-beam_search}"

echo "=== $(date -Is) node=$(hostname) job=$SLURM_JOB_ID cores=$SLURM_CPUS_PER_TASK name=$NAME ==="
uname -m
venv/bin/python -c "import numpy, polars, sklearn; print('numpy', numpy.__version__, 'polars', polars.__version__, 'sklearn', sklearn.__version__)"
# ESS and TORANN drive the space-filling seeding policies. They are an optional extra, and
# `seeding.empty_space` degrades to the classical maximin rule without them -- which would
# silently turn four policies into duplicates of two others, so the job says which it has.
venv/bin/python -c "
try:
    import ess, torann; print('seeding: ess + torann available')
except ImportError as error:
    print('seeding: DEGRADED to maximin --', error)
"

echo "--- stage 1: sweep (the shortlist) ---"
venv/bin/python -m ml_meta_perf.beam_search_cli sweep \
  --scope full \
  --jobs "$SLURM_CPUS_PER_TASK" \
  --output results/cluster \
  --name "$NAME"

echo "--- stage 2: compare (the decision) ---"
exec venv/bin/python -m ml_meta_perf.beam_search_cli compare \
  --output results/cluster \
  --name "$NAME"
