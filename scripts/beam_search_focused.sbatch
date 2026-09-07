#!/bin/bash
# The same comparison on the arity-2 four-feature band the full sweep already settled.
# Submit with:
#   sbatch --job-name=beamf --cpus-per-task=32 scripts/beam_search_focused.sbatch
#
# Why a second, smaller job exists. The full cross in `beam_search.sbatch` is seven hours and
# answers "is there a beam policy that, given its own configuration sweep, finds a better
# equation than the published one anywhere in the grid". This one is about forty minutes and
# answers the narrower question that comes up first: "at the operating point the study already
# chose -- arity 2, the two load-bearing model columns -- does a different beam do better?"
#
# Run this one first. If nothing here beats the incumbent, the full sweep is still worth
# running (a different beam may prefer a different operating point entirely), but the result
# arrives the same day rather than the next morning.
#
# 1 feature subset x 11 penalties x 23 lengths x 3 z-caps x 1 arity x 15 policies = 11,385
# points at about 2.2 s, so roughly 7 core-hours.

#SBATCH --partition=cpuPartition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=results/cluster/slurm-%x-%j.out
#SBATCH --error=results/cluster/slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/cluster

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JOBLIB_START_METHOD=loky
export PYTHONUNBUFFERED=1

NAME="${BEAM_NAME:-beam_search_focused}"

echo "=== $(date -Is) node=$(hostname) job=$SLURM_JOB_ID cores=$SLURM_CPUS_PER_TASK name=$NAME ==="
venv/bin/python -c "
try:
    import ess, torann; print('seeding: ess + torann available')
except ImportError as error:
    print('seeding: DEGRADED to maximin --', error)
"

# `--scope focused` fixes the feature subset to the two load-bearing columns and the arity to
# 2, and narrows the z-cap grid to three values around the published 4.25. The penalty and
# length axes stay at full resolution, because those are the two the published configuration
# is most sensitive to.
echo "--- stage 1: sweep (the shortlist) ---"
venv/bin/python -m ml_meta_perf.beam_search_cli sweep \
  --scope focused \
  --jobs "$SLURM_CPUS_PER_TASK" \
  --output results/cluster \
  --name "$NAME"

echo "--- stage 2: compare (the decision) ---"
exec venv/bin/python -m ml_meta_perf.beam_search_cli compare \
  --output results/cluster \
  --name "$NAME"
