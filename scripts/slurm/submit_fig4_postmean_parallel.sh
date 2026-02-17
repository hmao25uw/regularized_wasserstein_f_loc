#!/bin/bash
#
# Submit the *parallel* (job-array) Figure-4 posterior-mean pipeline.
#
# This script should be run on the **login node** (not via sbatch).
# It submits:
#   1) a precompute job (both priors)
#   2) an array of point jobs (one per (prior, z0))
#   3) an aggregation job (writes final CSVs + 4 plots)
#
# Example:
#   bash scripts/slurm/submit_fig4_postmean_parallel.sh --outdir results/paper_fig4 --nreps 4000
#

set -euo pipefail

# Defaults (keep in sync with Julia scripts)
OUTDIR="results/paper_fig4"
ZMIN="-3.0"
ZMAX="3.0"
ZSTEP="0.2"

# Parse a small subset of args so we can compute the job-array size.
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      OUTDIR="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --prior)
      # This submit helper always runs BOTH priors (spiky + negspiky) via a
      # (prior, z0) job-array mapping. Passing --prior would break that mapping.
      echo "[submit] Ignoring user-supplied --prior $2 (this pipeline always runs both priors)." >&2
      shift 2 ;;
    --zmin)
      ZMIN="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --zmax)
      ZMAX="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    --zstep)
      ZSTEP="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
    *)
      PASSTHRU+=("$1"); shift ;;
  esac
done

# Compute number of z grid points K = round((ZMAX-ZMIN)/ZSTEP) + 1.
K=$(python3 - <<PY
import math
zmin=float("$ZMIN")
zmax=float("$ZMAX")
zstep=float("$ZSTEP")
K=int(round((zmax-zmin)/zstep))+1
if K<=0:
    raise SystemExit("Invalid grid")
print(K)
PY
)

# Two priors by default (spiky + negspiky)
NTASKS=$((2 * K))

echo "Submitting Figure-4 posterior-mean PARALLEL pipeline"
echo "  OUTDIR=$OUTDIR"
echo "  z-grid: [$ZMIN:$ZSTEP:$ZMAX] => K=$K"
echo "  array tasks: NTASKS=$NTASKS (2 priors × K points)"

PRE_JOB=$(sbatch --parsable scripts/slurm/fig4_postmean_precompute.sbatch --prior both "${PASSTHRU[@]}")
echo "  submitted precompute job: $PRE_JOB"

ARRAY_JOB=$(sbatch --parsable --dependency=afterok:${PRE_JOB} --array=1-${NTASKS} scripts/slurm/fig4_postmean_point.sbatch "${PASSTHRU[@]}")
echo "  submitted point array job: $ARRAY_JOB"

AGG_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} scripts/slurm/fig4_postmean_aggregate.sbatch "${PASSTHRU[@]}")
echo "  submitted aggregate job: $AGG_JOB"

echo
echo "You can monitor with: squeue -j $PRE_JOB,$ARRAY_JOB,$AGG_JOB"
echo "Final plots will be in: $OUTDIR/plots"
