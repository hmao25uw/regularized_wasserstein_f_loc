#!/bin/bash
#SBATCH --account=foster
#SBATCH --partition=compute
#SBATCH --job-name=nbci_prostate
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

# ----------------------------------------------------------------------
# Real-data experiment: Prostate z-scores (Figure 2-style)
#
# This runs the prostate analysis and writes plots/CSVs under:
#   results/paper_fig2_prostate/
#
# Usage:
#   sbatch scripts/slurm/fig2_prostate.sbatch
#   sbatch scripts/slurm/fig2_prostate.sbatch --outdir results/my_prostate_run
# ----------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load elbert/julia/1.11.1/1.11.1

# If needed, set Gurobi license path (LPs are solved with Gurobi, conic with Clarabel)
export GRB_LICENSE_FILE=/mmfs1/home/hmao25/gurobi.lic

OUTDIR="results/paper_fig2_prostate"

# Allow an optional --outdir override
if [[ $# -ge 2 && "$1" == "--outdir" ]]; then
  OUTDIR="$2"
  shift 2
fi

mkdir -p "$OUTDIR"

echo "SLURM job $SLURM_JOB_ID"
echo "Running prostate real-data experiment -> $OUTDIR"

julia --project=. scripts/paper_fig2_prostate.jl --outdir "$OUTDIR" "$@"

echo "Done. Outputs are in: $OUTDIR"
