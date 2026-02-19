#!/bin/bash
#SBATCH --account=foster
#SBATCH --partition=compute
#SBATCH --job-name=nbci_fig4_pt
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Example module loads (edit for your cluster)
# module load julia/1.10
# module load gurobi

# If needed, set license path
# export GRB_LICENSE_FILE=/path/to/gurobi.lic

echo "SLURM job $SLURM_JOB_ID (array task ${SLURM_ARRAY_TASK_ID:-NA})"
echo "Running one (prior, z0) point job"

# The Julia script reads SLURM_ARRAY_TASK_ID automatically.
julia --project=. scripts/paper_fig4_postmean_point.jl "$@"
