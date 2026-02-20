#!/bin/bash
#SBATCH --job-name=nbci_fig4_submit
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# ----------------------------------------------------------------------
# SUBMITTER job for the Figure-4 posterior-mean simulation (parallel, multi-node)
#
# Why a "submitter" job?
#   The Figure-4 pipeline consists of:
#     (1) precompute (single job)
#     (2) many independent point jobs (SLURM job-array)
#     (3) aggregate (single job)
#   with dependencies between stages.
#
# This script simply calls:
#   scripts/slurm/submit_fig4_postmean_parallel.sh
# which submits (1)-(3) with the correct dependencies.
#
# You can also run the .sh directly on the login node:
#   bash scripts/slurm/submit_fig4_postmean_parallel.sh
#
# Defaults match the Ignatiadis & Wager paper Figure-4 settings:
#   n=5000, nreps=400, z=-3:0.2:3
#
# Parallelization:
#   - block_size=100 replicates per task
#   - max_concurrent=6 array tasks at once (≈ up to 6 nodes at once)
#
# Usage:
#   sbatch scripts/slurm/fig4_postmean_submit_parallel.sbatch
#   sbatch scripts/slurm/fig4_postmean_submit_parallel.sbatch --outdir results/paper_fig4 --block_size 50 --max_concurrent 6
# ----------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Example module loads (edit for your cluster)
# module load julia/1.10
# module load gurobi

# If needed, set license path
# export GRB_LICENSE_FILE=/path/to/gurobi.lic

echo "SLURM submitter job $SLURM_JOB_ID"
echo "Submitting Figure-4 posterior-mean *parallel* pipeline..."

bash scripts/slurm/submit_fig4_postmean_parallel.sh "$@"
