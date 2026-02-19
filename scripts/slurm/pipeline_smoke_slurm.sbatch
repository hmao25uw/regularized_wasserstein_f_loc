#!/bin/bash
#SBATCH --job-name=nbci_smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# ----------------------------------------------------------------------
# SMOKE-TEST SLURM pipeline for NonparBayesCI
#
# This runs *all* TOML configs in ./configs sequentially (same behavior as
# scripts/pipeline_local.jl) and then generates the simple demo plots:
#   - results/plots/ci_bands.png
#   - results/plots/ci_length.png
#
# IMPORTANT:
#   - These configs are NOT the Ignatiadis & Wager (2021) paper Figure-4
#     posterior-mean simulation.
#   - They are intentionally small, fast "sanity-check" experiments.
#
# For Figure-4 style posterior-mean results, use ONE of:
#   - scripts/slurm/pipeline_fig4_postmean.sbatch  (single sequential job)
#   - scripts/slurm/submit_fig4_postmean_parallel.sh (job-array workflow)
#
# For the prostate real-data experiment (Figure 2-style), run:
#   - julia --project=. scripts/paper_fig2_prostate.jl --outdir results/paper_fig2_prostate
# ----------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Example module loads (edit as needed)
# module load julia/1.10
# module load gurobi

# If needed, set license path
# export GRB_LICENSE_FILE=/path/to/gurobi.lic

mkdir -p results

echo "SLURM job $SLURM_JOB_ID"
echo "Running smoke-test configs in ./configs and then plotting."

julia --project=. scripts/pipeline_local.jl --outdir results
