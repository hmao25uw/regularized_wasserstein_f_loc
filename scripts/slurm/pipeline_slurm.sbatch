#!/bin/bash
#SBATCH --account=foster
#SBATCH --partition=compute
#SBATCH --job-name=nbci
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# ----------------------------------------------------------------------
# SLURM pipeline for NonparBayesCI
#
# This runs *all* TOML configs in ./configs sequentially (same behavior as
# scripts/pipeline_local.jl) and then generates plots.
#
# IMPORTANT: You must edit the module loads and license paths to match
# your cluster environment.
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
echo "Running all configs in ./configs and then plotting."

julia --project=. scripts/pipeline_local.jl --outdir results

echo "Running real-data experiment: prostate z-scores (Figure 2-style)."
julia --project=. scripts/paper_fig2_prostate.jl --outdir results/paper_fig2_prostate

# NOTE: `paper_fig2_prostate.jl` will try to auto-download `data/prostate/prostz.txt`
# if it is missing. If your cluster blocks outbound internet, copy `prostz.txt`
# into `data/prostate/` before submitting this job.
