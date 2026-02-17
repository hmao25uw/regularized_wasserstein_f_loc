#!/bin/bash
#SBATCH --job-name=nbci_fig4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# ----------------------------------------------------------------------
# SLURM pipeline for the *posterior mean* simulation (Figure 4-style)
#
# This runs BOTH priors (Spiky and NegSpiky) and produces FOUR figures:
#   - results/paper_fig4/plots/postmean_ci_bands_spiky.png
#   - results/paper_fig4/plots/postmean_ci_bands_negspiky.png
#   - results/paper_fig4/plots/postmean_coverage_spiky.png
#   - results/paper_fig4/plots/postmean_coverage_negspiky.png
#
# IMPORTANT: Edit module loads / license paths for your cluster.
# ----------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Example module loads (edit as needed)
# module load julia/1.10
# module load gurobi

# If needed, set license path
# export GRB_LICENSE_FILE=/path/to/gurobi.lic

mkdir -p results/paper_fig4

echo "SLURM job $SLURM_JOB_ID"
echo "Running posterior mean simulation pipeline (Spiky + NegSpiky)."

julia --project=. scripts/paper_fig4_postmean_pipeline.jl --outdir results/paper_fig4
