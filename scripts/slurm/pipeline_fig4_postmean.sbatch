#!/bin/bash
#SBATCH --account=foster
#SBATCH --partition=compute
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

module load elbert/julia/1.11.1/1.11.1
export GRB_LICENSE_FILE=/mmfs1/home/hmao25/gurobi.lic

# Cluster environment setup (Julia + optional Gurobi module). See:
#   scripts/slurm/_setup_env.sh
source scripts/slurm/_setup_env.sh

mkdir -p results/paper_fig4

echo "SLURM job $SLURM_JOB_ID"
echo "Running posterior mean simulation pipeline (Spiky + NegSpiky)."
echo "Defaulting to paper settings: n=5000, nreps=400, z=-3:0.2:3 (override by passing CLI args to sbatch)."

# NOTE:
# - We pass the paper defaults first.
# - Any user-supplied args (\"$@\") come last and therefore override defaults.

julia --project=. scripts/paper_fig4_postmean_pipeline.jl \
  --outdir results/paper_fig4 \
  --n 5000 \
  --nreps 400 \
  --zmin -3.0 --zmax 3.0 --zstep 0.2 \
  --alpha 0.05 \
  --boot_B 500 \
  --clt_B 2000 \
  "$@"
