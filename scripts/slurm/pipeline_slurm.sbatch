#!/bin/bash
#SBATCH --job-name=nbci_paper
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

# ----------------------------------------------------------------------
# PAPER-REPLICATION SLURM pipeline for NonparBayesCI
#
# This script is intended to produce the *paper-style* results you asked for:
#   1) Figure-4 style posterior-mean simulation (Spiky + NegSpiky)
#   2) Prostate real-data experiment (Figure-2 style)
#
# IMPORTANT:
#   - The original Ignatiadis & Wager (2021) Fig-4 simulation uses:
#       n = 5000, nreps = 400, z-grid = -3:0.2:3
#   - We call the corresponding Julia pipeline with those settings.
#
# If you want the small "smoke-test" configs under ./configs instead, run:
#   sbatch scripts/slurm/pipeline_smoke_slurm.sbatch
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

# -------------------------
# Figure 4: posterior mean
# -------------------------

echo "Running Figure-4 posterior-mean pipeline (paper settings: n=5000, nreps=400, z=-3:0.2:3)."
mkdir -p results/paper_fig4

julia --project=. scripts/paper_fig4_postmean_pipeline.jl \
  --outdir results/paper_fig4 \
  --n 5000 \
  --nreps 400 \
  --zmin -3.0 --zmax 3.0 --zstep 0.2 \
  --alpha 0.05 \
  --boot_B 500 \
  --clt_B 2000

# -------------------------
# Figure 2: prostate (real data)
# -------------------------

echo "Running real-data experiment: prostate z-scores (Figure 2-style)."
mkdir -p results/paper_fig2_prostate

julia --project=. scripts/paper_fig2_prostate.jl --outdir results/paper_fig2_prostate

echo "Done. See results/ for outputs."
