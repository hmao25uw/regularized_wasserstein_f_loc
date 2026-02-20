#!/bin/bash
#SBATCH --job-name=nbci_paper_submit
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# ----------------------------------------------------------------------
# PAPER-REPLICATION SUBMITTER for NonparBayesCI
#
# This file is now a *submitter* job (fast) that submits TWO separate jobs:
#   1) Figure-4 posterior-mean simulation (parallel, multi-node via job arrays)
#   2) Prostate real-data experiment (single job)
#
# If you want to run them individually, submit:
#   sbatch scripts/slurm/fig4_postmean_submit_parallel.sbatch
#   sbatch scripts/slurm/fig2_prostate.sbatch
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

echo "SLURM submitter job $SLURM_JOB_ID"
echo "Submitting Figure-4 posterior-mean (parallel) and prostate real-data jobs..."

# Submit simulation (parallel pipeline)
FIG4_JOB=$(sbatch --parsable scripts/slurm/fig4_postmean_submit_parallel.sbatch)
echo "  submitted fig4 submitter job: $FIG4_JOB"

# Submit prostate (real data)
PROSTATE_JOB=$(sbatch --parsable scripts/slurm/fig2_prostate.sbatch)
echo "  submitted prostate job: $PROSTATE_JOB"

echo
echo "Monitor with: squeue -j $FIG4_JOB,$PROSTATE_JOB"
echo "Figure-4 results will end up in: results/paper_fig4"
echo "Prostate results will end up in: results/paper_fig2_prostate"
