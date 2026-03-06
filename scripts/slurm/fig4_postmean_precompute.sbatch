#!/bin/bash
#SBATCH --account=foster
#SBATCH --partition=compute
#SBATCH --job-name=nbci_fig4_pre
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Example module loads (edit for your cluster)
module load elbert/julia/1.11.1/1.11.1
export GRB_LICENSE_FILE=/mmfs1/home/hmao25/gurobi.lic

# If needed, set license path
# export GRB_LICENSE_FILE=/path/to/gurobi.lic

echo "SLURM job $SLURM_JOB_ID"
echo "Precomputing localization radii + stats for Figure-4 posterior-mean simulation"

# Pass through any CLI args you supply to sbatch.
julia --project=. scripts/paper_fig4_postmean_precompute.jl "$@"
