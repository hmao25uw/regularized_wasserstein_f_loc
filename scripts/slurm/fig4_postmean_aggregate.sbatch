#!/bin/bash
#SBATCH --job-name=nbci_fig4_agg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Example module loads (edit for your cluster)
# module load julia/1.10

echo "SLURM job $SLURM_JOB_ID"
echo "Aggregating partial Figure-4 posterior-mean results and producing plots"

julia --project=. scripts/paper_fig4_postmean_aggregate.jl "$@"
