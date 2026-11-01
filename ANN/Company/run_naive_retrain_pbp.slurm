#!/bin/bash
#SBATCH --job-name=naive-retrain-pbp
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --mail-user=f20231101@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Submit this job from ANN/:
#   sbatch --export=ALL,PREP_TYPE=2 Company/run_naive_retrain_pbp.slurm
cd "$SLURM_SUBMIT_DIR/Company"

PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found or not executable: $PYTHON" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PREP_TYPE="${PREP_TYPE:?Submit with --export=ALL,PREP_TYPE=2|3|4|5}"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname)"
echo "Working directory: $PWD"
echo "Python: $PYTHON"
"$PYTHON" --version

"$PYTHON" -u NaiveRetrain_PBP_Company.py \
    --mode multithread --prep_type="$PREP_TYPE"

echo "Naive retraining completed for prep type $PREP_TYPE."
