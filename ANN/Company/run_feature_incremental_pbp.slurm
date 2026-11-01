#!/bin/bash
#SBATCH --job-name=feat-incr-pbp
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
#   sbatch Company/run_feature_incremental_pbp.slurm
cd "$SLURM_SUBMIT_DIR/Company"

# The environment is located in ANN/.venv.
PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found or not executable: $PYTHON" >&2
    echo "Create the environment with uv or submit with --export=PYTHON=/path/to/python." >&2
    exit 1
fi

# The Python program launches four CPU worker processes internally.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Submit with: sbatch --export=ALL,PREP_TYPE=2 Company/run_feature_incremental_pbp.slurm
PREP_TYPE="${PREP_TYPE:?Submit with --export=ALL,PREP_TYPE=2|3|4|5}"

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname)"
echo "Working directory: $PWD"
echo "Python: $PYTHON"
"$PYTHON" --version

# No --prep_type means the program runs prep types 2, 3, 4, and 5 in order:
# ARFE, AVC, ARFE_AVC, and NaiveNA. Sequential execution keeps the shared
# consolidated_results.csv and Final/ aggregation outputs race-free.
"$PYTHON" -u FeatureIncremental_PBP_Company.py --mode multithread --prep_type="$PREP_TYPE" 

echo "Feature-incremental PBP completed ON PREP TYPE $PREP_TYPE."
