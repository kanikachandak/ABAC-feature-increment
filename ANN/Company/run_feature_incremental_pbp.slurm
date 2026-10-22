#!/bin/bash
#SBATCH --job-name=feat-incr-pbp
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-user=f20231101@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# SLURM may execute a copied batch script from /var/spool, so BASH_SOURCE is
# not a reliable way to locate the repository. SLURM_SUBMIT_DIR is the actual
# directory from which sbatch was invoked.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$SUBMIT_DIR/Company/FeatureIncremental_PBP_Company.py" ]]; then
    ANN_DIR="$(cd -- "$SUBMIT_DIR" && pwd)"
elif [[ -f "$SUBMIT_DIR/ANN/Company/FeatureIncremental_PBP_Company.py" ]]; then
    ANN_DIR="$(cd -- "$SUBMIT_DIR/ANN/Company" && pwd)"
else
    echo "ERROR: Could not locate ANN/Company from submit directory: $SUBMIT_DIR" >&2
    exit 1
fi
cd "$ANN_DIR"

# The one-time setup inside ANN/ is:
#   uv venv .venv --python /usr/bin/python3.11
#   uv pip install --python .venv/bin/python -r requirements.txt
# Override PYTHON only if the ANN environment has a nonstandard location.
PYTHON="${PYTHON:-$ANN_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found or not executable: $PYTHON" >&2
    echo "Create the environment with uv or submit with --export=PYTHON=/path/to/python." >&2
    exit 1
fi

# The Python program launches four CPU worker processes internally.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname)"
echo "Working directory: $PWD"
echo "Python: $PYTHON"
"$PYTHON" --version

# No --prep_type means the program runs prep types 2, 3, 4, and 5 in order:
# ARFE, AVC, ARFE_AVC, and NaiveNA. Sequential execution keeps the shared
# consolidated_results.csv and Final/ aggregation outputs race-free.
"$PYTHON" FeatureIncremental_PBP_Company.py --mode multithread

echo "Feature-incremental PBP job completed."
