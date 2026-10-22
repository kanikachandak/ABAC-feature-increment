#!/bin/bash
#SBATCH --job-name=feat-incr-pbp
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-user=f20231101@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Run from the directory containing this script so all paths used by the
# Python program (Dataset/, Results/, Model/) resolve correctly.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# The recommended one-time setup is:
#   cd "$PROJECT_DIR"
#   uv venv .venv --python /usr/bin/python3.11
#   uv pip install --python .venv/bin/python -r ANN/requirements.txt
# Override PYTHON with an absolute path if the environment lives elsewhere.
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
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
