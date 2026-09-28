#!/usr/bin/env bash
# start_server.sh  –  Launch the Flower ServerApp on the server machine.
#
# Usage:
#   chmod +x start_server.sh
#   ./start_server.sh [OPTIONS]
#
# Environment variables (override defaults):
#   DATASET_NAME        company | university1 | university2  (default: company)
#   PARTITION_STRATEGY  rnp | pbp | sbp                     (default: sbp)
#   PREP_TYPE           2 | 3 | 4 | 5                       (default: 2)
#   TRAIN_METHOD        bagging | cyclic                     (default: bagging)
#   FLWR_SEED           integer seed                        (default: 0)
#   SERVER_ADDRESS      host:port                           (default: 0.0.0.0:9092)
#   NUM_ROUNDS          number of FL rounds                 (default: 50)
#   OUTPUT_DIR          results output path                 (default: ./results)
#
# Example:
#   DATASET_NAME=university2 PARTITION_STRATEGY=sbp PREP_TYPE=4 ./start_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── defaults ──────────────────────────────────────────────────────────────────
export DATASET_NAME="${DATASET_NAME:-company}"
export PARTITION_STRATEGY="${PARTITION_STRATEGY:-sbp}"
export PREP_TYPE="${PREP_TYPE:-2}"
export TRAIN_METHOD="${TRAIN_METHOD:-bagging}"
export FLWR_SEED="${FLWR_SEED:-0}"
export PYTHONHASHSEED="${FLWR_SEED}"
export SERVER_ADDRESS="${SERVER_ADDRESS:-0.0.0.0:9092}"
export NUM_ROUNDS="${NUM_ROUNDS:-50}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"

# ── resolve CSV paths ─────────────────────────────────────────────────────────
TRAIN_CSV="${SCRIPT_DIR}/train_${DATASET_NAME}.csv"
TEST_CSV="${SCRIPT_DIR}/test_${DATASET_NAME}.csv"

if [[ ! -f "$TRAIN_CSV" ]]; then
    TRAIN_CSV="${SCRIPT_DIR}/Dataset/train_${DATASET_NAME}.csv"
    TEST_CSV="${SCRIPT_DIR}/Dataset/test_${DATASET_NAME}.csv"
fi

if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "ERROR: train_${DATASET_NAME}.csv not found in $SCRIPT_DIR or $SCRIPT_DIR/Dataset/"
    exit 1
fi

export TRAIN_CSV TEST_CSV

PREP_NAMES=([2]="ARFE" [3]="AVC" [4]="ARFE_AVC" [5]="NaiveNA")
export CASE_NAME="${PREP_NAMES[$PREP_TYPE]:-PrepType${PREP_TYPE}}"

export SERVER_EVAL_LOG="${OUTPUT_DIR}/${DATASET_NAME}_${PARTITION_STRATEGY}_${CASE_NAME}_seed${FLWR_SEED}.log"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  Flower ServerApp — Distributed Mode"
echo "  Dataset   : $DATASET_NAME"
echo "  Strategy  : $PARTITION_STRATEGY"
echo "  PrepType  : $PREP_TYPE ($CASE_NAME)"
echo "  Method    : $TRAIN_METHOD"
echo "  Seed      : $FLWR_SEED"
echo "  Address   : $SERVER_ADDRESS"
echo "  Rounds    : $NUM_ROUNDS"
echo "  Output    : $OUTPUT_DIR"
echo "============================================================"

# Launch the Flower server-side app.
# In Flower 1.x distributed mode the server app runs via flower-server-app.
# The SuperLink (network relay) must also be running; if you use the all-in-one
# flower-superlink binary, start that first on the same machine:
#
#   flower-superlink --insecure &
#   ./start_server.sh
#
exec flower-server-app server:app \
    --superlink "${SERVER_ADDRESS}" \
    --insecure
