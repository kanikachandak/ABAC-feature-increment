#!/usr/bin/env bash
# start_client.sh  –  Launch a Flower SuperNode (client) on a client machine.
#
# Run this script on EACH client machine that will participate in training.
# The SuperNode connects to the SuperLink running on the server machine.
#
# Usage:
#   chmod +x start_client.sh
#   ./start_client.sh [OPTIONS]
#
# Required environment variable:
#   SERVER_ADDRESS  IP or hostname of the server machine, e.g. 192.168.1.10:9092
#
# Optional environment variables (must match server's configuration):
#   DATASET_NAME        company | university1 | university2  (default: company)
#   PARTITION_STRATEGY  rnp | pbp | sbp                     (default: sbp)
#   PREP_TYPE           2 | 3 | 4 | 5                       (default: 2)
#   TRAIN_METHOD        bagging | cyclic                     (default: bagging)
#   FLWR_SEED           integer seed                        (default: 0)
#   CLIENT_API_ADDRESS  local address for ClientApp API     (default: 0.0.0.0:9094)
#
# Example (run on each of 4 client machines):
#   SERVER_ADDRESS=192.168.1.10:9092 DATASET_NAME=company ./start_client.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${SERVER_ADDRESS:-}" ]]; then
    echo "ERROR: SERVER_ADDRESS environment variable is required."
    echo "  Example: SERVER_ADDRESS=192.168.1.10:9092 ./start_client.sh"
    exit 1
fi

# ── defaults ──────────────────────────────────────────────────────────────────
export DATASET_NAME="${DATASET_NAME:-company}"
export PARTITION_STRATEGY="${PARTITION_STRATEGY:-sbp}"
export PREP_TYPE="${PREP_TYPE:-2}"
export TRAIN_METHOD="${TRAIN_METHOD:-bagging}"
export FLWR_SEED="${FLWR_SEED:-0}"
export PYTHONHASHSEED="${FLWR_SEED}"
export CLIENT_API_ADDRESS="${CLIENT_API_ADDRESS:-0.0.0.0:9094}"

# ── resolve CSV paths ─────────────────────────────────────────────────────────
TRAIN_CSV="${SCRIPT_DIR}/train_${DATASET_NAME}.csv"
TEST_CSV="${SCRIPT_DIR}/test_${DATASET_NAME}.csv"

if [[ ! -f "$TRAIN_CSV" ]]; then
    TRAIN_CSV="${SCRIPT_DIR}/Dataset/train_${DATASET_NAME}.csv"
    TEST_CSV="${SCRIPT_DIR}/Dataset/test_${DATASET_NAME}.csv"
fi

if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "ERROR: train_${DATASET_NAME}.csv not found."
    echo "  Copy the dataset CSV files to this machine before starting."
    exit 1
fi

export TRAIN_CSV TEST_CSV

echo "============================================================"
echo "  Flower SuperNode (Client) — Distributed Mode"
echo "  Dataset   : $DATASET_NAME"
echo "  Strategy  : $PARTITION_STRATEGY"
echo "  PrepType  : $PREP_TYPE"
echo "  Method    : $TRAIN_METHOD"
echo "  Seed      : $FLWR_SEED"
echo "  Server    : $SERVER_ADDRESS"
echo "  ClientAPI : $CLIENT_API_ADDRESS"
echo "============================================================"
echo ""
echo "NOTE: The ClientApp (client.py) will be loaded by the SuperNode."
echo "      Ensure all Python dependencies are installed on this machine:"
echo "      pip install 'flwr[simulation]>=1.9,<2.0' xgboost scikit-learn torch pandas numpy"
echo ""

# Launch the Flower SuperNode.
# --superlink        : address of the SuperLink (server machine)
# --clientappio-api-address : local port where the ClientApp runner connects
exec flower-supernode \
    --insecure \
    --superlink "${SERVER_ADDRESS}" \
    --clientappio-api-address "${CLIENT_API_ADDRESS}"
