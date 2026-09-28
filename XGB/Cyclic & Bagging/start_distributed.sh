#!/usr/bin/env bash
# start_distributed.sh  –  Test multi-process mode on a SINGLE machine.
# Launches: 1 SuperLink + 1 ServerApp + N SuperNodes (each in own process).
# This simulates the true distributed architecture without needing N machines.
#
# For REAL multi-machine deployment use start_server.sh + start_client.sh.
#
# Usage:
#   ./start_distributed.sh [--clients N] [--dataset DS] ...
#
# Example:
#   DATASET_NAME=company PARTITION_STRATEGY=sbp PREP_TYPE=2 ./start_distributed.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_CLIENTS="${NUM_CLIENTS:-4}"
export DATASET_NAME="${DATASET_NAME:-company}"
export PARTITION_STRATEGY="${PARTITION_STRATEGY:-sbp}"
export PREP_TYPE="${PREP_TYPE:-2}"
export TRAIN_METHOD="${TRAIN_METHOD:-bagging}"
export FLWR_SEED="${FLWR_SEED:-0}"
export PYTHONHASHSEED="${FLWR_SEED}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"

SUPERLINK_ADDR="127.0.0.1:9092"
CLIENTAPPIO_BASE=9094   # each SuperNode gets base+i port

# Resolve CSVs
TRAIN_CSV="${SCRIPT_DIR}/train_${DATASET_NAME}.csv"
TEST_CSV="${SCRIPT_DIR}/test_${DATASET_NAME}.csv"
if [[ ! -f "$TRAIN_CSV" ]]; then
    TRAIN_CSV="${SCRIPT_DIR}/Dataset/train_${DATASET_NAME}.csv"
    TEST_CSV="${SCRIPT_DIR}/Dataset/test_${DATASET_NAME}.csv"
fi
export TRAIN_CSV TEST_CSV
mkdir -p "$OUTPUT_DIR"

PREP_NAMES=([2]="ARFE" [3]="AVC" [4]="ARFE_AVC" [5]="NaiveNA")
export CASE_NAME="${PREP_NAMES[$PREP_TYPE]:-PrepType${PREP_TYPE}}"
export SERVER_EVAL_LOG="${OUTPUT_DIR}/${DATASET_NAME}_${PARTITION_STRATEGY}_${CASE_NAME}_seed${FLWR_SEED}.log"

echo "============================================================"
echo "  Multi-process Distributed Test"
echo "  Dataset=$DATASET_NAME  Strategy=$PARTITION_STRATEGY  Prep=$PREP_TYPE"
echo "  Method=$TRAIN_METHOD  Seed=$FLWR_SEED  Clients=$NUM_CLIENTS"
echo "============================================================"

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down all processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    echo "Done."
}
trap cleanup EXIT INT TERM

# 1) Start SuperLink (relay between server and client processes)
echo "[1/3] Starting SuperLink on $SUPERLINK_ADDR ..."
flower-superlink --insecure \
    --fleet-api-address "$SUPERLINK_ADDR" \
    &>> "${OUTPUT_DIR}/superlink.log" &
PIDS+=($!)
sleep 2   # give SuperLink time to bind

# 2) Start ServerApp
echo "[2/3] Starting ServerApp ..."
flower-server-app server:app \
    --superlink "$SUPERLINK_ADDR" \
    --insecure \
    &>> "${OUTPUT_DIR}/server.log" &
PIDS+=($!)
sleep 2

# 3) Start N SuperNodes (one per logical client)
echo "[3/3] Starting $NUM_CLIENTS SuperNodes ..."
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    PORT=$((CLIENTAPPIO_BASE + i))
    flower-supernode \
        --insecure \
        --superlink "$SUPERLINK_ADDR" \
        --clientappio-api-address "0.0.0.0:${PORT}" \
        &>> "${OUTPUT_DIR}/client_${i}.log" &
    PIDS+=($!)
    echo "  SuperNode $i → port $PORT (PID $!)"
done

echo ""
echo "All processes started. Waiting for experiment to finish..."
echo "Logs: $OUTPUT_DIR/"
echo "Press Ctrl+C to stop."

wait "${PIDS[0]}"   # wait for SuperLink (proxy for entire run)
