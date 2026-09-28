#!/bin/bash
# run_all_company_cases.sh
# ========================
# Runs all 12 Company PE cases sequentially.
# Each case opens TWO visible terminal windows:
#
#   Terminal 1 — PE server   (stays open until test is done)
#   Terminal 2 — Test runner (closes when test finishes)
#
# Lifecycle per case:
#   1. Open Terminal 1  → PE server
#   2. Wait until all 4 nodes respond on /queue_status
#   3. Open Terminal 2  → Test runner
#   4. Wait for Terminal 2 to finish (rc-file handshake)
#   5. Kill Terminal 2, then kill Terminal 1
#   6. Free ports 5000-5003
#   7. Pause 3s → next case
#
# Usage:
#   chmod +x run_all_company_cases.sh
#   ./run_all_company_cases.sh
#   ./run_all_company_cases.sh 2>&1 | tee run_all_company_cases.log
#
# Requirements:
#   gnome-terminal  (Ubuntu/GNOME default)
#   wmctrl          → sudo apt install wmctrl   (for closing windows by name)
#   curl            → sudo apt install curl

PYTHON=python3.10
PE_SCRIPT=policyEnforcement_withPrivacy_AES256_XGB_University1.py
ATR_SCRIPT=auto_test_runner_XGB_University1.py
READY_WAIT=90      # max seconds to wait for all 4 nodes
PAUSE_BETWEEN=3    # seconds between cases

# ── Helper: wait for all 4 nodes via /queue_status ───────────────────────────
wait_for_nodes() {
    local deadline=$(( $(date +%s) + READY_WAIT ))
    echo "[$(date +%T)] Waiting for nodes on ports 5000-5003 (max ${READY_WAIT}s)..."
    while [ $(date +%s) -lt $deadline ]; do
        all_up=true
        for port in 5000 5001 5002 5003; do
            if ! curl -sf "http://localhost:${port}/queue_status" > /dev/null 2>&1; then
                all_up=false
                break
            fi
        done
        if $all_up; then
            echo "[$(date +%T)] All 4 nodes are ready."
            return 0
        fi
        sleep 2
    done
    echo "[$(date +%T)] ERROR: Nodes not ready within ${READY_WAIT}s."
    return 1
}

# ── Helper: close a terminal window by its --class name (needs wmctrl) ───────
close_window() {
    local CLASS="$1"
    wmctrl -x -c "$CLASS" 2>/dev/null || true
    sleep 1
}

# ── Helper: kill a PID and wait for it to exit ───────────────────────────────
kill_pid() {
    local PID=$1
    local NAME=$2
    if kill -0 "$PID" 2>/dev/null; then
        echo "[$(date +%T)] Stopping $NAME (pid=$PID)..."
        kill -TERM "$PID" 2>/dev/null
        sleep 4
        kill -0 "$PID" 2>/dev/null && kill -KILL "$PID" 2>/dev/null || true
        echo "[$(date +%T)] $NAME stopped."
    fi
}

# ── Helper: kill any process still holding ports 5000-5003 ───────────────────
free_ports() {
    for port in 5000 5001 5002 5003; do
        local pid
        pid=$(lsof -ti tcp:$port 2>/dev/null)
        if [ -n "$pid" ]; then
            kill -9 "$pid" 2>/dev/null && \
                echo "[$(date +%T)] Freed port $port (pid=$pid)" || true
        fi
    done
}

# ── Core: run one case in two visible terminal windows ───────────────────────
run_case() {
    local ALGO=$1
    local PREP=$2
    declare -A PMAP=([2]="ARFE" [3]="AVC" [4]="ARFE_AVC" [5]="NaiveNA")
    local PNAME="${PMAP[$PREP]:-Prep$PREP}"

    local PE_CMD="$PYTHON $PE_SCRIPT --algo $ALGO --prep-type $PREP"
    local ATR_CMD="$PYTHON $ATR_SCRIPT --algo $ALGO --prep-type $PREP --all-nodes"

    # Unique WM class names so wmctrl can target exactly this case's windows
    local PE_CLASS="PE_${ALGO}_${PREP}"
    local ATR_CLASS="ATR_${ALGO}_${PREP}"

    # RC handshake: test runner writes its exit code here when done
    local RC_FILE="/tmp/atr_rc_${ALGO}_${PREP}_$$.txt"
    local ATR_CMD_WRAPPED="$ATR_CMD; echo \$? > '$RC_FILE'"

    mkdir -p logs
    local PE_LOG="logs/pe_${ALGO}_${PREP}.log"
    local ATR_LOG="logs/atr_${ALGO}_${PREP}.log"

    echo ""
    echo "======================================================"
    echo "  CASE: ${ALGO}  prep-type=${PREP} (${PNAME})"
    echo "======================================================"
    echo "[$(date +%T)] Terminal 1 → $PE_CMD"
    echo "[$(date +%T)] Terminal 2 → $ATR_CMD"

    # ── Open Terminal 1: PE server (stays open) ───────────────────────────────
    echo "[$(date +%T)] Opening Terminal 1 (PE server)..."
    gnome-terminal \
        --title="[T1] PE  ${ALGO}  prep=${PREP} (${PNAME})" \
        --class="$PE_CLASS" \
        --working-directory="$HOME/Project/nvflare_xgb_simulation_PE" \
        -- bash -c "$PE_CMD 2>&1 | tee '$PE_LOG'; echo '[PE SERVER EXITED]'; exec bash" &
    local T1_PID=$!
    echo "[$(date +%T)] Terminal 1 opened (pid=$T1_PID)"

    # ── Wait until all 4 nodes are ready ─────────────────────────────────────
    if ! wait_for_nodes; then
        echo "[$(date +%T)] Nodes not ready — aborting case and closing Terminal 1."
        close_window "$PE_CLASS"
        kill_pid "$T1_PID" "Terminal 1"
        free_ports
        return 1
    fi

    # ── Open Terminal 2: test runner ──────────────────────────────────────────
    echo "[$(date +%T)] Opening Terminal 2 (test runner)..."
    gnome-terminal \
        --title="[T2] ATR  ${ALGO}  prep=${PREP} (${PNAME})" \
        --class="$ATR_CLASS" \
        --working-directory="$HOME/Project/nvflare_xgb_simulation_PE" \
        -- bash -c "$ATR_CMD_WRAPPED 2>&1 | tee '$ATR_LOG'; echo '[TEST RUNNER DONE]'; exec bash" &
    local T2_PID=$!
    echo "[$(date +%T)] Terminal 2 opened (pid=$T2_PID)"

    # ── Wait for Terminal 2 to finish (poll for RC file) ─────────────────────
    echo "[$(date +%T)] Waiting for test runner to finish..."
    local wait_deadline=$(( $(date +%s) + 3600 ))   # max 1 hour per case
    while [ $(date +%s) -lt $wait_deadline ]; do
        [ -f "$RC_FILE" ] && break
        sleep 3
    done

    local ATR_RC=1
    if [ -f "$RC_FILE" ]; then
        ATR_RC=$(cat "$RC_FILE" | tr -d '[:space:]')
        rm -f "$RC_FILE"
        echo "[$(date +%T)] Test runner finished (rc=$ATR_RC)"
    else
        echo "[$(date +%T)] WARNING: Test runner timed out — RC file not found."
    fi

    # ── Close Terminal 2 (test runner) first ─────────────────────────────────
    echo "[$(date +%T)] Closing Terminal 2 (test runner)..."
    close_window "$ATR_CLASS"
    kill_pid "$T2_PID" "Terminal 2"

    # ── Close Terminal 1 (PE server) ─────────────────────────────────────────
    echo "[$(date +%T)] Closing Terminal 1 (PE server)..."
    close_window "$PE_CLASS"
    kill_pid "$T1_PID" "Terminal 1"

    # ── Kill any remaining Flask processes on 5000-5003 ──────────────────────
    free_ports

    sleep $PAUSE_BETWEEN
    return $ATR_RC
}

# ============================================================
# MAIN — ALL 12 CASES
# ============================================================
PASS=0; FAIL=0

echo "======================================================"
echo "  Company PE — 12 cases  (2 visible terminals each)"
echo "  PE  script : $PE_SCRIPT"
echo "  ATR script : $ATR_SCRIPT"
echo "======================================================"
echo "  Case  1  →  SBC  ARFE       (prep=2)"
echo "  Case  2  →  SBC  AVC        (prep=3)"
echo "  Case  3  →  SBC  ARFE_AVC   (prep=4)"
echo "  Case  4  →  SBC  NaiveNA    (prep=5)"
echo "  Case  5  →  PBP  ARFE       (prep=2)"
echo "  Case  6  →  PBP  AVC        (prep=3)"
echo "  Case  7  →  PBP  ARFE_AVC   (prep=4)"
echo "  Case  8  →  PBP  NaiveNA    (prep=5)"
echo "  Case  9  →  RNP  ARFE       (prep=2)"
echo "  Case 10  →  RNP  AVC        (prep=3)"
echo "  Case 11  →  RNP  ARFE_AVC   (prep=4)"
echo "  Case 12  →  RNP  NaiveNA    (prep=5)"
echo ""
echo "[$(date +%T)] Starting..."
echo ""

run_case SBC 2 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case SBC 3 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case SBC 4 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case SBC 5 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case PBP 2 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case PBP 3 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case PBP 4 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case PBP 5 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case RNP 2 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case RNP 3 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case RNP 4 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
run_case RNP 5 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo "======================================================"
echo "  ALL 12 CASES DONE"
echo "  Passed : $PASS"
echo "  Failed : $FAIL"
echo "======================================================"
echo ""
echo "[$(date +%T)] Consolidating all results..."
$PYTHON $ATR_SCRIPT --consolidate

[ $FAIL -eq 0 ] && exit 0 || exit 1
