"""
policyEnforcement_withPrivacy_AES256_XGB_Company.py
====================================================
XGBoost (SecureBoost / NVFlare) Policy Enforcement Server — Company dataset.

Mirrors policyEnforcement_withPrivacy_AES256_Company.py exactly, but uses
the federated XGBoost model trained by run_simulation.py instead of the ANN.

The XGBoost model is loaded from the path produced by server_eval.py:
  results/{dataset}/{PARTITION}/{ENCODING}/models/{CaseName}_Seed{seed}.json

The mapping pkl is saved by the patched prepare_data.py:
  results/{dataset}/{PARTITION}/{ENCODING}/Mapping/mapping_Company_{ALGO}_{PREP_NAME}.pkl

Client data (site-1 … site-4 train CSVs) are produced by prepare_data_split():
  results/{dataset}/{PARTITION}/{ENCODING}/data/site-{1-4}/train.csv

All PE features are identical to the ANN version:
  ✓ AES-256-GCM encryption on all peer communication
  ✓ FIFO PDPQueue per node
  ✓ Sequential ReqID  (next_req_id)
  ✓ Dedup via served_table
  ✓ Fault-tolerant retransmission with exponential back-off
  ✓ Parallel peer forwarding  (first-valid-response wins)
  ✓ /queue_status  /health  /check_resource  /access_resource  /access_request

Usage:
  python3.10 policyEnforcement_withPrivacy_AES256_XGB_Company.py \\
      --algo SBC --prep-type 2 --seed 0
  python3.10 policyEnforcement_withPrivacy_AES256_XGB_Company.py \\
      --algo PBP --prep-type 5 --seed 0 --results-root results

Prep-type → encoding name mapping (must match run_simulation.py):
  2 → ARFE       3 → AVC      4 → ARFE_AVC    5 → Naive_NACol

Partition mapping (must match run_simulation.py):
  SBC → SBP      PBP → PBP    RNP → RNP
"""

import os
import json
import queue
import threading
import time
import concurrent.futures
import base64
import hashlib
import pickle
import requests
from collections import defaultdict, Counter

import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# =====================================================
# CLI ARGUMENT PARSING
# =====================================================
import argparse as _ap

def _parse_args():
    p = _ap.ArgumentParser(add_help=False)
    p.add_argument("--algo",         default="SBC", choices=["SBC", "PBP", "RNP"])
    p.add_argument("--prep-type",    type=int, default=5, choices=[2, 3, 4, 5])
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--results-root", default="results",
                   help="Root folder written by run_simulation.py (default: results)")
    args, _ = p.parse_known_args()
    return args

_A = _parse_args()

# ── Naming tables (must match run_simulation.py) ──────────────────────────────
_PARTITION_CODES = {"SBC": "SBP", "PBP": "PBP", "RNP": "RNP"}
_ENCODING_NAMES  = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "Naive_NACol"}
_PREP_NAMES      = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}

_ALGO       = _A.algo
_PREP_TYPE  = _A.prep_type
_SEED       = _A.seed
_PREP_NAME  = _PREP_NAMES[_PREP_TYPE]        # used for mapping pkl filename
_PARTITION  = _PARTITION_CODES[_ALGO]        # used for folder path
_ENCODING   = _ENCODING_NAMES[_PREP_TYPE]   # used for folder path
_DATASET    = "company"
_DS_DISPLAY = "Company"

# case name exactly as server_eval.py builds it
_CASE_NAME  = f"{_DS_DISPLAY}_{_PARTITION}_{_ENCODING}"

# ── Paths ─────────────────────────────────────────────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))
_case_dir = os.path.join(_here, _A.results_root,
                          _DATASET, _PARTITION, _ENCODING)

# XGBoost model saved by server_eval.py
XGB_MODEL_PATH = os.path.join(_case_dir, "models",
                               f"{_CASE_NAME}_Seed{_SEED}.json")

# Mapping pkl saved by patched prepare_data.py
_MAPPING_PKL = os.path.join(_case_dir, "Mapping",
                              f"mapping_Company_{_ALGO}_{_PREP_NAME}.pkl")

# Client data directory (site-1 … site-4 train CSVs)
_CLIENT_DATA_ROOT = os.path.join(_case_dir, "data")

print(f"[CONFIG] ALGO={_ALGO}  PREP_TYPE={_PREP_TYPE} ({_PREP_NAME})  SEED={_SEED}")
print(f"[CONFIG] Case      : {_CASE_NAME}")
print(f"[CONFIG] Model     : {XGB_MODEL_PATH}")
print(f"[CONFIG] Mapping   : {_MAPPING_PKL}")
print(f"[CONFIG] ClientData: {_CLIENT_DATA_ROOT}")

pd.set_option('future.no_silent_downcasting', True)

# ── Client config (4 nodes, one per site) ─────────────────────────────────────
CLIENTS_CONFIG = [
    {"node_id": "node1", "port": 5000,
     "dataset": os.path.join(_CLIENT_DATA_ROOT, "site-1", "train.csv"),
     "resources": [],
     "peers": ["http://localhost:5001","http://localhost:5002","http://localhost:5003"]},
    {"node_id": "node2", "port": 5001,
     "dataset": os.path.join(_CLIENT_DATA_ROOT, "site-2", "train.csv"),
     "resources": [],
     "peers": ["http://localhost:5000","http://localhost:5002","http://localhost:5003"]},
    {"node_id": "node3", "port": 5002,
     "dataset": os.path.join(_CLIENT_DATA_ROOT, "site-3", "train.csv"),
     "resources": [],
     "peers": ["http://localhost:5000","http://localhost:5001","http://localhost:5003"]},
    {"node_id": "node4", "port": 5003,
     "dataset": os.path.join(_CLIENT_DATA_ROOT, "site-4", "train.csv"),
     "resources": [],
     "peers": ["http://localhost:5000","http://localhost:5001","http://localhost:5002"]},
]

# ── Retransmission parameters ─────────────────────────────────────────────────
PEER_CONNECT_TIMEOUT   = 5
RETRY_MAX_ATTEMPTS     = 3
RETRY_DELAY_SEC        = 2.0
CHECK_RESOURCE_TIMEOUT = 3
CHECK_RESOURCE_RETRIES = 2

# =====================================================
# AUTO-POPULATE CLIENT RESOURCES
# =====================================================
def _populate_resources():
    """Read site-N/train.csv and collect unique Resource values per node."""
    for i, cfg in enumerate(CLIENTS_CONFIG, 1):
        path = cfg["dataset"]
        if not os.path.exists(path):
            print(f"[WARN] Client {i} data not found: {path}"); continue
        try:
            df = pd.read_csv(path)
            # site CSV: first col = Access (label), rest = encoded features
            # Resource column not raw here, so fall back to unique encoded vals
            # The resource summary is not written for XGB — populate from raw data
            # Try to find Resource column among original train.csv instead
            raw_train = os.path.join(_case_dir, "train.csv")
            if os.path.exists(raw_train):
                raw_df = pd.read_csv(raw_train)
                if "Resource" in raw_df.columns:
                    cfg["resources"] = raw_df["Resource"].dropna().unique().tolist()
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")

_populate_resources()

# =====================================================
# SEQUENTIAL REQUEST ID
# =====================================================
_req_id_counter = 0
_req_id_lock    = threading.Lock()

def next_req_id() -> int:
    global _req_id_counter
    with _req_id_lock:
        _req_id_counter += 1
        return _req_id_counter

# =====================================================
# AES-256-GCM ENCRYPTION
# =====================================================
SHARED_KEY = hashlib.sha256(b"secure_shared_key").digest()

def encrypt_payload(payload: dict) -> bytes:
    nonce = os.urandom(12)
    ct    = AESGCM(SHARED_KEY).encrypt(nonce, json.dumps(payload).encode(), None)
    return base64.urlsafe_b64encode(nonce + ct)

def decrypt_payload(data: bytes) -> dict:
    raw       = base64.urlsafe_b64decode(data)
    nonce, ct = raw[:12], raw[12:]
    return json.loads(AESGCM(SHARED_KEY).decrypt(nonce, ct, None).decode())

# =====================================================
# XGBoost MODEL LOADING
# =====================================================
def load_xgb_model(model_path: str) -> xgb.Booster:
    """Load federated XGBoost model produced by server_eval.py."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"XGBoost model not found: {model_path}\n"
            f"  Run NVFlare simulation first:\n"
            f"  python3.10 run_simulation.py --datasets {_DATASET} "
            f"--strategies {_ALGO.lower()} --prep-types {_PREP_TYPE} --seeds {_SEED}"
        )
    bst = xgb.Booster()
    bst.load_model(model_path)
    print(f"[MODEL] XGBoost loaded from {model_path}")
    return bst

# =====================================================
# FEATURE ENCODING  (identical logic to Company ANN PE)
# =====================================================
_mapping_cache: dict = {}

def _load_mapping():
    global _mapping_cache
    if _mapping_cache: return
    if not os.path.exists(_MAPPING_PKL):
        raise FileNotFoundError(
            f"Mapping not found: {_MAPPING_PKL}\n"
            f"  Run NVFlare simulation (prepare_data_split saves this file)."
        )
    with open(_MAPPING_PKL, "rb") as f:
        _mapping_cache = pickle.load(f)
    print(f"[ENC] Mapping loaded from {_MAPPING_PKL} ({len(_mapping_cache)} entries)")


def _same_conditions(col1, col2):
    """Replica of same_conditions() in data_preprocessor.py."""
    if col1 == -1 or col2 == -1:
        return 2
    return 1 if col1 == col2 else 0


def encode_features_to_dataframe(features_dict: dict) -> pd.DataFrame:
    """
    Encode raw feature dict → a single-row DataFrame whose column order
    exactly matches the XGBoost training feature matrix.

    This replicates the logic of GetPreparedData() in data_preprocessor.py
    for every prep_type, and returns a DataFrame ready for xgb.DMatrix().

    Column order per prep_type (Company dataset):
      prep_type 2 (ARFE)     : DESIGNATION, Resource, sameProj, sameDep        (4 cols)
      prep_type 3 (AVC)      : DESIGNATION, Project_name, Department,
                                Resource, Project_Name, Department.1            (6 cols)
      prep_type 4 (ARFE+AVC) : DESIGNATION, Resource, sameProj, sameDep        (4 cols)
      prep_type 5 (NaiveNA)  : DESIGNATION, Project_name, Department,
                                Resource, Project_Name, Department.1, Proj_NA  (7 cols)
    """
    _load_mapping()

    def _u(v):
        return str(v).strip() if isinstance(v, str) else v

    def _enc(val) -> float:
        key = _u(val)
        m   = _mapping_cache.get(key, _mapping_cache.get(val, 0))
        return float(m) if not isinstance(m, str) else 0.0

    # Extract raw feature values — keep original case (mapping is case-sensitive)
    desig  = _u(features_dict.get("Designation",  "") or features_dict.get("DESIGNATION", ""))
    proj   = _u(features_dict.get("Project_name", ""))
    dept   = _u(features_dict.get("Department",   ""))
    res    = _u(features_dict.get("Resource",     ""))
    proj_n = _u(features_dict.get("Project_Name", ""))
    dept1  = _u(features_dict.get("Department.1", "") or features_dict.get("Department_1", ""))

    de = _enc(desig); pr = _enc(proj); dp = _enc(dept)
    rs = _enc(res);   pn = _enc(proj_n); d1 = _enc(dept1)

    if _PREP_TYPE == 5:        # NaiveNA — 7 cols
        proj_na = 1.0 if pr == -1.0 else 0.0
        row = {
            "DESIGNATION": de, "Project_name": pr, "Department": dp,
            "Resource":    rs, "Project_Name": pn, "Department.1": d1,
            "Proj_NA":     proj_na,
        }

    elif _PREP_TYPE == 3:      # AVC — 6 cols (group encoding, same column set)
        row = {
            "DESIGNATION": de, "Project_name": pr, "Department": dp,
            "Resource":    rs, "Project_Name": pn, "Department.1": d1,
        }

    elif _PREP_TYPE == 2:      # ARFE — 4 cols
        sp = _same_conditions(pr, pn)
        sd = _same_conditions(dp, d1)
        row = {"DESIGNATION": de, "Resource": rs, "sameProj": sp, "sameDep": sd}

    elif _PREP_TYPE == 4:      # ARFE+AVC — 4 cols
        sp = _same_conditions(pr, pn)
        sd = _same_conditions(dp, d1)
        row = {"DESIGNATION": de, "Resource": rs, "sameProj": sp, "sameDep": sd}

    else:                      # fallback: NaiveNA
        proj_na = 1.0 if pr == -1.0 else 0.0
        row = {
            "DESIGNATION": de, "Project_name": pr, "Department": dp,
            "Resource":    rs, "Project_Name": pn, "Department.1": d1,
            "Proj_NA":     proj_na,
        }

    df = pd.DataFrame([row])
    print(f"[ENC] prep_type={_PREP_TYPE}  cols={list(df.columns)}  "
          f"values={df.values.tolist()[0]}")
    return df

# =====================================================
# PDP — POLICY DECISION POINT
# =====================================================
def evaluate_access_policy(bst: xgb.Booster, features: dict) -> str:
    """
    Encode features and run XGBoost inference.
    Returns 'YES' (grant) or 'NO' (deny).
    """
    try:
        if isinstance(features, dict) and "features" in features:
            features = features["features"]

        df    = encode_features_to_dataframe(features)
        dtest = xgb.DMatrix(df)
        proba = float(bst.predict(dtest)[0])
        decision = "YES" if proba >= 0.5 else "NO"
        print(f"[XGB] score={proba:.4f} → {decision}")
    except Exception as e:
        print(f"[ERROR PDP] {e}")
        decision = "NO"
    return decision

# =====================================================
# PDP REQUEST QUEUE
# =====================================================
class PDPJob:
    __slots__ = ("req_id","node_id","user","resource","features",
                 "arrival_time","queue_position","done_event","result")

    def __init__(self, req_id, node_id, user, resource, features):
        self.req_id         = req_id
        self.node_id        = node_id
        self.user           = user
        self.resource       = resource
        self.features       = features
        self.arrival_time   = time.perf_counter()
        self.queue_position = None
        self.done_event     = threading.Event()
        self.result         = None


class PDPQueue:
    """Per-node FIFO queue with a single sequential worker thread."""

    def __init__(self, node_id, bst):
        self.node_id    = node_id
        self.bst        = bst
        self._q         = queue.Queue()
        self._counter   = 0
        self._lock      = threading.Lock()
        self._processed = 0
        self._worker    = threading.Thread(
            target=self._worker_loop,
            name=f"pdp-worker-{node_id}",
            daemon=True,
        )
        self._worker.start()
        print(f"[{node_id}][PDPQueue] Worker thread started")

    def submit(self, job: PDPJob, timeout: float = 30.0) -> PDPJob:
        with self._lock:
            self._counter     += 1
            job.queue_position = self._counter
            depth              = self._q.qsize()

        print(f"[{self.node_id}][PDPQueue] ENQUEUED  "
              f"ReqID={job.req_id}  user={job.user}  resource={job.resource}  "
              f"pos=#{job.queue_position}  depth={depth}")

        self._q.put(job)
        if not job.done_event.wait(timeout=timeout):
            job.result = {
                "decision":       "NO",
                "error":          f"PDP timeout after {timeout}s",
                "queue_position": job.queue_position,
                "queue_wait_ms":  round((time.perf_counter()-job.arrival_time)*1000, 2),
                "pdp_eval_ms":    0.0,
                "jobs_processed": self._processed,
            }
        return job

    def depth(self)           -> int:  return self._q.qsize()
    def total_processed(self) -> int:  return self._processed
    def worker_alive(self)    -> bool: return self._worker.is_alive()

    def _worker_loop(self):
        while True:
            job: PDPJob   = self._q.get()
            wait_ms       = round((time.perf_counter()-job.arrival_time)*1000, 2)
            t0            = time.perf_counter()

            print(f"[{self.node_id}][PDPQueue] PROCESSING "
                  f"ReqID={job.req_id}  pos=#{job.queue_position}  "
                  f"user={job.user}  wait={wait_ms}ms")

            try:
                decision = evaluate_access_policy(self.bst, {"features": job.features})
            except Exception as e:
                print(f"[{self.node_id}][PDPQueue] ERROR {job.req_id}: {e}")
                decision = "NO"

            eval_ms          = round((time.perf_counter()-t0)*1000, 2)
            self._processed += 1

            job.result = {
                "decision":       decision,
                "queue_position": job.queue_position,
                "queue_wait_ms":  wait_ms,
                "pdp_eval_ms":    eval_ms,
                "jobs_processed": self._processed,
            }

            print(f"[{self.node_id}][PDPQueue] COMPLETED "
                  f"ReqID={job.req_id}  → {decision}  eval={eval_ms}ms  "
                  f"total={self._processed}")

            job.done_event.set()
            self._q.task_done()

# =====================================================
# FAULT-TOLERANT PEER COMMUNICATION
# =====================================================
def send_with_retry(node_id, req_id, peer_url, encrypted_data,
                    endpoint="/access_resource"):
    retry_log = []
    delay     = RETRY_DELAY_SEC

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        t0 = time.perf_counter()
        if attempt > 1:
            print(f"[{node_id}][RETRY] ReqID={req_id}  attempt={attempt}  "
                  f"waiting {delay:.1f}s …")
            time.sleep(delay)
            delay *= 2
        try:
            resp       = requests.post(f"{peer_url}{endpoint}",
                                       data=encrypted_data,
                                       timeout=PEER_CONNECT_TIMEOUT)
            attempt_ms = round((time.perf_counter()-t0)*1000, 2)
            if resp.status_code == 200:
                body = resp.json()
                retry_log.append({"attempt": attempt, "outcome": "success",
                                   "attempt_ms": attempt_ms})
                return body, retry_log
            raise requests.exceptions.RequestException(f"HTTP {resp.status_code}")
        except Exception as exc:
            attempt_ms = round((time.perf_counter()-t0)*1000, 2)
            retry_log.append({"attempt": attempt, "outcome": "failure",
                               "error": str(exc), "attempt_ms": attempt_ms})
            print(f"[{node_id}][RETRY] attempt {attempt} FAILED: {exc}")

    return None, retry_log


def discover_peers(node_id, req_id, peers, encrypted_body):
    candidates = []
    for peer in peers:
        for attempt in range(1, CHECK_RESOURCE_RETRIES + 1):
            try:
                res = requests.post(f"{peer}/check_resource",
                                    data=encrypted_body,
                                    timeout=CHECK_RESOURCE_TIMEOUT)
                if res.status_code == 200 and res.json().get("has_resource"):
                    candidates.append(peer)
                break
            except Exception as exc:
                if attempt < CHECK_RESOURCE_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)
    return candidates

# =====================================================
# FORWARDING WITH RETRANSMISSION
# =====================================================
def forward_remote_request(node_cfg, request_payload, decision,
                           req_id, served_table, served_table_lock):
    start_time = time.perf_counter()
    node_id    = node_cfg["node_id"]
    resource   = request_payload.get("resource", "unknown")

    with served_table_lock:
        if req_id in served_table:
            return dict(served_table[req_id])

    encrypted_body = encrypt_payload({
        "resource":    resource.strip().upper(),
        "origin_node": node_id,
        "req_id":      req_id,
    })
    candidates = discover_peers(node_id, req_id, node_cfg["peers"], encrypted_body)

    if decision != "YES":
        result = {"status": "denied", "resource": resource}
        with served_table_lock: served_table[req_id] = result
        return result

    if not candidates:
        result = {"status": "granted", "resource": resource}
        with served_table_lock: served_table[req_id] = result
        return result

    fwd_payload           = dict(request_payload)
    fwd_payload["req_id"] = req_id
    encrypted_request     = encrypt_payload(fwd_payload)

    final_result = None
    result_lock  = threading.Lock()
    result_ready = threading.Event()

    def _send(peer_url):
        nonlocal final_result
        peer_node = next((c["node_id"] for c in CLIENTS_CONFIG
                          if peer_url.endswith(str(c["port"]))), "?")
        body, _ = send_with_retry(node_id, req_id, peer_url, encrypted_request)
        if body and body.get("status") == "granted":
            with result_lock:
                if final_result is None:
                    ms = round((time.perf_counter()-start_time)*1000, 2)
                    final_result = {
                        "status":     "granted_remote",
                        "owner":      peer_node,
                        "resource":   resource,
                        "latency_ms": ms,
                    }
                    result_ready.set()

    max_wait = (PEER_CONNECT_TIMEOUT * RETRY_MAX_ATTEMPTS +
                RETRY_DELAY_SEC * (2 ** RETRY_MAX_ATTEMPTS))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        futures = [ex.submit(_send, p) for p in candidates]
        result_ready.wait(timeout=max_wait)
        for f in futures: f.cancel()

    result = final_result or {"status": "denied", "resource": resource}
    with served_table_lock: served_table[req_id] = result
    return result

# =====================================================
# FLASK APP
# =====================================================
def create_app(node_cfg, bst, local_resources) -> Flask:
    app               = Flask(node_cfg["node_id"])
    pdp_queue         = PDPQueue(node_cfg["node_id"], bst)
    served_table      = {}
    served_table_lock = threading.Lock()

    @app.route("/access_request", methods=["POST"])
    def handle_access_request():
        req_id  = next_req_id()
        arrival = time.perf_counter()
        data    = request.get_json(force=True, silent=True) or {}
        user     = data.get("user",     "anonymous")
        resource = data.get("resource", "")
        features = data.get("features", {})

        print(f"[{node_cfg['node_id']}][PEP] "
              f"ReqID={req_id}  user={user}  resource={resource}  "
              f"prep={_PREP_TYPE}({_PREP_NAME})")

        with served_table_lock:
            if req_id in served_table:
                cached = dict(served_table[req_id])
                cached["req_id"]           = req_id
                cached["cached"]           = True
                cached["total_latency_ms"] = round(
                    (time.perf_counter()-arrival)*1000, 2)
                return jsonify(cached)

        job = PDPJob(req_id, node_cfg["node_id"], user, resource, features)
        job = pdp_queue.submit(job, timeout=30.0)
        res = job.result
        decision = res["decision"]

        if decision.upper() == "YES":
            total_ms = round((time.perf_counter()-arrival)*1000, 2)
            result = {
                "status":           "granted",
                "resource":         resource.upper(),
                "owner":            node_cfg["node_id"],
                "req_id":           req_id,
                "queue_position":   res["queue_position"],
                "queue_wait_ms":    res["queue_wait_ms"],
                "pdp_eval_ms":      res["pdp_eval_ms"],
                "total_latency_ms": total_ms,
            }
            with served_table_lock: served_table[req_id] = result
            return jsonify(result)

        result = forward_remote_request(
            node_cfg, data, decision, req_id, served_table, served_table_lock
        )
        result["req_id"]           = req_id
        result["queue_position"]   = res["queue_position"]
        result["queue_wait_ms"]    = res["queue_wait_ms"]
        result["pdp_eval_ms"]      = res["pdp_eval_ms"]
        result["total_latency_ms"] = round((time.perf_counter()-arrival)*1000, 2)
        return jsonify(result)

    @app.route("/queue_status", methods=["GET"])
    def queue_status():
        with served_table_lock:
            served_count = len(served_table)
        return jsonify({
            "node_id":          node_cfg["node_id"],
            "model":            "XGBoost",
            "dataset":          _DS_DISPLAY,
            "algo":             _ALGO,
            "prep_type":        _PREP_TYPE,
            "prep_name":        _PREP_NAME,
            "seed":             _SEED,
            "queue_depth":      pdp_queue.depth(),
            "total_processed":  pdp_queue.total_processed(),
            "worker_alive":     pdp_queue.worker_alive(),
            "served_req_count": served_count,
        })

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "node":      node_cfg["node_id"],
            "status":    "running",
            "model":     "XGBoost",
            "dataset":   _DS_DISPLAY,
            "algo":      _ALGO,
            "prep_type": _PREP_TYPE,
            "prep_name": _PREP_NAME,
            "seed":      _SEED,
            "resources": local_resources,
        })

    @app.route("/check_resource", methods=["POST"])
    def check_resource():
        try:
            d = decrypt_payload(request.data)
            r = d.get("resource", "").strip().upper()
            return jsonify({"has_resource": any(
                r == x.strip().upper() for x in local_resources
            )})
        except Exception:
            return jsonify({"has_resource": False})

    @app.route("/access_resource", methods=["POST"])
    def access_resource():
        try:
            d         = decrypt_payload(request.data)
            resource  = d.get("resource")
            req_id_in = d.get("req_id")
            if req_id_in is not None:
                with served_table_lock:
                    if req_id_in in served_table:
                        return jsonify({"status": "already_served",
                                        "req_id": req_id_in})
                    served_table[req_id_in] = {
                        "status":    "granted",
                        "served_by": node_cfg["node_id"],
                        "resource":  resource,
                    }
            return jsonify({"status": "granted", "req_id": req_id_in})
        except Exception as e:
            print(f"[{node_cfg['node_id']}][ERROR] /access_resource: {e}")
            return jsonify({"status": "denied"})

    return app

# =====================================================
# RESOURCE OWNERSHIP INFERENCE
# =====================================================
def infer_resource_ownership(client_configs: list) -> dict:
    owners = defaultdict(str)
    print("\n=== RESOURCE OWNERSHIP (Company: Resource column) ===")
    raw_train = os.path.join(_case_dir, "train.csv")
    if os.path.exists(raw_train):
        df = pd.read_csv(raw_train)
        if "Resource" in df.columns:
            for res in df["Resource"].dropna().unique():
                owners[str(res).upper()] = "global"
    for cfg in client_configs:
        path = cfg["dataset"]
        if not os.path.exists(path): continue
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().upper() for c in df.columns]
            # site CSVs have encoded Resource (first non-label col varies)
            # Resources will be inferred from raw train instead
        except Exception as e:
            print(f"  [WARN] {cfg['node_id']}: {e}")
    print(f"  {len(owners)} unique resources inferred")
    return owners

# =====================================================
# NODE STARTUP
# =====================================================
def start_node(cfg, bst):
    print(f"[{cfg['node_id']}] port={cfg['port']}  "
          f"resources={cfg['resources'][:5]}{'...' if len(cfg['resources'])>5 else ''}")
    app = create_app(cfg, bst, cfg["resources"])
    app.run(host="0.0.0.0", port=cfg["port"],
            debug=False, use_reloader=False, threaded=True)

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    import argparse as _main_ap
    _p = _main_ap.ArgumentParser(
        description="XGBoost SecureBoost Policy Enforcement Server — Company dataset"
    )
    _p.add_argument("--algo",         default=_ALGO)
    _p.add_argument("--prep-type",    type=int, default=_PREP_TYPE)
    _p.add_argument("--seed",         type=int, default=_SEED)
    _p.add_argument("--results-root", default=_A.results_root)
    _args = _p.parse_args()

    print(f"\n[START] XGBoost PE — Company  "
          f"ALGO={_ALGO}  PREP_TYPE={_PREP_TYPE}({_PREP_NAME})  "
          f"SEED={_SEED}  case={_CASE_NAME}\n")

    # Load model once — shared read-only across all 4 nodes
    bst = load_xgb_model(XGB_MODEL_PATH)

    # Infer resource ownership from raw training data
    RESOURCE_OWNERSHIP = infer_resource_ownership(CLIENTS_CONFIG)

    threads = []
    for cfg in CLIENTS_CONFIG:
        t = threading.Thread(target=start_node, args=(cfg, bst))
        t.daemon = True
        t.start()
        threads.append(t)

    print(f"\n[MAIN] 4 nodes started (5000–5003)  "
          f"model=XGBoost  algo={_ALGO}  prep={_PREP_TYPE}({_PREP_NAME})  "
          f"seed={_SEED}\nCtrl-C to stop.\n")
    try:
        for t in threads: t.join()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutdown.")
