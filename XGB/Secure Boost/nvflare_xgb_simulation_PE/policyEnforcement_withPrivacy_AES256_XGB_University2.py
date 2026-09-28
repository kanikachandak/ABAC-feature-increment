"""
policyEnforcement_withPrivacy_AES256_XGB_U2.py
===============================================
XGBoost (SecureBoost / NVFlare) Policy Enforcement Server — University_2 dataset.

Mirrors policyEnforcement_withPrivacy_AES256_U2.py exactly, but uses the
federated XGBoost model trained by run_simulation.py instead of the ANN.

University_2 dataset columns:
  Designation, Post, Course, Department, Degree, Year,
  Type (= resource), Department.1, Course.1, Degree.1, Year.1, Access

Feature vector per prep_type (matches GetPreparedData in U2 data_preprocessor.py):
  prep_type 2 (ARFE)     : Designation, Post, Type,
                            sameCourse, sameDep, sameDeg, sameYr                -> 7
  prep_type 3 (AVC)      : all 11 raw cols (group-encoded)                      -> 11
  prep_type 4 (ARFE+AVC) : Designation, Post, Type,
                            sameCourse, sameDep, sameDeg, sameYr                -> 7
  prep_type 5 (NaiveNA)  : all 11 raw + Post_NA, Course_NA, Degree_NA,
                            Year_NA, Course.1_NA, Year.1_NA                     -> 17

Artefact paths (produced by run_simulation.py + patched prepare_data.py):
  results/university2/{PARTITION}/{ENCODING}/models/{CaseName}_Seed{seed}.json
  results/university2/{PARTITION}/{ENCODING}/Mapping/mapping_U2_{ALGO}_{PREP_NAME}.pkl
  results/university2/{PARTITION}/{ENCODING}/TestData/raw_test_data_{seed}.csv
  results/university2/{PARTITION}/{ENCODING}/data/site-{1-4}/train.csv

All PE infrastructure identical to XGB_Company version:
  AES-256-GCM  |  FIFO PDPQueue  |  Sequential ReqID  |  Dedup  |
  Retransmission  |  Parallel forwarding  |  /queue_status /health

Usage:
  python3.10 policyEnforcement_withPrivacy_AES256_XGB_U2.py --algo SBC --prep-type 5 --seed 0
  python3.10 policyEnforcement_withPrivacy_AES256_XGB_U2.py --algo PBP --prep-type 2 --seed 0

FIX NOTES (why NVFlare SecureBoost metrics now match Policy Enforcement):
  Root cause: GetAttributeMapping() calls data[col].unique() which returns NaN as
  a legitimate unique value. The condition `if val != "NotA"` is True for NaN, so
  NaN is added to the mapping as float('nan') → <sequential_int>.
  data.replace(mapping) then replaces NaN cells with that integer (pandas handles
  NaN keys in replace() correctly via pd.isna() checks internally).
  XGBoost trains treating NaN-origin cells as ordinary integers, NOT as missing.

  The PE server's _MAPPING_CI construction stored float('nan') as a dict key, but
  Python dict lookup for NaN always fails (nan != nan → equality check fails after
  hash match) so _enc() fell through to the default of 0 — sending the wrong value
  to XGBoost at inference time.

  FIX: Replace the float('nan') key with a string sentinel _NAN_SENTINEL in
  _MAPPING_CI so it can be reliably looked up. _enc() now detects NaN inputs
  (both actual float-NaN and the "nan" string produced by str(pd.NA)) and maps
  them to the same integer the training pipeline used.
"""

import math
import os, json, queue, threading, time, concurrent.futures
import base64, hashlib, pickle, requests
from collections import defaultdict
from functools import lru_cache

import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# =====================================================
# CLI / NAMING
# =====================================================
import argparse as _ap

def _parse_args():
    p = _ap.ArgumentParser(add_help=False)
    p.add_argument("--algo",         default="SBC", choices=["SBC","PBP","RNP"])
    p.add_argument("--prep-type",    type=int, default=5, choices=[2,3,4,5])
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--results-root", default="results")
    return p.parse_known_args()[0]

_A = _parse_args()

_PARTITION_CODES = {"SBC":"SBP","PBP":"PBP","RNP":"RNP"}
_ENCODING_NAMES  = {2:"ARFE",3:"AVC",4:"ARFE_AVC",5:"Naive_NACol"}
_PREP_NAMES      = {2:"ARFE",3:"AVC",4:"ARFE_AVC",5:"NaiveNA"}

_ALGO       = _A.algo
_PREP_TYPE  = _A.prep_type
_SEED       = _A.seed
_PREP_NAME  = _PREP_NAMES[_PREP_TYPE]
_PARTITION  = _PARTITION_CODES[_ALGO]
_ENCODING   = _ENCODING_NAMES[_PREP_TYPE]
_DATASET    = "university2"
_DS_DISPLAY = "University2"
_CASE_NAME  = f"{_DS_DISPLAY}_{_PARTITION}_{_ENCODING}"

_here     = os.path.dirname(os.path.abspath(__file__))
_case_dir = os.path.join(_here, _A.results_root, _DATASET, _PARTITION, _ENCODING)

XGB_MODEL_PATH = os.path.join(_case_dir, "models", f"{_CASE_NAME}_Seed{_SEED}.json")
_MAPPING_PKL   = os.path.join(_case_dir, "Mapping", f"mapping_U2_{_ALGO}_{_PREP_NAME}.pkl")
_CLIENT_DATA_ROOT = os.path.join(_case_dir, "data")

print(f"[CONFIG] DATASET={_DS_DISPLAY}  ALGO={_ALGO}  PREP_TYPE={_PREP_TYPE}({_PREP_NAME})  SEED={_SEED}")
print(f"[CONFIG] Case      : {_CASE_NAME}")
print(f"[CONFIG] Model     : {XGB_MODEL_PATH}")
print(f"[CONFIG] Mapping   : {_MAPPING_PKL}")
print(f"[CONFIG] ClientData: {_CLIENT_DATA_ROOT}")

pd.set_option('future.no_silent_downcasting', True)

CLIENTS_CONFIG = [
    {"node_id":"node1","port":5000,
     "dataset":os.path.join(_CLIENT_DATA_ROOT,"site-1","train.csv"),"resources":[],
     "peers":["http://localhost:5001","http://localhost:5002","http://localhost:5003"]},
    {"node_id":"node2","port":5001,
     "dataset":os.path.join(_CLIENT_DATA_ROOT,"site-2","train.csv"),"resources":[],
     "peers":["http://localhost:5000","http://localhost:5002","http://localhost:5003"]},
    {"node_id":"node3","port":5002,
     "dataset":os.path.join(_CLIENT_DATA_ROOT,"site-3","train.csv"),"resources":[],
     "peers":["http://localhost:5000","http://localhost:5001","http://localhost:5003"]},
    {"node_id":"node4","port":5003,
     "dataset":os.path.join(_CLIENT_DATA_ROOT,"site-4","train.csv"),"resources":[],
     "peers":["http://localhost:5000","http://localhost:5001","http://localhost:5002"]},
]

PEER_CONNECT_TIMEOUT   = 5
RETRY_MAX_ATTEMPTS     = 3
RETRY_DELAY_SEC        = 2.0
CHECK_RESOURCE_TIMEOUT = 3
CHECK_RESOURCE_RETRIES = 2

# =====================================================
# RESOURCES  (Type = resource in U2)
# =====================================================
def _populate_resources():
    raw_train = os.path.join(_case_dir, "train.csv")
    if os.path.exists(raw_train):
        try:
            df = pd.read_csv(raw_train)
            if "Type" in df.columns:
                res = df["Type"].dropna().unique().tolist()
                for cfg in CLIENTS_CONFIG:
                    cfg["resources"] = res
                print(f"[INFO] {len(res)} unique Type values as resources")
                return
        except Exception as e:
            print(f"[WARN] {e}")

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
# AES-256-GCM
# =====================================================
SHARED_KEY = hashlib.sha256(b"secure_shared_key").digest()

def encrypt_payload(payload: dict) -> bytes:
    nonce = os.urandom(12)
    ct    = AESGCM(SHARED_KEY).encrypt(nonce, json.dumps(payload).encode(), None)
    return base64.urlsafe_b64encode(nonce + ct)

def decrypt_payload(data: bytes) -> dict:
    raw = base64.urlsafe_b64decode(data)
    return json.loads(AESGCM(SHARED_KEY).decrypt(raw[:12], raw[12:], None).decode())

# =====================================================
# XGBoost MODEL
# =====================================================
def load_xgb_model(model_path: str) -> xgb.Booster:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"  Run: python3.10 run_simulation.py --datasets university2 "
            f"--strategies {_ALGO} --prep-types {_PREP_TYPE} --seeds {_SEED}")
    bst = xgb.Booster()
    bst.load_model(model_path)
    print(f"[MODEL] XGBoost loaded from {model_path}")
    return bst

# =====================================================
# FEATURE ENCODING
# =====================================================
# Sentinel key used in _MAPPING_CI to store the mapping for NaN values.
#
# WHY: GetAttributeMapping() does col_un = data[col].unique() which includes
# actual NaN rows, and since (NaN != "NotA") is True, NaN gets added to the
# mapping as mapping[float('nan')] = <int>.  data.replace(mapping) then
# correctly swaps NaN cells for that integer (pandas handles NaN keys in
# replace() via pd.isna() internally).
#
# The old _MAPPING_CI kept float('nan') as a dict key, but dict lookup for
# NaN ALWAYS fails in Python because (nan != nan) makes the equality check
# fail after the hash match.  The fix: on load, replace the float('nan') key
# with this sentinel string so it can be retrieved reliably.
_NAN_SENTINEL = "__nan__"

_MAPPING_CACHE: dict = {}
_MAPPING_CI:   dict = {}

def _load_mapping():
    global _MAPPING_CACHE, _MAPPING_CI
    if _MAPPING_CACHE:
        return
    if not os.path.exists(_MAPPING_PKL):
        raise FileNotFoundError(f"Mapping not found: {_MAPPING_PKL}")
    with open(_MAPPING_PKL, "rb") as f:
        _MAPPING_CACHE = pickle.load(f)

    _MAPPING_CI = {}
    nan_count = 0
    for k, v in _MAPPING_CACHE.items():
        # ── FIX: float('nan') key cannot be looked up via dict.get() because
        # nan != nan.  Store it under a reliable string sentinel instead.
        if isinstance(k, float) and math.isnan(k):
            _MAPPING_CI[_NAN_SENTINEL] = v
            nan_count += 1
        elif isinstance(k, str):
            _MAPPING_CI[k.upper()] = v
        else:
            _MAPPING_CI[k] = v

    print(f"[ENC] Mapping loaded: {_MAPPING_PKL}  "
          f"({len(_MAPPING_CACHE)} entries, {nan_count} NaN key(s) re-mapped to sentinel)")


def _enc(v) -> float:
    """
    Encode a single raw feature value using the saved mapping pickle.

    Replicates data.replace(mapping) from GetPreparedData() for scalar values:
      - "NotA" (explicit missing sentinel) → -1.0
      - NaN (float NaN or the string "nan" from JSON transport)
            → the sequential integer assigned to NaN during training
              (same value data.replace() would have substituted)
      - Known value  → its mapped integer
      - Truly unknown value (not in training data at all) → float('nan')
            so XGBoost applies its learned default direction (matches training
            behaviour where unseen strings stay as NaN in the DMatrix)
    """
    _load_mapping()

    # ── FIX: detect NaN input ─────────────────────────────────────────────
    # Path 1: caller passes an actual float NaN (e.g. from pd.isna() branch)
    if isinstance(v, float) and math.isnan(v):
        nan_enc = _MAPPING_CI.get(_NAN_SENTINEL)
        return float(nan_enc) if nan_enc is not None else float('nan')

    # Path 2: NaN was serialised to the string "nan" / "NaN" during JSON
    # transport (str(pd.NA) == "nan", str(float('nan')) == "nan")
    if isinstance(v, str) and v.strip().upper() == 'NAN':
        nan_enc = _MAPPING_CI.get(_NAN_SENTINEL)
        return float(nan_enc) if nan_enc is not None else float('nan')
    # ── end NaN fix ───────────────────────────────────────────────────────

    key = str(v).strip().upper() if isinstance(v, str) else v
    m   = _MAPPING_CI.get(key, _MAPPING_CI.get(v, None))

    if m is None:
        # Value was not seen during training: let XGBoost handle as missing.
        return float('nan')

    return float(m) if not isinstance(m, str) else float('nan')


def _same_cond(v1, v2) -> float:
    """
    Replicates same_conditions() from data_preprocessor.py on encoded values.
    Works correctly with float('nan') inputs (nan != -1.0 and nan != nan both
    evaluate to False in Python, so NaN pairs return 0.0 — matching training).
    """
    if v1 == -1.0 or v2 == -1.0:
        return 2.0
    return 1.0 if v1 == v2 else 0.0


@lru_cache(maxsize=4096)
def _encode_p5(desig, post, course, dept, deg, yr, typ, dept1, crs1, deg1, yr1):
    """NaiveNA (prep_type 5): 17-dim vector."""
    de=_enc(desig); po=_enc(post);  co=_enc(course); dp=_enc(dept)
    dg=_enc(deg);   ye=_enc(yr);    ty=_enc(typ);    d1=_enc(dept1)
    c1=_enc(crs1);  g1=_enc(deg1);  y1=_enc(yr1)
    # NA indicators: 1 if the encoded value is -1 (the "NotA" sentinel).
    # NaN-origin values encode to their training integer (not -1), so their
    # NA flag is correctly 0 — matching chk_nota() in data_preprocessor.py.
    return (de, po, co, dp, dg, ye, ty, d1, c1, g1, y1,
            1.0 if po==-1.0 else 0.0,   # Post_NA
            1.0 if co==-1.0 else 0.0,   # Course_NA
            1.0 if dg==-1.0 else 0.0,   # Degree_NA
            1.0 if ye==-1.0 else 0.0,   # Year_NA
            1.0 if c1==-1.0 else 0.0,   # Course.1_NA
            1.0 if y1==-1.0 else 0.0)   # Year.1_NA

@lru_cache(maxsize=4096)
def _encode_p3(desig, post, course, dept, deg, yr, typ, dept1, crs1, deg1, yr1):
    """AVC (prep_type 3): 11-dim vector — all 11 raw cols."""
    return (_enc(desig),_enc(post), _enc(course),_enc(dept), _enc(deg),
            _enc(yr),   _enc(typ),  _enc(dept1),  _enc(crs1),_enc(deg1),_enc(yr1))

@lru_cache(maxsize=4096)
def _encode_arfe(desig, post, typ, course, crs1, dept, dept1, deg, deg1, yr, yr1):
    """ARFE/ARFE+AVC (prep_types 2,4): 7-dim vector."""
    return (_enc(desig), _enc(post), _enc(typ),
            _same_cond(_enc(course), _enc(crs1)),
            _same_cond(_enc(dept),   _enc(dept1)),
            _same_cond(_enc(deg),    _enc(deg1)),
            _same_cond(_enc(yr),     _enc(yr1)))


def encode_features_to_dataframe(features_dict: dict) -> pd.DataFrame:
    """
    Encode raw U2 feature dict -> single-row DataFrame matching XGBoost training columns.

    Replicates GetPreparedData() logic from U2 data_preprocessor.py for all prep_types.
    Returns a DataFrame ready for xgb.DMatrix().

    Column order per prep_type (University_2):
      prep_type 2/4 (ARFE)  : Designation, Post, Type,
                               sameCourse, sameDep, sameDeg, sameYr              (7)
      prep_type 3   (AVC)   : Designation, Post, Course, Department, Degree, Year,
                               Type, Department.1, Course.1, Degree.1, Year.1    (11)
      prep_type 5 (NaiveNA) : Designation, Post, Course, Department, Degree, Year,
                               Type, Department.1, Course.1, Degree.1, Year.1,
                               Post_NA, Course_NA, Degree_NA, Year_NA,
                               Course.1_NA, Year.1_NA                            (17)
    """
    def _u(v):
        # Preserve NaN as the string "nan" so _enc() can detect it via the
        # NAN sentinel path.  strip().upper() on genuine strings is fine.
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return float('nan')
        return str(v).strip().upper() if isinstance(v, str) else v

    desig  = _u(features_dict.get("Designation",  ""))
    post   = _u(features_dict.get("Post",   "NOTA"))
    course = _u(features_dict.get("Course", "NOTA"))
    dept   = _u(features_dict.get("Department",   ""))
    deg    = _u(features_dict.get("Degree", "NOTA"))
    yr     = _u(features_dict.get("Year",   "NOTA"))
    typ    = _u(features_dict.get("Type",   "") or features_dict.get("Resource", ""))
    dept1  = _u(features_dict.get("Department.1","") or features_dict.get("Department_1",""))
    crs1   = _u(features_dict.get("Course.1","NOTA") or features_dict.get("Course_1",""))
    deg1   = _u(features_dict.get("Degree.1","NOTA") or features_dict.get("Degree_1",""))
    yr1    = _u(features_dict.get("Year.1",  "NOTA") or features_dict.get("Year_1",  ""))

    if _PREP_TYPE == 5:
        vec  = _encode_p5(desig,post,course,dept,deg,yr,typ,dept1,crs1,deg1,yr1)
        cols = ["Designation","Post","Course","Department","Degree","Year","Type",
                "Department.1","Course.1","Degree.1","Year.1",
                "Post_NA","Course_NA","Degree_NA","Year_NA","Course.1_NA","Year.1_NA"]
    elif _PREP_TYPE == 3:
        vec  = _encode_p3(desig,post,course,dept,deg,yr,typ,dept1,crs1,deg1,yr1)
        cols = ["Designation","Post","Course","Department","Degree","Year",
                "Type","Department.1","Course.1","Degree.1","Year.1"]
    elif _PREP_TYPE in (2, 4):
        vec  = _encode_arfe(desig,post,typ,course,crs1,dept,dept1,deg,deg1,yr,yr1)
        cols = ["Designation","Post","Type","sameCourse","sameDep","sameDeg","sameYr"]
    else:  # fallback: NaiveNA
        vec  = _encode_p5(desig,post,course,dept,deg,yr,typ,dept1,crs1,deg1,yr1)
        cols = ["Designation","Post","Course","Department","Degree","Year","Type",
                "Department.1","Course.1","Degree.1","Year.1",
                "Post_NA","Course_NA","Degree_NA","Year_NA","Course.1_NA","Year.1_NA"]

    df = pd.DataFrame([list(vec)], columns=cols)
    print(f"[ENC] prep_type={_PREP_TYPE}  cols={cols}  values={list(vec)}")
    return df

# =====================================================
# PDP
# =====================================================
def evaluate_access_policy(bst: xgb.Booster, features: dict) -> str:
    """XGBoost inference for U2 — called for every request, no policy table bypass."""
    try:
        if isinstance(features, dict) and "features" in features:
            features = features["features"]
        df    = encode_features_to_dataframe(features)
        # Explicitly mark float('nan') as missing so XGBoost applies its
        # trained default-direction for those cells — matching training behaviour.
        dmat  = xgb.DMatrix(df, missing=float('nan'))
        proba = float(bst.predict(dmat)[0])
        decision = "YES" if proba >= 0.5 else "NO"
        print(f"[XGB] prep={_PREP_TYPE}({_PREP_NAME})  score={proba:.4f} -> {decision}")
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
        self.req_id=req_id; self.node_id=node_id; self.user=user
        self.resource=resource; self.features=features
        self.arrival_time=time.perf_counter(); self.queue_position=None
        self.done_event=threading.Event(); self.result=None

class PDPQueue:
    def __init__(self, node_id, bst):
        self.node_id=node_id; self.bst=bst
        self._q=queue.Queue(); self._counter=0
        self._lock=threading.Lock(); self._processed=0
        self._worker=threading.Thread(target=self._worker_loop,
                                      name=f"pdp-worker-{node_id}",daemon=True)
        self._worker.start()
        print(f"[{node_id}][PDPQueue] Worker thread started")

    def submit(self, job: PDPJob, timeout: float = 30.0) -> PDPJob:
        with self._lock:
            self._counter+=1; job.queue_position=self._counter; depth=self._q.qsize()
        print(f"[{self.node_id}][PDPQueue] ENQUEUED  ReqID={job.req_id}  "
              f"user={job.user}  resource={job.resource}  pos=#{job.queue_position}  depth={depth}")
        self._q.put(job)
        if not job.done_event.wait(timeout=timeout):
            job.result={"decision":"NO","error":f"PDP timeout after {timeout}s",
                        "queue_position":job.queue_position,
                        "queue_wait_ms":round((time.perf_counter()-job.arrival_time)*1000,2),
                        "pdp_eval_ms":0.0,"jobs_processed":self._processed}
        return job

    def depth(self) -> int:          return self._q.qsize()
    def total_processed(self) -> int: return self._processed
    def worker_alive(self) -> bool:   return self._worker.is_alive()

    def _worker_loop(self):
        while True:
            job=self._q.get(); wait_ms=round((time.perf_counter()-job.arrival_time)*1000,2)
            t0=time.perf_counter()
            print(f"[{self.node_id}][PDPQueue] PROCESSING ReqID={job.req_id}  "
                  f"pos=#{job.queue_position}  user={job.user}  wait={wait_ms}ms")
            try:
                decision=evaluate_access_policy(self.bst,{"features":job.features})
            except Exception as e:
                print(f"[{self.node_id}][PDPQueue] ERROR {job.req_id}: {e}"); decision="NO"
            eval_ms=round((time.perf_counter()-t0)*1000,2); self._processed+=1
            job.result={"decision":decision,"queue_position":job.queue_position,
                        "queue_wait_ms":wait_ms,"pdp_eval_ms":eval_ms,
                        "jobs_processed":self._processed}
            print(f"[{self.node_id}][PDPQueue] COMPLETED ReqID={job.req_id}  "
                  f"-> {decision}  eval={eval_ms}ms  total={self._processed}")
            job.done_event.set(); self._q.task_done()

# =====================================================
# FAULT-TOLERANT PEER COMMUNICATION
# =====================================================
def send_with_retry(node_id, req_id, peer_url, encrypted_data, endpoint="/access_resource"):
    retry_log=[]; delay=RETRY_DELAY_SEC
    for attempt in range(1, RETRY_MAX_ATTEMPTS+1):
        t0=time.perf_counter()
        if attempt>1:
            print(f"[{node_id}][RETRY] ReqID={req_id}  attempt={attempt}  waiting {delay:.1f}s ...")
            time.sleep(delay); delay*=2
        try:
            resp=requests.post(f"{peer_url}{endpoint}",data=encrypted_data,timeout=PEER_CONNECT_TIMEOUT)
            ms=round((time.perf_counter()-t0)*1000,2)
            if resp.status_code==200:
                retry_log.append({"attempt":attempt,"outcome":"success","attempt_ms":ms})
                return resp.json(), retry_log
            raise requests.exceptions.RequestException(f"HTTP {resp.status_code}")
        except Exception as exc:
            ms=round((time.perf_counter()-t0)*1000,2)
            retry_log.append({"attempt":attempt,"outcome":"failure","error":str(exc),"attempt_ms":ms})
            print(f"[{node_id}][RETRY] attempt {attempt} FAILED: {exc}")
    return None, retry_log

def discover_peers(node_id, req_id, peers, encrypted_body):
    candidates=[]
    for peer in peers:
        for attempt in range(1, CHECK_RESOURCE_RETRIES+1):
            try:
                res=requests.post(f"{peer}/check_resource",data=encrypted_body,
                                  timeout=CHECK_RESOURCE_TIMEOUT)
                if res.status_code==200 and res.json().get("has_resource"):
                    candidates.append(peer)
                break
            except Exception:
                if attempt<CHECK_RESOURCE_RETRIES: time.sleep(RETRY_DELAY_SEC)
    return candidates

def forward_remote_request(node_cfg, request_payload, decision,
                           req_id, served_table, served_table_lock):
    start_time=time.perf_counter(); node_id=node_cfg["node_id"]
    resource=request_payload.get("resource","unknown")
    with served_table_lock:
        if req_id in served_table: return dict(served_table[req_id])
    encrypted_body=encrypt_payload({"resource":resource.strip().upper(),
                                     "origin_node":node_id,"req_id":req_id})
    candidates=discover_peers(node_id,req_id,node_cfg["peers"],encrypted_body)
    if decision!="YES":
        result={"status":"denied","resource":resource}
        with served_table_lock: served_table[req_id]=result
        return result
    if not candidates:
        result={"status":"granted","resource":resource}
        with served_table_lock: served_table[req_id]=result
        return result
    fwd_payload=dict(request_payload); fwd_payload["req_id"]=req_id
    encrypted_request=encrypt_payload(fwd_payload)
    final_result=None; result_lock=threading.Lock(); result_ready=threading.Event()
    def _send(peer_url):
        nonlocal final_result
        peer_node=next((c["node_id"] for c in CLIENTS_CONFIG
                        if peer_url.endswith(str(c["port"]))),"?")
        body,_=send_with_retry(node_id,req_id,peer_url,encrypted_request)
        if body and body.get("status")=="granted":
            with result_lock:
                if final_result is None:
                    ms=round((time.perf_counter()-start_time)*1000,2)
                    final_result={"status":"granted_remote","owner":peer_node,
                                  "resource":resource,"latency_ms":ms}
                    result_ready.set()
    max_wait=PEER_CONNECT_TIMEOUT*RETRY_MAX_ATTEMPTS+RETRY_DELAY_SEC*(2**RETRY_MAX_ATTEMPTS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        futures=[ex.submit(_send,p) for p in candidates]
        result_ready.wait(timeout=max_wait)
        for f in futures: f.cancel()
    result=final_result or {"status":"denied","resource":resource}
    with served_table_lock: served_table[req_id]=result
    return result

# =====================================================
# FLASK APP
# =====================================================
def create_app(node_cfg, bst, local_resources) -> Flask:
    app=Flask(node_cfg["node_id"]); pdp_queue=PDPQueue(node_cfg["node_id"],bst)
    served_table={}; served_table_lock=threading.Lock()

    @app.route("/access_request", methods=["POST"])
    def handle_access_request():
        req_id=next_req_id(); arrival=time.perf_counter()
        data=request.get_json(force=True,silent=True) or {}
        user=data.get("user","anonymous"); resource=data.get("resource","")
        features=data.get("features",{})
        print(f"[{node_cfg['node_id']}][PEP] ReqID={req_id}  user={user}  "
              f"resource={resource}  prep={_PREP_TYPE}({_PREP_NAME})")
        with served_table_lock:
            if req_id in served_table:
                cached=dict(served_table[req_id])
                cached["req_id"]=req_id; cached["cached"]=True
                cached["total_latency_ms"]=round((time.perf_counter()-arrival)*1000,2)
                return jsonify(cached)
        job=PDPJob(req_id,node_cfg["node_id"],user,resource,features)
        job=pdp_queue.submit(job,timeout=30.0); res=job.result; decision=res["decision"]
        if decision.upper()=="YES":
            total_ms=round((time.perf_counter()-arrival)*1000,2)
            result={"status":"granted","resource":resource.upper(),
                    "owner":node_cfg["node_id"],"req_id":req_id,
                    "queue_position":res["queue_position"],
                    "queue_wait_ms":res["queue_wait_ms"],
                    "pdp_eval_ms":res["pdp_eval_ms"],"total_latency_ms":total_ms}
            with served_table_lock: served_table[req_id]=result
            return jsonify(result)
        result=forward_remote_request(node_cfg,data,decision,req_id,served_table,served_table_lock)
        result["req_id"]=req_id; result["queue_position"]=res["queue_position"]
        result["queue_wait_ms"]=res["queue_wait_ms"]; result["pdp_eval_ms"]=res["pdp_eval_ms"]
        result["total_latency_ms"]=round((time.perf_counter()-arrival)*1000,2)
        return jsonify(result)

    @app.route("/queue_status", methods=["GET"])
    def queue_status():
        with served_table_lock: sc=len(served_table)
        return jsonify({"node_id":node_cfg["node_id"],"model":"XGBoost",
                        "dataset":_DS_DISPLAY,"algo":_ALGO,
                        "prep_type":_PREP_TYPE,"prep_name":_PREP_NAME,"seed":_SEED,
                        "queue_depth":pdp_queue.depth(),
                        "total_processed":pdp_queue.total_processed(),
                        "worker_alive":pdp_queue.worker_alive(),"served_req_count":sc})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"node":node_cfg["node_id"],"status":"running","model":"XGBoost",
                        "dataset":_DS_DISPLAY,"algo":_ALGO,
                        "prep_type":_PREP_TYPE,"prep_name":_PREP_NAME,
                        "seed":_SEED,"resources":local_resources})

    @app.route("/check_resource", methods=["POST"])
    def check_resource():
        try:
            d=decrypt_payload(request.data); r=d.get("resource","").strip().upper()
            return jsonify({"has_resource":any(r==x.strip().upper() for x in local_resources)})
        except Exception:
            return jsonify({"has_resource":False})

    @app.route("/access_resource", methods=["POST"])
    def access_resource():
        try:
            d=decrypt_payload(request.data)
            resource=d.get("resource"); req_id_in=d.get("req_id")
            if req_id_in is not None:
                with served_table_lock:
                    if req_id_in in served_table:
                        return jsonify({"status":"already_served","req_id":req_id_in})
                    served_table[req_id_in]={"status":"granted",
                                             "served_by":node_cfg["node_id"],"resource":resource}
            return jsonify({"status":"granted","req_id":req_id_in})
        except Exception as e:
            print(f"[{node_cfg['node_id']}][ERROR] /access_resource: {e}")
            return jsonify({"status":"denied"})

    return app

def infer_resource_ownership(client_configs: list) -> dict:
    owners=defaultdict(str)
    print(f"\n=== RESOURCE OWNERSHIP ({_DS_DISPLAY}: Type column) ===")
    raw_train=os.path.join(_case_dir,"train.csv")
    if os.path.exists(raw_train):
        try:
            df=pd.read_csv(raw_train)
            if "Type" in df.columns:
                for res in df["Type"].dropna().unique():
                    owners[str(res).upper()]="global"
        except Exception as e:
            print(f"  [WARN] {e}")
    print(f"  {len(owners)} unique resources inferred")
    return owners

def start_node(cfg, bst):
    print(f"[{cfg['node_id']}] port={cfg['port']}  "
          f"resources={cfg['resources'][:5]}{'...' if len(cfg['resources'])>5 else ''}")
    create_app(cfg, bst, cfg["resources"]).run(
        host="0.0.0.0", port=cfg["port"], debug=False, use_reloader=False, threaded=True)

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print(f"\n[START] XGBoost PE -- {_DS_DISPLAY}  "
          f"ALGO={_ALGO}  PREP_TYPE={_PREP_TYPE}({_PREP_NAME})  SEED={_SEED}  case={_CASE_NAME}\n")
    bst = load_xgb_model(XGB_MODEL_PATH)
    RESOURCE_OWNERSHIP = infer_resource_ownership(CLIENTS_CONFIG)
    threads = []
    for cfg in CLIENTS_CONFIG:
        t = threading.Thread(target=start_node, args=(cfg, bst))
        t.daemon = True; t.start(); threads.append(t)
    print(f"\n[MAIN] 4 nodes started (5000-5003)  model=XGBoost  "
          f"dataset={_DS_DISPLAY}  algo={_ALGO}  prep={_PREP_TYPE}({_PREP_NAME})  seed={_SEED}\n"
          "Ctrl-C to stop.\n")
    try:
        for t in threads: t.join()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutdown.")
