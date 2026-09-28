"""
auto_test_runner_XGB_Company.py
================================
Tests the XGBoost (SecureBoost/NVFlare) policy enforcement server
for the Company dataset.

Test CSV path (produced by prepare_data_split → test_processed.csv):
  results/company/{PARTITION}/{ENCODING}/test_processed.csv

Output:
  Results/XGB/Company/{ALGO}/{PREP_NAME}/PolicyEnforcement/test_results.csv
  Results/XGB/Company/{ALGO}/{PREP_NAME}/PolicyEnforcement/test_summary.json
  Results/XGB/Company/{ALGO}/{PREP_NAME}/PolicyEnforcement/test_summary.csv

Usage:
  python3.10 auto_test_runner_XGB_Company.py --algo SBC --prep-type 2 --seed 0
  python3.10 auto_test_runner_XGB_Company.py --algo PBP --prep-type 5 --all-nodes
  python3.10 auto_test_runner_XGB_Company.py --consolidate
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from itertools import cycle

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET = "Company"

DEFAULT_NODES    = ["http://localhost:5000","http://localhost:5001",
                    "http://localhost:5002","http://localhost:5003"]
DEFAULT_ENDPOINT = "/access_request"

# Naming tables — must match run_simulation.py and PE server
_PARTITION_CODES = {"SBC": "SBP", "PBP": "PBP", "RNP": "RNP"}
_ENCODING_NAMES  = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "Naive_NACol"}
_PREP_NAMES      = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}

# Company dataset raw feature columns sent to PE server
# (test_processed.csv has label as first col, then encoded features)
# The test runner reads raw columns from test_processed.csv and passes them
# directly — the PE server does the encoding internally.
COLUMN_MAP = {
    "DESIGNATION":  "Designation",
    "Project_name": "Project_name",
    "Department":   "Department",
    "Resource":     "Resource",
    "Project_Name": "Project_Name",
    "Department.1": "Department.1",
}
ACCESS_LABEL_COL = "Access"


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_label(val) -> int:
    v = str(val).strip().upper()
    if v in ("YES","1","TRUE","GRANT","GRANTED"): return 1
    if v in ("NO","0","FALSE","DENY","DENIED"):   return 0
    raise ValueError(f"Unknown label: {val!r}")

def response_to_label(resp_json: dict) -> int:
    return 1 if "grant" in str(resp_json.get("status","")).lower() else 0

def build_payload(row: pd.Series, row_idx: int) -> dict:
    features = {}
    for csv_col, api_key in COLUMN_MAP.items():
        if csv_col in row.index:
            features[api_key] = str(row[csv_col]).strip()
    resource = features.get("Resource", "UNKNOWN")
    return {"user": f"TEST_U{row_idx}", "resource": resource, "features": features}


# ── Single request ─────────────────────────────────────────────────────────────
def send_request(args):
    row_idx, row, url, timeout = args
    payload = build_payload(row, row_idx)
    result  = {"row_idx": row_idx, "user": payload["user"],
               "resource": payload["resource"], "node_url": url,
               "actual_label": None, "pred_label": None, "pred_status": None,
               "latency_ms": None, "error": None, "response_json": None}
    try:
        result["actual_label"] = normalize_label(row[ACCESS_LABEL_COL])
    except Exception as e:
        result["error"] = f"Label error: {e}"; return result

    t0 = time.perf_counter()
    try:
        resp = requests.post(url + DEFAULT_ENDPOINT, json=payload,
                             headers={"Content-Type":"application/json"},
                             timeout=timeout)
        result["latency_ms"]    = round((time.perf_counter()-t0)*1000, 2)
        resp_json               = resp.json()
        result["response_json"] = resp_json
        result["pred_status"]   = resp_json.get("status","unknown")
        result["pred_label"]    = response_to_label(resp_json)
    except requests.exceptions.ConnectionError:
        result["error"] = f"Connection refused — is the server running at {url}?"
    except requests.exceptions.Timeout:
        result["error"] = f"Timeout after {timeout}s"
    except Exception as e:
        result["error"] = str(e)
    return result


# ── Metrics report ─────────────────────────────────────────────────────────────
def print_metrics(y_true, y_pred, latencies, total_rows, error_count, algo, prep_name):
    print(f"\n{'='*60}")
    print(f"  XGBoost PE TEST REPORT — {DATASET}  {algo}/{prep_name}")
    print(f"{'='*60}")
    print(f"  Total rows  : {total_rows}")
    print(f"  Success     : {total_rows - error_count}")
    print(f"  Errors      : {error_count}")
    if not y_true:
        print("  [!] No successful predictions."); return
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if latencies:
        print(f"\n  Avg latency : {sum(latencies)/len(latencies):.2f} ms")
        print(f"  Min latency : {min(latencies):.2f} ms")
        print(f"  Max latency : {max(latencies):.2f} ms")
    print("\n  Confusion Matrix  (rows=Actual, cols=Predicted)")
    print("                 Pred NO   Pred YES")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print(f"  Actual NO  :   {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"  Actual YES :   {cm[1][0]:>6}    {cm[1][1]:>6}")
    print("\n  Per-class Report:")
    print(classification_report(y_true, y_pred, target_names=["NO","YES"], zero_division=0))
    print("="*60)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XGBoost PE test runner — Company dataset"
    )
    parser.add_argument("--algo",         default="SBC", choices=["SBC","PBP","RNP"])
    parser.add_argument("--prep-type",    type=int, default=5, choices=[2,3,4,5])
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--file",         default=None,
                        help="Custom CSV path (overrides auto-resolved path)")
    parser.add_argument("--port",         type=int, default=5000)
    parser.add_argument("--all-nodes",    action="store_true")
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--timeout",      type=int, default=8)
    parser.add_argument("--limit",        type=int, default=None)
    args = parser.parse_args()

    algo      = args.algo
    prep_type = args.prep_type
    prep_name = _PREP_NAMES[prep_type]
    partition = _PARTITION_CODES[algo]
    encoding  = _ENCODING_NAMES[prep_type]

    print(f"[INFO] DATASET={DATASET}  ALGO={algo}  PREP_TYPE={prep_type}({prep_name})  "
          f"SEED={args.seed}  MODEL=XGBoost")

    # ── Resolve CSV path ─────────────────────────────────────────────────────
    # test_processed.csv is created by prepare_data_split inside the case dir
    if args.file:
        csv_path = args.file
    else:
        case_dir = os.path.join(os.getcwd(), args.results_root,
                                "company", partition, encoding)
        # Use raw_test_data_{seed}.csv — contains original string values
        # (DESIGNATION="CEO", Resource="DATABASE", etc.) so the PE server can
        # re-encode them correctly via the mapping pkl.
        # test_processed.csv has encoded integers which the mapping can't decode.
        csv_path = os.path.join(case_dir, "TestData",
                                f"raw_test_data_{args.seed}.csv")

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        print("        Run NVFlare simulation first: python3.10 run_simulation.py")
        sys.exit(1)

    print(f"[INFO] Loading test data: {csv_path}")
    df = pd.read_csv(csv_path)

    # raw_test_data_{seed}.csv has original string values (e.g. "CEO", "DATABASE")
    # plus ENC_* prefixed encoded columns appended for reference.
    # Drop the ENC_* columns — the PE server encodes internally from the raw strings.
    raw_cols = [c for c in df.columns if not c.startswith("ENC_")]
    df = df[raw_cols]

    if ACCESS_LABEL_COL not in df.columns and df.columns[0] != ACCESS_LABEL_COL:
        df = df.rename(columns={df.columns[0]: ACCESS_LABEL_COL})

    print("[INFO] CSV columns found:", list(df.columns))
    matched = [k for k in COLUMN_MAP if k in df.columns]
    print(f"[INFO] COLUMN_MAP matched: {matched}")
    missing = [k for k in COLUMN_MAP if k not in df.columns]
    if missing:
        print(f"[WARN] Keys not in CSV (will send 0): {missing}")

    if ACCESS_LABEL_COL not in df.columns:
        print(f"[ERROR] '{ACCESS_LABEL_COL}' not found. Cols: {list(df.columns)}")
        sys.exit(1)

    if args.limit:
        df = df.head(args.limit)

    total_rows = len(df)
    print(f"[INFO] Total test rows: {total_rows}")

    if args.all_nodes:
        node_cycle = cycle(DEFAULT_NODES)
        urls = [next(node_cycle) for _ in range(total_rows)]
        print(f"[INFO] Round-robin: {DEFAULT_NODES}")
    else:
        target = f"http://localhost:{args.port}"
        urls   = [target] * total_rows
        print(f"[INFO] Targeting: {target}")

    tasks = [(i, df.iloc[i], urls[i], args.timeout) for i in range(total_rows)]

    print(f"[INFO] Sending {total_rows} requests ({args.workers} workers) …\n")
    results = []
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, res in enumerate(ex.map(send_request, tasks), 1):
            results.append(res)
            if idx % 100 == 0 or idx == total_rows:
                rps = idx / (time.perf_counter()-t_start)
                print(f"  Progress: {idx}/{total_rows}  |  {rps:.1f} req/s", end="\r")
    print()

    y_true, y_pred, latencies, errors = [], [], [], []
    for r in results:
        if r["error"]: errors.append(r)
        elif r["pred_label"] is not None and r["actual_label"] is not None:
            y_true.append(r["actual_label"])
            y_pred.append(r["pred_label"])
            if r["latency_ms"] is not None:
                latencies.append(r["latency_ms"])

    print_metrics(y_true, y_pred, latencies, total_rows, len(errors), algo, prep_name)

    # ── Output ────────────────────────────────────────────────────────────────
    # Results/XGB/Company/{ALGO}/{PREP_NAME}/PolicyEnforcement/
    base_dir = os.path.join(os.getcwd(), "Results", "XGB",
                            DATASET, algo, prep_name, "PolicyEnforcement")
    os.makedirs(base_dir, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df["resp_owner"]   = results_df["response_json"].apply(
        lambda x: x.get("owner","") if isinstance(x,dict) else "")
    results_df["resp_latency"] = results_df["response_json"].apply(
        lambda x: x.get("total_latency_ms","") if isinstance(x,dict) else "")
    results_df.drop(columns=["response_json"], inplace=True)

    out_path = os.path.join(base_dir, "test_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n[INFO] Results → {out_path}")

    if errors:
        err_df = pd.DataFrame(errors)[["row_idx","resource","node_url","error"]]
        err_df.to_csv(os.path.join(base_dir,"test_errors.csv"), index=False)
        print(f"[WARN] {len(errors)} errors logged")

    if y_true:
        summary = {
            "model":          "XGBoost",
            "dataset":        DATASET,
            "algo":           algo,
            "prep_type":      prep_type,
            "prep_name":      prep_name,
            "seed":           args.seed,
            "csv_file":       csv_path,
            "total_rows":     total_rows,
            "success":        total_rows - len(errors),
            "errors":         len(errors),
            "accuracy":       round(accuracy_score(y_true,y_pred)*100, 4),
            "precision":      round(precision_score(y_true,y_pred,zero_division=0), 4),
            "recall":         round(recall_score(y_true,y_pred,zero_division=0), 4),
            "f1":             round(f1_score(y_true,y_pred,zero_division=0), 4),
            "avg_latency_ms": round(sum(latencies)/len(latencies),2) if latencies else None,
        }
        with open(os.path.join(base_dir,"test_summary.json"),"w") as f:
            json.dump(summary, f, indent=2)
        pd.DataFrame([summary]).to_csv(
            os.path.join(base_dir,"test_summary.csv"), index=False)
        print(f"[INFO] Summary → {os.path.join(base_dir,'test_summary.csv')}")


# ── Consolidate ────────────────────────────────────────────────────────────────
def consolidate_results(base="Results/XGB/Company", out_file=None):
    import glob
    base  = os.path.abspath(base)
    found = sorted(glob.glob(
        os.path.join(base,"*","*","PolicyEnforcement","test_summary.csv")))
    if not found:
        print(f"[WARN] No summaries found under {base}"); return

    rows = []
    for path in found:
        try:
            df = pd.read_csv(path)
            if df.empty: continue
            parts   = path.replace("\\","/").split("/")
            pe_idx  = parts.index("PolicyEnforcement")
            row     = df.iloc[0].to_dict()
            row.setdefault("dataset",   DATASET)
            row.setdefault("algo",      parts[pe_idx-2])
            row.setdefault("prep_name", parts[pe_idx-1])
            row["source_file"] = path
            rows.append(row)
            print(f"[INFO] {parts[pe_idx-2]}/{parts[pe_idx-1]}  "
                  f"acc={row.get('accuracy','?')}%  f1={row.get('f1','?')}")
        except Exception as e:
            print(f"[WARN] {path}: {e}")

    if not rows:
        print("[WARN] No valid summaries."); return

    df = pd.DataFrame(rows)
    ordered = ["model","dataset","algo","prep_type","prep_name","seed",
               "total_rows","success","errors","accuracy","precision",
               "recall","f1","avg_latency_ms","csv_file","source_file"]
    cols = [c for c in ordered if c in df.columns] + \
           [c for c in df.columns if c not in ordered]
    df = df[cols].sort_values(["algo","prep_name"]).reset_index(drop=True)

    if out_file is None:
        out_file = os.path.join(base, "consolidated_results.csv")
    df.to_csv(out_file, index=False)

    print(f"\n{'='*65}")
    print(f"  XGBoost PE Consolidated — {DATASET}  ({len(df)} runs)")
    print(f"{'='*65}")
    dcols = ["algo","prep_name","accuracy","precision","recall","f1"]
    dcols = [c for c in dcols if c in df.columns]
    print(df[dcols].to_string(index=False))
    print(f"{'='*65}")
    print(f"\n[INFO] Consolidated → {out_file}")
    return df


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse as _pre_ap, sys as _pre_sys
    if "--consolidate" in _pre_sys.argv:
        _cp = _pre_ap.ArgumentParser()
        _cp.add_argument("--consolidate", action="store_true")
        _cp.add_argument("--base", default="Results/XGB/Company")
        _cp.add_argument("--out",  default=None)
        _ca = _cp.parse_args()
        consolidate_results(base=_ca.base, out_file=_ca.out)
    else:
        main()
