# server.py  —  XGBoost Flower server
#   • AES-256-GCM encrypted model exchange (encryption.py)
#   • SPW_POWER-scaled scale_pos_weight
#   • Optimal-F1 threshold
#   • One row per experiment → consolidated_results.csv with
#     TrainTime_s / AvgInference_ms / TrainRecords / TestRecords
#
# Encryption points:
#   EncryptedFedXgbBagging.aggregate_fit()
#       DECRYPT every client tensor → parent merges trees on raw bytes
#       → evaluate on raw global model → RE-ENCRYPT before broadcast
#   EncryptedFedXgbCyclic.evaluate()
#       Cyclic passes the single client tensor through unchanged (still
#       encrypted); only DECRYPT is needed here for evaluation.

import csv
import logging
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import (f1_score, precision_recall_curve,
                             precision_score, recall_score)

from flwr.common import FitRes, Parameters, Scalar, Context
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from dataset_registry import get_dataset_config
from utils import GetPreparedData, replace_keys, compute_scale_pos_weight
from encryption import (get_fl_key, decrypt_bytes,
                        encrypt_tensors, decrypt_tensors)

_log = logging.getLogger(__name__)

PREP_NAMES = {1: "Naive", 2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}


# ─── Optimal threshold ────────────────────────────────────────────────────────

def find_best_threshold(labels: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    pr, rc, thr = precision_recall_curve(labels, proba)
    f1s = np.where((pr + rc) > 0, 2 * pr * rc / (pr + rc), 0.0)
    return float(thr[int(np.argmax(f1s[:-1]))]) if len(thr) > 0 else 0.5


# ─── Consolidated results CSV ─────────────────────────────────────────────────

def _write_final_result(output_dir: str, row: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "consolidated_results.csv")
    fieldnames = [
        "dataset", "train_method", "prep_type", "partition_strategy", "seed",
        "AUC", "accuracy", "precision", "recall", "f1", "threshold",
        "TrainTime_s", "AvgInference_ms", "TrainRecords", "TestRecords",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
    _log.info("[Results] Written → %s", path)


# ─── Shared metrics computation ───────────────────────────────────────────────

def _compute_metrics(bst, test_dmatrix, test_labels, meta,
                     server_round, num_rounds, output_dir,
                     total_train_time_s=0.0, total_train_records=0) -> dict:
    eval_str = bst.eval_set(
        evals=[(test_dmatrix, "valid")],
        iteration=bst.num_boosted_rounds() - 1,
    )
    auc = round(float(eval_str.split("\t")[1].split(":")[1]), 4)

    t0          = time.perf_counter()
    proba       = bst.predict(test_dmatrix)
    infer_s     = time.perf_counter() - t0
    n_test      = len(test_labels)
    avg_inf_ms  = round(infer_s / max(n_test, 1) * 1000, 6)

    labels = test_labels.astype(int)
    thr    = find_best_threshold(labels, proba)
    preds  = (proba >= thr).astype(int)

    acc  = round(float((preds == labels).mean()), 4)
    prec = round(float(precision_score(labels, preds, zero_division=0)), 4)
    rec  = round(float(recall_score(labels,    preds, zero_division=0)), 4)
    f1   = round(float(f1_score(labels,        preds, zero_division=0)), 4)

    _log.info("[%s] Round %3d/%d | AUC=%.4f Acc=%.4f Thr=%.3f "
              "P=%.4f R=%.4f F1=%.4f  InfAvg=%.4fms",
              meta["train_method"].upper(), server_round, num_rounds,
              auc, acc, thr, prec, rec, f1, avg_inf_ms)

    if server_round == num_rounds:
        _write_final_result(output_dir, {
            **meta,
            "AUC": auc, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1,
            "threshold": round(thr, 4),
            "TrainTime_s":     round(total_train_time_s, 4),
            "AvgInference_ms": avg_inf_ms,
            "TrainRecords":    total_train_records,
            "TestRecords":     n_test,
        })

    return {"AUC": auc, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}


# ─── Config builder ───────────────────────────────────────────────────────────

def _build_cfg(run_config: dict) -> dict:
    rc     = replace_keys(unflatten_dict(run_config)) if run_config else {}
    seed   = int(os.environ.get("FLWR_SEED", "0"))
    params = rc.get("params", {})
    return {
        "num_server_rounds": int(rc.get("num_server_rounds", os.environ.get("NUM_ROUNDS", "50"))),
        "fraction_fit":    float(rc.get("fraction_fit",      os.environ.get("FRACTION_FIT", "1.0"))),
        "centralised_eval":  str(rc.get("centralised_eval",  os.environ.get("CENTRALISED_EVAL", "true"))).lower() == "true",
        "local_epochs":    int(  rc.get("local_epochs",      os.environ.get("LOCAL_EPOCHS", "3"))),
        "train_method":         rc.get("train_method",      os.environ.get("TRAIN_METHOD", "bagging")),
        "params": {
            "objective":         params.get("objective",         "binary:logistic"),
            "eta":          float(params.get("eta",              os.environ.get("ETA",              "0.05"))),
            "max_depth":      int(params.get("max_depth",        os.environ.get("MAX_DEPTH",        "6"))),
            "min_child_weight":int(params.get("min_child_weight",os.environ.get("MIN_CHILD_WEIGHT", "5"))),
            "subsample":    float(params.get("subsample",        os.environ.get("SUBSAMPLE",        "0.8"))),
            "colsample_bytree":float(params.get("colsample_bytree",os.environ.get("COLSAMPLE_BYTREE","0.8"))),
            "gamma":        float(params.get("gamma",            os.environ.get("GAMMA",            "0.1"))),
            "eval_metric":       params.get("eval_metric",       os.environ.get("EVAL_METRIC",      "auc")),
            "nthread":        int(params.get("nthread",          os.environ.get("NTHREAD",          "1"))),
            "tree_method":       params.get("tree_method",       os.environ.get("TREE_METHOD",      "hist")),
            "seed":              seed,
            "scale_pos_weight": float(params.get("scale_pos_weight", "1")),
        },
    }


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def fit_metrics_aggregation(fit_metrics):
    if not fit_metrics:
        return {}
    accs = [m["accuracy"] for _, m in fit_metrics if "accuracy" in m]
    aucs = [m["auc"]      for _, m in fit_metrics if "auc"      in m]
    out  = {}
    if accs: out["accuracy_mean"] = float(np.mean(accs))
    if aucs: out["auc_mean"]      = float(np.mean(aucs))
    return out

def evaluate_metrics_aggregation(eval_metrics):
    return eval_metrics[0][1] if eval_metrics else {}

def config_func(rnd: int) -> Dict[str, str]:
    return {"global_round": str(rnd)}


# ─── Encrypted Bagging Strategy ───────────────────────────────────────────────

class EncryptedFedXgbBagging(FedXgbBagging):
    """FedXgbBagging with AES-256-GCM encrypted model parameters.

    FedXgbBagging.aggregate_fit() parses model bytes as JSON to merge trees,
    so client tensors must be DECRYPTED first; the aggregated model is
    RE-ENCRYPTED before it is broadcast to clients.
    """

    def __init__(self, test_dmatrix, test_labels, xgb_params,
                 meta, num_rounds, output_dir, centralised_eval, **kwargs):
        super().__init__(**kwargs)
        self.test_dmatrix        = test_dmatrix
        self.test_labels         = test_labels
        self.xgb_params          = xgb_params
        self.meta                = meta
        self.num_rounds          = num_rounds
        self.output_dir          = output_dir
        self.centralised_eval    = centralised_eval
        self.total_train_time_s  = 0.0
        self._records_by_client  = {}   # cid → num_examples

    @property
    def total_train_records(self) -> int:
        return int(sum(self._records_by_client.values()))

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        key = get_fl_key()

        # Accumulate client timing / record counts (per client id, any round)
        for proxy, fit_res in results:
            self.total_train_time_s += float(fit_res.metrics.get("train_time_s", 0.0))
            self._records_by_client[proxy.cid] = int(fit_res.num_examples)

        # ── DECRYPT each client's tensors ────────────────────────────────────
        dec_results = []
        for proxy, fit_res in results:
            dec_results.append((proxy, FitRes(
                status=fit_res.status,
                parameters=Parameters(
                    tensor_type="",
                    tensors=decrypt_tensors(fit_res.parameters.tensors, key)),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )))

        # ── Merge trees on raw bytes (parent logic) ──────────────────────────
        agg_params, agg_metrics = super().aggregate_fit(server_round, dec_results, failures)
        if agg_params is None or not agg_params.tensors:
            return agg_params, agg_metrics

        # ── Centralised evaluation on raw aggregated model ───────────────────
        if self.centralised_eval and self.test_dmatrix is not None:
            bst = xgb.Booster(params=self.xgb_params)
            bst.load_model(bytearray(agg_params.tensors[0]))
            _compute_metrics(bst, self.test_dmatrix, self.test_labels,
                             self.meta, server_round, self.num_rounds,
                             self.output_dir,
                             total_train_time_s=self.total_train_time_s,
                             total_train_records=self.total_train_records)

        # ── RE-ENCRYPT before broadcast ──────────────────────────────────────
        enc_tensors = encrypt_tensors(agg_params.tensors, key)
        _log.debug("[Bagging] Round %d: re-encrypted global model %d → %d bytes",
                   server_round, len(agg_params.tensors[0]), len(enc_tensors[0]))
        return Parameters(tensor_type="", tensors=enc_tensors), agg_metrics

    def evaluate(self, server_round, parameters):
        return None   # evaluation is done inside aggregate_fit


# ─── Encrypted Cyclic Strategy ────────────────────────────────────────────────

class EncryptedFedXgbCyclic(FedXgbCyclic):
    """FedXgbCyclic with AES-256-GCM encrypted model parameters.

    FedXgbCyclic.aggregate_fit() passes the single client tensor through
    unchanged, so the encrypted bytes flow transparently.  Only DECRYPT
    is needed in evaluate() for metrics.
    """

    def __init__(self, test_dmatrix, test_labels, xgb_params,
                 meta, num_rounds, output_dir, **kwargs):
        super().__init__(**kwargs)
        self.test_dmatrix        = test_dmatrix
        self.test_labels         = test_labels
        self.xgb_params          = xgb_params
        self.meta                = meta
        self.num_rounds          = num_rounds
        self.output_dir          = output_dir
        self.total_train_time_s  = 0.0
        self._records_by_client  = {}   # cid → num_examples

    @property
    def total_train_records(self) -> int:
        return int(sum(self._records_by_client.values()))

    def aggregate_fit(self, server_round, results, failures):
        for proxy, fit_res in results:
            self.total_train_time_s += float(fit_res.metrics.get("train_time_s", 0.0))
            self._records_by_client[proxy.cid] = int(fit_res.num_examples)
        return super().aggregate_fit(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: Parameters
                 ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if server_round == 0 or not parameters.tensors or self.test_dmatrix is None:
            return None
        key = get_fl_key()
        raw = decrypt_bytes(bytes(parameters.tensors[0]), key)   # ── DECRYPT ──
        bst = xgb.Booster(params=self.xgb_params)
        bst.load_model(bytearray(raw))
        metrics = _compute_metrics(bst, self.test_dmatrix, self.test_labels,
                                   self.meta, server_round, self.num_rounds,
                                   self.output_dir,
                                   total_train_time_s=self.total_train_time_s,
                                   total_train_records=self.total_train_records)
        return 0.0, metrics


class CyclicClientManager(SimpleClientManager):
    def sample(self, num_clients, min_num_clients=None, criterion=None):
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        cids = list(self.clients)
        if criterion:
            cids = [c for c in cids if criterion.select(self.clients[c])]
        return [self.clients[c] for c in cids] if num_clients <= len(cids) else []


# ─── ServerApp ────────────────────────────────────────────────────────────────

def server_fn(context: Context):
    cfg = _build_cfg(context.run_config)

    seed = int(os.environ.get("FLWR_SEED", "0"))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    num_rounds   = cfg["num_server_rounds"]
    fraction_fit = cfg["fraction_fit"]
    params       = dict(cfg["params"])
    train_method = os.environ.get("TRAIN_METHOD",       cfg.get("train_method", "bagging"))
    dataset_name = os.environ.get("DATASET_NAME",       "company")
    prep_type    = int(os.environ.get("PREP_TYPE",      "2"))
    partition    = os.environ.get("PARTITION_STRATEGY", "sbp").lower()
    output_dir   = os.environ.get("OUTPUT_DIR",         "results")
    dataset_csv  = os.environ.get("DATASET_CSV",        "dataset.csv")
    spw_power    = float(os.environ.get("SPW_POWER",    "0.5"))
    prep_name    = PREP_NAMES.get(prep_type, f"PrepType{prep_type}")

    meta = {
        "dataset":            dataset_name,
        "train_method":       train_method,
        "prep_type":          prep_name,
        "partition_strategy": partition,
        "seed":               seed,
    }

    _ = get_fl_key()   # derive key once, log source

    test_dmatrix = None
    test_labels  = None
    if cfg["centralised_eval"]:
        ds_cfg = get_dataset_config(dataset_name)
        _, X_te, _, y_te, _, _, _ = GetPreparedData(
            dataset_csv, prep_type, seed, k_best=1, cfg=ds_cfg,
        )
        test_labels  = np.array(y_te).ravel()
        test_dmatrix = xgb.DMatrix(X_te, label=test_labels)

        raw_spw = compute_scale_pos_weight(y_te)
        spw     = round(raw_spw ** spw_power, 4)
        if float(params.get("scale_pos_weight", 1)) == 1.0:
            params["scale_pos_weight"] = spw
            _log.info("[Server] dataset=%s  spw_raw=%.2f  spw_power=%.2f  spw=%.2f",
                      dataset_name, raw_spw, spw_power, spw)

    initial_parameters = Parameters(tensor_type="", tensors=[])
    common = dict(
        fraction_evaluate=0.0,
        on_fit_config_fn=config_func,
        on_evaluate_config_fn=config_func,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        initial_parameters=initial_parameters,
    )

    if train_method == "bagging":
        strategy = EncryptedFedXgbBagging(
            test_dmatrix, test_labels, params, meta, num_rounds, output_dir,
            centralised_eval=cfg["centralised_eval"],
            evaluate_function=None,
            fraction_fit=fraction_fit,
            **common,
        )
        client_manager = None
    else:
        strategy = EncryptedFedXgbCyclic(
            test_dmatrix, test_labels, params, meta, num_rounds, output_dir,
            **common,
        )
        client_manager = CyclicClientManager()

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
        client_manager=client_manager,
    )


app = ServerApp(server_fn=server_fn)
