# client.py  —  XGBoost Flower client
#   • AES-256-GCM encrypted model exchange (encryption.py)
#   • SPW_POWER-scaled scale_pos_weight
#   • Optimal-F1 threshold for local metrics
#
# Encryption points:
#   fit()      round 1   : ENCRYPT outgoing model bytes
#   fit()      round > 1 : DECRYPT incoming global model, then ENCRYPT outgoing
#   evaluate()           : DECRYPT incoming global model
#
# Key = HKDF-SHA256(FL_SHARED_SECRET). Same secret on server and all clients.

import logging
import os
import random
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score,
                             precision_recall_curve)

from flwr.client import Client, ClientApp
from flwr.common import (Code, EvaluateIns, EvaluateRes,
                         FitIns, FitRes, Parameters, Status)
from flwr.common.config import unflatten_dict
from flwr.common.context import Context

from utils import load_data, replace_keys
from encryption import get_fl_key, encrypt_bytes, decrypt_bytes

_log = logging.getLogger(__name__)


# ─── Optimal threshold ────────────────────────────────────────────────────────

def _best_threshold(labels: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    pr, rc, thr = precision_recall_curve(labels, proba)
    f1s = np.where((pr + rc) > 0, 2 * pr * rc / (pr + rc), 0.0)
    return float(thr[int(np.argmax(f1s[:-1]))]) if len(thr) > 0 else 0.5


# ─── Config builder ───────────────────────────────────────────────────────────

def _build_cfg(run_config: dict) -> dict:
    rc     = replace_keys(unflatten_dict(run_config)) if run_config else {}
    seed   = int(os.environ.get("FLWR_SEED", "0"))
    params = rc.get("params", {})
    return {
        "local_epochs": int(rc.get("local_epochs", os.environ.get("LOCAL_EPOCHS", "3"))),
        "train_method":     rc.get("train_method", os.environ.get("TRAIN_METHOD", "bagging")),
        "scaled_lr": str(rc.get("scaled_lr", os.environ.get("SCALED_LR", "false"))).lower() == "true",
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


# ─── XGBClient ────────────────────────────────────────────────────────────────

class XGBClient(Client):
    def __init__(self, train_dmatrix, valid_dmatrix, num_train, num_val,
                 num_local_round, params, train_method):
        self.train_dmatrix   = train_dmatrix
        self.valid_dmatrix   = valid_dmatrix
        self.num_train       = num_train
        self.num_val         = num_val
        self.num_local_round = num_local_round
        self.params          = params
        self.train_method    = train_method

    def _local_boost(self, bst: xgb.Booster) -> xgb.Booster:
        for _ in range(self.num_local_round):
            bst.update(self.train_dmatrix, bst.num_boosted_rounds())
        if self.train_method == "bagging":
            return bst[bst.num_boosted_rounds() - self.num_local_round:
                       bst.num_boosted_rounds()]
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        key = get_fl_key()
        t0  = time.perf_counter()

        if global_round == 1:
            bst = xgb.train(
                self.params, self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "valid"), (self.train_dmatrix, "train")],
                verbose_eval=False,
            )
        else:
            # ── DECRYPT incoming global model ────────────────────────────────
            raw = decrypt_bytes(bytes(ins.parameters.tensors[0]), key)
            bst = xgb.Booster(params=self.params)
            bst.load_model(bytearray(raw))
            bst = self._local_boost(bst)

        train_time_s = time.perf_counter() - t0

        preds_proba = bst.predict(self.train_dmatrix)
        labels      = self.train_dmatrix.get_label()
        thr         = _best_threshold(labels, preds_proba)
        preds       = (preds_proba >= thr).astype(int)

        metrics: dict = {"train_time_s": float(train_time_s)}
        if len(np.unique(labels)) > 1:
            metrics["accuracy"] = float(accuracy_score(labels, preds))
            metrics["loss"]     = float(log_loss(labels, preds_proba))
            try:
                metrics["auc"] = float(roc_auc_score(labels, preds_proba))
            except ValueError:
                pass

        _log.info("Fit round=%d  acc=%.4f  time=%.3fs",
                  global_round, metrics.get("accuracy", 0), train_time_s)

        # ── ENCRYPT outgoing model ───────────────────────────────────────────
        raw_model = bytes(bst.save_raw("json"))
        enc_model = encrypt_bytes(raw_model, key)
        _log.debug("fit round=%d: encrypted %d → %d bytes",
                   global_round, len(raw_model), len(enc_model))

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[enc_model]),
            num_examples=self.num_train,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        key = get_fl_key()

        # ── DECRYPT incoming global model ────────────────────────────────────
        raw = decrypt_bytes(bytes(ins.parameters.tensors[0]), key)
        bst = xgb.Booster(params=self.params)
        bst.load_model(bytearray(raw))

        preds_proba = bst.predict(self.valid_dmatrix)
        labels      = self.valid_dmatrix.get_label()
        thr         = _best_threshold(labels, preds_proba)
        preds       = (preds_proba >= thr).astype(int)
        accuracy    = round(float((preds == labels).mean()), 4)

        eval_str = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_str.split("\t")[1].split(":")[1]), 4)

        _log.info("Eval round=%s  AUC=%.4f  acc=%.4f",
                  ins.config.get("global_round", "?"), auc, accuracy)

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"AUC": auc, "accuracy": accuracy},
        )


# ─── ClientApp entry point ────────────────────────────────────────────────────

def client_fn(context: Context):
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    os.environ["FLWR_CLIENT_ID"] = str(partition_id)

    cfg  = _build_cfg(context.run_config)
    seed = int(os.environ.get("FLWR_SEED", "42"))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    Partition_Type = int(os.environ.get("PREP_TYPE", "2"))
    dataset_csv    = os.environ.get("DATASET_CSV", "dataset.csv")
    train_method   = os.environ.get("TRAIN_METHOD", cfg.get("train_method", "bagging"))

    train_dmatrix, valid_dmatrix, num_train, num_val, spw = load_data(
        partition_id, num_partitions, seed,
        dataset_csv, Partition_Type, "Access",
    )

    params = dict(cfg["params"])

    # SPW_POWER: 0.5 = sqrt (ARFE/NaiveNA), 0.33 = cbrt (AVC/ARFE_AVC)
    spw_power = float(os.environ.get("SPW_POWER", "0.5"))
    if float(params.get("scale_pos_weight", 1)) == 1.0:
        spw_adj = round(spw ** spw_power, 4)
        params["scale_pos_weight"] = spw_adj
        _log.info("Client %d | spw_raw=%.2f  spw_power=%.2f  spw=%.2f",
                  partition_id, spw, spw_power, spw_adj)

    if cfg["scaled_lr"]:
        params["eta"] = params["eta"] / num_partitions

    _ = get_fl_key()   # derive key once, log source

    return XGBClient(
        train_dmatrix, valid_dmatrix,
        num_train, num_val,
        cfg["local_epochs"], params, train_method,
    )


app = ClientApp(client_fn)
