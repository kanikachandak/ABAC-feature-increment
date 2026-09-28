# server_eval.py
# Server-side validation + metrics/model saving for NVFlare XGBoost simulation.
#
# Key env vars (set by run_simulation.py):
# - METRICS_CSV_PATH: where to append per-seed validation metrics CSV
# - METRICS_GROUP: optional tag (e.g., "prep2")
# - CASE_NAME: used for model naming (e.g., "University2_SBP_ARFE")
# - DATASET, PARTITION, ENCODING, PREP_TYPE, EXPERIMENT_SEED
# - MODEL_SAVE_DIR: directory to save final models

import os
import time
import csv
from pathlib import Path

import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from nvflare.apis.dxo import from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.signal import Signal


def _safe_slug(s: str) -> str:
    """Make a filename-safe slug."""
    if s is None:
        return "model"
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch in (" ", "/", "\\", ":", "|", "+"):
            keep.append("_")
        # otherwise: drop
    slug = "".join(keep).strip("_")
    return slug or "model"


def _coerce_binary_labels(y_series: pd.Series) -> pd.Series:
    """Coerce YES/NO, 0/1, True/False style labels to 0/1 ints when possible."""
    y = y_series.copy()
    if y.dtype == object:
        upper = y.astype(str).str.strip().str.upper()
        mapped = upper.map({"NO": 0, "YES": 1, "FALSE": 0, "TRUE": 1, "0": 0, "1": 1})
        # If mapping failed for some rows, try raw int conversion
        if mapped.isna().any():
            try:
                mapped = pd.to_numeric(y, errors="coerce")
            except Exception:
                pass
        y = mapped
    try:
        y = y.astype(int)
    except Exception:
        # Last resort: keep as-is (metrics may still work for accuracy/precision/recall if consistent)
        pass
    return y


class XGBoostValidator(Executor):
    def __init__(self, test_data_path: str, label_col: str):
        super().__init__()
        self.test_data_path = test_data_path
        self.label_col = label_col
        self.bst = None

        # CSV path where we accumulate per-seed results.
        # Prefer env var from the runner; else write alongside CWD.
        default_csv = Path(os.getcwd()) / "server_metrics.csv"
        self.metrics_csv_path = Path(os.getenv("METRICS_CSV_PATH", str(default_csv)))

        # Optional grouping tag (e.g., prep type) so you can filter later
        self.metrics_group = os.getenv("METRICS_GROUP", "")

    def _append_metrics_csv(self, row: dict):
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.metrics_csv_path.exists()

        fieldnames = [
            "case_name",
            "dataset",
            "partition",
            "encoding",
            "prep_type",
            "seed",
            "group",
            "auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "inference_time_s",
            "n_samples",
            "timestamp",
        ]

        with open(self.metrics_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_running_averages(self, fl_ctx: FLContext):
        if not self.metrics_csv_path.exists():
            return
        try:
            df = pd.read_csv(self.metrics_csv_path)
            if self.metrics_group and "group" in df.columns:
                df = df[df["group"] == self.metrics_group]
            if len(df) == 0:
                return

            metric_cols = ["auc", "accuracy", "precision", "recall", "f1", "inference_time_s"]
            existing = [c for c in metric_cols if c in df.columns]
            means = df[existing].mean(numeric_only=True)
            n = len(df)

            suffix = f" | Group: {self.metrics_group}" if self.metrics_group else ""
            self.log_info(fl_ctx, "---------------- RUNNING AVERAGES ----------------")
            self.log_info(fl_ctx, f"Seeds aggregated: {n}{suffix}")
            if "auc" in means:
                self.log_info(fl_ctx, f"AUC:       {means['auc']:.6f}")
            if "accuracy" in means:
                self.log_info(fl_ctx, f"Accuracy:  {means['accuracy'] * 100:.2f}%")
            if "precision" in means:
                self.log_info(fl_ctx, f"Precision: {means['precision']:.6f}")
            if "recall" in means:
                self.log_info(fl_ctx, f"Recall:    {means['recall']:.6f}")
            if "f1" in means:
                self.log_info(fl_ctx, f"F1-Score:  {means['f1']:.6f}")
            if "inference_time_s" in means:
                self.log_info(fl_ctx, f"InferTime: {means['inference_time_s']:.6f} s")
            self.log_info(fl_ctx, "--------------------------------------------------")
        except Exception as e:
            self.log_exception(fl_ctx, f"Failed to compute running averages: {e}")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != "validate":
            self.log_error(fl_ctx, f"Validator received unknown task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

        seed = os.getenv("EXPERIMENT_SEED", "NA")

        try:
            dxo = from_shareable(shareable)
            model_bytes = dxo.data.get("model_data")
            self.bst = xgb.Booster()
            self.bst.load_model(bytearray(model_bytes))

            df = pd.read_csv(self.test_data_path)
            y_raw = df[self.label_col]
            y = _coerce_binary_labels(y_raw)
            X = df.drop(self.label_col, axis=1)
            dtest = xgb.DMatrix(X, label=y)

            # --- TIME INFERENCE ---
            t0 = time.perf_counter()
            preds_proba = self.bst.predict(dtest)
            t1 = time.perf_counter()
            inference_time_s = t1 - t0

            # Binary predictions (threshold at 0.5)
            pred_labels = (pd.Series(preds_proba) >= 0.5).astype(int)

            # --- METRICS (robust to degenerate cases) ---
            accuracy = accuracy_score(y, pred_labels)
            precision = precision_score(y, pred_labels, zero_division=0)
            recall = recall_score(y, pred_labels, zero_division=0)
            f1 = f1_score(y, pred_labels, zero_division=0)

            try:
                auc = roc_auc_score(y, preds_proba)
            except Exception:
                auc = float("nan")

            # --- LOG WITH SEED ---
            self.log_info(fl_ctx, "================================================================")
            self.log_info(fl_ctx, f"[Seed {seed}] SERVER-SIDE VALIDATION RESULTS")
            self.log_info(fl_ctx, "----------------------------------------------------------------")
            self.log_info(fl_ctx, f"AUC:       {auc:.6f}")
            self.log_info(fl_ctx, f"Accuracy:  {accuracy * 100:.2f}%")
            self.log_info(fl_ctx, f"Precision: {precision:.6f}")
            self.log_info(fl_ctx, f"Recall:    {recall:.6f}")
            self.log_info(fl_ctx, f"F1-Score:  {f1:.6f}")
            self.log_info(fl_ctx, f"InferTime: {inference_time_s:.6f} s")
            self.log_info(fl_ctx, "================================================================")

            # --- APPEND TO CSV ---
            case_name = os.getenv("CASE_NAME", "")
            dataset = os.getenv("DATASET", "")
            partition = os.getenv("PARTITION", "")
            encoding = os.getenv("ENCODING", "")
            prep_type = os.getenv("PREP_TYPE", "")

            self._append_metrics_csv(
                {
                    "case_name": case_name,
                    "dataset": dataset,
                    "partition": partition,
                    "encoding": encoding,
                    "prep_type": str(prep_type),
                    "seed": str(seed),
                    "group": self.metrics_group,
                    "auc": f"{auc:.6f}" if auc == auc else "",
                    "accuracy": f"{accuracy:.6f}",
                    "precision": f"{precision:.6f}",
                    "recall": f"{recall:.6f}",
                    "f1": f"{f1:.6f}",
                    "inference_time_s": f"{inference_time_s:.9f}",
                    "n_samples": str(len(df)),
                    "timestamp": f"{time.time():.3f}",
                }
            )

            self._log_running_averages(fl_ctx)

            # --- SAVE MODEL ---
            try:
                model_dir_env = os.getenv("MODEL_SAVE_DIR", "")
                if model_dir_env:
                    save_dir = Path(model_dir_env)
                else:
                    save_dir = Path(os.getcwd()) / "xgb_models" / "secure"
                save_dir.mkdir(parents=True, exist_ok=True)

                prep_type = os.getenv("PREP_TYPE", "NA")
                partition = os.getenv("PARTITION", "NA")
                seed = os.getenv("EXPERIMENT_SEED", "NA")
                case_name = os.getenv("CASE_NAME", "")

                base_name = case_name or f"PrepType{prep_type}_Partition{partition}"
                base_name = _safe_slug(base_name)

                save_path = save_dir / f"{base_name}_Seed{seed}.json"
                self.bst.save_model(save_path)
                self.log_info(fl_ctx, f"Saved model to {save_path}")
            except Exception as e:
                self.log_exception(fl_ctx, f"Failed to save model: {e}")

            return make_reply(ReturnCode.OK)

        except Exception as e:
            self.log_exception(fl_ctx, f"[Seed {seed}] Exception during server-side validation: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
