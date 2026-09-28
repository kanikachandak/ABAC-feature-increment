# server_eval.py
import os
import time
import csv
from pathlib import Path

import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nvflare.apis.dxo import from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.signal import Signal


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
        with open(self.metrics_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "seed",
                    "group",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "inference_time_s",
                    "n_samples",
                    "timestamp",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_running_averages(self, fl_ctx: FLContext):
        try:
            df = pd.read_csv(self.metrics_csv_path)
            if self.metrics_group:
                df = df[df["group"] == self.metrics_group]
            if len(df) == 0:
                return
            means = df[["accuracy", "precision", "recall", "f1", "inference_time_s"]].mean(numeric_only=True)
            n = len(df)
            self.log_info(fl_ctx, "---------------- RUNNING AVERAGES ----------------")
            self.log_info(fl_ctx, f"Seeds aggregated: {n}" + (f" | Group: {self.metrics_group}" if self.metrics_group else ""))
            self.log_info(fl_ctx, f"Accuracy:  {means['accuracy'] * 100:.2f}%")
            self.log_info(fl_ctx, f"Precision: {means['precision']:.4f}")
            self.log_info(fl_ctx, f"Recall:    {means['recall']:.4f}")
            self.log_info(fl_ctx, f"F1-Score:  {means['f1']:.4f}")
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
            y = df[self.label_col]
            X = df.drop(self.label_col, axis=1)
            dtest = xgb.DMatrix(X, label=y)

            # --- TIME INFERENCE ---
            t0 = time.perf_counter()
            preds = self.bst.predict(dtest)
            t1 = time.perf_counter()
            inference_time_s = t1 - t0

            # Binary predictions (round probs)
            predictions = [round(float(v)) for v in preds]

            # --- METRICS (robust to degenerate cases) ---
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions, zero_division=0)
            recall = recall_score(y, predictions, zero_division=0)
            f1 = f1_score(y, predictions, zero_division=0)

            # --- LOG WITH SEED ---
            self.log_info(fl_ctx, "================================================================")
            self.log_info(fl_ctx, f"[Seed {seed}] SERVER-SIDE VALIDATION RESULTS")
            self.log_info(fl_ctx, "----------------------------------------------------------------")
            self.log_info(fl_ctx, f"Accuracy:  {accuracy * 100:.2f}%")
            self.log_info(fl_ctx, f"Precision: {precision:.4f}")
            self.log_info(fl_ctx, f"Recall:    {recall:.4f}")
            self.log_info(fl_ctx, f"F1-Score:  {f1:.4f}")
            self.log_info(fl_ctx, f"InferTime: {inference_time_s:.6f} s")
            self.log_info(fl_ctx, "================================================================")

            # --- APPEND TO CSV ---
            self._append_metrics_csv(
                {
                    "seed": str(seed),
                    "group": self.metrics_group,
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
            try:
                save_dir = Path("/scratch/bmitra/xgb_models/secure")
                save_dir.mkdir(parents=True, exist_ok=True)

                prep_type = os.getenv("PREP_TYPE", "NA")
                partition = os.getenv("PARTITION", "NA")
                seed = os.getenv("EXPERIMENT_SEED", "NA")

                save_path = save_dir / f"PrepType{prep_type}_Partition{partition}_Seed{seed}.json"
                self.bst.save_model(save_path)
                self.log_info(fl_ctx, f"Saved model to {save_path}")
            except Exception as e:
                self.log_exception(fl_ctx, f"Failed to save model: {e}")

            return make_reply(ReturnCode.OK)

        except Exception as e:
            self.log_exception(fl_ctx, f"[Seed {seed}] Exception during server-side validation: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

