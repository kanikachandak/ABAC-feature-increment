import xgboost as xgb
import pandas as pd
# Import the new metrics from Scikit-learn
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

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != "validate":
            self.log_error(fl_ctx, f"Validator received unknown task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

        try:
            dxo = from_shareable(shareable)
            model_bytes = dxo.data.get("model_data")
            
            self.bst = xgb.Booster()
            self.bst.load_model(bytearray(model_bytes))

            df = pd.read_csv(self.test_data_path)
            y = df[self.label_col]
            X = df.drop(self.label_col, axis=1)

            dtest = xgb.DMatrix(X, label=y)

            preds = self.bst.predict(dtest)
            # Rounding probabilities to get binary 0/1 predictions
            predictions = [round(value) for value in preds]
            
            # --- CALCULATE ALL METRICS ---
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions)
            recall = recall_score(y, predictions)
            f1 = f1_score(y, predictions)

            # --- LOG ALL METRICS ---
            self.log_info(fl_ctx, "================================================================")
            self.log_info(fl_ctx, f"SERVER-SIDE VALIDATION RESULTS")
            self.log_info(fl_ctx, "----------------------------------------------------------------")
            self.log_info(fl_ctx, f"Accuracy: {accuracy * 100:.2f}%")
            self.log_info(fl_ctx, f"Precision: {precision:.4f}")
            self.log_info(fl_ctx, f"Recall: {recall:.4f}")
            self.log_info(fl_ctx, f"F1-Score: {f1:.4f}")
            self.log_info(fl_ctx, "================================================================")
            
            return make_reply(ReturnCode.OK)

        except Exception as e:
            self.log_exception(fl_ctx, f"Exception during server-side validation: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
