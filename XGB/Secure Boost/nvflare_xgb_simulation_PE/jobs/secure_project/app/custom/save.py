import os
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.handler import Handler


class XGBoostValidator(Handler):
    """
    A custom Handler to validate the final XGBoost model on the server.
    Enhanced with comprehensive debug logging.
    """

    def __init__(self, test_data_path: str, label_col: str = "y"):
        """
        Args:
            test_data_path (str): The absolute path to the test CSV file.
            label_col (str): The name of the label column in the test data.
        """
        super().__init__()
        self.test_data_path = test_data_path
        self.label_col = label_col
        self.bst_model = None
        self.evaluation_triggered = False
        
        # Debug: Print to console as well
        print(f"XGBoostValidator initialized with test_data_path: {test_data_path}")
        print(f"XGBoostValidator initialized with label_col: {label_col}")

    def _preprocess_data(self, df: pd.DataFrame, fl_ctx: FLContext) -> (pd.DataFrame, pd.Series):
        """
        Preprocesses the raw dataframe to be ready for XGBoost.
        - Converts the label column to binary (0/1).
        - Applies one-hot encoding to categorical features.
        """
        print("Starting data preprocessing for evaluation...")
        self.log_info(fl_ctx, "Starting data preprocessing for evaluation...")

        # 1. Separate features and labels
        if self.label_col not in df.columns:
            error_msg = f"Label column '{self.label_col}' not found in the data."
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

        y = df[self.label_col]
        X = df.drop(self.label_col, axis=1)

        # 2. Convert label 'Yes'/'No' to 1/0
        print(f"Converting label column '{self.label_col}' to binary")
        self.log_info(fl_ctx, f"Converting label column '{self.label_col}' to binary (Yes=1, No=0).")
        y = y.apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

        # 3. Apply one-hot encoding to all feature columns
        print("Applying one-hot encoding to feature columns.")
        self.log_info(fl_ctx, "Applying one-hot encoding to feature columns.")
        X_encoded = pd.get_dummies(X, columns=X.columns, dummy_na=False)

        print(f"Data preprocessed. Number of features after encoding: {len(X_encoded.columns)}")
        self.log_info(fl_ctx, f"Data preprocessed. Number of features after encoding: {len(X_encoded.columns)}")
        return X_encoded, y

    def _evaluate_model(self, fl_ctx: FLContext):
        """
        The core logic for loading the model and running evaluation.
        """
        print("=== _evaluate_model called ===")
        
        # Ensure evaluation runs only once
        if self.evaluation_triggered:
            print("Evaluation already triggered, skipping...")
            return
        self.evaluation_triggered = True

        print("--- Triggering server-side evaluation ---")
        self.log_info(fl_ctx, f"--- Triggering server-side evaluation ---")

        # Try multiple possible model paths
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        job_id = fl_ctx.get_job_id()
        app_dir = workspace.get_app_dir(job_id)
        
        print(f"Job ID: {job_id}")
        print(f"App directory: {app_dir}")
        
        # Check multiple possible model locations
        possible_paths = [
            os.path.join(app_dir, "xgboost_model.json"),
            os.path.join(app_dir, "models", "xgboost_model.json"),
            os.path.join(app_dir, "model.json"),
            os.path.join(app_dir, "final_model.json")
        ]
        
        print("Checking for model files in:")
        for path in possible_paths:
            print(f"  {path} - {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")
        
        # Try to list all files in app_dir
        try:
            print(f"Files in app directory {app_dir}:")
            if os.path.exists(app_dir):
                for root, dirs, files in os.walk(app_dir):
                    level = root.replace(app_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"{subindent}{file}")
            else:
                print(f"App directory {app_dir} does not exist!")
        except Exception as e:
            print(f"Error listing files: {e}")

        # Find the model file
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if not model_path:
            error_msg = f"Model file not found in any expected location. Evaluation skipped."
            print(f"ERROR: {error_msg}")
            self.log_error(fl_ctx, error_msg)
            return

        # Load the trained XGBoost model
        print(f"Loading model from: {model_path}")
        self.log_info(fl_ctx, f"Loading model from: {model_path}")
        
        try:
            self.bst_model = xgb.Booster()
            self.bst_model.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            error_msg = f"Error loading model: {e}"
            print(f"ERROR: {error_msg}")
            self.log_error(fl_ctx, error_msg)
            return

        # Load and prepare the test data
        print(f"Loading test data from: {self.test_data_path}")
        self.log_info(fl_ctx, f"Loading test data from: {self.test_data_path}")
        
        if not os.path.exists(self.test_data_path):
            error_msg = f"Test data file not found at {self.test_data_path}. Evaluation skipped."
            print(f"ERROR: {error_msg}")
            self.log_error(fl_ctx, error_msg)
            return

        try:
            df = pd.read_csv(self.test_data_path)
            print(f"Test data loaded. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            X_test_processed, y_test = self._preprocess_data(df, fl_ctx)

            # Align columns of test data with model's feature names
            print("Aligning test data columns with model's feature names.")
            self.log_info(fl_ctx, "Aligning test data columns with model's feature names.")
            
            model_features = self.bst_model.feature_names
            print(f"Model expects {len(model_features)} features")
            print(f"Test data has {len(X_test_processed.columns)} features")
            
            X_test_aligned = X_test_processed.reindex(columns=model_features, fill_value=0)
            print(f"Aligned test data shape: {X_test_aligned.shape}")

            dtest = xgb.DMatrix(X_test_aligned)

            # Run predictions
            print("Running predictions on the processed test dataset...")
            self.log_info(fl_ctx, "Running predictions on the processed test dataset...")
            
            preds_proba = self.bst_model.predict(dtest)
            pred_labels = (preds_proba > 0.5).astype(int)
            print(f"Predictions completed. Shape: {preds_proba.shape}")

            # Calculate metrics
            accuracy = accuracy_score(y_test, pred_labels)
            auc = roc_auc_score(y_test, preds_proba)

            # Log the results (both to console and FL logs)
            results_msg = f"""
------ Server-side Evaluation Results ------
Test Accuracy: {accuracy:.4f}
Test AUC: {auc:.4f}
------------------------------------------
            """
            print(results_msg)
            self.log_info(fl_ctx, results_msg)

        except Exception as e:
            error_msg = f"An error occurred during evaluation: {e}"
            print(f"ERROR: {error_msg}")
            self.log_error(fl_ctx, error_msg)
            import traceback
            traceback.print_exc()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """
        This method is called when events occur during the FL run.
        """
        print(f"=== Event received: {event_type} ===")
        
        # Log all events we receive
        self.log_info(fl_ctx, f"XGBoostValidator received event: {event_type}")
        
        if event_type == EventType.START_RUN:
            print("--- START_RUN event caught ---")
            self.log_info(fl_ctx, f"--- XGBoostValidator loaded and caught event {event_type} ---")
        
        elif event_type == EventType.ROUND_DONE:
            print(f"--- ROUND_DONE event caught ---")
            self.log_info(fl_ctx, f"--- Round completed ---")
            
        elif event_type == EventType.ABOUT_TO_END_RUN:
            print("--- ABOUT_TO_END_RUN event caught ---")
            self.log_info(fl_ctx, f"--- Caught event {event_type} ---")
            self._evaluate_model(fl_ctx)
        
        elif event_type == EventType.END_RUN:
            print("--- END_RUN event caught ---")
            self.log_info(fl_ctx, f"--- Caught event {event_type} ---")
            self._evaluate_model(fl_ctx)
            
        else:
            print(f"--- Other event caught: {event_type} ---")
