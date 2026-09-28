import os
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.fl_constant import ReturnCode
from nvflare.app_opt.xgboost.histogram_based_v2.fed_executor import FedXGBHistogramExecutor

class CustomFedXGBHistogramExecutor(FedXGBHistogramExecutor):
    
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal) -> Shareable:
        """
        This method now handles the 'get_model' task by reading the saved model file from the correct path.
        """
        if task_name == "get_model":
            self.log_info(fl_ctx, "Received 'get_model' task from the server.")
            
            try:
                engine = fl_ctx.get_engine()
                workspace = engine.get_workspace()
                job_id = fl_ctx.get_job_id()
                
                # --- THIS IS THE CORRECTED PATH LOGIC ---
                # 1. Get the app-specific directory (e.g., .../app_site-1)
                app_dir = workspace.get_app_dir(job_id)

                # 2. Get the parent job directory (e.g., .../simulate_job)
                job_dir = os.path.dirname(app_dir)
                
                # 3. Construct the correct path to the model file
                model_path = os.path.join(
                    job_dir,
                    self.model_file_name
                )
                # --- END OF CORRECTED PATH LOGIC ---
                
                self.log_info(fl_ctx, f"Attempting to read model from correct path: {model_path}")

                with open(model_path, "rb") as f:
                    model_bytes = f.read()

                self.log_info(fl_ctx, f"Successfully read model file, size: {len(model_bytes)} bytes.")
                reply = make_reply(ReturnCode.OK)
                reply["model_data"] = model_bytes
                return reply
                
            except FileNotFoundError:
                self.log_error(fl_ctx, f"Model file not found at {model_path}. Training might have failed.")
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
            except Exception as e:
                self.log_exception(fl_ctx, f"Error reading model file: {e}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        return super().execute(task_name, shareable, fl_ctx, abort_signal)
