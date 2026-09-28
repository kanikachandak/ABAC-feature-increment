import time
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.controller_spec import Task, ClientTask
# --- THIS IS THE CORRECTED IMPORT ---
from nvflare.apis.dxo import DXO, DataKind  # 'to_shareable' is removed from the import
# --- END OF CORRECTION ---
from nvflare.app_opt.xgboost.histogram_based_v2.fed_controller import XGBFedController

class CustomXGBController(XGBFedController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_evaluator_id = "server_evaluator"
        self._final_model_received = False

    def _process_get_model_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        reply = client_task.result
        client_name = client_task.client.name
        
        if not reply or reply.get_return_code() != ReturnCode.OK:
            self.log_error(fl_ctx, f"Client '{client_name}' failed to return model. Aborting.")
            self._final_model_received = True
            return

        final_model = reply.get("model_data")
        self.log_info(fl_ctx, f"Callback received final model from client '{client_name}'.")

        # Create a DXO with the model data.
        dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data={"model_data": final_model}
        )

        # Convert the DXO object into a properly formatted Shareable object using the method.
        validate_task_shareable = dxo.to_shareable()

        self.log_info(fl_ctx, f"Directly calling execute on server validator: '{self.server_evaluator_id}'")
        engine = fl_ctx.get_engine()
        validator = engine.get_component(self.server_evaluator_id)

        if validator:
            validator.execute(
                task_name="validate",
                shareable=validate_task_shareable,
                fl_ctx=fl_ctx,
                abort_signal=None
            )
        else:
            self.log_error(fl_ctx, f"Could not find server component '{self.server_evaluator_id}' to run validation.")
        
        self.log_info(fl_ctx, "Validation task complete. Check server logs for evaluation results.")
        self._final_model_received = True

    def control_flow(self, abort_signal, fl_ctx: FLContext):
        try:
            self.log_info(fl_ctx, "Starting standard federated XGBoost training...")
            super().control_flow(abort_signal, fl_ctx)
            self.log_info(fl_ctx, "Standard training finished successfully.")
            
            self.log_info(fl_ctx, "Starting post-training validation: requesting model from client...")
            target_client = self._engine.get_clients()[0]

            task = Task(
                name="get_model",
                data=Shareable(),
                result_received_cb=self._process_get_model_reply
            )

            self.broadcast_and_wait(
                task=task,
                targets=[target_client],
                min_responses=1,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal
            )

            for _ in range(15):
                if self._final_model_received:
                    self.log_info(fl_ctx, "Post-training validation complete. Workflow finished.")
                    break
                time.sleep(1)

        except Exception as e:
            self.log_exception(fl_ctx, f"An exception occurred during the workflow: {e}")
