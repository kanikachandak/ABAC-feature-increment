import os
from nvflare.apis.aux_helper import AuxHelper
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.fl_constant import ReturnCode

class ModelSharer(AuxHelper):
    def __init__(self, model_file_name="model.json"):
        super().__init__()
        self.model_file_name = model_file_name

    def handle_topic(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        if topic != "aux_topic":
            return make_reply(ReturnCode.TOPIC_UNKNOWN)

        task_name = request.get_header("task_name")
        if task_name == "get_model":
            try:
                # The workspace is the secure, temporary directory for the job run
                workspace = fl_ctx.get_engine().get_workspace()
                model_path = os.path.join(workspace.get_app_dir(fl_ctx.get_job_id()), self.model_file_name)

                self.log_info(fl_ctx, f"ModelSharer attempting to read model from: {model_path}")
                
                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        model_bytes = f.read()
                    
                    self.log_info(fl_ctx, "Successfully read model file, preparing reply.")
                    reply = make_reply(ReturnCode.OK)
                    reply["model_data"] = model_bytes
                    return reply
                else:
                    self.log_error(fl_ctx, f"Model file not found at {model_path}")
                    return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

            except Exception as e:
                self.log_exception(fl_ctx, f"Exception in ModelSharer: {e}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
