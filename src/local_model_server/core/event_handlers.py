import os
from typing import Callable

from fastapi import FastAPI
from modelscope import pipeline
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.input_output import get_task_schemas, get_task_input_examples
from modelscope.utils.logger import get_logger

# control the model start stop

logger = get_logger()


def _create_pipeline(model_path: str, revision: str = None,
                     llm_first: bool = False,
                     device: str = None):
    config_file_path = os.path.join(model_path, ModelFile.CONFIGURATION)
    cfg = Config.from_file(config_file_path)
    return pipeline(
        task=cfg.task,
        model=model_path,
        model_revision=revision,
        device=device or "gpu",
        llm_first=llm_first, )


def _startup_model(app: FastAPI) -> None:
    logger.info('load model and create pipeline')
    state = app.state
    args = state.args
    state.pipeline = _create_pipeline(args.model_id,
                                      args.revision,
                                      args.llm_first,
                                      args.device)
    info = {
        "task_name": state.pipeline.group_key,
        "schema": get_task_schemas(state.pipeline.group_key)
    }
    state.pipeline_info = info
    state.pipeline_sample = get_task_input_examples(state.pipeline.group_key)
    logger.info('pipeline created.')


def _shutdown_model(app: FastAPI) -> None:
    app.state.pipeline = None
    logger.info('shutdown model service')


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        _startup_model(app)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_model(app)

    return shutdown
