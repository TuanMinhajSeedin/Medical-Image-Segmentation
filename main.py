import os
from dotenv import load_dotenv
load_dotenv()

# Silence noisy TensorFlow INFO logs in terminal output.
# Must be set BEFORE importing tensorflow (directly or indirectly).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=filter all
# Optional: disable oneDNN verbose info line about custom ops.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

os.environ["MLFLOW_TRACKING_URI"]=os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"]=os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"]=os.getenv("MLFLOW_TRACKING_PASSWORD")

from LiverTumorSegmentation.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from LiverTumorSegmentation.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from LiverTumorSegmentation.pipeline.stage_03_model_training import ModelTrainingPipeline
from LiverTumorSegmentation.pipeline.stage_04_model_evaluation import EvaluationPipeline
from LiverTumorSegmentation import logger


STAGE_NAME="Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Prepare Base Model Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Training Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Evaluation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e