from LiverTumorSegmentation.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
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