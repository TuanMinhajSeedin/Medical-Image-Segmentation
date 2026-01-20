from LiverTumorSegmentation.config.configuration import ConfigurationManager
from LiverTumorSegmentation.components.prepare_base_model import PrepareBaseModel
from LiverTumorSegmentation import logger

STAGE_NAME="Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        preparer = PrepareBaseModel(config=prepare_base_model_config)
        model = preparer.update_base_model()
        # Save the updated base model to the configured updated_base_model_path
        PrepareBaseModel.save_model(prepare_base_model_config.updated_base_model_path, model)
        preparer.copy_model()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e