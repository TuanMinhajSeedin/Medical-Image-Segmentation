from LiverTumorSegmentation.config.configuration import ConfigurationManager
from LiverTumorSegmentation.components.training import ModelTrainer
from LiverTumorSegmentation import logger

STAGE_NAME="Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config_manager = ConfigurationManager()
        training_config = config_manager.get_training_config()
        learning_rate = config_manager.params.LEARNING_RATE
        
        trainer = ModelTrainer(config=training_config, learning_rate=learning_rate)
        trainer.load_model()
        trainer.prepare_datasets(predictions_subdir="Dice")
        trainer.train()
        trainer.copy_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>  {STAGE_NAME} started <<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>  {STAGE_NAME} completed <<<<<<<\n\n x===========x")
    except Exception as e:
        logger.exception(e)
        raise e
