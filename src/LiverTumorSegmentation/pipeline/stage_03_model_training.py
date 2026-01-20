from LiverTumorSegmentation.config.configuration import ConfigurationManager
from LiverTumorSegmentation.components.training import ModelTrainer
from LiverTumorSegmentation import logger
from pathlib import Path
from typing import Optional

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self, experiment_config_path: Optional[Path] = None):
        self.experiment_config_path = experiment_config_path
    
    def main(self):
        config_manager = ConfigurationManager(experiment_params_path=self.experiment_config_path)
        training_config = config_manager.get_training_config()
        learning_rate = config_manager.params.LEARNING_RATE
        
        # Get experiment-specific parameters if available
        predictions_subdir = getattr(config_manager.params, "PREDICTIONS_SUBDIR", "Dice")
        val_split = getattr(config_manager.params, "VAL_SPLIT", 0.1)
        shuffle_buffer = getattr(config_manager.params, "SHUFFLE_BUFFER", 128)
        
        trainer = ModelTrainer(config=training_config, learning_rate=learning_rate)
        trainer.load_model()
        trainer.prepare_datasets(
            predictions_subdir=predictions_subdir,
            val_split=val_split,
            shuffle_buffer=shuffle_buffer
        )
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
