from LiverTumorSegmentation.constants import *
from LiverTumorSegmentation.utils.common import read_yaml, create_directories
from LiverTumorSegmentation.entity.config_entity import DataIngestionConfig
from LiverTumorSegmentation.entity.config_entity import PrepareBaseModelConfig
from LiverTumorSegmentation.entity.config_entity import TrainingConfig
from pathlib import Path
from typing import Optional

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        experiment_params_path: Optional[Path] = None
    ):
        self.config = read_yaml(config_filepath)
        
        # Use experiment config if provided, otherwise use default params
        if experiment_params_path is not None:
            self.params = read_yaml(experiment_params_path)
        else:
            self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        create_directories([config.copy_data])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            copy_data=config.copy_data,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            copy_updated_base_model_path=Path(config.copy_updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_base_filters=self.params.BASE_FILTERS,
        )

        return prepare_base_model_config


    def get_training_config(self) -> TrainingConfig:
        """Create and return TrainingConfig from loaded configs."""
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        create_directories([Path(training.root_dir)])

        return TrainingConfig(
                root_dir=Path(training.root_dir),
                trained_model_path=Path(training.trained_model_path),
                updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
                copy_trained_model_path=Path(training.copy_trained_model_path),
                training_data=Path(training.training_data),
                    params_epochs=params.EPOCHS,
                    params_batch_size=params.BATCH_SIZE,
                    params_is_augmentation=params.AUGMENTATION,
                    params_image_size=params.IMAGE_SIZE
            )
