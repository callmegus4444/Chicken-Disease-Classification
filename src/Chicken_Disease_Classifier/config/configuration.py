# src/Chicken_Disease_Classifier/config/configuration.py
from __future__ import annotations

import os
from pathlib import Path

from Chicken_Disease_Classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Chicken_Disease_Classifier.utils.common import read_yaml, create_directories
from Chicken_Disease_Classifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareCallbacksConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        # read_yaml in your project returns an object with attribute-style access
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir)])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([Path(config.root_dir)])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=tuple(self.params.IMAGE_SIZE),
            params_learning_rate=float(self.params.LEARNING_RATE),
            params_include_top=bool(self.params.INCLUDE_TOP),
            params_weights=str(self.params.WEIGHTS),
            params_classes=int(self.params.CLASSES),
        )

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([Path(model_ckpt_dir), Path(config.tensorboard_root_log_dir)])
        return PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        # same directory training uses
        training_data_dir = Path(self.config.data_ingestion.unzip_dir) / "Chicken-fecal-images"
        create_directories([Path(training.root_dir)])

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),            # artifacts/training/model.keras
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=training_data_dir,
            params_epochs=int(params.EPOCHS),
            params_batch_size=int(params.BATCH_SIZE),
            params_is_augmentation=bool(params.AUGMENTATION),
            params_image_size=tuple(params.IMAGE_SIZE),
        )

    def get_validation_config(self) -> EvaluationConfig:
        """
        Build EvaluationConfig from config.yaml + params.yaml.
        - Uses the trained .keras model path from config.yaml
        - Points training_data to the same directory used in training
        - Supplies full params plus the specific image_size/batch_size used by Evaluation
        """
        path_of_model = Path(self.config.training.trained_model_path)  # artifacts/training/model.keras
        training_data = Path(self.config.data_ingestion.unzip_dir) / "Chicken-fecal-images"

        image_size = list(self.params.IMAGE_SIZE)   # EvaluationConfig expects a list
        batch_size = int(self.params.BATCH_SIZE)

        return EvaluationConfig(
            path_of_model=path_of_model,
            training_data=training_data,
            all_params=dict(self.params),            # keep full params available if needed
            params_image_size=image_size,
            params_batch_size=batch_size,
        )
