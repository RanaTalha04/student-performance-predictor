from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)


class TrainingPipeline:

    def run(self):

        ingestion = DataIngestion(
            DataIngestionConfig()
        )

        ingestion_artifact = ingestion.run()

        transformation = DataTransformation(
            DataTransformationConfig()
        )

        transformation_artifact = transformation.run(
            ingestion_artifact.train_data_path,
            ingestion_artifact.test_data_path,
        )

        trainer = ModelTrainer(
            ModelTrainerConfig()
        )

        trainer_artifact = trainer.run(
            transformation_artifact
        )

        return trainer_artifact