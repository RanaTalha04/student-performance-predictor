import sys
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

from src.utils.common import save_object
from src.utils.model_utils import evaluate_models

from src.entity.config import ModelTrainerConfig
from src.entity.artifact import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def run(
        self,
        transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:

        try:

            logging.info("Starting model training.")

            train_array = transformation_artifact.train_array
            test_array = transformation_artifact.test_array

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            }

            logging.info("Evaluating models.")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = models[best_model_name]

            logging.info(
                f"Best Model: {best_model_name} "
                f"with R2 Score: {model_report[best_model_name]:.4f}"
            )

            save_object(
                self.config.trained_model_path,
                best_model
            )

            train_prediction = best_model.predict(X_train)
            test_prediction = best_model.predict(X_test)

            train_score = r2_score(
                y_train,
                train_prediction
            )

            test_score = r2_score(
                y_test,
                test_prediction
            )

            logging.info("Model training completed successfully.")

            return ModelTrainerArtifact(
                model_path=self.config.trained_model_path,
                train_score=train_score,
                test_score=test_score,
                model_name=best_model_name
            )

        except Exception as e:
            logging.exception("Model training failed.")
            raise CustomException(e, sys)