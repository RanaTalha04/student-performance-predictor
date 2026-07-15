import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils.common import save_object

from src.entity.config import DataTransformationConfig
from src.entity.artifact import DataTransformationArtifact

from src.constant import TARGET_COLUMN


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config    
        
        
    def get_data_transformer(self, numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:

        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features)
            ]
        )

        return preprocessor
    
    def run(self, train_data_path, test_data_path) -> DataTransformationArtifact:

        try:

            logging.info("Reading train and test datasets.")

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(f"{TARGET_COLUMN} not found in training data.")

            if TARGET_COLUMN not in test_df.columns:
                raise ValueError(f"{TARGET_COLUMN} not found in testing data.")

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            numerical_features = X_train.select_dtypes(
                exclude=["object"]
            ).columns.tolist()

            categorical_features = X_train.select_dtypes(
                include=["object"]
            ).columns.tolist()

            logging.info("Creating preprocessing pipeline.")

            preprocessor = self.get_data_transformer(
                numerical_features,
                categorical_features
            )

            X_train_transformed = preprocessor.fit_transform(X_train)

            X_test_transformed = preprocessor.transform(X_test)

            train_array = np.c_[X_train_transformed, y_train.to_numpy()]

            test_array = np.c_[X_test_transformed, y_test.to_numpy()]

            save_object(
                self.config.preprocessor_path,
                preprocessor
            )

            logging.info("Preprocessor saved successfully.")

            return DataTransformationArtifact(
                train_array=train_array,
                test_array=test_array,
                preprocessor_path=self.config.preprocessor_path
            )

        except Exception as e:
            logging.exception("Data transformation failed.")
            
            raise CustomException(e, sys)