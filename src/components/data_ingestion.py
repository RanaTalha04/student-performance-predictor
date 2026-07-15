import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.entity.config import DataIngestionConfig
from src.entity.artifact import DataIngestionArtifact
from src.constant import RANDOM_STATE, TEST_SIZE,TARGET_COLUMN

from sklearn.model_selection import train_test_split



class DataIngestion: 
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
        
    def run(self):
        
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info("Dataset loaded successfully")
            
            train_df, test_df = train_test_split(
                df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )
            
            train_df.to_csv(    
                self.config.train_data_path,
                index=False
            )
            
            test_df.to_csv(
                self.config.test_data_path,
                index=False
            )
            
            return DataIngestionArtifact(
                train_data_path=self.config.train_data_path,
                test_data_path=self.config.test_data_path,
            )
            
        except Exception as e:
            
            logging.exception("Data ingestion failed.")
            raise CustomException(e, sys)