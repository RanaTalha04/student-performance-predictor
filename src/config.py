from dataclasses import dataclass
from src.constant import  ARTIFICATS_DIR, RAW_DATA_DIR, DATASET_NAME


@dataclass
class DataIngestionConfig: 
    
    raw_data_path = RAW_DATA_DIR / DATASET_NAME
    
    train_data_path = ARTIFICATS_DIR / "train.csv"
    
    test_data_path = ARTIFICATS_DIR / "test.csv"