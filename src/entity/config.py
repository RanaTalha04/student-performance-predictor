from pathlib import Path
from dataclasses import dataclass
from src.constant import ARTIFICATS_DIR, RAW_DATA_DIR, DATASET_NAME, MODELS_DIR


@dataclass(frozen=True)
class DataIngestionConfig: 
    
    raw_data_path = RAW_DATA_DIR / DATASET_NAME
    
    train_data_path = ARTIFICATS_DIR / "train.csv"
    
    test_data_path = ARTIFICATS_DIR / "test.csv"
    

@dataclass(frozen=True)
class DataTransformationConfig:
    
    preprocessor_path: Path = ARTIFICATS_DIR / "preprocessor.pkl"


@dataclass(frozen=True)
class ModelTrainerConfig:

    trained_model_path = MODELS_DIR / "model.pkl"
    
