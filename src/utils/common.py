import sys
import joblib

from pathlib import Path

from src.logger import logging
from src.exception import CustomException



def save_object(file_path: Path, obj) -> None:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved successfully: {file_path}")
        
    except Exception as e:
        logging.error("Failed to save the object")
        raise CustomException(e, sys)
    


def load_object(file_path: Path):
    
    try:
        
        obj = joblib.load(file_path)
        logging.info(f"Object loaded successfully {file_path}")
        return obj
    
    except Exception as e:
        logging.error(f"Failed to load the object")
        raise CustomException(e, sys)
    