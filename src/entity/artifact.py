import numpy as np
from pathlib import Path

from dataclasses import dataclass

@dataclass(frozen=True)
class DataTransformationArtifact:

    train_array: np.ndarray
    test_array: np.ndarray

    preprocessor_path: Path
    

@dataclass(frozen=True)
class ModelTrainerArtifact:

    model_path: Path

    train_score: float

    test_score: float

    model_name: str
    
    
@dataclass(frozen=True)
class DataIngestionArtifact:
    train_data_path: Path
    test_data_path: Path