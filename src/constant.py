from pathlib import Path

# Root Directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project Directory

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artificats"

MODELS_DIR = PROJECT_ROOT / "models"


# Dataset

DATASET_NAME = "student_data.csv"
TARGET_COLUMN = "G3"


# Defaults

RANDOM_STATE = 42
TEST_SIZE = 0.2