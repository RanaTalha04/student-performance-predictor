import os
from pathlib import Path
import logging
from datetime import datetime

PROJECT_ROOT = Path.cwd()
LOG_FILE = f"{datetime.now().strftime('%y_%m_%d_%H_%M_%S')}.log"
log_path = PROJECT_ROOT / "logs"
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = log_path / LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s | %(levelname)s | %(filename)s: %(lineno)d | %(message)s",
    level=logging.INFO
)
