# Student Performance Predictor

A machine learning system that predicts student academic performance (final grade) from demographic, social, and school-related features. The project includes a full training pipeline and a FastAPI service for real-time inference.

## Overview

This repository is designed to:
- Load and validate raw student data
- Preprocess and transform input features (encoding, scaling, handling missing values)
- Train and evaluate a regression model for final grade estimation
- Serve the trained model through a REST API for inference

## Project Structure

```text
student-performance-predictor/
├── app.py                     # FastAPI application entry point
├── README.md                  # Project documentation
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependency versions
├── .python-version            # Python version pin
├── .gitignore
├── config/
│   └── config.yaml            # Data paths, split ratios, model hyperparameters
├── data/
│   └── raw/
│       └── student_data.csv   # Raw input dataset
├── artifacts/                 # Generated pipeline outputs (preprocessor, train/test splits)
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
├── models/
│   └── model.pkl              # Final trained model
├── logs/                      # Runtime logs
├── notebooks/                 # EDA and experimentation notebooks
├── tests/                     # Unit tests
└── src/
    ├── components/            # Data ingestion, transformation, and model training
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    ├── entity/                # Config and artifact entity (dataclass) definitions
    │   ├── artifact.py
    │   └── config.py
    ├── pipeline/               # Training and prediction pipelines
    │   ├── training_pipeline.py
    │   └── prediction_pipeline.py
    ├── utils/                  # Shared helpers
    │   ├── common.py
    │   └── model_utils.py
    ├── constant.py             # Project-wide constants
    ├── exception.py            # Custom exception classes
    └── logger.py               # Logging setup
```

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

Using `uv` (recommended, matches the included `uv.lock`):

```bash
uv venv
source .venv/bin/activate
uv sync
```

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Pipeline settings (dataset paths, train/test split ratio, random seed, model hyperparameters) live in `config/config.yaml` rather than being hardcoded. Update this file before running training if you want to change:

- Raw data location
- Train/test split ratio
- Target column
- Model type and hyperparameters
- Output paths for artifacts and the final model

## Training the Model

Run the training pipeline to generate a fresh preprocessor and model:

```bash
python -m src.pipeline.training_pipeline
```

This will:
1. Ingest raw data from `data/raw/`
2. Split it into train/test sets and write them to `artifacts/`
3. Fit and save a preprocessing pipeline to `artifacts/preprocessor.pkl`
4. Train a regression model and save it to `models/model.pkl`
5. Write run logs to `logs/`

## Running the API

Start the FastAPI server locally:

```bash
uvicorn app:app --reload
```

The API will be available at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Prediction Endpoint

Send a POST request to `/predict` with student feature values in JSON format:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "school": "GP",
    "sex": "F",
    "age": 16,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "A",
    "Medu": 4,
    "Fedu": 4,
    "Mjob": "at_home",
    "Fjob": "teacher",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 2,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "no",
    "nursery": "yes",
    "higher": "yes",
    "internet": "no",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 4,
    "Dalc": 1,
    "Walc": 1,
    "health": 3,
    "absences": 6,
    "G1": 12,
    "G2": 13
  }'
```

Example response:

```json
{
  "predicted_grade": 13.42
}
```

## Roadmap

Ideas for extending this project further:
- Add cross-validation and model comparison (e.g., RandomForest, XGBoost, linear models) with logged metrics
- Track experiments (MLflow or similar) instead of overwriting a single `models/model.pkl`
- Containerize the API with a `Dockerfile` for deployment
- Add CI (GitHub Actions) to run tests and linting on every push
- Deploy to a cloud service (e.g., Render, AWS, GCP)

## License

Specify your license here (e.g., MIT).
