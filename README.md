# Student Performance Predictor

This project is a machine learning-based student performance prediction system. It includes a training pipeline for building a regression model and a FastAPI application for serving predictions through an API.

## Overview

The repository is designed to help you:

- load student data from the raw dataset
- preprocess and transform the input features
- train a predictive model for final grade estimation
- expose the model through a REST API for inference

## Project Structure

```text
student-performance-predictor/
├── app.py                  # FastAPI application entry point
├── README.md               # Project documentation
├── pyproject.toml          # Project metadata and dependencies
├── data/
│   └── raw/                # Raw input dataset files
├── src/
│   ├── components/         # Data ingestion, transformation, and model training
│   ├── entity/             # Config and artifact entity classes
│   ├── pipeline/           # Training and prediction pipelines
│   ├── utils/              # Helper utilities and model loading logic
│   ├── constant.py         # Project-wide constants
│   ├── exception.py        # Custom exception classes
│   └── logger.py           # Logging setup
├── models/                 # Trained model artifacts
├── notebooks/              # EDA and experimentation notebooks
└── artificats/             # Intermediate/generated artifacts
```

## Requirements

- Python 3.10+
- uv (recommended) or pip

## Installation

Create and activate a virtual environment, then install the project dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If you prefer pip, install the dependencies directly:

```bash
pip install fastapi[standard] joblib matplotlib notebook numpy pandas scikit-learn seaborn
```

## Running the API

Start the FastAPI server locally:

```bash
uvicorn app:app --reload
```

The API will be available at:

- http://127.0.0.1:8000/docs for Swagger UI
- http://127.0.0.1:8000/redoc for ReDoc

## Prediction Endpoint

Send a POST request to `/predict` with student feature values in JSON format.

Example:

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

## Training the Model

The project includes training and prediction pipelines under [src/pipeline](src/pipeline). Update the configuration and dataset paths as needed, then run the training workflow from your environment to generate the model artifacts.

## Notes

This repository provides a solid starting point for a student performance prediction project. You can extend it further by improving preprocessing, trying additional models, adding evaluation metrics, and deploying it to a cloud service.
