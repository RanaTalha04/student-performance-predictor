from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.prediction_pipeline import (
    PredictionPipeline,
    CustomData
)

app = FastAPI(
    title="Student Performance Predictor",
    version="1.0.0"
)


class StudentRequest(BaseModel):

    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int
    G1: int
    G2: int


@app.get("/")
def home():
    return {"message": "Student Performance Predictor API"}


@app.post("/predict")
def predict(request: StudentRequest):

    data = CustomData(
        **request.model_dump()
    )

    df = data.get_data_as_dataframe()

    prediction = PredictionPipeline().predict(df)

    return {
        "Predicted Final Grade": float(prediction[0])
    }