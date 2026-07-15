import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils.common import load_object

from src.constant import (
    MODELS_DIR,
    ARTIFACTS_DIR
)


class PredictionPipeline:

    def __init__(self):
        self.model_path = MODELS_DIR / "model.pkl"
        self.preprocessor_path = ARTIFACTS_DIR / "preprocessor.pkl"

    def predict(self, features: pd.DataFrame):

        try:

            logging.info("Loading preprocessor...")

            preprocessor = load_object(
                self.preprocessor_path
            )

            logging.info("Loading trained model...")

            model = load_object(
                self.model_path
            )

            logging.info("Transforming input data...")

            transformed_data = preprocessor.transform(features)

            logging.info("Making prediction...")

            prediction = model.predict(transformed_data)

            return prediction

        except Exception as e:

            logging.exception("Prediction failed.")

            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        school,
        sex,
        age,
        address,
        famsize,
        Pstatus,
        Medu,
        Fedu,
        Mjob,
        Fjob,
        reason,
        guardian,
        traveltime,
        studytime,
        failures,
        schoolsup,
        famsup,
        paid,
        activities,
        nursery,
        higher,
        internet,
        romantic,
        famrel,
        freetime,
        goout,
        Dalc,
        Walc,
        health,
        absences,
        G1,
        G2,
    ):

        self.school = school
        self.sex = sex
        self.age = age
        self.address = address
        self.famsize = famsize
        self.Pstatus = Pstatus
        self.Medu = Medu
        self.Fedu = Fedu
        self.Mjob = Mjob
        self.Fjob = Fjob
        self.reason = reason
        self.guardian = guardian
        self.traveltime = traveltime
        self.studytime = studytime
        self.failures = failures
        self.schoolsup = schoolsup
        self.famsup = famsup
        self.paid = paid
        self.activities = activities
        self.nursery = nursery
        self.higher = higher
        self.internet = internet
        self.romantic = romantic
        self.famrel = famrel
        self.freetime = freetime
        self.goout = goout
        self.Dalc = Dalc
        self.Walc = Walc
        self.health = health
        self.absences = absences
        self.G1 = G1
        self.G2 = G2

    def get_data_as_dataframe(self):

        try:

            custom_data = {
                "school": [self.school],
                "sex": [self.sex],
                "age": [self.age],
                "address": [self.address],
                "famsize": [self.famsize],
                "Pstatus": [self.Pstatus],
                "Medu": [self.Medu],
                "Fedu": [self.Fedu],
                "Mjob": [self.Mjob],
                "Fjob": [self.Fjob],
                "reason": [self.reason],
                "guardian": [self.guardian],
                "traveltime": [self.traveltime],
                "studytime": [self.studytime],
                "failures": [self.failures],
                "schoolsup": [self.schoolsup],
                "famsup": [self.famsup],
                "paid": [self.paid],
                "activities": [self.activities],
                "nursery": [self.nursery],
                "higher": [self.higher],
                "internet": [self.internet],
                "romantic": [self.romantic],
                "famrel": [self.famrel],
                "freetime": [self.freetime],
                "goout": [self.goout],
                "Dalc": [self.Dalc],
                "Walc": [self.Walc],
                "health": [self.health],
                "absences": [self.absences],
                "G1": [self.G1],
                "G2": [self.G2],
            }

            return pd.DataFrame(custom_data)

        except Exception as e:

            logging.exception("Failed to create dataframe.")

            raise CustomException(e, sys)