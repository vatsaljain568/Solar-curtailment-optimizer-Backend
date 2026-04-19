from pydantic import BaseModel
from datetime import datetime, date
from typing import List

class PredictionInput(BaseModel):
    """
    Input schema for prediction, expecting a single date.
    """
    prediction_date: date

class PredictionOutput(BaseModel):
    """
    Output schema for prediction, returning a list of floats.
    """
    predictions: List[float]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}