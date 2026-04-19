from fastapi import FastAPI
from .schemas import PredictionInput, PredictionOutput
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'models/solar_forecaster_v1.joblib'
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Ensure '01_Solar_Forecaster_Training.ipynb' has been run.")


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates time-series features from a datetime index."""
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_year'] = df['Timestamp'].dt.dayofyear
    return df

def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates cyclical features for time-based attributes."""
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
    return df


FEATURES = [
    'Temperature_C',
    'Cloud_Cover_Pct',
    'hour_sin',
    'hour_cos',
    'day_of_year_sin',
    'day_of_year_cos'
]


def generate_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all necessary features for a prediction request DataFrame.
    This includes time features, cyclical features, and deterministic weather placeholders.
    """

    df = create_time_features(df)
    df = create_cyclical_features(df)

    day_of_year = df['day_of_year']
    hour = df['hour']

    seasonal_avg = 25.5 + 17.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_offset = 12 * np.sin(2 * np.pi * (hour - 4) / 24)
    df['Temperature_C'] = seasonal_avg + daily_offset

    base_cloud = 0.05 + 0.1 * np.sin(np.pi * day_of_year / 365)
    is_monsoon = df['Timestamp'].dt.month.isin([7, 8])
    monsoon_effect = 0.5 * (day_of_year % 14) / 14 

    is_spike_day = (day_of_year % 20 == 0)
    spike_effect = 0.6

    df['Cloud_Cover_Pct'] = np.where(is_monsoon, base_cloud + monsoon_effect,
                                     np.where(is_spike_day, base_cloud + spike_effect, base_cloud))
    df['Cloud_Cover_Pct'] = df['Cloud_Cover_Pct'].clip(0, 1)
    return df


app = FastAPI(
    title="SolarCOptimizer - Prediction Backend",
    description="API for forecasting solar generation and grid demand.",
    version="1.0.0"
)

@app.get("/")
async def read_root():
    return {"status": "Prediction Backend is online"}

@app.post("/predict/solar", response_model=PredictionOutput)
async def predict_solar(data: PredictionInput):
    """
    Predicts solar generation for all 24 hours of a given date.
    """

    timestamps = pd.date_range(start=data.prediction_date, periods=24, freq='h')
    df = pd.DataFrame(timestamps, columns=['Timestamp'])

    df_features = generate_features_for_prediction(df)
    predictions = model.predict(df_features[FEATURES]).clip(0)
    return {"predictions": [round(p, 2) for p in predictions.tolist()]}
