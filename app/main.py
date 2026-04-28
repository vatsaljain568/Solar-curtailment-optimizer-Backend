from fastapi import FastAPI, HTTPException
from .schemas import PredictionInput, PredictionOutput
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from .optimizer import create_dispatch_schedule
import joblib
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SOLAR_MODEL_PATH = BASE_DIR / 'models' / 'solar_forecaster_v1.joblib'
try:
    solar_model = joblib.load(SOLAR_MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {SOLAR_MODEL_PATH}. Ensure '01_Solar_Forecaster_Training.ipynb' has been run.")

DEMAND_MODEL_PATH = BASE_DIR / 'models' / 'demand_forecaster_v1.joblib'
try:
    demand_model = joblib.load(DEMAND_MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {DEMAND_MODEL_PATH}. Ensure '02_Demand_Forecaster_Training.ipynb' has been run.")

DATA_PATH = BASE_DIR / 'data' / 'hybrid_park_dataset.csv'
try:
    historical_df = pd.read_csv(
        DATA_PATH,
        parse_dates=['Timestamp'],
        index_col='Timestamp',
        usecols=['Timestamp', 'Demand_MW']
    )
except FileNotFoundError:
    raise RuntimeError(f"Data file not found at {DATA_PATH}. Ensure the data is in the correct location.")


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates time-series features from a datetime index."""
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_year'] = df['Timestamp'].dt.dayofyear
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = (df['Timestamp'].dt.dayofweek >= 5).astype(int)
    df['month'] = df['Timestamp'].dt.month
    return df

def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates cyclical features for time-based attributes."""
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
    return df


SOLAR_FEATURES = [
    'Temperature_C',
    'Cloud_Cover_Pct',
    'hour_sin',
    'hour_cos',
    'day_of_year_sin',
    'day_of_year_cos'
]

DEMAND_FEATURES = [
    'Temperature_C',
    'Demand_Lag_24h',
    'hour_sin',
    'hour_cos',
    'day_of_year_sin',
    'day_of_year_cos',
    'day_of_week',
    'is_weekend'
]


def generate_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Generates features for a solar prediction request."""
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


def generate_demand_features(df: pd.DataFrame, previous_day_demand: list) -> pd.DataFrame:
    """Generates features for a demand prediction request."""
    df = create_time_features(df)
    df = create_cyclical_features(df)

    day_of_year = df['day_of_year']
    hour = df['hour']

    seasonal_avg = 25.5 + 17.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_offset = 12 * np.sin(2 * np.pi * (hour - 4) / 24)
    df['Temperature_C'] = seasonal_avg + daily_offset

    df['Demand_Lag_24h'] = previous_day_demand
    return df


app = FastAPI(
    title="SolarCOptimizer - Prediction Backend",
    description="API for forecasting solar generation, grid demand, and optimizing power dispatch.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
async def read_root():
    return {"status": "Prediction Backend is online"}


@app.post("/predict/solar", response_model=PredictionOutput, tags=["Forecasting"])
async def predict_solar(data: PredictionInput):
    """Predicts solar generation for all 24 hours of a given date."""
    timestamps = pd.date_range(start=data.prediction_date, periods=24, freq='h')
    df = pd.DataFrame(timestamps, columns=['Timestamp'])

    df_features = generate_features_for_prediction(df)
    predictions = solar_model.predict(df_features[SOLAR_FEATURES]).clip(0)
    return {"predictions": [round(p, 2) for p in predictions.tolist()]}


@app.post("/predict/demand", response_model=PredictionOutput, tags=["Forecasting"])
async def predict_demand(data: PredictionInput):
    """
    Predicts grid demand for all 24 hours of a given date.
    To create the 24h-lag feature, this endpoint uses the demand data from the
    same day of the previous year as a proxy, simplifying data requirements.
    """
    previous_year_date = data.prediction_date - timedelta(days=365)
    
    previous_day_demand_series = historical_df.loc[
        historical_df.index.date == previous_year_date, 'Demand_MW'
    ]

    if len(previous_day_demand_series) < 24:
        raise HTTPException(
            status_code=422,
            detail=f"Incomplete or missing historical demand data for the fallback date ({previous_year_date}). Cannot generate lag features."
        )
    
    previous_day_demand = previous_day_demand_series.head(24).tolist()

    timestamps = pd.date_range(start=data.prediction_date, periods=24, freq='h')
    df = pd.DataFrame(timestamps, columns=['Timestamp'])

    df_features = generate_demand_features(df, previous_day_demand)

    predictions = demand_model.predict(df_features[DEMAND_FEATURES]).clip(0)
    return {"predictions": [round(p, 2) for p in predictions.tolist()]}


@app.post("/optimize/schedule", response_model=dict, tags=["Optimization"])
async def optimize_schedule(data: PredictionInput):
    """
    Generates an optimal 24-hour power dispatch schedule for a given date.
    """
    try:
        solar_response = await predict_solar(data)
        demand_response = await predict_demand(data)
        hourly_solar = solar_response['predictions']
        hourly_demand = demand_response['predictions']
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    output_json, error_detail = create_dispatch_schedule(
        prediction_date=data.prediction_date,
        hourly_solar=hourly_solar,
        hourly_demand=hourly_demand
    )

    if error_detail:
        raise HTTPException(status_code=500, detail=error_detail)

    return jsonable_encoder(output_json)
