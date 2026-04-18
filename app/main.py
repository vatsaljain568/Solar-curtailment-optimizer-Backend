from fastapi import FastAPI

app = FastAPI(
    title="SolarCOptimizer - Prediction Brain",
    description="API for forecasting solar generation and grid demand.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "Prediction Brain is online"}