# SolarCOptimizer - Prediction Brain

This repository contains the machine learning backend for the **SolarCOptimizer** project, part of the Google Solution Challenge 2026.

## Overview

The core component is a FastAPI application that serves two predictive models:
1.  **`/predict/demand`**: Predicts the next 24 hours of electricity demand (MW).
2.  **`/predict/solar`**: Predicts the next 24 hours of solar generation (MW).
3.  **`/optimize/schedule`**: Takes the two forecasts and generates an optimal 24-hour power dispatch schedule to minimize coal consumption.

These models are trained on a synthetic dataset and will be deployed to Google Vertex AI. The API, running on Google Cloud Run, provides the foundational curves for the main optimization engine.

See the `notebooks/` directory for model training and `app/` for the API implementation.

*Note: You will need to install `ortools` to run the optimization endpoint (`pip install ortools`).*