# SolarCOptimizer - Backend

This repository contains the machine learning backend for the **SolarCOptimizer** project, part of the Google Solution Challenge 2026.

## Overview

The core component is a self-contained FastAPI application that provides three key endpoints:
1.  **`/predict/demand`**: Predicts the next 24 hours of electricity demand (MW).
2.  **`/predict/solar`**: Predicts the next 24 hours of solar generation (MW).
3.  **`/optimize/schedule`**: Takes the two forecasts and generates an optimal 24-hour power dispatch schedule to minimize coal consumption.

The application loads pre-trained models and runs all predictions and optimizations locally. It is designed to be containerized using the provided `Dockerfile` for easy deployment.

See the `notebooks/` directory for model training and `app/` for the API implementation.

*Note: You will need to install `ortools` to run the optimization endpoint (`pip install ortools`).*