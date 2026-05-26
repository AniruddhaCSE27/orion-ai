# Real-Time ML Prediction API System

## Project Description

This project is a production-ready machine learning inference service built with FastAPI, Scikit-learn, and Docker. It serves real-time binary classification predictions through a clean REST API, with startup model loading, structured validation, logging, and containerized deployment.

The example model uses a healthcare-style binary classification problem from Scikit-learn's breast cancer dataset and exposes four numeric input features through a simple churn-style prediction interface.

## Features

- FastAPI-based REST API for low-latency prediction serving
- Binary classification model trained with Scikit-learn
- Startup model initialization for efficient inference
- Request validation with Pydantic
- Health check endpoint for deployment monitoring
- Structured error handling and JSON responses
- Containerized deployment with Docker
- Modular code organization for maintainability

## Tech Stack

- Python 3.10
- FastAPI
- Uvicorn
- Scikit-learn
- Joblib
- Pydantic
- Pandas
- Docker

## Project Structure

```text
real-time-ml-api/
|-- app/
|   |-- main.py
|   |-- schema.py
|   `-- utils.py
|-- models/
|   `-- model.pkl
|-- train_model.py
|-- requirements.txt
|-- Dockerfile
`-- README.md
```

## Local Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This generates the serialized artifact at `models/model.pkl`.

### 4. Start the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## Run with Docker

### 1. Build the image

```bash
docker build -t real-time-ml-api .
```

### 2. Ensure the model is available

Train the model before building if `models/model.pkl` does not already exist:

```bash
python train_model.py
```

### 3. Run the container

```bash
docker run -p 8000:8000 real-time-ml-api
```

## API Endpoints

### `GET /`

Returns a welcome message and basic service information.

Example response:

```json
{
  "message": "Welcome to the Real-Time ML Prediction API System.",
  "model_type": "Binary classification"
}
```

### `GET /health`

Used for service health monitoring.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Accepts a JSON payload containing the four model features and returns the predicted class and probabilities.

Example request:

```json
{
  "feature1": 10,
  "feature2": 5,
  "feature3": 2,
  "feature4": 1
}
```

Example response:

```json
{
  "prediction": 1,
  "prediction_label": "benign",
  "probability": 0.998321,
  "class_probabilities": {
    "malignant": 0.001679,
    "benign": 0.998321
  },
  "model_features": [
    "feature1",
    "feature2",
    "feature3",
    "feature4"
  ]
}
```

## Production Notes

- The model is loaded once at application startup for efficiency.
- Validation errors return clear `422` responses.
- Unexpected runtime failures are logged and converted into clean JSON error messages.
- The current example uses a lightweight classification model, but the structure supports swapping in a richer production model artifact later.
