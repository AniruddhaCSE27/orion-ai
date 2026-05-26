# Real-Time ML Prediction API System

## Overview

This project is a production-oriented machine learning inference service built with FastAPI, Scikit-learn, and Docker. It serves real-time binary classification predictions through an authenticated API that is ready for local development, containerized deployment, and cloud hosting on platforms such as Render, AWS EC2, or AWS ECS.

The system keeps the architecture intentionally lean: environment-based configuration, JWT authentication, structured logging, request tracing, rate limiting, health probes, and a small automated test suite. The result is a realistic backend and ML serving foundation rather than a demo-only prototype.

## Features

- FastAPI application organized with routers, dependencies, and lifespan startup
- Scikit-learn inference pipeline serialized with metadata-rich model artifacts
- JWT authentication using environment-managed credentials
- Rate limiting on login and inference endpoints
- Structured logging with request ID propagation
- Liveness and readiness probes for monitoring and deployment platforms
- CORS, trusted host support, and response security headers
- Docker, Docker Compose, Render, and AWS deployment readiness
- Pytest suite covering health, authentication, and prediction flows

## Architecture

```text
real-time-ml-api/
|-- app/
|   |-- main.py
|   |-- config.py
|   |-- dependencies.py
|   |-- logging_config.py
|   |-- middleware.py
|   |-- auth.py
|   |-- rate_limiter.py
|   |-- schema.py
|   |-- utils.py
|   |-- model_loader.py
|   `-- routers/
|       |-- health.py
|       |-- inference.py
|       `-- auth.py
|-- models/
|   `-- model.pkl
|-- tests/
|   |-- test_health.py
|   |-- test_auth.py
|   `-- test_predict.py
|-- train_model.py
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
|-- .dockerignore
|-- .env.example
|-- render.yaml
`-- README.md
```

## API Summary

- `GET /`
  Returns service metadata.
- `GET /health/live`
  Returns liveness information for platform health checks.
- `GET /health/ready`
  Returns readiness information and confirms model availability.
- `POST /auth/token`
  Issues a JWT access token from environment-based credentials.
- `POST /predict`
  Authenticated real-time inference endpoint with rate limiting.

## Environment Variables

Copy `.env.example` to `.env` and update the values before running the service outside local experimentation.

```env
APP_NAME=Real-Time ML Prediction API
APP_VERSION=2.0.0
ENVIRONMENT=development
DEBUG=false
API_USERNAME=admin
API_PASSWORD=change-me
JWT_SECRET_KEY=replace-with-a-long-random-secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
AUTH_RATE_LIMIT=5/minute
PREDICT_RATE_LIMIT=20/minute
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
TRUSTED_HOSTS=localhost,127.0.0.1
MODEL_PATH=models/model.pkl
```

## Local Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
```

### 5. Train or refresh the model artifact

```powershell
python train_model.py
```

### 6. Run the API locally

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Authentication Flow

The API uses JWT Bearer authentication for inference. Credentials are sourced from environment variables rather than a database to keep the example operationally simple and deployment-friendly.

1. Send username and password to `POST /auth/token`.
2. Receive a short-lived access token.
3. Include the token in the `Authorization: Bearer <token>` header when calling `POST /predict`.

## Sample curl Requests

### Login

```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=change-me"
```

### Liveness check

```bash
curl "http://localhost:8000/health/live"
```

### Readiness check

```bash
curl "http://localhost:8000/health/ready"
```

### Predict

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d "{\"feature1\": 10, \"feature2\": 5, \"feature3\": 2, \"feature4\": 1}"
```

## Docker Usage

### Build the image

```powershell
docker build -t real-time-ml-api .
```

### Run the container

```powershell
docker run --env-file .env -p 8000:8000 real-time-ml-api
```

## Docker Compose Usage

```powershell
docker compose up --build
```

## Render Deployment

The repository includes `render.yaml` for deployment on Render. Configure the service environment variables in the Render dashboard, especially:

- `API_USERNAME`
- `API_PASSWORD`
- `JWT_SECRET_KEY`

Recommended deployment notes:

- Keep `ENVIRONMENT=production`
- Use a strong secret for `JWT_SECRET_KEY`
- Restrict `ALLOWED_ORIGINS` to trusted client origins
- Point the health check at `/health/live`

## AWS Deployment Notes

The service is ready for Docker-based deployment on AWS EC2 or ECS.

Practical deployment pattern:

1. Build and push the Docker image to Amazon ECR.
2. Deploy the container to ECS or run it on EC2 with Docker Compose or systemd.
3. Put Nginx or an AWS load balancer in front of the service for TLS termination and traffic management.
4. Expose only the necessary security group ports, typically `80/443` externally and `8000` internally where appropriate.
5. Store production secrets in AWS Systems Manager Parameter Store or AWS Secrets Manager rather than plain environment files.

## Testing

Run the automated tests with:

```powershell
pytest
```

The test suite covers:

- Liveness and readiness endpoints
- Valid and invalid authentication
- Authenticated and unauthenticated prediction requests
- Validation failure handling

## Security Considerations

- Replace the default credentials and JWT secret before any shared deployment.
- Restrict allowed CORS origins and trusted hosts in production.
- Keep docs disabled in production by setting `ENVIRONMENT=production`.
- Avoid logging secrets, passwords, or raw token values.
- Place the service behind HTTPS in production using a reverse proxy or cloud load balancer.

## Model Training Notes

`train_model.py` trains a Logistic Regression pipeline on a Scikit-learn dataset and stores a serialized artifact at `models/model.pkl`.

The artifact includes:

- `model_version`
- `feature_names`
- `source_feature_names`
- `target_names`
- `metrics`

This metadata is surfaced by the API so inference responses remain traceable and deployment-friendly.
