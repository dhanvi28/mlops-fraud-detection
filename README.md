# MLOps Fraud Detection System

An end-to-end MLOps-based fraud detection project built using Python, FastAPI, MLflow, XGBoost, and Scikit-learn.

This project demonstrates a complete machine learning workflow including:
- data ingestion
- preprocessing and scaling
- model training
- experiment tracking
- API deployment
- inference validation
- modular MLOps architecture

---

# Project Structure

```bash
mlops-fraud-detection/
│
├── app/
│   └── main.py
│
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── metrics.txt
│
├── data/
│   └── creditcard.csv
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   └── train_pipeline.py
│   │
│   └── __init__.py
│
├── tests/
│
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

---

# Features

- Fraud detection using machine learning
- End-to-end MLOps pipeline
- Modular project architecture
- Data ingestion pipeline
- Train/test split
- Feature scaling using StandardScaler
- XGBoost model training
- MLflow experiment tracking
- FastAPI deployment
- Swagger API documentation
- Prediction probability scoring
- Input validation
- Saved model and scaler artifacts
- Docker support

---

# Technologies Used

## Machine Learning & Data

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Backend & API

- FastAPI
- Uvicorn
- Pydantic

## MLOps & Deployment

- MLflow
- Docker
- Joblib

## Development Tools

- Git
- GitHub
- VS Code
- Virtual Environments

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/mlops-fraud-detection.git
```

Move into the project directory:

```bash
cd mlops-fraud-detection
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

## Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Training Pipeline

Run the complete ML pipeline:

```bash
python -m src.pipeline.train_pipeline
```

Pipeline flow:

```text
Data Ingestion
    ↓
Data Transformation
    ↓
Train/Test Split
    ↓
Feature Scaling
    ↓
Model Training
    ↓
Evaluation
    ↓
Artifact Generation
```

---

# Generated Artifacts

The training pipeline automatically generates:

```bash
artifacts/model.pkl
artifacts/scaler.pkl
artifacts/features.pkl
artifacts/metrics.txt
```

---

# Run FastAPI Server

Start the API server:

```bash
uvicorn app.main:app --reload
```

Server runs at:

```bash
http://127.0.0.1:8000
```

Swagger API documentation:

```bash
http://127.0.0.1:8000/docs
```

---

# Run MLflow UI

Start MLflow tracking dashboard:

```bash
mlflow ui
```

MLflow UI runs at:

```bash
http://127.0.0.1:5000
```

---

# API Endpoints

## Root Endpoint

```http
GET /
```

Response:

```json
{
  "message": "Fraud Detection API"
}
```

---

## Model Information

```http
GET /model-info
```

Response:

```json
{
  "model": "XGBoost Fraud Detection",
  "features": 30,
  "scaler": "StandardScaler"
}
```

---

## Prediction Endpoint

```http
POST /predict
```

Example Request:

```json
{
  "features": [
    0.1,
    0.2,
    0.3
  ]
}
```

Example Response:

```json
{
  "prediction": "Legitimate Transaction",
  "fraud_probability": 0.0021
}
```

---

# Model Performance

- Fraud detection classification model
- Accuracy achieved: ~99.95%

---

# Docker Support

Build Docker image:

```bash
docker build -t fraud-detection .
```

Run Docker container:

```bash
docker run -p 8000:8000 fraud-detection
```

---

# Future Improvements

- Cloud deployment
- CI/CD integration
- Kubernetes deployment
- Real-time fraud detection
- Monitoring and logging
- Database integration
- Authentication & security
- Streamlit frontend dashboard

---

# Author

Developed by Dhanvi
