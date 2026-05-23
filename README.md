# MLOps Fraud Detection System

An end-to-end MLOps-based fraud detection project built using Python, FastAPI, Scikit-learn, XGBoost, and MLflow.

This project demonstrates:
- Data ingestion
- Data transformation
- Model training
- Experiment tracking with MLflow
- FastAPI deployment
- Modular MLOps project structure

---

# Project Structure

```bash
mlops-fraud-detection/
│
├── app/
│   └── main.py
│
├── artifacts/
│   └── model.pkl
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
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

---

# Features

- Fraud detection using machine learning
- Modular MLOps architecture
- FastAPI backend
- MLflow experiment tracking
- Model artifact generation
- Scalable project structure

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- MLflow
- Joblib

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

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Training Pipeline

```bash
python -m src.pipeline.train_pipeline
```

---

# Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

Server will run at:

```bash
http://127.0.0.1:8000
```

Swagger API Docs:

```bash
http://127.0.0.1:8000/docs
```

---

# Run MLflow UI

```bash
mlflow ui
```

Open:

```bash
http://127.0.0.1:5000
```

# Model Performance

- Fraud detection classification model
- Accuracy achieved: ~99.95%

---

# Future Improvements

- Docker deployment
- CI/CD pipeline
- Cloud deployment
- Database integration
- Real-time prediction endpoint

---

# Author

Developed by Dhanvi
