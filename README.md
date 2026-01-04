This README is:

* ✅ Aligned with your **actual implementation**
* ✅ Matches **MLOps assignment rubrics**
* ✅ Easy for evaluators to understand
* ✅ Industry-style (not academic fluff)

---

# ❤️ Heart Disease Prediction – End-to-End MLOps Pipeline

This repository implements an **end-to-end MLOps pipeline** for a Heart Disease Prediction system, covering the complete lifecycle from data processing and model training to deployment, CI/CD, Kubernetes orchestration, and monitoring using Prometheus and Grafana.

---

## 📌 Project Overview

The goal of this project is to demonstrate practical **MLOps principles** by building a production-ready machine learning system that includes:

* Data acquisition and preprocessing
* Model training and evaluation
* Automated testing and CI/CD
* Containerized model serving API
* Kubernetes deployment
* Monitoring and observability

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **ML Framework:** Scikit-learn
* **API Framework:** FastAPI
* **Containerization:** Docker
* **Orchestration:** Kubernetes (Docker Desktop)
* **CI/CD:** GitHub Actions
* **Monitoring:** Prometheus & Grafana

---

## 📂 Repository Structure

```
mlops-heart-disease/
│
├── app/                    # FastAPI application
├── data/                   # Raw and cleaned datasets
├── models/                 # Trained models & features
├── src/                    # Training and utility scripts
├── tests/                  # Unit tests (Pytest)
├── k8s/                    # Kubernetes manifests
├── monitoring/             # Prometheus & Grafana configs
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile              # Docker build file
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📊 Dataset

* **Source:** UCI Machine Learning Repository
* **Dataset:** Heart Disease Dataset
* **Target Variable:** Presence of heart disease (binary classification)

---

## 🧠 Model Development

* **Models Trained:**

  * Logistic Regression
  * Random Forest Classifier
* **Feature Engineering:**

  * Numerical feature scaling
  * Consistent preprocessing pipeline
* **Evaluation Metrics:**

  * Accuracy
  * Precision
  * Recall
  * ROC-AUC
* **Final Model:** Saved using `joblib` for reproducibility

---

## 🚀 Model Serving API

* Built using **FastAPI**
* Exposes `/predict` endpoint
* Accepts JSON input and returns:

  * Prediction (0 / 1)
  * Probability score

### Example Request

```json
{
  "age": 62,
  "sex": 1,
  "cp": 3,
  "trestbps": 150,
  "chol": 290,
  "fbs": 1,
  "restecg": 2,
  "thalach": 120,
  "exang": 1,
  "oldpeak": 2.5,
  "slope": 1,
  "ca": 2,
  "thal": 3
}
```

---

## 🐳 Dockerization

Build Docker image:

```bash
docker build -t heart-disease-api .
```

Run container:

```bash
docker run -p 8000:8000 heart-disease-api
```

Access Swagger UI:

```
http://localhost:8000/docs
```

---

## ☸️ Kubernetes Deployment

* Deployed on **Docker Desktop Kubernetes**
* Uses:

  * Deployment YAML
  * Service (NodePort)
* API exposed via:

```
http://localhost:30007/docs
```

Check deployment:

```bash
kubectl get pods
kubectl get services
```

---

## 🔁 CI/CD Pipeline

Implemented using **GitHub Actions**:

* Code linting (flake8)
* Unit testing (pytest)
* Model training execution

Pipeline runs automatically on every push to `main`.

---

## 📈 Monitoring & Logging

* **Prometheus** scrapes API metrics
* **Grafana** visualizes:

  * API request count
  * Request trends over time

Grafana Dashboard:

```
http://localhost:3000
```

Metrics endpoint:

```
/metrics
```

---

## 🧪 Testing

Run tests locally:

```bash
pytest tests/
```

Tests include:

* Model artifact validation
* Feature file checks
* Prediction sanity checks

---

## 📄 Documentation

A detailed **10–12 page report** is included separately covering:

* Task-wise implementation
* Architecture
* Screenshots
* Results and observations

---

## 📌 Conclusion

This project demonstrates a **complete MLOps workflow** following industry best practices, integrating automation, scalability, monitoring, and reproducibility into a real-world ML system.

---

## 🔗 GitHub Repository

```
https://github.com/Soniyap16/mlops-heart-disease
```
