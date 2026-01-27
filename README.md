# 🎯 Customer Churn Prediction Platform

**End-to-end ML system for predicting customer churn with real-time business insights**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-blue.svg)](https://mlflow.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10-yellow.svg)](https://duckdb.org/)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Business Impact](#business-impact)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

This project implements a production-grade machine learning system to predict customer churn for telecommunications companies. The platform identifies high-risk customers and calculates revenue at risk, enabling data-driven retention strategies.

**Key Highlights:**
- 84% ROC-AUC score on test data
- Real-time churn probability predictions
- Interactive business dashboard
- MLflow experiment tracking
- SQL-based feature engineering with DuckDB
- Docker-ready deployment

---

## ✨ Key Features

### 🔮 Predictive Analytics
- Binary classification for churn prediction
- Probability-based risk scoring (High/Medium/Low)
- Customer lifetime value (CLV) estimation

### 📊 Business Intelligence
- Revenue-at-risk calculations
- Actionable retention recommendations
- Customer segmentation insights

### 🛠️ Engineering Excellence
- Modular, production-ready code
- Config-driven architecture
- Comprehensive logging system
- Experiment tracking with MLflow
- Cloud-ready design (Databricks/Azure compatible)

---

## 🔧 Tech Stack

### Data & Analytics
- **Python 3.13** - Core programming language
- **Pandas** - Data manipulation
- **DuckDB** - In-memory SQL analytics (cloud-ready)

### Machine Learning
- **Scikit-learn** - Model training and evaluation
- **MLflow** - Experiment tracking and model registry
- **Joblib** - Model persistence

### Visualization & Deployment
- **Streamlit** - Interactive web dashboard
- **Docker** - Containerization
- **PyYAML** - Configuration management

---

## 📁 Project Structure
```
churn-clv-realtime/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned data
│   └── features/               # DuckDB feature store
│
├── src/
│   ├── pipelines/
│   │   └── data_preprocessing.py    # SQL-based ETL pipeline
│   ├── models/
│   │   └── train_churn_model.py     # Model training with MLflow
│   └── utils/                       # Helper functions
│
├── app/
│   └── streamlit_app.py        # Interactive dashboard
│
├── config/
│   └── config.yaml             # Configuration file
│
├── models/
│   ├── churn_model.pkl         # Trained model
│   └── scaler.pkl              # Feature scaler
│
├── logs/                        # Execution logs
├── mlruns/                      # MLflow tracking data
├── notebooks/                   # EDA notebooks
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/churn-clv-realtime.git
cd churn-clv-realtime

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_data.py
```

---

## 💻 Usage

### 1. Data Preprocessing
```bash
python src/pipelines/data_preprocessing.py
```

**What it does:**
- Loads raw CSV data
- Cleans missing values
- Encodes categorical features using SQL
- Saves processed data to DuckDB feature store

**Output:** `data/processed/cleaned_churn_data.csv`

---

### 2. Model Training
```bash
python src/models/train_churn_model.py
```

**What it does:**
- Trains Logistic Regression and Random Forest models
- Logs experiments to MLflow
- Selects best model by ROC-AUC
- Saves model artifacts

**View experiments:**
```bash
mlflow ui
```
Navigate to `http://localhost:5000`

---

### 3. Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501`

**Features:**
- Input customer details via interactive form
- Get real-time churn probability
- View business insights and recommendations
- Risk level classification (High/Medium/Low)

---

## 📊 Model Performance

### Best Model: Logistic Regression

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **84.0%** |
| Accuracy | 79.8% |
| Precision | 64.1% |
| Recall | 54.8% |
| F1-Score | 59.1% |

### Confusion Matrix
- True Negatives: 920
- False Positives: 115
- False Negatives: 169
- True Positives: 205

### Model Selection Rationale
Logistic Regression was selected over Random Forest despite similar performance due to:
- Better interpretability for business stakeholders
- Faster inference time
- Lower computational cost
- Easier to explain feature importance

---

## 💼 Business Impact

### Use Cases
1. **Proactive Retention** - Identify at-risk customers before they churn
2. **Resource Optimization** - Prioritize high-value customer interventions
3. **Campaign Targeting** - Design personalized retention offers
4. **Revenue Protection** - Quantify and mitigate revenue at risk

### Expected ROI
- **15-20%** reduction in churn rate
- **$500K-$1M** annual revenue saved (mid-size telco)
- **3-5x** ROI on targeted retention campaigns
- **40%** improvement in campaign efficiency

---

## 🔮 Future Enhancements

### Technical
- [ ] Add XGBoost/LightGBM models for comparison
- [ ] Implement SHAP for explainability
- [ ] Add feature importance visualization
- [ ] Deploy REST API with FastAPI
- [ ] Implement A/B testing framework

### Cloud Migration
- [ ] Deploy on Azure ML or AWS SageMaker
- [ ] Migrate DuckDB queries to Databricks
- [ ] Set up CI/CD pipeline
- [ ] Add monitoring and alerting

### Business Features
- [ ] Customer segmentation clustering
- [ ] Cohort analysis dashboard
- [ ] Churn propensity trends over time
- [ ] Recommendation engine for retention strategies

---

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Production-ready code architecture
- Experiment tracking and model versioning
- SQL-based feature engineering
- Business metric translation (churn → revenue impact)
- Interactive dashboard development
- Cloud-ready design patterns

---

## 📝 License

MIT License - see LICENSE file for details

---

## 👤 Author


**Vinit Bahua**
- Graduate Student, Data Science @ Northeastern University
- Seeking Fall 2026 Co-op Opportunities
- LinkedIn: [linkedin.com/in/vinitbahua](https://www.linkedin.com/in/vinit-bahua-586466246/)
- Email: bahua.v@northeastern.edu
- GitHub: [@vinitbahua123](https://github.com/vinitbahua123)

---

## 🙏 Acknowledgments

- Dataset: IBM Telco Customer Churn
- Framework: Scikit-learn, MLflow, Streamlit, DuckDB
- Inspiration: Real-world churn prediction systems at Netflix, Spotify

---

**Built with ❤️ for production-grade data science**
