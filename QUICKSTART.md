# 🚀 Quick Start Guide

Get the churn prediction platform running in 5 minutes!

## Prerequisites
- Python 3.8+ installed
- Git installed

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/churn-clv-realtime.git
cd churn-clv-realtime
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data
```bash
python download_data.py
```

### 5. Run Pipeline
```bash
# Preprocess data
python src/pipelines/data_preprocessing.py

# Train models
python src/models/train_churn_model.py
```

### 6. Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

Open browser: `http://localhost:8501`

---

## 🐳 Docker Deployment (Alternative)
```bash
# Build image
docker build -t churn-prediction .

# Run container
docker run -p 8501:8501 churn-prediction
```

---

## ✅ Verify Installation

You should see:
- ✅ Data downloaded to `data/raw/`
- ✅ Models saved to `models/`
- ✅ MLflow experiments in `mlruns/`
- ✅ Dashboard running on port 8501

---

## 🆘 Troubleshooting

**Issue:** SSL certificate error during data download
**Fix:** Use `pip install requests` and the script will auto-handle SSL

**Issue:** Module not found errors
**Fix:** Ensure virtual environment is activated before installing packages

**Issue:** Port 8501 already in use
**Fix:** `streamlit run app/streamlit_app.py --server.port=8502`

---

## 📞 Support

Questions? Open an issue or contact: bahua.v@northeastern.edu
