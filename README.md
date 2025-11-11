# Phishing Classifier

Complete feature engineering pipeline with 20 features, comprehensive testing, and production-ready architecture.

---

## 📈 Status

- **Status:** active (Active)
- **Focus:** Complete feature engineering pipeline with 20 features, comprehensive testing, and production-ready architecture.
- **Last updated:** 11/11/2025
- **Target completion:** 18/11/2025

---

## 🔑 Highlights

- **Dataset** → UCI Phishing Websites Dataset with 11,055 samples
- **Features** → 30 engineered features (URL length, domain age, suspicious patterns)
- **Models** → Multiple baseline algorithms (Logistic Regression, Random Forest, SVM)
- **Evaluation** → Comprehensive metrics (accuracy, precision, recall, F1-score)
- **Pipeline** → End-to-end ML workflow from EDA to model export
- **Export** → Pickle serialization for API integration

---

## 🏗 Architecture Overview

```
src/
 ├─ data/           # load.py, preprocess.py
 ├─ models/         # train.py, evaluate.py
 └─ pipeline.py     # main execution script
```

**Patterns used:**

- **load.py** handles data ingestion and validation
- **preprocess.py** performs feature engineering and scaling
- **train.py** implements model training with cross-validation
- **evaluate.py** provides comprehensive model evaluation
- **pipeline.py** orchestrates the entire ML workflow

---

## 📱 What It Demonstrates

- End-to-end machine learning project structure
- Feature engineering and data preprocessing techniques
- Model training, evaluation, and comparison
- Production-ready model export and serialization

---

## 🚀 Getting Started

```bash
git clone https://github.com/Krispy145/phishing-classifier.git
cd phishing-classifier
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
python src/pipeline.py
```

**Train specific models:**
```bash
python src/models/train.py --model logistic_regression
python src/models/train.py --model random_forest
```

---

## 🧪 Testing

```bash
python -m pytest tests/
```

- Unit tests → Data loading and preprocessing functions
- Integration tests → Full pipeline execution
- Model tests → Training and evaluation workflows

---

## 🔒 Security & Next Steps

- Follow security best practices for the technology stack
- Implement proper authentication and authorization
- Add comprehensive error handling and validation
- Set up monitoring and logging

---

## 🗓 Roadmap

| Milestone                    | Category              | Target Date | Status     |
| ---------------------------- | --------------------- | ----------- | ---------- |
| Scaffold repo | AI Engineering Projects | 26/10/2025 | ✅ Done |
| EDA and feature engineering | AI Engineering Projects | 26/10/2025 | ✅ Done |
| Comprehensive testing infrastructure | AI Engineering Projects | 26/10/2025 | ✅ Done |
| Dataset management system | AI Engineering Projects | 26/10/2025 | ✅ Done |
| Train and export baseline | AI Engineering Projects | 26/10/2025 | ⏳ In Progress |
| Model evaluation suite | AI Engineering Projects | 18/11/2025 | ⏳ In Progress |
| Secure AI API integration | AI Engineering Projects | 22/11/2025 | ⏳ In Progress |


---

## 📄 License

MIT © Krispy145