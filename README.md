# Phishing Website Classifier

Supervised model to detect phishing: EDA, features, baselines, metrics.

---

## 📈 Status

- **Status:** scaffolded (initial setup complete)
- **Focus:** Machine learning fundamentals, feature engineering, and model evaluation
- **Last updated:** 07/10/2025
- **Upcoming integration:** Secure AI API (as inference backend)

---

## 🔑 Highlights

- **Dataset:** UCI Phishing Websites Dataset with 11,055 samples
- **Features:** 30 engineered features (URL length, domain age, suspicious patterns)
- **Models:** Multiple baseline algorithms (Logistic Regression, Random Forest, SVM)
- **Evaluation:** Comprehensive metrics (accuracy, precision, recall, F1-score)
- **Pipeline:** End-to-end ML workflow from EDA to model export
- **Export:** Pickle serialization for API integration
- **Testing:** Cross-validation and holdout testing strategies

---

## 🏗 Architecture Overview

Clean ML pipeline with modular components:

```
src/
 ├─ data/           # load.py, preprocess.py
 ├─ models/         # train.py, evaluate.py
 └─ pipeline.py     # main execution script
```

**Patterns used:**

- `load.py` handles data ingestion and validation
- `preprocess.py` performs feature engineering and scaling
- `train.py` implements model training with cross-validation
- `evaluate.py` provides comprehensive model evaluation
- `pipeline.py` orchestrates the entire ML workflow

---

## 📱 What It Demonstrates

- End-to-end machine learning project structure
- Feature engineering and data preprocessing techniques
- Model training, evaluation, and comparison
- Production-ready model export and serialization
- Clean, maintainable ML code architecture

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

**Evaluate models:**

```bash
python src/models/evaluate.py --model-path models/logistic_regression.pkl
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

- Follow secure ML practices (data validation, model versioning)
- Integrate with **Secure AI API** for real-time inference
- Add **model monitoring** and performance tracking
- Implement **A/B testing** for model comparison

---

## 🗓 Roadmap

| Milestone                    | Target Date | Status     |
| ---------------------------- | ----------- | ---------- |
| Scaffold repo                | 12/10/2025  | ✅ Done    |
| EDA and feature engineering  | 15/10/2025  | ⏳ Pending |
| Train and export baseline    | 18/10/2025  | ⏳ Planned |
| Model evaluation suite       | 20/10/2025  | ⏳ Planned |
| Secure AI API integration    | 24/10/2025  | ⏳ Planned |

---

## 📄 License

MIT © Krispy145