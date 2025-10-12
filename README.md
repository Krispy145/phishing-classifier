# Phishing Classifier

Supervised model to detect phishing: EDA, features, baselines, metrics.

---

## 📈 Status

- **Status:** active (Active)
- **Focus:** Supervised model to detect phishing: EDA, features, baselines, metrics.
- **Last updated:** 11/10/2025
- **Target completion:** 20/10/2025

---

## 🔑 Highlights

- **Dataset** → UCI Phishing Websites Dataset with 11,055 samples
- **Features** → 20 engineered features covering URL, domain, and content characteristics
- **Models** → Multiple baseline algorithms (Logistic Regression, Random Forest, SVM)
- **Evaluation** → Comprehensive metrics (accuracy, precision, recall, F1-score)
- **Pipeline** → End-to-end ML workflow from EDA to model export
- **Export** → Pickle serialization for API integration

### 📊 Feature Engineering (20 Features)

| Category             | Feature                  | Description                              |
| -------------------- | ------------------------ | ---------------------------------------- |
| **URL Features**     | URL length               | Total character count in the URL         |
|                      | Subdomain count          | Number of subdomains in the URL          |
|                      | Suspicious characters    | Count of special characters (@, #, etc.) |
|                      | URL shortening           | Detection of URL shortening services     |
|                      | IP address in URL        | Presence of IP address instead of domain |
|                      | Redirect chain length    | Number of redirects before final page    |
| **Domain Features**  | Domain age               | Age of the domain in days                |
|                      | Domain registrar         | Registrar reputation score               |
|                      | Country of origin        | Geographic location of domain            |
|                      | Alexa rank               | Website popularity ranking               |
|                      | SSL certificate validity | SSL certificate status and validity      |
|                      | Domain length            | Length of the domain name                |
| **Content Features** | Suspicious keywords      | Count of phishing-related keywords       |
|                      | HTML form count          | Number of forms on the page              |
|                      | External link ratio      | Ratio of external to internal links      |
|                      | Image-to-text ratio      | Ratio of images to text content          |
|                      | JavaScript ratio         | Percentage of JavaScript content         |
|                      | Page load time           | Time taken to load the page              |
|                      | Meta tag count           | Number of meta tags in HTML              |
|                      | Title length             | Length of the page title                 |
|                      | Suspicious TLD           | Use of suspicious top-level domains      |

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

| Milestone                   | Category                | Target Date | Status         |
| --------------------------- | ----------------------- | ----------- | -------------- |
| Scaffold repo               | AI Engineering Projects | 12/10/2025  | ✅ Done        |
| EDA and feature engineering | AI Engineering Projects | 15/10/2025  | ⏳ In Progress |
| Train and export baseline   | AI Engineering Projects | 18/10/2025  | ⏳ In Progress |
| Model evaluation suite      | AI Engineering Projects | 20/10/2025  | ⏳ In Progress |
| Secure AI API integration   | AI Engineering Projects | 24/10/2025  | ⏳ In Progress |

---

## 📄 License

MIT © Krispy145
