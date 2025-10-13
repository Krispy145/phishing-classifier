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

- **Dataset** → Sample dataset with 1,000 samples (UCI dataset: 11,055 samples)
- **Features** → 20 engineered features covering URL, domain, and content characteristics
- **Models** → Multiple baseline algorithms (Logistic Regression, Random Forest, SVM)
- **Evaluation** → Comprehensive metrics (accuracy, precision, recall, F1-score)
- **Pipeline** → End-to-end ML workflow from EDA to model export
- **Export** → Pickle serialization for API integration

---

## 🏗 Architecture Overview

```
src/
 ├─ features/                    # Feature Engineering Pipeline
 │  ├─ __init__.py              # Module initialization
 │  ├─ base.py                  # Base classes and interfaces
 │  ├─ url_features.py          # URL-based features (6 features)
 │  ├─ domain_features.py       # Domain-based features (6 features)
 │  ├─ content_features.py      # Content-based features (8 features)
 │  └─ pipeline.py              # Main feature pipeline orchestrator
 ├─ data/                       # Data Processing
 │  ├─ load.py                  # Data loading and validation
 │  └─ preprocess.py            # Data preprocessing and splitting
 ├─ models/                     # Model Training & Evaluation
 │  ├─ train.py                 # Model training with cross-validation
 │  └─ evaluate.py              # Comprehensive model evaluation
 └─ pipeline.py                 # Main ML pipeline orchestrator

scripts/                        # Utility Scripts
 ├─ create_sample_dataset.py    # Generate sample dataset
 └─ download_uci_dataset.py     # Download UCI dataset

tests/                          # Comprehensive Test Suite
 ├─ test_features.py            # Feature engineering tests (25 tests)
 └─ test_smoke.py               # Smoke tests

examples/                       # Demo and Examples
 └─ feature_engineering_demo.py # Interactive feature demo

test_results/                   # Test Reports & Coverage
 ├─ html/                       # Interactive HTML reports
 ├─ json/                       # Machine-readable JSON reports
 ├─ xml/                        # JUnit XML for CI/CD
 └─ coverage/                   # Code coverage analysis
```

**Architecture Patterns:**

- **Modular Feature Engineering**: Separate extractors for URL, Domain, and Content features
- **Abstract Base Classes**: Consistent interfaces across all feature extractors
- **Pipeline Orchestration**: End-to-end ML workflow with feature engineering
- **Comprehensive Testing**: 25 tests with 95%+ code coverage
- **Production Ready**: Clean, maintainable, and well-documented code

---

## 📱 What It Demonstrates

- **Advanced Feature Engineering**: 20 carefully engineered features across 3 categories
- **Modular ML Architecture**: Clean separation of concerns with abstract base classes
- **Comprehensive Testing**: 25 tests with 95%+ code coverage and structured reporting
- **Production-Ready Pipeline**: End-to-end ML workflow with error handling
- **Industry Best Practices**: Clean code, documentation, and maintainable structure
- **Interactive Demos**: Feature engineering demonstration and analysis tools

---

## 🚀 Getting Started

```bash
git clone https://github.com/Krispy145/phishing-classifier.git
cd phishing-classifier
pip install -r requirements.txt
```

**Run the complete ML pipeline:**

```bash
python src/pipeline.py --stage all
```

**Run feature engineering only:**

```bash
python src/pipeline.py --stage features
```

**Interactive feature demo:**

```bash
python examples/feature_engineering_demo.py
```

**Run comprehensive tests:**

```bash
python run_tests.py --verbose
```

**Setup datasets:**

```bash
# Create sample dataset (already done)
python3 scripts/create_sample_dataset.py

# Download full UCI dataset
python3 scripts/download_uci_dataset.py
```

---

## 🧪 Testing

**Comprehensive test suite with structured reporting:**

```bash
# Run all tests with detailed reporting
python run_tests.py --verbose

# Run specific test categories
python run_tests.py --format unit
python run_tests.py --category URL
python run_tests.py --category Domain
python run_tests.py --category Content
```

**Test Coverage:**

- **25 tests** with **100% pass rate**
- **95%+ code coverage** across all modules
- **Structured reporting**: HTML, JSON, XML, and Coverage reports
- **Interactive reports**: Open `test_results/html/test_report.html` in browser
- **CI/CD ready**: JUnit XML output for automated pipelines

**Test Categories:**

- **Unit tests** → Individual feature extractors and components
- **Integration tests** → Complete feature engineering pipeline
- **Edge case tests** → Error handling and boundary conditions

---

## 🔧 Feature Engineering Pipeline

**Modular Architecture with 20 Engineered Features:**

### URL Features (6 features)

- **URL Length**: Character count analysis
- **Subdomain Count**: Domain structure analysis
- **Suspicious Characters**: Special character detection
- **URL Shortening**: Shortener service detection
- **IP Address Detection**: Direct IP usage detection
- **Redirect Chain Length**: URL redirection analysis

### Domain Features (6 features)

- **Domain Age**: Registration age estimation
- **Registrar Reputation**: Registrar trust scoring
- **Country Risk**: Geographic risk assessment
- **Alexa Rank**: Website popularity scoring
- **SSL Validity**: Certificate validation
- **Domain Length**: Domain name analysis

### Content Features (8 features)

- **Suspicious Keywords**: Phishing language detection
- **HTML Form Count**: Form complexity analysis
- **External Link Ratio**: Link structure analysis
- **Image-to-Text Ratio**: Content composition
- **JavaScript Ratio**: Dynamic content analysis
- **Page Load Time**: Performance estimation
- **Meta Tag Count**: SEO completeness
- **Title Length**: Page title analysis

**Key Features:**

- **Modular Design**: Separate extractors for each category
- **Abstract Base Classes**: Consistent interfaces
- **Comprehensive Testing**: 95%+ code coverage
- **Production Ready**: Error handling and validation
- **Extensible**: Easy to add new features

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
| EDA and feature engineering | AI Engineering Projects | 15/10/2025  | ✅ Done        |
| Train and export baseline   | AI Engineering Projects | 18/10/2025  | ⏳ In Progress |
| Model evaluation suite      | AI Engineering Projects | 20/10/2025  | ⏳ In Progress |
| Secure AI API integration   | AI Engineering Projects | 24/10/2025  | ⏳ In Progress |

### Recent Achievements

- ✅ **Complete Feature Engineering Pipeline**: 20 engineered features across 3 categories
- ✅ **Comprehensive Testing Infrastructure**: 25 tests with 95%+ code coverage
- ✅ **Production-Ready Architecture**: Modular design with abstract base classes
- ✅ **Interactive Demo**: Feature engineering demonstration and analysis tools
- ✅ **Structured Test Reporting**: HTML, JSON, XML, and Coverage reports

---

## 📄 License

MIT © Krispy145
