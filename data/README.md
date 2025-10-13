# 📊 Data Directory

This directory contains the datasets used by the Phishing Classifier project.

## 📁 Directory Structure

```
data/
├── raw/                    # Raw datasets (used by the system)
│   └── phishing.csv       # Main dataset (1000 samples)
├── sample/                 # Sample datasets (for testing)
│   └── phishing_sample.csv # Sample dataset copy
├── uci/                    # UCI dataset downloads
│   └── (downloaded files)  # Original UCI dataset files
└── README.md              # This file
```

## 📈 Current Dataset

### **Sample Dataset (1000 samples)**

- **Location**: `data/raw/phishing.csv`
- **Type**: Balanced sample dataset
- **Legitimate**: 500 samples
- **Phishing**: 500 samples
- **Format**: CSV with columns `url` and `Result`

### **Sample Copy**

- **Location**: `data/sample/phishing_sample.csv`
- **Purpose**: Backup copy for testing
- **Content**: Identical to main dataset

## 🚀 Getting the Full UCI Dataset

### **Option 1: Download Script**

```bash
python3 scripts/download_uci_dataset.py
```

### **Option 2: Manual Download**

1. Visit: https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
2. Download: `PhishingWebsites_Dataset.arff`
3. Convert to CSV using the provided script

### **Option 3: Alternative Sources**

- **Kaggle**: Search for "Phishing Websites Dataset"
- **GitHub**: Look for converted CSV versions

## 📊 Dataset Format

The dataset should have the following structure:

```csv
url,Result
https://www.google.com,0
https://suspicious-site.tk,1
https://www.bank.com,0
https://phishing-attempt.com,1
```

Where:

- `url`: The website URL to analyze
- `Result`: 0 = Legitimate, 1 = Phishing

## 🔧 Usage

The system automatically loads the dataset from `data/raw/phishing.csv`:

```python
from src.data.load import load_raw

# Load the dataset
df = load_raw()
print(f"Loaded {len(df)} samples")
```

## 📝 Notes

- **Sample dataset**: Good for testing and development
- **UCI dataset**: Full dataset with 11,055 samples for production
- **Format**: Must be CSV with `url` and `Result` columns
- **Encoding**: UTF-8 encoding recommended
- **Size**: Sample dataset is ~33KB, full UCI dataset is ~2MB

## 🚨 Important

- The `data/raw/` directory is used by the system
- The `data/sample/` directory is for backup/testing
- The `data/uci/` directory is for downloaded files
- Always keep a backup of your datasets!
