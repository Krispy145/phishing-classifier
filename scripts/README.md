# 🔧 Scripts Directory

This directory contains utility scripts for the Phishing Classifier project.

## 📁 Available Scripts

### **Dataset Management**

#### `create_sample_dataset.py`
Creates a sample dataset for testing and development.

**Usage:**
```bash
python3 scripts/create_sample_dataset.py
```

**What it does:**
- Creates 1,000 sample URLs (500 legitimate + 500 phishing)
- Saves to `data/raw/phishing.csv` (main dataset)
- Creates backup copy in `data/sample/phishing_sample.csv`
- Generates balanced dataset for immediate testing

**Output:**
- `data/raw/phishing.csv` - Main dataset for the system
- `data/sample/phishing_sample.csv` - Backup copy

#### `download_uci_dataset.py`
Downloads and converts the official UCI Phishing Websites Dataset.

**Usage:**
```bash
python3 scripts/download_uci_dataset.py
```

**What it does:**
- Downloads the official UCI dataset (11,055 samples)
- Converts from ARFF to CSV format
- Saves to `data/raw/phishing.csv`
- Falls back to comprehensive sample if download fails

**Features:**
- Multiple download sources
- Automatic format conversion
- Error handling and fallbacks
- Progress reporting

#### `sync_model_to_api.py`
Syncs the trained model to the Secure AI API for integration.

**Usage:**
```bash
python3 scripts/sync_model_to_api.py
```

**What it does:**
- Copies `app/models/model.joblib` to `secure-ai-api/app/models/model.joblib`
- Verifies the copy was successful
- Provides status feedback

**Prerequisites:**
- Model must be trained first (`python src/pipeline.py`)
- Both repositories must be in the same parent directory

**Output:**
- Model copied to `secure-ai-api/app/models/model.joblib`
- Status messages indicating success or failure

## 🚀 Quick Start

### **1. Create Sample Dataset (Immediate Testing)**
```bash
python3 scripts/create_sample_dataset.py
```

### **2. Download Full UCI Dataset (Production)**
```bash
python3 scripts/download_uci_dataset.py
```

### **3. Verify Dataset**
```bash
python3 -c "import pandas as pd; df = pd.read_csv('data/raw/phishing.csv'); print(f'Dataset: {df.shape[0]} samples')"
```

## 📊 Dataset Information

### **Sample Dataset**
- **Size**: 1,000 samples
- **Balance**: 500 legitimate + 500 phishing
- **Format**: CSV with `url` and `Result` columns
- **Purpose**: Testing and development

### **UCI Dataset**
- **Size**: 11,055 samples
- **Source**: UCI Machine Learning Repository
- **Format**: Converted from ARFF to CSV
- **Purpose**: Production and research

## 🔧 Script Requirements

### **Dependencies**
```bash
pip install pandas numpy scipy requests
```

### **Python Version**
- Python 3.7 or higher
- Compatible with the main project requirements

## 📝 Notes

- Scripts are designed to be run from the project root directory
- All scripts create necessary directories automatically
- Error handling includes fallback options
- Scripts are safe to run multiple times (idempotent)

## 🚨 Important

- Always backup your data before running scripts
- The `data/raw/phishing.csv` file is used by the main system
- Scripts will overwrite existing datasets
- Check the output messages for success/failure status
