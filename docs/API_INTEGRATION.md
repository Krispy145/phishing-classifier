# Secure AI API Integration

This document describes how to integrate the phishing classifier model with the Secure AI API.

## Overview

The phishing classifier can be integrated with the Secure AI API to provide real-time phishing detection via REST API endpoints. The integration involves:

1. Training and exporting the model
2. Syncing the model to the Secure AI API
3. Using the API endpoints for predictions

## Prerequisites

- Both `phishing-classifier` and `secure-ai-api` repositories should be in the same parent directory
- The model must be trained and saved in `app/models/model.joblib`

## Integration Steps

### 1. Train the Model

Train the phishing classifier model:

```bash
cd phishing-classifier
python src/pipeline.py
```

This will:
- Load and preprocess the data
- Engineer features
- Train the model
- Save the model to `app/models/model.joblib`
- Generate evaluation reports

### 2. Sync Model to API

Use the sync script to copy the model to the Secure AI API:

```bash
python scripts/sync_model_to_api.py
```

This script:
- Copies `app/models/model.joblib` to `secure-ai-api/app/models/model.joblib`
- Verifies the copy was successful
- Provides status feedback

### 3. Start the API Server

Start the Secure AI API:

```bash
cd ../secure-ai-api
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will automatically load the model on startup.

### 4. Use the API

The phishing detection endpoint is available at:

**POST** `/phishing` or `/v1/predict`

**Request:**
```json
{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "input_url": "https://example.com",
  "prediction": "legitimate",
  "confidence": 0.95,
  "score": 0.95
}
```

## API Endpoints

### Phishing Detection

- **Endpoint:** `POST /phishing`
- **Description:** Classify a URL as phishing or legitimate
- **Request Body:**
  ```json
  {
    "url": "string"
  }
  ```
- **Response:**
  ```json
  {
    "input_url": "string",
    "prediction": "phishing" | "legitimate",
    "confidence": 0.0-1.0,
    "score": 0.0-1.0
  }
  ```

### Get Samples

- **Endpoint:** `GET /phishing/samples`
- **Description:** Get sample URLs for testing
- **Query Parameters:**
  - `limit` (optional): Number of samples (1-100, default: 10)
  - `label` (optional): Filter by label ("phishing" or "legitimate")
- **Response:**
  ```json
  {
    "samples": [
      {
        "url": "string",
        "label": "phishing" | "legitimate",
        "score": 0.0-1.0
      }
    ]
  }
  ```

## Model Updates

When you retrain the model:

1. Run the training pipeline: `python src/pipeline.py`
2. Sync the new model: `python scripts/sync_model_to_api.py`
3. Restart the API server to load the new model

## Troubleshooting

### Model Not Found

If the API reports "Model file not found":
- Ensure the model has been synced: `python scripts/sync_model_to_api.py`
- Check that `secure-ai-api/app/models/model.joblib` exists
- Verify file permissions

### Feature Pipeline Import Error

If you see "Could not import feature pipeline":
- Ensure both repositories are in the same parent directory
- The API will fall back to simplified feature extraction
- For full functionality, ensure the directory structure matches

### Low Confidence Scores

If predictions have low confidence:
- Retrain the model with more data
- Check feature engineering pipeline
- Review evaluation metrics in `app/evaluations/`

## Architecture

```
phishing-classifier/
  ├── app/models/model.joblib          # Trained model
  └── src/features/pipeline.py         # Feature engineering

secure-ai-api/
  ├── app/models/model.joblib          # Synced model (from phishing-classifier)
  ├── app/services/
  │   └── phishing_classifier.py      # Service that loads model
  └── app/api/v1/phishing.py          # API endpoint
```

The Secure AI API service:
1. Loads the model from `app/models/model.joblib`
2. Imports the feature pipeline from phishing-classifier (if available)
3. Extracts features from input URLs
4. Makes predictions using the loaded model
5. Returns results via the REST API

## Testing

Test the integration:

```bash
# Test the API endpoint
curl -X POST "http://localhost:8000/phishing" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Get sample URLs
curl "http://localhost:8000/phishing/samples?limit=5"
```

## Next Steps

- Set up authentication (JWT/OAuth2)
- Add rate limiting
- Implement logging and monitoring
- Deploy with Docker
- Set up CI/CD for automated model updates

