# Feature Engineering Documentation

## Overview

This document describes the comprehensive feature engineering pipeline for the phishing classifier. The pipeline extracts 20 carefully engineered features from URL data, organized into three categories: URL Features, Domain Features, and Content Features.

## Architecture

### Design Principles

- **Modularity**: Each feature category has its own extractor class
- **Testability**: All components are unit tested with comprehensive test coverage
- **Maintainability**: Clean, well-documented code following ML engineering best practices
- **Scalability**: Pipeline can handle large datasets efficiently
- **Robustness**: Comprehensive error handling and input validation

### Pipeline Structure

```
src/features/
├── __init__.py              # Module initialization
├── base.py                  # Base classes and interfaces
├── url_features.py          # URL-based feature extractor (6 features)
├── domain_features.py       # Domain-based feature extractor (6 features)
├── content_features.py      # Content-based feature extractor (8 features)
└── pipeline.py              # Main pipeline orchestrator
```

## Feature Categories

### 1. URL Features (6 features)

URL features analyze the structure and characteristics of the URL itself.

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `url_length` | Total character count in the URL | Phishing URLs are often longer as attackers try to make them look legitimate |
| `subdomain_count` | Number of subdomains in the URL | Phishing sites often use multiple subdomains to appear legitimate |
| `suspicious_char_count` | Count of special characters (@, #, etc.) | Phishing URLs often contain unusual characters |
| `is_url_shortened` | Detection of URL shortening services | Phishing attacks often use URL shorteners to hide destinations |
| `has_ip_address` | Presence of IP address instead of domain | Legitimate sites rarely use IP addresses directly |
| `redirect_chain_length` | Number of redirects before final page | Phishing URLs often have complex redirect chains |

### 2. Domain Features (6 features)

Domain features analyze the reputation, age, and characteristics of the domain.

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `domain_age_days` | Age of the domain in days | New domains are more suspicious than established ones |
| `registrar_reputation_score` | Registrar reputation score (0-1) | Reputable registrars are less likely to host phishing sites |
| `country_risk_score` | Geographic risk score (0-1) | Some countries are known for hosting more phishing sites |
| `alexa_rank_score` | Website popularity ranking (0-1) | Legitimate sites typically have better rankings |
| `ssl_validity_score` | SSL certificate validity (0-1) | Legitimate sites have valid SSL certificates |
| `domain_length` | Length of the domain name | Phishing domains are often longer to appear legitimate |

### 3. Content Features (8 features)

Content features analyze the actual webpage content and structure.

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `suspicious_keywords_count` | Count of phishing-related keywords | Phishing pages often contain urgent or suspicious language |
| `html_form_count` | Number of forms on the page | Phishing sites often have simple credential harvesting forms |
| `external_link_ratio` | Ratio of external to internal links | Phishing sites often have many external links |
| `image_to_text_ratio` | Ratio of images to text content | Phishing sites often use more images and less text |
| `javascript_ratio` | Percentage of JavaScript content | Phishing sites often use less dynamic content |
| `page_load_time_score` | Estimated page load time (0-1) | Phishing sites load differently than legitimate sites |
| `meta_tag_count` | Number of meta tags in HTML | Legitimate sites have more comprehensive meta tags |
| `title_length` | Length of the page title | Phishing sites often have shorter, generic titles |
| `suspicious_tld_score` | Suspicious TLD detection (0-1) | Some TLDs are commonly used for phishing |

## Usage

### Basic Usage

```python
from src.features.pipeline import create_feature_pipeline
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'url': [
        'https://www.google.com',
        'https://suspicious-site.tk/urgent-verify',
        'https://legitimate-bank.com/login'
    ]
})

# Create and use pipeline
pipeline = create_feature_pipeline()
engineered_df = pipeline.fit_transform(df)

print(f"Engineered {len(engineered_df.columns)} features")
print(engineered_df.head())
```

### Advanced Usage

```python
# Create pipeline with custom configuration
pipeline = create_feature_pipeline(config={'verbose': True})

# Fit and transform separately
pipeline.fit(df)
engineered_df = pipeline.transform(df)

# Get feature information
feature_names = pipeline.get_feature_names()
categories = pipeline.get_feature_categories()

# Save pipeline for later use
pipeline.save_pipeline('models/feature_pipeline.joblib')

# Load saved pipeline
new_pipeline = create_feature_pipeline()
new_pipeline.load_pipeline('models/feature_pipeline.joblib')
```

### Individual Extractors

```python
from src.features.url_features import URLFeatureExtractor
from src.features.domain_features import DomainFeatureExtractor
from src.features.content_features import ContentFeatureExtractor

# Use individual extractors
url_extractor = URLFeatureExtractor()
url_features = url_extractor.fit_transform(df)

domain_extractor = DomainFeatureExtractor()
domain_features = domain_extractor.fit_transform(df)

content_extractor = ContentFeatureExtractor()
content_features = content_extractor.fit_transform(df)
```

## Integration with ML Pipeline

The feature engineering pipeline is fully integrated with the main ML pipeline:

```bash
# Run complete pipeline with feature engineering
python src/pipeline.py --stage all

# Run only feature engineering
python src/pipeline.py --stage features

# Save engineered features
python src/pipeline.py --stage features --save-features

# Load pre-engineered features
python src/pipeline.py --stage train --load-features data/processed/engineered_features.csv
```

## Testing

Comprehensive test suite covers all components:

```bash
# Run all feature engineering tests
python -m pytest tests/test_features.py -v

# Run specific test categories
python -m pytest tests/test_features.py::TestURLFeatureExtractor -v
python -m pytest tests/test_features.py::TestDomainFeatureExtractor -v
python -m pytest tests/test_features.py::TestContentFeatureExtractor -v
python -m pytest tests/test_features.py::TestPhishingFeaturePipeline -v
```

## Demo and Examples

Run the feature engineering demo to see the pipeline in action:

```bash
python examples/feature_engineering_demo.py
```

This demo shows:
- Individual feature extractor usage
- Complete pipeline operation
- Feature analysis and insights
- Correlation analysis with labels

## Performance Considerations

### Memory Usage
- Pipeline processes data in chunks to minimize memory usage
- Feature extractors are optimized for efficiency
- Large datasets are handled gracefully with progress logging

### Speed Optimization
- URL parsing is cached where possible
- Regex patterns are compiled once and reused
- Vectorized operations are used throughout

### Error Handling
- Comprehensive input validation
- Graceful handling of malformed URLs
- Robust null value handling
- Detailed error logging

## Extensibility

### Adding New Features

To add new features, create a new extractor class:

```python
from src.features.base import BaseFeatureExtractor

class CustomFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__(['custom_feature_1', 'custom_feature_2'])
    
    def fit(self, df):
        # Learn parameters from data
        self.is_fitted = True
        return self
    
    def transform(self, df):
        # Extract features
        features_df = pd.DataFrame(index=df.index)
        features_df['custom_feature_1'] = df['url'].apply(self._extract_feature_1)
        features_df['custom_feature_2'] = df['url'].apply(self._extract_feature_2)
        return features_df
```

### Adding New Categories

To add a new feature category:

1. Create a new extractor class in a new file
2. Add it to the pipeline in `pipeline.py`
3. Update the feature categories in `get_feature_categories()`
4. Add comprehensive tests

## Monitoring and Logging

The pipeline includes comprehensive logging:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Pipeline logs all major operations
# - Feature extraction progress
# - Data validation results
# - Performance metrics
# - Error conditions
```

## Best Practices

### Code Quality
- All code follows PEP 8 style guidelines
- Comprehensive docstrings for all functions
- Type hints throughout
- Extensive error handling

### Testing
- Unit tests for all components
- Integration tests for the complete pipeline
- Edge case testing
- Performance benchmarking

### Documentation
- Clear, comprehensive documentation
- Usage examples for all components
- API reference documentation
- Troubleshooting guides

## Troubleshooting

### Common Issues

1. **Missing URL column**: Ensure your DataFrame has a 'url' column
2. **Memory issues**: Use the pipeline with smaller data chunks
3. **Feature extraction errors**: Check URL format and encoding
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Real-time feature extraction for streaming data
- Advanced feature selection algorithms
- Feature importance analysis
- Automated feature engineering
- Integration with feature stores
- Real-time monitoring and alerting

## Contributing

When contributing to the feature engineering pipeline:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure backward compatibility
5. Follow ML engineering best practices
