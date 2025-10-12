#!/usr/bin/env python3
"""
Feature Engineering Demo

This script demonstrates the complete feature engineering pipeline for the
phishing classifier. It shows how to use individual extractors and the
complete pipeline to engineer 20 features from URL data.

Usage:
    python examples/feature_engineering_demo.py
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.pipeline import create_feature_pipeline
from features.url_features import URLFeatureExtractor
from features.domain_features import DomainFeatureExtractor
from features.content_features import ContentFeatureExtractor


def create_sample_data():
    """Create sample data for demonstration."""
    sample_urls = [
        # Legitimate URLs
        'https://www.google.com/search?q=machine+learning',
        'https://github.com/microsoft/vscode',
        'https://stackoverflow.com/questions/tagged/python',
        'https://www.microsoft.com/en-us/',
        'https://www.amazon.com/dp/B08N5WRWNW',
        
        # Suspicious URLs
        'https://bit.ly/abc123xyz',
        'https://urgent-verify-account.com/update-now',
        'https://suspicious-site.tk/click-here-immediately',
        'https://192.168.1.100/login',
        'https://secure-bank-login.example.com/verify',
        
        # Edge cases
        'http://insecure-site.com',
        'https://very-long-domain-name-that-looks-suspicious.com',
        'https://phishing-site.pw/urgent-action-required',
        'https://legitimate-bank.com/secure-login',
        'https://fake-paypal-verification.ml/confirm-account'
    ]
    
    return pd.DataFrame({
        'url': sample_urls,
        'label': ['legitimate'] * 5 + ['suspicious'] * 5 + ['edge_case'] * 5
    })


def demonstrate_individual_extractors(df):
    """Demonstrate individual feature extractors."""
    print("="*60)
    print("INDIVIDUAL FEATURE EXTRACTOR DEMONSTRATION")
    print("="*60)
    
    # URL Features
    print("\n1. URL FEATURE EXTRACTOR")
    print("-" * 30)
    url_extractor = URLFeatureExtractor()
    url_features = url_extractor.fit_transform(df)
    
    print(f"URL Features extracted: {len(url_extractor.get_feature_names())}")
    print("Feature names:", url_extractor.get_feature_names())
    print("\nSample URL features:")
    print(url_features.head())
    
    # Domain Features
    print("\n2. DOMAIN FEATURE EXTRACTOR")
    print("-" * 30)
    domain_extractor = DomainFeatureExtractor()
    domain_features = domain_extractor.fit_transform(df)
    
    print(f"Domain Features extracted: {len(domain_extractor.get_feature_names())}")
    print("Feature names:", domain_extractor.get_feature_names())
    print("\nSample domain features:")
    print(domain_features.head())
    
    # Content Features
    print("\n3. CONTENT FEATURE EXTRACTOR")
    print("-" * 30)
    content_extractor = ContentFeatureExtractor()
    content_features = content_extractor.fit_transform(df)
    
    print(f"Content Features extracted: {len(content_extractor.get_feature_names())}")
    print("Feature names:", content_extractor.get_feature_names())
    print("\nSample content features:")
    print(content_features.head())


def demonstrate_complete_pipeline(df):
    """Demonstrate the complete feature engineering pipeline."""
    print("\n" + "="*60)
    print("COMPLETE FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Create pipeline
    pipeline = create_feature_pipeline()
    
    print(f"Pipeline created with {len(pipeline.get_feature_names())} features")
    print("Feature categories:")
    categories = pipeline.get_feature_categories()
    for category, features in categories.items():
        print(f"  {category}: {len(features)} features")
        for feature in features:
            print(f"    - {feature}")
    
    # Fit and transform
    print("\nFitting and transforming data...")
    engineered_df = pipeline.fit_transform(df)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Engineered data shape: {engineered_df.shape}")
    
    # Show feature statistics
    print("\nFEATURE STATISTICS")
    print("-" * 30)
    feature_stats = engineered_df.describe()
    print(feature_stats)
    
    # Show correlation with labels (if available)
    if 'label' in df.columns:
        print("\nFEATURE CORRELATION WITH LABELS")
        print("-" * 30)
        
        # Create numeric labels
        label_mapping = {'legitimate': 0, 'suspicious': 1, 'edge_case': 0.5}
        numeric_labels = df['label'].map(label_mapping)
        
        # Calculate correlations
        correlations = {}
        for feature in pipeline.get_feature_names():
            if feature in engineered_df.columns:
                corr = engineered_df[feature].corr(numeric_labels)
                correlations[feature] = corr
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
        
        print("Top 10 features by correlation with labels:")
        for feature, corr in sorted_correlations[:10]:
            print(f"  {feature}: {corr:.4f}")


def demonstrate_feature_analysis(df):
    """Demonstrate feature analysis and insights."""
    print("\n" + "="*60)
    print("FEATURE ANALYSIS AND INSIGHTS")
    print("="*60)
    
    pipeline = create_feature_pipeline()
    engineered_df = pipeline.fit_transform(df)
    
    # Analyze URL length patterns
    print("\n1. URL LENGTH ANALYSIS")
    print("-" * 30)
    url_lengths = engineered_df['url_length']
    print(f"URL length statistics:")
    print(f"  Mean: {url_lengths.mean():.2f}")
    print(f"  Median: {url_lengths.median():.2f}")
    print(f"  Min: {url_lengths.min():.2f}")
    print(f"  Max: {url_lengths.max():.2f}")
    
    # Analyze suspicious features
    print("\n2. SUSPICIOUS FEATURE ANALYSIS")
    print("-" * 30)
    suspicious_features = ['is_url_shortened', 'has_ip_address', 'suspicious_keywords_count']
    
    for feature in suspicious_features:
        if feature in engineered_df.columns:
            count = (engineered_df[feature] > 0).sum()
            percentage = (count / len(engineered_df)) * 100
            print(f"  {feature}: {count} samples ({percentage:.1f}%)")
    
    # Analyze domain features
    print("\n3. DOMAIN FEATURE ANALYSIS")
    print("-" * 30)
    domain_features = ['domain_age_days', 'ssl_validity_score', 'suspicious_tld_score']
    
    for feature in domain_features:
        if feature in engineered_df.columns:
            mean_val = engineered_df[feature].mean()
            print(f"  {feature}: {mean_val:.4f} (mean)")
    
    # Show most suspicious samples
    print("\n4. MOST SUSPICIOUS SAMPLES")
    print("-" * 30)
    
    # Calculate suspiciousness score (sum of suspicious features)
    suspicious_cols = ['is_url_shortened', 'has_ip_address', 'suspicious_keywords_count', 
                      'suspicious_tld_score', 'country_risk_score']
    available_suspicious_cols = [col for col in suspicious_cols if col in engineered_df.columns]
    
    if available_suspicious_cols:
        engineered_df['suspiciousness_score'] = engineered_df[available_suspicious_cols].sum(axis=1)
        
        # Get top 5 most suspicious
        top_suspicious = engineered_df.nlargest(5, 'suspiciousness_score')
        
        for idx, row in top_suspicious.iterrows():
            print(f"  Sample {idx}: {df.loc[idx, 'url']}")
            print(f"    Suspiciousness: {row['suspiciousness_score']:.2f}")
            print(f"    URL Length: {row['url_length']}")
            print(f"    Suspicious Keywords: {row.get('suspicious_keywords_count', 0)}")
            print()


def main():
    """Main demonstration function."""
    print("PHISHING CLASSIFIER - FEATURE ENGINEERING DEMO")
    print("=" * 60)
    print("This demo shows how to engineer 20 features from URL data")
    print("for phishing detection using industry best practices.")
    print()
    
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data()
    print(f"Created {len(df)} sample URLs")
    print()
    
    # Demonstrate individual extractors
    demonstrate_individual_extractors(df)
    
    # Demonstrate complete pipeline
    demonstrate_complete_pipeline(df)
    
    # Demonstrate feature analysis
    demonstrate_feature_analysis(df)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("The feature engineering pipeline successfully extracted 20 features")
    print("from URL data, following ML engineering best practices:")
    print()
    print("✓ Modular design with separate extractors")
    print("✓ Comprehensive error handling and validation")
    print("✓ Detailed logging and monitoring")
    print("✓ Extensive testing and documentation")
    print("✓ Industry-standard code organization")
    print()
    print("Next steps:")
    print("1. Run the complete ML pipeline: python src/pipeline.py")
    print("2. Run tests: python -m pytest tests/test_features.py")
    print("3. Integrate with your ML models")


if __name__ == "__main__":
    main()
