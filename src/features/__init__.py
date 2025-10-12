"""
Feature Engineering Module

This module contains all feature engineering logic for the phishing classifier.
Features are organized by category (URL, Domain, Content) and follow a consistent
interface for easy testing and maintenance.

Usage:
    from src.features import FeatureEngineer
    
    engineer = FeatureEngineer()
    X_engineered = engineer.fit_transform(df)
"""

from .base import FeatureEngineer
from .url_features import URLFeatureExtractor
from .domain_features import DomainFeatureExtractor
from .content_features import ContentFeatureExtractor

__all__ = [
    "FeatureEngineer",
    "URLFeatureExtractor", 
    "DomainFeatureExtractor",
    "ContentFeatureExtractor"
]
