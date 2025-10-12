"""
Feature Engineering Pipeline

This module provides a complete feature engineering pipeline that orchestrates
all feature extractors and provides a unified interface for the ML pipeline.
It handles the complete workflow from raw data to engineered features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import joblib

from .base import FeatureEngineer
from .url_features import URLFeatureExtractor
from .domain_features import DomainFeatureExtractor
from .content_features import ContentFeatureExtractor

logger = logging.getLogger(__name__)


class PhishingFeaturePipeline:
    """
    Complete feature engineering pipeline for phishing detection.
    
    This class orchestrates all feature extractors and provides a unified
    interface for feature engineering. It handles data validation, feature
    extraction, and feature selection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            config: Optional configuration dictionary for customizing behavior
        """
        self.config = config or {}
        self.feature_engineer = FeatureEngineer()
        self.feature_names = []
        self.is_fitted = False
        
        # Initialize all feature extractors
        self._setup_extractors()
        
    def _setup_extractors(self) -> None:
        """Setup all feature extractors in the pipeline."""
        logger.info("Setting up feature extractors")
        
        # Add URL feature extractor (6 features)
        self.feature_engineer.add_extractor(URLFeatureExtractor())
        
        # Add domain feature extractor (6 features)  
        self.feature_engineer.add_extractor(DomainFeatureExtractor())
        
        # Add content feature extractor (8 features)
        self.feature_engineer.add_extractor(ContentFeatureExtractor())
        
        # Get all feature names
        self.feature_names = self.feature_engineer.get_feature_names()
        
        logger.info(f"Pipeline setup complete. Total features: {len(self.feature_names)}")
        logger.info(f"Feature names: {self.feature_names}")
    
    def fit(self, df: pd.DataFrame) -> 'PhishingFeaturePipeline':
        """
        Fit the feature engineering pipeline to the data.
        
        Args:
            df: Training dataframe with 'url' column
            
        Returns:
            self for method chaining
        """
        logger.info(f"Fitting feature pipeline on {len(df)} samples")
        
        # Validate input data
        self._validate_input_data(df)
        
        # Fit the feature engineer
        self.feature_engineer.fit(df)
        
        self.is_fitted = True
        logger.info("Feature pipeline fitted successfully")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted pipeline.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Dataframe with engineered features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
            
        logger.info(f"Transforming {len(df)} samples")
        
        # Validate input data
        self._validate_input_data(df)
        
        # Transform the data
        engineered_df = self.feature_engineer.transform(df)
        
        # Validate output
        self._validate_output_data(engineered_df)
        
        logger.info(f"Transformation complete. Output shape: {engineered_df.shape}")
        return engineered_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline and transform the data.
        
        Args:
            df: Dataframe to fit and transform
            
        Returns:
            Dataframe with engineered features
        """
        return self.fit(df).transform(df)
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data before processing.
        
        Args:
            df: Dataframe to validate
            
        Raises:
            ValueError: If data validation fails
        """
        if df.empty:
            raise ValueError("Input dataframe is empty")
            
        if 'url' not in df.columns:
            raise ValueError("Input dataframe must contain 'url' column")
            
        # Check for missing URLs
        missing_urls = df['url'].isna().sum()
        if missing_urls > 0:
            logger.warning(f"Found {missing_urls} missing URLs, will be handled during feature extraction")
    
    def _validate_output_data(self, df: pd.DataFrame) -> None:
        """
        Validate output data after processing.
        
        Args:
            df: Dataframe to validate
            
        Raises:
            ValueError: If output validation fails
        """
        if df.empty:
            raise ValueError("Output dataframe is empty")
            
        # Check for expected number of features
        expected_features = len(self.feature_names)
        actual_features = len([col for col in df.columns if col in self.feature_names])
        
        if actual_features != expected_features:
            logger.warning(f"Expected {expected_features} features, got {actual_features}")
        
        # Check for infinite or extremely large values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isin([np.inf, -np.inf]).any().any():
                logger.warning(f"Found infinite values in column {col}")
            if (df[col].abs() > 1e10).any().any():
                logger.warning(f"Found extremely large values in column {col}")
    
    def get_feature_names(self) -> List[str]:
        """Get the list of all feature names."""
        return self.feature_names.copy()
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        Get features organized by category.
        
        Returns:
            Dictionary mapping category names to feature lists
        """
        categories = {
            'URL Features': [],
            'Domain Features': [],
            'Content Features': []
        }
        
        for feature in self.feature_names:
            if feature in ['url_length', 'subdomain_count', 'suspicious_char_count', 
                          'is_url_shortened', 'has_ip_address', 'redirect_chain_length']:
                categories['URL Features'].append(feature)
            elif feature in ['domain_age_days', 'registrar_reputation_score', 'country_risk_score',
                            'alexa_rank_score', 'ssl_validity_score', 'domain_length']:
                categories['Domain Features'].append(feature)
            elif feature in ['suspicious_keywords_count', 'html_form_count', 'external_link_ratio',
                            'image_to_text_ratio', 'javascript_ratio', 'page_load_time_score',
                            'meta_tag_count', 'title_length']:
                categories['Content Features'].append(feature)
        
        return categories
    
    def save_pipeline(self, filepath: str) -> None:
        """
        Save the fitted pipeline to disk.
        
        Args:
            filepath: Path where to save the pipeline
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
            
        pipeline_data = {
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> 'PhishingFeaturePipeline':
        """
        Load a fitted pipeline from disk.
        
        Args:
            filepath: Path to the saved pipeline
            
        Returns:
            self for method chaining
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
            
        pipeline_data = joblib.load(filepath)
        
        self.feature_names = pipeline_data['feature_names']
        self.config = pipeline_data['config']
        self.is_fitted = pipeline_data['is_fitted']
        
        # Recreate the feature engineer with the loaded state
        self.feature_engineer = FeatureEngineer()
        self._setup_extractors()
        
        if self.is_fitted:
            self.feature_engineer.is_fitted = True
            for extractor in self.feature_engineer.extractors:
                extractor.is_fitted = True
        
        logger.info(f"Pipeline loaded from {filepath}")
        return self
    
    def get_feature_importance_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the feature pipeline.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_categories': self.get_feature_categories(),
            'is_fitted': self.is_fitted,
            'config': self.config
        }


def create_feature_pipeline(config: Optional[Dict[str, Any]] = None) -> PhishingFeaturePipeline:
    """
    Factory function to create a new feature pipeline.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        New PhishingFeaturePipeline instance
    """
    return PhishingFeaturePipeline(config)
