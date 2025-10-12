"""
Base Feature Engineering Classes

This module provides the base classes and interfaces for feature engineering.
All feature extractors should inherit from BaseFeatureExtractor and implement
the required methods for consistency and testability.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

# Configure logging for feature engineering
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    This ensures consistent interface across all feature extractors and makes
    testing and maintenance easier. Each extractor should focus on one category
    of features (URL, Domain, Content).
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize the feature extractor.
        
        Args:
            feature_names: List of feature names this extractor will create
        """
        self.feature_names = feature_names
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BaseFeatureExtractor':
        """
        Fit the feature extractor to the data.
        
        This method should learn any parameters needed for feature extraction
        (e.g., thresholds, scaling parameters, vocabulary).
        
        Args:
            df: Training dataframe
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by extracting features.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Dataframe with extracted features
        """
        pass
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the extractor and transform the data in one step.
        
        Args:
            df: Dataframe to fit and transform
            
        Returns:
            Dataframe with extracted features
        """
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names this extractor creates."""
        return self.feature_names.copy()


class FeatureEngineer:
    """
    Main feature engineering orchestrator.
    
    This class coordinates all feature extractors and provides a unified
    interface for feature engineering. It handles the pipeline from raw
    data to engineered features ready for ML models.
    """
    
    def __init__(self):
        """Initialize the feature engineer with all extractors."""
        self.extractors = []
        self.feature_names = []
        self.is_fitted = False
        
    def add_extractor(self, extractor: BaseFeatureExtractor) -> 'FeatureEngineer':
        """
        Add a feature extractor to the pipeline.
        
        Args:
            extractor: Feature extractor to add
            
        Returns:
            self for method chaining
        """
        self.extractors.append(extractor)
        self.feature_names.extend(extractor.get_feature_names())
        logger.info(f"Added extractor with {len(extractor.get_feature_names())} features")
        return self
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit all feature extractors to the data.
        
        Args:
            df: Training dataframe
            
        Returns:
            self for method chaining
        """
        logger.info(f"Fitting feature engineering pipeline on {len(df)} samples")
        
        for i, extractor in enumerate(self.extractors):
            logger.info(f"Fitting extractor {i+1}/{len(self.extractors)}")
            extractor.fit(df)
            
        self.is_fitted = True
        logger.info(f"Feature engineering pipeline fitted. Total features: {len(self.feature_names)}")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using all fitted extractors.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Dataframe with all engineered features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
            
        logger.info(f"Transforming {len(df)} samples")
        
        # Start with original dataframe
        result_df = df.copy()
        
        # Apply each extractor
        for i, extractor in enumerate(self.extractors):
            logger.info(f"Applying extractor {i+1}/{len(self.extractors)}")
            features_df = extractor.transform(df)
            result_df = pd.concat([result_df, features_df], axis=1)
            
        logger.info(f"Feature engineering complete. Shape: {result_df.shape}")
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit all extractors and transform the data.
        
        Args:
            df: Dataframe to fit and transform
            
        Returns:
            Dataframe with all engineered features
        """
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names from all extractors."""
        return self.feature_names.copy()
    
    def get_feature_importance_info(self) -> Dict[str, Any]:
        """
        Get information about feature categories and counts.
        
        Returns:
            Dictionary with feature category information
        """
        info = {}
        for extractor in self.extractors:
            extractor_name = extractor.__class__.__name__
            info[extractor_name] = {
                'feature_count': len(extractor.get_feature_names()),
                'features': extractor.get_feature_names()
            }
        return info
