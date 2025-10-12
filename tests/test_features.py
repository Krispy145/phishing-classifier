"""
Test Suite for Feature Engineering

This module contains comprehensive tests for all feature engineering components.
Tests cover individual extractors, the complete pipeline, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

from src.features.base import BaseFeatureExtractor, FeatureEngineer
from src.features.url_features import URLFeatureExtractor
from src.features.domain_features import DomainFeatureExtractor
from src.features.content_features import ContentFeatureExtractor
from src.features.pipeline import PhishingFeaturePipeline, create_feature_pipeline

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.mark.unit
@pytest.mark.url
class TestURLFeatureExtractor:
    """Test cases for URL feature extractor."""
    
    def setup_method(self):
        """Setup test data."""
        self.extractor = URLFeatureExtractor()
        self.test_data = pd.DataFrame({
            'url': [
                'https://www.google.com/search?q=test',
                'http://bit.ly/abc123',
                'https://192.168.1.1/login',
                'https://secure-bank-login.example.com/verify',
                'https://phishing-site.com/urgent/verify-account',
                None,  # Test null handling
                ''  # Test empty string
            ]
        })
    
    def test_initialization(self):
        """Test extractor initialization."""
        assert len(self.extractor.get_feature_names()) == 6
        assert 'url_length' in self.extractor.get_feature_names()
        assert 'subdomain_count' in self.extractor.get_feature_names()
    
    def test_fit(self):
        """Test fit method."""
        result = self.extractor.fit(self.test_data)
        assert result is self.extractor
        assert self.extractor.is_fitted
    
    def test_fit_missing_url_column(self):
        """Test fit with missing URL column."""
        bad_data = pd.DataFrame({'other_column': [1, 2, 3]})
        with pytest.raises(ValueError, match="must contain 'url' column"):
            self.extractor.fit(bad_data)
    
    def test_transform(self):
        """Test transform method."""
        self.extractor.fit(self.test_data)
        result = self.extractor.transform(self.test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.test_data)
        assert len(result.columns) == 6
        
        # Check specific features
        assert 'url_length' in result.columns
        assert 'is_url_shortened' in result.columns
        assert 'has_ip_address' in result.columns
    
    def test_url_length_extraction(self):
        """Test URL length feature extraction."""
        self.extractor.fit(self.test_data)
        result = self.extractor.transform(self.test_data)
        
        # Check that URL lengths are calculated correctly
        assert result.loc[0, 'url_length'] > 0  # Google URL should have length
        assert result.loc[5, 'url_length'] == 0  # None should be 0
        assert result.loc[6, 'url_length'] == 0  # Empty string should be 0
    
    def test_url_shortening_detection(self):
        """Test URL shortening detection."""
        self.extractor.fit(self.test_data)
        result = self.extractor.transform(self.test_data)
        
        # bit.ly should be detected as shortened
        assert result.loc[1, 'is_url_shortened'] == 1
        # Google URL should not be shortened
        assert result.loc[0, 'is_url_shortened'] == 0
    
    def test_ip_address_detection(self):
        """Test IP address detection."""
        self.extractor.fit(self.test_data)
        result = self.extractor.transform(self.test_data)
        
        # IP address should be detected
        assert result.loc[2, 'has_ip_address'] == 1
        # Regular domain should not have IP
        assert result.loc[0, 'has_ip_address'] == 0


@pytest.mark.unit
@pytest.mark.domain
class TestDomainFeatureExtractor:
    """Test cases for domain feature extractor."""
    
    def setup_method(self):
        """Setup test data."""
        self.extractor = DomainFeatureExtractor()
        self.test_data = pd.DataFrame({
            'url': [
                'https://www.google.com',
                'https://secure-bank-login.example.com',
                'https://suspicious-site.tk',
                'https://legitimate-site.com',
                None
            ]
        })
    
    def test_initialization(self):
        """Test extractor initialization."""
        assert len(self.extractor.get_feature_names()) == 6
        assert 'domain_age_days' in self.extractor.get_feature_names()
        assert 'ssl_validity_score' in self.extractor.get_feature_names()
    
    def test_domain_age_extraction(self):
        """Test domain age feature extraction."""
        self.extractor.fit(self.test_data)
        result = self.extractor.transform(self.test_data)
        
        # Google should have high age
        assert result.loc[0, 'domain_age_days'] > 0
        # Suspicious site should have low age
        assert result.loc[2, 'domain_age_days'] < result.loc[0, 'domain_age_days']
    
    def test_ssl_validity_extraction(self):
        """Test SSL validity feature extraction."""
        self.extractor.fit(self.test_data)
        result = self.extractor.transform(self.test_data)
        
        # HTTPS URLs should have higher SSL scores
        assert result.loc[0, 'ssl_validity_score'] > 0.5
        assert result.loc[1, 'ssl_validity_score'] > 0.5


@pytest.mark.unit
@pytest.mark.content
class TestContentFeatureExtractor:
    """Test cases for content feature extractor."""
    
    def setup_method(self):
        """Setup test data."""
        self.extractor = ContentFeatureExtractor()
        self.test_data = pd.DataFrame({
            'url': [
                'https://legitimate-bank.com/login',
                'https://urgent-verify-account.com/update',
                'https://suspicious-site.tk/click-here',
                'https://normal-site.com/about',
                None
            ]
        })
    
    def test_initialization(self):
        """Test extractor initialization."""
        assert len(self.extractor.get_feature_names()) == 8
        assert 'suspicious_keywords_count' in self.extractor.get_feature_names()
        assert 'title_length' in self.extractor.get_feature_names()
    
    def test_suspicious_keywords_extraction(self):
        """Test suspicious keywords feature extraction."""
        self.extractor.fit(self.test_data)
        result = self.extractor.transform(self.test_data)
        
        # URL with 'urgent' and 'verify' should have higher count
        assert result.loc[1, 'suspicious_keywords_count'] > result.loc[0, 'suspicious_keywords_count']
        # URL with 'click-here' should have keywords
        assert result.loc[2, 'suspicious_keywords_count'] > 0
    
    def test_title_length_extraction(self):
        """Test title length feature extraction."""
        self.extractor.fit(self.test_data)
        result = self.extractor.transform(self.test_data)
        
        # All URLs should have title length > 0
        assert result.loc[0, 'title_length'] > 0
        assert result.loc[1, 'title_length'] > 0
        assert result.loc[2, 'title_length'] > 0


@pytest.mark.integration
@pytest.mark.feature
class TestFeatureEngineer:
    """Test cases for the main feature engineer."""
    
    def setup_method(self):
        """Setup test data."""
        self.engineer = FeatureEngineer()
        self.test_data = pd.DataFrame({
            'url': [
                'https://www.google.com',
                'https://suspicious-site.tk/urgent-verify',
                'https://legitimate-bank.com/login'
            ]
        })
    
    def test_add_extractor(self):
        """Test adding extractors."""
        url_extractor = URLFeatureExtractor()
        self.engineer.add_extractor(url_extractor)
        
        assert len(self.engineer.extractors) == 1
        assert len(self.engineer.feature_names) == 6
    
    def test_fit_transform(self):
        """Test fit and transform pipeline."""
        # Add all extractors
        self.engineer.add_extractor(URLFeatureExtractor())
        self.engineer.add_extractor(DomainFeatureExtractor())
        self.engineer.add_extractor(ContentFeatureExtractor())
        
        # Fit and transform
        result = self.engineer.fit_transform(self.test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.test_data)
        assert len(result.columns) > 20  # Should have all 20 features


@pytest.mark.integration
@pytest.mark.pipeline
class TestPhishingFeaturePipeline:
    """Test cases for the complete feature pipeline."""
    
    def setup_method(self):
        """Setup test data."""
        self.pipeline = create_feature_pipeline()
        self.test_data = pd.DataFrame({
            'url': [
                'https://www.google.com/search',
                'https://bit.ly/abc123',
                'https://urgent-verify-account.com/update-now',
                'https://suspicious-site.tk/click-here',
                'https://legitimate-bank.com/secure-login'
            ]
        })
    
    def test_pipeline_creation(self):
        """Test pipeline creation."""
        assert isinstance(self.pipeline, PhishingFeaturePipeline)
        assert len(self.pipeline.get_feature_names()) == 20
    
    def test_fit_transform(self):
        """Test complete pipeline fit and transform."""
        result = self.pipeline.fit_transform(self.test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.test_data)
        
        # Check that we have exactly 20 engineered features
        feature_columns = self.pipeline.get_feature_names()
        assert len(feature_columns) == 20
        assert all(feature in result.columns for feature in feature_columns)
        
        # Check that all expected features are present
        expected_features = [
            'url_length', 'subdomain_count', 'suspicious_char_count',
            'is_url_shortened', 'has_ip_address', 'redirect_chain_length',
            'domain_age_days', 'registrar_reputation_score', 'country_risk_score',
            'alexa_rank_score', 'ssl_validity_score', 'domain_length',
            'suspicious_keywords_count', 'html_form_count', 'external_link_ratio',
            'image_to_text_ratio', 'javascript_ratio', 'page_load_time_score',
            'meta_tag_count', 'title_length'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_feature_categories(self):
        """Test feature categorization."""
        categories = self.pipeline.get_feature_categories()
        
        assert 'URL Features' in categories
        assert 'Domain Features' in categories
        assert 'Content Features' in categories
        
        # Check that categories have expected number of features
        assert len(categories['URL Features']) == 6
        assert len(categories['Domain Features']) == 6
        assert len(categories['Content Features']) == 8
    
    def test_validation_errors(self):
        """Test input validation."""
        # Test empty dataframe
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Input dataframe is empty"):
            self.pipeline.fit(empty_df)
        
        # Test missing URL column
        bad_df = pd.DataFrame({'other_column': [1, 2, 3]})
        with pytest.raises(ValueError, match="must contain 'url' column"):
            self.pipeline.fit(bad_df)
    
    def test_transform_without_fit(self):
        """Test transform without fitting first."""
        with pytest.raises(ValueError, match="Pipeline must be fitted before transform"):
            self.pipeline.transform(self.test_data)
    
    def test_save_load_pipeline(self, tmp_path):
        """Test saving and loading pipeline."""
        # Fit the pipeline
        self.pipeline.fit(self.test_data)
        
        # Save pipeline
        save_path = tmp_path / "pipeline.joblib"
        self.pipeline.save_pipeline(str(save_path))
        
        # Load pipeline
        new_pipeline = create_feature_pipeline()
        new_pipeline.load_pipeline(str(save_path))
        
        # Test that loaded pipeline works
        result = new_pipeline.transform(self.test_data)
        
        # Check that we have exactly 20 engineered features
        feature_columns = new_pipeline.get_feature_names()
        assert len(feature_columns) == 20
        assert all(feature in result.columns for feature in feature_columns)


@pytest.mark.unit
@pytest.mark.slow
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_null_url_handling(self):
        """Test handling of null URLs."""
        pipeline = create_feature_pipeline()
        test_data = pd.DataFrame({
            'url': [None, '', 'https://valid.com', np.nan]
        })
        
        result = pipeline.fit_transform(test_data)
        
        # Should not raise errors and should handle nulls gracefully
        assert len(result) == 4
        
        # Check that engineered features don't have NaN values (original URL column may have NaN)
        feature_columns = pipeline.get_feature_names()
        engineered_features = result[feature_columns]
        assert not engineered_features.isnull().any().any()  # No NaN values in engineered features
    
    def test_malformed_url_handling(self):
        """Test handling of malformed URLs."""
        pipeline = create_feature_pipeline()
        test_data = pd.DataFrame({
            'url': [
                'not-a-url',
                'http://',
                'https://',
                'ftp://invalid',
                'javascript:alert(1)',
                'https://valid.com'
            ]
        })
        
        result = pipeline.fit_transform(test_data)
        
        # Should not raise errors
        assert len(result) == 6
        assert not result.isnull().any().any()
    
    def test_very_long_url(self):
        """Test handling of very long URLs."""
        pipeline = create_feature_pipeline()
        long_url = 'https://' + 'a' * 1000 + '.com'
        test_data = pd.DataFrame({'url': [long_url]})
        
        result = pipeline.fit_transform(test_data)
        
        # Should handle long URLs without issues
        assert len(result) == 1
        assert result.loc[0, 'url_length'] == len(long_url)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
