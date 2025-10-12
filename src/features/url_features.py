"""
URL Feature Extractor

This module extracts features from URLs that are indicative of phishing behavior.
URL-based features are often the most reliable indicators as they're harder
for attackers to manipulate compared to content features.

Features extracted:
- URL length
- Subdomain count  
- Suspicious characters
- URL shortening detection
- IP address in URL
- Redirect chain length
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from typing import List, Dict, Any
import logging

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class URLFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts URL-based features for phishing detection.
    
    URL features are particularly effective because they're difficult for
    attackers to manipulate and often contain clear indicators of malicious
    intent (e.g., suspicious characters, unusual length patterns).
    """
    
    def __init__(self):
        """Initialize URL feature extractor with predefined feature names."""
        feature_names = [
            'url_length',
            'subdomain_count', 
            'suspicious_char_count',
            'is_url_shortened',
            'has_ip_address',
            'redirect_chain_length'
        ]
        super().__init__(feature_names)
        
        # Define suspicious characters commonly used in phishing URLs
        self.suspicious_chars = r'[@#%&+=\s]'
        
        # Common URL shortening services
        self.shortening_services = {
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 
            'short.link', 'is.gd', 'v.gd', 'tiny.cc', 'shorturl.at'
        }
        
        # IP address regex pattern
        self.ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        
    def fit(self, df: pd.DataFrame) -> 'URLFeatureExtractor':
        """
        Fit the URL feature extractor.
        
        For URL features, we don't need to learn any parameters from the data
        as all features are based on URL structure and patterns.
        
        Args:
            df: Training dataframe (must contain 'url' column)
            
        Returns:
            self for method chaining
        """
        if 'url' not in df.columns:
            raise ValueError("Dataframe must contain 'url' column for URL feature extraction")
            
        logger.info("Fitting URL feature extractor")
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract URL features from the dataframe.
        
        Args:
            df: Dataframe containing 'url' column
            
        Returns:
            Dataframe with URL features
        """
        if not self.is_fitted:
            raise ValueError("URLFeatureExtractor must be fitted before transform")
            
        logger.info(f"Extracting URL features from {len(df)} URLs")
        
        # Initialize result dataframe
        features_df = pd.DataFrame(index=df.index)
        
        # Extract each feature
        features_df['url_length'] = df['url'].apply(self._extract_url_length)
        features_df['subdomain_count'] = df['url'].apply(self._extract_subdomain_count)
        features_df['suspicious_char_count'] = df['url'].apply(self._extract_suspicious_char_count)
        features_df['is_url_shortened'] = df['url'].apply(self._detect_url_shortening)
        features_df['has_ip_address'] = df['url'].apply(self._detect_ip_address)
        features_df['redirect_chain_length'] = df['url'].apply(self._extract_redirect_chain_length)
        
        # Handle any NaN values that might have been created
        features_df = features_df.fillna(0)
        
        logger.info(f"URL feature extraction complete. Shape: {features_df.shape}")
        return features_df
    
    def _extract_url_length(self, url: str) -> int:
        """
        Extract the total length of the URL.
        
        Phishing URLs are often longer than legitimate ones as attackers
        try to make them look more legitimate by adding extra characters.
        
        Args:
            url: URL string
            
        Returns:
            Length of the URL
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
        return len(url)
    
    def _extract_subdomain_count(self, url: str) -> int:
        """
        Count the number of subdomains in the URL.
        
        Phishing sites often use multiple subdomains to make URLs look
        more legitimate (e.g., secure-bank-login.example.com).
        
        Args:
            url: URL string
            
        Returns:
            Number of subdomains
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0
                
            # Split by dots and count subdomains (exclude TLD)
            parts = hostname.split('.')
            return max(0, len(parts) - 2)  # -2 for domain and TLD
            
        except Exception:
            return 0
    
    def _extract_suspicious_char_count(self, url: str) -> int:
        """
        Count suspicious characters in the URL.
        
        Phishing URLs often contain special characters that legitimate
        URLs typically don't have.
        
        Args:
            url: URL string
            
        Returns:
            Count of suspicious characters
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        return len(re.findall(self.suspicious_chars, url))
    
    def _detect_url_shortening(self, url: str) -> int:
        """
        Detect if URL uses a shortening service.
        
        Phishing attacks often use URL shorteners to hide the true destination
        and make malicious links look more legitimate.
        
        Args:
            url: URL string
            
        Returns:
            1 if URL is shortened, 0 otherwise
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0
                
            # Check if hostname matches any known shortening service
            return 1 if hostname.lower() in self.shortening_services else 0
            
        except Exception:
            return 0
    
    def _detect_ip_address(self, url: str) -> int:
        """
        Detect if URL contains an IP address instead of domain name.
        
        Legitimate websites rarely use IP addresses directly in URLs,
        while phishing sites often do to avoid detection.
        
        Args:
            url: URL string
            
        Returns:
            1 if URL contains IP address, 0 otherwise
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        return 1 if re.search(self.ip_pattern, url) else 0
    
    def _extract_redirect_chain_length(self, url: str) -> int:
        """
        Estimate redirect chain length based on URL structure.
        
        Phishing URLs often have complex redirect chains to hide the
        final malicious destination. This is a simplified estimation.
        
        Args:
            url: URL string
            
        Returns:
            Estimated redirect chain length (simplified)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        # Simple heuristic: count redirect indicators in URL
        redirect_indicators = ['redirect', 'goto', 'link', 'url=']
        count = 0
        url_lower = url.lower()
        
        for indicator in redirect_indicators:
            count += url_lower.count(indicator)
            
        return min(count, 5)  # Cap at 5 to avoid extreme values
