"""
Domain Feature Extractor

This module extracts domain-based features that are indicative of phishing behavior.
Domain features focus on the reputation, age, and characteristics of the domain
itself rather than the URL structure.

Features extracted:
- Domain age
- Domain registrar reputation
- Country of origin
- Alexa rank
- SSL certificate validity
- Domain length
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class DomainFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts domain-based features for phishing detection.
    
    Domain features are valuable because they're harder for attackers to
    manipulate and often reveal the true nature of a website through
    reputation, age, and registration details.
    """
    
    def __init__(self):
        """Initialize domain feature extractor with predefined feature names."""
        feature_names = [
            'domain_age_days',
            'registrar_reputation_score',
            'country_risk_score', 
            'alexa_rank_score',
            'ssl_validity_score',
            'domain_length'
        ]
        super().__init__(feature_names)
        
        # Reputable registrars (higher score = more reputable)
        self.registrar_scores = {
            'godaddy.com': 0.9,
            'namecheap.com': 0.8,
            'enom.com': 0.8,
            'tucows.com': 0.7,
            'network solutions': 0.7,
            'register.com': 0.6,
            '1and1.com': 0.6,
            'hostgator.com': 0.5,
            'bluehost.com': 0.5,
            'name.com': 0.4,
            'default': 0.3  # Unknown registrars get lower score
        }
        
        # Country risk scores (higher = more risky for phishing)
        self.country_risk_scores = {
            'US': 0.1, 'CA': 0.1, 'GB': 0.1, 'AU': 0.1, 'DE': 0.1,
            'FR': 0.2, 'JP': 0.2, 'NL': 0.2, 'SE': 0.2, 'CH': 0.2,
            'CN': 0.7, 'RU': 0.8, 'UA': 0.8, 'BY': 0.8, 'KZ': 0.7,
            'default': 0.5  # Unknown countries get medium risk
        }
        
        # Suspicious TLDs that are commonly used for phishing
        self.suspicious_tlds = {
            '.tk', '.ml', '.ga', '.cf', '.pw', '.top', '.click',
            '.download', '.stream', '.online', '.site', '.website'
        }
        
    def fit(self, df: pd.DataFrame) -> 'DomainFeatureExtractor':
        """
        Fit the domain feature extractor.
        
        For domain features, we don't need to learn parameters from the data
        as all features are based on domain characteristics and reputation.
        
        Args:
            df: Training dataframe (must contain 'url' column)
            
        Returns:
            self for method chaining
        """
        if 'url' not in df.columns:
            raise ValueError("Dataframe must contain 'url' column for domain feature extraction")
            
        logger.info("Fitting domain feature extractor")
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract domain features from the dataframe.
        
        Args:
            df: Dataframe containing 'url' column
            
        Returns:
            Dataframe with domain features
        """
        if not self.is_fitted:
            raise ValueError("DomainFeatureExtractor must be fitted before transform")
            
        logger.info(f"Extracting domain features from {len(df)} URLs")
        
        # Initialize result dataframe
        features_df = pd.DataFrame(index=df.index)
        
        # Extract each feature
        features_df['domain_age_days'] = df['url'].apply(self._extract_domain_age)
        features_df['registrar_reputation_score'] = df['url'].apply(self._extract_registrar_score)
        features_df['country_risk_score'] = df['url'].apply(self._extract_country_risk)
        features_df['alexa_rank_score'] = df['url'].apply(self._extract_alexa_rank)
        features_df['ssl_validity_score'] = df['url'].apply(self._extract_ssl_validity)
        features_df['domain_length'] = df['url'].apply(self._extract_domain_length)
        
        # Handle any NaN values
        features_df = features_df.fillna(0)
        
        logger.info(f"Domain feature extraction complete. Shape: {features_df.shape}")
        return features_df
    
    def _extract_domain_age(self, url: str) -> int:
        """
        Extract domain age in days.
        
        New domains are more suspicious as legitimate businesses typically
        have established domains. This is a simplified estimation.
        
        Args:
            url: URL string
            
        Returns:
            Estimated domain age in days
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0
                
            # Simplified domain age estimation based on domain characteristics
            # In a real implementation, you'd query WHOIS data
            domain = hostname.lower()
            
            # Very old domains (established companies)
            if any(established in domain for established in ['google', 'microsoft', 'apple', 'amazon']):
                return 3650  # 10+ years
                
            # Medium age domains (common patterns)
            if len(domain) < 10 and '.' in domain:
                return 1095  # 3 years
                
            # New domains (suspicious patterns)
            if any(suspicious in domain for suspicious in ['secure', 'login', 'verify', 'update']):
                return 30  # 1 month
                
            # Default estimation
            return 365  # 1 year
            
        except Exception:
            return 0
    
    def _extract_registrar_score(self, url: str) -> float:
        """
        Extract registrar reputation score.
        
        Reputable registrars are less likely to host phishing sites,
        while unknown or cheap registrars are more suspicious.
        
        Args:
            url: URL string
            
        Returns:
            Registrar reputation score (0-1, higher = more reputable)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0.0
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0.0
                
            # Simplified registrar detection based on domain patterns
            # In a real implementation, you'd query WHOIS data
            domain = hostname.lower()
            
            # Check for known registrar patterns
            for registrar, score in self.registrar_scores.items():
                if registrar in domain or any(reg in domain for reg in registrar.split()):
                    return score
                    
            # Check for suspicious patterns
            if any(suspicious in domain for suspicious in ['temp', 'fake', 'test', 'demo']):
                return 0.1
                
            return self.registrar_scores['default']
            
        except Exception:
            return 0.0
    
    def _extract_country_risk(self, url: str) -> float:
        """
        Extract country risk score.
        
        Some countries are known for hosting more phishing sites,
        while others have better cybersecurity practices.
        
        Args:
            url: URL string
            
        Returns:
            Country risk score (0-1, higher = more risky)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0.5
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0.5
                
            # Simplified country detection based on TLD and patterns
            # In a real implementation, you'd use GeoIP databases
            domain = hostname.lower()
            
            # Check TLD for country indicators
            if domain.endswith('.us'):
                return self.country_risk_scores['US']
            elif domain.endswith('.ca'):
                return self.country_risk_scores['CA']
            elif domain.endswith('.uk') or domain.endswith('.co.uk'):
                return self.country_risk_scores['GB']
            elif domain.endswith('.de'):
                return self.country_risk_scores['DE']
            elif domain.endswith('.ru'):
                return self.country_risk_scores['RU']
            elif domain.endswith('.cn'):
                return self.country_risk_scores['CN']
            elif domain.endswith('.tk') or domain.endswith('.ml'):
                return 0.9  # Very high risk TLDs
                
            # Check for suspicious country-related patterns
            if any(country in domain for country in ['russia', 'china', 'ukraine']):
                return 0.8
                
            return self.country_risk_scores['default']
            
        except Exception:
            return 0.5
    
    def _extract_alexa_rank(self, url: str) -> float:
        """
        Extract Alexa rank score.
        
        Legitimate websites typically have better Alexa rankings,
        while phishing sites are usually unranked or very low ranked.
        
        Args:
            url: URL string
            
        Returns:
            Alexa rank score (0-1, higher = better ranking)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0.0
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0.0
                
            # Simplified Alexa rank estimation
            # In a real implementation, you'd query Alexa API
            domain = hostname.lower()
            
            # Top websites (very high score)
            if any(top in domain for top in ['google', 'facebook', 'youtube', 'amazon', 'microsoft']):
                return 1.0
                
            # Well-known domains (high score)
            if any(known in domain for known in ['github', 'stackoverflow', 'reddit', 'wikipedia']):
                return 0.8
                
            # Suspicious patterns (low score)
            if any(suspicious in domain for suspicious in ['secure', 'login', 'verify', 'update', 'bank']):
                return 0.1
                
            # Default for unknown domains
            return 0.3
            
        except Exception:
            return 0.0
    
    def _extract_ssl_validity(self, url: str) -> float:
        """
        Extract SSL certificate validity score.
        
        Legitimate websites typically have valid SSL certificates,
        while phishing sites often have invalid or missing certificates.
        
        Args:
            url: URL string
            
        Returns:
            SSL validity score (0-1, higher = more valid)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0.0
            
        try:
            parsed = urlparse(url)
            if parsed.scheme == 'https':
                # HTTPS URLs get higher score
                return 0.8
            elif parsed.scheme == 'http':
                # HTTP URLs get lower score
                return 0.2
            else:
                # Other schemes get very low score
                return 0.1
                
        except Exception:
            return 0.0
    
    def _extract_domain_length(self, url: str) -> int:
        """
        Extract domain name length.
        
        Phishing domains are often longer as attackers try to make them
        look more legitimate by adding extra characters.
        
        Args:
            url: URL string
            
        Returns:
            Length of the domain name
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0
                
            return len(hostname)
            
        except Exception:
            return 0
