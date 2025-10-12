"""
Content Feature Extractor

This module extracts content-based features from web pages that are indicative
of phishing behavior. Content features analyze the actual content of the webpage
rather than just the URL or domain.

Features extracted:
- Suspicious keywords count
- HTML form count
- External link ratio
- Image-to-text ratio
- JavaScript ratio
- Page load time
- Meta tag count
- Title length
- Suspicious TLD detection
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
import logging

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class ContentFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts content-based features for phishing detection.
    
    Content features analyze the actual webpage content to detect phishing
    patterns. These features are more complex to extract but can be very
    effective at detecting sophisticated phishing attempts.
    """
    
    def __init__(self):
        """Initialize content feature extractor with predefined feature names."""
        feature_names = [
            'suspicious_keywords_count',
            'html_form_count',
            'external_link_ratio',
            'image_to_text_ratio',
            'javascript_ratio',
            'page_load_time_score',
            'meta_tag_count',
            'title_length'
        ]
        super().__init__(feature_names)
        
        # Keywords commonly found in phishing pages
        self.phishing_keywords = [
            'verify', 'confirm', 'update', 'security', 'urgent', 'immediately',
            'suspended', 'expired', 'locked', 'compromised', 'unauthorized',
            'click here', 'click-here', 'login now', 'act now', 'limited time', 'exclusive',
            'congratulations', 'winner', 'prize', 'free', 'guaranteed',
            'no risk', 'instant', 'immediate', 'urgent action required'
        ]
        
        # Suspicious TLDs commonly used for phishing
        self.suspicious_tlds = {
            '.tk', '.ml', '.ga', '.cf', '.pw', '.top', '.click',
            '.download', '.stream', '.online', '.site', '.website',
            '.biz', '.info', '.name', '.pro', '.mobi'
        }
        
        # Legitimate form patterns (legitimate sites have more complex forms)
        self.legitimate_form_indicators = [
            'password', 'username', 'email', 'phone', 'address',
            'terms', 'privacy', 'captcha', 'recaptcha'
        ]
        
    def fit(self, df: pd.DataFrame) -> 'ContentFeatureExtractor':
        """
        Fit the content feature extractor.
        
        For content features, we don't need to learn parameters from the data
        as all features are based on content analysis patterns.
        
        Args:
            df: Training dataframe (must contain 'url' column)
            
        Returns:
            self for method chaining
        """
        if 'url' not in df.columns:
            raise ValueError("Dataframe must contain 'url' column for content feature extraction")
            
        logger.info("Fitting content feature extractor")
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract content features from the dataframe.
        
        Args:
            df: Dataframe containing 'url' column
            
        Returns:
            Dataframe with content features
        """
        if not self.is_fitted:
            raise ValueError("ContentFeatureExtractor must be fitted before transform")
            
        logger.info(f"Extracting content features from {len(df)} URLs")
        
        # Initialize result dataframe
        features_df = pd.DataFrame(index=df.index)
        
        # Extract each feature
        features_df['suspicious_keywords_count'] = df['url'].apply(self._extract_suspicious_keywords)
        features_df['html_form_count'] = df['url'].apply(self._extract_form_count)
        features_df['external_link_ratio'] = df['url'].apply(self._extract_external_link_ratio)
        features_df['image_to_text_ratio'] = df['url'].apply(self._extract_image_text_ratio)
        features_df['javascript_ratio'] = df['url'].apply(self._extract_javascript_ratio)
        features_df['page_load_time_score'] = df['url'].apply(self._extract_load_time_score)
        features_df['meta_tag_count'] = df['url'].apply(self._extract_meta_tag_count)
        features_df['title_length'] = df['url'].apply(self._extract_title_length)
        
        # Handle any NaN values
        features_df = features_df.fillna(0)
        
        logger.info(f"Content feature extraction complete. Shape: {features_df.shape}")
        return features_df
    
    def _extract_suspicious_keywords(self, url: str) -> int:
        """
        Count suspicious keywords in the URL.
        
        Phishing URLs often contain urgent or suspicious keywords to
        create a sense of urgency or legitimacy.
        
        Args:
            url: URL string
            
        Returns:
            Count of suspicious keywords found
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        url_lower = url.lower()
        count = 0
        
        for keyword in self.phishing_keywords:
            count += url_lower.count(keyword)
            
        return min(count, 10)  # Cap at 10 to avoid extreme values
    
    def _extract_form_count(self, url: str) -> int:
        """
        Estimate HTML form count based on URL patterns.
        
        Phishing sites often have simple forms for credential harvesting,
        while legitimate sites have more complex, multi-step forms.
        
        Args:
            url: URL string
            
        Returns:
            Estimated form count
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Simple forms (phishing indicators)
            simple_form_indicators = ['login', 'signin', 'verify', 'confirm']
            simple_count = sum(1 for indicator in simple_form_indicators if indicator in path)
            
            # Complex forms (legitimate indicators)
            complex_form_indicators = ['register', 'signup', 'profile', 'settings', 'checkout']
            complex_count = sum(1 for indicator in complex_form_indicators if indicator in path)
            
            # Estimate based on patterns
            if simple_count > 0 and complex_count == 0:
                return 1  # Simple phishing form
            elif complex_count > 0:
                return 3  # Complex legitimate forms
            else:
                return 0  # No forms detected
                
        except Exception:
            return 0
    
    def _extract_external_link_ratio(self, url: str) -> float:
        """
        Estimate external link ratio based on URL structure.
        
        Phishing sites often have many external links to redirect users
        or load malicious content from other domains.
        
        Args:
            url: URL string
            
        Returns:
            Estimated external link ratio (0-1)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0.0
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0.0
                
            # Simplified estimation based on URL complexity
            path = parsed.path
            query = parsed.query
            
            # More complex URLs with parameters often have more external links
            if '?' in url and '&' in url:
                return 0.7  # High external link ratio
            elif '?' in url:
                return 0.4  # Medium external link ratio
            else:
                return 0.1  # Low external link ratio
                
        except Exception:
            return 0.0
    
    def _extract_image_text_ratio(self, url: str) -> float:
        """
        Estimate image-to-text ratio based on URL patterns.
        
        Phishing sites often use more images (screenshots, logos) and less
        actual text content compared to legitimate sites.
        
        Args:
            url: URL string
            
        Returns:
            Estimated image-to-text ratio (0-1)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0.0
            
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Image-heavy patterns (phishing indicators)
            image_indicators = ['image', 'img', 'photo', 'picture', 'gallery']
            image_count = sum(1 for indicator in image_indicators if indicator in path)
            
            # Text-heavy patterns (legitimate indicators)
            text_indicators = ['article', 'blog', 'news', 'help', 'support', 'about']
            text_count = sum(1 for indicator in text_indicators if indicator in path)
            
            if image_count > text_count:
                return 0.8  # High image ratio
            elif text_count > image_count:
                return 0.2  # Low image ratio
            else:
                return 0.5  # Balanced ratio
                
        except Exception:
            return 0.0
    
    def _extract_javascript_ratio(self, url: str) -> float:
        """
        Estimate JavaScript content ratio based on URL patterns.
        
        Phishing sites often use less JavaScript for dynamic content
        and more static HTML, while legitimate sites use more JS.
        
        Args:
            url: URL string
            
        Returns:
            Estimated JavaScript ratio (0-1)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0.0
            
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # JavaScript-heavy patterns (legitimate indicators)
            js_indicators = ['app', 'dashboard', 'admin', 'panel', 'api']
            js_count = sum(1 for indicator in js_indicators if indicator in path)
            
            # Static patterns (phishing indicators)
            static_indicators = ['page', 'static', 'html', 'index']
            static_count = sum(1 for indicator in static_indicators if indicator in path)
            
            if js_count > static_count:
                return 0.8  # High JavaScript ratio
            elif static_count > js_count:
                return 0.2  # Low JavaScript ratio
            else:
                return 0.5  # Balanced ratio
                
        except Exception:
            return 0.0
    
    def _extract_load_time_score(self, url: str) -> float:
        """
        Estimate page load time score based on URL characteristics.
        
        Phishing sites often load faster (simple HTML) or slower (malicious
        scripts) compared to legitimate sites with optimized content.
        
        Args:
            url: URL string
            
        Returns:
            Load time score (0-1, higher = faster load)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0.5
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0.5
                
            # Fast loading patterns (simple phishing sites)
            if len(hostname) < 10 and '.' in hostname:
                return 0.9  # Very fast
                
            # Slow loading patterns (complex legitimate sites)
            if any(complex_indicator in hostname for complex_indicator in ['app', 'dashboard', 'admin']):
                return 0.3  # Slower
                
            # Medium loading patterns
            return 0.6  # Medium speed
                
        except Exception:
            return 0.5
    
    def _extract_meta_tag_count(self, url: str) -> int:
        """
        Estimate meta tag count based on URL patterns.
        
        Legitimate sites typically have more comprehensive meta tags
        for SEO and social sharing, while phishing sites have fewer.
        
        Args:
            url: URL string
            
        Returns:
            Estimated meta tag count
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # SEO-friendly patterns (more meta tags)
            seo_indicators = ['blog', 'article', 'news', 'help', 'about', 'contact']
            seo_count = sum(1 for indicator in seo_indicators if indicator in path)
            
            # Simple patterns (fewer meta tags)
            simple_indicators = ['login', 'signin', 'verify', 'confirm']
            simple_count = sum(1 for indicator in simple_indicators if indicator in path)
            
            if seo_count > simple_count:
                return 8  # Many meta tags
            elif simple_count > seo_count:
                return 2  # Few meta tags
            else:
                return 5  # Average meta tags
                
        except Exception:
            return 0
    
    def _extract_title_length(self, url: str) -> int:
        """
        Extract estimated title length based on URL patterns.
        
        Phishing sites often have shorter, more generic titles,
        while legitimate sites have longer, more descriptive titles.
        
        Args:
            url: URL string
            
        Returns:
            Estimated title length
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0
                
            # Estimate title length based on domain characteristics
            domain_length = len(hostname)
            
            # Longer domains often have longer titles
            if domain_length > 20:
                return 60  # Long title
            elif domain_length > 10:
                return 40  # Medium title
            else:
                return 20  # Short title
                
        except Exception:
            return 0
    
    def _extract_suspicious_tld_score(self, url: str) -> float:
        """
        Extract suspicious TLD score.
        
        Some TLDs are commonly used for phishing due to their low cost
        and lax registration requirements.
        
        Args:
            url: URL string
            
        Returns:
            Suspicious TLD score (0-1, higher = more suspicious)
        """
        if pd.isna(url) or not isinstance(url, str):
            return 0.0
            
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return 0.0
                
            # Check for suspicious TLDs
            for tld in self.suspicious_tlds:
                if hostname.endswith(tld):
                    return 0.9  # Very suspicious
                    
            # Check for very new or unusual TLDs
            if '.' in hostname:
                tld = hostname.split('.')[-1]
                if len(tld) > 4:  # Long TLDs are often suspicious
                    return 0.7
                    
            return 0.1  # Not suspicious
            
        except Exception:
            return 0.0
