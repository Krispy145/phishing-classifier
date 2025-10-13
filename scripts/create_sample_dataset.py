#!/usr/bin/env python3
"""
Create a sample dataset for testing the phishing classifier.
This generates a balanced dataset with legitimate and suspicious URLs.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_dataset():
    """Create a sample dataset for testing purposes."""
    
    # Create data directory structure
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    
    # Sample URLs (mix of legitimate and suspicious)
    legitimate_urls = [
        "https://www.google.com",
        "https://www.bank.com",
        "https://www.amazon.com",
        "https://www.paypal.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.linkedin.com",
        "https://www.github.com",
        "https://www.netflix.com",
        "https://www.spotify.com",
        "https://www.youtube.com",
        "https://www.instagram.com",
        "https://www.whatsapp.com",
        "https://www.zoom.us",
        "https://www.slack.com",
        "https://www.dropbox.com",
        "https://www.adobe.com",
        "https://www.oracle.com"
    ]
    
    suspicious_urls = [
        "https://suspicious-site.tk",
        "https://phishing-attempt.com",
        "https://urgent-verify-account.com",
        "https://congratulations-winner.com",
        "https://account-suspended.tk",
        "https://security-alert.ml",
        "https://verify-now.ga",
        "https://click-here-immediately.com",
        "https://limited-time-offer.tk",
        "https://exclusive-deal.ml",
        "https://urgent-action-required.com",
        "https://verify-your-account.tk",
        "https://security-breach-alert.ml",
        "https://account-locked-immediately.ga",
        "https://click-here-to-claim.tk",
        "https://congratulations-you-won.ml",
        "https://urgent-verification-needed.ga",
        "https://security-update-required.tk",
        "https://account-verification-pending.ml",
        "https://click-here-for-prize.ga"
    ]
    
    # Create balanced dataset
    all_urls = legitimate_urls + suspicious_urls
    labels = [0] * len(legitimate_urls) + [1] * len(suspicious_urls)
    
    # Expand to 1000 samples
    expanded_urls = []
    expanded_labels = []
    
    for i in range(1000):
        url = all_urls[i % len(all_urls)]
        label = labels[i % len(labels)]
        
        # Add some variation to URLs
        if i > len(all_urls):
            url = url.replace('.com', f'-{i}.com')
            url = url.replace('.tk', f'-{i}.tk')
            url = url.replace('.ml', f'-{i}.ml')
            url = url.replace('.ga', f'-{i}.ga')
        
        expanded_urls.append(url)
        expanded_labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({
        'url': expanded_urls,
        'Result': expanded_labels
    })
    
    # Save to both locations
    df.to_csv('data/raw/phishing.csv', index=False)
    df.to_csv('data/sample/phishing_sample.csv', index=False)
    
    print(f"✅ Created sample dataset with {len(df)} samples")
    print(f"   📊 Legitimate: {sum(df['Result'] == 0)} samples")
    print(f"   🚨 Phishing: {sum(df['Result'] == 1)} samples")
    print(f"   💾 Saved to: data/raw/phishing.csv")
    print(f"   💾 Sample copy: data/sample/phishing_sample.csv")
    
    return df

if __name__ == "__main__":
    create_sample_dataset()
