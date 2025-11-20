#!/usr/bin/env python3
"""
Sync Model to Secure AI API

This script copies the trained model from phishing-classifier to secure-ai-api
for API integration. It ensures the model is available for the FastAPI service.
"""

import shutil
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sync_model_to_api():
    """
    Copy the trained model from phishing-classifier to secure-ai-api.
    
    This script assumes both repositories are in the same parent directory:
    - ai-journey/phishing-classifier/
    - ai-journey/secure-ai-api/
    """
    # Get the script directory and navigate to project root
    script_dir = Path(__file__).parent
    phishing_classifier_root = script_dir.parent
    
    # Paths
    source_model = phishing_classifier_root / "app" / "models" / "model.joblib"
    target_dir = phishing_classifier_root.parent / "secure-ai-api" / "app" / "models"
    target_model = target_dir / "model.joblib"
    
    # Check if source model exists
    if not source_model.exists():
        logger.error(f"Source model not found: {source_model}")
        logger.error("Please train the model first by running: python src/pipeline.py")
        return False
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Copy the model
        logger.info(f"Copying model from: {source_model}")
        logger.info(f"To: {target_model}")
        shutil.copy2(source_model, target_model)
        
        # Verify the copy
        if target_model.exists():
            source_size = source_model.stat().st_size
            target_size = target_model.stat().st_size
            if source_size == target_size:
                logger.info(f"✅ Model synced successfully! ({source_size:,} bytes)")
                logger.info(f"   Model is now available at: {target_model}")
                return True
            else:
                logger.error(f"File size mismatch: source={source_size}, target={target_size}")
                return False
        else:
            logger.error("Model copy failed - target file not found")
            return False
            
    except Exception as e:
        logger.error(f"Error syncing model: {e}")
        return False


def main():
    """Main entry point."""
    logger.info("🔄 Syncing phishing classifier model to Secure AI API...")
    
    success = sync_model_to_api()
    
    if success:
        logger.info("✅ Model sync completed successfully!")
        logger.info("   The Secure AI API can now use the trained model.")
        logger.info("   Restart the API server to load the new model.")
        return 0
    else:
        logger.error("❌ Model sync failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

