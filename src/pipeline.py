# Complete ML pipeline with feature engineering
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml

from src.data.load import load_raw
from src.data.preprocess import split_xy, make_splits
from src.features.pipeline import create_feature_pipeline
from src.models.train import train_and_save
from src.models.evaluate import evaluate, print_evaluation_summary

# Load configuration
with open("config/config.yaml") as f:
    CFG = yaml.safe_load(f)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main pipeline execution function.
    
    This function orchestrates the complete ML pipeline including:
    1. Data loading
    2. Feature engineering (20 features)
    3. Model training
    4. Model evaluation
    """
    parser = argparse.ArgumentParser(description="Phishing Classifier ML Pipeline")
    parser.add_argument("--stage", choices=["all", "data", "features", "train", "eval"], 
                       default="all", help="Pipeline stage to execute")
    parser.add_argument("--save-features", action="store_true", 
                       help="Save engineered features to disk")
    parser.add_argument("--load-features", type=str, 
                       help="Load pre-engineered features from file")
    args = parser.parse_args()

    logger.info(f"Starting pipeline execution - Stage: {args.stage}")

    # Initialize variables
    X_train = X_test = y_train = y_test = None
    feature_pipeline = None

    try:
        # Stage 1: Data Loading
        if args.stage in ("all", "data", "features", "train", "eval"):
            logger.info("Loading raw data...")
            df = load_raw()
            logger.info(f"Loaded {len(df)} samples")
            
            # Split features and target
            X, y = split_xy(df)
            logger.info(f"Split data: X shape {X.shape}, y shape {y.shape}")

        # Stage 2: Feature Engineering
        if args.stage in ("all", "features", "train", "eval"):
            if args.load_features:
                logger.info(f"Loading pre-engineered features from {args.load_features}")
                # Load pre-engineered features
                features_df = pd.read_csv(args.load_features)
                X = features_df.values
            else:
                logger.info("Engineering features...")
                
                # Create feature pipeline
                feature_pipeline = create_feature_pipeline()
                
                # Convert X back to DataFrame for feature engineering
                # (This assumes the original data had column names)
                df_for_features = df.copy()
                if 'Result' in df_for_features.columns:
                    df_for_features = df_for_features.drop(columns=['Result'])
                
                # Add URL column if not present (assuming it's the first column)
                if 'url' not in df_for_features.columns and len(df_for_features.columns) > 0:
                    df_for_features['url'] = df_for_features.iloc[:, 0]
                
                # Engineer features
                engineered_df = feature_pipeline.fit_transform(df_for_features)
                
                # Extract only the engineered features
                feature_columns = feature_pipeline.get_feature_names()
                X = engineered_df[feature_columns].values
                
                logger.info(f"Feature engineering complete. X shape: {X.shape}")
                logger.info(f"Features: {feature_columns}")
                
                # Save features if requested
                if args.save_features:
                    features_path = Path("data/processed/engineered_features.csv")
                    features_path.parent.mkdir(parents=True, exist_ok=True)
                    engineered_df.to_csv(features_path, index=False)
                    logger.info(f"Features saved to {features_path}")

        # Stage 3: Train-Test Split
        if args.stage in ("all", "train", "eval"):
            logger.info("Creating train-test split...")
            X_train, X_test, y_train, y_test = make_splits(X, y)
            logger.info(f"Split complete - Train: {X_train.shape}, Test: {X_test.shape}")

        # Stage 4: Model Training
        if args.stage in ("all", "train"):
            logger.info("Training model...")
            train_and_save(X_train, y_train)
            logger.info("Model training complete")

        # Stage 5: Model Evaluation
        if args.stage in ("all", "eval"):
            logger.info("Evaluating model...")
            model_path = Path(CFG["paths"]["models"]) / "model.joblib"
            if not model_path.exists():
                raise SystemExit("No model found. Run with --stage train first.")
            
            model = joblib.load(model_path)
            
            # Get evaluation output directory
            eval_dir = Path(CFG["paths"]["evaluations"])
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Run comprehensive evaluation
            # For eval-only stage, we need to retrain for cross-validation
            # For all stage, we already have X_train/y_train
            results = evaluate(
                model, 
                X_test, 
                y_test,
                X_train=X_train,
                y_train=y_train,
                save_dir=eval_dir,
                generate_plots=CFG.get("evaluation", {}).get("generate_plots", True)
            )
            
            # Print formatted summary
            print_evaluation_summary(results)
            
            logger.info("Model evaluation complete")

        logger.info("Pipeline execution completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
