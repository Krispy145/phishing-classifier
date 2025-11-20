"""
Comprehensive Model Evaluation Suite

This module provides a complete evaluation framework for the phishing classifier,
including metrics, visualizations, and report generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/seaborn not available. Visualizations will be skipped.")

logger = logging.getLogger(__name__)


def evaluate(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
    generate_plots: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with metrics, visualizations, and reports.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        X_train: Training features (optional, for feature importance)
        y_train: Training labels (optional)
        feature_names: List of feature names (optional)
        save_dir: Directory to save reports and plots
        generate_plots: Whether to generate visualization plots
        
    Returns:
        Dictionary containing all evaluation metrics and metadata
    """
    logger.info("Starting comprehensive model evaluation...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    # Get prediction probabilities if available
    if hasattr(model, "predict_proba"):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
    
    # Basic metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0, average='weighted'))
    recall = float(recall_score(y_test, y_pred, zero_division=0, average='weighted'))
    f1 = float(f1_score(y_test, y_pred, zero_division=0, average='weighted'))
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, zero_division=0, average=None)
    recall_per_class = recall_score(y_test, y_pred, zero_division=0, average=None)
    f1_per_class = f1_score(y_test, y_pred, zero_division=0, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # ROC-AUC and PR-AUC (if probabilities available)
    roc_auc = None
    pr_auc = None
    if y_pred_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_test, y_pred_proba))
            pr_auc = float(average_precision_score(y_test, y_pred_proba))
        except Exception as e:
            logger.warning(f"Could not calculate ROC/PR-AUC: {e}")
    
    # Cross-validation scores (if training data provided)
    # Note: We need to create a new model instance for CV since it will be retrained
    cv_scores = None
    if X_train is not None and y_train is not None:
        try:
            from sklearn.linear_model import LogisticRegression
            import yaml
            # Load model config to recreate model for CV
            with open("config/config.yaml") as f:
                cfg = yaml.safe_load(f)
            cv_model = LogisticRegression(**cfg["model"]["params"])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='f1_weighted')
            logger.info(f"Cross-validation F1 scores: {cv_scores}")
        except Exception as e:
            logger.warning(f"Could not perform cross-validation: {e}")
    
    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "test_set_size": len(y_test),
        "metrics": {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        },
        "per_class_metrics": {
            "precision": precision_per_class.tolist() if isinstance(precision_per_class, np.ndarray) else precision_per_class,
            "recall": recall_per_class.tolist() if isinstance(recall_per_class, np.ndarray) else recall_per_class,
            "f1": f1_per_class.tolist() if isinstance(f1_per_class, np.ndarray) else f1_per_class,
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "cross_validation": {
            "mean": float(cv_scores.mean()) if cv_scores is not None else None,
            "std": float(cv_scores.std()) if cv_scores is not None else None,
            "scores": cv_scores.tolist() if cv_scores is not None else None,
        } if cv_scores is not None else None,
    }
    
    # Generate visualizations
    if generate_plots and PLOTTING_AVAILABLE and save_dir:
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            _generate_plots(model, X_test, y_test, y_pred, y_pred_proba, cm, save_dir)
            results["plots_generated"] = True
            results["plot_dir"] = str(save_dir)
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            results["plots_generated"] = False
    else:
        results["plots_generated"] = False
    
    # Save results to JSON
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        results_path = save_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {results_path}")
    
    logger.info("Model evaluation complete")
    return results


def _generate_plots(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    cm: np.ndarray,
    save_dir: Path
):
    """Generate visualization plots for model evaluation."""
    logger.info("Generating evaluation plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve (if probabilities available)
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Plots saved to {save_dir}")


def print_evaluation_summary(results: Dict[str, Any]):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Evaluation results dictionary from evaluate()
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION SUMMARY")
    print("="*70)
    print(f"Model Type: {results['model_type']}")
    print(f"Test Set Size: {results['test_set_size']}")
    print(f"Timestamp: {results['timestamp']}")
    print("\n" + "-"*70)
    print("OVERALL METRICS")
    print("-"*70)
    
    metrics = results['metrics']
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision_weighted']:.4f}")
    print(f"Recall:         {metrics['recall_weighted']:.4f}")
    print(f"F1-Score:       {metrics['f1_weighted']:.4f}")
    
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
    if metrics['pr_auc'] is not None:
        print(f"PR-AUC:          {metrics['pr_auc']:.4f}")
    
    if results.get('cross_validation'):
        cv = results['cross_validation']
        print(f"\nCross-Validation F1: {cv['mean']:.4f} (+/- {cv['std']:.4f})")
    
    print("\n" + "-"*70)
    print("PER-CLASS METRICS")
    print("-"*70)
    per_class = results['per_class_metrics']
    classes = ['Legitimate', 'Phishing']
    for i, class_name in enumerate(classes):
        if i < len(per_class['precision']):
            print(f"\n{class_name}:")
            print(f"  Precision: {per_class['precision'][i]:.4f}")
            print(f"  Recall:    {per_class['recall'][i]:.4f}")
            print(f"  F1-Score:  {per_class['f1'][i]:.4f}")
    
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)
    cm = results['confusion_matrix']
    print(f"                Predicted")
    print(f"              Legitimate  Phishing")
    print(f"Actual Legitimate    {cm[0][0]:4d}      {cm[0][1]:4d}")
    print(f"        Phishing      {cm[1][0]:4d}      {cm[1][1]:4d}")
    
    if results.get('plots_generated'):
        print(f"\nVisualizations saved to: {results['plot_dir']}")
    
    print("="*70 + "\n")
