"""
Training pipeline for SDSS stellar classification
"""

import logging
from pathlib import Path
from typing import Dict, Any
import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from .data_loader import SDSSDataLoader
from .model import StellarClassifier


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file_path = Path("logs") / log_file
        log_file_path.parent.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def train_model(
    data_dir: str = "data",
    model_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100
) -> Dict[str, Any]:
    """
    Train stellar classification model

    Args:
        data_dir: Directory containing data
        model_dir: Directory to save model
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        n_estimators: Number of trees in random forest

    Returns:
        Dictionary with training results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting SDSS stellar classification training pipeline")

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data_loader = SDSSDataLoader(data_dir=data_dir)
    df = data_loader.load_raw_data()
    X, y, objids = data_loader.preprocess_data(df)

    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split data
    logger.info(f"Splitting data: {1-test_size:.1%} train, {test_size:.1%} test")

    # Check if we have enough samples per class for stratification
    unique, counts = np.unique(y, return_counts=True)
    min_samples_per_class = min(counts)

    if min_samples_per_class < 2:
        logger.warning("Not enough samples per class for stratification, using random split")
        stratify_arg = None
    else:
        stratify_arg = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Train model
    logger.info("Training Random Forest model...")
    model = StellarClassifier(n_estimators=n_estimators, random_state=random_state)

    train_results = model.train(
        X_train.values, y_train.values,
        feature_names=list(X_train.columns),
        class_names=data_loader.get_class_names()
    )

    # Evaluate model
    logger.info("Evaluating model on test set...")
    eval_results = model.evaluate(X_test.values, y_test.values)

    # Save model
    model_path = Path(model_dir) / "stellar_classifier.joblib"
    model.save_model(model_path)

    # Save preprocessing metadata
    import joblib
    preprocessing_path = Path(model_dir) / "data_preprocessing.joblib"
    preprocessing_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessing_data = {
        "label_encoder": data_loader.label_encoder,
        "feature_columns": data_loader.feature_columns,
        "columns_to_drop": data_loader.columns_to_drop
    }
    joblib.dump(preprocessing_data, preprocessing_path)
    logger.info(f"Preprocessing metadata saved to {preprocessing_path}")

    # Compile results
    results = {
        "dataset_info": {
            "total_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(data_loader.get_class_names()),
            "class_names": data_loader.get_class_names(),
            "feature_names": list(X.columns)
        },
        "data_split": {
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "test_size": test_size
        },
        "model_config": {
            "n_estimators": n_estimators,
            "random_state": random_state
        },
        "training_results": train_results,
        "evaluation_results": eval_results,
        "model_path": str(model_path),
        "preprocessing_path": str(preprocessing_path)
    }

    logger.info("Training pipeline completed successfully!")
    logger.info(f"Final test accuracy: {eval_results['accuracy']:.4f}")

    return results


def main():
    """Main entry point for training pipeline"""
    parser = argparse.ArgumentParser(description="Train SDSS stellar classification model")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--model-dir", default="models", help="Model output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of RF estimators")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", help="Log file name (optional)")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    try:
        # Train model
        results = train_model(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators
        )

        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Dataset: {results['dataset_info']['total_samples']} samples, {results['dataset_info']['n_features']} features")
        print(f"Classes: {', '.join(results['dataset_info']['class_names'])}")
        print(f"Model: Random Forest ({results['model_config']['n_estimators']} estimators)")
        print(f"Test Accuracy: {results['evaluation_results']['accuracy']:.4f}")
        print(f"Model saved to: {results['model_path']}")
        print("="*60)

    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()