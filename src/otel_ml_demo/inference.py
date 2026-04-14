"""
Inference script for SDSS stellar classification
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Union
import argparse

import joblib
import numpy as np

from .model import StellarClassifier


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class StellarInference:
    """Handles inference for stellar classification"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessing_data = None
        self.logger = logging.getLogger(__name__)

    def load_model(self) -> None:
        """Load trained model and preprocessing metadata"""
        model_path = self.model_dir / "stellar_classifier.joblib"
        preprocessing_path = self.model_dir / "data_preprocessing.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not preprocessing_path.exists():
            raise FileNotFoundError(f"Preprocessing file not found: {preprocessing_path}")

        # Load model
        self.model = StellarClassifier.load_model(model_path)
        self.logger.info("Model loaded successfully")

        # Load preprocessing metadata
        self.preprocessing_data = joblib.load(preprocessing_path)
        self.logger.info("Preprocessing metadata loaded successfully")

    def predict(self, observation: Dict[str, Union[float, int]], include_objid: bool = True) -> Dict[str, Any]:
        """
        Predict stellar class for a single observation

        Args:
            observation: Dictionary with photometric features and optional objid
            include_objid: Whether to include objid in response

        Returns:
            Prediction results with class, confidence, and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Extract features in correct order
        feature_columns = self.preprocessing_data["feature_columns"]
        features = {}

        # Map input observation to expected feature names
        for feature_name in feature_columns:
            if feature_name in observation:
                features[feature_name] = float(observation[feature_name])
            else:
                raise ValueError(f"Missing required feature: {feature_name}")

        self.logger.info(f"Making prediction for observation with features: {list(features.keys())}")

        # Make prediction
        prediction_result = self.model.predict_single(features)

        # Add objid if provided and requested
        if include_objid and 'objid' in observation:
            prediction_result['objid'] = observation['objid']

        # Add input features for reference
        prediction_result['input_features'] = features

        return prediction_result

    def batch_predict(self, observations: list) -> list:
        """
        Predict stellar classes for multiple observations

        Args:
            observations: List of observation dictionaries

        Returns:
            List of prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.logger.info(f"Making predictions for {len(observations)} observations")

        results = []
        for i, obs in enumerate(observations):
            try:
                result = self.predict(obs)
                result['observation_index'] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to predict observation {i}: {e}")
                results.append({
                    'observation_index': i,
                    'error': str(e),
                    'objid': obs.get('objid', None)
                })

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        feature_importance = self.model.get_feature_importance()

        return {
            "feature_names": self.model.feature_names,
            "class_names": self.model.class_names,
            "feature_importance": feature_importance,
            "top_features": sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        }


def predict_from_json(json_input: str, model_dir: str = "models") -> Dict[str, Any]:
    """
    Predict from JSON input (single observation or list)

    Args:
        json_input: JSON string with observation data
        model_dir: Directory containing trained model

    Returns:
        Prediction results
    """
    # Parse JSON input
    try:
        data = json.loads(json_input)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

    # Initialize inference
    inference = StellarInference(model_dir)
    inference.load_model()

    # Handle single observation vs batch
    if isinstance(data, dict):
        # Single observation
        return inference.predict(data)
    elif isinstance(data, list):
        # Batch of observations
        return {
            "batch_predictions": inference.batch_predict(data),
            "total_observations": len(data)
        }
    else:
        raise ValueError("Input must be a JSON object or array of objects")


def main():
    """Main entry point for inference"""
    parser = argparse.ArgumentParser(description="SDSS stellar classification inference")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--input", help="JSON input file or string")
    parser.add_argument("--output", help="Output file (optional, prints to stdout if not provided)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    # Example usage arguments
    parser.add_argument("--example", action="store_true", help="Show example usage")
    parser.add_argument("--model-info", action="store_true", help="Show model information")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Show example usage
        if args.example:
            example_observation = {
                "objid": 1237648720693755918,
                "u": 23.87882,
                "g": 22.27530,
                "r": 20.39501,
                "i": 19.16573,
                "z": 18.79371,
                "alpha": 135.689,
                "delta": 32.494,
                "redshift": 0.634794
            }

            print("Example usage:")
            print("Single observation:")
            print(json.dumps(example_observation, indent=2))
            print("\nBatch prediction:")
            print(json.dumps([example_observation], indent=2))
            return

        # Show model info
        if args.model_info:
            inference = StellarInference(args.model_dir)
            inference.load_model()
            info = inference.get_model_info()

            print("Model Information:")
            print("="*50)
            print(f"Classes: {', '.join(info['class_names'])}")
            print(f"Features: {', '.join(info['feature_names'])}")
            print("\nTop 5 Most Important Features:")
            for i, (feature, importance) in enumerate(info['top_features'], 1):
                print(f"{i}. {feature}: {importance:.4f}")
            return

        # Require input for actual prediction
        if not args.input:
            parser.error("--input is required for prediction (or use --example/--model-info)")

        # Read input
        if Path(args.input).exists():
            # Input is a file
            with open(args.input, 'r') as f:
                json_input = f.read()
            logger.info(f"Reading input from file: {args.input}")
        else:
            # Input is a JSON string
            json_input = args.input
            logger.info("Using command-line JSON input")

        # Make prediction
        result = predict_from_json(json_input, args.model_dir)

        # Output result
        output_json = json.dumps(result, indent=2)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
            logger.info(f"Results saved to: {args.output}")
        else:
            print(output_json)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()