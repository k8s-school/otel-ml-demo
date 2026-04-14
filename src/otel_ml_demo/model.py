"""
Model definition and utilities for SDSS stellar classification
"""

import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


logger = logging.getLogger(__name__)


class StellarClassifier:
    """Random Forest classifier for stellar object classification"""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = None
        self.class_names = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list = None, class_names: list = None) -> Dict[str, Any]:
        """
        Train the Random Forest model

        Args:
            X_train: Training features
            y_train: Training targets (encoded)
            feature_names: List of feature names
            class_names: List of class names

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training Random Forest with {X_train.shape[0]} samples")
        logger.info(f"Feature shape: {X_train.shape[1]} features")

        # Store metadata
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self.class_names = class_names or [f"class_{i}" for i in range(len(np.unique(y_train)))]

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Get feature importances
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

        # Training accuracy
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)

        logger.info(f"Training completed. Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Top 3 important features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")

        return {
            "train_accuracy": train_accuracy,
            "feature_importance": feature_importance,
            "n_samples": X_train.shape[0],
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train))
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data

        Args:
            X_test: Test features
            y_test: Test targets (encoded)

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        logger.info(f"Evaluating model on {X_test.shape[0]} test samples")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        conf_matrix = confusion_matrix(y_test, y_pred)

        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred, target_names=self.class_names)}")

        return {
            "accuracy": accuracy,
            "classification_report": classification_rep,
            "confusion_matrix": conf_matrix.tolist(),
            "predictions": y_pred,
            "probabilities": y_pred_proba
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data

        Args:
            X: Features for prediction

        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return predictions, probabilities

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict class for a single observation

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Convert features dict to array in correct order
        feature_array = np.array([features[name] for name in self.feature_names]).reshape(1, -1)

        # Predict
        prediction = self.model.predict(feature_array)[0]
        probabilities = self.model.predict_proba(feature_array)[0]

        # Format results
        class_probabilities = {
            class_name: prob for class_name, prob
            in zip(self.class_names, probabilities)
        }

        return {
            "predicted_class": self.class_names[prediction],
            "predicted_class_encoded": int(prediction),
            "confidence": float(probabilities[prediction]),
            "class_probabilities": class_probabilities
        }

    def save_model(self, filepath: Path) -> None:
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "is_trained": self.is_trained
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Path) -> "StellarClassifier":
        """Load trained model from disk"""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        # Create instance
        instance = cls()
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.class_names = model_data["class_names"]
        instance.is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        return dict(zip(self.feature_names, self.model.feature_importances_))