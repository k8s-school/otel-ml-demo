"""
Unit tests for model training and evaluation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.otel_ml_demo.model import StellarClassifier


class TestStellarClassifier:
    """Test cases for Stellar Classifier model"""

    def test_init(self):
        """Test model initialization"""
        model = StellarClassifier(n_estimators=50, random_state=123)

        assert model.model.n_estimators == 50
        assert model.model.random_state == 123
        assert not model.is_trained
        assert model.feature_names is None
        assert model.class_names is None

    def test_init_defaults(self):
        """Test model initialization with defaults"""
        model = StellarClassifier()

        assert model.model.n_estimators == 100
        assert model.model.random_state == 42
        assert model.model.n_jobs == -1

    def test_train_basic(self, sample_data):
        """Test basic model training"""
        # Prepare data
        X = sample_data[['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']].values
        y = np.array([0, 1, 2])  # Encoded classes
        feature_names = ['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']
        class_names = ['GALAXY', 'STAR', 'QSO']

        # Train model
        model = StellarClassifier(n_estimators=5, random_state=42)
        results = model.train(X, y, feature_names, class_names)

        # Check model state
        assert model.is_trained
        assert model.feature_names == feature_names
        assert model.class_names == class_names

        # Check results
        assert isinstance(results, dict)
        assert 'train_accuracy' in results
        assert 'feature_importance' in results
        assert 'n_samples' in results
        assert results['n_samples'] == 3
        assert results['n_features'] == 8
        assert results['n_classes'] == 3

        # Check feature importance
        assert len(results['feature_importance']) == 8
        assert all(importance >= 0 for importance in results['feature_importance'].values())

    def test_train_without_metadata(self):
        """Test training without explicit feature/class names"""
        X = np.random.random((10, 5))
        y = np.random.randint(0, 3, 10)

        model = StellarClassifier(n_estimators=5)
        results = model.train(X, y)

        assert model.feature_names == [f"feature_{i}" for i in range(5)]
        assert model.class_names == [f"class_{i}" for i in range(3)]

    def test_evaluate_trained_model(self, trained_model):
        """Test model evaluation on test data"""
        model, loader = trained_model

        # Create test data (same as training for simplicity)
        X_test = np.array([[23.87882, 22.2753, 20.39501, 19.16573, 18.79371, 135.689, 32.494, 0.634794],
                          [19.43718, 17.58028, 16.49747, 15.97711, 15.54461, 345.282, 21.183, 0.000010]])
        y_test = np.array([0, 2])  # GALAXY, STAR

        results = model.evaluate(X_test, y_test)

        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        assert 'predictions' in results
        assert 'probabilities' in results

        assert 0 <= results['accuracy'] <= 1
        assert len(results['predictions']) == 2
        assert results['probabilities'].shape == (2, 3)  # 2 samples, 3 classes

    def test_evaluate_untrained_model(self):
        """Test evaluation on untrained model raises error"""
        model = StellarClassifier()
        X_test = np.random.random((5, 8))
        y_test = np.random.randint(0, 3, 5)

        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            model.evaluate(X_test, y_test)

    def test_predict_trained_model(self, trained_model):
        """Test predictions on trained model"""
        model, loader = trained_model

        X = np.array([[23.87882, 22.2753, 20.39501, 19.16573, 18.79371, 135.689, 32.494, 0.634794]])

        predictions, probabilities = model.predict(X)

        assert len(predictions) == 1
        assert probabilities.shape == (1, 3)
        assert 0 <= predictions[0] <= 2
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_untrained_model(self):
        """Test prediction on untrained model raises error"""
        model = StellarClassifier()
        X = np.random.random((1, 8))

        with pytest.raises(ValueError, match="Model must be trained before prediction"):
            model.predict(X)

    def test_predict_single(self, trained_model, sample_observation):
        """Test single observation prediction"""
        model, loader = trained_model

        # Remove objid from features for prediction
        features = {k: v for k, v in sample_observation.items() if k != 'objid'}

        result = model.predict_single(features)

        assert isinstance(result, dict)
        assert 'predicted_class' in result
        assert 'predicted_class_encoded' in result
        assert 'confidence' in result
        assert 'class_probabilities' in result

        assert result['predicted_class'] in ['GALAXY', 'STAR', 'QSO']
        assert 0 <= result['predicted_class_encoded'] <= 2
        assert 0 <= result['confidence'] <= 1
        assert len(result['class_probabilities']) == 3

    def test_predict_single_wrong_features(self, trained_model):
        """Test single prediction with wrong features"""
        model, loader = trained_model

        wrong_features = {'wrong_feature': 1.0, 'another_wrong': 2.0}

        with pytest.raises(KeyError):
            # This will fail because we don't have the right features
            model.predict_single(wrong_features)

    def test_save_and_load_model(self, trained_model, temp_dir):
        """Test model saving and loading"""
        model, loader = trained_model
        save_path = temp_dir / "test_model.joblib"

        # Save model
        model.save_model(save_path)
        assert save_path.exists()

        # Load model
        loaded_model = StellarClassifier.load_model(save_path)

        assert loaded_model.is_trained
        assert loaded_model.feature_names == model.feature_names
        assert loaded_model.class_names == model.class_names

        # Test that loaded model works
        X = np.array([[23.87882, 22.2753, 20.39501, 19.16573, 18.79371, 135.689, 32.494, 0.634794]])
        orig_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)

        np.testing.assert_array_equal(orig_pred[0], loaded_pred[0])
        np.testing.assert_array_almost_equal(orig_pred[1], loaded_pred[1])

    def test_save_untrained_model(self, temp_dir):
        """Test saving untrained model raises error"""
        model = StellarClassifier()
        save_path = temp_dir / "test_model.joblib"

        with pytest.raises(ValueError, match="Cannot save untrained model"):
            model.save_model(save_path)

    def test_load_nonexistent_model(self, temp_dir):
        """Test loading non-existent model raises error"""
        nonexistent_path = temp_dir / "nonexistent.joblib"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            StellarClassifier.load_model(nonexistent_path)

    def test_get_feature_importance(self, trained_model):
        """Test feature importance retrieval"""
        model, loader = trained_model

        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 8  # Number of features
        assert all(isinstance(v, (float, np.floating)) for v in importance.values())
        assert all(v >= 0 for v in importance.values())
        assert sum(importance.values()) == pytest.approx(1.0, rel=1e-10)

    def test_get_feature_importance_untrained(self):
        """Test feature importance on untrained model"""
        model = StellarClassifier()

        with pytest.raises(ValueError, match="Model must be trained to get feature importance"):
            model.get_feature_importance()

    def test_model_reproducibility(self, sample_data):
        """Test that model training is reproducible with same random state"""
        X = sample_data[['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']].values
        y = np.array([0, 1, 2])
        feature_names = ['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']
        class_names = ['GALAXY', 'STAR', 'QSO']

        # Train two models with same random state
        model1 = StellarClassifier(n_estimators=5, random_state=42)
        model2 = StellarClassifier(n_estimators=5, random_state=42)

        model1.train(X, y, feature_names, class_names)
        model2.train(X, y, feature_names, class_names)

        # Predictions should be identical
        test_X = X[:1]  # First sample
        pred1, prob1 = model1.predict(test_X)
        pred2, prob2 = model2.predict(test_X)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(prob1, prob2)

    def test_model_different_random_states(self, sample_data):
        """Test that models with different random states produce different results"""
        X = sample_data[['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']].values
        y = np.array([0, 1, 2])
        feature_names = ['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']
        class_names = ['GALAXY', 'STAR', 'QSO']

        # Train models with different random states
        model1 = StellarClassifier(n_estimators=10, random_state=42)
        model2 = StellarClassifier(n_estimators=10, random_state=123)

        model1.train(X, y, feature_names, class_names)
        model2.train(X, y, feature_names, class_names)

        # Feature importance should be different (with high probability)
        importance1 = model1.get_feature_importance()
        importance2 = model2.get_feature_importance()

        # At least one feature importance should be different
        differences = [abs(importance1[k] - importance2[k]) for k in importance1.keys()]
        assert any(diff > 1e-10 for diff in differences)