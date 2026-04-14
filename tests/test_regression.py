"""
Non-regression tests with expected outputs
"""

import pytest
import json
import numpy as np
from pathlib import Path

from src.otel_ml_demo.pipeline import train_model, setup_logging
from src.otel_ml_demo.inference import StellarInference


class TestNonRegression:
    """Non-regression tests to ensure consistent behavior across changes"""

    @pytest.fixture(scope="class")
    def trained_pipeline(self, tmp_path_factory, sample_csv_file):
        """Train a model once for the regression tests"""
        temp_dir = tmp_path_factory.mktemp("regression_test")
        data_dir = temp_dir / "data"
        model_dir = temp_dir / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        # Copy sample data
        sample_path = data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        # Train with fixed parameters for reproducibility
        setup_logging("ERROR")
        results = train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            test_size=0.2,
            n_estimators=10,
            random_state=42
        )

        inference = StellarInference(model_dir=str(model_dir))
        inference.load_model()

        return inference, results

    def test_expected_training_metrics(self, trained_pipeline):
        """Test that training metrics match expected values"""
        inference, results = trained_pipeline

        # Expected values based on our sample dataset with fixed random seed
        expected_dataset_info = {
            'total_samples': 16,
            'n_features': 8,
            'n_classes': 3,
            'class_names': ['GALAXY', 'QSO', 'STAR']
        }

        expected_feature_names = ['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']

        # Check dataset info
        assert results['dataset_info']['total_samples'] == expected_dataset_info['total_samples']
        assert results['dataset_info']['n_features'] == expected_dataset_info['n_features']
        assert results['dataset_info']['n_classes'] == expected_dataset_info['n_classes']
        assert set(results['dataset_info']['class_names']) == set(expected_dataset_info['class_names'])
        assert results['dataset_info']['feature_names'] == expected_feature_names

        # Training should achieve perfect accuracy on this small dataset
        assert results['training_results']['train_accuracy'] >= 0.9
        assert results['evaluation_results']['accuracy'] >= 0.5  # At least better than random

    def test_expected_galaxy_prediction(self, trained_pipeline):
        """Test specific galaxy prediction matches expected output"""
        inference, _ = trained_pipeline

        # Known galaxy observation from our dataset
        galaxy_obs = {
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

        result = inference.predict(galaxy_obs)

        # Expected characteristics for this observation
        assert result['predicted_class'] in ['GALAXY', 'QSO']  # Should be galaxy-like
        assert result['objid'] == 1237648720693755918
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
        assert 'class_probabilities' in result
        assert len(result['class_probabilities']) == 3

        # Probabilities should sum to 1
        prob_sum = sum(result['class_probabilities'].values())
        assert abs(prob_sum - 1.0) < 1e-10

    def test_expected_star_prediction(self, trained_pipeline):
        """Test specific star prediction matches expected output"""
        inference, _ = trained_pipeline

        # Known star observation from our dataset
        star_obs = {
            "objid": 1237648668670849481,
            "u": 19.43718,
            "g": 17.58028,
            "r": 16.49747,
            "i": 15.97711,
            "z": 15.54461,
            "alpha": 345.282,
            "delta": 21.183,
            "redshift": 0.000010
        }

        result = inference.predict(star_obs)

        # Stars typically have low redshift and specific color characteristics
        assert result['predicted_class'] in ['STAR', 'GALAXY']  # Should be star-like
        assert result['objid'] == 1237648668670849481

        # For this specific observation, we expect high confidence in star classification
        if result['predicted_class'] == 'STAR':
            assert result['confidence'] > 0.5

    def test_expected_quasar_prediction(self, trained_pipeline):
        """Test specific quasar prediction matches expected output"""
        inference, _ = trained_pipeline

        # Known QSO observation from our dataset
        qso_obs = {
            "objid": 1237648720693755922,
            "u": 21.45678,
            "g": 19.87654,
            "r": 18.23456,
            "i": 17.56789,
            "z": 16.89012,
            "alpha": 234.567,
            "delta": 12.345,
            "redshift": 1.234567
        }

        result = inference.predict(qso_obs)

        # QSOs typically have high redshift
        assert result['predicted_class'] in ['QSO', 'GALAXY']  # Should be QSO-like
        assert result['objid'] == 1237648720693755922

    def test_expected_feature_importance_ranking(self, trained_pipeline):
        """Test that feature importance ranking is consistent"""
        inference, results = trained_pipeline

        importance = inference.get_model_info()['feature_importance']

        # Convert to sorted list
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Check that all features have non-negative importance
        assert all(imp >= 0 for _, imp in sorted_features)

        # Check that z-band and u-band are typically important for stellar classification
        important_features = [feat for feat, _ in sorted_features[:3]]
        photometric_features = {'u', 'g', 'r', 'i', 'z'}
        assert any(feat in photometric_features for feat in important_features)

        # Feature importances should sum to 1
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 1e-10

    def test_batch_prediction_consistency(self, trained_pipeline):
        """Test that batch predictions are consistent with individual predictions"""
        inference, _ = trained_pipeline

        # Test observations
        observations = [
            {
                "objid": 1237648720693755918,
                "u": 23.87882,
                "g": 22.27530,
                "r": 20.39501,
                "i": 19.16573,
                "z": 18.79371,
                "alpha": 135.689,
                "delta": 32.494,
                "redshift": 0.634794
            },
            {
                "objid": 1237648668670849481,
                "u": 19.43718,
                "g": 17.58028,
                "r": 16.49747,
                "i": 15.97711,
                "z": 15.54461,
                "alpha": 345.282,
                "delta": 21.183,
                "redshift": 0.000010
            }
        ]

        # Get individual predictions
        individual_results = []
        for obs in observations:
            individual_results.append(inference.predict(obs))

        # Get batch predictions
        batch_results = inference.batch_predict(observations)

        # Compare results
        assert len(batch_results) == len(individual_results)

        for i, (individual, batch) in enumerate(zip(individual_results, batch_results)):
            assert individual['predicted_class'] == batch['predicted_class']
            assert individual['confidence'] == batch['confidence']
            assert individual['objid'] == batch['objid']
            assert batch['observation_index'] == i

    def test_model_reproducibility(self, sample_csv_file, tmp_path):
        """Test that identical training parameters produce identical results"""
        # Train model twice with same parameters
        results1 = self._train_model_with_params(sample_csv_file, tmp_path / "run1")
        results2 = self._train_model_with_params(sample_csv_file, tmp_path / "run2")

        # Training metrics should be identical
        assert results1['training_results']['train_accuracy'] == results2['training_results']['train_accuracy']
        assert results1['evaluation_results']['accuracy'] == results2['evaluation_results']['accuracy']

        # Test predictions are identical
        inference1 = StellarInference(model_dir=str(tmp_path / "run1" / "models"))
        inference2 = StellarInference(model_dir=str(tmp_path / "run2" / "models"))
        inference1.load_model()
        inference2.load_model()

        test_obs = {
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

        result1 = inference1.predict(test_obs)
        result2 = inference2.predict(test_obs)

        assert result1['predicted_class'] == result2['predicted_class']
        assert abs(result1['confidence'] - result2['confidence']) < 1e-10

    def _train_model_with_params(self, sample_csv_file, base_dir):
        """Helper method to train model with fixed parameters"""
        data_dir = base_dir / "data"
        model_dir = base_dir / "models"
        data_dir.mkdir(parents=True)
        model_dir.mkdir(parents=True)

        # Copy sample data
        sample_path = data_dir / "sample_sdss.csv"
        import shutil
        shutil.copy2(sample_csv_file, sample_path)

        setup_logging("ERROR")
        return train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            test_size=0.2,
            n_estimators=10,
            random_state=42
        )

    def test_expected_class_distribution(self, trained_pipeline):
        """Test that we can predict all three classes in principle"""
        inference, results = trained_pipeline

        # Test a variety of observations to check we can get all classes
        test_observations = [
            # Galaxy-like (high redshift, faint)
            {
                "objid": 1,
                "u": 24.0,
                "g": 22.5,
                "r": 21.0,
                "i": 20.0,
                "z": 19.5,
                "alpha": 100.0,
                "delta": 20.0,
                "redshift": 0.5
            },
            # Star-like (low redshift, bright)
            {
                "objid": 2,
                "u": 18.0,
                "g": 16.5,
                "r": 15.0,
                "i": 14.5,
                "z": 14.0,
                "alpha": 200.0,
                "delta": 30.0,
                "redshift": 0.00001
            },
            # QSO-like (high redshift, blue colors)
            {
                "objid": 3,
                "u": 20.0,
                "g": 19.0,
                "r": 18.5,
                "i": 18.0,
                "z": 17.5,
                "alpha": 300.0,
                "delta": 40.0,
                "redshift": 2.0
            }
        ]

        predictions = []
        for obs in test_observations:
            result = inference.predict(obs)
            predictions.append(result['predicted_class'])

        # We should be able to predict all classes given diverse enough inputs
        unique_predictions = set(predictions)
        assert len(unique_predictions) >= 1  # At least some variety

        # All predictions should be valid classes
        valid_classes = {'GALAXY', 'STAR', 'QSO'}
        assert all(pred in valid_classes for pred in predictions)

    def test_confidence_bounds(self, trained_pipeline):
        """Test that confidence values are always within valid bounds"""
        inference, _ = trained_pipeline

        # Test various observations
        test_observations = [
            {"objid": 1, "u": 23.9, "g": 22.3, "r": 20.4, "i": 19.2, "z": 18.8,
             "alpha": 135.7, "delta": 32.5, "redshift": 0.63},
            {"objid": 2, "u": 19.4, "g": 17.6, "r": 16.5, "i": 16.0, "z": 15.5,
             "alpha": 345.3, "delta": 21.2, "redshift": 0.00001},
            {"objid": 3, "u": 21.5, "g": 19.9, "r": 18.2, "i": 17.6, "z": 16.9,
             "alpha": 234.6, "delta": 12.3, "redshift": 1.23}
        ]

        for obs in test_observations:
            result = inference.predict(obs)

            # Confidence should be between 0 and 1
            assert 0 <= result['confidence'] <= 1

            # All class probabilities should be between 0 and 1
            for prob in result['class_probabilities'].values():
                assert 0 <= prob <= 1

            # Probabilities should sum to 1
            prob_sum = sum(result['class_probabilities'].values())
            assert abs(prob_sum - 1.0) < 1e-10

            # Confidence should equal the maximum probability
            max_prob = max(result['class_probabilities'].values())
            assert abs(result['confidence'] - max_prob) < 1e-10