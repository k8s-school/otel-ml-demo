"""
Integration tests for complete ML pipeline
"""

import pytest
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.otel_ml_demo.pipeline import train_model, setup_logging
from src.otel_ml_demo.inference import StellarInference, predict_from_json


class TestPipelineIntegration:
    """Integration tests for the complete ML pipeline"""

    def test_complete_pipeline_flow(self, temp_dir, sample_csv_file):
        """Test complete pipeline: data loading -> training -> inference"""
        # Setup directories
        data_dir = temp_dir / "data"
        model_dir = temp_dir / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        # Copy sample data
        sample_path = data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        # Step 1: Train model
        setup_logging("ERROR")  # Minimize log output during tests
        results = train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            test_size=0.2,
            n_estimators=5,
            random_state=42
        )

        # Verify training results
        assert isinstance(results, dict)
        assert 'evaluation_results' in results
        assert results['evaluation_results']['accuracy'] > 0
        assert Path(results['model_path']).exists()
        assert Path(results['preprocessing_path']).exists()

        # Step 2: Test inference
        inference = StellarInference(model_dir=str(model_dir))
        inference.load_model()

        # Test single prediction
        observation = {
            "objid": 1237648720693755918,
            "u": 23.87882,
            "g": 22.2753,
            "r": 20.39501,
            "i": 19.16573,
            "z": 18.79371,
            "alpha": 135.689,
            "delta": 32.494,
            "redshift": 0.634794
        }

        result = inference.predict(observation)

        assert 'predicted_class' in result
        assert result['predicted_class'] in ['GALAXY', 'STAR', 'QSO']
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
        assert 'objid' in result
        assert result['objid'] == 1237648720693755918

    def test_batch_inference_integration(self, temp_dir, sample_csv_file):
        """Test batch inference after training"""
        # Setup and train model (similar to above)
        data_dir = temp_dir / "data"
        model_dir = temp_dir / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        sample_path = data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        setup_logging("ERROR")
        train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            n_estimators=5,
            random_state=42
        )

        # Test batch inference
        inference = StellarInference(model_dir=str(model_dir))
        inference.load_model()

        observations = [
            {
                "objid": 1237648720693755918,
                "u": 23.87882,
                "g": 22.2753,
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

        results = inference.batch_predict(observations)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert 'predicted_class' in result
            assert 'objid' in result
            assert result['objid'] == observations[i]['objid']
            assert 'observation_index' in result
            assert result['observation_index'] == i

    def test_model_persistence_integration(self, temp_dir, sample_csv_file):
        """Test that saved models can be loaded and produce consistent results"""
        # Setup and train model
        data_dir = temp_dir / "data"
        model_dir = temp_dir / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        sample_path = data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        setup_logging("ERROR")
        train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            n_estimators=5,
            random_state=42
        )

        # Create two inference instances
        inference1 = StellarInference(model_dir=str(model_dir))
        inference2 = StellarInference(model_dir=str(model_dir))

        inference1.load_model()
        inference2.load_model()

        # Test that they produce identical results
        observation = {
            "objid": 1237648720693755918,
            "u": 23.87882,
            "g": 22.2753,
            "r": 20.39501,
            "i": 19.16573,
            "z": 18.79371,
            "alpha": 135.689,
            "delta": 32.494,
            "redshift": 0.634794
        }

        result1 = inference1.predict(observation)
        result2 = inference2.predict(observation)

        assert result1['predicted_class'] == result2['predicted_class']
        assert result1['confidence'] == result2['confidence']
        assert result1['predicted_class_encoded'] == result2['predicted_class_encoded']

    def test_json_inference_integration(self, temp_dir, sample_csv_file):
        """Test JSON-based inference functionality"""
        # Setup and train model
        data_dir = temp_dir / "data"
        model_dir = temp_dir / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        sample_path = data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        setup_logging("ERROR")
        train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            n_estimators=5,
            random_state=42
        )

        # Test single observation JSON
        single_json = json.dumps({
            "objid": 1237648720693755918,
            "u": 23.87882,
            "g": 22.2753,
            "r": 20.39501,
            "i": 19.16573,
            "z": 18.79371,
            "alpha": 135.689,
            "delta": 32.494,
            "redshift": 0.634794
        })

        result = predict_from_json(single_json, str(model_dir))
        assert 'predicted_class' in result
        assert 'objid' in result

        # Test batch JSON
        batch_json = json.dumps([
            {
                "objid": 1237648720693755918,
                "u": 23.87882,
                "g": 22.2753,
                "r": 20.39501,
                "i": 19.16573,
                "z": 18.79371,
                "alpha": 135.689,
                "delta": 32.494,
                "redshift": 0.634794
            }
        ])

        result = predict_from_json(batch_json, str(model_dir))
        assert 'batch_predictions' in result
        assert 'total_observations' in result
        assert result['total_observations'] == 1

    def test_invalid_json_inference(self, temp_dir, sample_csv_file):
        """Test error handling for invalid JSON input"""
        # Setup and train model (minimal)
        data_dir = temp_dir / "data"
        model_dir = temp_dir / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        sample_path = data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        setup_logging("ERROR")
        train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            n_estimators=5,
            random_state=42
        )

        # Test invalid JSON
        with pytest.raises(ValueError, match="Invalid JSON input"):
            predict_from_json("invalid json", str(model_dir))

        # Test invalid data structure
        with pytest.raises(ValueError, match="Input must be a JSON object or array"):
            predict_from_json('"just a string"', str(model_dir))

    def test_missing_features_error(self, temp_dir, sample_csv_file):
        """Test error handling for missing features during inference"""
        # Setup and train model
        data_dir = temp_dir / "data"
        model_dir = temp_dir / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        sample_path = data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        setup_logging("ERROR")
        train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            n_estimators=5,
            random_state=42
        )

        # Test inference with missing features
        inference = StellarInference(model_dir=str(model_dir))
        inference.load_model()

        incomplete_observation = {
            "objid": 1237648720693755918,
            "u": 23.87882,
            "g": 22.2753,
            # Missing other required features
        }

        with pytest.raises(ValueError, match="Missing required feature"):
            inference.predict(incomplete_observation)

    def test_model_info_integration(self, temp_dir, sample_csv_file):
        """Test model info retrieval after training"""
        # Setup and train model
        data_dir = temp_dir / "data"
        model_dir = temp_dir / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        sample_path = data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        setup_logging("ERROR")
        train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            n_estimators=5,
            random_state=42
        )

        # Test model info
        inference = StellarInference(model_dir=str(model_dir))
        inference.load_model()

        info = inference.get_model_info()

        assert 'feature_names' in info
        assert 'class_names' in info
        assert 'feature_importance' in info
        assert 'top_features' in info

        assert len(info['feature_names']) == 8
        assert set(info['class_names']) == {'GALAXY', 'QSO', 'STAR'}
        assert len(info['top_features']) == 5  # Top 5 features

        # Check that top features are sorted by importance
        importances = [feat[1] for feat in info['top_features']]
        assert importances == sorted(importances, reverse=True)

    def test_reproducible_pipeline(self, temp_dir, sample_csv_file):
        """Test that pipeline produces reproducible results with same random seed"""
        # Setup data
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        sample_path = data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        # Train two models with same random state
        model_dir1 = temp_dir / "models1"
        model_dir2 = temp_dir / "models2"
        model_dir1.mkdir()
        model_dir2.mkdir()

        setup_logging("ERROR")

        results1 = train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir1),
            n_estimators=5,
            random_state=42
        )

        results2 = train_model(
            data_dir=str(data_dir),
            model_dir=str(model_dir2),
            n_estimators=5,
            random_state=42
        )

        # Results should be identical (or very close due to floating point)
        assert abs(results1['evaluation_results']['accuracy'] -
                  results2['evaluation_results']['accuracy']) < 1e-10

        # Test that inference produces identical results
        inference1 = StellarInference(model_dir=str(model_dir1))
        inference2 = StellarInference(model_dir=str(model_dir2))

        inference1.load_model()
        inference2.load_model()

        observation = {
            "objid": 1237648720693755918,
            "u": 23.87882,
            "g": 22.2753,
            "r": 20.39501,
            "i": 19.16573,
            "z": 18.79371,
            "alpha": 135.689,
            "delta": 32.494,
            "redshift": 0.634794
        }

        result1 = inference1.predict(observation)
        result2 = inference2.predict(observation)

        assert result1['predicted_class'] == result2['predicted_class']
        assert abs(result1['confidence'] - result2['confidence']) < 1e-10