"""
Pytest configuration and fixtures
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from src.otel_ml_demo.data_loader import SDSSDataLoader
from src.otel_ml_demo.model import StellarClassifier


@pytest.fixture(scope="class")
def sample_data():
    """Sample SDSS data for testing"""
    data = {
        'objid': [1237648720693755918, 1237648668670849481, 1237648720693755922],
        'alpha': [135.689, 345.282, 234.567],
        'delta': [32.494, 21.183, 12.345],
        'u': [23.87882, 19.43718, 21.45678],
        'g': [22.27530, 17.58028, 19.87654],
        'r': [20.39501, 16.49747, 18.23456],
        'i': [19.16573, 15.97711, 17.56789],
        'z': [18.79371, 15.54461, 16.89012],
        'run_ID': [1345, 1345, 1345],
        'rerun_ID': [40, 40, 40],
        'cam_col': [3, 2, 3],
        'field_ID': [11, 75, 15],
        'spec_obj_ID': [1237648720693755918, 1237648668670849481, 1237648720693755922],
        'class': ['GALAXY', 'STAR', 'QSO'],
        'redshift': [0.634794, 0.000010, 1.234567],
        'plate': [752, 752, 752],
        'mjd': [52251, 52251, 52251],
        'fiber_ID': [381, 323, 385]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Temporary directory fixture"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="class")
def sample_csv_file(tmp_path_factory, sample_data):
    """Sample CSV file fixture"""
    temp_dir = tmp_path_factory.mktemp("test_data")
    csv_path = temp_dir / "sample_sdss.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def data_loader(temp_dir):
    """Data loader fixture with temp directory"""
    return SDSSDataLoader(data_dir=str(temp_dir))


@pytest.fixture
def trained_model(sample_data):
    """Trained model fixture"""
    # Prepare data
    loader = SDSSDataLoader()
    X, y, objids = loader.preprocess_data(sample_data, fit_encoder=True)

    # Train model
    model = StellarClassifier(n_estimators=5, random_state=42)
    model.train(X.values, y.values,
               feature_names=list(X.columns),
               class_names=loader.get_class_names())

    return model, loader


@pytest.fixture
def sample_observation():
    """Single observation for inference testing"""
    return {
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