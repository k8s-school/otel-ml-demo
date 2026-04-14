"""
Unit tests for data loading and preprocessing
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import requests

from src.otel_ml_demo.data_loader import SDSSDataLoader


class TestSDSSDataLoader:
    """Test cases for SDSS data loader"""

    def test_init(self, temp_dir):
        """Test data loader initialization"""
        loader = SDSSDataLoader(data_dir=str(temp_dir))

        assert loader.data_dir == temp_dir
        assert loader.data_dir.exists()
        assert isinstance(loader.feature_columns, list)
        assert isinstance(loader.columns_to_drop, list)
        assert 'u' in loader.feature_columns
        assert 'spec_obj_ID' in loader.columns_to_drop

    def test_load_raw_data_from_sample(self, data_loader, sample_csv_file):
        """Test loading data from sample CSV"""
        # Create sample file in data dir
        sample_path = data_loader.data_dir / "sample_sdss.csv"
        sample_csv_file.rename(sample_path)

        df = data_loader.load_raw_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'class' in df.columns
        assert 'objid' in df.columns
        assert set(df['class'].unique()) == {'GALAXY', 'STAR', 'QSO'}

    @patch('requests.get')
    def test_download_dataset_success(self, mock_get, data_loader):
        """Test successful dataset download"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"test,data\n1,2"
        mock_get.return_value = mock_response

        data_loader.download_dataset()

        assert data_loader.dataset_path.exists()
        mock_get.assert_called_once_with(data_loader.dataset_url, timeout=30)

    @patch('requests.get')
    def test_download_dataset_failure(self, mock_get, data_loader):
        """Test dataset download failure"""
        # Mock failed response
        mock_get.side_effect = requests.RequestException("Network error")

        with pytest.raises(requests.RequestException):
            data_loader.download_dataset()

    def test_preprocess_data_basic(self, data_loader, sample_data):
        """Test basic data preprocessing"""
        X, y, objids = data_loader.preprocess_data(sample_data, fit_encoder=True)

        # Check shapes and types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(objids, pd.Series)

        # Check dimensions
        assert X.shape[0] == 3  # 3 samples
        assert X.shape[1] == 8  # 8 features
        assert len(y) == 3
        assert len(objids) == 3

        # Check feature columns
        expected_features = ['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']
        assert list(X.columns) == expected_features

        # Check target encoding
        assert set(y.unique()) == {0, 1, 2}  # Encoded classes
        assert data_loader.get_class_names() == ['GALAXY', 'QSO', 'STAR']

    def test_preprocess_data_drops_columns(self, data_loader, sample_data):
        """Test that irrelevant columns are dropped"""
        original_columns = set(sample_data.columns)
        X, y, objids = data_loader.preprocess_data(sample_data)

        # Ensure dropped columns are not in features
        for col in data_loader.columns_to_drop:
            if col in original_columns:
                assert col not in X.columns

    def test_preprocess_data_missing_class(self, data_loader, sample_data):
        """Test preprocessing data without class column (inference mode)"""
        # Remove class column
        data_no_class = sample_data.drop(columns=['class'])

        X, y, objids = data_loader.preprocess_data(data_no_class, fit_encoder=False)

        assert isinstance(X, pd.DataFrame)
        assert y is None
        assert len(objids) == 3

    def test_preprocess_data_missing_features(self, data_loader, sample_data):
        """Test preprocessing with missing feature columns"""
        # Remove a required feature
        data_missing_feature = sample_data.drop(columns=['u'])

        X, y, objids = data_loader.preprocess_data(data_missing_feature)

        # Should only include available features
        assert 'u' not in X.columns
        assert X.shape[1] == 7  # One less feature

    def test_get_class_name(self, data_loader, sample_data):
        """Test class name decoding"""
        # Fit encoder first
        data_loader.preprocess_data(sample_data, fit_encoder=True)

        assert data_loader.get_class_name(0) == 'GALAXY'
        assert data_loader.get_class_name(1) == 'QSO'
        assert data_loader.get_class_name(2) == 'STAR'

    def test_get_class_name_not_fitted(self, data_loader):
        """Test class name retrieval without fitted encoder"""
        with pytest.raises(ValueError, match="Label encoder not fitted yet"):
            data_loader.get_class_name(0)

    def test_get_class_names(self, data_loader, sample_data):
        """Test getting all class names"""
        data_loader.preprocess_data(sample_data, fit_encoder=True)

        class_names = data_loader.get_class_names()
        assert isinstance(class_names, list)
        assert set(class_names) == {'GALAXY', 'QSO', 'STAR'}

    def test_preprocess_data_with_missing_values(self, data_loader, sample_data):
        """Test preprocessing with missing values"""
        # Introduce missing values
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.loc[0, 'u'] = np.nan
        sample_data_with_nan.loc[1, 'redshift'] = np.nan

        X, y, objids = data_loader.preprocess_data(sample_data_with_nan)

        # Should have no missing values after preprocessing
        assert not X.isnull().any().any()

    def test_feature_columns_consistency(self, data_loader):
        """Test that feature columns are consistently defined"""
        expected_features = ['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']
        assert data_loader.feature_columns == expected_features

    def test_columns_to_drop_consistency(self, data_loader):
        """Test that columns to drop are consistently defined"""
        expected_drops = [
            'spec_obj_ID', 'run_ID', 'rerun_ID', 'cam_col',
            'field_ID', 'plate', 'mjd', 'fiber_ID'
        ]
        assert data_loader.columns_to_drop == expected_drops