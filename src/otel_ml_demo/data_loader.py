"""
Data loading and preprocessing utilities for SDSS stellar classification
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder


logger = logging.getLogger(__name__)


class SDSSDataLoader:
    """Handles downloading and preprocessing of SDSS stellar classification dataset"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.dataset_url = "https://raw.githubusercontent.com/AronDJacobsen/Stellar_Classification/main/star_classification.csv"
        self.dataset_path = self.data_dir / "star_classification.csv"
        self.sample_dataset_path = self.data_dir / "sample_sdss.csv"

        # Columns to drop (irrelevant for classification)
        self.columns_to_drop = [
            'spec_obj_ID', 'run_ID', 'rerun_ID', 'cam_col',
            'field_ID', 'plate', 'mjd', 'fiber_ID'
        ]

        # Feature columns (photometric + coordinates + redshift)
        self.feature_columns = ['u', 'g', 'r', 'i', 'z', 'alpha', 'delta', 'redshift']

        self.label_encoder = LabelEncoder()

    def download_dataset(self) -> None:
        """Download SDSS dataset if not already present"""
        if self.dataset_path.exists():
            logger.info(f"Dataset already exists at {self.dataset_path}")
            return

        logger.info(f"Downloading dataset from {self.dataset_url}")
        try:
            response = requests.get(self.dataset_url, timeout=30)
            response.raise_for_status()

            with open(self.dataset_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Dataset downloaded successfully to {self.dataset_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset from CSV"""
        # Try sample dataset first, then download main dataset
        if self.sample_dataset_path.exists():
            logger.info(f"Loading sample dataset from {self.sample_dataset_path}")
            df = pd.read_csv(self.sample_dataset_path)
        else:
            if not self.dataset_path.exists():
                self.download_dataset()

            logger.info(f"Loading dataset from {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)

        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    def preprocess_data(self, df: pd.DataFrame, fit_encoder: bool = True) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Preprocess the dataset by dropping irrelevant columns and encoding target

        Args:
            df: Raw dataframe
            fit_encoder: Whether to fit the label encoder (True for training, False for inference)

        Returns:
            Tuple of (features_df, encoded_targets, objid_series)
        """
        logger.info("Starting data preprocessing")

        # Check for missing columns
        missing_cols = [col for col in self.columns_to_drop if col in df.columns]
        if missing_cols:
            logger.info(f"Dropping columns: {missing_cols}")
            df = df.drop(columns=missing_cols)

        # Extract objid if present (for identification)
        objid_series = df.get('objid', None)

        # Extract target column
        if 'class' in df.columns:
            target_series = df['class']
            if fit_encoder:
                encoded_targets = pd.Series(self.label_encoder.fit_transform(target_series))
                logger.info(f"Target classes: {list(self.label_encoder.classes_)}")
            else:
                encoded_targets = pd.Series(self.label_encoder.transform(target_series))

            # Remove target from features
            df = df.drop(columns=['class'])
        else:
            encoded_targets = None

        # Select feature columns (excluding objid for training)
        available_features = [col for col in self.feature_columns if col in df.columns]
        features_df = df[available_features].copy()

        logger.info(f"Selected {len(available_features)} feature columns: {available_features}")
        logger.info(f"Feature matrix shape: {features_df.shape}")

        # Handle missing values
        if features_df.isnull().sum().sum() > 0:
            logger.warning("Missing values detected, filling with median")
            features_df = features_df.fillna(features_df.median())

        return features_df, encoded_targets, objid_series

    def get_class_name(self, encoded_class: int) -> str:
        """Convert encoded class back to original class name"""
        if not hasattr(self.label_encoder, 'classes_'):
            raise ValueError("Label encoder not fitted yet")
        return self.label_encoder.classes_[encoded_class]

    def get_class_names(self) -> list:
        """Get all class names"""
        if not hasattr(self.label_encoder, 'classes_'):
            raise ValueError("Label encoder not fitted yet")
        return list(self.label_encoder.classes_)