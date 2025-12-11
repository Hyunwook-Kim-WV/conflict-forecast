"""
Feature engineering for time series modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("features")


class FeatureEngineer:
    """Engineer features for LSTM Autoencoder"""

    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize feature engineer

        Args:
            output_dir: Directory to save feature data and scalers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scalers = {}
        logger.info("Feature Engineer initialized")

    def select_features(
        self,
        df: pd.DataFrame,
        exclude_cols: List[str] = ['date']
    ) -> pd.DataFrame:
        """
        Select relevant features for modeling

        Args:
            df: Processed DataFrame
            exclude_cols: Columns to exclude

        Returns:
            DataFrame with selected features
        """
        # Exclude non-numeric and unwanted columns
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64']
        ]

        features = df[feature_cols].copy()
        logger.info(f"Selected {len(feature_cols)} features")

        return features

    def prepare_monthly_classification_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        sequence_length: int,
        threshold: float,
        window_size: int = 30,
        scaler_name: str = 'train',
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare data for Monthly Classification (Window Sum > Threshold)
        """
        logger.info(f"Preparing monthly classification data (Threshold: {threshold})")
        
        # Calculate Rolling Sum Target
        # shift(-window_size) is not quite right if we want to predict NEXT 30 days from T.
        # Logic: Input X is [T-seq...T], Target y is 1 if sum(T+1...T+30) > Threshold
        
        target_series = df[target_col].rolling(window=window_size).sum().shift(-window_size)
        binary_target = (target_series > threshold).astype(int)
        
        # Select features
        features = self.select_features(df, exclude_cols=['date', 'is_conflict', 'conflict_name', target_col])
        features = self.make_stationary(features)
        
        normalized = self.normalize_features(features, fit=fit_scaler, scaler_name=scaler_name)
        
        X_arr = normalized.values
        y_arr = binary_target.values
        date_arr = df['date'].values
        
        X, y, dates = [], [], []
        
        # We need to stop earlier because of the forward looking window
        valid_end = len(X_arr) - window_size 
        
        for i in range(sequence_length, valid_end):
            # Input sequence: [i-seq : i]
            seq_x = X_arr[i-sequence_length : i]
            # Target: at time i (representing sum of i...i+30)
            target = y_arr[i]
            
            if np.isnan(target): continue # Should be handled by loop range, but safety check
            
            X.append(seq_x)
            y.append(target)
            dates.append(date_arr[i])
            
        return np.array(X), np.array(y), pd.DataFrame({'date': dates, 'label': y})

    def make_stationary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make features stationary using differencing
        
        Args:
            df: Input DataFrame
            
        Returns:
            Stationary DataFrame (first row will be NaN)
        """
        # distinct columns that are not features
        non_feature_cols = ['date', 'is_conflict', 'conflict_name', 'target_fatalities']
        feature_cols = [c for c in df.columns if c not in non_feature_cols and df[c].dtype in ['int64', 'float64']]
        
        df_stationary = df.copy()
        df_stationary[feature_cols] = df_stationary[feature_cols].diff()
        
        # Fill NaN created by diff (usually first row) with 0 or drop
        # Here we fill with 0 to keep shape, but ideally we should drop
        df_stationary[feature_cols] = df_stationary[feature_cols].fillna(0)
        
        logger.info("Applied differencing to features for stationarity")
        return df_stationary

    def normalize_features(
        self,
        df: pd.DataFrame,
        scaler_type: str = 'standard',
        fit: bool = True,
        scaler_name: str = 'default'
    ) -> pd.DataFrame:
        """
        Normalize features

        Args:
            df: Feature DataFrame
            scaler_type: 'standard' or 'minmax'
            fit: Whether to fit scaler or use existing
            scaler_name: Name to save/load scaler

        Returns:
            Normalized DataFrame
        """
        if fit:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")

            normalized = pd.DataFrame(
                scaler.fit_transform(df),
                columns=df.columns,
                index=df.index
            )

            # Save scaler
            self.scalers[scaler_name] = scaler
            scaler_file = self.output_dir / f"scaler_{scaler_name}.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Fitted and saved {scaler_type} scaler: {scaler_file}")

        else:
            if scaler_name not in self.scalers:
                # Load scaler
                scaler_file = self.output_dir / f"scaler_{scaler_name}.pkl"
                if not scaler_file.exists():
                    raise FileNotFoundError(f"Scaler not found: {scaler_file}")

                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                self.scalers[scaler_name] = scaler
                logger.info(f"Loaded scaler: {scaler_file}")

            normalized = pd.DataFrame(
                self.scalers[scaler_name].transform(df),
                columns=df.columns,
                index=df.index
            )

        return normalized

    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int,
        step_size: int = 1
    ) -> np.ndarray:
        """
        Create sequences for LSTM input

        Args:
            data: Feature array (n_samples, n_features)
            sequence_length: Length of each sequence
            step_size: Step size for sliding window

        Returns:
            Sequences array (n_sequences, sequence_length, n_features)
        """
        sequences = []

        for i in range(0, len(data) - sequence_length + 1, step_size):
            seq = data[i:i + sequence_length]
            sequences.append(seq)

        sequences = np.array(sequences)
        logger.info(f"Created {len(sequences)} sequences of length {sequence_length}")

        return sequences

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        sequence_length: int,
        step_size: int = 1,
        use_normal_only: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare training data with sequences

        Args:
            df: Processed feature DataFrame with 'date' column
            labels: Ground truth labels DataFrame
            sequence_length: Sequence length for LSTM
            step_size: Step size for sliding window
            use_normal_only: If True, only use normal periods for training

        Returns:
            (sequences, sequence_labels, sequence_dates)
        """
        logger.info("Preparing training data")

        # Merge features with labels
        df_with_labels = df.merge(labels[['date', 'is_conflict']], on='date', how='left')
        df_with_labels['is_conflict'] = df_with_labels['is_conflict'].fillna(0).astype(int)

        # Select features (exclude date and labels)
        features = self.select_features(df_with_labels, exclude_cols=['date', 'is_conflict'])
        
        # Make stationary
        features = self.make_stationary(features)

        # Normalize
        normalized = self.normalize_features(
            features,
            scaler_type='standard',
            fit=True,
            scaler_name='train'
        )

        # Create sequences
        feature_array = normalized.values
        label_array = df_with_labels['is_conflict'].values
        date_array = df_with_labels['date'].values

        sequences = []
        sequence_labels = []
        sequence_dates = []

        for i in range(0, len(feature_array) - sequence_length + 1, step_size):
            seq = feature_array[i:i + sequence_length]
            seq_label = label_array[i + sequence_length - 1]  # Label of last day in sequence
            seq_date = date_array[i + sequence_length - 1]

            # If use_normal_only, skip sequences that contain conflict
            if use_normal_only:
                seq_labels_window = label_array[i:i + sequence_length]
                if seq_labels_window.sum() > 0:  # Contains conflict
                    continue

            sequences.append(seq)
            sequence_labels.append(seq_label)
            sequence_dates.append(seq_date)

        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        sequence_dates = pd.DataFrame({'date': sequence_dates, 'is_conflict': sequence_labels})

        logger.info(f"Prepared {len(sequences)} sequences")
        logger.info(f"  Normal sequences: {(sequence_labels == 0).sum()}")
        logger.info(f"  Conflict sequences: {(sequence_labels == 1).sum()}")

        return sequences, sequence_labels, sequence_dates

    def prepare_test_data(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        sequence_length: int,
        step_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare test data (includes both normal and conflict)

        Args:
            df: Processed feature DataFrame
            labels: Ground truth labels
            sequence_length: Sequence length
            step_size: Step size

        Returns:
            (sequences, sequence_labels, sequence_dates)
        """
        # Use existing scaler
        df_with_labels = df.merge(labels[['date', 'is_conflict']], on='date', how='left')
        df_with_labels['is_conflict'] = df_with_labels['is_conflict'].fillna(0).astype(int)

        features = self.select_features(df_with_labels, exclude_cols=['date', 'is_conflict'])

        normalized = self.normalize_features(
            features,
            scaler_type='standard',
            fit=False,
            scaler_name='train'
        )

        feature_array = normalized.values
        label_array = df_with_labels['is_conflict'].values
        date_array = df_with_labels['date'].values

        sequences = []
        sequence_labels = []
        sequence_dates = []

        for i in range(0, len(feature_array) - sequence_length + 1, step_size):
            seq = feature_array[i:i + sequence_length]
            seq_label = label_array[i + sequence_length - 1]
            seq_date = date_array[i + sequence_length - 1]

            sequences.append(seq)
            sequence_labels.append(seq_label)
            sequence_dates.append(seq_date)

        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        sequence_dates = pd.DataFrame({'date': sequence_dates, 'is_conflict': sequence_labels})

        logger.info(f"Prepared {len(sequences)} test sequences")
        logger.info(f"  Normal: {(sequence_labels == 0).sum()}")
        logger.info(f"  Conflict: {(sequence_labels == 1).sum()}")

        return sequences, sequence_labels, sequence_dates

    def create_forecasting_sequences(
        self,
        df: pd.DataFrame,
        target_col: str,
        sequence_length: int,
        forecast_horizon: int = 1,
        step_size: int = 1,
        scaler_name: str = 'train',
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create sequences for forecasting (X, y)
        
        Args:
            df: Processed DataFrame with features and target
            target_col: Name of target column (e.g., 'target_fatalities')
            sequence_length: Input sequence length
            forecast_horizon: How many steps ahead to predict (currently supports 1)
            step_size: Sliding window step
            scaler_name: Name of scaler to use/save
            fit_scaler: Whether to fit scaler on this data
            
        Returns:
            (X, y, dates)
            X: (n_samples, seq_len, n_features)
            y: (n_samples, forecast_horizon)
            dates: DataFrame with dates corresponding to target
        """
        logger.info(f"Preparing forecasting sequences for target: {target_col}")
        
        # Select features (exclude date and target columns for X)
        # We assume target columns start with 'target_' or are explicitly excluded
        exclude_cols = ['date', 'is_conflict', 'conflict_name']
        
        features = self.select_features(df, exclude_cols=exclude_cols)
        
        # Make stationary
        features = self.select_features(df, exclude_cols=exclude_cols)
        
        # Make stationary
        features = self.make_stationary(features)
        
        # Normalize
        normalized = self.normalize_features(
            features,
            scaler_type='standard',
            fit=fit_scaler,
            scaler_name=scaler_name
        )
        
        if target_col in normalized.columns:
            target_array = normalized[target_col].values
            feature_array = normalized.values
        else:
            logger.warning(f"Target {target_col} not found in normalized features. Using raw values.")
            target_array = df[target_col].values
            feature_array = normalized.values

        date_array = df['date'].values
        
        X = []
        y = []
        dates = []
        
        for i in range(0, len(feature_array) - sequence_length - forecast_horizon + 1, step_size):
            # Input sequence
            seq_x = feature_array[i : i + sequence_length]
            
            # Target
            target_idx = i + sequence_length + forecast_horizon - 1
            seq_y = target_array[target_idx]
            
            # Date of the target
            target_date = date_array[target_idx]
            
            X.append(seq_x)
            y.append(seq_y)
            dates.append(target_date)
            
        X = np.array(X)
        y = np.array(y)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        dates_df = pd.DataFrame({'date': dates})
        
        logger.info(f"Created {len(X)} forecasting sequences")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, dates_df


if __name__ == "__main__":
    from src.utils.config_loader import load_config

    config = load_config()
    engineer = FeatureEngineer()

    # Example usage
    region = 'israel_palestine'
    processed_file = Path(f"data/processed/{region}_processed.parquet")
    labels_file = Path(f"data/ground_truth/{region}_labels.csv")

    if processed_file.exists() and labels_file.exists():
        df = pd.read_parquet(processed_file)
        labels = pd.read_csv(labels_file, parse_dates=['date'])

        # Prepare training data (normal only)
        seq, seq_labels, seq_dates = engineer.prepare_training_data(
            df, labels,
            sequence_length=30,
            use_normal_only=True
        )

        print(f"Training sequences: {seq.shape}")
        print(f"Labels: {seq_labels.shape}")
