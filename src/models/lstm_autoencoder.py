"""
LSTM Autoencoder for time series anomaly detection
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("lstm_autoencoder")


class LSTMEncoder(nn.Module):
    """LSTM Encoder"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM Encoder

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            outputs: LSTM outputs (batch_size, seq_len, hidden_dim * num_directions)
            hidden: Hidden state tuple (h_n, c_n)
        """
        outputs, hidden = self.lstm(x)
        return outputs, hidden


class LSTMDecoder(nn.Module):
    """LSTM Decoder"""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM Decoder

        Args:
            output_dim: Output feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=hidden_dim * self.num_directions,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x, hidden=None):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim * num_directions)
            hidden: Initial hidden state (optional)

        Returns:
            outputs: Reconstructed outputs (batch_size, seq_len, output_dim)
        """
        lstm_out, _ = self.lstm(x, hidden)
        outputs = self.fc(lstm_out)
        return outputs


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for anomaly detection"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM Autoencoder

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Encoder
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # Decoder
        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        logger.info(f"LSTM Autoencoder initialized:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Bidirectional: {bidirectional}")

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            reconstructed: Reconstructed tensor (batch_size, seq_len, input_dim)
        """
        # Encode
        encoded, hidden = self.encoder(x)

        # Decode
        reconstructed = self.decoder(encoded)

        return reconstructed

    def encode(self, x):
        """
        Encode input to latent representation

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            encoded: Encoded representation
        """
        encoded, hidden = self.encoder(x)
        return encoded

    def compute_reconstruction_error(self, x, reduction='mean'):
        """
        Compute reconstruction error (MSE)

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Reconstruction error
        """
        reconstructed = self.forward(x)

        if reduction == 'none':
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))  # Per sample
        elif reduction == 'mean':
            error = torch.mean((x - reconstructed) ** 2)  # Overall mean
        elif reduction == 'sum':
            error = torch.sum((x - reconstructed) ** 2)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        return error


class AnomalyDetector:
    """Anomaly detector using LSTM Autoencoder"""

    def __init__(
        self,
        model: LSTMAutoencoder,
        device: str = 'cpu'
    ):
        """
        Initialize anomaly detector

        Args:
            model: Trained LSTM Autoencoder
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = None
        logger.info(f"Anomaly Detector initialized on {device}")

    def compute_threshold(
        self,
        normal_data: np.ndarray,
        method: str = 'percentile',
        percentile: float = 95.0,
        std_multiplier: float = 3.0
    ):
        """
        Compute anomaly threshold from normal data

        Args:
            normal_data: Normal training data (n_samples, seq_len, input_dim)
            method: 'percentile' or 'std'
            percentile: Percentile for threshold (if method='percentile')
            std_multiplier: Std multiplier for threshold (if method='std')
        """
        self.model.eval()

        with torch.no_grad():
            x = torch.FloatTensor(normal_data).to(self.device)
            errors = self.model.compute_reconstruction_error(x, reduction='none')
            errors = errors.cpu().numpy()

        if method == 'percentile':
            self.threshold = np.percentile(errors, percentile)
            logger.info(f"Threshold (percentile {percentile}): {self.threshold:.6f}")

        elif method == 'std':
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            self.threshold = mean_error + std_multiplier * std_error
            logger.info(f"Threshold (mean + {std_multiplier} std): {self.threshold:.6f}")
            logger.info(f"  Mean error: {mean_error:.6f}")
            logger.info(f"  Std error: {std_error:.6f}")

        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.threshold

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies

        Args:
            data: Input data (n_samples, seq_len, input_dim)

        Returns:
            (anomaly_scores, anomaly_labels)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call compute_threshold first.")

        self.model.eval()

        with torch.no_grad():
            x = torch.FloatTensor(data).to(self.device)
            errors = self.model.compute_reconstruction_error(x, reduction='none')
            errors = errors.cpu().numpy()

        labels = (errors > self.threshold).astype(int)

        logger.info(f"Predicted {labels.sum()} anomalies out of {len(labels)} samples")

        return errors, labels

    def save_model(self, path: str):
        """Save model and threshold"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'bidirectional': self.model.bidirectional
            }
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model and threshold"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        logger.info(f"Model loaded from {path}")
        logger.info(f"  Threshold: {self.threshold}")


if __name__ == "__main__":
    # Test model
    batch_size = 32
    seq_len = 30
    input_dim = 50

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    reconstructed = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstructed.shape}")

    # Compute error
    error = model.compute_reconstruction_error(x, reduction='none')
    print(f"Reconstruction error shape: {error.shape}")
    print(f"Mean error: {error.mean():.6f}")
