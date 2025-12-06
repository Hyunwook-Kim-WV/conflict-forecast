"""
Transformer Autoencoder for Time Series Anomaly Detection
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Transformer Autoencoder for Time Series

        Args:
            input_dim: Number of features per time step
            sequence_length: Length of input sequence
            d_model: Hidden dimension of Transformer
            nhead: Number of attention heads
            num_layers: Number of encoder/decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            device: Device to run on
        """
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.device = device

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length + 100)

        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Decoder
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)

        # Output projection
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            src: Input sequence (batch_size, seq_len, input_dim)

        Returns:
            Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        # Embed and add position encoding
        src_emb = self.embedding(src)
        src_emb = self.pos_encoder(src_emb.permute(1, 0, 2)).permute(1, 0, 2)

        # Encode
        memory = self.transformer_encoder(src_emb)

        # Decode (reconstruct)
        # For autoencoder, we can use the memory as target for decoder or use same input
        # Here we use a standard AE structure where we decode from memory
        output = self.transformer_decoder(src_emb, memory)

        # Project back to input dim
        reconstruction = self.output_layer(output)

        return reconstruction

class TransformerAnomalyDetector:
    """Anomaly Detector using Transformer Autoencoder"""

    def __init__(self, model: TransformerAutoencoder, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.criterion = nn.MSELoss(reduction='none')
        self.threshold = None

    def compute_reconstruction_error(self, sequences: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error for sequences"""
        self.model.eval()
        dataset = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        
        batch_size = 32
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                reconstruction = self.model(batch)
                
                # Compute error per sequence
                error = self.criterion(reconstruction, batch)
                error = torch.mean(error, dim=[1, 2]) # Mean over seq_len and features
                errors.extend(error.cpu().numpy())
                
        return np.array(errors)

    def compute_threshold(
        self,
        normal_data: np.ndarray,
        method: str = 'percentile',
        percentile: float = 95,
        std_multiplier: float = 3.0
    ):
        """Compute anomaly threshold from normal data"""
        errors = self.compute_reconstruction_error(normal_data)
        
        if method == 'percentile':
            self.threshold = np.percentile(errors, percentile)
        elif method == 'std':
            self.threshold = np.mean(errors) + std_multiplier * np.std(errors)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
            
        return self.threshold

    def compute_dynamic_threshold(
        self,
        errors: np.ndarray,
        window_size: int = 30,
        std_multiplier: float = 3.0
    ) -> np.ndarray:
        """
        Compute dynamic threshold using moving average and std
        
        Args:
            errors: Reconstruction errors
            window_size: Size of moving window
            std_multiplier: Multiplier for standard deviation
            
        Returns:
            Dynamic threshold array
        """
        # Calculate moving statistics
        series = pd.Series(errors)
        moving_mean = series.rolling(window=window_size, min_periods=1).mean()
        moving_std = series.rolling(window=window_size, min_periods=1).std().fillna(0)
        
        dynamic_threshold = moving_mean + std_multiplier * moving_std
        
        # For the first few points where rolling might be unstable, use global stats
        global_mean = np.mean(errors[:window_size])
        global_std = np.std(errors[:window_size])
        dynamic_threshold[:window_size] = global_mean + std_multiplier * global_std
        
        self.threshold = dynamic_threshold.values
        return self.threshold

    def predict(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies
        
        Returns:
            (anomaly_scores, predictions)
        """
        if self.threshold is None:
            raise ValueError("Threshold not computed. Call compute_threshold first.")
            
        scores = self.compute_reconstruction_error(sequences)
        
        if isinstance(self.threshold, np.ndarray):
            # Dynamic threshold
            if len(self.threshold) != len(scores):
                # If lengths mismatch (e.g. inference on new data), use last threshold value or recompute
                # For simplicity in this context, we'll warn and use the mean of the dynamic threshold
                # Ideally, dynamic thresholding should be updated online
                import warnings
                warnings.warn("Dynamic threshold length mismatch. Using mean of dynamic threshold.")
                threshold_val = np.mean(self.threshold)
                predictions = (scores > threshold_val).astype(int)
            else:
                predictions = (scores > self.threshold).astype(int)
        else:
            # Static threshold
            predictions = (scores > self.threshold).astype(int)
        
        return scores, predictions

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
