import torch
import torch.nn as nn
import math
from typing import Dict, Any

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class TransformerForecaster(nn.Module):
    """
    Transformer-based model for forecasting conflict intensity.
    Uses an Encoder-only architecture to predict a target variable (e.g., fatalities)
    from a sequence of past events.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model parameters
        self.input_dim = config.get('input_dim', 10)
        self.d_model = config.get('d_model', 64)
        self.nhead = config.get('nhead', 4)
        self.num_layers = config.get('num_layers', 2)
        self.dim_feedforward = config.get('dim_feedforward', 256)
        self.dropout = config.get('dropout', 0.1)
        self.sequence_length = config.get('sequence_length', 30)
        self.forecast_horizon = config.get('forecast_horizon', 1) # Predict t+1
        
        # Layers
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.sequence_length + 100)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        
        # Output head: Predict scalar value (intensity)
        # We can take the last time step's embedding or pool them.
        # Here we take the last time step.
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_feedforward, self.forecast_horizon) 
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input sequence [batch_size, seq_len, input_dim]
            
        Returns:
            Prediction [batch_size, forecast_horizon]
        """
        # Permute for Transformer [seq_len, batch_size, d_model]
        src = src.permute(1, 0, 2)
        
        # Embedding and Positional Encoding
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        # Transformer Encoder
        # output: [seq_len, batch_size, d_model]
        output = self.transformer_encoder(src)
        
        # Take the last time step for forecasting
        # [batch_size, d_model]
        last_step_output = output[-1, :, :]
        
        # Prediction
        # [batch_size, forecast_horizon]
        prediction = self.output_head(last_step_output)
        
        return prediction
