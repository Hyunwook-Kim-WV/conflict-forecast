import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.forecast_horizon = config.get('forecast_horizon', 1)
        self.bidirectional = config.get('bidirectional', True)
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        lstm_out_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.forecast_horizon)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_step = lstm_out[:, -1, :]
        
        output = self.head(last_step)
        return output
