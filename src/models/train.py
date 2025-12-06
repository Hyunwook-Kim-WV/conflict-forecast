"""
Training pipeline for LSTM Autoencoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.lstm_autoencoder import LSTMAutoencoder, AnomalyDetector as LSTMAnomalyDetector
from src.models.transformer_autoencoder import TransformerAutoencoder, TransformerAnomalyDetector
from src.models.transformer_autoencoder import TransformerAutoencoder, TransformerAnomalyDetector
from src.models.transformer_forecaster import TransformerForecaster
from src.models.lstm_forecaster import LSTMForecaster
from src.utils.logger import setup_logger
from src.utils.device import get_device, print_gpu_info, optimize_for_device

logger = setup_logger("trainer")


class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop

        Args:
            val_loss: Current validation loss

        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class Trainer:
    """Train LSTM Autoencoder"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: Path,
        model_config: dict = None
    ):
        """
        Initialize trainer

        Args:
            model: LSTM Autoencoder model
            device: Device to train on
            save_dir: Directory to save models
            model_config: Configuration dictionary for the model
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.model_config = model_config or {}
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []

        logger.info(f"Trainer initialized on {device}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> float:
        """
        Train one epoch

        Args:
            dataloader: Training data loader
            criterion: Loss function
            optimizer: Optimizer

        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(self.device)
            
            # Forward pass
            output = self.model(x)
            
            # Calculate loss
            if len(batch) > 1: # Supervised (Forecasting)
                y = batch[1].to(self.device)
                loss = criterion(output, y)
            else: # Unsupervised (Autoencoder)
                loss = criterion(output, x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """
        Validate model

        Args:
            dataloader: Validation data loader
            criterion: Loss function

        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                output = self.model(x)
                
                if len(batch) > 1: # Supervised
                    y = batch[1].to(self.device)
                    loss = criterion(output, y)
                else: # Unsupervised
                    loss = criterion(output, x)
                    
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        train_targets: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train model
        
        Args:
            train_data: Training data (n_samples, seq_len, input_dim)
            val_data: Validation data (optional)
            train_targets: Training targets (optional, for supervised learning)
            val_targets: Validation targets (optional)
            batch_size: Batch size
            epochs: Number of epochs
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        logger.info(f"Starting training:")
        logger.info(f"  Train samples: {len(train_data)}")
        if val_data is not None:
            logger.info(f"  Val samples: {len(val_data)}")
        logger.info(f"Model hidden dim: {getattr(self.model, 'hidden_dim', getattr(self.model, 'd_model', 'N/A'))}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Learning rate: {learning_rate}")

        # Create data loaders
        if train_targets is not None:
            train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_targets))
        else:
            train_dataset = TensorDataset(torch.FloatTensor(train_data))
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data is not None:
            if val_targets is not None:
                val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.FloatTensor(val_targets))
            else:
                val_dataset = TensorDataset(torch.FloatTensor(val_data))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)

            # Validate
            if val_data is not None:
                val_loss = self.validate(val_loader, criterion)
                self.val_losses.append(val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    logger.info(f"  Saved best model (val_loss: {val_loss:.6f})")

                # Early stopping
                if early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")

        # Save final model
        self.save_checkpoint('final_model.pt')

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses if val_data is not None else None
        }

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config, # Use the stored model_config
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved training plot to {save_path}")
        else:
            plt.show()

        plt.close()


def train_region_model(
    region_name: str,
    config: Dict,
    train_sequences: np.ndarray,
    val_sequences: Optional[np.ndarray] = None,
    train_targets: Optional[np.ndarray] = None,
    val_targets: Optional[np.ndarray] = None
) -> Tuple[nn.Module, Trainer]:
    """
    Train model for a specific region
    
    Args:
        region_name: Name of region
        config: Configuration dictionary
        train_sequences: Training sequences
        val_sequences: Validation sequences (optional)
        train_targets: Training targets (optional)
        val_targets: Validation targets (optional)
        
    Returns:
        (trained_model, trainer)
    """
    logger.info(f"Training model for {region_name}")

    # Get model config
    model_config = config['model']
    input_dim = train_sequences.shape[2]
    sequence_length = train_sequences.shape[1]

    # Get device from config
    device = get_device(config)

    # Initialize model
    model_type = config['model'].get('type', 'lstm')
    
    if model_type == 'lstm':
        lstm_config = config['model']['lstm']
        model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=lstm_config['hidden_dim'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'],
            bidirectional=lstm_config['bidirectional']
        )
        saved_config = lstm_config
    elif model_type == 'transformer':
        trans_config = config['model']['transformer']
        model = TransformerAutoencoder(
            input_dim=input_dim,
            sequence_length=sequence_length,
            d_model=trans_config['d_model'],
            nhead=trans_config['nhead'],
            num_layers=trans_config['num_layers'],
            dim_feedforward=trans_config['dim_feedforward'],
            dropout=trans_config['dropout'],
            device=device
        )
        saved_config = trans_config
    elif model_type == 'transformer_forecaster':
        trans_config = config['model']['transformer'] # Use same config structure for now
        model = TransformerForecaster(
            config={
                'input_dim': input_dim,
                'sequence_length': sequence_length,
                'd_model': trans_config['d_model'],
                'nhead': trans_config['nhead'],
                'num_layers': trans_config['num_layers'],
                'dim_feedforward': trans_config['dim_feedforward'],
                'dropout': trans_config['dropout'],
                'forecast_horizon': 1 # Hardcoded for now or add to config
            }
        )
        saved_config = trans_config
        saved_config = trans_config
    elif model_type == 'lstm_forecaster':
        lstm_config = config['model']['lstm']
        model = LSTMForecaster(
            config={
                'input_dim': input_dim,
                'hidden_dim': lstm_config['hidden_dim'],
                'num_layers': lstm_config['num_layers'],
                'dropout': lstm_config['dropout'],
                'bidirectional': lstm_config['bidirectional'],
                'forecast_horizon': 1
            }
        )
        saved_config = lstm_config
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Optimize model for device
    model = optimize_for_device(model, device)

    # Create trainer
    output_dir = Path(config['paths']['models']) / region_name
    trainer = Trainer(
        model=model,
        device=device,
        save_dir=output_dir,
        model_config=saved_config
    )

    # Train
    history = trainer.train(
        train_data=train_sequences,
        val_data=val_sequences,
        train_targets=train_targets,
        val_targets=val_targets,
        batch_size=model_config['training']['batch_size'],
        epochs=model_config['training']['epochs'],
        learning_rate=model_config['training']['learning_rate'],
        early_stopping_patience=model_config['training']['early_stopping_patience']
    )

    # Plot training history
    plot_path = output_dir / 'training_history.png'
    trainer.plot_training_history(str(plot_path))

    return model, trainer


if __name__ == "__main__":
    from src.utils.config_loader import load_config
    from src.features.feature_engineering import FeatureEngineer

    config = load_config()

    # Example: Train on Israel-Palestine
    region = 'israel_palestine'

    # Load processed data
    processed_file = Path(f"data/processed/{region}_processed.parquet")
    labels_file = Path(f"data/ground_truth/{region}_labels.csv")

    if processed_file.exists() and labels_file.exists():
        df = pd.read_parquet(processed_file)
        labels = pd.read_csv(labels_file, parse_dates=['date'])

        # Prepare training data
        engineer = FeatureEngineer()
        train_seq, train_labels, train_dates = engineer.prepare_training_data(
            df, labels,
            sequence_length=config['model']['lstm']['sequence_length'],
            use_normal_only=True
        )

        # Split train/val
        val_split = config['model']['training']['validation_split']
        split_idx = int(len(train_seq) * (1 - val_split))
        train_data = train_seq[:split_idx]
        val_data = train_seq[split_idx:]

        # Train
        model, trainer = train_region_model(
            region_name=region,
            config=config,
            train_sequences=train_data,
            val_sequences=val_data
        )

        print(f"\nTraining completed for {region}")
