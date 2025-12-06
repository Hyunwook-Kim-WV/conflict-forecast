import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys
import copy

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.features.feature_engineering import FeatureEngineer
from src.models.train import train_region_model
from src.utils.device import get_device

logger = setup_logger("model_comparison")

def compare_models(region_name: str):
    logger.info(f"Comparing models for {region_name}")
    config = load_config()
    
    # Load data
    processed_file = Path(f"data/processed/{region_name}_processed.parquet")
    if not processed_file.exists():
        logger.error("Data not found")
        return

    df = pd.read_parquet(processed_file)
    
    # Prepare data
    engineer = FeatureEngineer()
    target_col = 'target_fatalities'
    
    X, y, dates = engineer.create_forecasting_sequences(
        df=df,
        target_col=target_col,
        sequence_length=30,
        forecast_horizon=1,
        fit_scaler=True
    )
    
    # Split
    val_split = 0.2
    split_idx = int(len(X) * (1 - val_split))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    dates_val = dates.iloc[split_idx:]
    
    results = {}
    
    # 1. Train Transformer
    logger.info("Training Transformer...")
    config_trans = copy.deepcopy(config)
    config_trans['model']['type'] = 'transformer_forecaster'
    
    model_trans, _ = train_region_model(
        region_name, config_trans, X_train, X_val, y_train, y_val
    )
    
    model_trans.eval()
    with torch.no_grad():
        pred_trans = model_trans(torch.FloatTensor(X_val).to(get_device(config))).cpu().numpy()
        
    results['Transformer'] = pred_trans
    
    # 2. Train LSTM
    logger.info("Training LSTM...")
    config_lstm = copy.deepcopy(config)
    config_lstm['model']['type'] = 'lstm_forecaster'
    
    model_lstm, _ = train_region_model(
        region_name, config_lstm, X_train, X_val, y_train, y_val
    )
    
    model_lstm.eval()
    with torch.no_grad():
        pred_lstm = model_lstm(torch.FloatTensor(X_val).to(get_device(config))).cpu().numpy()
        
    results['LSTM'] = pred_lstm
    
    # 3. Visualize
    logger.info("Generating visualization...")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Timeline
    plt.subplot(2, 1, 1)
    plt.plot(dates_val['date'], y_val, label='Actual', color='black', alpha=0.6, linewidth=2)
    plt.plot(dates_val['date'], results['Transformer'], label='Transformer', color='blue', alpha=0.8)
    plt.plot(dates_val['date'], results['LSTM'], label='LSTM', color='red', alpha=0.8, linestyle='--')
    plt.title(f'Conflict Intensity Forecasting: Transformer vs LSTM ({region_name})')
    plt.ylabel('Fatalities (Scaled)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Error Distribution
    plt.subplot(2, 1, 2)
    error_trans = np.abs(y_val - results['Transformer'])
    error_lstm = np.abs(y_val - results['LSTM'])
    
    plt.hist(error_trans, bins=30, alpha=0.5, label=f'Transformer MAE={np.mean(error_trans):.4f}', color='blue')
    plt.hist(error_lstm, bins=30, alpha=0.5, label=f'LSTM MAE={np.mean(error_lstm):.4f}', color='red')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    save_path = Path(f"results/{region_name}_model_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    compare_models('israel_palestine')
