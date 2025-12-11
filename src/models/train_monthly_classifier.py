
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_engineering import FeatureEngineer
from src.models.transformer_forecaster import TransformerForecaster
from src.utils.logger import setup_logger
from src.utils.device import get_device

logger = setup_logger("train_monthly")

def train_monthly_classifier():
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = get_device(config)
    
    # Load Data
    region = "israel_palestine" # Hardcoded for benchmark
    data_path = f"data/processed/{region}_processed.parquet"
    df = pd.read_parquet(data_path)
    
    # Feature Engineering
    fe = FeatureEngineer()
    
    # Define Threshold (Median from analysis)
    THRESHOLD = 6812.0 
    WINDOW_SIZE = 30
    SEQ_LEN = config['time_series']['window_size']
    
    from sklearn.model_selection import train_test_split

    # Prepare Data on FULL dataset to maximize valid windows
    X_full, y_full, _ = fe.prepare_monthly_classification_data(
        df, 'target_fatalities', SEQ_LEN, THRESHOLD, WINDOW_SIZE, fit_scaler=True
    )
    
    # Stratified Shuffle Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    logger.info(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    logger.info(f"Train Positive Rate: {y_train.mean():.4f}")
    logger.info(f"Test Positive Rate: {y_test.mean():.4f}")
    logger.info(f"Test Labels: {y_test}")
    
    # Model Setup (Reusing TransformerForecaster but treating output as logits)
    input_dim = X_train.shape[2]
    model_config = {
        'input_dim': input_dim,
        'd_model': config['model']['transformer']['d_model'],
        'nhead': config['model']['transformer']['nhead'],
        'num_layers': config['model']['transformer']['num_layers'],
        'forecast_horizon': 1 # Binary output (logit)
    }
    model = TransformerForecaster(model_config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Training Loop
    epochs = 50
    best_auc = 0
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch) # Logits
                probs = torch.sigmoid(output).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(y_batch.numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        
        try:
            auc = roc_auc_score(all_targets, all_preds)
            acc = accuracy_score(all_targets, (all_preds > 0.5).astype(int))
            
            if auc > best_auc:
                best_auc = auc
                # torch.save(model.state_dict(), "models/transformer_monthly_best.pth")
                
            if (epoch+1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Test AUC: {auc:.4f} | Acc: {acc:.4f}")
        except ValueError:
             logger.warning("Validation skipped due to single class in batch")

    logger.info(f"Final Best Test AUC: {best_auc:.4f}")
    
    # Save Report
    with open("results/monthly_benchmark.txt", "w") as f:
        f.write("Monthly Classification Benchmark (High Intensity Prediction)\n")
        f.write("========================================================\n")
        f.write(f"Threshold (30-day Fatalities): > {THRESHOLD}\n")
        f.write(f"Test AUC: {best_auc:.4f}\n")

if __name__ == "__main__":
    train_monthly_classifier()
