
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_engineering import FeatureEngineer
from src.models.transformer_forecaster import TransformerForecaster
from src.utils.logger import setup_logger
from src.utils.device import get_device

logger = setup_logger("window_ablation")

def run_experiment(window_size, df, config, device, fe):
    logger.info(f"--- Starting Experiment: Window Size {window_size} ---")
    
    # 1. Determine Threshold (Median of rolling sum)
    # Shift is handled in prepare_data, so we just check the distribution here
    rolling_sum = df['target_fatalities'].rolling(window=window_size).sum()
    threshold = rolling_sum.median()
    
    # If median is 0 (sparse data), default to > 0 (Any conflict)
    if threshold == 0:
        threshold = 0.5 # effectively > 0 for integers
        logger.info(f"Median is 0, setting threshold to > 0 (Any Conflict)")
    else:
        logger.info(f"Dynamic Threshold (Median): {threshold}")

    SEQ_LEN = config['time_series']['window_size']
    
    # 2. Prepare Data
    try:
        X_full, y_full, _ = fe.prepare_monthly_classification_data(
            df, 'target_fatalities', SEQ_LEN, threshold, window_size, fit_scaler=True
        )
    except Exception as e:
        logger.error(f"Data prep failed: {e}")
        return 0.0

    # check if we have enough data
    if len(y_full) < 50:
        logger.warning("Not enough data points.")
        return 0.0
        
    # Check class balance
    if y_full.mean() == 0 or y_full.mean() == 1:
        logger.warning(f"Single class target (mean={y_full.mean()}). AUC undefined.")
        return 0.5

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    # 4. Model Setup
    input_dim = X_train.shape[2]
    model_config = {
        'input_dim': input_dim,
        'd_model': config['model']['transformer']['d_model'],
        'nhead': config['model']['transformer']['nhead'],
        'num_layers': config['model']['transformer']['num_layers'],
        'forecast_horizon': 1
    }
    model = TransformerForecaster(model_config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    
    # 5. Training
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    epochs = 30 # Reduced epochs for speed
    best_auc = 0.5
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                probs = torch.sigmoid(output).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(y_batch.numpy())
        
        try:
            auc = roc_auc_score(all_targets, all_preds)
            if auc > best_auc:
                best_auc = auc
        except:
            pass
            
    logger.info(f"Window {window_size} Result: AUC {best_auc:.4f}")
    return best_auc

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    device = get_device(config)
    
    region = "israel_palestine"
    data_path = f"data/processed/{region}_processed.parquet"
    df = pd.read_parquet(data_path)
    
    fe = FeatureEngineer()
    
    windows = [1, 7, 14, 21, 28]
    results = {}
    
    for w in windows:
        auc = run_experiment(w, df, config, device, fe)
        results[w] = auc
        
    # Save Results
    results_path = Path("results/ablation")
    results_path.mkdir(exist_ok=True, parents=True)
    
    # DataFrame
    res_df = pd.DataFrame(list(results.items()), columns=['Window', 'AUC'])
    res_df.to_csv(results_path / "window_ablation.csv", index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['Window'], res_df['AUC'], marker='o', linestyle='-', color='b')
    plt.title('Conflict Prediction Performance vs Window Size')
    plt.xlabel('Prediction Window (Days)')
    plt.ylabel('ROC AUC Score')
    plt.grid(True)
    plt.ylim(0.4, 1.0)
    plt.savefig(results_path / "auc_vs_window.png")
    
    print("\n=== Ablation Results ===")
    print(res_df)

if __name__ == "__main__":
    main()
