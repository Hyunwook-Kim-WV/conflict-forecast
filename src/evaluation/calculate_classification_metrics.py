import pandas as pd
import numpy as np
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
from src.evaluation.evaluate import ConflictEvaluator

logger = setup_logger("classification_metrics")

def calculate_metrics(region_name: str):
    logger.info(f"Calculating classification metrics for {region_name}")
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
    
    # Split (Same as training)
    val_split = 0.2
    split_idx = int(len(X) * (1 - val_split))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    dates_val = dates.iloc[split_idx:]
    
    # Train Model (or load if we had saving logic, but training is fast enough for now to ensure consistency)
    logger.info("Training Transformer Model...")
    model, _ = train_region_model(
        region_name, config, X_train, X_val, y_train, y_val
    )
    
    model.eval()
    with torch.no_grad():
        pred_val = model(torch.FloatTensor(X_val).to(get_device(config))).cpu().numpy()
        
    # Flatten arrays to ensure 1D
    pred_val = pred_val.flatten()
    y_val = y_val.flatten()
    
    logger.info(f"Shapes - Pred: {pred_val.shape}, True: {y_val.shape}")
    
    # Debug stats
    logger.info(f"True Values (Fatalities): Min={y_val.min()}, Max={y_val.max()}, Mean={y_val.mean()}")
    logger.info(f"Predictions: Min={pred_val.min()}, Max={pred_val.max()}, Mean={pred_val.mean()}")
        
    # Convert to binary
    # Ground truth: Conflict if fatalities > 0
    y_true_binary = (y_val > 0).astype(int)
    logger.info(f"True Positives (Conflict Days): {y_true_binary.sum()} / {len(y_true_binary)}")
    
    # Find optimal threshold for predictions
    best_f1 = 0
    best_threshold = 0
    
    # Search threshold from 0 to max prediction
    thresholds = np.linspace(0, max(pred_val.max(), 1.0), 100)
    
    for thresh in thresholds:
        y_pred_binary = (pred_val > thresh).astype(int)
        f1 = 0
        # Manual F1 to avoid sklearn overhead in loop if needed, but sklearn is fine
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
    logger.info(f"Optimal Threshold: {best_threshold:.4f} (Validation F1: {best_f1:.4f})")
    
    # Apply optimal threshold
    y_pred_binary = (pred_val > best_threshold).astype(int)
    
    # Compute detailed metrics
    evaluator = ConflictEvaluator()
    metrics = evaluator.compute_metrics(y_true_binary, y_pred_binary, pred_val)
    
    # Generate report
    evaluator.generate_report(
        region_name=f"{region_name}_classification",
        metrics=metrics,
        dates=dates_val['date'],
        y_true=y_true_binary,
        y_pred=y_pred_binary,
        y_scores=pred_val.flatten(), # Scores are the predicted fatalities
        threshold=best_threshold
    )
    
    print("\n" + "="*50)
    print(f"Classification Metrics for {region_name}")
    print("="*50)
    print(f"Optimal Threshold (Predicted Fatalities): > {best_threshold:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    calculate_metrics('israel_palestine')
