
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_engineering import FeatureEngineer

def analyze_feature_importance():
    # 1. Load Data
    region = "israel_palestine"
    data_path = f"data/processed/{region}_processed.parquet"
    df = pd.read_parquet(data_path)
    
    # 2. Prepare Data (Same Monthly Classification setup)
    fe = FeatureEngineer()
    SEQ_LEN = 30
    WINDOW_SIZE = 30
    
    # Determine threshold (Median)
    rolling_sum = df['target_fatalities'].rolling(window=WINDOW_SIZE).sum()
    threshold = rolling_sum.median()
    if threshold == 0: threshold = 0.5
    
    print(f"Analyzing importance for Monthly Prediction (Threshold > {threshold})")
    
    # We use prepare_monthly_classification_data to get the exact X and y
    # X shape: (samples, seq_len, n_features)
    X_seq, y_seq, _ = fe.prepare_monthly_classification_data(
        df, 'target_fatalities', SEQ_LEN, threshold, WINDOW_SIZE, fit_scaler=True
    )
    
    # Flatten the sequence for Random Forest
    # Instead of (samples, 30, features), we can aggregate (mean/std) or just take the flattened vector.
    # For interpretability, let's take the MEAN of the 30-day window features.
    # This tells us "Average level of X over 30 days" importance.
    
    # Get feature names
    dummy_features = fe.select_features(df, exclude_cols=['date', 'is_conflict', 'conflict_name', 'target_fatalities'])
    feature_names = dummy_features.columns.tolist()
    
    # X_seq is un-flattened. Let's aggregate for RF
    # X_seq: (N, 30, F) -> X_flat: (N, F) using Mean
    X_flat = np.mean(X_seq, axis=1)
    
    # 3. Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_flat, y_seq)
    
    # 4. Get Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_features = []
    for f in range(10):
        name = feature_names[indices[f]]
        score = importances[indices[f]]
        top_features.append((name, score))

    print("\n=== TOP FEATURES START ===")
    with open("results/feature_importance.txt", "w") as f:
        for name, score in top_features:
            print(f"{name}: {score:.4f}")
            f.write(f"{name}: {score:.4f}\n")
    print("=== TOP FEATURES END ===\n")

if __name__ == "__main__":
    analyze_feature_importance()

if __name__ == "__main__":
    analyze_feature_importance()
