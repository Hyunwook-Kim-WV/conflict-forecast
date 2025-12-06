"""
Granger Causality Test for GDELT features vs UCDP Fatalities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests
import sys
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("causality_test")

def run_granger_test(region_name: str, max_lag: int = 7):
    """
    Run Granger Causality test for a region
    """
    logger.info(f"Running Granger Causality Test for {region_name}")
    
    # Load data
    processed_file = Path(f"data/processed/{region_name}_processed.parquet")
    if not processed_file.exists():
        logger.error(f"Data not found for {region_name}")
        return

    df = pd.read_parquet(processed_file)
    
    # Check if target exists
    target_col = 'target_fatalities'
    if target_col not in df.columns:
        # Try to reconstruct if missing (or use target_conflict as proxy)
        logger.warning(f"{target_col} not found. Using 'target_conflict' if available.")
        target_col = 'target_conflict'
        if target_col not in df.columns:
             logger.error("No target column found.")
             return

    # Select top features (e.g., event codes) to test
    # We'll test a few aggregated features to avoid testing hundreds
    # Let's create some aggregate features if they don't exist
    
    # Example: Total Events, Goldstein Scale Mean
    if 'GoldsteinScale' in df.columns:
        # Assuming GoldsteinScale is already aggregated or we take the mean of features?
        # Actually, let's look for columns that look like features.
        pass
        
    # Let's pick a few representative features
    # 1. Total Event Count (if we can sum up event codes)
    # 2. Avg Tone (if available)
    # 3. Specific Event Codes (e.g., 'event_code_190' - Use of armed force)
    
    feature_candidates = [c for c in df.columns if c.startswith('event_code_') or c.startswith('actor_type_')]
    
    # Sort by variance or sum to pick most active ones
    top_features = df[feature_candidates].sum().sort_values(ascending=False).head(5).index.tolist()
    logger.info(f"Testing top 5 active features: {top_features}")
    
    results = {}
    
    for feature in top_features:
        # Prepare data for test: [target, predictor]
        # Granger test checks if predictor causes target. 
        # statsmodels expects [y, x] where x is the predictor.
        
        data = df[[target_col, feature]].dropna()
        
        # Make stationary? The test assumes stationarity. 
        # Our data might not be stationary. Let's diff once.
        data_diff = data.diff().dropna()
        
        logger.info(f"Testing {feature} -> {target_col}")
        try:
            # verbose=False to suppress stdout
            test_result = grangercausalitytests(data_diff, maxlag=max_lag, verbose=False)
            
            # Extract p-values for each lag
            p_values = {}
            for lag, val in test_result.items():
                # val[0] is the test result dictionary
                # 'ssr_ftest' is usually robust. val[0]['ssr_ftest'][1] is the p-value
                p_value = val[0]['ssr_ftest'][1]
                p_values[lag] = p_value
            
            min_p_value = min(p_values.values())
            best_lag = min(p_values, key=p_values.get)
            
            results[feature] = {
                'min_p_value': min_p_value,
                'best_lag': best_lag,
                'significant': min_p_value < 0.05
            }
            
            logger.info(f"  Min P-value: {min_p_value:.4f} (Lag {best_lag}) - Significant: {min_p_value < 0.05}")
            
        except Exception as e:
            logger.error(f"Test failed for {feature}: {e}")

    # Save results
    results_df = pd.DataFrame(results).T
    save_path = Path(f"results/{region_name}_granger_causality.csv")
    results_df.to_csv(save_path)
    logger.info(f"Saved Granger Causality results to {save_path}")
    print(results_df)

if __name__ == "__main__":
    run_granger_test('israel_palestine')
