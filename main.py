"""
Main pipeline for GDELT conflict prediction
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.device import print_gpu_info, get_device
from src.data_collection.fetch_gdelt import GDELTFetcher
from src.data_collection.create_ground_truth import GroundTruthCreator
from src.preprocessing.preprocess import GDELTPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.lstm_autoencoder import LSTMAutoencoder, AnomalyDetector
from src.models.transformer_autoencoder import TransformerAutoencoder, TransformerAnomalyDetector
from src.models.train import train_region_model
from src.evaluation.evaluate import ConflictEvaluator

logger = setup_logger("main")


def run_full_pipeline(
    regions: list = None,
    skip_fetch: bool = False,
    skip_train: bool = False,
    use_bigquery: bool = None
):
    """
    Run complete pipeline for conflict prediction

    Args:
        regions: List of region names to process (None = all regions)
        skip_fetch: Skip data fetching step
        skip_train: Skip training step
        use_bigquery: Use BigQuery for data fetching (None = use config setting)
    """
    logger.info("=" * 50)
    logger.info("GDELT CONFLICT PREDICTION PIPELINE")
    logger.info("=" * 50)

    # Load config
    config = load_config()

    # Print GPU info
    print_gpu_info()

    # Determine regions to process
    if regions is None:
        regions = list(config['regions'].keys())

    logger.info(f"Processing regions: {regions}")

    # Step 1: Fetch GDELT data
    if not skip_fetch:
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: Fetching GDELT Data")
        logger.info("=" * 50)

        # Determine fetch method
        fetch_method = use_bigquery if use_bigquery is not None else (
            config.get('data_collection', {}).get('method', 'download') == 'bigquery'
        )

        if fetch_method:
            # Use BigQuery (fast)
            try:
                from src.data_collection.fetch_gdelt_bigquery import GDELTBigQueryFetcher

                bq_config = config.get('data_collection', {}).get('bigquery', {})
                fetcher = GDELTBigQueryFetcher(
                    credentials_path=bq_config.get('credentials_path'),
                    project_id=bq_config.get('project_id')
                )

                # Filter regions
                region_configs = {k: v for k, v in config['regions'].items() if k in regions}
                fetcher.fetch_all_regions({'regions': region_configs})

                logger.info("Data fetched successfully using BigQuery")

            except ImportError:
                logger.error("BigQuery not available. Install: pip install google-cloud-bigquery")
                logger.info("Falling back to direct download method")
                fetch_method = False

        if not fetch_method:
            # Use direct download (slow but no GCP required)
            fetcher = GDELTFetcher()
            for region in regions:
                region_config = config['regions'][region]
                fetcher.fetch_region_data(
                    region_name=region,
                    start_date=region_config['date_range']['start'],
                    end_date=region_config['date_range']['end'],
                    countries=region_config['countries'],
                    actor_keywords=region_config['actor_keywords']
                )
    else:
        logger.info("Skipping data fetching")

    # Step 2: Create ground truth labels
    logger.info("\n" + "=" * 50)
    logger.info("STEP 2: Creating Ground Truth Labels")
    logger.info("=" * 50)

    from src.data_collection.fetch_ucdp import UCDPFetcher
    
    ucdp_fetcher = UCDPFetcher()
    labels_dict = {} # Keep for compatibility if needed, though we use merged data now
    
    for region in regions:
        if region in ucdp_fetcher.REGION_MAPPING:
            logger.info(f"Fetching UCDP data for {region}")
            region_config = config['regions'][region]
            ucdp_fetcher.run_for_region(
                region_name=region,
                start_date=region_config['date_range']['start'],
                end_date=region_config['date_range']['end'],
                ucdp_ids=ucdp_fetcher.REGION_MAPPING[region]
            )
        else:
            logger.warning(f"No UCDP mapping for {region}, skipping ground truth fetch")
            
    # Generate labels from fetched UCDP data
    from src.data_collection.create_ground_truth import GroundTruthCreator
    creator = GroundTruthCreator()
    
    for region in regions:
        region_config = config['regions'][region]
        logger.info(f"Generating labels for {region} from UCDP data...")
        creator.create_labels_from_ucdp(
            region_name=region,
            start_date=region_config['date_range']['start'],
            end_date=region_config['date_range']['end']
        )

    # Step 3: Preprocess data
    logger.info("\n" + "=" * 50)
    logger.info("STEP 3: Preprocessing Data")
    logger.info("=" * 50)

    preprocessor = GDELTPreprocessor()
    processed_dict = preprocessor.process_all_regions(config)

    # Step 4: Train models and evaluate
    logger.info("\n" + "=" * 50)
    logger.info("STEP 4: Training and Evaluation")
    logger.info("=" * 50)

    engineer = FeatureEngineer()
    evaluator = ConflictEvaluator()

    results = {}

    for region in regions:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {region}")
        logger.info(f"{'='*50}")

        # Load data
        processed_df = processed_dict[region]
        
        # Handle labels
        if region in labels_dict:
            labels_df = labels_dict[region]
        else:
            # If no manual labels, try to use target_conflict from processed_df
            if 'target_conflict' in processed_df.columns:
                labels_df = processed_df[['date', 'target_conflict']].rename(columns={'target_conflict': 'is_conflict'})
            else:
                labels_df = pd.DataFrame({'date': processed_df['date'], 'is_conflict': 0})
                logger.warning(f"No labels found for {region}, using all zeros")

        if len(processed_df) == 0:
            logger.warning(f"No data for {region}, skipping")
            continue

        # Determine model type
        model_type = config['model'].get('type', 'lstm')
        is_forecasting = model_type == 'transformer_forecaster'

        # Prepare training data
        if is_forecasting:
            target_col = config['model'].get('target_col', 'target_fatalities')
            forecast_horizon = config['model'].get('forecast_horizon', 1)
            
            logger.info(f"Preparing forecasting data for {target_col}")
            train_seq, train_targets, train_dates = engineer.create_forecasting_sequences(
                df=processed_df,
                target_col=target_col,
                sequence_length=config['model']['lstm']['sequence_length'], # Use same seq len config for now
                forecast_horizon=forecast_horizon,
                fit_scaler=True,
                scaler_name='train'
            )
            # Flatten targets if needed
            # train_targets is (N, 1) usually
        else:
            # Anomaly Detection (Autoencoder)
            train_seq, train_labels, train_dates = engineer.prepare_training_data(
                df=processed_df,
                labels=labels_df,
                sequence_length=config['model']['lstm']['sequence_length'],
                use_normal_only=True
            )
            train_targets = None # No targets for autoencoder

        if len(train_seq) == 0:
            logger.warning(f"No training data for {region}, skipping")
            continue

        # Split train/val
        val_split = config['model']['training']['validation_split']
        split_idx = int(len(train_seq) * (1 - val_split))
        train_data = train_seq[:split_idx]
        val_data = train_seq[split_idx:]
        
        if is_forecasting:
            train_y = train_targets[:split_idx]
            val_y = train_targets[split_idx:]
        else:
            train_y = None
            val_y = None

        # Train model
        if not skip_train:
            model, trainer = train_region_model(
                region_name=region,
                config=config,
                train_sequences=train_data,
                val_sequences=val_data,
                train_targets=train_y,
                val_targets=val_y
            )
        else:
            # Load existing model logic (simplified for now)
            # ... (Existing loading logic needs update for forecasting, skipping for brevity/safety)
            logger.warning("Skip train not fully supported for forecasting yet in this refactor. Training from scratch.")
            model, trainer = train_region_model(
                region_name=region,
                config=config,
                train_sequences=train_data,
                val_sequences=val_data,
                train_targets=train_y,
                val_targets=val_y
            )

        if is_forecasting:
            # Evaluation for Forecasting
            logger.info("Evaluating Forecasting Model...")
            model.eval()
            with torch.no_grad():
                # Predict on validation set
                device = get_device(config)
                val_tensor = torch.FloatTensor(val_data).to(device)
                predictions = model(val_tensor).cpu().numpy()
                
                # Metrics
                mse = np.mean((predictions - val_y) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - val_y))
                
                logger.info(f"Validation RMSE: {rmse:.4f}")
                logger.info(f"Validation MAE: {mae:.4f}")
                
                results[region] = {
                    'rmse': rmse,
                    'mae': mae
                }
                
                # Plot predictions vs actual
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                plt.plot(val_y, label='Actual')
                plt.plot(predictions, label='Predicted')
                plt.title(f'{region} Conflict Intensity Prediction (RMSE={rmse:.2f})')
                plt.legend()
                plt.savefig(Path(config['paths']['results']) / f'{region}_forecast.png')
                plt.close()
                
        else:
            # Evaluation for Anomaly Detection
            # Compute threshold from normal data
            # ... (Existing logic)
            # Create anomaly detector
            device = get_device(config)
            if model_type == 'transformer':
                 detector = TransformerAnomalyDetector(model, device=device)
            else:
                 detector = AnomalyDetector(model, device=device)

            # Prepare test data (all periods)
            test_seq, test_labels, test_dates = engineer.prepare_test_data(
                df=processed_df,
                labels=labels_df,
                sequence_length=config['model']['lstm']['sequence_length']
            )

            # Compute dynamic threshold
            # First get errors on test set
            test_scores = detector.compute_reconstruction_error(test_seq)
            
            # Compute dynamic threshold based on test scores (simulating online adaptive threshold)
            detector.compute_dynamic_threshold(
                test_scores,
                window_size=30, # 1 month window
                std_multiplier=config['model']['anomaly_detection']['threshold_std_multiplier']
            )
            
            # Predict (uses the stored dynamic threshold)
            anomaly_scores, anomaly_predictions = detector.predict(test_seq)

            # Evaluate
            metrics = evaluator.compute_metrics(
                y_true=test_labels,
                y_pred=anomaly_predictions,
                y_scores=anomaly_scores
            )

            # Generate report
            evaluator.generate_report(
                region_name=region,
                metrics=metrics,
                dates=test_dates['date'],
                y_true=test_labels,
                y_pred=anomaly_predictions,
                y_scores=anomaly_scores,
                threshold=detector.threshold
            )

            # Save detector
            detector_path = Path(config['paths']['models']) / region / 'detector.pt'
            detector.save_model(str(detector_path))

            results[region] = {
                'metrics': metrics,
                'threshold': detector.threshold
            }

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 50)

    for region, result in results.items():
        logger.info(f"\n{region}:")
        if 'metrics' in result:
            logger.info(f"  Threshold: {result['threshold']:.6f}")
            logger.info(f"  Accuracy:  {result['metrics']['accuracy']:.4f}")
            logger.info(f"  F1 Score:  {result['metrics']['f1']:.4f}")
        else:
            logger.info(f"  RMSE: {result['rmse']:.4f}")
            logger.info(f"  MAE:  {result['mae']:.4f}")

    logger.info(f"\nResults saved to: {config['paths']['results']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT Conflict Prediction Pipeline")

    parser.add_argument(
        '--regions',
        nargs='+',
        default=None,
        help='Regions to process (default: all)'
    )
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip GDELT data fetching'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip model training (use existing models)'
    )
    parser.add_argument(
        '--bigquery',
        action='store_true',
        help='Use BigQuery for data fetching (fast, requires GCP)'
    )
    parser.add_argument(
        '--no-bigquery',
        action='store_true',
        help='Use direct download for data fetching (slow, no GCP required)'
    )

    args = parser.parse_args()

    # Determine BigQuery usage
    use_bigquery = None
    if args.bigquery:
        use_bigquery = True
    elif args.no_bigquery:
        use_bigquery = False

    run_full_pipeline(
        regions=args.regions,
        skip_fetch=args.skip_fetch,
        skip_train=args.skip_train,
        use_bigquery=use_bigquery
    )
