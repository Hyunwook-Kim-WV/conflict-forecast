"""
Data preprocessing pipeline for GDELT events
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("preprocessor")


class GDELTPreprocessor:
    """Preprocess GDELT event data for time series modeling"""

    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        """
        Initialize preprocessor

        Args:
            input_dir: Directory with raw GDELT data
            output_dir: Directory to save processed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preprocessor initialized")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw GDELT data

        Args:
            df: Raw GDELT DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data: {len(df)} rows")

        # Convert date
        df['date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d', errors='coerce')

        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])

        # Fill missing numeric values
        numeric_cols = ['GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Fill missing categorical values
        categorical_cols = [
            'Actor1CountryCode', 'Actor2CountryCode', 'EventCode',
            'EventBaseCode', 'EventRootCode', 'QuadClass',
            'Actor1Type1Code', 'Actor2Type1Code'
        ]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('UNKNOWN')

        logger.info(f"After cleaning: {len(df)} rows")
        return df

    def aggregate_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate event features by day

        Args:
            df: Cleaned GDELT DataFrame

        Returns:
            Daily aggregated features
        """
        logger.info("Aggregating daily features")

        # Group by date
        daily = df.groupby('date').agg({
            'GLOBALEVENTID': 'count',  # Event count
            'GoldsteinScale': ['mean', 'std', 'min', 'max'],
            'NumMentions': ['sum', 'mean'],
            'NumSources': ['sum', 'mean'],
            'NumArticles': ['sum', 'mean'],
            'AvgTone': ['mean', 'std'],
        }).reset_index()

        # Flatten column names
        daily.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                        for col in daily.columns.values]

        # Rename for clarity
        daily = daily.rename(columns={
            'GLOBALEVENTID_count': 'event_count',
            'GoldsteinScale_mean': 'goldstein_mean',
            'GoldsteinScale_std': 'goldstein_std',
            'GoldsteinScale_min': 'goldstein_min',
            'GoldsteinScale_max': 'goldstein_max',
            'NumMentions_sum': 'mentions_total',
            'NumMentions_mean': 'mentions_mean',
            'NumSources_sum': 'sources_total',
            'NumSources_mean': 'sources_mean',
            'NumArticles_sum': 'articles_total',
            'NumArticles_mean': 'articles_mean',
            'AvgTone_mean': 'tone_mean',
            'AvgTone_std': 'tone_std',
        })

        # Event code distribution (Using EventBaseCode for more granularity)
        event_codes = df.groupby(['date', 'EventBaseCode']).size().unstack(fill_value=0)
        event_codes.columns = [f'event_code_{col}' for col in event_codes.columns]
        event_codes = event_codes.reset_index()

        # Actor Type distribution
        # Combine Actor1 and Actor2 types
        actor1_types = df.groupby(['date', 'Actor1Type1Code']).size().unstack(fill_value=0)
        actor2_types = df.groupby(['date', 'Actor2Type1Code']).size().unstack(fill_value=0)
        
        # Align columns and sum
        all_types = set(actor1_types.columns) | set(actor2_types.columns)
        for col in all_types:
            if col not in actor1_types: actor1_types[col] = 0
            if col not in actor2_types: actor2_types[col] = 0
            
        actor_types = actor1_types.add(actor2_types, fill_value=0)
        actor_types.columns = [f'actor_type_{col}' for col in actor_types.columns]
        actor_types = actor_types.reset_index()

        # QuadClass distribution (1=Verbal Cooperation, 2=Material Cooperation,
        # 3=Verbal Conflict, 4=Material Conflict)
        quad_class = df.groupby(['date', 'QuadClass']).size().unstack(fill_value=0)
        quad_class.columns = [f'quad_class_{col}' for col in quad_class.columns]
        quad_class = quad_class.reset_index()

        # Merge all features
        daily = daily.merge(event_codes, on='date', how='left')
        daily = daily.merge(quad_class, on='date', how='left')
        daily = daily.merge(actor_types, on='date', how='left')

        # Fill NaN with 0
        daily = daily.fillna(0)

        logger.info(f"Created {len(daily.columns)} features for {len(daily)} days")
        return daily

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features (day of week, month, etc.)

        Args:
            df: Daily aggregated DataFrame

        Returns:
            DataFrame with temporal features
        """
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

        # Cyclical encoding for temporal features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Add rolling window features

        Args:
            df: Daily aggregated DataFrame
            windows: List of window sizes in days

        Returns:
            DataFrame with rolling features
        """
        # Key features to compute rolling statistics
        key_features = [
            'event_count', 'goldstein_mean', 'mentions_total',
            'tone_mean'
        ]

        for feature in key_features:
            if feature not in df.columns:
                continue

            for window in windows:
                # Rolling mean
                df[f'{feature}_rolling_mean_{window}d'] = (
                    df[feature].rolling(window=window, min_periods=1).mean()
                )

                # Rolling std
                df[f'{feature}_rolling_std_{window}d'] = (
                    df[feature].rolling(window=window, min_periods=1).std().fillna(0)
                )

        return df

    def merge_ground_truth(self, features_df: pd.DataFrame, region_name: str) -> pd.DataFrame:
        """
        Merge UCDP ground truth data with features
        
        Args:
            features_df: Daily aggregated features
            region_name: Name of region
            
        Returns:
            DataFrame with target variables
        """
        ucdp_file = Path("data/ground_truth") / f"ucdp_{region_name}.csv"
        
        if not ucdp_file.exists():
            logger.warning(f"UCDP data not found for {region_name}, skipping ground truth merge")
            return features_df
            
        logger.info(f"Merging UCDP ground truth from {ucdp_file}")
        ucdp_df = pd.read_csv(ucdp_file)
        
        # Convert date
        ucdp_df['date_start'] = pd.to_datetime(ucdp_df['date_start'])
        
        # Aggregate fatalities by day
        # We use date_start as the event date
        daily_fatalities = ucdp_df.groupby('date_start')['best'].sum().reset_index()
        daily_fatalities.columns = ['date', 'target_fatalities']
        
        # Merge with features
        merged = features_df.merge(daily_fatalities, on='date', how='left')
        
        # Fill missing fatalities with 0 (assuming no record = no fatalities)
        merged['target_fatalities'] = merged['target_fatalities'].fillna(0)
        
        # Create binary conflict target (for backward compatibility or classification)
        merged['target_conflict'] = (merged['target_fatalities'] > 0).astype(int)
        
        logger.info(f"Merged ground truth. Max fatalities: {merged['target_fatalities'].max()}")
        return merged

    def process_region(
        self,
        region_name: str,
        add_rolling: bool = True
    ) -> pd.DataFrame:
        """
        Process data for a specific region
        
        Args:
            region_name: Name of region
            add_rolling: Whether to add rolling features
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {region_name}")

        # Load raw data
        input_file = self.input_dir / f"{region_name}_raw.parquet"
        if not input_file.exists():
            logger.error(f"Raw data not found: {input_file}")
            return pd.DataFrame()

        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {len(df)} raw events")

        # Clean
        df = self.clean_data(df)

        # Aggregate daily
        daily = self.aggregate_daily_features(df)

        # Add temporal features
        daily = self.add_temporal_features(daily)

        # Add rolling features
        if add_rolling:
            daily = self.add_rolling_features(daily)
            
        # Merge Ground Truth (UCDP)
        daily = self.merge_ground_truth(daily, region_name)

        # Sort by date
        daily = daily.sort_values('date').reset_index(drop=True)

        # Save
        output_file = self.output_dir / f"{region_name}_processed.parquet"
        daily.to_parquet(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")

        return daily

    def process_all_regions(self, config: Dict) -> Dict[str, pd.DataFrame]:
        """
        Process data for all regions in config

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary mapping region names to processed DataFrames
        """
        results = {}

        for region_key in config['regions'].keys():
            df = self.process_region(region_key)
            results[region_key] = df

        return results


if __name__ == "__main__":
    from src.utils.config_loader import load_config

    config = load_config()
    preprocessor = GDELTPreprocessor()

    # Process all regions
    data = preprocessor.process_all_regions(config)

    # Print summary
    for region, df in data.items():
        print(f"\n{region}: {len(df)} days, {len(df.columns)} features")
        if len(df) > 0:
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Feature columns: {df.columns.tolist()[:10]}...")
