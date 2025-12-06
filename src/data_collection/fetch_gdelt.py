"""
GDELT data fetching module
Downloads and processes GDELT event data for specified regions and time periods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import zipfile
import io
from tqdm import tqdm
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("gdelt_fetcher")


class GDELTFetcher:
    """Fetch GDELT data for conflict prediction"""

    GDELT_BASE_URL = "http://data.gdeltproject.org/events/"

    # GDELT 2.0 column names
    GDELT_COLUMNS = [
        'GLOBALEVENTID', 'SQLDATE', 'MonthYear', 'Year', 'FractionDate',
        'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
        'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
        'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
        'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
        'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
        'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
        'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode',
        'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
        'NumArticles', 'AvgTone',
        'Actor1Geo_Type', 'Actor1Geo_FullName', 'Actor1Geo_CountryCode',
        'Actor1Geo_ADM1Code', 'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
        'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode',
        'Actor2Geo_ADM1Code', 'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID',
        'ActionGeo_Type', 'ActionGeo_FullName', 'ActionGeo_CountryCode',
        'ActionGeo_ADM1Code', 'ActionGeo_Lat', 'ActionGeo_Long', 'ActionGeo_FeatureID',
        'DATEADDED', 'SOURCEURL'
    ]

    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize GDELT fetcher

        Args:
            output_dir: Directory to save raw data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"GDELT Fetcher initialized. Output: {self.output_dir}")

    def fetch_daily_data(self, date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch GDELT data for a single day

        Args:
            date: Date to fetch

        Returns:
            DataFrame with GDELT events or None if failed
        """
        date_str = date.strftime("%Y%m%d")
        url = f"{self.GDELT_BASE_URL}{date_str}.export.CSV.zip"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Extract zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = f"{date_str}.export.CSV"
                with z.open(csv_filename) as f:
                    df = pd.read_csv(
                        f,
                        sep='\t',
                        header=None,
                        names=self.GDELT_COLUMNS,
                        low_memory=False,
                        encoding='utf-8',
                        on_bad_lines='skip'
                    )

            logger.info(f"Fetched {len(df)} events for {date_str}")
            return df

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch {date_str}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            return None

    def filter_region_data(
        self,
        df: pd.DataFrame,
        countries: List[str],
        actor_keywords: List[str]
    ) -> pd.DataFrame:
        """
        Filter GDELT data for specific region

        Args:
            df: GDELT DataFrame
            countries: List of country codes (e.g., ['ISR', 'PSE'])
            actor_keywords: Keywords to match in actor names

        Returns:
            Filtered DataFrame
        """
        # Filter by country codes in actors or action location
        country_mask = (
            df['Actor1CountryCode'].isin(countries) |
            df['Actor2CountryCode'].isin(countries) |
            df['ActionGeo_CountryCode'].isin(countries)
        )

        # Filter by actor keywords
        keyword_pattern = '|'.join(actor_keywords)
        keyword_mask = (
            df['Actor1Name'].str.contains(keyword_pattern, case=False, na=False) |
            df['Actor2Name'].str.contains(keyword_pattern, case=False, na=False)
        )

        filtered = df[country_mask | keyword_mask].copy()
        logger.info(f"Filtered to {len(filtered)} region-specific events")

        return filtered

    def fetch_region_data(
        self,
        region_name: str,
        start_date: str,
        end_date: str,
        countries: List[str],
        actor_keywords: List[str]
    ) -> pd.DataFrame:
        """
        Fetch GDELT data for a region over a date range

        Args:
            region_name: Name of region
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            countries: List of country codes
            actor_keywords: Actor keywords to filter

        Returns:
            Combined DataFrame for the region
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = []
        current = start

        logger.info(f"Fetching data for {region_name} from {start_date} to {end_date}")

        with tqdm(total=(end - start).days + 1, desc=f"Fetching {region_name}") as pbar:
            while current <= end:
                df = self.fetch_daily_data(current)

                if df is not None:
                    filtered = self.filter_region_data(df, countries, actor_keywords)
                    if len(filtered) > 0:
                        all_data.append(filtered)

                current += timedelta(days=1)
                pbar.update(1)

                # Rate limiting
                time.sleep(0.1)

        if not all_data:
            logger.warning(f"No data found for {region_name}")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        # Save raw data
        output_file = self.output_dir / f"{region_name}_raw.parquet"
        combined.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(combined)} events to {output_file}")

        return combined

    def fetch_all_regions(self, config: Dict) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all regions in config

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary mapping region names to DataFrames
        """
        results = {}

        for region_key, region_config in config['regions'].items():
            logger.info(f"Processing region: {region_config['name']}")

            df = self.fetch_region_data(
                region_name=region_key,
                start_date=region_config['date_range']['start'],
                end_date=region_config['date_range']['end'],
                countries=region_config['countries'],
                actor_keywords=region_config['actor_keywords']
            )

            results[region_key] = df

        return results


if __name__ == "__main__":
    from src.utils.config_loader import load_config

    config = load_config()
    fetcher = GDELTFetcher()

    # Fetch data for all regions
    data = fetcher.fetch_all_regions(config)

    # Print summary
    for region, df in data.items():
        print(f"\n{region}: {len(df)} events")
        if len(df) > 0:
            print(f"  Date range: {df['SQLDATE'].min()} to {df['SQLDATE'].max()}")
