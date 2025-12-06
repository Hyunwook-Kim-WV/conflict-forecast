"""
GDELT data fetching using Google BigQuery
Much faster than downloading individual files
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import sys

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    print("Warning: google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("gdelt_bigquery_fetcher")


class GDELTBigQueryFetcher:
    """Fetch GDELT data using Google BigQuery"""

    # GDELT BigQuery table
    BIGQUERY_TABLE = "gdelt-bq.gdeltv2.events"

    def __init__(
        self,
        output_dir: str = "data/raw",
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize BigQuery GDELT fetcher

        Args:
            output_dir: Directory to save data
            credentials_path: Path to GCP service account JSON (optional, uses default credentials if None)
            project_id: GCP project ID (optional, uses default if None)
        """
        if not BIGQUERY_AVAILABLE:
            raise ImportError("google-cloud-bigquery is required. Install with: pip install google-cloud-bigquery")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize BigQuery client
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = bigquery.Client(credentials=credentials, project=project_id)
        else:
            # Use default credentials (requires: gcloud auth application-default login)
            self.client = bigquery.Client(project=project_id)

        logger.info(f"BigQuery GDELT Fetcher initialized. Output: {self.output_dir}")
        logger.info(f"Project: {self.client.project}")

    def build_query(
        self,
        start_date: str,
        end_date: str,
        countries: List[str],
        actor_keywords: List[str],
        limit: Optional[int] = None
    ) -> str:
        """
        Build SQL query for GDELT data

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            countries: List of country codes
            actor_keywords: Actor keywords to filter
            limit: Optional row limit for testing

        Returns:
            SQL query string
        """
        # Convert dates to YYYYMMDD format
        start_date_int = int(datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d"))
        end_date_int = int(datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d"))

        # Build country filter
        country_filter = " OR ".join([
            f"Actor1CountryCode = '{c}'" for c in countries
        ] + [
            f"Actor2CountryCode = '{c}'" for c in countries
        ] + [
            f"ActionGeo_CountryCode = '{c}'" for c in countries
        ])

        # Build actor keyword filter
        keyword_conditions = []
        for keyword in actor_keywords:
            keyword_conditions.append(f"UPPER(Actor1Name) LIKE '%{keyword.upper()}%'")
            keyword_conditions.append(f"UPPER(Actor2Name) LIKE '%{keyword.upper()}%'")
        keyword_filter = " OR ".join(keyword_conditions)

        # Complete query
        query = f"""
        SELECT
            GLOBALEVENTID,
            SQLDATE,
            MonthYear,
            Year,
            FractionDate,
            Actor1Code,
            Actor1Name,
            Actor1CountryCode,
            Actor1KnownGroupCode,
            Actor1EthnicCode,
            Actor1Religion1Code,
            Actor1Religion2Code,
            Actor1Type1Code,
            Actor1Type2Code,
            Actor1Type3Code,
            Actor2Code,
            Actor2Name,
            Actor2CountryCode,
            Actor2KnownGroupCode,
            Actor2EthnicCode,
            Actor2Religion1Code,
            Actor2Religion2Code,
            Actor2Type1Code,
            Actor2Type2Code,
            Actor2Type3Code,
            IsRootEvent,
            EventCode,
            EventBaseCode,
            EventRootCode,
            QuadClass,
            GoldsteinScale,
            NumMentions,
            NumSources,
            NumArticles,
            AvgTone,
            Actor1Geo_Type,
            Actor1Geo_FullName,
            Actor1Geo_CountryCode,
            Actor1Geo_ADM1Code,
            Actor1Geo_Lat,
            Actor1Geo_Long,
            Actor1Geo_FeatureID,
            Actor2Geo_Type,
            Actor2Geo_FullName,
            Actor2Geo_CountryCode,
            Actor2Geo_ADM1Code,
            Actor2Geo_Lat,
            Actor2Geo_Long,
            Actor2Geo_FeatureID,
            ActionGeo_Type,
            ActionGeo_FullName,
            ActionGeo_CountryCode,
            ActionGeo_ADM1Code,
            ActionGeo_Lat,
            ActionGeo_Long,
            ActionGeo_FeatureID,
            DATEADDED,
            SOURCEURL
        FROM
            `{self.BIGQUERY_TABLE}`
        WHERE
            SQLDATE BETWEEN {start_date_int} AND {end_date_int}
            AND (
                ({country_filter})
                OR ({keyword_filter})
            )
        """

        if limit:
            query += f"\nLIMIT {limit}"

        return query

    def fetch_region_data(
        self,
        region_name: str,
        start_date: str,
        end_date: str,
        countries: List[str],
        actor_keywords: List[str],
        batch_months: int = 3,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch GDELT data for a region using BigQuery

        Args:
            region_name: Name of region
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            countries: List of country codes
            actor_keywords: Actor keywords to filter
            batch_months: Fetch data in batches of N months (to avoid timeout)
            limit: Optional row limit for testing

        Returns:
            DataFrame with GDELT events
        """
        logger.info(f"Fetching data for {region_name} from BigQuery")
        logger.info(f"Date range: {start_date} to {end_date}")

        # If limit is specified, just run one query
        if limit:
            logger.info(f"Running test query with limit {limit}")
            query = self.build_query(start_date, end_date, countries, actor_keywords, limit)
            df = self.client.query(query).to_dataframe()
            logger.info(f"Fetched {len(df)} events (test mode)")
            return df

        # Otherwise, fetch in batches to avoid timeout
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = []
        current = start

        with tqdm(desc=f"Fetching {region_name}", unit="batch") as pbar:
            while current < end:
                # Calculate batch end date
                batch_end = min(
                    datetime(current.year + (current.month + batch_months - 1) // 12,
                            (current.month + batch_months - 1) % 12 + 1, 1),
                    end
                )

                batch_start_str = current.strftime("%Y-%m-%d")
                batch_end_str = batch_end.strftime("%Y-%m-%d")

                logger.info(f"Fetching batch: {batch_start_str} to {batch_end_str}")

                # Build and run query
                query = self.build_query(
                    batch_start_str, batch_end_str,
                    countries, actor_keywords
                )

                try:
                    df = self.client.query(query).to_dataframe()
                    logger.info(f"  Retrieved {len(df)} events")

                    if len(df) > 0:
                        all_data.append(df)

                except Exception as e:
                    logger.error(f"Error fetching batch {batch_start_str} to {batch_end_str}: {e}")

                # Move to next batch
                current = batch_end
                pbar.update(1)

        if not all_data:
            logger.warning(f"No data found for {region_name}")
            return pd.DataFrame()

        # Combine all batches
        combined = pd.concat(all_data, ignore_index=True)

        # Remove duplicates
        combined = combined.drop_duplicates(subset=['GLOBALEVENTID'])

        logger.info(f"Total events fetched: {len(combined)}")

        # Save raw data
        output_file = self.output_dir / f"{region_name}_raw.parquet"
        combined.to_parquet(output_file, index=False)
        logger.info(f"Saved to {output_file}")

        return combined

    def fetch_all_regions(
        self,
        config: Dict,
        limit: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all regions in config

        Args:
            config: Configuration dictionary
            limit: Optional row limit for testing

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
                actor_keywords=region_config['actor_keywords'],
                limit=limit
            )

            results[region_key] = df

        return results


if __name__ == "__main__":
    from src.utils.config_loader import load_config
    import argparse

    parser = argparse.ArgumentParser(description="Fetch GDELT data from BigQuery")
    parser.add_argument('--credentials', type=str, help='Path to GCP service account JSON')
    parser.add_argument('--project', type=str, help='GCP project ID')
    parser.add_argument('--limit', type=int, help='Limit rows for testing (e.g., 1000)')
    parser.add_argument('--regions', nargs='+', help='Specific regions to fetch')
    args = parser.parse_args()

    config = load_config()

    # Filter regions if specified
    if args.regions:
        config['regions'] = {k: v for k, v in config['regions'].items() if k in args.regions}

    # Get credentials from args or config
    credentials_path = args.credentials or config.get('data_collection', {}).get('bigquery', {}).get('credentials_path')
    project_id = args.project or config.get('data_collection', {}).get('bigquery', {}).get('project_id')

    # Create fetcher
    try:
        fetcher = GDELTBigQueryFetcher(
            credentials_path=credentials_path,
            project_id=project_id
        )

        # Fetch data
        data = fetcher.fetch_all_regions(config, limit=args.limit)

        # Print summary
        print("\n" + "="*50)
        print("FETCH COMPLETE - SUMMARY")
        print("="*50)
        for region, df in data.items():
            print(f"\n{region}: {len(df)} events")
            if len(df) > 0:
                print(f"  Date range: {df['SQLDATE'].min()} to {df['SQLDATE'].max()}")

    except Exception as e:
        logger.error(f"Failed to initialize BigQuery: {e}")
        print("\nTo use BigQuery:")
        print("1. Install: pip install google-cloud-bigquery")
        print("2. Setup GCP credentials:")
        print("   Option A: gcloud auth application-default login")
        print("   Option B: Use service account JSON with --credentials flag")
        print("3. Make sure you have a GCP project with BigQuery API enabled")
