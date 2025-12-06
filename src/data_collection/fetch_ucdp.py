"""
Fetch conflict data from UCDP (Uppsala Conflict Data Program) API.
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("fetch_ucdp")

class UCDPFetcher:
    """Fetch conflict data from UCDP API"""
    
    BASE_URL = "https://ucdpapi.pcr.uu.se/api/gedevents/25.1"
    
    # Mapping from our region names to UCDP Country IDs
    # IDs can be found at: https://ucdp.uu.se/downloads/
    REGION_MAPPING = {
        'israel_palestine': [666, 667],  # Israel, Palestine (if separate) - checking UCDP codes
        'russia_ukraine': [365, 369],    # Russia, Ukraine
        'india_pakistan': [750, 770],    # India, Pakistan
        'china_taiwan': [710, 713],      # China, Taiwan
        'koreas': [731, 732],            # North Korea, South Korea
    }

    def __init__(self, output_dir: str = "data/ground_truth"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_events(self, start_date: str, end_date: str, country_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch events from UCDP API with pagination
        """
        all_events = []
        page = 1
        page_size = 1000
        
        params = {
            'pagesize': page_size,
            'StartDate': start_date,
            'EndDate': end_date,
        }
        
        if country_ids:
            # UCDP API expects comma-separated string for multiple IDs? 
            # Or we might need to filter client-side if API doesn't support multiple country filters easily in one go.
            # Let's try fetching all for the date range and filtering, or iterate.
            # Actually, 'country_id' parameter accepts comma-separated list.
            params['country_id'] = ','.join(map(str, country_ids))

        logger.info(f"Fetching UCDP data for countries {country_ids} from {start_date} to {end_date}...")

        while True:
            params['page'] = page
            try:
                response = requests.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'Result' not in data:
                    break
                    
                events = data['Result']
                if not events:
                    break
                    
                all_events.extend(events)
                logger.info(f"Fetched page {page} ({len(events)} events)")
                
                if len(events) < page_size:
                    break
                    
                page += 1
                time.sleep(0.5) # Be nice to the API
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break
        
        if not all_events:
            logger.warning("No events found.")
            return pd.DataFrame()

        df = pd.DataFrame(all_events)
        logger.info(f"Total events fetched: {len(df)}")
        return df

    def process_and_save(self, df: pd.DataFrame, region_name: str):
        """
        Process raw UCDP data and save to CSV
        """
        if df.empty:
            return

        # Select relevant columns
        cols = [
            'id', 'year', 'date_start', 'date_end', 'type_of_violence',
            'conflict_new_id', 'conflict_name', 'dyad_new_id', 'dyad_name',
            'side_a', 'side_b', 'best', 'high', 'low', # Fatalities
            'country', 'region', 'latitude', 'longitude'
        ]
        
        # Filter for columns that exist
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]
        
        # Convert dates
        df['date_start'] = pd.to_datetime(df['date_start'])
        df['date_end'] = pd.to_datetime(df['date_end'])
        
        # Save
        output_file = self.output_dir / f"ucdp_{region_name}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")

    def run_for_region(self, region_name: str, start_date: str, end_date: str, ucdp_ids: List[int]):
        """
        Run fetcher for a specific region
        """
        logger.info(f"Processing region: {region_name} (UCDP IDs: {ucdp_ids})")
        df = self.fetch_events(start_date, end_date, ucdp_ids)
        self.process_and_save(df, region_name)

if __name__ == "__main__":
    from src.utils.config_loader import load_config
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch UCDP conflict data")
    parser.add_argument('--regions', nargs='+', help='Specific regions to fetch')
    args = parser.parse_args()
    
    config = load_config()
    fetcher = UCDPFetcher()
    
    # UCDP ID Mapping (Still needed as it's not in config yet, or we should move it to config)
    # For now, we keep the mapping here but use config for dates and iteration
    # Ideally, we should add ucdp_ids to config.yaml, but let's stick to the mapping for now
    # and just use the keys from config.
    
    regions_to_process = args.regions if args.regions else config['regions'].keys()
    
    for region_name in regions_to_process:
        if region_name not in config['regions']:
            logger.warning(f"Region {region_name} not found in config, skipping")
            continue
            
        region_config = config['regions'][region_name]
        
        # Check if we have UCDP mapping
        if region_name not in fetcher.REGION_MAPPING:
            logger.warning(f"No UCDP ID mapping for {region_name}, skipping")
            continue
            
        ucdp_ids = fetcher.REGION_MAPPING[region_name]
        start_date = region_config['date_range']['start']
        end_date = region_config['date_range']['end']
        
        fetcher.run_for_region(region_name, start_date, end_date, ucdp_ids)
