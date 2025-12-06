"""
Create ground truth labels from Wikipedia conflict timelines
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("ground_truth")


class GroundTruthCreator:
    """Create ground truth labels for conflicts based on Wikipedia data"""

    # Major conflict events from Wikipedia timelines
    CONFLICT_EVENTS = {
        'israel_palestine': [
            # Gaza Wars and Major Operations
            ('2008-12-27', '2009-01-18', 'Operation Cast Lead'),
            ('2012-11-14', '2012-11-21', 'Operation Pillar of Defense'),
            ('2014-07-08', '2014-08-26', 'Operation Protective Edge'),
            ('2021-05-10', '2021-05-21', 'Israel-Gaza Crisis 2021'),
            ('2023-10-07', '2024-12-31', 'Israel-Hamas War 2023'),
            # Major escalations
            ('2018-03-30', '2019-12-27', 'Gaza Border Protests'),
            ('2022-08-05', '2022-08-07', 'Operation Breaking Dawn'),
        ],

        'russia_ukraine': [
            # Crimea and Donbas
            ('2014-02-20', '2014-03-18', 'Crimea Annexation'),
            ('2014-04-06', '2015-02-15', 'War in Donbas - Initial Phase'),
            ('2015-02-15', '2022-02-24', 'Donbas War - Low Intensity'),
            # Full-scale invasion
            ('2022-02-24', '2024-12-31', 'Russian Invasion of Ukraine 2022'),
            # Major battles
            ('2022-02-24', '2022-04-07', 'Battle of Kyiv'),
            ('2022-02-24', '2022-05-20', 'Siege of Mariupol'),
            ('2022-08-29', '2022-11-11', 'Ukrainian Counteroffensives'),
        ],

        'india_pakistan': [
            # Kargil War aftermath and major incidents
            ('2001-12-13', '2002-10-16', 'Parliament Attack & Military Standoff'),
            ('2008-11-26', '2008-11-29', 'Mumbai Attacks'),
            ('2016-09-18', '2016-09-29', 'Uri Attack & Surgical Strikes'),
            ('2019-02-14', '2019-03-01', 'Pulwama Attack & Balakot Airstrike'),
            # Border skirmishes
            ('2016-11-01', '2017-01-31', 'LoC Ceasefire Violations 2016-17'),
            ('2019-08-05', '2019-12-31', 'Kashmir Lockdown & Tensions'),
            ('2020-01-01', '2020-06-30', 'Border Tensions 2020'),
            ('2021-02-25', '2021-02-25', 'LoC Ceasefire Agreement'),
        ]
    }

    def __init__(self, output_dir: str = "data/ground_truth"):
        """
        Initialize ground truth creator

        Args:
            output_dir: Directory to save ground truth labels
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ground Truth Creator initialized. Output: {self.output_dir}")

    def create_labels(
        self,
        region_name: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Create binary labels for conflict/non-conflict periods

        Args:
            region_name: Name of region
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date and conflict label
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Create date range
        dates = pd.date_range(start, end, freq='D')
        labels = pd.DataFrame({
            'date': dates,
            'is_conflict': 0,
            'conflict_name': ''
        })

        # Mark conflict periods
        if region_name not in self.CONFLICT_EVENTS:
            logger.warning(f"No conflict events defined for {region_name}")
            return labels

        for conflict_start, conflict_end, conflict_name in self.CONFLICT_EVENTS[region_name]:
            conflict_start_dt = datetime.strptime(conflict_start, "%Y-%m-%d")
            conflict_end_dt = datetime.strptime(conflict_end, "%Y-%m-%d")

            mask = (labels['date'] >= conflict_start_dt) & (labels['date'] <= conflict_end_dt)
            labels.loc[mask, 'is_conflict'] = 1
            labels.loc[mask, 'conflict_name'] = conflict_name

        # Statistics
        conflict_days = labels['is_conflict'].sum()
        total_days = len(labels)
        conflict_pct = (conflict_days / total_days) * 100

        logger.info(f"{region_name}:")
        logger.info(f"  Total days: {total_days}")
        logger.info(f"  Conflict days: {conflict_days} ({conflict_pct:.2f}%)")
        logger.info(f"  Normal days: {total_days - conflict_days} ({100-conflict_pct:.2f}%)")

        # List conflicts
        conflicts = labels[labels['is_conflict'] == 1].groupby('conflict_name').size()
        logger.info(f"  Conflicts: {len(conflicts)}")
        for conflict, days in conflicts.items():
            logger.info(f"    - {conflict}: {days} days")

        # Save
        output_file = self.output_dir / f"{region_name}_labels.csv"
        labels.to_csv(output_file, index=False)
        logger.info(f"Saved labels to {output_file}")

        return labels

    def create_labels_from_ucdp(
        self,
        region_name: str,
        start_date: str,
        end_date: str,
        ucdp_file: Path = None
    ) -> pd.DataFrame:
        """
        Create labels dynamically from UCDP data
        
        Args:
            region_name: Name of region
            start_date: Start date
            end_date: End date
            ucdp_file: Path to UCDP CSV file (optional, defaults to standard path)
            
        Returns:
            DataFrame with date, is_conflict, and target_fatalities
        """
        if ucdp_file is None:
            ucdp_file = self.output_dir / f"ucdp_{region_name}.csv"
            
        if not ucdp_file.exists():
            logger.warning(f"UCDP file not found: {ucdp_file}. Falling back to hardcoded events.")
            return self.create_labels(region_name, start_date, end_date)
            
        # Load UCDP data
        try:
            ucdp_df = pd.read_csv(ucdp_file)
        except Exception as e:
            logger.error(f"Failed to read UCDP file: {e}")
            return self.create_labels(region_name, start_date, end_date)
            
        # Parse dates
        ucdp_df['date_start'] = pd.to_datetime(ucdp_df['date_start'])
        
        # Create full date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start, end, freq='D')
        
        labels = pd.DataFrame({'date': dates})
        
        # Aggregate fatalities by day
        # UCDP events have date_start and date_end. For simplicity, we assign fatalities to date_start
        # or distribute them? Most events are 1 day. Let's use date_start.
        daily_fatalities = ucdp_df.groupby('date_start')['best'].sum().reindex(dates, fill_value=0)
        
        labels['target_fatalities'] = daily_fatalities.values
        
        # Create binary label (1 if fatalities > 0)
        # We can also use a rolling window to smooth "conflict periods"
        # For now, strictly daily:
        labels['is_conflict'] = (labels['target_fatalities'] > 0).astype(int)
        labels['conflict_name'] = 'UCDP Event' # Placeholder
        
        # Statistics
        conflict_days = labels['is_conflict'].sum()
        total_days = len(labels)
        total_fatalities = labels['target_fatalities'].sum()
        
        logger.info(f"{region_name} (UCDP Source):")
        logger.info(f"  Total days: {total_days}")
        logger.info(f"  Conflict days: {conflict_days} ({(conflict_days/total_days)*100:.2f}%)")
        logger.info(f"  Total Fatalities: {total_fatalities}")
        
        # Save
        output_file = self.output_dir / f"{region_name}_labels.csv"
        labels.to_csv(output_file, index=False)
        logger.info(f"Saved UCDP-based labels to {output_file}")
        
        return labels

    def create_all_labels(self, config: Dict) -> Dict[str, pd.DataFrame]:
        """
        Create labels for all regions in config

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary mapping region names to label DataFrames
        """
        results = {}

        for region_key, region_config in config['regions'].items():
            logger.info(f"Creating labels for {region_config['name']}")

            labels = self.create_labels(
                region_name=region_key,
                start_date=region_config['date_range']['start'],
                end_date=region_config['date_range']['end']
            )

            results[region_key] = labels

        return results

    def get_normal_periods(
        self,
        labels: pd.DataFrame,
        min_period_days: int = 90
    ) -> List[Tuple[datetime, datetime]]:
        """
        Extract continuous normal (non-conflict) periods

        Args:
            labels: DataFrame with conflict labels
            min_period_days: Minimum length for normal period

        Returns:
            List of (start_date, end_date) tuples for normal periods
        """
        normal_periods = []

        # Find continuous sequences of normal days
        labels['is_normal'] = labels['is_conflict'] == 0
        labels['period_id'] = (labels['is_normal'] != labels['is_normal'].shift()).cumsum()

        for period_id, group in labels.groupby('period_id'):
            if group['is_normal'].iloc[0] and len(group) >= min_period_days:
                start = group['date'].min()
                end = group['date'].max()
                normal_periods.append((start, end))
                logger.info(f"  Normal period: {start.date()} to {end.date()} ({len(group)} days)")

        return normal_periods


if __name__ == "__main__":
    from src.utils.config_loader import load_config

    config = load_config()
    creator = GroundTruthCreator()

    # Create labels for all regions
    labels = creator.create_all_labels(config)

    # Show normal periods
    print("\n" + "="*50)
    print("NORMAL PERIODS FOR TRAINING")
    print("="*50)
    for region, label_df in labels.items():
        print(f"\n{region.upper()}:")
        normal_periods = creator.get_normal_periods(
            label_df,
            min_period_days=config['normal_state']['min_normal_period_days']
        )
        print(f"  Found {len(normal_periods)} normal periods")
