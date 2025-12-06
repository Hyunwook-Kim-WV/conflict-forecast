"""Configuration loader utility"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_region_config(region_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get configuration for specific region

    Args:
        region_name: Name of region (e.g., 'israel_palestine')
        config: Full configuration dictionary

    Returns:
        Region-specific configuration
    """
    if region_name not in config['regions']:
        raise ValueError(f"Region {region_name} not found in config")

    return config['regions'][region_name]
