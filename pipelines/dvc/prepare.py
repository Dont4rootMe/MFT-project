"""
DVC Pipeline Stage: Data Preparation

This script fetches raw cryptocurrency data and prepares it for training.
It downloads OHLCV data from exchanges and saves it in a format ready for 
feature engineering.
"""

import sys
from pathlib import Path
import argparse
import yaml
import os

# Ensure project root is on the path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from scripts.fetch_currency import fetch_historical_data, list_currencies


def prepare_data(config_path: str, output_dir: str) -> None:
    """
    Prepare raw data based on configuration.
    
    Args:
        config_path: Path to the data configuration YAML file
        output_dir: Directory to save raw data files
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config.get('symbols', ['LINK'])
    time_freq = config.get('time_freq', '1d')
    main_currency = config.get('main_currency', 'USDT')
    days = config.get('days', 1000)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Map time_freq to Binance interval format
    interval_map = {
        '1m': '1m',
        '5m': '5m', 
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
    }
    interval = interval_map.get(time_freq, '1d')
    
    print(f"📊 Preparing data for {len(symbols)} symbols...")
    print(f"   Interval: {interval}")
    print(f"   Days: {days}")
    print(f"   Output: {output_dir}")
    
    for symbol in symbols:
        currency_pair = f"{symbol}{main_currency}"
        output_file = os.path.join(output_dir, f"{symbol}_{main_currency}_{time_freq}.csv")
        
        print(f"\n📥 Fetching {currency_pair}...")
        
        try:
            fetch_historical_data(
                currency=currency_pair,
                interval=interval,
                days=days,
                output=output_file
            )
            print(f"✓ Saved to {output_file}")
        except Exception as e:
            print(f"✗ Failed to fetch {currency_pair}: {e}")
    
    # Save metadata about the prepared data
    metadata = {
        'symbols': symbols,
        'time_freq': time_freq,
        'main_currency': main_currency,
        'days': days,
        'files': [f"{s}_{main_currency}_{time_freq}.csv" for s in symbols]
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata, f)
    
    print(f"\n✅ Data preparation complete!")
    print(f"   Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare raw cryptocurrency data for training"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='conf/data/dvc_data.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/raw',
        help='Output directory for raw data'
    )
    
    args = parser.parse_args()
    prepare_data(args.config, args.output)


if __name__ == '__main__':
    main()

