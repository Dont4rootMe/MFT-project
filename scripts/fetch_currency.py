"""
Fetching history of specified currency via BINANCE API
"""

import argparse
import csv
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import os

def list_currencies(filter_str=None):
    """List available cryptocurrency trading pairs on Binance."""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        response.raise_for_status()
        symbols = [s['symbol'] for s in response.json()['symbols']]
        if filter_str:
            symbols = [s for s in symbols if filter_str.lower() in s.lower()]
        return symbols
    except Exception as e:
        print(f"🚨 Error fetching currencies: {e}")
        return []

def fetch_historical_data(currency, interval, days, output):
    """
    Fetch historical klines data and save to CSV.
    
    Args:
        currency (str): Trading pair symbol (e.g., BTCUSDT)
        interval (str): Kline interval (e.g., 1d, 1h)
        days (int): Number of days of historical data (0 = all available)
        output (str): Output CSV file path
    """
    end_time = int(time.time() * 1000)  # Current time in milliseconds
    
    # Calculate start time based on days
    if days <= 0:
        print("🌐 Fetching ALL available historical data...")
        start_time = 0  # Beginning of time for Binance
    else:
        days_in_ms = days * 24 * 60 * 60 * 1000
        start_time = end_time - days_in_ms
        print(f"📅 Fetching {days} days of data ({datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(end_time/1000).strftime('%Y-%m-%d')})")

    all_klines = []
    current_start = start_time
    request_count = 0
    max_requests = 500  # Safety limit to prevent infinite loops

    print("⏳ Downloading data...", end="", flush=True)
    
    while current_start < end_time and request_count < max_requests:
        params = {
            'symbol': currency,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': 1000
        }
        
        try:
            response = requests.get('https://api.binance.com/api/v3/klines', params=params)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
                
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1  # Next start time is last kline's open time + 1ms
            request_count += 1
            print(".", end="", flush=True)  # Progress indicator
            time.sleep(0.1)  # Avoid rate limiting
            
            # Check if we've reached current time
            if current_start > end_time:
                break
                
        except Exception as e:
            print(f"\n🚨 Request failed: {e}")
            break

    print()  # New line after progress dots
    
    if not all_klines:
        print("❌ No data fetched")
        return

    # Extract actual time range of fetched data
    first_ts = all_klines[0][0] / 1000
    last_ts = all_klines[-1][0] / 1000
    actual_days = (last_ts - first_ts) / (24 * 3600)
    
    print(f"📊 Fetched {len(all_klines)} records covering {actual_days:.1f} days")

    header = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    # check validity of dump file name
    file_name = output.split('/')[-1]
    if '.' not in file_name or file_name.split('.')[-1] != 'csv':
        output = os.path.join(*output.split('/')[:-1], file_name.split('.')[0] + '.csv')

    try:
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_klines)
        print(f"💾 Data saved to {output}")
    except Exception as e:
        print(f"🚨 CSV write failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="📊 Fetch cryptocurrency data from Binance API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 'list' command
    list_parser = subparsers.add_parser('list', help='List available currency pairs')
    list_parser.add_argument('--filter', type=str, default='', 
                            help='Filter currency pairs by string (case-insensitive)')

    # 'fetch' command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch historical data for a currency')
    fetch_parser.add_argument('--currency', type=str, help='Currency pair symbol (e.g., BTCUSDT)')
    fetch_parser.add_argument('--output', type=str, help='Output CSV file path')
    fetch_parser.add_argument('--interval', type=str, 
                             help='Kline interval (e.g., 1m, 5m, 1h, 1d)')
    fetch_parser.add_argument('--days', type=int, default=0,
                             help='Number of days of historical data (default 0 = all available)')

    args = parser.parse_args()

    if args.command == 'list':
        currencies = list_currencies(args.filter)
        print(f"\n🔎 Found {len(currencies)} currencies matching '{args.filter}':")
        for c in currencies:
            print(f"  - {c}")
        print()

    elif args.command == 'fetch':
        print(f"\n📥 Fetching {args.currency} data ({args.interval} intervals)...")
        fetch_historical_data(
            currency=args.currency,
            interval=args.interval,
            days=args.days,
            output=args.output
        )

if __name__ == '__main__':
    main()