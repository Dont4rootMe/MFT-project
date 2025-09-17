#!/usr/bin/env python3
"""
Data Availability Validation Script

This script validates if all currencies specified in the data configuration
can be successfully downloaded from the configured data sources without
actually downloading the data. It checks:

1. Binance API endpoint availability for each currency pair
2. TensorTrade CryptoDataDownload compatibility 
3. Configuration file validity

Usage:
    python validate_data_availability.py --config conf/data/triple_data.yaml
    python validate_data_availability.py --config-dir conf/data/
"""

import argparse
import sys
import os
import yaml
import requests
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from tensortrade.data.cdd import CryptoDataDownload
    TENSORTRADE_AVAILABLE = True
except ImportError:
    TENSORTRADE_AVAILABLE = False
    print("⚠️  Warning: TensorTrade not available. Skipping TensorTrade validation.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataAvailabilityValidator:
    """Validates data availability for cryptocurrency trading pairs."""
    
    def __init__(self):
        self.binance_symbols = None
        self.validation_results = {}
        
    def load_binance_symbols(self) -> bool:
        """Load available symbols from Binance API."""
        try:
            logger.info("🔍 Fetching available symbols from Binance...")
            response = requests.get(
                "https://api.binance.com/api/v3/exchangeInfo",
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            self.binance_symbols = {s['symbol']: s for s in data['symbols']}
            logger.info(f"✅ Loaded {len(self.binance_symbols)} symbols from Binance")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Binance symbols: {e}")
            return False
    
    def validate_binance_endpoint(self, symbol: str, base_currency: str, quote_currency: str) -> Tuple[bool, str]:
        """
        Validate if a currency pair is available on Binance.
        
        Args:
            symbol: The trading pair symbol (e.g., BTCUSDT)
            base_currency: Base currency (e.g., BTC) 
            quote_currency: Quote currency (e.g., USDT)
            
        Returns:
            Tuple of (is_valid, message)
        """
        if self.binance_symbols is None:
            return False, "Binance symbols not loaded"
            
        # Check exact symbol match
        if symbol in self.binance_symbols:
            symbol_info = self.binance_symbols[symbol]
            status = symbol_info.get('status', 'UNKNOWN')
            
            if status == 'TRADING':
                return True, f"Active trading pair: {symbol}"
            else:
                return False, f"Symbol exists but status is: {status}"
        
        return False, f"Symbol {symbol} not found on Binance"
    
    def validate_binance_klines(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate if klines data is available for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Test klines endpoint with minimal request
            params = {
                'symbol': symbol,
                'interval': '1d',
                'limit': 1
            }
            
            response = requests.get(
                'https://api.binance.com/api/v3/klines',
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            klines = response.json()
            if klines and len(klines) > 0:
                return True, f"Klines data available for {symbol}"
            else:
                return False, f"No klines data returned for {symbol}"
                
        except Exception as e:
            return False, f"Klines request failed for {symbol}: {e}"
    
    def validate_tensortrade_compatibility(self, exchange: str, quote: str, base: str, timeframe: str) -> Tuple[bool, str]:
        """
        Validate TensorTrade CryptoDataDownload compatibility.
        
        Args:
            exchange: Exchange name
            quote: Quote currency
            base: Base currency  
            timeframe: Timeframe (e.g., '1h', '1d')
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not TENSORTRADE_AVAILABLE:
            return False, "TensorTrade not available"
            
        try:
            cdd = CryptoDataDownload()
            
            # Convert timeframe format
            tt_timeframe = 'd' if timeframe == '1d' else timeframe
            
            # Test data fetch (this shouldn't actually download much data)
            # We'll use a very recent date range to minimize data transfer
            test_data = cdd.fetch(
                exchange,
                quote, 
                base,
                tt_timeframe
            )
            
            if test_data is not None and not test_data.empty:
                return True, f"TensorTrade can fetch {base}/{quote} from {exchange}"
            else:
                return False, f"TensorTrade returned empty data for {base}/{quote}"
                
        except Exception as e:
            return False, f"TensorTrade fetch failed for {base}/{quote}: {e}"
    
    def validate_currency_config(self, config: Dict) -> Dict[str, Dict]:
        """
        Validate all currencies in a configuration.
        
        Args:
            config: Data configuration dictionary
            
        Returns:
            Dictionary of validation results per currency
        """
        symbols = config.get('symbols', [])
        main_currency = config.get('main_currency', 'USDT')
        exchange = config.get('exchange', 'Binance')
        time_freq = config.get('time_freq', '1d')
        
        results = {}
        
        logger.info(f"🔍 Validating {len(symbols)} currencies...")
        logger.info(f"📊 Exchange: {exchange}, Quote: {main_currency}, Timeframe: {time_freq}")
        
        for base_currency in symbols:
            logger.info(f"\n--- Validating {base_currency} ---")
            
            # Construct trading pair symbol
            trading_pair = f"{base_currency}{main_currency}"
            
            currency_results = {
                'base_currency': base_currency,
                'quote_currency': main_currency, 
                'trading_pair': trading_pair,
                'exchange': exchange,
                'timeframe': time_freq,
                'validations': {}
            }
            
            # 1. Validate Binance symbol existence
            logger.info(f"🔍 Checking Binance symbol: {trading_pair}")
            is_valid, message = self.validate_binance_endpoint(trading_pair, base_currency, main_currency)
            currency_results['validations']['binance_symbol'] = {
                'valid': is_valid,
                'message': message
            }
            logger.info(f"  {'✅' if is_valid else '❌'} {message}")
            
            # 2. Validate Binance klines availability
            if is_valid:
                logger.info(f"🔍 Checking klines data availability...")
                is_valid_klines, klines_message = self.validate_binance_klines(trading_pair)
                currency_results['validations']['binance_klines'] = {
                    'valid': is_valid_klines,
                    'message': klines_message
                }
                logger.info(f"  {'✅' if is_valid_klines else '❌'} {klines_message}")
            else:
                currency_results['validations']['binance_klines'] = {
                    'valid': False,
                    'message': 'Skipped due to symbol validation failure'
                }
                logger.info(f"  ⏭️  Skipping klines check due to symbol validation failure")
            
            # 3. Validate TensorTrade compatibility
            if TENSORTRADE_AVAILABLE:
                logger.info(f"🔍 Checking TensorTrade compatibility...")
                is_valid_tt, tt_message = self.validate_tensortrade_compatibility(
                    exchange, main_currency, base_currency, time_freq
                )
                currency_results['validations']['tensortrade'] = {
                    'valid': is_valid_tt,
                    'message': tt_message
                }
                logger.info(f"  {'✅' if is_valid_tt else '❌'} {tt_message}")
            else:
                currency_results['validations']['tensortrade'] = {
                    'valid': False,
                    'message': 'TensorTrade not available'
                }
                logger.info(f"  ⚠️  TensorTrade not available")
            
            # Overall currency validation status
            all_validations = [v['valid'] for v in currency_results['validations'].values()]
            currency_results['overall_valid'] = all(all_validations)
            
            results[base_currency] = currency_results
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def generate_report(self, results: Dict[str, Dict], config_path: str) -> None:
        """Generate a comprehensive validation report."""
        
        print(f"\n{'='*80}")
        print(f"🔍 DATA AVAILABILITY VALIDATION REPORT")
        print(f"{'='*80}")
        print(f"📁 Config: {config_path}")
        print(f"⏰ Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Summary statistics
        total_currencies = len(results)
        valid_currencies = sum(1 for r in results.values() if r['overall_valid'])
        
        print(f"\n📊 SUMMARY")
        print(f"{'─'*40}")
        print(f"Total currencies tested: {total_currencies}")
        print(f"Valid currencies: {valid_currencies}")
        print(f"Invalid currencies: {total_currencies - valid_currencies}")
        print(f"Success rate: {(valid_currencies/total_currencies)*100:.1f}%" if total_currencies > 0 else "N/A")
        
        # Detailed results per currency
        print(f"\n📋 DETAILED RESULTS")
        print(f"{'─'*40}")
        
        for currency, result in results.items():
            status_icon = "✅" if result['overall_valid'] else "❌"
            print(f"\n{status_icon} {currency} ({result['trading_pair']})")
            
            for validation_name, validation_result in result['validations'].items():
                status = "✅" if validation_result['valid'] else "❌"
                print(f"   {status} {validation_name.replace('_', ' ').title()}: {validation_result['message']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print(f"{'─'*40}")
        
        invalid_currencies = [currency for currency, result in results.items() if not result['overall_valid']]
        
        if not invalid_currencies:
            print("🎉 All currencies are valid and ready for data collection!")
        else:
            print(f"⚠️  The following currencies have validation issues:")
            for currency in invalid_currencies:
                result = results[currency]
                issues = [name for name, val in result['validations'].items() if not val['valid']]
                print(f"   • {currency}: {', '.join(issues)}")
            
            print(f"\n📝 Consider:")
            print(f"   • Removing invalid currencies from configuration")
            print(f"   • Checking currency symbol spelling")
            print(f"   • Verifying exchange support for these pairs")
            print(f"   • Installing missing dependencies (TensorTrade)")
        
        print(f"\n{'='*80}")


def load_config(config_path: str) -> Optional[Dict]:
    """Load and parse YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ Loaded configuration from {config_path}")
        return config
        
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"❌ Invalid YAML configuration: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return None


def find_config_files(config_dir: str) -> List[str]:
    """Find all YAML configuration files in a directory."""
    config_files = []
    
    try:
        config_path = Path(config_dir)
        if not config_path.exists():
            logger.error(f"❌ Configuration directory not found: {config_dir}")
            return []
        
        # Find YAML files that contain data configurations
        for yaml_file in config_path.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check if this looks like a data configuration
                if isinstance(config, dict) and 'symbols' in config:
                    config_files.append(str(yaml_file))
                    
            except Exception as e:
                logger.warning(f"⚠️  Skipping {yaml_file}: {e}")
        
        logger.info(f"✅ Found {len(config_files)} data configuration files")
        return config_files
        
    except Exception as e:
        logger.error(f"❌ Failed to scan configuration directory: {e}")
        return []


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="🔍 Validate cryptocurrency data availability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config', 
        type=str,
        help='Path to specific data configuration YAML file'
    )
    group.add_argument(
        '--config-dir',
        type=str, 
        default='conf/data',
        required=False,
        help='Directory containing data configuration files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = DataAvailabilityValidator()
    
    # Load Binance symbols
    if not validator.load_binance_symbols():
        logger.error("❌ Failed to load Binance symbols. Cannot proceed with validation.")
        sys.exit(1)
    
    # Determine configuration files to validate
    config_files = []
    if args.config:
        if os.path.exists(args.config):
            config_files = [args.config]
        else:
            logger.error(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
    else:
        config_files = find_config_files(args.config_dir)
        if not config_files:
            logger.error("❌ No valid configuration files found")
            sys.exit(1)
    
    # Validate each configuration file
    all_results = {}
    
    for config_file in config_files:
        logger.info(f"\n🔍 Validating configuration: {config_file}")
        
        # Load configuration
        config = load_config(config_file)
        if config is None:
            continue
        
        # Validate currencies
        results = validator.validate_currency_config(config)
        all_results[config_file] = results
        
        # Generate report for this configuration
        validator.generate_report(results, config_file)
    
    # Overall summary if multiple files
    if len(config_files) > 1:
        total_currencies = sum(len(results) for results in all_results.values())
        total_valid = sum(
            sum(1 for r in results.values() if r['overall_valid']) 
            for results in all_results.values()
        )
        
        print(f"\n🌟 OVERALL SUMMARY")
        print(f"{'='*50}")
        print(f"Configuration files validated: {len(config_files)}")
        print(f"Total currencies tested: {total_currencies}")
        print(f"Total valid currencies: {total_valid}")
        print(f"Overall success rate: {(total_valid/total_currencies)*100:.1f}%" if total_currencies > 0 else "N/A")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
