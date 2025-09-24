# Data Availability Validation Script

## Overview

The `validate_data_availability.py` script validates whether all cryptocurrency pairs specified in your data configuration files can be successfully downloaded from the configured data sources **without actually downloading the data**. 

This script performs the following validations:

1. **Binance API Endpoint Validation**: Checks if trading pairs exist and are active on Binance
2. **Binance Klines Data Availability**: Verifies that historical klines data can be retrieved 
3. **TensorTrade Compatibility**: Tests if TensorTrade CryptoDataDownload can fetch the data (if available)

## Usage

### Validate a Single Configuration File

```bash
python scripts/validate_data_availability.py --config conf/data/triple_data.yaml
```

### Validate All Configuration Files in a Directory

```bash
python scripts/validate_data_availability.py --config-dir conf/data/
```

### Enable Verbose Logging

```bash
python scripts/validate_data_availability.py --config conf/data/triple_data.yaml --verbose
```

## Sample Output

```
================================================================================
🔍 DATA AVAILABILITY VALIDATION REPORT
================================================================================
📁 Config: conf/data/triple_data.yaml
⏰ Validation Time: 2025-09-17 13:56:55
================================================================================

📊 SUMMARY
────────────────────────────────────────
Total currencies tested: 3
Valid currencies: 3
Invalid currencies: 0
Success rate: 100.0%

📋 DETAILED RESULTS
────────────────────────────────────────

✅ PEPE (PEPEUSDT)
   ✅ Binance Symbol: Active trading pair: PEPEUSDT
   ✅ Binance Klines: Klines data available for PEPEUSDT
   ✅ Tensortrade: TensorTrade can fetch PEPE/USDT from Binance

✅ LINK (LINKUSDT)
   ✅ Binance Symbol: Active trading pair: LINKUSDT
   ✅ Binance Klines: Klines data available for LINKUSDT
   ✅ Tensortrade: TensorTrade can fetch LINK/USDT from Binance

✅ SOL (SOLUSDT)
   ✅ Binance Symbol: Active trading pair: SOLUSDT
   ✅ Binance Klines: Klines data available for SOLUSDT
   ✅ Tensortrade: TensorTrade can fetch SOL/USDT from Binance

💡 RECOMMENDATIONS
────────────────────────────────────────
🎉 All currencies are valid and ready for data collection!
```

## Configuration File Requirements

The script expects YAML configuration files with the following structure:

```yaml
symbols: [PEPE, LINK, SOL]
time_freq: "1h"
exchange: "Binance"
main_currency: "USDT"
```

Required fields:
- `symbols`: List of base currency symbols (e.g., BTC, ETH, PEPE)
- `main_currency`: Quote currency (e.g., USDT, BTC)
- `exchange`: Exchange name (currently supports Binance)
- `time_freq`: Timeframe for data (e.g., 1h, 1d, 5m)

## Dependencies

- `requests`: For Binance API calls
- `pyyaml`: For configuration file parsing
- `tensortrade` (optional): For TensorTrade compatibility validation

Install missing dependencies:
```bash
pip install requests pyyaml
pip install tensortrade  # Optional, for full validation
```

## Error Handling

The script gracefully handles various error conditions:

- **Missing configuration files**: Clear error messages
- **Invalid YAML syntax**: Detailed parsing errors  
- **Network connectivity issues**: Timeout and retry logic
- **Missing dependencies**: Graceful degradation with warnings
- **Invalid currency symbols**: Clear identification of problematic pairs

## Integration with Data Pipeline

This validation script is designed to work with:

- `fetch_currency.py`: Binance API data fetching script
- `data_parser.py`: TensorTrade-based data processing pipeline
- Configuration files in `conf/data/`: YAML-based data configurations

Run this validation script before starting any data collection to ensure all configured currencies are available and accessible.
