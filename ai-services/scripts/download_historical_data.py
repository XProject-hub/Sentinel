#!/usr/bin/env python3
"""
SENTINEL AI - Historical Data Downloader
Downloads historical data from multiple sources for AI training

Sources:
1. Binance Data Vision - OHLCV klines
2. Bybit API - Klines + Funding rates
3. CryptoCompare - News sentiment data

Run: python download_historical_data.py

Data is saved to: /mnt/sentinel-data/historical/ (or ./data/historical/)
"""

import os
import sys
import json
import asyncio
import zipfile
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Configuration
DATA_DIR = Path("/mnt/sentinel-data/historical")
if not DATA_DIR.exists():
    DATA_DIR = Path("./data/historical")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Symbols to download
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT', 'AAVEUSDT'
]

# Timeframes
TIMEFRAMES = ['1h', '4h', '1d']


def download_binance_klines(symbol: str, interval: str, start_date: str, end_date: str = None):
    """
    Download klines from Binance Data Vision
    https://data.binance.vision/
    """
    print(f"\n📥 Downloading {symbol} {interval} from Binance...")
    
    base_url = "https://data.binance.vision/data/futures/um/daily/klines"
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    
    # Create output directory
    output_dir = DATA_DIR / "binance" / symbol / interval
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    current = start
    downloaded = 0
    errors = 0
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        filename = f"{symbol}-{interval}-{date_str}.zip"
        url = f"{base_url}/{symbol}/{interval}/{filename}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Save zip file
                zip_path = output_dir / filename
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                # Extract and read CSV
                with zipfile.ZipFile(zip_path, 'r') as z:
                    csv_name = z.namelist()[0]
                    z.extractall(output_dir)
                    
                    # Read and append data
                    csv_path = output_dir / csv_name
                    df = pd.read_csv(csv_path, header=None)
                    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                                 'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                 'taker_buy_quote', 'ignore']
                    all_data.append(df)
                    
                    # Cleanup
                    os.remove(zip_path)
                    os.remove(csv_path)
                    
                downloaded += 1
                if downloaded % 30 == 0:
                    print(f"  ✓ Downloaded {downloaded} days...")
            else:
                errors += 1
                
        except Exception as e:
            errors += 1
            
        current += timedelta(days=1)
    
    # Combine all data
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['open_time'])
        combined = combined.sort_values('open_time')
        
        # Save as parquet (efficient format)
        output_file = output_dir / f"{symbol}_{interval}_combined.parquet"
        combined.to_parquet(output_file, index=False)
        
        # Also save as CSV for compatibility
        csv_file = output_dir / f"{symbol}_{interval}_combined.csv"
        combined.to_csv(csv_file, index=False)
        
        print(f"  ✅ {symbol} {interval}: {len(combined)} candles saved")
        print(f"     Downloaded: {downloaded} days, Errors: {errors}")
        return len(combined)
    else:
        print(f"  ❌ {symbol} {interval}: No data downloaded")
        return 0


def download_bybit_klines(symbol: str, interval: str = "D", limit: int = 1000):
    """
    Download klines from Bybit API
    """
    print(f"\n📥 Downloading {symbol} from Bybit API...")
    
    url = "https://api.bybit.com/v5/market/kline"
    
    # Map interval
    interval_map = {'1h': '60', '4h': '240', '1d': 'D', '1w': 'W'}
    bybit_interval = interval_map.get(interval, interval)
    
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': bybit_interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data.get('retCode') == 0:
            klines = data['result']['list']
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert types
            df['open_time'] = pd.to_numeric(df['open_time'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col])
            
            # Sort by time
            df = df.sort_values('open_time')
            
            # Save
            output_dir = DATA_DIR / "bybit" / symbol
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{symbol}_{interval}.parquet"
            df.to_parquet(output_file, index=False)
            
            csv_file = output_dir / f"{symbol}_{interval}.csv"
            df.to_csv(csv_file, index=False)
            
            print(f"  ✅ {symbol}: {len(df)} candles saved")
            return len(df)
        else:
            print(f"  ❌ {symbol}: API error - {data.get('retMsg')}")
            return 0
            
    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        return 0


def download_bybit_funding_rates(symbol: str, limit: int = 200):
    """
    Download funding rate history from Bybit
    """
    print(f"\n📥 Downloading {symbol} funding rates...")
    
    url = "https://api.bybit.com/v5/market/funding/history"
    
    params = {
        'category': 'linear',
        'symbol': symbol,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data.get('retCode') == 0:
            funding = data['result']['list']
            
            df = pd.DataFrame(funding)
            
            # Save
            output_dir = DATA_DIR / "funding" / symbol
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{symbol}_funding.csv"
            df.to_csv(output_file, index=False)
            
            print(f"  ✅ {symbol}: {len(df)} funding rates saved")
            return len(df)
        else:
            print(f"  ❌ {symbol}: {data.get('retMsg')}")
            return 0
            
    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        return 0


def download_fear_greed_index():
    """
    Download Crypto Fear & Greed Index history
    https://alternative.me/crypto/fear-and-greed-index/
    """
    print("\n📥 Downloading Fear & Greed Index...")
    
    url = "https://api.alternative.me/fng/?limit=0"  # All history
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['value'] = pd.to_numeric(df['value'])
            
            output_dir = DATA_DIR / "sentiment"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "fear_greed_index.csv"
            df.to_csv(output_file, index=False)
            
            print(f"  ✅ Fear & Greed: {len(df)} days saved")
            return len(df)
        else:
            print("  ❌ Fear & Greed: No data")
            return 0
            
    except Exception as e:
        print(f"  ❌ Fear & Greed: {e}")
        return 0


def create_summary():
    """Create a summary of all downloaded data"""
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            filepath = Path(root) / file
            total_files += 1
            total_size += filepath.stat().st_size
    
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print(f"Data directory: {DATA_DIR}")
    
    # List directories
    print("\nData structure:")
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            sub_files = list(item.rglob("*"))
            print(f"  📁 {item.name}: {len([f for f in sub_files if f.is_file()])} files")


def main():
    print("="*60)
    print("🚀 SENTINEL AI - Historical Data Downloader")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timeframes: {TIMEFRAMES}")
    
    # 1. Download Binance data (last 2 years)
    print("\n" + "="*60)
    print("📊 PART 1: Binance OHLCV Data (2 years)")
    print("="*60)
    
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    for symbol in SYMBOLS[:5]:  # Top 5 first
        for tf in TIMEFRAMES:
            download_binance_klines(symbol, tf, start_date, end_date)
    
    # 2. Download Bybit data
    print("\n" + "="*60)
    print("📊 PART 2: Bybit API Data")
    print("="*60)
    
    for symbol in SYMBOLS:
        download_bybit_klines(symbol, '1d', 1000)
        download_bybit_funding_rates(symbol)
    
    # 3. Download sentiment data
    print("\n" + "="*60)
    print("📊 PART 3: Sentiment Data")
    print("="*60)
    
    download_fear_greed_index()
    
    # Summary
    create_summary()
    
    print("\n" + "="*60)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python load_data_to_redis.py")
    print("2. AI will automatically use this data for training")


if __name__ == "__main__":
    main()
