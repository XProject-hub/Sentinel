#!/usr/bin/env python3
"""
SENTINEL AI - Load Historical Data to Redis
Loads downloaded historical data into Redis for AI training

Run after: download_historical_data.py
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd
import redis.asyncio as redis

# Configuration
DATA_DIR = Path("/mnt/sentinel-data/historical")
if not DATA_DIR.exists():
    DATA_DIR = Path("./data/historical")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


async def load_klines_to_redis(r: redis.Redis):
    """Load OHLCV klines to Redis for AI training"""
    print("\n📊 Loading Klines to Redis...")
    
    loaded = 0
    
    # Check Binance data
    binance_dir = DATA_DIR / "binance"
    if binance_dir.exists():
        for symbol_dir in binance_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                
                for tf_dir in symbol_dir.iterdir():
                    if tf_dir.is_dir():
                        # Find combined file (CSV only)
                        csv_files = list(tf_dir.glob("*_combined.csv"))
                        
                        data_file = csv_files[0] if csv_files else None
                        
                        if data_file:
                            try:
                                df = pd.read_csv(data_file)
                                
                                # Store summary stats for AI
                                stats = {
                                    'symbol': symbol,
                                    'timeframe': tf_dir.name,
                                    'candles': len(df),
                                    'start_time': int(df['open_time'].min()),
                                    'end_time': int(df['open_time'].max()),
                                    'avg_volume': float(df['volume'].mean()),
                                    'volatility': float(df['high'].astype(float).sub(df['low'].astype(float)).div(df['close'].astype(float)).mean() * 100),
                                    'source': 'binance'
                                }
                                
                                await r.hset(
                                    f"ai:historical:{symbol}:{tf_dir.name}",
                                    mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in stats.items()}
                                )
                                
                                # Store recent data for quick access (last 500 candles)
                                recent = df.tail(500).to_dict('records')
                                await r.set(
                                    f"ai:klines:{symbol}:{tf_dir.name}",
                                    json.dumps(recent),
                                    ex=86400 * 7  # 7 days expiry
                                )
                                
                                loaded += 1
                                print(f"  ✅ {symbol} {tf_dir.name}: {len(df)} candles")
                                
                            except Exception as e:
                                print(f"  ❌ {symbol} {tf_dir.name}: {e}")
    
    # Check Bybit data
    bybit_dir = DATA_DIR / "bybit"
    if bybit_dir.exists():
        for symbol_dir in bybit_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                
                csv_files = list(symbol_dir.glob("*.csv"))
                
                for data_file in csv_files:
                    try:
                        df = pd.read_csv(data_file)
                        
                        tf = data_file.stem.split('_')[-1]
                        
                        stats = {
                            'symbol': symbol,
                            'timeframe': tf,
                            'candles': len(df),
                            'source': 'bybit'
                        }
                        
                        await r.hset(
                            f"ai:historical:{symbol}:{tf}",
                            mapping={k: str(v) for k, v in stats.items()}
                        )
                        
                        loaded += 1
                        print(f"  ✅ {symbol} {tf}: {len(df)} candles (Bybit)")
                        
                    except Exception as e:
                        print(f"  ❌ {data_file}: {e}")
    
    print(f"\n  Total loaded: {loaded} datasets")
    return loaded


async def load_funding_rates_to_redis(r: redis.Redis):
    """Load funding rate history to Redis"""
    print("\n📊 Loading Funding Rates to Redis...")
    
    loaded = 0
    funding_dir = DATA_DIR / "funding"
    
    if funding_dir.exists():
        for symbol_dir in funding_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                
                csv_files = list(symbol_dir.glob("*.csv"))
                for data_file in csv_files:
                    try:
                        df = pd.read_csv(data_file)
                        
                        # Calculate average funding
                        if 'fundingRate' in df.columns:
                            avg_funding = df['fundingRate'].astype(float).mean()
                            max_funding = df['fundingRate'].astype(float).max()
                            min_funding = df['fundingRate'].astype(float).min()
                            
                            stats = {
                                'symbol': symbol,
                                'records': len(df),
                                'avg_funding': float(avg_funding),
                                'max_funding': float(max_funding),
                                'min_funding': float(min_funding)
                            }
                            
                            await r.hset(
                                f"ai:funding:{symbol}",
                                mapping={k: str(v) for k, v in stats.items()}
                            )
                            
                            # Store recent rates
                            recent = df.tail(100).to_dict('records')
                            await r.set(
                                f"ai:funding:history:{symbol}",
                                json.dumps(recent),
                                ex=86400  # 1 day expiry
                            )
                            
                            loaded += 1
                            print(f"  ✅ {symbol}: {len(df)} funding rates (avg: {avg_funding*100:.4f}%)")
                            
                    except Exception as e:
                        print(f"  ❌ {symbol}: {e}")
    
    print(f"\n  Total loaded: {loaded} symbols")
    return loaded


async def load_sentiment_to_redis(r: redis.Redis):
    """Load sentiment data to Redis"""
    print("\n📊 Loading Sentiment Data to Redis...")
    
    loaded = 0
    sentiment_dir = DATA_DIR / "sentiment"
    
    if sentiment_dir.exists():
        # Fear & Greed Index
        fg_file = sentiment_dir / "fear_greed_index.csv"
        if fg_file.exists():
            try:
                df = pd.read_csv(fg_file)
                
                # Calculate stats
                stats = {
                    'records': len(df),
                    'avg_value': float(df['value'].mean()),
                    'current': int(df.iloc[0]['value']) if len(df) > 0 else 50,
                    'classification': df.iloc[0]['value_classification'] if 'value_classification' in df.columns else 'neutral'
                }
                
                await r.hset("ai:sentiment:fear_greed", mapping={k: str(v) for k, v in stats.items()})
                
                # Store recent data
                recent = df.head(30).to_dict('records')
                await r.set("ai:sentiment:fear_greed:history", json.dumps(recent), ex=86400)
                
                loaded += 1
                print(f"  ✅ Fear & Greed Index: {len(df)} days (current: {stats['current']})")
                
            except Exception as e:
                print(f"  ❌ Fear & Greed: {e}")
    
    print(f"\n  Total loaded: {loaded} sentiment sources")
    return loaded


async def update_ai_learning_stats(r: redis.Redis, total_loaded: int):
    """Update AI learning statistics"""
    print("\n📊 Updating AI Learning Stats...")
    
    await r.hset("ai:learning:data_sources", mapping={
        'total_datasets': str(total_loaded),
        'last_updated': datetime.now().isoformat(),
        'data_directory': str(DATA_DIR)
    })
    
    # Trigger AI to reload data
    await r.publish("ai:reload_data", json.dumps({
        'action': 'reload',
        'datasets': total_loaded,
        'timestamp': datetime.now().isoformat()
    }))
    
    print(f"  ✅ AI notified: {total_loaded} datasets available")


async def main():
    print("="*60)
    print("🚀 SENTINEL AI - Load Data to Redis")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Redis: {REDIS_URL}")
    
    # Connect to Redis
    r = redis.from_url(REDIS_URL)
    
    try:
        await r.ping()
        print("✅ Redis connected")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return
    
    total = 0
    
    # Load all data
    total += await load_klines_to_redis(r)
    total += await load_funding_rates_to_redis(r)
    total += await load_sentiment_to_redis(r)
    
    # Update stats
    await update_ai_learning_stats(r, total)
    
    await r.close()
    
    print("\n" + "="*60)
    print("✅ DATA LOADING COMPLETE!")
    print("="*60)
    print(f"Total datasets loaded: {total}")
    print("\nAI will now use this data for improved predictions!")


if __name__ == "__main__":
    asyncio.run(main())
