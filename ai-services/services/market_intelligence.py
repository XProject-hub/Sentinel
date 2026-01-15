"""
SENTINEL AI - Market Intelligence Service
Real-time market data collection and analysis
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from loguru import logger
import ccxt.async_support as ccxt
import redis.asyncio as redis
import json

from config import settings


class MarketIntelligenceService:
    """
    Collects and processes real-time market data:
    - Price ticks
    - Order book depth
    - Volume analysis
    - Volatility metrics
    - Funding rates
    - Liquidation data
    - Open interest
    """
    
    def __init__(self):
        self.is_running = False
        self.exchange = None
        self.redis_client = None
        self.symbols = settings.DEFAULT_SYMBOLS
        self.last_update = {}
        self._tasks = []
        
    async def initialize(self):
        """Initialize connections and data stores"""
        logger.info("Initializing Market Intelligence Service...")
        
        # Initialize exchange connection
        self.exchange = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
        # Initialize Redis
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        
        self.is_running = True
        logger.info("Market Intelligence Service initialized")
        
    async def shutdown(self):
        """Cleanup resources"""
        self.is_running = False
        
        for task in self._tasks:
            task.cancel()
            
        if self.exchange:
            await self.exchange.close()
            
        if self.redis_client:
            await self.redis_client.close()
            
    async def start_data_collection(self):
        """Start all data collection tasks"""
        self._tasks = [
            asyncio.create_task(self._collect_prices()),
            asyncio.create_task(self._collect_orderbooks()),
            asyncio.create_task(self._collect_funding_rates()),
            asyncio.create_task(self._calculate_indicators()),
        ]
        
    async def _collect_prices(self):
        """Collect real-time price data"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    
                    price_data = {
                        'symbol': symbol,
                        'price': ticker['last'],
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'volume_24h': ticker['quoteVolume'],
                        'change_24h': ticker['percentage'],
                        'high_24h': ticker['high'],
                        'low_24h': ticker['low'],
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    
                    # Store in Redis
                    await self.redis_client.hset(
                        f"market:price:{symbol}",
                        mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in price_data.items()}
                    )
                    
                    # Update last update time
                    self.last_update[symbol] = datetime.utcnow()
                    
                await asyncio.sleep(1)  # 1 second interval
                
            except Exception as e:
                logger.error(f"Price collection error: {e}")
                await asyncio.sleep(5)
                
    async def _collect_orderbooks(self):
        """Collect order book depth data"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    orderbook = await self.exchange.fetch_order_book(symbol, limit=20)
                    
                    bid_volume = sum([b[1] for b in orderbook['bids'][:10]])
                    ask_volume = sum([a[1] for a in orderbook['asks'][:10]])
                    
                    spread = orderbook['asks'][0][0] - orderbook['bids'][0][0]
                    spread_percent = (spread / orderbook['bids'][0][0]) * 100
                    
                    orderbook_data = {
                        'symbol': symbol,
                        'best_bid': orderbook['bids'][0][0],
                        'best_ask': orderbook['asks'][0][0],
                        'spread': spread,
                        'spread_percent': spread_percent,
                        'bid_volume': bid_volume,
                        'ask_volume': ask_volume,
                        'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume),
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    
                    await self.redis_client.hset(
                        f"market:orderbook:{symbol}",
                        mapping={k: str(v) for k, v in orderbook_data.items()}
                    )
                    
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Orderbook collection error: {e}")
                await asyncio.sleep(5)
                
    async def _collect_funding_rates(self):
        """Collect funding rate data for perpetual futures"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    funding = await self.exchange.fetch_funding_rate(symbol)
                    
                    funding_data = {
                        'symbol': symbol,
                        'funding_rate': funding['fundingRate'],
                        'funding_timestamp': funding['fundingTimestamp'],
                        'next_funding_time': funding.get('nextFundingTime'),
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    
                    await self.redis_client.hset(
                        f"market:funding:{symbol}",
                        mapping={k: str(v) for k, v in funding_data.items()}
                    )
                    
                await asyncio.sleep(60)  # Funding rates update less frequently
                
            except Exception as e:
                logger.error(f"Funding rate collection error: {e}")
                await asyncio.sleep(60)
                
    async def _calculate_indicators(self):
        """Calculate technical indicators from collected data"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # Get OHLCV data
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
                    
                    if len(ohlcv) < 50:
                        continue
                        
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Calculate volatility (ATR)
                    high_low = df['high'] - df['low']
                    high_close = np.abs(df['high'] - df['close'].shift())
                    low_close = np.abs(df['low'] - df['close'].shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    atr = true_range.rolling(14).mean().iloc[-1]
                    
                    # Calculate RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs.iloc[-1]))
                    
                    # Calculate volatility percentage
                    volatility_pct = (atr / df['close'].iloc[-1]) * 100
                    
                    # Calculate trend (SMA crossover)
                    sma_20 = df['close'].rolling(20).mean().iloc[-1]
                    sma_50 = df['close'].rolling(50).mean().iloc[-1]
                    trend = 'bullish' if sma_20 > sma_50 else 'bearish'
                    trend_strength = abs(sma_20 - sma_50) / sma_50 * 100
                    
                    indicators = {
                        'symbol': symbol,
                        'atr': atr,
                        'volatility_percent': volatility_pct,
                        'rsi': rsi,
                        'sma_20': sma_20,
                        'sma_50': sma_50,
                        'trend': trend,
                        'trend_strength': trend_strength,
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    
                    await self.redis_client.hset(
                        f"market:indicators:{symbol}",
                        mapping={k: str(v) for k, v in indicators.items()}
                    )
                    
                await asyncio.sleep(60)  # Calculate every minute
                
            except Exception as e:
                logger.error(f"Indicator calculation error: {e}")
                await asyncio.sleep(60)
                
    async def get_current_state(self) -> Dict[str, Any]:
        """Get aggregated current market state"""
        state = {}
        
        for symbol in self.symbols:
            try:
                price_data = await self.redis_client.hgetall(f"market:price:{symbol}")
                orderbook_data = await self.redis_client.hgetall(f"market:orderbook:{symbol}")
                indicators_data = await self.redis_client.hgetall(f"market:indicators:{symbol}")
                funding_data = await self.redis_client.hgetall(f"market:funding:{symbol}")
                
                state[symbol] = {
                    'price': self._decode_redis_hash(price_data),
                    'orderbook': self._decode_redis_hash(orderbook_data),
                    'indicators': self._decode_redis_hash(indicators_data),
                    'funding': self._decode_redis_hash(funding_data),
                }
            except Exception as e:
                logger.error(f"Error getting state for {symbol}: {e}")
                
        return state
        
    async def get_symbol_state(self, symbol: str) -> Dict[str, Any]:
        """Get market state for a specific symbol"""
        try:
            price_data = await self.redis_client.hgetall(f"market:price:{symbol}")
            orderbook_data = await self.redis_client.hgetall(f"market:orderbook:{symbol}")
            indicators_data = await self.redis_client.hgetall(f"market:indicators:{symbol}")
            funding_data = await self.redis_client.hgetall(f"market:funding:{symbol}")
            
            return {
                'price': self._decode_redis_hash(price_data),
                'orderbook': self._decode_redis_hash(orderbook_data),
                'indicators': self._decode_redis_hash(indicators_data),
                'funding': self._decode_redis_hash(funding_data),
            }
        except Exception as e:
            logger.error(f"Error getting state for {symbol}: {e}")
            return {}
            
    async def get_data_age(self) -> int:
        """Get age of most recent data in seconds"""
        if not self.last_update:
            return 9999
            
        oldest = min(self.last_update.values())
        return int((datetime.utcnow() - oldest).total_seconds())
        
    def _decode_redis_hash(self, data: Dict[bytes, bytes]) -> Dict[str, Any]:
        """Decode Redis hash data"""
        decoded = {}
        for k, v in data.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            try:
                decoded[key] = float(val)
            except ValueError:
                decoded[key] = val
        return decoded

