"""
SENTINEL AI - Data Collection & Replay Buffer
Collects and stores EVERYTHING for AI training

This is the MEMORY of the system:
- Market data (OHLCV, orderbook, funding)
- Trade decisions and outcomes
- News and sentiment
- Regime snapshots
- Feature vectors

Data is used for:
- Periodic model training
- Backtesting
- Performance analysis
- Continuous improvement

NOTHING is wasted - every data point is valuable.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import gzip
import os
from pathlib import Path
from loguru import logger
import redis.asyncio as redis
import httpx

from config import settings


@dataclass
class MarketSnapshot:
    """Complete market state at a point in time"""
    timestamp: str
    symbol: str
    
    # Price data
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Derived
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    atr: float
    
    # Market data
    funding_rate: float
    open_interest: float
    bid_price: float
    ask_price: float
    spread: float
    
    # Regime
    regime: str
    regime_confidence: float
    volatility: float
    liquidity_score: float
    trend_strength: float
    trend_direction: str


@dataclass
class TradeRecord:
    """Complete record of a trade decision"""
    timestamp: str
    trade_id: str
    symbol: str
    
    # Decision
    action: str  # 'buy', 'sell', 'hold'
    direction: str  # 'long', 'short'
    confidence: float
    
    # Edge analysis at entry
    edge_score: float
    technical_edge: float
    momentum_edge: float
    volume_edge: float
    sentiment_edge: float
    
    # Regime at entry
    regime: str
    regime_action: str
    
    # Position details
    entry_price: float
    quantity: float
    position_value: float
    leverage: int
    
    # Risk parameters
    stop_loss: float
    take_profit: float
    kelly_fraction: float
    
    # Outcome (filled after close)
    exit_price: float = 0.0
    exit_time: str = ""
    pnl_percent: float = 0.0
    pnl_value: float = 0.0
    duration_seconds: int = 0
    exit_reason: str = ""
    won: bool = False


@dataclass
class NewsRecord:
    """News article with sentiment"""
    timestamp: str
    source: str
    title: str
    content: str
    url: str
    
    # Analysis
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'bullish', 'bearish', 'neutral'
    relevance: float  # 0-1
    
    # Related assets
    related_symbols: List[str]
    
    # Impact (measured later)
    actual_impact: float = 0.0


@dataclass
class FeatureVector:
    """Feature vector for ML training"""
    timestamp: str
    symbol: str
    
    # Price features (normalized)
    price_change_1h: float
    price_change_4h: float
    price_change_24h: float
    
    # Technical features
    rsi_normalized: float
    macd_normalized: float
    bb_position: float  # 0=lower, 0.5=middle, 1=upper
    atr_normalized: float
    
    # Volume features
    volume_ratio: float  # vs 20-period average
    volume_trend: float  # increasing/decreasing
    
    # Market structure
    regime_encoded: int  # One-hot or label encoded
    trend_strength: float
    volatility_normalized: float
    liquidity_normalized: float
    
    # Sentiment features
    fear_greed_normalized: float
    news_sentiment: float
    funding_rate_normalized: float
    
    # Target (for supervised learning)
    target_1h: float  # Actual price change in next hour
    target_4h: float  # Actual price change in 4 hours
    target_24h: float  # Actual price change in 24 hours
    target_direction: int  # 1=up, 0=down, -1=hold


class DataCollector:
    """
    Central Data Collection System
    
    Collects and stores:
    - Market snapshots every minute
    - All trade decisions and outcomes
    - News with sentiment
    - Feature vectors for training
    
    Data is stored in:
    - Redis (hot data, last 24h)
    - Disk (cold data, compressed)
    """
    
    def __init__(self):
        self.redis_client = None
        self.http_client = None
        self.is_running = False
        
        # Data paths
        self.data_dir = Path("/opt/sentinel/data")
        self.live_dir = self.data_dir / "live"
        self.training_dir = self.data_dir / "training"
        self.replay_dir = self.data_dir / "replay"
        
        # Collection intervals
        self.snapshot_interval = 60  # 1 minute
        self.feature_interval = 300  # 5 minutes
        self.flush_interval = 3600  # 1 hour (save to disk)
        
        # Buffers (in memory before flushing)
        self.snapshot_buffer: List[MarketSnapshot] = []
        self.trade_buffer: List[TradeRecord] = []
        self.news_buffer: List[NewsRecord] = []
        self.feature_buffer: List[FeatureVector] = []
        
        # Tracked symbols
        self.tracked_symbols: List[str] = []
        
        # Stats
        self.stats = {
            'snapshots_collected': 0,
            'trades_recorded': 0,
            'news_collected': 0,
            'features_generated': 0,
            'bytes_stored': 0
        }
        
    async def initialize(self):
        """Initialize data collector"""
        logger.info("Initializing Data Collector (Replay Buffer)...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Create directories
        self._create_directories()
        
        # Load tracked symbols
        await self._load_tracked_symbols()
        
        # Load stats
        await self._load_stats()
        
        self.is_running = True
        
        # Start collection tasks
        asyncio.create_task(self._snapshot_collection_loop())
        asyncio.create_task(self._feature_generation_loop())
        asyncio.create_task(self._disk_flush_loop())
        
        logger.info(f"Data Collector initialized - Tracking {len(self.tracked_symbols)} symbols")
        
    async def shutdown(self):
        """Graceful shutdown - flush all buffers"""
        logger.info("Shutting down Data Collector...")
        self.is_running = False
        
        # Flush remaining data
        await self._flush_to_disk()
        await self._save_stats()
        
        if self.http_client:
            await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.aclose()
            
    def _create_directories(self):
        """Create data directories if they don't exist"""
        for d in [self.data_dir, self.live_dir, self.training_dir, self.replay_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories
        (self.live_dir / "snapshots").mkdir(exist_ok=True)
        (self.live_dir / "features").mkdir(exist_ok=True)
        (self.replay_dir / "trades").mkdir(exist_ok=True)
        (self.replay_dir / "news").mkdir(exist_ok=True)
        (self.training_dir / "datasets").mkdir(exist_ok=True)
        
    async def _load_tracked_symbols(self):
        """Load symbols to track"""
        try:
            symbols_data = await self.redis_client.get('trading:available_symbols')
            if symbols_data:
                self.tracked_symbols = symbols_data.decode().split(',')[:100]  # Top 100
            else:
                self.tracked_symbols = [
                    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
                    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT'
                ]
        except:
            self.tracked_symbols = ['BTCUSDT', 'ETHUSDT']
            
    async def _snapshot_collection_loop(self):
        """Collect market snapshots every minute"""
        logger.info("Starting snapshot collection loop...")
        
        while self.is_running:
            try:
                await self._collect_snapshots()
                await asyncio.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"Snapshot collection error: {e}")
                await asyncio.sleep(30)
                
    async def _collect_snapshots(self):
        """Collect snapshot for all tracked symbols"""
        timestamp = datetime.utcnow().isoformat()
        
        for symbol in self.tracked_symbols[:50]:  # Batch of 50
            try:
                snapshot = await self._get_market_snapshot(symbol, timestamp)
                if snapshot:
                    self.snapshot_buffer.append(snapshot)
                    self.stats['snapshots_collected'] += 1
                    
                    # Store in Redis (hot data)
                    await self._store_snapshot_redis(snapshot)
                    
            except Exception as e:
                logger.debug(f"Snapshot error for {symbol}: {e}")
                
        logger.debug(f"Collected {len(self.snapshot_buffer)} snapshots")
        
    async def _get_market_snapshot(self, symbol: str, timestamp: str) -> Optional[MarketSnapshot]:
        """Get complete market snapshot for a symbol"""
        try:
            # Get kline data
            url = "https://api.bybit.com/v5/market/kline"
            params = {'category': 'linear', 'symbol': symbol, 'interval': '1', 'limit': 100}
            
            response = await self.http_client.get(url, params=params)
            if response.status_code != 200:
                return None
                
            data = response.json()
            klines = data.get('result', {}).get('list', [])
            
            if len(klines) < 20:
                return None
                
            # Parse OHLCV
            current = klines[0]
            closes = [float(k[4]) for k in reversed(klines)]
            highs = [float(k[2]) for k in reversed(klines)]
            lows = [float(k[3]) for k in reversed(klines)]
            volumes = [float(k[5]) for k in reversed(klines)]
            
            # Calculate indicators
            rsi = self._calculate_rsi(closes)
            macd, signal = self._calculate_macd(closes)
            bb_upper, bb_lower = self._calculate_bollinger(closes)
            atr = self._calculate_atr(highs, lows, closes)
            
            # Get ticker for more data
            ticker_url = "https://api.bybit.com/v5/market/tickers"
            ticker_params = {'category': 'linear', 'symbol': symbol}
            ticker_resp = await self.http_client.get(ticker_url, params=ticker_params)
            
            ticker_data = {}
            if ticker_resp.status_code == 200:
                ticker_list = ticker_resp.json().get('result', {}).get('list', [])
                if ticker_list:
                    ticker_data = ticker_list[0]
                    
            # Get regime from Redis
            regime_data = await self.redis_client.hgetall(f'regime:{symbol}')
            regime = regime_data.get(b'regime', b'unknown').decode() if regime_data else 'unknown'
            regime_conf = float(regime_data.get(b'confidence', 0)) if regime_data else 0
            volatility = float(regime_data.get(b'volatility', 0)) if regime_data else 0
            liquidity = float(regime_data.get(b'liquidity', 50)) if regime_data else 50
            trend_str = float(regime_data.get(b'trend_strength', 0)) if regime_data else 0
            trend_dir = regime_data.get(b'trend_direction', b'neutral').decode() if regime_data else 'neutral'
            
            return MarketSnapshot(
                timestamp=timestamp,
                symbol=symbol,
                open=float(current[1]),
                high=float(current[2]),
                low=float(current[3]),
                close=float(current[4]),
                volume=float(current[5]),
                rsi=rsi,
                macd=macd,
                macd_signal=signal,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                atr=atr,
                funding_rate=float(ticker_data.get('fundingRate', 0)),
                open_interest=float(ticker_data.get('openInterest', 0)),
                bid_price=float(ticker_data.get('bid1Price', 0)),
                ask_price=float(ticker_data.get('ask1Price', 0)),
                spread=float(ticker_data.get('ask1Price', 0)) - float(ticker_data.get('bid1Price', 0)),
                regime=regime,
                regime_confidence=regime_conf,
                volatility=volatility,
                liquidity_score=liquidity,
                trend_strength=trend_str,
                trend_direction=trend_dir
            )
            
        except Exception as e:
            logger.debug(f"Snapshot creation error: {e}")
            return None
            
    async def _store_snapshot_redis(self, snapshot: MarketSnapshot):
        """Store snapshot in Redis (hot data)"""
        try:
            key = f"snapshots:{snapshot.symbol}"
            await self.redis_client.lpush(key, json.dumps(asdict(snapshot)))
            await self.redis_client.ltrim(key, 0, 1439)  # Keep 24h of 1-min data
        except:
            pass
            
    async def record_trade(self, trade: TradeRecord):
        """Record a trade decision"""
        self.trade_buffer.append(trade)
        self.stats['trades_recorded'] += 1
        
        # Store in Redis immediately
        try:
            key = f"trades:history:{trade.symbol}"
            await self.redis_client.lpush(key, json.dumps(asdict(trade)))
            await self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 trades
            
            # Also store in global trades list
            await self.redis_client.lpush('trades:all', json.dumps(asdict(trade)))
            await self.redis_client.ltrim('trades:all', 0, 9999)
            
        except Exception as e:
            logger.error(f"Trade recording error: {e}")
            
    async def update_trade_outcome(self, trade_id: str, exit_price: float, 
                                    pnl_percent: float, pnl_value: float,
                                    exit_reason: str, duration_seconds: int):
        """Update trade with outcome (called when trade closes)"""
        try:
            # Find and update trade in buffer
            for trade in self.trade_buffer:
                if trade.trade_id == trade_id:
                    trade.exit_price = exit_price
                    trade.exit_time = datetime.utcnow().isoformat()
                    trade.pnl_percent = pnl_percent
                    trade.pnl_value = pnl_value
                    trade.exit_reason = exit_reason
                    trade.duration_seconds = duration_seconds
                    trade.won = pnl_percent > 0
                    
                    # Store updated trade
                    await self.redis_client.lpush('trades:completed', json.dumps(asdict(trade)))
                    break
                    
        except Exception as e:
            logger.error(f"Trade outcome update error: {e}")
            
    async def record_news(self, news: NewsRecord):
        """Record news article with sentiment"""
        self.news_buffer.append(news)
        self.stats['news_collected'] += 1
        
        # Store in Redis
        try:
            await self.redis_client.lpush('news:history', json.dumps(asdict(news)))
            await self.redis_client.ltrim('news:history', 0, 4999)  # Keep last 5000 articles
        except:
            pass
            
    async def _feature_generation_loop(self):
        """Generate feature vectors for training"""
        logger.info("Starting feature generation loop...")
        
        while self.is_running:
            try:
                await self._generate_features()
                await asyncio.sleep(self.feature_interval)
            except Exception as e:
                logger.error(f"Feature generation error: {e}")
                await asyncio.sleep(60)
                
    async def _generate_features(self):
        """Generate feature vectors for all tracked symbols"""
        timestamp = datetime.utcnow().isoformat()
        
        for symbol in self.tracked_symbols[:30]:  # Top 30 for features
            try:
                feature = await self._create_feature_vector(symbol, timestamp)
                if feature:
                    self.feature_buffer.append(feature)
                    self.stats['features_generated'] += 1
                    
                    # Store in Redis
                    key = f"features:{symbol}"
                    await self.redis_client.lpush(key, json.dumps(asdict(feature)))
                    await self.redis_client.ltrim(key, 0, 287)  # Keep 24h of 5-min features
                    
            except Exception as e:
                logger.debug(f"Feature generation error for {symbol}: {e}")
                
    async def _create_feature_vector(self, symbol: str, timestamp: str) -> Optional[FeatureVector]:
        """Create normalized feature vector for ML training"""
        try:
            # Get recent snapshots from Redis
            snapshots_data = await self.redis_client.lrange(f"snapshots:{symbol}", 0, 59)
            if len(snapshots_data) < 20:
                return None
                
            snapshots = [json.loads(s) for s in snapshots_data]
            current = snapshots[0]
            
            closes = [s['close'] for s in reversed(snapshots)]
            
            # Calculate price changes
            price_1h = (closes[-1] / closes[-60] - 1) * 100 if len(closes) >= 60 else 0
            price_4h = (closes[-1] / closes[0] - 1) * 100  # Approx
            price_24h = float(current.get('price_change_24h', 0))
            
            # Normalize RSI (0-100 to 0-1)
            rsi_normalized = current.get('rsi', 50) / 100
            
            # Normalize MACD
            macd = current.get('macd', 0)
            macd_normalized = np.tanh(macd / 100)  # Squash to -1, 1
            
            # BB position (0=lower, 1=upper)
            bb_upper = current.get('bb_upper', closes[-1] * 1.02)
            bb_lower = current.get('bb_lower', closes[-1] * 0.98)
            bb_range = bb_upper - bb_lower if bb_upper != bb_lower else 1
            bb_position = (closes[-1] - bb_lower) / bb_range
            bb_position = max(0, min(1, bb_position))
            
            # Volume ratio
            volumes = [s['volume'] for s in reversed(snapshots)]
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # Volume trend
            recent_vol = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            older_vol = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else volumes[-1]
            volume_trend = (recent_vol / older_vol - 1) if older_vol > 0 else 0
            
            # Regime encoding
            regime_map = {
                'high_liquidity_trend': 1,
                'accumulation': 2,
                'range_bound': 3,
                'distribution': 4,
                'high_volatility': 5,
                'low_liquidity': 6,
                'news_event': 7,
                'unknown': 0
            }
            regime_encoded = regime_map.get(current.get('regime', 'unknown'), 0)
            
            # Fear & Greed
            fg_data = await self.redis_client.get('data:fear_greed')
            fg_value = 50
            if fg_data:
                fg = json.loads(fg_data)
                fg_value = int(fg.get('value', 50))
            fg_normalized = fg_value / 100
            
            # News sentiment
            news_sentiment = 0
            news_data = await self.redis_client.get('data:news_sentiment')
            if news_data:
                ns = json.loads(news_data)
                news_sentiment = float(ns.get('overall_sentiment', 0))
                
            return FeatureVector(
                timestamp=timestamp,
                symbol=symbol,
                price_change_1h=price_1h,
                price_change_4h=price_4h,
                price_change_24h=price_24h,
                rsi_normalized=rsi_normalized,
                macd_normalized=macd_normalized,
                bb_position=bb_position,
                atr_normalized=current.get('atr', 0) / closes[-1] if closes[-1] > 0 else 0,
                volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                regime_encoded=regime_encoded,
                trend_strength=current.get('trend_strength', 0),
                volatility_normalized=min(1, current.get('volatility', 0) / 5),
                liquidity_normalized=current.get('liquidity_score', 50) / 100,
                fear_greed_normalized=fg_normalized,
                news_sentiment=news_sentiment,
                funding_rate_normalized=np.tanh(current.get('funding_rate', 0) * 1000),
                target_1h=0.0,  # To be filled during training
                target_4h=0.0,
                target_24h=0.0,
                target_direction=0
            )
            
        except Exception as e:
            logger.debug(f"Feature vector error: {e}")
            return None
            
    async def _disk_flush_loop(self):
        """Periodically flush buffers to disk"""
        logger.info("Starting disk flush loop...")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_to_disk()
            except Exception as e:
                logger.error(f"Disk flush error: {e}")
                
    async def _flush_to_disk(self):
        """Flush all buffers to disk (compressed)"""
        if not self.snapshot_buffer and not self.trade_buffer:
            return
            
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Flush snapshots
            if self.snapshot_buffer:
                path = self.live_dir / "snapshots" / f"snapshots_{timestamp}.json.gz"
                data = [asdict(s) for s in self.snapshot_buffer]
                self._write_compressed(path, data)
                self.stats['bytes_stored'] += path.stat().st_size
                self.snapshot_buffer.clear()
                logger.info(f"Flushed {len(data)} snapshots to disk")
                
            # Flush trades
            if self.trade_buffer:
                path = self.replay_dir / "trades" / f"trades_{timestamp}.json.gz"
                data = [asdict(t) for t in self.trade_buffer]
                self._write_compressed(path, data)
                self.stats['bytes_stored'] += path.stat().st_size
                self.trade_buffer.clear()
                logger.info(f"Flushed {len(data)} trades to disk")
                
            # Flush features
            if self.feature_buffer:
                path = self.live_dir / "features" / f"features_{timestamp}.json.gz"
                data = [asdict(f) for f in self.feature_buffer]
                self._write_compressed(path, data)
                self.stats['bytes_stored'] += path.stat().st_size
                self.feature_buffer.clear()
                
            # Flush news
            if self.news_buffer:
                path = self.replay_dir / "news" / f"news_{timestamp}.json.gz"
                data = [asdict(n) for n in self.news_buffer]
                self._write_compressed(path, data)
                self.stats['bytes_stored'] += path.stat().st_size
                self.news_buffer.clear()
                
        except Exception as e:
            logger.error(f"Flush to disk error: {e}")
            
    def _write_compressed(self, path: Path, data: List[Dict]):
        """Write data to gzipped JSON file"""
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
            
    def _read_compressed(self, path: Path) -> List[Dict]:
        """Read data from gzipped JSON file"""
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
            
    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, closes: List[float]) -> Tuple[float, float]:
        """Calculate MACD and signal"""
        if len(closes) < 26:
            return 0.0, 0.0
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd = ema12 - ema26
        signal = self._ema([macd], 9) if macd else 0
        return macd, signal
        
    def _ema(self, data: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(data) < period:
            return data[-1] if data else 0
        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
        
    def _calculate_bollinger(self, closes: List[float], period: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        if len(closes) < period:
            return closes[-1] * 1.02, closes[-1] * 0.98
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        return sma + 2 * std, sma - 2 * std
        
    def _calculate_atr(self, highs: List[float], lows: List[float], 
                       closes: List[float], period: int = 14) -> float:
        """Calculate ATR"""
        if len(closes) < period + 1:
            return 0.0
        tr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        return np.mean(tr_values[-period:])
        
    async def get_training_dataset(self, symbol: str, days: int = 30) -> List[FeatureVector]:
        """Get feature vectors for training"""
        features = []
        
        # From Redis (recent)
        try:
            data = await self.redis_client.lrange(f"features:{symbol}", 0, -1)
            features.extend([FeatureVector(**json.loads(d)) for d in data])
        except:
            pass
            
        # From disk (historical)
        feature_dir = self.live_dir / "features"
        if feature_dir.exists():
            for path in sorted(feature_dir.glob("*.json.gz"))[-days*24:]:  # Approx
                try:
                    data = self._read_compressed(path)
                    for d in data:
                        if d['symbol'] == symbol:
                            features.append(FeatureVector(**d))
                except:
                    continue
                    
        return features
        
    async def get_trade_history(self, symbol: str = None, limit: int = 1000) -> List[TradeRecord]:
        """Get trade history for analysis"""
        trades = []
        
        try:
            if symbol:
                data = await self.redis_client.lrange(f"trades:history:{symbol}", 0, limit-1)
            else:
                data = await self.redis_client.lrange('trades:all', 0, limit-1)
                
            trades = [TradeRecord(**json.loads(d)) for d in data]
        except:
            pass
            
        return trades
        
    async def _load_stats(self):
        """Load stats from Redis"""
        try:
            data = await self.redis_client.get('collector:stats')
            if data:
                self.stats = json.loads(data)
        except:
            pass
            
    async def _save_stats(self):
        """Save stats to Redis"""
        try:
            await self.redis_client.set('collector:stats', json.dumps(self.stats))
        except:
            pass
            
    async def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            **self.stats,
            'buffer_sizes': {
                'snapshots': len(self.snapshot_buffer),
                'trades': len(self.trade_buffer),
                'features': len(self.feature_buffer),
                'news': len(self.news_buffer)
            },
            'tracked_symbols': len(self.tracked_symbols)
        }


# Global instance
data_collector = DataCollector()

