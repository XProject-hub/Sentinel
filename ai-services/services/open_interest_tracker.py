"""
Open Interest Tracker - Analyzes OI changes to improve entry signals

Key Patterns:
- OI ↑ + Price ↑ = Strong bullish (new longs entering) - CONFIRM LONG
- OI ↑ + Price ↓ = Bearish pressure (new shorts entering) - AVOID LONG / GO SHORT
- OI ↓ + Price ↑ = Weak rally (shorts closing) - CAUTION on LONG
- OI ↓ + Price ↓ = Capitulation (longs closing) - Possible bottom, wait

This helps distinguish real breakouts from fake-outs.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

import redis.asyncio as redis
from config import settings


class OISignal(Enum):
    STRONG_BULLISH = "strong_bullish"      # OI ↑ + Price ↑ - Best for longs
    WEAK_BULLISH = "weak_bullish"          # OI ↓ + Price ↑ - Caution on longs
    STRONG_BEARISH = "strong_bearish"      # OI ↑ + Price ↓ - Best for shorts
    WEAK_BEARISH = "weak_bearish"          # OI ↓ + Price ↓ - Possible capitulation
    NEUTRAL = "neutral"                     # No significant change
    ACCUMULATION = "accumulation"           # OI ↑↑ + Price flat - Big move coming
    DISTRIBUTION = "distribution"           # OI ↓↓ + Price flat - Interest fading


@dataclass
class OIAnalysis:
    """Result of OI analysis for a symbol"""
    symbol: str
    signal: OISignal
    oi_change_pct: float          # OI change % (e.g., +5.2%)
    price_change_pct: float       # Price change % (e.g., +2.1%)
    oi_trend: str                 # 'rising', 'falling', 'flat'
    confidence: int               # 0-100
    recommendation: str           # 'confirm', 'caution', 'avoid', 'wait'
    reasoning: str                # Human-readable explanation
    timestamp: datetime


class OpenInterestTracker:
    """
    Tracks and analyzes Open Interest patterns for trading signals
    """
    
    def __init__(self):
        self.redis_client = None
        self.oi_cache: Dict[str, List[Dict]] = {}  # symbol -> [oi_data_points]
        self.price_cache: Dict[str, List[Dict]] = {}  # symbol -> [price_data_points]
        self.analysis_cache: Dict[str, OIAnalysis] = {}  # symbol -> latest analysis
        
        # Thresholds for signal generation
        self.SIGNIFICANT_OI_CHANGE = 2.0      # % change to be significant
        self.STRONG_OI_CHANGE = 5.0           # % change for strong signal
        self.SIGNIFICANT_PRICE_CHANGE = 1.0   # % price move to consider
        self.ACCUMULATION_THRESHOLD = 8.0     # OI spike for accumulation
        
        # Analysis window
        self.ANALYSIS_PERIODS = 12  # Look at last 12 data points (1 hour for 5min intervals)
        
        self._initialized = False
        
    async def initialize(self):
        """Initialize the tracker"""
        try:
            self.redis_client = await redis.from_url(settings.REDIS_URL)
            await self._load_cache()
            self._initialized = True
            logger.info("Open Interest Tracker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OI Tracker: {e}")
            
    async def shutdown(self):
        """Cleanup"""
        await self._save_cache()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def _load_cache(self):
        """Load cached OI data from Redis"""
        try:
            data = await self.redis_client.get('oi_tracker:cache')
            if data:
                cache = json.loads(data)
                self.oi_cache = cache.get('oi', {})
                self.price_cache = cache.get('price', {})
                logger.info(f"Loaded OI cache: {len(self.oi_cache)} symbols")
        except Exception as e:
            logger.debug(f"Could not load OI cache: {e}")
            
    async def _save_cache(self):
        """Save OI cache to Redis"""
        try:
            cache = {
                'oi': self.oi_cache,
                'price': self.price_cache,
                'updated': datetime.utcnow().isoformat()
            }
            await self.redis_client.set('oi_tracker:cache', json.dumps(cache), ex=3600)
        except Exception as e:
            logger.debug(f"Could not save OI cache: {e}")
            
    async def update_oi_data(self, symbol: str, client) -> Optional[OIAnalysis]:
        """
        Fetch and analyze OI data for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            client: BybitV5Client instance
            
        Returns:
            OIAnalysis with signal and recommendation
        """
        try:
            # Fetch OI history
            oi_result = await client.get_open_interest(
                symbol=symbol,
                interval="5min",
                limit=self.ANALYSIS_PERIODS + 5  # Extra buffer
            )
            
            if not oi_result.get('success'):
                return None
                
            oi_list = oi_result.get('data', {}).get('list', [])
            if len(oi_list) < 3:
                return None
                
            # Store in cache
            if symbol not in self.oi_cache:
                self.oi_cache[symbol] = []
            
            # Parse OI data (newest first in Bybit API)
            oi_values = []
            for item in oi_list[:self.ANALYSIS_PERIODS]:
                oi_values.append({
                    'timestamp': int(item.get('timestamp', 0)),
                    'oi': float(item.get('openInterest', 0))
                })
            
            self.oi_cache[symbol] = oi_values
            
            # Fetch current ticker for price comparison
            ticker_result = await client.get_tickers(symbol=symbol, category="linear")
            if not ticker_result.get('success'):
                return None
                
            ticker_list = ticker_result.get('data', {}).get('list', [])
            if not ticker_list:
                return None
                
            ticker = ticker_list[0]
            current_price = float(ticker.get('lastPrice', 0))
            price_change_24h = float(ticker.get('price24hPcnt', 0)) * 100  # Convert to %
            
            # Get klines for recent price action
            klines_result = await client.get_klines(
                symbol=symbol,
                interval="5",  # 5 minute
                limit=self.ANALYSIS_PERIODS + 5
            )
            
            price_values = []
            if klines_result.get('success'):
                klines = klines_result.get('data', {}).get('list', [])
                for k in klines[:self.ANALYSIS_PERIODS]:
                    price_values.append({
                        'timestamp': int(k[0]),
                        'close': float(k[4])
                    })
                self.price_cache[symbol] = price_values
            
            # Analyze the data
            analysis = self._analyze_oi_pattern(symbol, oi_values, price_values, current_price)
            
            if analysis:
                self.analysis_cache[symbol] = analysis
                
                # Store in Redis for dashboard
                await self._store_analysis(symbol, analysis)
                
            return analysis
            
        except Exception as e:
            logger.debug(f"OI update error for {symbol}: {e}")
            return None
            
    def _analyze_oi_pattern(
        self, 
        symbol: str, 
        oi_data: List[Dict], 
        price_data: List[Dict],
        current_price: float
    ) -> Optional[OIAnalysis]:
        """
        Analyze OI and price patterns to generate signal
        """
        try:
            if len(oi_data) < 3:
                return None
                
            # Calculate OI change (oldest to newest)
            # Note: Bybit returns newest first, so reverse
            oi_values = [d['oi'] for d in reversed(oi_data)]
            
            if len(oi_values) < 3 or oi_values[0] == 0:
                return None
                
            # Recent OI change (last 3 periods)
            recent_oi = oi_values[-3:]
            oi_change_recent = ((recent_oi[-1] - recent_oi[0]) / recent_oi[0]) * 100 if recent_oi[0] > 0 else 0
            
            # Overall OI change
            oi_change_total = ((oi_values[-1] - oi_values[0]) / oi_values[0]) * 100 if oi_values[0] > 0 else 0
            
            # Calculate price change
            price_change_pct = 0
            if price_data and len(price_data) >= 3:
                price_values = [d['close'] for d in reversed(price_data)]
                if price_values[0] > 0:
                    price_change_pct = ((price_values[-1] - price_values[0]) / price_values[0]) * 100
            
            # Determine OI trend
            if oi_change_total > self.SIGNIFICANT_OI_CHANGE:
                oi_trend = 'rising'
            elif oi_change_total < -self.SIGNIFICANT_OI_CHANGE:
                oi_trend = 'falling'
            else:
                oi_trend = 'flat'
                
            # Generate signal based on OI + Price patterns
            signal, confidence, recommendation, reasoning = self._classify_pattern(
                oi_change_total, 
                oi_change_recent,
                price_change_pct,
                oi_trend
            )
            
            return OIAnalysis(
                symbol=symbol,
                signal=signal,
                oi_change_pct=round(oi_change_total, 2),
                price_change_pct=round(price_change_pct, 2),
                oi_trend=oi_trend,
                confidence=confidence,
                recommendation=recommendation,
                reasoning=reasoning,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.debug(f"OI analysis error: {e}")
            return None
            
    def _classify_pattern(
        self,
        oi_change: float,
        oi_change_recent: float,
        price_change: float,
        oi_trend: str
    ) -> Tuple[OISignal, int, str, str]:
        """
        Classify the OI/Price pattern into a signal
        
        Returns: (signal, confidence, recommendation, reasoning)
        """
        # Check for accumulation (big OI spike, flat price)
        if oi_change > self.ACCUMULATION_THRESHOLD and abs(price_change) < self.SIGNIFICANT_PRICE_CHANGE:
            return (
                OISignal.ACCUMULATION,
                85,
                "wait",
                f"Big OI spike (+{oi_change:.1f}%) with flat price - major move incoming"
            )
            
        # Check for distribution (big OI drop, flat price)
        if oi_change < -self.ACCUMULATION_THRESHOLD and abs(price_change) < self.SIGNIFICANT_PRICE_CHANGE:
            return (
                OISignal.DISTRIBUTION,
                70,
                "avoid",
                f"OI dropping ({oi_change:.1f}%) with flat price - interest fading"
            )
        
        # STRONG BULLISH: OI rising + Price rising
        if oi_change > self.SIGNIFICANT_OI_CHANGE and price_change > self.SIGNIFICANT_PRICE_CHANGE:
            confidence = min(95, 70 + int(oi_change) + int(price_change))
            return (
                OISignal.STRONG_BULLISH,
                confidence,
                "confirm",
                f"OI +{oi_change:.1f}% + Price +{price_change:.1f}% = New longs entering (REAL breakout)"
            )
            
        # STRONG BEARISH: OI rising + Price falling
        if oi_change > self.SIGNIFICANT_OI_CHANGE and price_change < -self.SIGNIFICANT_PRICE_CHANGE:
            confidence = min(95, 70 + int(oi_change) + int(abs(price_change)))
            return (
                OISignal.STRONG_BEARISH,
                confidence,
                "avoid_long",
                f"OI +{oi_change:.1f}% + Price {price_change:.1f}% = New shorts entering (DUMP incoming)"
            )
            
        # WEAK BULLISH: OI falling + Price rising
        if oi_change < -self.SIGNIFICANT_OI_CHANGE and price_change > self.SIGNIFICANT_PRICE_CHANGE:
            return (
                OISignal.WEAK_BULLISH,
                55,
                "caution",
                f"OI {oi_change:.1f}% + Price +{price_change:.1f}% = Shorts closing (weak rally, may reverse)"
            )
            
        # WEAK BEARISH: OI falling + Price falling (capitulation)
        if oi_change < -self.SIGNIFICANT_OI_CHANGE and price_change < -self.SIGNIFICANT_PRICE_CHANGE:
            return (
                OISignal.WEAK_BEARISH,
                60,
                "wait",
                f"OI {oi_change:.1f}% + Price {price_change:.1f}% = Longs capitulating (possible bottom)"
            )
            
        # No significant pattern
        return (
            OISignal.NEUTRAL,
            50,
            "neutral",
            f"OI {oi_change:+.1f}%, Price {price_change:+.1f}% - No clear signal"
        )
        
    async def _store_analysis(self, symbol: str, analysis: OIAnalysis):
        """Store analysis in Redis for dashboard access"""
        try:
            data = {
                'symbol': symbol,
                'signal': analysis.signal.value,
                'oi_change_pct': analysis.oi_change_pct,
                'price_change_pct': analysis.price_change_pct,
                'oi_trend': analysis.oi_trend,
                'confidence': analysis.confidence,
                'recommendation': analysis.recommendation,
                'reasoning': analysis.reasoning,
                'timestamp': analysis.timestamp.isoformat()
            }
            
            # Store individual symbol analysis
            await self.redis_client.hset('oi:analysis', symbol, json.dumps(data))
            
            # Store in sorted set for quick lookups (sorted by confidence)
            await self.redis_client.zadd(
                'oi:signals',
                {symbol: analysis.confidence}
            )
            
            # Expire after 15 minutes
            await self.redis_client.expire('oi:analysis', 900)
            await self.redis_client.expire('oi:signals', 900)
            
        except Exception as e:
            logger.debug(f"Failed to store OI analysis: {e}")
            
    async def get_analysis(self, symbol: str) -> Optional[OIAnalysis]:
        """Get cached analysis for a symbol"""
        # Check memory cache first
        if symbol in self.analysis_cache:
            analysis = self.analysis_cache[symbol]
            # Return if fresh (< 5 minutes old)
            if datetime.utcnow() - analysis.timestamp < timedelta(minutes=5):
                return analysis
                
        # Try Redis
        try:
            data = await self.redis_client.hget('oi:analysis', symbol)
            if data:
                d = json.loads(data)
                return OIAnalysis(
                    symbol=d['symbol'],
                    signal=OISignal(d['signal']),
                    oi_change_pct=d['oi_change_pct'],
                    price_change_pct=d['price_change_pct'],
                    oi_trend=d['oi_trend'],
                    confidence=d['confidence'],
                    recommendation=d['recommendation'],
                    reasoning=d['reasoning'],
                    timestamp=datetime.fromisoformat(d['timestamp'])
                )
        except:
            pass
            
        return None
        
    async def get_top_signals(self, limit: int = 10, signal_type: Optional[OISignal] = None) -> List[Dict]:
        """Get top OI signals sorted by confidence"""
        try:
            # Get all analyses
            all_data = await self.redis_client.hgetall('oi:analysis')
            
            signals = []
            for symbol, data in all_data.items():
                d = json.loads(data)
                
                # Filter by signal type if specified
                if signal_type and d['signal'] != signal_type.value:
                    continue
                    
                signals.append(d)
                
            # Sort by confidence descending
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            return signals[:limit]
            
        except Exception as e:
            logger.debug(f"Failed to get top signals: {e}")
            return []
            
    def should_confirm_entry(self, analysis: Optional[OIAnalysis], direction: str) -> Tuple[bool, str]:
        """
        Check if OI analysis confirms an entry
        
        Args:
            analysis: OIAnalysis object
            direction: 'long' or 'short'
            
        Returns:
            (should_enter, reason)
        """
        if not analysis:
            return True, "No OI data - proceed with caution"
            
        signal = analysis.signal
        
        if direction == 'long':
            if signal == OISignal.STRONG_BULLISH:
                return True, f"OI CONFIRMS LONG: {analysis.reasoning}"
            elif signal == OISignal.STRONG_BEARISH:
                return False, f"OI BLOCKS LONG: {analysis.reasoning}"
            elif signal == OISignal.WEAK_BULLISH:
                return True, f"OI CAUTION: {analysis.reasoning}"
            elif signal == OISignal.ACCUMULATION:
                return True, f"OI ACCUMULATION: {analysis.reasoning}"
            elif signal == OISignal.DISTRIBUTION:
                return False, f"OI DISTRIBUTION: {analysis.reasoning}"
                
        elif direction == 'short':
            if signal == OISignal.STRONG_BEARISH:
                return True, f"OI CONFIRMS SHORT: {analysis.reasoning}"
            elif signal == OISignal.STRONG_BULLISH:
                return False, f"OI BLOCKS SHORT: {analysis.reasoning}"
            elif signal == OISignal.WEAK_BEARISH:
                return True, f"OI CAUTION: {analysis.reasoning}"
                
        return True, f"OI NEUTRAL: {analysis.reasoning}"
        
    def get_oi_score_adjustment(self, analysis: Optional[OIAnalysis], direction: str) -> int:
        """
        Get score adjustment based on OI analysis
        
        Returns: Score adjustment (-20 to +20)
        """
        if not analysis:
            return 0
            
        signal = analysis.signal
        
        if direction == 'long':
            if signal == OISignal.STRONG_BULLISH:
                return 15  # Boost score
            elif signal == OISignal.STRONG_BEARISH:
                return -15  # Reduce score
            elif signal == OISignal.WEAK_BULLISH:
                return -5  # Slight penalty
            elif signal == OISignal.ACCUMULATION:
                return 10  # Good sign
            elif signal == OISignal.DISTRIBUTION:
                return -10  # Bad sign
                
        elif direction == 'short':
            if signal == OISignal.STRONG_BEARISH:
                return 15
            elif signal == OISignal.STRONG_BULLISH:
                return -15
            elif signal == OISignal.WEAK_BEARISH:
                return 5  # Capitulation can be good for shorts
                
        return 0


# Singleton instance
_oi_tracker: Optional[OpenInterestTracker] = None

async def get_oi_tracker() -> OpenInterestTracker:
    """Get or create the OI tracker singleton"""
    global _oi_tracker
    if _oi_tracker is None:
        _oi_tracker = OpenInterestTracker()
        await _oi_tracker.initialize()
    return _oi_tracker
