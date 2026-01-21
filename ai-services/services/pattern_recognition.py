"""
SENTINEL AI - Advanced Pattern Recognition
Detects chart patterns with high accuracy

Features:
- Candlestick pattern recognition
- Support/Resistance detection
- Trend analysis
- Volume profile analysis
- Multi-timeframe confirmation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from loguru import logger
import redis.asyncio as redis
import json
import httpx

from config import settings


@dataclass
class PatternSignal:
    """Detected pattern with trading signal"""
    symbol: str
    pattern_name: str
    pattern_type: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-100
    confidence: float  # 0-100
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward: float
    timeframe: str
    description: str
    timestamp: str


class PatternRecognition:
    """
    Advanced Pattern Recognition System
    
    Detects multiple pattern types:
    - Candlestick patterns (doji, hammer, engulfing, etc.)
    - Chart patterns (double top/bottom, head & shoulders, triangles)
    - Support/Resistance levels
    - Trend channels
    """
    
    def __init__(self):
        self.redis_client = None
        self.http_client = None
        
        # Pattern statistics
        self.patterns_detected = 0
        self.pattern_accuracy: Dict[str, Dict] = {}
        
        # Support/Resistance cache
        self.sr_levels: Dict[str, List[float]] = {}
        
    async def initialize(self):
        """Initialize pattern recognition"""
        logger.info("Initializing Pattern Recognition System...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load pattern accuracy history
        await self._load_pattern_stats()
        
        logger.info("Pattern Recognition initialized - Scanning for patterns")
        
    async def shutdown(self):
        """Cleanup"""
        await self._save_pattern_stats()
        if self.http_client:
            await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def analyze_patterns(self, symbol: str) -> List[PatternSignal]:
        """
        Analyze symbol for all pattern types
        Returns list of detected patterns with signals
        """
        patterns = []
        
        try:
            # Fetch multi-timeframe data
            data_15m = await self._fetch_ohlcv(symbol, '15')
            data_1h = await self._fetch_ohlcv(symbol, '60')
            data_4h = await self._fetch_ohlcv(symbol, '240')
            
            current_price = data_15m[-1, 3] if data_15m is not None else 0
            
            # Detect candlestick patterns (15m for quick signals)
            if data_15m is not None:
                candle_patterns = self._detect_candlestick_patterns(data_15m, symbol, '15m')
                patterns.extend(candle_patterns)
                
            # Detect chart patterns (1h for medium-term)
            if data_1h is not None:
                chart_patterns = self._detect_chart_patterns(data_1h, symbol, '1h')
                patterns.extend(chart_patterns)
                
                # Calculate support/resistance
                self.sr_levels[symbol] = self._calculate_support_resistance(data_1h)
                
            # Detect trend patterns (4h for confirmation)
            if data_4h is not None:
                trend_patterns = self._detect_trend_patterns(data_4h, symbol, '4h')
                patterns.extend(trend_patterns)
                
            # Add support/resistance based signals
            if symbol in self.sr_levels and current_price > 0:
                sr_patterns = self._analyze_sr_levels(symbol, current_price, self.sr_levels[symbol])
                patterns.extend(sr_patterns)
                
            self.patterns_detected += len(patterns)
            
            # Store detected patterns
            if patterns:
                await self._store_patterns(symbol, patterns)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern analysis error for {symbol}: {e}")
            return []
            
    def _detect_candlestick_patterns(self, data: np.ndarray, symbol: str, 
                                      timeframe: str) -> List[PatternSignal]:
        """Detect candlestick patterns"""
        patterns = []
        
        if len(data) < 5:
            return patterns
            
        # Get last few candles
        opens = data[-5:, 0]
        highs = data[-5:, 1]
        lows = data[-5:, 2]
        closes = data[-5:, 3]
        
        current_price = closes[-1]
        
        # Calculate candle properties
        body = closes[-1] - opens[-1]
        upper_wick = highs[-1] - max(opens[-1], closes[-1])
        lower_wick = min(opens[-1], closes[-1]) - lows[-1]
        candle_range = highs[-1] - lows[-1]
        
        prev_body = closes[-2] - opens[-2]
        
        # === BULLISH PATTERNS ===
        
        # Hammer (bullish reversal)
        if candle_range > 0:
            body_ratio = abs(body) / candle_range
            lower_wick_ratio = lower_wick / candle_range
            
            if body_ratio < 0.3 and lower_wick_ratio > 0.6 and body >= 0:
                patterns.append(PatternSignal(
                    symbol=symbol,
                    pattern_name='Hammer',
                    pattern_type='bullish',
                    strength=75,
                    confidence=70,
                    entry_price=current_price,
                    target_price=current_price * 1.02,
                    stop_loss=lows[-1] * 0.995,
                    risk_reward=2.0,
                    timeframe=timeframe,
                    description='Bullish reversal - hammer pattern with long lower wick',
                    timestamp=datetime.utcnow().isoformat()
                ))
                
        # Bullish Engulfing
        if prev_body < 0 and body > 0 and abs(body) > abs(prev_body) * 1.5:
            if opens[-1] <= closes[-2] and closes[-1] >= opens[-2]:
                patterns.append(PatternSignal(
                    symbol=symbol,
                    pattern_name='Bullish Engulfing',
                    pattern_type='bullish',
                    strength=80,
                    confidence=75,
                    entry_price=current_price,
                    target_price=current_price * 1.025,
                    stop_loss=lows[-1] * 0.99,
                    risk_reward=2.5,
                    timeframe=timeframe,
                    description='Strong bullish reversal - current candle engulfs previous',
                    timestamp=datetime.utcnow().isoformat()
                ))
                
        # Morning Star (3 candle bullish reversal)
        if len(data) >= 3:
            body_3 = closes[-3] - opens[-3]
            body_2 = closes[-2] - opens[-2]
            body_1 = closes[-1] - opens[-1]
            
            if body_3 < 0 and abs(body_2) < abs(body_3) * 0.3 and body_1 > 0:
                if closes[-1] > (opens[-3] + closes[-3]) / 2:
                    patterns.append(PatternSignal(
                        symbol=symbol,
                        pattern_name='Morning Star',
                        pattern_type='bullish',
                        strength=85,
                        confidence=80,
                        entry_price=current_price,
                        target_price=current_price * 1.03,
                        stop_loss=lows[-2] * 0.99,
                        risk_reward=3.0,
                        timeframe=timeframe,
                        description='Strong bullish reversal - 3 candle morning star pattern',
                        timestamp=datetime.utcnow().isoformat()
                    ))
                    
        # === BEARISH PATTERNS ===
        
        # Shooting Star (bearish reversal)
        if candle_range > 0:
            upper_wick_ratio = upper_wick / candle_range
            
            if body_ratio < 0.3 and upper_wick_ratio > 0.6 and body <= 0:
                patterns.append(PatternSignal(
                    symbol=symbol,
                    pattern_name='Shooting Star',
                    pattern_type='bearish',
                    strength=75,
                    confidence=70,
                    entry_price=current_price,
                    target_price=current_price * 0.98,
                    stop_loss=highs[-1] * 1.005,
                    risk_reward=2.0,
                    timeframe=timeframe,
                    description='Bearish reversal - shooting star with long upper wick',
                    timestamp=datetime.utcnow().isoformat()
                ))
                
        # Bearish Engulfing
        if prev_body > 0 and body < 0 and abs(body) > abs(prev_body) * 1.5:
            if opens[-1] >= closes[-2] and closes[-1] <= opens[-2]:
                patterns.append(PatternSignal(
                    symbol=symbol,
                    pattern_name='Bearish Engulfing',
                    pattern_type='bearish',
                    strength=80,
                    confidence=75,
                    entry_price=current_price,
                    target_price=current_price * 0.975,
                    stop_loss=highs[-1] * 1.01,
                    risk_reward=2.5,
                    timeframe=timeframe,
                    description='Strong bearish reversal - current candle engulfs previous',
                    timestamp=datetime.utcnow().isoformat()
                ))
                
        # Doji (indecision - look at context)
        if candle_range > 0 and abs(body) / candle_range < 0.1:
            # Doji after uptrend = bearish
            trend = self._calculate_trend(closes[:-1])
            if trend > 0.5:
                patterns.append(PatternSignal(
                    symbol=symbol,
                    pattern_name='Doji (Top)',
                    pattern_type='bearish',
                    strength=60,
                    confidence=55,
                    entry_price=current_price,
                    target_price=current_price * 0.985,
                    stop_loss=highs[-1] * 1.01,
                    risk_reward=1.5,
                    timeframe=timeframe,
                    description='Indecision after uptrend - potential reversal',
                    timestamp=datetime.utcnow().isoformat()
                ))
            elif trend < -0.5:
                patterns.append(PatternSignal(
                    symbol=symbol,
                    pattern_name='Doji (Bottom)',
                    pattern_type='bullish',
                    strength=60,
                    confidence=55,
                    entry_price=current_price,
                    target_price=current_price * 1.015,
                    stop_loss=lows[-1] * 0.99,
                    risk_reward=1.5,
                    timeframe=timeframe,
                    description='Indecision after downtrend - potential reversal',
                    timestamp=datetime.utcnow().isoformat()
                ))
                
        return patterns
        
    def _detect_chart_patterns(self, data: np.ndarray, symbol: str,
                                timeframe: str) -> List[PatternSignal]:
        """Detect larger chart patterns"""
        patterns = []
        
        if len(data) < 30:
            return patterns
            
        closes = data[:, 3]
        highs = data[:, 1]
        lows = data[:, 2]
        current_price = closes[-1]
        
        # Find local highs and lows
        local_highs = self._find_local_extrema(highs, 'max')
        local_lows = self._find_local_extrema(lows, 'min')
        
        # === Double Bottom ===
        if len(local_lows) >= 2:
            last_two_lows = local_lows[-2:]
            low1_price = lows[last_two_lows[0]]
            low2_price = lows[last_two_lows[1]]
            
            # Check if lows are similar (within 2%)
            if abs(low1_price - low2_price) / low1_price < 0.02:
                # Check if price has bounced
                if current_price > max(low1_price, low2_price) * 1.01:
                    neckline = max(closes[last_two_lows[0]:last_two_lows[1]])
                    target = neckline + (neckline - min(low1_price, low2_price))
                    
                    patterns.append(PatternSignal(
                        symbol=symbol,
                        pattern_name='Double Bottom',
                        pattern_type='bullish',
                        strength=85,
                        confidence=80,
                        entry_price=current_price,
                        target_price=target,
                        stop_loss=min(low1_price, low2_price) * 0.99,
                        risk_reward=2.5,
                        timeframe=timeframe,
                        description=f'Double bottom at ${min(low1_price, low2_price):.2f} - bullish reversal',
                        timestamp=datetime.utcnow().isoformat()
                    ))
                    
        # === Double Top ===
        if len(local_highs) >= 2:
            last_two_highs = local_highs[-2:]
            high1_price = highs[last_two_highs[0]]
            high2_price = highs[last_two_highs[1]]
            
            if abs(high1_price - high2_price) / high1_price < 0.02:
                if current_price < min(high1_price, high2_price) * 0.99:
                    neckline = min(closes[last_two_highs[0]:last_two_highs[1]])
                    target = neckline - (max(high1_price, high2_price) - neckline)
                    
                    patterns.append(PatternSignal(
                        symbol=symbol,
                        pattern_name='Double Top',
                        pattern_type='bearish',
                        strength=85,
                        confidence=80,
                        entry_price=current_price,
                        target_price=target,
                        stop_loss=max(high1_price, high2_price) * 1.01,
                        risk_reward=2.5,
                        timeframe=timeframe,
                        description=f'Double top at ${max(high1_price, high2_price):.2f} - bearish reversal',
                        timestamp=datetime.utcnow().isoformat()
                    ))
                    
        # === Ascending Triangle ===
        if len(local_highs) >= 3 and len(local_lows) >= 3:
            recent_highs = [highs[i] for i in local_highs[-3:]]
            recent_lows = [lows[i] for i in local_lows[-3:]]
            
            # Flat top, rising bottom
            highs_flat = max(recent_highs) - min(recent_highs) < max(recent_highs) * 0.01
            lows_rising = recent_lows[-1] > recent_lows[0] * 1.02
            
            if highs_flat and lows_rising:
                resistance = max(recent_highs)
                if current_price > resistance * 0.99:  # Near breakout
                    patterns.append(PatternSignal(
                        symbol=symbol,
                        pattern_name='Ascending Triangle',
                        pattern_type='bullish',
                        strength=75,
                        confidence=70,
                        entry_price=current_price,
                        target_price=resistance * 1.03,
                        stop_loss=min(recent_lows) * 0.99,
                        risk_reward=2.0,
                        timeframe=timeframe,
                        description=f'Ascending triangle - resistance at ${resistance:.2f}',
                        timestamp=datetime.utcnow().isoformat()
                    ))
                    
        return patterns
        
    def _detect_trend_patterns(self, data: np.ndarray, symbol: str,
                                timeframe: str) -> List[PatternSignal]:
        """Detect trend-based patterns"""
        patterns = []
        
        if len(data) < 50:
            return patterns
            
        closes = data[:, 3]
        current_price = closes[-1]
        
        # Calculate moving averages
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        
        # Calculate trend strength
        trend_strength = (sma_20 - sma_50) / sma_50 * 100
        
        # === Golden Cross (20 crosses above 50) ===
        prev_sma_20 = np.mean(closes[-21:-1])
        prev_sma_50 = np.mean(closes[-51:-1])
        
        if prev_sma_20 < prev_sma_50 and sma_20 > sma_50:
            patterns.append(PatternSignal(
                symbol=symbol,
                pattern_name='Golden Cross',
                pattern_type='bullish',
                strength=90,
                confidence=85,
                entry_price=current_price,
                target_price=current_price * 1.05,
                stop_loss=current_price * 0.97,
                risk_reward=1.67,
                timeframe=timeframe,
                description='20 SMA crossed above 50 SMA - strong bullish signal',
                timestamp=datetime.utcnow().isoformat()
            ))
            
        # === Death Cross (20 crosses below 50) ===
        if prev_sma_20 > prev_sma_50 and sma_20 < sma_50:
            patterns.append(PatternSignal(
                symbol=symbol,
                pattern_name='Death Cross',
                pattern_type='bearish',
                strength=90,
                confidence=85,
                entry_price=current_price,
                target_price=current_price * 0.95,
                stop_loss=current_price * 1.03,
                risk_reward=1.67,
                timeframe=timeframe,
                description='20 SMA crossed below 50 SMA - strong bearish signal',
                timestamp=datetime.utcnow().isoformat()
            ))
            
        # === Strong Uptrend ===
        if trend_strength > 3 and current_price > sma_20 > sma_50:
            patterns.append(PatternSignal(
                symbol=symbol,
                pattern_name='Strong Uptrend',
                pattern_type='bullish',
                strength=70,
                confidence=75,
                entry_price=current_price,
                target_price=current_price * 1.03,
                stop_loss=sma_20 * 0.99,
                risk_reward=2.0,
                timeframe=timeframe,
                description=f'Strong uptrend - price above both MAs ({trend_strength:.1f}% spread)',
                timestamp=datetime.utcnow().isoformat()
            ))
            
        # === Strong Downtrend ===
        if trend_strength < -3 and current_price < sma_20 < sma_50:
            patterns.append(PatternSignal(
                symbol=symbol,
                pattern_name='Strong Downtrend',
                pattern_type='bearish',
                strength=70,
                confidence=75,
                entry_price=current_price,
                target_price=current_price * 0.97,
                stop_loss=sma_20 * 1.01,
                risk_reward=2.0,
                timeframe=timeframe,
                description=f'Strong downtrend - price below both MAs ({abs(trend_strength):.1f}% spread)',
                timestamp=datetime.utcnow().isoformat()
            ))
            
        return patterns
        
    def _calculate_support_resistance(self, data: np.ndarray) -> List[float]:
        """Calculate key support and resistance levels"""
        closes = data[:, 3]
        highs = data[:, 1]
        lows = data[:, 2]
        
        levels = []
        
        # Recent high and low
        levels.append(max(highs[-20:]))
        levels.append(min(lows[-20:]))
        
        # Find swing points
        for i in range(2, len(data) - 2):
            # Swing high
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                levels.append(highs[i])
                
            # Swing low
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                levels.append(lows[i])
                
        # Cluster similar levels
        levels = sorted(set(levels))
        clustered = []
        
        for level in levels:
            # Check if close to existing cluster
            added = False
            for i, c in enumerate(clustered):
                if abs(level - c) / c < 0.01:  # Within 1%
                    clustered[i] = (c + level) / 2  # Average
                    added = True
                    break
            if not added:
                clustered.append(level)
                
        return sorted(clustered)
        
    def _analyze_sr_levels(self, symbol: str, current_price: float,
                           levels: List[float]) -> List[PatternSignal]:
        """Generate signals based on support/resistance"""
        patterns = []
        
        # Find nearest support and resistance
        supports = [l for l in levels if l < current_price]
        resistances = [l for l in levels if l > current_price]
        
        if supports:
            nearest_support = max(supports)
            distance = (current_price - nearest_support) / current_price * 100
            
            if distance < 1:  # Within 1% of support
                patterns.append(PatternSignal(
                    symbol=symbol,
                    pattern_name='Support Bounce',
                    pattern_type='bullish',
                    strength=65,
                    confidence=60,
                    entry_price=current_price,
                    target_price=current_price * 1.02,
                    stop_loss=nearest_support * 0.99,
                    risk_reward=2.0,
                    timeframe='1h',
                    description=f'Price near support at ${nearest_support:.2f}',
                    timestamp=datetime.utcnow().isoformat()
                ))
                
        if resistances:
            nearest_resistance = min(resistances)
            distance = (nearest_resistance - current_price) / current_price * 100
            
            if distance < 1:  # Within 1% of resistance
                patterns.append(PatternSignal(
                    symbol=symbol,
                    pattern_name='Resistance Test',
                    pattern_type='bearish',
                    strength=65,
                    confidence=60,
                    entry_price=current_price,
                    target_price=current_price * 0.98,
                    stop_loss=nearest_resistance * 1.01,
                    risk_reward=2.0,
                    timeframe='1h',
                    description=f'Price testing resistance at ${nearest_resistance:.2f}',
                    timestamp=datetime.utcnow().isoformat()
                ))
                
        return patterns
        
    def _find_local_extrema(self, data: np.ndarray, extrema_type: str,
                            window: int = 5) -> List[int]:
        """Find local maxima or minima"""
        extrema = []
        
        for i in range(window, len(data) - window):
            if extrema_type == 'max':
                if data[i] == max(data[i-window:i+window+1]):
                    extrema.append(i)
            else:
                if data[i] == min(data[i-window:i+window+1]):
                    extrema.append(i)
                    
        return extrema
        
    def _calculate_trend(self, closes: np.ndarray) -> float:
        """Calculate trend direction and strength (-1 to 1)"""
        if len(closes) < 2:
            return 0
            
        # Simple linear regression slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        
        # Normalize by average price
        avg_price = np.mean(closes)
        normalized_slope = slope / avg_price * len(closes)
        
        return np.clip(normalized_slope, -1, 1)
        
    async def _fetch_ohlcv(self, symbol: str, interval: str) -> Optional[np.ndarray]:
        """Fetch OHLCV data"""
        try:
            url = f"https://api.bybit.com/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': interval,
                'limit': 200
            }
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            klines = data.get('result', {}).get('list', [])
            
            if not klines:
                return None
                
            ohlcv = []
            for k in reversed(klines):
                ohlcv.append([
                    float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
                ])
                
            return np.array(ohlcv)
            
        except Exception as e:
            logger.debug(f"OHLCV fetch error: {e}")
            return None
            
    async def _store_patterns(self, symbol: str, patterns: List[PatternSignal]):
        """Store detected patterns"""
        try:
            for pattern in patterns:
                await self.redis_client.lpush(
                    f'patterns:{symbol}',
                    json.dumps({
                        'name': pattern.pattern_name,
                        'type': pattern.pattern_type,
                        'strength': pattern.strength,
                        'confidence': pattern.confidence,
                        'timeframe': pattern.timeframe,
                        'description': pattern.description,
                        'timestamp': pattern.timestamp
                    })
                )
            await self.redis_client.ltrim(f'patterns:{symbol}', 0, 49)
        except:
            pass
            
    async def _load_pattern_stats(self):
        """Load pattern accuracy statistics"""
        try:
            data = await self.redis_client.get('ai:pattern:stats')
            if data:
                self.pattern_accuracy = json.loads(data)
        except:
            pass
            
    async def _save_pattern_stats(self):
        """Save pattern statistics"""
        try:
            await self.redis_client.set('ai:pattern:stats', json.dumps(self.pattern_accuracy))
        except:
            pass
            
    async def get_pattern_stats(self) -> Dict:
        """Get pattern detection statistics"""
        return {
            'patterns_detected': self.patterns_detected,
            'pattern_accuracy': self.pattern_accuracy,
            'sr_levels_tracked': len(self.sr_levels)
        }


# Global instance
pattern_recognition = PatternRecognition()



