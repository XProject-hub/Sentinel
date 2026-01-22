"""
Market Regime Detector - Identifies current market conditions

Regimes:
- BULL: Strong uptrend
- BEAR: Strong downtrend  
- RANGE: Sideways/consolidation
- VOLATILE: High volatility, uncertain direction
- CHOPPY: Erratic price action
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current market regime state"""
    regime: str  # BULL, BEAR, RANGE, VOLATILE, CHOPPY
    confidence: float  # 0-1
    volatility: float  # ATR-based
    trend_strength: float  # -1 to 1
    recommended_action: str = 'HOLD'  # BUY, SELL, HOLD
    liquidity_score: float = 50.0  # 0-100
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        # Auto-set recommended action based on regime
        if not self.recommended_action or self.recommended_action == 'HOLD':
            if self.regime == 'BULL':
                self.recommended_action = 'BUY'
            elif self.regime == 'BEAR':
                self.recommended_action = 'SELL'
            else:
                self.recommended_action = 'HOLD'


class RegimeDetector:
    """
    Detects current market regime using multiple indicators
    
    Methods:
    - detect_regime(symbol): Get current regime for a symbol
    - get_current_regime(): Get overall market regime (BTC-based)
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self._cache: Dict[str, RegimeState] = {}
        self._cache_ttl = 60  # 1 minute cache
        self._last_update: Dict[str, datetime] = {}
        
    async def initialize(self, redis_client=None):
        """Initialize with Redis client"""
        if redis_client:
            self.redis_client = redis_client
        logger.info("Regime Detector initialized")
        
    async def detect_regime(self, symbol: str, klines: list = None) -> RegimeState:
        """
        Detect market regime for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            klines: Optional kline data (if not provided, uses cached)
            
        Returns:
            RegimeState with regime classification
        """
        try:
            # Check cache
            if symbol in self._cache:
                last_update = self._last_update.get(symbol, datetime.min)
                if (datetime.utcnow() - last_update).total_seconds() < self._cache_ttl:
                    return self._cache[symbol]
            
            # Calculate regime from klines if provided
            if klines and len(klines) >= 20:
                regime = self._calculate_regime(klines)
            else:
                # Default to neutral regime
                regime = RegimeState(
                    regime='RANGE',
                    confidence=0.5,
                    volatility=1.5,
                    trend_strength=0.0
                )
            
            # Cache result
            self._cache[symbol] = regime
            self._last_update[symbol] = datetime.utcnow()
            
            return regime
            
        except Exception as e:
            logger.error(f"Regime detection failed for {symbol}: {e}")
            return RegimeState(
                regime='RANGE',
                confidence=0.3,
                volatility=1.5,
                trend_strength=0.0,
                recommended_action='HOLD',
                liquidity_score=50.0
            )
    
    def _calculate_regime(self, klines: list) -> RegimeState:
        """Calculate regime from kline data"""
        try:
            # Extract price data
            closes = [float(k[4]) for k in klines[:50]]
            highs = [float(k[2]) for k in klines[:50]]
            lows = [float(k[3]) for k in klines[:50]]
            
            if len(closes) < 20:
                return RegimeState(regime='RANGE', confidence=0.5, volatility=1.5, trend_strength=0.0, recommended_action='HOLD', liquidity_score=50.0)
            
            # Calculate indicators
            closes_arr = np.array(closes)
            
            # SMA for trend
            sma_20 = np.mean(closes_arr[:20])
            sma_50 = np.mean(closes_arr[:50]) if len(closes_arr) >= 50 else sma_20
            
            # Price relative to SMAs
            current_price = closes_arr[0]
            above_sma20 = current_price > sma_20
            above_sma50 = current_price > sma_50
            
            # Trend strength (-1 to 1)
            price_change_20 = (current_price - closes_arr[19]) / closes_arr[19] if closes_arr[19] > 0 else 0
            trend_strength = np.clip(price_change_20 * 10, -1, 1)  # Scale to -1 to 1
            
            # Volatility (ATR-like)
            ranges = [highs[i] - lows[i] for i in range(min(14, len(highs)))]
            avg_range = np.mean(ranges) if ranges else 0
            volatility = (avg_range / current_price * 100) if current_price > 0 else 1.5
            
            # Determine regime
            if volatility > 3.0:
                regime = 'VOLATILE'
                confidence = min(0.9, volatility / 5)
            elif abs(trend_strength) > 0.5:
                if trend_strength > 0:
                    regime = 'BULL'
                else:
                    regime = 'BEAR'
                confidence = min(0.9, abs(trend_strength))
            elif volatility > 1.5:
                regime = 'CHOPPY'
                confidence = 0.6
            else:
                regime = 'RANGE'
                confidence = 0.7
            
            # Determine recommended action
            if regime == 'BULL' and trend_strength > 0.3:
                action = 'BUY'
            elif regime == 'BEAR' and trend_strength < -0.3:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # Estimate liquidity score (based on volatility - lower vol = higher liquidity usually)
            liquidity = max(0, min(100, 100 - volatility * 20))
            
            return RegimeState(
                regime=regime,
                confidence=confidence,
                volatility=volatility,
                trend_strength=trend_strength,
                recommended_action=action,
                liquidity_score=liquidity
            )
            
        except Exception as e:
            logger.error(f"Regime calculation failed: {e}")
            return RegimeState(regime='RANGE', confidence=0.5, volatility=1.5, trend_strength=0.0, recommended_action='HOLD', liquidity_score=50.0)
    
    def get_current_regime(self) -> Dict[str, Any]:
        """Get current overall market regime (from cache)"""
        btc_regime = self._cache.get('BTCUSDT')
        if btc_regime:
            return {
                'regime': btc_regime.regime,
                'confidence': btc_regime.confidence,
                'volatility': btc_regime.volatility,
                'trend_strength': btc_regime.trend_strength
            }
        return {
            'regime': 'RANGE',
            'confidence': 0.5,
            'volatility': 1.5,
            'trend_strength': 0.0
        }
    
    async def shutdown(self):
        """Cleanup resources"""
        self._cache.clear()
        self._last_update.clear()
        logger.info("Regime Detector shutdown complete")


# Singleton instance
regime_detector = RegimeDetector()

