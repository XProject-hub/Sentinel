"""
Price Predictor - Multi-timeframe price prediction

Uses technical analysis and ML to predict price direction
"""

import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PricePrediction:
    """Price prediction result"""
    direction: str  # 'up', 'down', 'neutral'
    confidence: float  # 0-1
    predicted_change: float  # Expected % change
    timeframe: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class PricePredictor:
    """
    Multi-timeframe price prediction using technical indicators
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self._cache: Dict[str, PricePrediction] = {}
        
    async def initialize(self, redis_client=None):
        """Initialize with Redis client"""
        if redis_client:
            self.redis_client = redis_client
        logger.info("Price Predictor initialized")
        
    async def predict(self, symbol: str, klines: list = None, 
                     timeframe: str = '5m') -> PricePrediction:
        """
        Predict price direction for a symbol
        
        Args:
            symbol: Trading pair
            klines: OHLCV data
            timeframe: Prediction timeframe
            
        Returns:
            PricePrediction with direction and confidence
        """
        try:
            if not klines or len(klines) < 20:
                return PricePrediction(
                    direction='neutral',
                    confidence=0.3,
                    predicted_change=0.0,
                    timeframe=timeframe
                )
            
            # Extract price data
            closes = [float(k[4]) for k in klines[:50]]
            highs = [float(k[2]) for k in klines[:50]]
            lows = [float(k[3]) for k in klines[:50]]
            volumes = [float(k[5]) for k in klines[:50]]
            
            # Calculate indicators
            closes_arr = np.array(closes)
            
            # RSI
            rsi = self._calculate_rsi(closes_arr)
            
            # MACD
            macd_signal = self._calculate_macd_signal(closes_arr)
            
            # Moving average crossover
            sma_9 = np.mean(closes_arr[:9])
            sma_21 = np.mean(closes_arr[:21])
            ma_signal = 1 if sma_9 > sma_21 else -1 if sma_9 < sma_21 else 0
            
            # Volume trend
            recent_vol = np.mean(volumes[:5])
            avg_vol = np.mean(volumes[:20])
            vol_signal = 1 if recent_vol > avg_vol * 1.2 else 0
            
            # Combine signals
            total_signal = 0
            weights = {'rsi': 0.3, 'macd': 0.3, 'ma': 0.25, 'vol': 0.15}
            
            # RSI signal
            if rsi < 30:
                total_signal += weights['rsi'] * 1  # Oversold = bullish
            elif rsi > 70:
                total_signal -= weights['rsi'] * 1  # Overbought = bearish
            
            # MACD signal
            total_signal += weights['macd'] * macd_signal
            
            # MA signal
            total_signal += weights['ma'] * ma_signal
            
            # Volume confirmation
            total_signal += weights['vol'] * vol_signal * np.sign(total_signal)
            
            # Determine direction
            if total_signal > 0.2:
                direction = 'up'
            elif total_signal < -0.2:
                direction = 'down'
            else:
                direction = 'neutral'
            
            # Confidence based on signal strength
            confidence = min(0.9, abs(total_signal) + 0.3)
            
            # Predicted change based on recent volatility
            volatility = np.std(closes_arr[:10]) / np.mean(closes_arr[:10]) * 100
            predicted_change = total_signal * volatility
            
            return PricePrediction(
                direction=direction,
                confidence=confidence,
                predicted_change=predicted_change,
                timeframe=timeframe
            )
            
        except Exception as e:
            logger.error(f"Price prediction failed for {symbol}: {e}")
            return PricePrediction(
                direction='neutral',
                confidence=0.3,
                predicted_change=0.0,
                timeframe=timeframe
            )
    
    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(closes) < period + 1:
                return 50.0
            
            deltas = np.diff(closes[:period + 1])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
        except:
            return 50.0
    
    def _calculate_macd_signal(self, closes: np.ndarray) -> float:
        """Calculate MACD signal (-1 to 1)"""
        try:
            if len(closes) < 26:
                return 0.0
            
            # Simple EMA approximation
            ema_12 = np.mean(closes[:12])
            ema_26 = np.mean(closes[:26])
            
            macd = ema_12 - ema_26
            signal = np.mean(closes[:9])  # Simplified signal line
            
            # Normalize to -1 to 1
            diff = (macd - signal) / closes[0] * 100 if closes[0] > 0 else 0
            return float(np.clip(diff * 10, -1, 1))
        except:
            return 0.0


# Singleton instance
price_predictor = PricePredictor()

