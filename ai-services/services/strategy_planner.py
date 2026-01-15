"""
SENTINEL AI - Strategy Planning Engine
Market regime detection and strategy selection
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from loguru import logger
import redis.asyncio as redis
import json

from config import settings


class MarketRegime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"


class Strategy(Enum):
    MOMENTUM = "momentum"
    GRID = "grid"
    SCALPING = "scalping"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    HEDGE = "hedge"
    HOLD = "hold"  # No trading


class StrategyPlanner:
    """
    AI-powered strategy planning engine that:
    1. Detects current market regime
    2. Selects optimal strategy for conditions
    3. Optimizes execution parameters
    4. Adapts to changing conditions
    """
    
    def __init__(self):
        self.redis_client = None
        self.active_strategies: Dict[str, Dict] = {}
        
        # Regime detection thresholds
        self.volatility_high_threshold = 3.0  # ATR %
        self.volatility_low_threshold = 0.8
        self.trend_strength_threshold = 2.0  # %
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Strategy-regime mapping
        self.regime_strategy_map = {
            MarketRegime.BULL_TREND: [Strategy.MOMENTUM, Strategy.BREAKOUT],
            MarketRegime.BEAR_TREND: [Strategy.HEDGE, Strategy.HOLD],
            MarketRegime.SIDEWAYS: [Strategy.GRID, Strategy.MEAN_REVERSION],
            MarketRegime.HIGH_VOLATILITY: [Strategy.SCALPING, Strategy.HOLD],
            MarketRegime.LOW_LIQUIDITY: [Strategy.HOLD],
            MarketRegime.BREAKOUT_UP: [Strategy.BREAKOUT, Strategy.MOMENTUM],
            MarketRegime.BREAKOUT_DOWN: [Strategy.HEDGE, Strategy.HOLD],
        }
        
        # Strategy parameters templates
        self.strategy_params = {
            Strategy.MOMENTUM: {
                'lookback_period': 20,
                'entry_threshold': 2.0,
                'exit_threshold': -0.5,
                'position_size': 0.1,
                'max_hold_hours': 48,
            },
            Strategy.GRID: {
                'grid_levels': 10,
                'grid_spacing': 0.5,
                'position_per_level': 0.05,
                'take_profit_percent': 1.0,
            },
            Strategy.SCALPING: {
                'tick_threshold': 3,
                'max_hold_seconds': 120,
                'position_size': 0.05,
                'profit_target': 0.3,
            },
            Strategy.MEAN_REVERSION: {
                'std_dev_entry': 2.0,
                'std_dev_exit': 0.5,
                'lookback_period': 50,
                'position_size': 0.1,
            },
            Strategy.BREAKOUT: {
                'atr_multiplier': 1.5,
                'confirmation_bars': 2,
                'position_size': 0.15,
                'trailing_stop': 2.0,
            },
            Strategy.HEDGE: {
                'hedge_ratio': 0.5,
                'rebalance_threshold': 5.0,
            },
        }
        
    async def initialize(self):
        """Initialize strategy planner"""
        logger.info("Initializing Strategy Planner...")
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        logger.info("Strategy Planner initialized")
        
    async def detect_regime(
        self, 
        market_state: Dict[str, Any], 
        sentiment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect current market regime based on multiple factors:
        - Price trend and strength
        - Volatility levels
        - Volume patterns
        - Sentiment indicators
        """
        # Default to first symbol or BTCUSDT
        symbol = 'BTCUSDT'
        
        if symbol not in market_state or not market_state[symbol].get('indicators'):
            return self._default_regime_response()
            
        indicators = market_state[symbol]['indicators']
        price_data = market_state[symbol].get('price', {})
        orderbook = market_state[symbol].get('orderbook', {})
        
        # Extract metrics
        volatility = float(indicators.get('volatility_percent', 1.5))
        rsi = float(indicators.get('rsi', 50))
        trend = indicators.get('trend', 'sideways')
        trend_strength = float(indicators.get('trend_strength', 0))
        
        # Get sentiment impact
        btc_sentiment = sentiment_data.get('BTC', {})
        sentiment_score = float(btc_sentiment.get('sentiment_score', 0))
        sentiment_momentum = float(btc_sentiment.get('sentiment_momentum', 0))
        
        # Determine regime
        regime = self._classify_regime(
            volatility, rsi, trend, trend_strength, 
            sentiment_score, sentiment_momentum
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            volatility, rsi, trend_strength, sentiment_score
        )
        
        # Select recommended strategy
        recommended_strategies = self.regime_strategy_map.get(regime, [Strategy.HOLD])
        recommended_strategy = recommended_strategies[0] if recommended_strategies else Strategy.HOLD
        
        # Get optimized parameters
        strategy_params = self._optimize_strategy_params(
            recommended_strategy, 
            volatility, 
            sentiment_score
        )
        
        result = {
            'regime': regime.value,
            'confidence': round(confidence, 2),
            'volatility': round(volatility, 2),
            'volatility_level': self._classify_volatility(volatility),
            'trend': trend,
            'trend_strength': round(trend_strength, 2),
            'rsi': round(rsi, 2),
            'sentiment_score': round(sentiment_score, 4),
            'sentiment_momentum': round(sentiment_momentum, 4),
            'recommended_strategy': recommended_strategy.value,
            'strategy_params': strategy_params,
            'alternative_strategies': [s.value for s in recommended_strategies[1:]],
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # Cache the result
        await self.redis_client.hset(
            f"regime:{symbol}",
            mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in result.items()}
        )
        
        return result
        
    def _classify_regime(
        self,
        volatility: float,
        rsi: float,
        trend: str,
        trend_strength: float,
        sentiment: float,
        sentiment_momentum: float
    ) -> MarketRegime:
        """Classify market regime based on indicators"""
        
        # High volatility takes precedence
        if volatility > self.volatility_high_threshold:
            return MarketRegime.HIGH_VOLATILITY
            
        # Detect breakouts
        if rsi > 75 and trend_strength > self.trend_strength_threshold and sentiment > 0.3:
            return MarketRegime.BREAKOUT_UP
            
        if rsi < 25 and trend_strength > self.trend_strength_threshold and sentiment < -0.3:
            return MarketRegime.BREAKOUT_DOWN
            
        # Detect trends
        if trend == 'bullish' and trend_strength > self.trend_strength_threshold:
            return MarketRegime.BULL_TREND
            
        if trend == 'bearish' and trend_strength > self.trend_strength_threshold:
            return MarketRegime.BEAR_TREND
            
        # Low volatility / sideways
        if volatility < self.volatility_low_threshold:
            return MarketRegime.LOW_LIQUIDITY
            
        # Default to sideways
        return MarketRegime.SIDEWAYS
        
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < self.volatility_low_threshold:
            return 'low'
        elif volatility > self.volatility_high_threshold:
            return 'high'
        else:
            return 'normal'
            
    def _calculate_confidence(
        self,
        volatility: float,
        rsi: float,
        trend_strength: float,
        sentiment: float
    ) -> float:
        """Calculate confidence in regime detection"""
        
        # Start with base confidence
        confidence = 0.5
        
        # Stronger signals increase confidence
        if 30 <= rsi <= 70:
            confidence += 0.1  # RSI in normal range
        else:
            confidence += 0.15  # RSI showing clear signal
            
        # Trend strength increases confidence
        if trend_strength > self.trend_strength_threshold:
            confidence += 0.15
        elif trend_strength > 1.0:
            confidence += 0.1
            
        # Clear sentiment increases confidence
        if abs(sentiment) > 0.3:
            confidence += 0.1
            
        # Very high volatility decreases confidence
        if volatility > 5.0:
            confidence -= 0.15
            
        return max(0.3, min(0.95, confidence))
        
    def _optimize_strategy_params(
        self,
        strategy: Strategy,
        volatility: float,
        sentiment: float
    ) -> Dict[str, Any]:
        """Optimize strategy parameters based on current conditions"""
        
        if strategy == Strategy.HOLD:
            return {'action': 'no_trade', 'reason': 'unfavorable conditions'}
            
        base_params = self.strategy_params.get(strategy, {}).copy()
        
        # Adjust for volatility
        if strategy == Strategy.MOMENTUM:
            if volatility > 2.0:
                base_params['position_size'] *= 0.7
                base_params['exit_threshold'] = -0.3
            elif volatility < 1.0:
                base_params['position_size'] *= 1.2
                
        elif strategy == Strategy.GRID:
            if volatility > 2.0:
                base_params['grid_spacing'] *= 1.5
            elif volatility < 1.0:
                base_params['grid_spacing'] *= 0.7
                
        elif strategy == Strategy.SCALPING:
            if volatility > 2.5:
                base_params['profit_target'] *= 1.5
                base_params['max_hold_seconds'] *= 0.5
                
        elif strategy == Strategy.MEAN_REVERSION:
            if volatility > 2.0:
                base_params['std_dev_entry'] = 2.5
            elif volatility < 1.0:
                base_params['std_dev_entry'] = 1.5
                
        elif strategy == Strategy.BREAKOUT:
            if sentiment > 0.3:
                base_params['position_size'] *= 1.2
            elif sentiment < -0.3:
                base_params['position_size'] *= 0.8
                
        return base_params
        
    async def get_active_count(self) -> int:
        """Get count of active strategies"""
        return len(self.active_strategies)
        
    async def get_recommended_strategy(
        self,
        user_id: str,
        symbol: str,
        risk_tolerance: str = 'medium'
    ) -> Dict[str, Any]:
        """Get recommended strategy for a user/symbol"""
        
        # Get cached regime
        regime_data = await self.redis_client.hgetall(f"regime:{symbol}")
        
        if not regime_data:
            return {'strategy': 'hold', 'reason': 'no_market_data'}
            
        regime_str = regime_data.get(b'regime', b'sideways').decode()
        confidence = float(regime_data.get(b'confidence', b'0.5').decode())
        
        # Get strategy params from cache
        strategy_params_raw = regime_data.get(b'strategy_params', b'{}').decode()
        try:
            strategy_params = json.loads(strategy_params_raw)
        except:
            strategy_params = {}
            
        # Adjust for risk tolerance
        if risk_tolerance == 'low':
            if strategy_params.get('position_size'):
                strategy_params['position_size'] *= 0.5
                
        elif risk_tolerance == 'high':
            if strategy_params.get('position_size'):
                strategy_params['position_size'] *= 1.5
                
        recommended_strategy = regime_data.get(b'recommended_strategy', b'hold').decode()
        
        return {
            'symbol': symbol,
            'strategy': recommended_strategy,
            'regime': regime_str,
            'confidence': confidence,
            'params': strategy_params,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
    def _default_regime_response(self) -> Dict[str, Any]:
        """Return default regime when data is unavailable"""
        return {
            'regime': 'sideways',
            'confidence': 0.5,
            'volatility': 1.5,
            'volatility_level': 'normal',
            'trend': 'sideways',
            'trend_strength': 0,
            'rsi': 50,
            'sentiment_score': 0,
            'sentiment_momentum': 0,
            'recommended_strategy': 'hold',
            'strategy_params': {'action': 'no_trade', 'reason': 'insufficient_data'},
            'alternative_strategies': [],
            'timestamp': datetime.utcnow().isoformat(),
        }

