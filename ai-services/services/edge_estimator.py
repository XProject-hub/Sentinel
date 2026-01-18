"""
SENTINEL AI - Edge Estimator
Calculates statistical edge for each trading opportunity

Edge Score: -1 to +1
- Negative = disadvantage (DON'T TRADE)
- 0 = no edge (SKIP)
- Positive = advantage (TRADE)
- >0.3 = good edge
- >0.5 = strong edge (increase size)

This is the difference between gambling and trading.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import redis.asyncio as redis
import json
import httpx

from config import settings


@dataclass
class EdgeScore:
    """Complete edge analysis for a trading opportunity"""
    symbol: str
    edge: float  # -1 to +1
    confidence: float  # 0-100, how confident in this edge
    
    # Components
    technical_edge: float
    regime_edge: float
    momentum_edge: float
    volume_edge: float
    correlation_edge: float
    sentiment_edge: float
    
    # Expected values
    expected_return: float  # Expected % return
    risk_reward_ratio: float
    win_probability: float  # 0-100
    
    # Position sizing recommendation
    kelly_fraction: float  # Optimal position size (0-1)
    recommended_size: str  # 'skip', 'probe', 'normal', 'aggressive'
    
    # Reasoning
    reasons: List[str]
    warnings: List[str]
    
    timestamp: str
    

class EdgeEstimator:
    """
    Advanced Edge Estimation System
    
    Combines multiple signals to calculate probability of profitable trade:
    - Technical setup quality
    - Regime alignment
    - Momentum quality
    - Volume confirmation
    - Cross-asset correlation
    - Sentiment alignment
    
    Uses Bayesian-inspired confidence calibration
    """
    
    def __init__(self):
        self.redis_client = None
        self.http_client = None
        
        # Historical accuracy tracking
        self.prediction_history: Dict[str, List[Dict]] = {}
        
        # Calibration factors
        self.calibration: Dict[str, float] = {
            'technical': 1.0,
            'regime': 1.0,
            'momentum': 1.0,
            'volume': 1.0,
            'correlation': 1.0,
            'sentiment': 1.0
        }
        
        # Component weights (updated through learning)
        self.weights = {
            'technical': 0.25,
            'regime': 0.20,
            'momentum': 0.20,
            'volume': 0.15,
            'correlation': 0.10,
            'sentiment': 0.10
        }
        
        # Win rate tracking per edge range
        self.edge_performance = {
            'negative': {'wins': 0, 'total': 0},
            'low': {'wins': 0, 'total': 0},      # 0-0.2
            'medium': {'wins': 0, 'total': 0},   # 0.2-0.4
            'high': {'wins': 0, 'total': 0},     # 0.4-0.6
            'very_high': {'wins': 0, 'total': 0} # 0.6+
        }
        
    async def initialize(self):
        """Initialize edge estimator"""
        logger.info("Initializing Edge Estimator (Bayesian-inspired)...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load calibration data
        await self._load_calibration()
        
        logger.info("Edge Estimator initialized - Statistical advantage detection active")
        
    async def shutdown(self):
        """Cleanup"""
        await self._save_calibration()
        if self.http_client:
            await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def calculate_edge(self, symbol: str, direction: str = 'long') -> EdgeScore:
        """
        Calculate statistical edge for a potential trade
        
        Args:
            symbol: Trading pair
            direction: 'long' or 'short'
            
        Returns:
            EdgeScore with complete analysis
        """
        try:
            reasons = []
            warnings = []
            
            # Fetch all required data
            market_data = await self._fetch_market_data(symbol)
            regime_data = await self._get_regime_data(symbol)
            
            if not market_data:
                return self._no_edge(symbol, "No market data")
                
            # === Calculate Individual Edge Components ===
            
            # 1. Technical Edge (chart setup quality)
            technical_edge, tech_reasons = self._calculate_technical_edge(
                market_data, direction
            )
            reasons.extend(tech_reasons)
            
            # 2. Regime Edge (is market regime favorable?)
            regime_edge, regime_reasons = self._calculate_regime_edge(
                regime_data, direction
            )
            reasons.extend(regime_reasons)
            
            # 3. Momentum Edge (is momentum confirming?)
            momentum_edge, mom_reasons = self._calculate_momentum_edge(
                market_data, direction
            )
            reasons.extend(mom_reasons)
            
            # 4. Volume Edge (is volume supporting?)
            volume_edge, vol_reasons = self._calculate_volume_edge(market_data)
            reasons.extend(vol_reasons)
            
            # 5. Correlation Edge (BTC/market alignment)
            correlation_edge, corr_reasons = await self._calculate_correlation_edge(
                symbol, direction
            )
            reasons.extend(corr_reasons)
            
            # 6. Sentiment Edge (news/social sentiment)
            sentiment_edge, sent_reasons = await self._calculate_sentiment_edge(symbol)
            reasons.extend(sent_reasons)
            
            # === Combine Edges with Weights ===
            
            weighted_edge = (
                technical_edge * self.weights['technical'] * self.calibration['technical'] +
                regime_edge * self.weights['regime'] * self.calibration['regime'] +
                momentum_edge * self.weights['momentum'] * self.calibration['momentum'] +
                volume_edge * self.weights['volume'] * self.calibration['volume'] +
                correlation_edge * self.weights['correlation'] * self.calibration['correlation'] +
                sentiment_edge * self.weights['sentiment'] * self.calibration['sentiment']
            )
            
            # Normalize to -1 to +1
            total_edge = max(-1.0, min(1.0, weighted_edge))
            
            # === Calculate Confidence ===
            # Confidence = how consistent are the signals?
            edges = [technical_edge, regime_edge, momentum_edge, 
                     volume_edge, correlation_edge, sentiment_edge]
            
            # If all edges agree, high confidence
            agreement = 1 - (np.std(edges) / 2) if len(edges) > 0 else 0
            base_confidence = agreement * 100
            
            # Adjust confidence based on historical accuracy
            calibrated_confidence = self._calibrate_confidence(base_confidence, total_edge)
            
            # === Calculate Win Probability ===
            # Based on edge + historical performance
            win_prob = self._estimate_win_probability(total_edge)
            
            # === Calculate Expected Return ===
            # Based on typical moves in this regime
            expected_return = self._estimate_expected_return(
                market_data, regime_data, total_edge
            )
            
            # === Risk/Reward Ratio ===
            risk_reward = self._calculate_risk_reward(market_data, direction)
            
            # === Kelly Fraction (Optimal Position Size) ===
            kelly = self._calculate_kelly_fraction(win_prob, risk_reward)
            
            # === Recommended Size ===
            recommended_size = self._get_size_recommendation(total_edge, calibrated_confidence, kelly)
            
            # === Generate Warnings ===
            if total_edge < 0:
                warnings.append("Negative edge - trade not recommended")
            if calibrated_confidence < 50:
                warnings.append("Low confidence in edge estimate")
            if regime_data.get('action') == 'avoid':
                warnings.append("Regime recommends avoiding trades")
            if volume_edge < -0.3:
                warnings.append("Weak volume - potential slippage")
            if kelly < 0.01:
                warnings.append("Kelly suggests very small or no position")
                
            edge_score = EdgeScore(
                symbol=symbol,
                edge=round(total_edge, 4),
                confidence=round(calibrated_confidence, 1),
                technical_edge=round(technical_edge, 3),
                regime_edge=round(regime_edge, 3),
                momentum_edge=round(momentum_edge, 3),
                volume_edge=round(volume_edge, 3),
                correlation_edge=round(correlation_edge, 3),
                sentiment_edge=round(sentiment_edge, 3),
                expected_return=round(expected_return, 2),
                risk_reward_ratio=round(risk_reward, 2),
                win_probability=round(win_prob, 1),
                kelly_fraction=round(kelly, 4),
                recommended_size=recommended_size,
                reasons=reasons[:10],  # Top 10 reasons
                warnings=warnings,
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Store for tracking
            await self._store_edge(edge_score)
            
            return edge_score
            
        except Exception as e:
            logger.error(f"Edge calculation error for {symbol}: {e}")
            return self._no_edge(symbol, str(e))
            
    def _calculate_technical_edge(self, data: Dict, direction: str) -> Tuple[float, List[str]]:
        """
        Calculate technical setup quality
        
        Looks for:
        - Support/resistance alignment
        - Moving average structure
        - RSI conditions
        - MACD alignment
        """
        closes = data.get('closes', [])
        highs = data.get('highs', [])
        lows = data.get('lows', [])
        
        if len(closes) < 50:
            return 0.0, []
            
        edge = 0.0
        reasons = []
        current = closes[-1]
        
        # === Moving Average Structure ===
        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        
        if direction == 'long':
            if current > sma_10 > sma_20:
                edge += 0.3
                reasons.append("Price above SMAs (bullish structure)")
            if sma_10 > sma_50:
                edge += 0.2
                reasons.append("SMA10 > SMA50 (uptrend)")
        else:
            if current < sma_10 < sma_20:
                edge += 0.3
                reasons.append("Price below SMAs (bearish structure)")
            if sma_10 < sma_50:
                edge += 0.2
                reasons.append("SMA10 < SMA50 (downtrend)")
                
        # === RSI Conditions ===
        rsi = self._calculate_rsi(closes, 14)
        
        if direction == 'long':
            if 30 <= rsi <= 50:
                edge += 0.25
                reasons.append(f"RSI {rsi:.0f} - oversold bounce zone")
            elif 50 <= rsi <= 65:
                edge += 0.1
                reasons.append(f"RSI {rsi:.0f} - momentum zone")
            elif rsi > 80:
                edge -= 0.3
                reasons.append(f"RSI {rsi:.0f} - overbought (warning)")
        else:
            if 50 <= rsi <= 70:
                edge += 0.25
                reasons.append(f"RSI {rsi:.0f} - overbought reversal zone")
            elif 35 <= rsi <= 50:
                edge += 0.1
                reasons.append(f"RSI {rsi:.0f} - weak momentum")
            elif rsi < 20:
                edge -= 0.3
                reasons.append(f"RSI {rsi:.0f} - oversold (warning)")
                
        # === Support/Resistance ===
        recent_low = min(lows[-20:])
        recent_high = max(highs[-20:])
        range_size = recent_high - recent_low
        
        if direction == 'long':
            # Near support is good for longs
            distance_from_support = (current - recent_low) / range_size if range_size > 0 else 0.5
            if distance_from_support < 0.3:
                edge += 0.3
                reasons.append("Near support level")
            elif distance_from_support > 0.8:
                edge -= 0.2
                reasons.append("Near resistance (caution)")
        else:
            # Near resistance is good for shorts
            distance_from_resistance = (recent_high - current) / range_size if range_size > 0 else 0.5
            if distance_from_resistance < 0.3:
                edge += 0.3
                reasons.append("Near resistance level")
            elif distance_from_resistance > 0.8:
                edge -= 0.2
                reasons.append("Near support (caution)")
                
        return max(-1, min(1, edge)), reasons
        
    def _calculate_regime_edge(self, regime_data: Dict, direction: str) -> Tuple[float, List[str]]:
        """Calculate edge from market regime alignment"""
        regime = regime_data.get('regime', 'unknown')
        action = regime_data.get('action', 'avoid')
        trend_dir = regime_data.get('trend_direction', 'neutral')
        
        edge = 0.0
        reasons = []
        
        # Regime-based edge
        regime_edges = {
            'high_liquidity_trend': 0.4,
            'accumulation': 0.2,
            'range_bound': 0.1,
            'distribution': -0.1,
            'high_volatility': -0.2,
            'low_liquidity': -0.5,
            'news_event': -0.6,
            'unknown': -0.3
        }
        
        edge = regime_edges.get(regime, 0)
        reasons.append(f"Regime: {regime} (edge: {edge:+.1f})")
        
        # Direction alignment with trend
        if direction == 'long' and trend_dir == 'up':
            edge += 0.2
            reasons.append("Long aligns with uptrend")
        elif direction == 'long' and trend_dir == 'down':
            edge -= 0.3
            reasons.append("Long against downtrend (warning)")
        elif direction == 'short' and trend_dir == 'down':
            edge += 0.2
            reasons.append("Short aligns with downtrend")
        elif direction == 'short' and trend_dir == 'up':
            edge -= 0.3
            reasons.append("Short against uptrend (warning)")
            
        # Action recommendation
        if action == 'avoid':
            edge -= 0.3
            reasons.append("Regime action: AVOID")
        elif action == 'reduced':
            edge -= 0.1
            reasons.append("Regime action: reduced size")
        elif action == 'aggressive':
            edge += 0.2
            reasons.append("Regime action: aggressive OK")
            
        return max(-1, min(1, edge)), reasons
        
    def _calculate_momentum_edge(self, data: Dict, direction: str) -> Tuple[float, List[str]]:
        """Calculate momentum quality edge"""
        closes = data.get('closes', [])
        
        if len(closes) < 20:
            return 0.0, []
            
        edge = 0.0
        reasons = []
        
        # Short-term momentum (5 candles)
        short_momentum = (closes[-1] / closes[-5] - 1) * 100
        
        # Medium-term momentum (20 candles)
        medium_momentum = (closes[-1] / closes[-20] - 1) * 100
        
        if direction == 'long':
            if short_momentum > 0 and medium_momentum > 0:
                edge += 0.4
                reasons.append(f"Momentum aligned: +{short_momentum:.1f}% / +{medium_momentum:.1f}%")
            elif short_momentum > 0:
                edge += 0.2
                reasons.append(f"Short-term momentum positive: +{short_momentum:.1f}%")
            elif short_momentum < -2:
                edge -= 0.3
                reasons.append(f"Momentum against long: {short_momentum:.1f}%")
        else:
            if short_momentum < 0 and medium_momentum < 0:
                edge += 0.4
                reasons.append(f"Momentum aligned: {short_momentum:.1f}% / {medium_momentum:.1f}%")
            elif short_momentum < 0:
                edge += 0.2
                reasons.append(f"Short-term momentum negative: {short_momentum:.1f}%")
            elif short_momentum > 2:
                edge -= 0.3
                reasons.append(f"Momentum against short: +{short_momentum:.1f}%")
                
        # Momentum acceleration (is momentum increasing?)
        if len(closes) >= 10:
            accel = (closes[-1] - closes[-5]) - (closes[-5] - closes[-10])
            if direction == 'long' and accel > 0:
                edge += 0.15
                reasons.append("Momentum accelerating")
            elif direction == 'short' and accel < 0:
                edge += 0.15
                reasons.append("Momentum accelerating down")
                
        return max(-1, min(1, edge)), reasons
        
    def _calculate_volume_edge(self, data: Dict) -> Tuple[float, List[str]]:
        """Calculate volume confirmation edge"""
        volumes = data.get('volumes', [])
        
        if len(volumes) < 20:
            return 0.0, []
            
        edge = 0.0
        reasons = []
        
        avg_volume = np.mean(volumes[-20:])
        recent_volume = np.mean(volumes[-5:])
        current_volume = volumes[-1]
        
        if avg_volume <= 0:
            return 0.0, []
            
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 1.5:
            edge += 0.4
            reasons.append(f"Strong volume: {volume_ratio:.1f}x average")
        elif volume_ratio > 1.2:
            edge += 0.2
            reasons.append(f"Above average volume: {volume_ratio:.1f}x")
        elif volume_ratio < 0.5:
            edge -= 0.4
            reasons.append(f"Low volume: {volume_ratio:.1f}x average")
        elif volume_ratio < 0.7:
            edge -= 0.2
            reasons.append(f"Below average volume: {volume_ratio:.1f}x")
            
        # Volume trend
        if len(volumes) >= 10:
            vol_first_half = np.mean(volumes[-10:-5])
            vol_second_half = np.mean(volumes[-5:])
            
            if vol_first_half > 0:
                vol_trend = (vol_second_half / vol_first_half) - 1
                if vol_trend > 0.2:
                    edge += 0.15
                    reasons.append("Volume increasing")
                elif vol_trend < -0.2:
                    edge -= 0.15
                    reasons.append("Volume decreasing")
                    
        return max(-1, min(1, edge)), reasons
        
    async def _calculate_correlation_edge(self, symbol: str, direction: str) -> Tuple[float, List[str]]:
        """Calculate edge from BTC/market correlation"""
        edge = 0.0
        reasons = []
        
        try:
            # Get BTC trend direction
            btc_regime = await self.redis_client.hgetall('regime:BTCUSDT')
            
            if btc_regime:
                btc_trend = btc_regime.get(b'trend_direction', b'neutral').decode()
                btc_regime_name = btc_regime.get(b'regime', b'unknown').decode()
                
                # Most alts follow BTC
                if direction == 'long' and btc_trend == 'up':
                    edge += 0.25
                    reasons.append("BTC uptrend supports longs")
                elif direction == 'long' and btc_trend == 'down':
                    edge -= 0.25
                    reasons.append("BTC downtrend - risk for longs")
                elif direction == 'short' and btc_trend == 'down':
                    edge += 0.25
                    reasons.append("BTC downtrend supports shorts")
                elif direction == 'short' and btc_trend == 'up':
                    edge -= 0.25
                    reasons.append("BTC uptrend - risk for shorts")
                    
                # BTC regime affects all
                if btc_regime_name == 'high_volatility':
                    edge -= 0.15
                    reasons.append("BTC high volatility - market risk")
                elif btc_regime_name == 'high_liquidity_trend':
                    edge += 0.1
                    reasons.append("BTC stable trend - good conditions")
                    
        except Exception as e:
            logger.debug(f"Correlation edge error: {e}")
            
        return max(-1, min(1, edge)), reasons
        
    async def _calculate_sentiment_edge(self, symbol: str) -> Tuple[float, List[str]]:
        """Calculate edge from sentiment data"""
        edge = 0.0
        reasons = []
        
        try:
            # Fear & Greed Index - handle different Redis key types
            try:
                key_type = await self.redis_client.type('data:fear_greed')
                if key_type == 'string':
                    fg_data = await self.redis_client.get('data:fear_greed')
                    if fg_data:
                        fg = json.loads(fg_data)
                        fg_value = int(fg.get('value', 50))
                elif key_type == 'hash':
                    fg = await self.redis_client.hgetall('data:fear_greed')
                    fg_value = int(fg.get('value', 50)) if fg else 50
                else:
                    fg_value = 50
                    
                # Contrarian indicator
                if fg_value < 25:
                    edge += 0.2
                    reasons.append(f"Extreme Fear ({fg_value}) - contrarian bullish")
                elif fg_value < 40:
                    edge += 0.1
                    reasons.append(f"Fear ({fg_value}) - potentially bullish")
                elif fg_value > 80:
                    edge -= 0.2
                    reasons.append(f"Extreme Greed ({fg_value}) - contrarian bearish")
                elif fg_value > 65:
                    edge -= 0.1
                    reasons.append(f"Greed ({fg_value}) - potentially bearish")
            except Exception:
                pass  # Skip fear/greed if not available
                    
            # Symbol-specific sentiment - handle different key types
            asset = symbol.replace('USDT', '').replace('PERP', '')
            try:
                key_type = await self.redis_client.type(f'sentiment:{asset}')
                if key_type == 'string':
                    sentiment_data = await self.redis_client.get(f'sentiment:{asset}')
                elif key_type == 'hash':
                    sentiment_data = json.dumps(await self.redis_client.hgetall(f'sentiment:{asset}'))
                else:
                    sentiment_data = None
            except Exception:
                sentiment_data = None
            
            if sentiment_data:
                sentiment = json.loads(sentiment_data)
                score = sentiment.get('score', 0)
                
                if score > 0.3:
                    edge += 0.15
                    reasons.append(f"Positive sentiment for {asset}")
                elif score < -0.3:
                    edge -= 0.15
                    reasons.append(f"Negative sentiment for {asset}")
                    
        except Exception as e:
            logger.debug(f"Sentiment edge error: {e}")
            
        return max(-1, min(1, edge)), reasons
        
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
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _estimate_win_probability(self, edge: float) -> float:
        """
        Estimate win probability based on edge
        
        Uses historical performance data if available
        """
        # Base probability from edge
        # Edge of 0.3 roughly corresponds to 55% win rate
        # Edge of 0.6 roughly corresponds to 65% win rate
        base_prob = 50 + (edge * 25)
        
        # Adjust based on historical performance
        if edge < 0:
            perf = self.edge_performance['negative']
        elif edge < 0.2:
            perf = self.edge_performance['low']
        elif edge < 0.4:
            perf = self.edge_performance['medium']
        elif edge < 0.6:
            perf = self.edge_performance['high']
        else:
            perf = self.edge_performance['very_high']
            
        if perf['total'] > 10:
            historical_wr = (perf['wins'] / perf['total']) * 100
            # Blend base with historical
            adjusted_prob = base_prob * 0.6 + historical_wr * 0.4
        else:
            adjusted_prob = base_prob
            
        return max(30, min(75, adjusted_prob))
        
    def _estimate_expected_return(self, data: Dict, regime_data: Dict, edge: float) -> float:
        """Estimate expected return based on volatility and edge"""
        closes = data.get('closes', [])
        
        if len(closes) < 21:
            return edge * 1.0  # Default 1% at full edge
            
        try:
            # Recent volatility - FIX: ensure arrays have matching shapes
            close_slice = np.array(closes[-21:])  # 21 elements
            returns = np.diff(close_slice) / close_slice[:-1]  # 20 / 20 = matching shapes
            volatility = np.std(returns) * 100
            
            # Expected return = edge * volatility * factor
            expected = edge * volatility * 0.5
            
            # Cap at reasonable values
            return max(-3, min(5, expected))
        except Exception as e:
            return edge * 1.0  # Fallback
        
    def _calculate_risk_reward(self, data: Dict, direction: str) -> float:
        """Calculate risk/reward ratio based on chart structure"""
        closes = data.get('closes', [])
        highs = data.get('highs', [])
        lows = data.get('lows', [])
        
        if len(closes) < 20:
            return 1.0  # Default 1:1
            
        current = closes[-1]
        recent_low = min(lows[-20:])
        recent_high = max(highs[-20:])
        
        if direction == 'long':
            risk = (current - recent_low) / current if current > 0 else 0.02
            reward = (recent_high - current) / current if current > 0 else 0.02
        else:
            risk = (recent_high - current) / current if current > 0 else 0.02
            reward = (current - recent_low) / current if current > 0 else 0.02
            
        # Minimum values
        risk = max(0.005, risk)
        reward = max(0.005, reward)
        
        return reward / risk
        
    def _calculate_kelly_fraction(self, win_prob: float, risk_reward: float) -> float:
        """
        Calculate Kelly Criterion fraction
        
        Kelly = (bp - q) / b
        where:
        b = odds (risk/reward ratio)
        p = probability of winning
        q = probability of losing (1-p)
        """
        p = win_prob / 100
        q = 1 - p
        b = risk_reward
        
        if b <= 0:
            return 0.0
            
        kelly = (b * p - q) / b
        
        # Use fractional Kelly (25-50% of full Kelly is safer)
        fractional_kelly = kelly * 0.25
        
        # Cap at reasonable values
        return max(0, min(0.15, fractional_kelly))
        
    def _get_size_recommendation(self, edge: float, confidence: float, kelly: float) -> str:
        """Get position size recommendation"""
        
        if edge < 0:
            return 'skip'
        elif edge < 0.15 or confidence < 40:
            return 'skip'
        elif edge < 0.25 or confidence < 50:
            return 'probe'
        elif edge < 0.4:
            return 'normal'
        else:
            return 'aggressive'
            
    def _calibrate_confidence(self, base_confidence: float, edge: float) -> float:
        """Calibrate confidence based on historical accuracy"""
        # Start with base
        calibrated = base_confidence
        
        # If we have history, adjust
        total_predictions = sum(p['total'] for p in self.edge_performance.values())
        
        if total_predictions > 50:
            # Calculate overall accuracy
            total_wins = sum(p['wins'] for p in self.edge_performance.values())
            overall_accuracy = total_wins / total_predictions
            
            # If our predictions are less accurate than we think, reduce confidence
            if overall_accuracy < 0.5:
                calibrated *= 0.8
            elif overall_accuracy > 0.6:
                calibrated *= 1.1
                
        return min(95, calibrated)
        
    async def record_outcome(self, symbol: str, edge: float, won: bool):
        """
        Record trade outcome for calibration
        
        This is how the system LEARNS
        """
        # Determine edge bucket
        if edge < 0:
            bucket = 'negative'
        elif edge < 0.2:
            bucket = 'low'
        elif edge < 0.4:
            bucket = 'medium'
        elif edge < 0.6:
            bucket = 'high'
        else:
            bucket = 'very_high'
            
        self.edge_performance[bucket]['total'] += 1
        if won:
            self.edge_performance[bucket]['wins'] += 1
            
        # Save to Redis
        await self.redis_client.set('edge:performance', json.dumps(self.edge_performance))
        
        logger.info(f"Edge outcome recorded: {symbol} edge={edge:.2f} won={won}")
        
    async def _fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch OHLCV data"""
        try:
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': '15',
                'limit': 100
            }
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            klines = data.get('result', {}).get('list', [])
            
            if len(klines) < 50:
                return None
                
            opens, highs, lows, closes, volumes = [], [], [], [], []
            
            for k in reversed(klines):
                opens.append(float(k[1]))
                highs.append(float(k[2]))
                lows.append(float(k[3]))
                closes.append(float(k[4]))
                volumes.append(float(k[5]))
                
            return {
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'volumes': volumes
            }
            
        except Exception as e:
            logger.debug(f"Market data fetch error: {e}")
            return None
            
    async def _get_regime_data(self, symbol: str) -> Dict:
        """Get regime data from Redis"""
        try:
            data = await self.redis_client.hgetall(f'regime:{symbol}')
            if data:
                return {
                    'regime': data.get(b'regime', b'unknown').decode(),
                    'action': data.get(b'action', b'avoid').decode(),
                    'trend_direction': data.get(b'trend_direction', b'neutral').decode(),
                    'confidence': float(data.get(b'confidence', 0))
                }
        except:
            pass
        return {'regime': 'unknown', 'action': 'avoid', 'trend_direction': 'neutral'}
        
    def _no_edge(self, symbol: str, reason: str = "") -> EdgeScore:
        """Return zero edge score"""
        return EdgeScore(
            symbol=symbol,
            edge=0.0,
            confidence=0.0,
            technical_edge=0.0,
            regime_edge=0.0,
            momentum_edge=0.0,
            volume_edge=0.0,
            correlation_edge=0.0,
            sentiment_edge=0.0,
            expected_return=0.0,
            risk_reward_ratio=1.0,
            win_probability=50.0,
            kelly_fraction=0.0,
            recommended_size='skip',
            reasons=[],
            warnings=[reason] if reason else [],
            timestamp=datetime.utcnow().isoformat()
        )
        
    async def _store_edge(self, score: EdgeScore):
        """Store edge score in Redis"""
        try:
            await self.redis_client.setex(
                f"edge:{score.symbol}",
                300,  # 5 minute TTL
                json.dumps({
                    'edge': score.edge,
                    'confidence': score.confidence,
                    'win_prob': score.win_probability,
                    'kelly': score.kelly_fraction,
                    'size': score.recommended_size,
                    'reasons': score.reasons,
                    'timestamp': score.timestamp
                })
            )
        except:
            pass
            
    async def _load_calibration(self):
        """Load calibration data from Redis"""
        try:
            perf = await self.redis_client.get('edge:performance')
            if perf:
                self.edge_performance = json.loads(perf)
                
            weights = await self.redis_client.get('edge:weights')
            if weights:
                self.weights = json.loads(weights)
                
            calib = await self.redis_client.get('edge:calibration')
            if calib:
                self.calibration = json.loads(calib)
                
            logger.info("Loaded edge estimation calibration data")
        except:
            pass
            
    async def _save_calibration(self):
        """Save calibration data to Redis"""
        try:
            await self.redis_client.set('edge:performance', json.dumps(self.edge_performance))
            await self.redis_client.set('edge:weights', json.dumps(self.weights))
            await self.redis_client.set('edge:calibration', json.dumps(self.calibration))
        except:
            pass


# Global instance
edge_estimator = EdgeEstimator()

