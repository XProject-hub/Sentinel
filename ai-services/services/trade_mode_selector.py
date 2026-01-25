"""
SENTINEL AI - Trade Mode Selector
Intelligently selects the best trading mode for each opportunity

Trading Modes:
1. SPOT - Buy and hold, no leverage, safest
2. FUTURES_LONG - Leveraged long position, profit when price goes up
3. FUTURES_SHORT - Leveraged short position, profit when price goes down

AI considers:
- Market regime (bull/bear/range)
- Volatility level
- Trend strength
- Funding rate (for futures)
- Risk/reward ratio
- User's risk settings
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from loguru import logger


class TradeMode(Enum):
    """Available trading modes"""
    SPOT = "spot"                    # Buy on spot market, no leverage
    FUTURES_LONG = "futures_long"    # Leveraged long position
    FUTURES_SHORT = "futures_short"  # Leveraged short position
    

@dataclass
class TradeModeDecision:
    """Result of trade mode selection"""
    mode: TradeMode
    confidence: float  # 0-100
    leverage: int  # 1 for spot, 2-10 for futures
    reasons: List[str]
    warnings: List[str]
    expected_hold_time: str  # "minutes", "hours", "days"
    risk_level: str  # "low", "medium", "high"
    spot_score: int = 0  # Score for spot trading
    futures_score: int = 0  # Score for futures trading
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.value,
            "mode_display": self.get_display_name(),
            "confidence": self.confidence,
            "leverage": self.leverage,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "expected_hold_time": self.expected_hold_time,
            "risk_level": self.risk_level
        }
    
    def get_display_name(self) -> str:
        """Human readable mode name for dashboard"""
        if self.mode == TradeMode.SPOT:
            return "Spot Buy"
        elif self.mode == TradeMode.FUTURES_LONG:
            return f"Futures Long {self.leverage}x"
        elif self.mode == TradeMode.FUTURES_SHORT:
            return f"Futures Short {self.leverage}x"
        return self.mode.value


class TradeModeSelector:
    """
    AI-powered trade mode selector
    
    Analyzes market conditions and selects the optimal trading mode:
    - SPOT for stable uptrends, lower risk
    - FUTURES_LONG for strong bullish signals with leverage
    - FUTURES_SHORT for bearish markets or reversals
    """
    
    def __init__(self):
        self.default_leverage = 3
        self.max_leverage = 10
        self.min_leverage = 2
        
    async def select_mode(
        self,
        symbol: str,
        direction: str,  # 'long' or 'short'
        regime: str,  # 'BULL', 'BEAR', 'RANGE', 'VOLATILE'
        volatility: float,  # 0-10 scale
        trend_strength: float,  # 0-1 scale
        confidence: float,  # 0-100
        funding_rate: float = 0.0,  # Current funding rate
        user_risk_mode: str = 'normal',  # 'safe', 'normal', 'aggressive'
        price_change_24h: float = 0.0,  # 24h price change %
        volume_ratio: float = 1.0,  # Current volume vs average
    ) -> TradeModeDecision:
        """
        Select optimal trading mode based on market conditions
        
        Returns:
            TradeModeDecision with mode, leverage, and reasoning
        """
        logger.info(f"🎯 TRADE MODE SELECTOR: Analyzing {symbol}")
        logger.info(f"   ├─ Direction: {direction.upper()}")
        logger.info(f"   ├─ Regime: {regime}, Volatility: {volatility:.2f}")
        logger.info(f"   ├─ Trend: {trend_strength:.1%}, Confidence: {confidence:.0f}%")
        logger.info(f"   ├─ Funding: {funding_rate:.4%}, 24h Change: {price_change_24h:+.2f}%")
        logger.info(f"   └─ Risk Mode: {user_risk_mode}, Volume Ratio: {volume_ratio:.1f}x")
        
        reasons = []
        warnings = []
        
        # === RULE 1: Short positions MUST use futures ===
        if direction == 'short':
            leverage = self._calculate_leverage(
                volatility, confidence, trend_strength, user_risk_mode
            )
            reasons.append(f"Short position requires futures")
            reasons.append(f"Bearish signal detected")
            
            if abs(funding_rate) > 0.01:
                if funding_rate > 0:
                    reasons.append(f"Positive funding ({funding_rate:.3%}) - shorts get paid")
                else:
                    warnings.append(f"Negative funding ({funding_rate:.3%}) - shorts pay longs")
            
            decision = TradeModeDecision(
                mode=TradeMode.FUTURES_SHORT,
                confidence=confidence,
                leverage=leverage,
                reasons=reasons,
                warnings=warnings,
                expected_hold_time=self._estimate_hold_time(volatility, regime),
                risk_level=self._assess_risk(leverage, volatility),
                spot_score=0,  # Can't short on spot
                futures_score=100  # Short requires futures
            )
            logger.info(f"🎯 TRADE MODE DECISION: {symbol} → FUTURES SHORT {leverage}x | Risk: {decision.risk_level}")
            logger.info(f"   Reasons: {', '.join(reasons)}")
            return decision
        
        # === RULE 2: For LONG positions, choose SPOT vs FUTURES ===
        
        # Base scores - SPOT has safety advantage
        spot_score = 20  # Base safety score for SPOT
        futures_score = 0
        reasons.append("Base safety score for spot")
        
        # Strong uptrend with low volatility = SPOT
        if regime == 'BULL' and volatility < 2.0:
            spot_score += 25
            reasons.append("Strong bull trend with low volatility")
        elif regime == 'BULL':
            spot_score += 15  # Bull market favors spot even with volatility
            reasons.append("Bull market favors spot")
        
        # Moderate volatility (1-3%) = SPOT is safer
        if 1.0 <= volatility <= 3.0:
            spot_score += 10
            reasons.append(f"Moderate volatility ({volatility:.1f}%) - spot is safer")
        
        # Very high volatility = FUTURES (can exit quickly with leverage)
        if volatility > 4.0:
            futures_score += 20
            reasons.append(f"High volatility ({volatility:.1f}) - futures for quick exits")
        
        # Strong trend = can use leverage, but not always
        if trend_strength > 0.8:
            futures_score += 15
            reasons.append(f"Very strong trend ({trend_strength:.0%}) - leverage opportunity")
        elif trend_strength > 0.6:
            spot_score += 5  # Moderate trend = spot is fine
        
        # Ranging market = SPOT is much safer
        if regime == 'RANGE':
            spot_score += 25
            reasons.append("Ranging market - spot is much safer")
        
        # BEAR market = be careful, prefer SPOT for longs
        if regime == 'BEAR':
            spot_score += 20
            reasons.append("Bear market - spot reduces risk")
        
        # Confidence scoring - more balanced
        if confidence > 90:
            futures_score += 10
            reasons.append(f"Very high confidence ({confidence:.0f}%) supports leverage")
        elif confidence > 75:
            # Neutral - both are fine
            pass
        else:
            spot_score += 15
            reasons.append(f"Moderate confidence ({confidence:.0f}%) - spot is safer")
        
        # Large price move already happened = SPOT is safer
        if abs(price_change_24h) > 10:
            spot_score += 15
            warnings.append(f"Large 24h move ({price_change_24h:+.1f}%) - spot reduces reversal risk")
        elif abs(price_change_24h) > 5:
            spot_score += 5
        
        # Funding rate consideration
        if funding_rate > 0.03:  # High funding
            spot_score += 20
            warnings.append(f"High funding rate ({funding_rate:.3%}) - spot avoids funding fees")
        elif funding_rate > 0.01:
            spot_score += 10
        elif funding_rate < -0.02:  # Negative funding
            futures_score += 10
            reasons.append(f"Negative funding ({funding_rate:.3%}) - longs get paid")
        
        # User risk mode adjustment - REBALANCED
        if user_risk_mode == 'safe':
            spot_score += 30  # Strong preference for spot
            reasons.append("Safe mode strongly prefers spot")
        elif user_risk_mode == 'normal':
            spot_score += 10  # Slight preference for spot in normal mode
            reasons.append("Normal mode slightly prefers spot for safety")
        elif user_risk_mode == 'aggressive':
            futures_score += 25
            reasons.append("Aggressive mode prefers leverage")
        
        # Volume confirmation - high volume can support both
        if volume_ratio > 2.0:
            futures_score += 10
            reasons.append(f"High volume ({volume_ratio:.1f}x avg) confirms move")
        
        # === DECISION ===
        logger.info(f"🎯 TRADE MODE DECISION: {symbol} | SPOT={spot_score} vs FUTURES={futures_score}")
        
        if spot_score > futures_score:
            decision = TradeModeDecision(
                mode=TradeMode.SPOT,
                confidence=confidence,
                leverage=1,
                reasons=reasons,
                warnings=warnings,
                expected_hold_time=self._estimate_hold_time(volatility, regime, is_spot=True),
                risk_level="low",
                spot_score=spot_score,
                futures_score=futures_score
            )
            logger.info(f"✅ DECISION: {symbol} → SPOT (no leverage) | Scores: {spot_score}>{futures_score} | Risk: low")
            logger.info(f"   Reasons: {', '.join(reasons)}")
            return decision
        else:
            leverage = self._calculate_leverage(
                volatility, confidence, trend_strength, user_risk_mode
            )
            decision = TradeModeDecision(
                mode=TradeMode.FUTURES_LONG,
                confidence=confidence,
                leverage=leverage,
                reasons=reasons,
                warnings=warnings,
                expected_hold_time=self._estimate_hold_time(volatility, regime),
                risk_level=self._assess_risk(leverage, volatility),
                spot_score=spot_score,
                futures_score=futures_score
            )
            logger.info(f"✅ DECISION: {symbol} → FUTURES LONG {leverage}x | Scores: {futures_score}>{spot_score} | Risk: {decision.risk_level}")
            logger.info(f"   Reasons: {', '.join(reasons)}")
            return decision
    
    def _calculate_leverage(
        self,
        volatility: float,
        confidence: float,
        trend_strength: float,
        risk_mode: str
    ) -> int:
        """Calculate optimal leverage based on conditions"""
        
        # Base leverage from risk mode
        base_leverage = {
            'safe': 2,
            'normal': 3,
            'aggressive': 5,
            'micro': 2
        }.get(risk_mode.lower(), 3)
        
        # Adjust based on volatility (higher vol = lower leverage)
        if volatility > 5:
            vol_adjustment = -2
        elif volatility > 3:
            vol_adjustment = -1
        elif volatility < 1.5:
            vol_adjustment = 1
        else:
            vol_adjustment = 0
        
        # Adjust based on confidence
        if confidence > 85:
            conf_adjustment = 1
        elif confidence < 50:
            conf_adjustment = -1
        else:
            conf_adjustment = 0
        
        # Adjust based on trend strength
        if trend_strength > 0.8:
            trend_adjustment = 1
        else:
            trend_adjustment = 0
        
        # Calculate final leverage
        leverage = base_leverage + vol_adjustment + conf_adjustment + trend_adjustment
        
        # Clamp to valid range
        leverage = max(self.min_leverage, min(self.max_leverage, leverage))
        
        return leverage
    
    def _estimate_hold_time(
        self,
        volatility: float,
        regime: str,
        is_spot: bool = False
    ) -> str:
        """Estimate expected hold time"""
        if is_spot:
            if regime == 'BULL':
                return "days"
            return "hours"
        
        # Futures positions
        if volatility > 4:
            return "minutes"
        elif volatility > 2:
            return "hours"
        else:
            return "hours"
    
    def _assess_risk(self, leverage: int, volatility: float) -> str:
        """Assess overall risk level"""
        risk_score = leverage * volatility
        
        if risk_score > 20:
            return "high"
        elif risk_score > 10:
            return "medium"
        else:
            return "low"
    
    def get_mode_info(self, mode: TradeMode) -> Dict:
        """Get information about a trading mode"""
        info = {
            TradeMode.SPOT: {
                "name": "Spot Trading",
                "description": "Buy and hold the actual asset. No leverage, lowest risk.",
                "pros": ["No liquidation risk", "Can hold indefinitely", "No funding fees"],
                "cons": ["Cannot profit from price drops", "Lower potential returns"],
                "best_for": "Stable uptrends, longer holds, lower risk tolerance"
            },
            TradeMode.FUTURES_LONG: {
                "name": "Futures Long",
                "description": "Leveraged long position. Profit when price goes up.",
                "pros": ["Amplified gains with leverage", "Capital efficient"],
                "cons": ["Liquidation risk", "Funding fees", "Higher risk"],
                "best_for": "Strong bullish signals, high confidence trades"
            },
            TradeMode.FUTURES_SHORT: {
                "name": "Futures Short",
                "description": "Leveraged short position. Profit when price goes down.",
                "pros": ["Profit from falling prices", "Hedge existing holdings"],
                "cons": ["Liquidation risk", "Unlimited loss potential", "Funding fees"],
                "best_for": "Bearish markets, reversals, hedging"
            }
        }
        return info.get(mode, {})


# Singleton instance
trade_mode_selector = TradeModeSelector()
