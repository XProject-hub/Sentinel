"""
SENTINEL AI - AI Coordinator (Brain)
Coordinates all AI models for trading decisions

This is the "brain" that combines:
- LSTM Price Predictions
- Pattern Recognition
- Sentiment Analysis
- Technical Indicators
- Learning Engine

To make unified trading decisions
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from loguru import logger
import redis.asyncio as redis
import json

from config import settings
from services.price_predictor import price_predictor, PricePrediction
from services.pattern_recognition import pattern_recognition, PatternSignal
from services.learning_engine import LearningEngine


@dataclass
class AIDecision:
    """Unified AI trading decision"""
    symbol: str
    action: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    confidence: float  # 0-100
    
    # Component scores
    lstm_score: float
    pattern_score: float
    sentiment_score: float
    technical_score: float
    learning_score: float
    
    # Prediction details
    predicted_direction: str
    predicted_change: float
    
    # Risk parameters
    suggested_entry: float
    suggested_stop_loss: float
    suggested_take_profit: float
    suggested_position_size: float  # % of portfolio
    risk_reward_ratio: float
    
    # Reasoning
    reasons: List[str]
    patterns_detected: List[str]
    
    # Metadata
    model_agreement: float
    timestamp: str


class AICoordinator:
    """
    Central AI Coordinator
    
    Combines all AI models into unified decisions:
    1. Gets LSTM price prediction
    2. Analyzes chart patterns
    3. Checks sentiment
    4. Applies learning adjustments
    5. Generates final decision with confidence
    """
    
    def __init__(self):
        self.redis_client = None
        self.learning_engine: Optional[LearningEngine] = None
        self.is_running = False
        
        # Decision statistics
        self.decisions_made = 0
        self.correct_decisions = 0
        
        # Model weights (adjusted based on performance)
        self.model_weights = {
            'lstm': 0.30,
            'patterns': 0.25,
            'sentiment': 0.15,
            'technical': 0.15,
            'learning': 0.15
        }
        
    async def initialize(self, learning_engine: LearningEngine):
        """Initialize coordinator with all AI components"""
        logger.info("Initializing AI Coordinator (Brain)...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.learning_engine = learning_engine
        
        # Initialize sub-components
        await price_predictor.initialize()
        await pattern_recognition.initialize()
        
        # Load saved weights
        await self._load_weights()
        
        self.is_running = True
        
        logger.info("AI Coordinator initialized - All models connected")
        logger.info(f"Model weights: LSTM={self.model_weights['lstm']:.0%}, "
                   f"Patterns={self.model_weights['patterns']:.0%}, "
                   f"Sentiment={self.model_weights['sentiment']:.0%}")
        
    async def shutdown(self):
        """Shutdown all components"""
        await self._save_weights()
        await price_predictor.shutdown()
        await pattern_recognition.shutdown()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def get_ai_decision(self, symbol: str, market_data: Dict) -> Optional[AIDecision]:
        """
        Get unified AI decision for a symbol
        
        This combines all AI models into one decision
        """
        try:
            reasons = []
            patterns_found = []
            
            # === 1. LSTM PRICE PREDICTION ===
            lstm_score = 0.0
            predicted_direction = 'sideways'
            predicted_change = 0.0
            
            prediction = await price_predictor.predict_price(symbol)
            if prediction:
                predicted_direction = prediction.direction
                predicted_change = prediction.predicted_change_percent
                
                if prediction.direction == 'up':
                    lstm_score = prediction.confidence
                    reasons.append(f"LSTM predicts +{predicted_change:.1f}% ({prediction.confidence:.0f}% conf)")
                elif prediction.direction == 'down':
                    lstm_score = -prediction.confidence
                    reasons.append(f"LSTM predicts {predicted_change:.1f}% ({prediction.confidence:.0f}% conf)")
                else:
                    lstm_score = 0
                    reasons.append("LSTM: sideways movement expected")
                    
            # === 2. PATTERN RECOGNITION ===
            pattern_score = 0.0
            patterns = await pattern_recognition.analyze_patterns(symbol)
            
            if patterns:
                bullish_patterns = [p for p in patterns if p.pattern_type == 'bullish']
                bearish_patterns = [p for p in patterns if p.pattern_type == 'bearish']
                
                for p in patterns:
                    patterns_found.append(f"{p.pattern_name} ({p.timeframe})")
                    
                if bullish_patterns:
                    best_bullish = max(bullish_patterns, key=lambda x: x.confidence)
                    pattern_score += best_bullish.confidence * 0.7
                    reasons.append(f"Pattern: {best_bullish.pattern_name} (bullish, {best_bullish.confidence:.0f}%)")
                    
                if bearish_patterns:
                    best_bearish = max(bearish_patterns, key=lambda x: x.confidence)
                    pattern_score -= best_bearish.confidence * 0.7
                    reasons.append(f"Pattern: {best_bearish.pattern_name} (bearish, {best_bearish.confidence:.0f}%)")
                    
            # === 3. SENTIMENT ANALYSIS ===
            sentiment_score = 0.0
            
            try:
                sentiment_data = await self.redis_client.get('data:crypto_news')
                if sentiment_data:
                    sentiment = json.loads(sentiment_data)
                    sentiment_info = sentiment.get('sentiment', {})
                    overall = sentiment_info.get('overall', 'neutral')
                    bullish_pct = sentiment_info.get('bullish_percent', 50)
                    
                    if overall == 'bullish':
                        sentiment_score = min(70, bullish_pct)
                        reasons.append(f"News sentiment: Bullish ({bullish_pct:.0f}%)")
                    elif overall == 'bearish':
                        sentiment_score = -min(70, 100 - bullish_pct)
                        reasons.append(f"News sentiment: Bearish ({100-bullish_pct:.0f}%)")
            except:
                pass
                
            # === 4. TECHNICAL INDICATORS ===
            technical_score = 0.0
            
            rsi = market_data.get('rsi', 50)
            price_change = market_data.get('price_change_24h', 0)
            funding_rate = market_data.get('funding_rate', 0)
            
            # RSI signals
            if rsi < 30:
                technical_score += 40
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                technical_score -= 40
                reasons.append(f"RSI overbought ({rsi:.0f})")
                
            # Momentum
            if 1 < price_change < 5:
                technical_score += 20
            elif -5 < price_change < -1:
                technical_score -= 20
                
            # Funding rate (contrarian)
            if funding_rate < -0.01:
                technical_score += 15
                reasons.append("Shorts paying (bullish)")
            elif funding_rate > 0.01:
                technical_score -= 15
                reasons.append("Longs paying (bearish)")
                
            # === 5. LEARNING ENGINE ===
            learning_score = 0.0
            
            if self.learning_engine:
                regime = market_data.get('regime', 'sideways')
                best_strategy, q_value = self.learning_engine.get_best_strategy(regime)
                
                # Positive Q-value = learned this works
                if q_value > 0.5:
                    learning_score = min(50, q_value * 30)
                    reasons.append(f"AI learned: {best_strategy} works in {regime}")
                elif q_value < -0.5:
                    learning_score = max(-50, q_value * 30)
                    reasons.append(f"AI learned: avoid {regime} markets")
                    
            # === COMBINE SCORES ===
            weighted_score = (
                lstm_score * self.model_weights['lstm'] +
                pattern_score * self.model_weights['patterns'] +
                sentiment_score * self.model_weights['sentiment'] +
                technical_score * self.model_weights['technical'] +
                learning_score * self.model_weights['learning']
            )
            
            # Determine action
            if weighted_score > 60:
                action = 'strong_buy'
            elif weighted_score > 30:
                action = 'buy'
            elif weighted_score < -60:
                action = 'strong_sell'
            elif weighted_score < -30:
                action = 'sell'
            else:
                action = 'hold'
                
            # Calculate confidence
            confidence = min(95, abs(weighted_score))
            
            # Model agreement (how aligned are the models)
            scores = [lstm_score, pattern_score, sentiment_score, technical_score, learning_score]
            positive_count = sum(1 for s in scores if s > 10)
            negative_count = sum(1 for s in scores if s < -10)
            
            if positive_count >= 4 or negative_count >= 4:
                model_agreement = 90
            elif positive_count >= 3 or negative_count >= 3:
                model_agreement = 70
            else:
                model_agreement = 50
                
            # Boost confidence if models agree
            if model_agreement > 70:
                confidence = min(95, confidence * 1.1)
                reasons.append(f"High model agreement ({model_agreement:.0f}%)")
                
            # === RISK PARAMETERS ===
            current_price = market_data.get('current_price', 0)
            
            if action in ['buy', 'strong_buy']:
                suggested_stop = current_price * 0.98  # 2% stop
                suggested_tp = current_price * (1 + abs(predicted_change) / 100 * 1.5)
            elif action in ['sell', 'strong_sell']:
                suggested_stop = current_price * 1.02
                suggested_tp = current_price * (1 - abs(predicted_change) / 100 * 1.5)
            else:
                suggested_stop = current_price * 0.98
                suggested_tp = current_price * 1.02
                
            # Position size based on confidence
            if confidence > 80:
                position_size = 15.0
            elif confidence > 60:
                position_size = 10.0
            else:
                position_size = 5.0
                
            # Risk/Reward ratio
            if current_price > 0:
                risk = abs(current_price - suggested_stop)
                reward = abs(suggested_tp - current_price)
                rr_ratio = reward / risk if risk > 0 else 0
            else:
                rr_ratio = 0
                
            decision = AIDecision(
                symbol=symbol,
                action=action,
                confidence=confidence,
                lstm_score=lstm_score,
                pattern_score=pattern_score,
                sentiment_score=sentiment_score,
                technical_score=technical_score,
                learning_score=learning_score,
                predicted_direction=predicted_direction,
                predicted_change=predicted_change,
                suggested_entry=current_price,
                suggested_stop_loss=suggested_stop,
                suggested_take_profit=suggested_tp,
                suggested_position_size=position_size,
                risk_reward_ratio=rr_ratio,
                reasons=reasons[:5],  # Top 5 reasons
                patterns_detected=patterns_found[:3],
                model_agreement=model_agreement,
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Store decision
            await self._store_decision(decision)
            
            self.decisions_made += 1
            
            # Log decision
            logger.info(f"AI DECISION: {symbol} → {action.upper()} "
                       f"(conf={confidence:.0f}%, LSTM={lstm_score:.0f}, "
                       f"Pattern={pattern_score:.0f}, Tech={technical_score:.0f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"AI decision error for {symbol}: {e}")
            return None
            
    async def learn_from_outcome(self, symbol: str, decision: AIDecision,
                                  actual_change: float, was_profitable: bool):
        """Learn from decision outcomes to improve future decisions"""
        
        # Was the direction correct?
        predicted_up = decision.predicted_direction == 'up'
        actual_up = actual_change > 0
        direction_correct = predicted_up == actual_up
        
        if direction_correct:
            self.correct_decisions += 1
            
        # Adjust model weights based on performance
        if was_profitable:
            # Increase weight of models that contributed to this decision
            if decision.lstm_score > 20:
                self.model_weights['lstm'] = min(0.40, self.model_weights['lstm'] * 1.02)
            if decision.pattern_score > 20:
                self.model_weights['patterns'] = min(0.35, self.model_weights['patterns'] * 1.02)
        else:
            # Decrease weight of models that misled
            if decision.lstm_score > 20:
                self.model_weights['lstm'] = max(0.15, self.model_weights['lstm'] * 0.98)
            if decision.pattern_score > 20:
                self.model_weights['patterns'] = max(0.10, self.model_weights['patterns'] * 0.98)
                
        # Normalize weights to sum to 1
        total = sum(self.model_weights.values())
        for key in self.model_weights:
            self.model_weights[key] /= total
            
        # Update price predictor
        await price_predictor.train_on_outcome(
            symbol, decision.predicted_direction,
            'up' if actual_up else 'down',
            decision.predicted_change, actual_change
        )
        
        logger.info(f"AI LEARNING: {symbol} {'CORRECT' if direction_correct else 'WRONG'} "
                   f"(predicted={decision.predicted_direction}, actual={'up' if actual_up else 'down'})")
                   
    async def _store_decision(self, decision: AIDecision):
        """Store decision for tracking"""
        try:
            await self.redis_client.hset(
                'ai:decisions:latest',
                decision.symbol,
                json.dumps({
                    'action': decision.action,
                    'confidence': decision.confidence,
                    'predicted_change': decision.predicted_change,
                    'reasons': decision.reasons,
                    'patterns': decision.patterns_detected,
                    'model_agreement': decision.model_agreement,
                    'timestamp': decision.timestamp
                })
            )
            
            # Store in history
            await self.redis_client.lpush(
                f'ai:decisions:history:{decision.symbol}',
                json.dumps({
                    'action': decision.action,
                    'confidence': decision.confidence,
                    'timestamp': decision.timestamp
                })
            )
            await self.redis_client.ltrim(f'ai:decisions:history:{decision.symbol}', 0, 99)
            
        except:
            pass
            
    async def _load_weights(self):
        """Load saved model weights"""
        try:
            data = await self.redis_client.get('ai:coordinator:weights')
            if data:
                self.model_weights = json.loads(data)
                logger.info(f"Loaded model weights from storage")
        except:
            pass
            
    async def _save_weights(self):
        """Save model weights"""
        try:
            await self.redis_client.set('ai:coordinator:weights', json.dumps(self.model_weights))
        except:
            pass
            
    async def get_coordinator_stats(self) -> Dict:
        """Get coordinator statistics"""
        accuracy = 0
        if self.decisions_made > 0:
            accuracy = (self.correct_decisions / self.decisions_made) * 100
            
        predictor_stats = await price_predictor.get_prediction_stats()
        pattern_stats = await pattern_recognition.get_pattern_stats()
        
        return {
            'decisions_made': self.decisions_made,
            'correct_decisions': self.correct_decisions,
            'accuracy': round(accuracy, 2),
            'model_weights': self.model_weights,
            'predictor_stats': predictor_stats,
            'pattern_stats': pattern_stats
        }


# Global instance
ai_coordinator = AICoordinator()



