"""
SENTINEL AI - Reinforcement Learning Engine
Self-improvement through trade analysis and reward optimization
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from loguru import logger
import redis.asyncio as redis
import json

from config import settings


@dataclass
class TradeOutcome:
    """Represents a completed trade with all relevant metrics"""
    symbol: str
    strategy: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_percent: float
    hold_time_seconds: int
    market_regime: str
    volatility_at_entry: float
    sentiment_at_entry: float
    timestamp: str


class LearningEngine:
    """
    Reinforcement Learning Engine for continuous improvement.
    
    Reward function: profit + risk_score
    Penalty: drawdown + excessive losses
    
    Learns:
    - Which strategies work best in which market regimes
    - Optimal position sizing based on conditions
    - Best entry/exit timing
    - Risk adjustment based on market conditions
    """
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1  # Start with 10% exploration
        self.min_exploration = 0.01
        self.exploration_decay = 0.995
        
        # Strategy performance tracking
        self.strategy_q_values: Dict[str, Dict[str, float]] = {}
        
        # Market regime definitions
        self.regimes = ['bull_trend', 'bear_trend', 'sideways', 'high_volatility', 'low_liquidity']
        self.strategies = ['momentum', 'grid', 'scalping', 'mean_reversion', 'breakout', 'hedge', 'hold']
        
        # Performance statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
    async def initialize(self):
        """Initialize learning engine"""
        logger.info("Initializing Learning Engine...")
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Load saved Q-values from Redis
        await self._load_q_values()
        
        # Initialize Q-table if empty
        if not self.strategy_q_values:
            self._initialize_q_table()
            
        self.is_running = True
        logger.info("Learning Engine initialized - AI self-improvement active")
        
    async def shutdown(self):
        """Save state and cleanup"""
        self.is_running = False
        await self._save_q_values()
        if self.redis_client:
            await self.redis_client.close()
            
    def _initialize_q_table(self):
        """Initialize Q-values for all state-action pairs"""
        for regime in self.regimes:
            self.strategy_q_values[regime] = {}
            for strategy in self.strategies:
                # Start with neutral value, slightly favoring 'hold' for safety
                self.strategy_q_values[regime][strategy] = 0.0 if strategy != 'hold' else 0.1
                
        # Add some prior knowledge
        # Momentum works well in trends
        self.strategy_q_values['bull_trend']['momentum'] = 0.3
        self.strategy_q_values['bear_trend']['hedge'] = 0.3
        # Grid trading works in sideways
        self.strategy_q_values['sideways']['grid'] = 0.3
        self.strategy_q_values['sideways']['mean_reversion'] = 0.25
        # Be cautious in high volatility
        self.strategy_q_values['high_volatility']['hold'] = 0.4
        self.strategy_q_values['high_volatility']['scalping'] = 0.2
        
    async def _load_q_values(self):
        """Load Q-values from Redis"""
        try:
            data = await self.redis_client.get('learning:q_values')
            if data:
                self.strategy_q_values = json.loads(data)
                logger.info("Loaded Q-values from storage")
                
            # Load stats
            stats = await self.redis_client.hgetall('learning:stats')
            if stats:
                self.total_trades = int(stats.get(b'total_trades', 0))
                self.winning_trades = int(stats.get(b'winning_trades', 0))
                self.total_pnl = float(stats.get(b'total_pnl', 0))
                self.max_drawdown = float(stats.get(b'max_drawdown', 0))
                
        except Exception as e:
            logger.error(f"Failed to load Q-values: {e}")
            
    async def _save_q_values(self):
        """Save Q-values to Redis"""
        try:
            await self.redis_client.set(
                'learning:q_values',
                json.dumps(self.strategy_q_values)
            )
            
            await self.redis_client.hset(
                'learning:stats',
                mapping={
                    'total_trades': str(self.total_trades),
                    'winning_trades': str(self.winning_trades),
                    'total_pnl': str(self.total_pnl),
                    'max_drawdown': str(self.max_drawdown),
                    'win_rate': str(self.winning_trades / max(1, self.total_trades) * 100),
                    'updated_at': datetime.utcnow().isoformat(),
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to save Q-values: {e}")
            
    def calculate_reward(self, trade: TradeOutcome) -> float:
        """
        Calculate reward for a trade.
        
        Reward = profit_factor + risk_factor - time_penalty
        
        Encourages:
        - Profitable trades
        - Quick profitable trades (efficiency)
        - Risk-adjusted returns
        
        Penalizes:
        - Losses (especially large ones)
        - Excessive drawdown
        - Very long hold times
        """
        reward = 0.0
        
        # Profit factor (primary reward)
        if trade.pnl > 0:
            # Positive P&L: reward proportional to profit %
            reward += trade.pnl_percent * 2.0
        else:
            # Negative P&L: penalize more heavily than reward
            reward += trade.pnl_percent * 3.0  # More negative = worse
            
        # Risk-adjusted return (Sharpe-like)
        if trade.volatility_at_entry > 0:
            risk_adjusted = trade.pnl_percent / trade.volatility_at_entry
            reward += risk_adjusted * 0.5
            
        # Time efficiency bonus/penalty
        hold_hours = trade.hold_time_seconds / 3600
        if trade.pnl > 0:
            # Reward quick profitable trades
            if hold_hours < 1:
                reward += 0.2
            elif hold_hours > 24:
                reward -= 0.1  # Penalize holding too long
        else:
            # Extra penalty for holding losers long
            if hold_hours > 6:
                reward -= 0.2
                
        # Sentiment alignment bonus
        if trade.sentiment_at_entry > 0 and trade.side == 'long' and trade.pnl > 0:
            reward += 0.1  # Aligned sentiment with position
        elif trade.sentiment_at_entry < 0 and trade.side == 'short' and trade.pnl > 0:
            reward += 0.1
            
        return reward
        
    async def update_from_trade(self, trade: TradeOutcome):
        """
        Update Q-values based on trade outcome.
        Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(s',a')) - Q(s,a))
        """
        regime = trade.market_regime
        strategy = trade.strategy
        
        if regime not in self.strategy_q_values:
            self.strategy_q_values[regime] = {s: 0.0 for s in self.strategies}
            
        if strategy not in self.strategy_q_values[regime]:
            self.strategy_q_values[regime][strategy] = 0.0
            
        # Calculate reward
        reward = self.calculate_reward(trade)
        
        # Current Q-value
        current_q = self.strategy_q_values[regime][strategy]
        
        # Best future Q-value (assuming regime doesn't change drastically)
        max_future_q = max(self.strategy_q_values[regime].values())
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        
        self.strategy_q_values[regime][strategy] = new_q
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += trade.pnl
        if trade.pnl > 0:
            self.winning_trades += 1
            
        # Track drawdown
        if trade.pnl < 0:
            current_drawdown = abs(trade.pnl_percent)
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
        
        # Save periodically
        if self.total_trades % 10 == 0:
            await self._save_q_values()
            
        # Log learning event
        logger.info(f"Learning update: {regime}/{strategy} Q={new_q:.4f} reward={reward:.4f}")
        
        # Store learning event
        await self.redis_client.lpush(
            'learning:events',
            json.dumps({
                'regime': regime,
                'strategy': strategy,
                'reward': reward,
                'new_q': new_q,
                'pnl': trade.pnl,
                'timestamp': datetime.utcnow().isoformat(),
            })
        )
        await self.redis_client.ltrim('learning:events', 0, 999)
        
    def get_best_strategy(self, regime: str) -> Tuple[str, float]:
        """
        Get the best strategy for a given market regime.
        Uses epsilon-greedy exploration.
        """
        if regime not in self.strategy_q_values:
            return 'hold', 0.5
            
        # Exploration: random strategy
        if np.random.random() < self.exploration_rate:
            random_strategy = np.random.choice(self.strategies)
            return random_strategy, self.strategy_q_values[regime].get(random_strategy, 0)
            
        # Exploitation: best known strategy
        best_strategy = max(
            self.strategy_q_values[regime].items(),
            key=lambda x: x[1]
        )
        
        return best_strategy[0], best_strategy[1]
        
    def get_strategy_confidence(self, regime: str, strategy: str) -> float:
        """Get confidence level for a strategy in a regime (0-100%)"""
        if regime not in self.strategy_q_values:
            return 50.0
            
        q_value = self.strategy_q_values[regime].get(strategy, 0)
        
        # Normalize Q-value to confidence (sigmoid-like)
        # Q values typically range from -2 to +2
        confidence = 50 + (q_value * 25)  # Map to 0-100
        return max(0, min(100, confidence))
        
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades) * 100,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'exploration_rate': self.exploration_rate * 100,
            'strategies_learned': sum(
                1 for regime in self.strategy_q_values.values()
                for q in regime.values() if abs(q) > 0.1
            ),
            'best_performing': self._get_best_performing_strategies(),
        }
        
    def _get_best_performing_strategies(self) -> List[Dict[str, Any]]:
        """Get top performing strategy-regime combinations"""
        all_combinations = []
        
        for regime, strategies in self.strategy_q_values.items():
            for strategy, q_value in strategies.items():
                all_combinations.append({
                    'regime': regime,
                    'strategy': strategy,
                    'q_value': q_value,
                    'confidence': self.get_strategy_confidence(regime, strategy),
                })
                
        # Sort by Q-value and return top 5
        sorted_combinations = sorted(all_combinations, key=lambda x: x['q_value'], reverse=True)
        return sorted_combinations[:5]
        
    async def get_recent_learning_events(self, limit: int = 20) -> List[Dict]:
        """Get recent learning events"""
        events = await self.redis_client.lrange('learning:events', 0, limit - 1)
        return [json.loads(e) for e in events]

