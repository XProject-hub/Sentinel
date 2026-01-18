"""
SENTINEL AI - Training Data Manager
Quality-filtered, multi-user learning system

PURPOSE:
- Filter trades to only train on QUALITY data
- Weight profitable trades higher
- Enable multi-user collective learning
- Prevent bad trades from polluting the model

QUALITY CRITERIA:
1. Profitable trades (PnL > 0.5%)
2. High edge trades (edge > 0.15)
3. XGBoost correct predictions
4. High confidence trades (> 60%)

MULTI-USER BENEFITS:
- More data = Better models
- Diverse strategies = Robust learning
- Collective wins benefit everyone
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import numpy as np
from loguru import logger
import redis.asyncio as redis

from config import settings


@dataclass
class QualityTrade:
    """A quality-filtered trade for training"""
    trade_id: str
    user_id: str
    symbol: str
    timestamp: str
    
    # Entry data
    direction: str
    entry_price: float
    quantity: float
    position_value: float
    
    # AI signals at entry
    edge_score: float
    confidence: float
    regime: str
    xgb_signal: str
    xgb_confidence: float
    finbert_sentiment: float
    
    # Outcome
    exit_price: float
    pnl_percent: float
    pnl_value: float
    duration_seconds: int
    exit_reason: str
    won: bool
    
    # Quality metrics
    quality_score: float  # 0-100
    training_weight: float  # 0.1 - 2.0


@dataclass
class UserContribution:
    """Track each user's contribution to learning"""
    user_id: str
    total_trades: int
    quality_trades: int
    contribution_score: float
    win_rate: float
    avg_pnl: float
    last_trade: str
    

class TrainingDataManager:
    """
    Manages training data quality and multi-user learning
    
    Features:
    - Quality filtering (only good trades)
    - Weighted sampling (winners weight more)
    - Multi-user aggregation
    - Contribution tracking
    - Training dataset generation
    """
    
    # Quality thresholds
    MIN_PNL_FOR_TRAINING = 0.3  # Min 0.3% profit to be "quality"
    MIN_EDGE_FOR_TRAINING = 0.10  # Min edge score
    MIN_CONFIDENCE_FOR_TRAINING = 55  # Min AI confidence
    MAX_LOSS_FOR_TRAINING = -0.5  # Losses smaller than this still useful
    
    # Weighting
    WINNER_WEIGHT_BASE = 1.5  # Winners start with 1.5x weight
    WINNER_WEIGHT_PER_PERCENT = 0.2  # +0.2 weight per % profit
    LOSER_WEIGHT = 0.5  # Losers have 0.5x weight (still learn from mistakes)
    HIGH_EDGE_BONUS = 0.3  # Bonus weight for high edge trades
    CORRECT_PREDICTION_BONUS = 0.5  # Bonus if XGBoost was correct
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # In-memory cache
        self.quality_trades: List[QualityTrade] = []
        self.user_contributions: Dict[str, UserContribution] = {}
        
        # Stats
        self.stats = {
            'total_trades_received': 0,
            'quality_trades_accepted': 0,
            'trades_rejected_low_quality': 0,
            'total_users': 0,
            'avg_quality_score': 0.0
        }
        
    async def initialize(self):
        """Initialize training data manager"""
        logger.info("Initializing Training Data Manager...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.is_running = True
        
        # Load existing quality trades
        await self._load_quality_trades()
        await self._load_user_contributions()
        
        logger.info(f"Training Data Manager initialized - {len(self.quality_trades)} quality trades loaded")
        
    async def shutdown(self):
        """Cleanup"""
        await self._save_quality_trades()
        await self._save_user_contributions()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def process_trade(self, trade_data: Dict, user_id: str = "default") -> Optional[QualityTrade]:
        """
        Process a completed trade and determine if it qualifies for training
        
        Returns QualityTrade if accepted, None if rejected
        """
        self.stats['total_trades_received'] += 1
        
        try:
            # Extract data
            pnl_percent = float(trade_data.get('pnl_percent', 0))
            edge_score = float(trade_data.get('edge_score', 0))
            confidence = float(trade_data.get('confidence', 0))
            xgb_signal = trade_data.get('xgb_signal', 'hold')
            xgb_confidence = float(trade_data.get('xgb_confidence', 50))
            direction = trade_data.get('direction', 'long')
            won = pnl_percent > 0
            
            # Calculate quality score (0-100)
            quality_score = self._calculate_quality_score(
                pnl_percent=pnl_percent,
                edge_score=edge_score,
                confidence=confidence,
                xgb_correct=(xgb_signal == ('buy' if direction == 'long' else 'sell')),
                xgb_confidence=xgb_confidence
            )
            
            # Check if trade meets quality threshold
            if not self._meets_quality_threshold(pnl_percent, edge_score, confidence, quality_score):
                self.stats['trades_rejected_low_quality'] += 1
                logger.debug(f"Trade rejected: quality={quality_score:.1f}, pnl={pnl_percent:.2f}%")
                return None
                
            # Calculate training weight
            training_weight = self._calculate_training_weight(
                pnl_percent=pnl_percent,
                edge_score=edge_score,
                xgb_correct=(xgb_signal == ('buy' if direction == 'long' else 'sell')),
                won=won
            )
            
            # Create quality trade record
            quality_trade = QualityTrade(
                trade_id=trade_data.get('trade_id', f"{trade_data.get('symbol')}_{datetime.utcnow().timestamp()}"),
                user_id=user_id,
                symbol=trade_data.get('symbol', 'UNKNOWN'),
                timestamp=datetime.utcnow().isoformat(),
                direction=direction,
                entry_price=float(trade_data.get('entry_price', 0)),
                quantity=float(trade_data.get('quantity', 0)),
                position_value=float(trade_data.get('position_value', 0)),
                edge_score=edge_score,
                confidence=confidence,
                regime=trade_data.get('regime', 'unknown'),
                xgb_signal=xgb_signal,
                xgb_confidence=xgb_confidence,
                finbert_sentiment=float(trade_data.get('finbert_sentiment', 0)),
                exit_price=float(trade_data.get('exit_price', 0)),
                pnl_percent=pnl_percent,
                pnl_value=float(trade_data.get('pnl_value', 0)),
                duration_seconds=int(trade_data.get('duration_seconds', 0)),
                exit_reason=trade_data.get('exit_reason', 'unknown'),
                won=won,
                quality_score=quality_score,
                training_weight=training_weight
            )
            
            # Store
            self.quality_trades.append(quality_trade)
            self.stats['quality_trades_accepted'] += 1
            
            # Update user contribution
            await self._update_user_contribution(user_id, quality_trade)
            
            # Store in Redis
            await self._store_quality_trade(quality_trade)
            
            # Update average quality
            self.stats['avg_quality_score'] = np.mean([t.quality_score for t in self.quality_trades])
            
            logger.info(f"Quality trade accepted: {quality_trade.symbol} | "
                       f"PnL: {pnl_percent:.2f}% | Quality: {quality_score:.1f} | "
                       f"Weight: {training_weight:.2f} | User: {user_id}")
            
            return quality_trade
            
        except Exception as e:
            logger.error(f"Trade processing error: {e}")
            return None
            
    def _calculate_quality_score(self, pnl_percent: float, edge_score: float,
                                  confidence: float, xgb_correct: bool,
                                  xgb_confidence: float) -> float:
        """Calculate quality score (0-100)"""
        score = 0.0
        
        # PnL component (max 40 points)
        if pnl_percent > 0:
            score += min(40, pnl_percent * 10)  # 10 points per % profit, max 40
        else:
            score += max(0, 20 + pnl_percent * 10)  # Losses reduce from 20
            
        # Edge component (max 20 points)
        score += min(20, edge_score * 40)  # 0.5 edge = 20 points
        
        # Confidence component (max 20 points)
        score += (confidence / 100) * 20
        
        # XGBoost correctness (max 20 points)
        if xgb_correct:
            score += 10
            score += (xgb_confidence / 100) * 10
            
        return min(100, max(0, score))
        
    def _meets_quality_threshold(self, pnl_percent: float, edge_score: float,
                                  confidence: float, quality_score: float) -> bool:
        """Check if trade meets quality threshold for training"""
        
        # Always accept big winners
        if pnl_percent >= 2.0:
            return True
            
        # Always accept high edge + winner
        if pnl_percent > 0 and edge_score >= 0.25:
            return True
            
        # Accept medium winners with good signals
        if pnl_percent >= self.MIN_PNL_FOR_TRAINING:
            if edge_score >= self.MIN_EDGE_FOR_TRAINING or confidence >= self.MIN_CONFIDENCE_FOR_TRAINING:
                return True
                
        # Accept instructive losses (learn what NOT to do)
        if self.MAX_LOSS_FOR_TRAINING <= pnl_percent < 0:
            if edge_score >= 0.2:  # Only if we had edge but still lost
                return True
                
        # Fallback to quality score
        return quality_score >= 50
        
    def _calculate_training_weight(self, pnl_percent: float, edge_score: float,
                                    xgb_correct: bool, won: bool) -> float:
        """Calculate training weight for this sample"""
        
        if won:
            # Winners: base weight + bonus per % profit
            weight = self.WINNER_WEIGHT_BASE + (pnl_percent * self.WINNER_WEIGHT_PER_PERCENT)
        else:
            # Losers: lower weight but still learn
            weight = self.LOSER_WEIGHT
            
        # High edge bonus
        if edge_score >= 0.3:
            weight += self.HIGH_EDGE_BONUS
            
        # XGBoost correct bonus
        if xgb_correct:
            weight += self.CORRECT_PREDICTION_BONUS
            
        # Cap weight
        return min(3.0, max(0.1, weight))
        
    async def _update_user_contribution(self, user_id: str, trade: QualityTrade):
        """Update user's contribution metrics"""
        if user_id not in self.user_contributions:
            self.user_contributions[user_id] = UserContribution(
                user_id=user_id,
                total_trades=0,
                quality_trades=0,
                contribution_score=0.0,
                win_rate=0.0,
                avg_pnl=0.0,
                last_trade=trade.timestamp
            )
            self.stats['total_users'] = len(self.user_contributions)
            
        contrib = self.user_contributions[user_id]
        contrib.total_trades += 1
        contrib.quality_trades += 1
        contrib.last_trade = trade.timestamp
        
        # Recalculate metrics
        user_trades = [t for t in self.quality_trades if t.user_id == user_id]
        if user_trades:
            contrib.win_rate = sum(1 for t in user_trades if t.won) / len(user_trades) * 100
            contrib.avg_pnl = np.mean([t.pnl_percent for t in user_trades])
            contrib.contribution_score = len(user_trades) * (contrib.win_rate / 100)
            
    async def _store_quality_trade(self, trade: QualityTrade):
        """Store quality trade in Redis"""
        try:
            await self.redis_client.lpush(
                'training:quality_trades',
                json.dumps(asdict(trade))
            )
            await self.redis_client.ltrim('training:quality_trades', 0, 49999)  # Keep 50K
            
            # Also store by user
            await self.redis_client.lpush(
                f'training:user:{trade.user_id}',
                json.dumps(asdict(trade))
            )
            await self.redis_client.ltrim(f'training:user:{trade.user_id}', 0, 9999)
            
        except Exception as e:
            logger.error(f"Store quality trade error: {e}")
            
    async def _load_quality_trades(self):
        """Load quality trades from Redis"""
        try:
            data = await self.redis_client.lrange('training:quality_trades', 0, -1)
            for item in data:
                trade_dict = json.loads(item)
                self.quality_trades.append(QualityTrade(**trade_dict))
            logger.info(f"Loaded {len(self.quality_trades)} quality trades")
        except Exception as e:
            logger.warning(f"Load quality trades error: {e}")
            
    async def _save_quality_trades(self):
        """Save quality trades to Redis (already done incrementally)"""
        pass
        
    async def _load_user_contributions(self):
        """Load user contributions from Redis"""
        try:
            data = await self.redis_client.hgetall('training:user_contributions')
            for user_id, contrib_json in data.items():
                user_id = user_id.decode() if isinstance(user_id, bytes) else user_id
                contrib_dict = json.loads(contrib_json)
                self.user_contributions[user_id] = UserContribution(**contrib_dict)
            self.stats['total_users'] = len(self.user_contributions)
        except Exception as e:
            logger.warning(f"Load user contributions error: {e}")
            
    async def _save_user_contributions(self):
        """Save user contributions to Redis"""
        try:
            for user_id, contrib in self.user_contributions.items():
                await self.redis_client.hset(
                    'training:user_contributions',
                    user_id,
                    json.dumps(asdict(contrib))
                )
        except Exception as e:
            logger.error(f"Save user contributions error: {e}")
            
    async def get_training_dataset(self, max_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training dataset with quality filtering and weighting
        
        Returns:
            X: Feature matrix
            y: Labels (0=hold, 1=buy, 2=sell)
            weights: Sample weights
        """
        if not self.quality_trades:
            return np.array([]), np.array([]), np.array([])
            
        # Sample with replacement based on weights
        trades = self.quality_trades[-max_samples:]  # Most recent
        
        X_list = []
        y_list = []
        weights_list = []
        
        for trade in trades:
            # Create feature vector
            features = [
                trade.edge_score,
                trade.confidence / 100,
                trade.xgb_confidence / 100,
                trade.finbert_sentiment,
                1 if trade.regime == 'trending' else 0,
                1 if trade.regime == 'ranging' else 0,
                1 if trade.regime == 'volatile' else 0,
            ]
            
            # Label: What SHOULD have happened
            if trade.won and trade.pnl_percent > 1.0:
                # Good trade - label as the action taken
                label = 1 if trade.direction == 'long' else 2
            elif trade.won:
                # Small win - could be hold
                label = 0
            else:
                # Loss - opposite action or hold
                if trade.pnl_percent < -1.0:
                    # Big loss - should have done opposite
                    label = 2 if trade.direction == 'long' else 1
                else:
                    # Small loss - hold
                    label = 0
                    
            X_list.append(features)
            y_list.append(label)
            weights_list.append(trade.training_weight)
            
        return np.array(X_list), np.array(y_list), np.array(weights_list)
        
    async def get_stats(self) -> Dict:
        """Get training data statistics"""
        return {
            **self.stats,
            'quality_trades_count': len(self.quality_trades),
            'user_count': len(self.user_contributions),
            'users': [
                {
                    'user_id': c.user_id,
                    'quality_trades': c.quality_trades,
                    'win_rate': c.win_rate,
                    'avg_pnl': c.avg_pnl,
                    'contribution_score': c.contribution_score
                }
                for c in sorted(
                    self.user_contributions.values(),
                    key=lambda x: x.contribution_score,
                    reverse=True
                )[:10]  # Top 10 contributors
            ]
        }
        
    async def get_leaderboard(self) -> List[Dict]:
        """Get user contribution leaderboard"""
        return [
            {
                'rank': i + 1,
                'user_id': c.user_id,
                'quality_trades': c.quality_trades,
                'win_rate': round(c.win_rate, 1),
                'avg_pnl': round(c.avg_pnl, 2),
                'contribution_score': round(c.contribution_score, 1),
                'last_trade': c.last_trade
            }
            for i, c in enumerate(sorted(
                self.user_contributions.values(),
                key=lambda x: x.contribution_score,
                reverse=True
            ))
        ]


# Global instance
training_data_manager = TrainingDataManager()

