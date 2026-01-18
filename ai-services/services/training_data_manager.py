"""
SENTINEL AI - Training Data Manager V2
Professional-grade quality filtering and multi-user learning

FIXES FROM REVIEW:
1. PnL is NOT primary criterion - multi-dimensional quality score
2. User ID dropped from learning - user is source, not signal
3. De-duplication - same market state = downweight or skip
4. Exploration isolated - shadow mode for experiments

QUALITY SCORE = MULTI-DIMENSIONAL:
  w1 * edge_strength
+ w2 * regime_stability
+ w3 * liquidity_score
+ w4 * RR_realized (risk/reward)
+ w5 * execution_quality
- w6 * slippage_penalty
- w7 * correlation_penalty

DE-DUPLICATION:
  If (market_state, regime, time_bucket, strategy) already exists:
     → downweight OR skip

USER HANDLING:
  User trade → map to (market_state, action, outcome) → DROP user_id → learning
  Bot learns WHAT happened in market, not WHO clicked.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import hashlib
import numpy as np
from loguru import logger
import redis.asyncio as redis

from config import settings


@dataclass
class MarketContext:
    """Market context at trade time - used for de-duplication"""
    symbol: str
    regime: str
    volatility_bucket: str  # 'low', 'medium', 'high'
    trend_bucket: str  # 'strong_up', 'up', 'neutral', 'down', 'strong_down'
    time_bucket: str  # Hour of day + day of week
    liquidity_bucket: str  # 'low', 'medium', 'high'
    
    def get_hash(self) -> str:
        """Get unique hash for this market context"""
        context_str = f"{self.symbol}:{self.regime}:{self.volatility_bucket}:{self.trend_bucket}:{self.time_bucket}:{self.liquidity_bucket}"
        return hashlib.md5(context_str.encode()).hexdigest()[:16]


@dataclass
class QualityTrade:
    """A quality-filtered trade for training - NO USER_ID in model"""
    trade_id: str
    timestamp: str
    
    # Market context (for de-duplication)
    context_hash: str
    symbol: str
    regime: str
    volatility: float
    liquidity: float
    trend_strength: float
    
    # Entry signals
    direction: str
    entry_price: float
    
    # AI signals at entry (what model predicted)
    edge_score: float
    confidence: float
    xgb_signal: str
    xgb_confidence: float
    sentiment_score: float
    
    # Outcome (what actually happened)
    exit_price: float
    pnl_percent: float
    realized_rr: float  # Realized risk/reward
    slippage_percent: float
    duration_seconds: int
    exit_reason: str
    
    # Quality metrics (multi-dimensional)
    quality_score: float  # 0-100, multi-dimensional
    training_weight: float  # 0.1 - 2.0
    is_duplicate: bool  # Flagged as similar to existing
    
    # For stats only - NOT used in model
    _user_id: str = ""  # Underscore = private, not for model


@dataclass
class LearningStats:
    """Statistics for learning (user-agnostic)"""
    total_trades_received: int = 0
    quality_trades_accepted: int = 0
    trades_rejected_low_quality: int = 0
    trades_downweighted_duplicate: int = 0
    unique_market_contexts: int = 0
    unique_symbols_seen: int = 0
    avg_quality_score: float = 0.0
    

class TrainingDataManager:
    """
    Professional Training Data Manager
    
    Key Principles:
    1. Multi-dimensional quality score (not just PnL)
    2. De-duplication (same market state = skip/downweight)
    3. User ID dropped from model (user is source, not signal)
    4. Diverse market situations valued over volume
    """
    
    # Quality weights (multi-dimensional)
    W_EDGE = 0.25       # Edge at entry
    W_REGIME = 0.15     # Regime stability
    W_LIQUIDITY = 0.10  # Liquidity score
    W_RR = 0.20         # Realized risk/reward
    W_EXECUTION = 0.10  # Execution quality
    W_SLIPPAGE = -0.10  # Slippage penalty
    W_CORRELATION = -0.10  # Correlation with recent trades
    
    # Thresholds
    MIN_QUALITY_SCORE = 40  # Minimum to accept
    DUPLICATE_WEIGHT_FACTOR = 0.3  # Duplicates get 30% weight
    MAX_SAME_CONTEXT_PER_DAY = 5  # Max similar trades per context per day
    
    # Time buckets
    TIME_BUCKETS = ['asian_early', 'asian_late', 'london_early', 'london_late', 
                    'ny_early', 'ny_late', 'weekend']
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # Quality trades (user_id stripped for model)
        self.quality_trades: List[QualityTrade] = []
        
        # De-duplication tracking
        self.seen_contexts: Dict[str, int] = {}  # context_hash -> count today
        self.context_last_reset: datetime = datetime.utcnow()
        
        # Statistics
        self.stats = LearningStats()
        
        # For UI only (not model) - track user contributions
        self._user_stats: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize training data manager"""
        logger.info("Initializing Training Data Manager V2 (Professional)...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.is_running = True
        
        await self._load_quality_trades()
        await self._load_seen_contexts()
        
        logger.info(f"Training Data Manager V2 ready - {len(self.quality_trades)} quality trades, "
                   f"{len(self.seen_contexts)} unique contexts")
        
    async def shutdown(self):
        """Cleanup"""
        await self._save_quality_trades()
        await self._save_seen_contexts()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def process_trade(self, trade_data: Dict, user_id: str = "anonymous") -> Optional[QualityTrade]:
        """
        Process a completed trade with professional quality filtering
        
        Key changes:
        - PnL is NOT primary criterion
        - User ID is tracked for stats but NOT used in model
        - De-duplication based on market context
        """
        self.stats.total_trades_received += 1
        
        try:
            # === 1. Build Market Context ===
            context = self._build_market_context(trade_data)
            context_hash = context.get_hash()
            
            # === 2. Check De-duplication ===
            is_duplicate, dup_reason = await self._check_duplication(context_hash)
            
            # === 3. Calculate Multi-Dimensional Quality Score ===
            quality_score, quality_breakdown = self._calculate_quality_score_v2(trade_data)
            
            # === 4. Decision: Accept, Downweight, or Reject ===
            if quality_score < self.MIN_QUALITY_SCORE and not self._is_instructive_loss(trade_data):
                self.stats.trades_rejected_low_quality += 1
                logger.debug(f"Trade rejected: quality={quality_score:.1f} | {quality_breakdown}")
                return None
                
            # === 5. Calculate Training Weight ===
            training_weight = self._calculate_training_weight_v2(
                quality_score=quality_score,
                is_duplicate=is_duplicate,
                trade_data=trade_data
            )
            
            if is_duplicate:
                self.stats.trades_downweighted_duplicate += 1
                
            # === 6. Create Quality Trade (NO USER_ID FOR MODEL) ===
            pnl = float(trade_data.get('pnl_percent', 0))
            entry_price = float(trade_data.get('entry_price', 0))
            exit_price = float(trade_data.get('exit_price', 0))
            
            # Calculate realized R:R
            sl_price = float(trade_data.get('stop_loss', entry_price * 0.99))
            risk = abs(entry_price - sl_price)
            reward = abs(exit_price - entry_price)
            realized_rr = reward / risk if risk > 0 else 0
            
            # Estimate slippage
            expected_price = float(trade_data.get('expected_price', entry_price))
            slippage = abs(entry_price - expected_price) / expected_price * 100 if expected_price > 0 else 0
            
            quality_trade = QualityTrade(
                trade_id=trade_data.get('trade_id', f"{context_hash}_{datetime.utcnow().timestamp()}"),
                timestamp=datetime.utcnow().isoformat(),
                context_hash=context_hash,
                symbol=trade_data.get('symbol', 'UNKNOWN'),
                regime=trade_data.get('regime', 'unknown'),
                volatility=float(trade_data.get('volatility', 0)),
                liquidity=float(trade_data.get('liquidity', 50)),
                trend_strength=float(trade_data.get('trend_strength', 0)),
                direction=trade_data.get('direction', 'long'),
                entry_price=entry_price,
                edge_score=float(trade_data.get('edge_score', 0)),
                confidence=float(trade_data.get('confidence', 0)),
                xgb_signal=trade_data.get('xgb_signal', 'hold'),
                xgb_confidence=float(trade_data.get('xgb_confidence', 50)),
                sentiment_score=float(trade_data.get('sentiment_score', 0)),
                exit_price=exit_price,
                pnl_percent=pnl,
                realized_rr=realized_rr,
                slippage_percent=slippage,
                duration_seconds=int(trade_data.get('duration_seconds', 0)),
                exit_reason=trade_data.get('exit_reason', 'unknown'),
                quality_score=quality_score,
                training_weight=training_weight,
                is_duplicate=is_duplicate,
                _user_id=user_id  # For stats only
            )
            
            # Store
            self.quality_trades.append(quality_trade)
            self.stats.quality_trades_accepted += 1
            
            # Update context tracking
            await self._update_context_tracking(context_hash)
            
            # Update user stats (for UI only, not model)
            await self._update_user_stats(user_id, quality_trade)
            
            # Save to Redis
            await self._store_quality_trade(quality_trade)
            
            # Update avg quality
            self.stats.avg_quality_score = np.mean([t.quality_score for t in self.quality_trades[-1000:]])
            
            logger.info(f"Quality trade: {quality_trade.symbol} | "
                       f"Q={quality_score:.1f} | W={training_weight:.2f} | "
                       f"Dup={is_duplicate} | {quality_breakdown}")
            
            return quality_trade
            
        except Exception as e:
            logger.error(f"Trade processing error: {e}")
            return None
            
    def _build_market_context(self, trade_data: Dict) -> MarketContext:
        """Build market context for de-duplication"""
        
        # Volatility bucket
        vol = float(trade_data.get('volatility', 1))
        if vol < 1:
            vol_bucket = 'low'
        elif vol < 3:
            vol_bucket = 'medium'
        else:
            vol_bucket = 'high'
            
        # Trend bucket
        trend = float(trade_data.get('trend_strength', 0))
        if trend > 0.6:
            trend_bucket = 'strong_up'
        elif trend > 0.2:
            trend_bucket = 'up'
        elif trend > -0.2:
            trend_bucket = 'neutral'
        elif trend > -0.6:
            trend_bucket = 'down'
        else:
            trend_bucket = 'strong_down'
            
        # Time bucket
        now = datetime.utcnow()
        hour = now.hour
        weekday = now.weekday()
        
        if weekday >= 5:
            time_bucket = 'weekend'
        elif hour < 8:
            time_bucket = 'asian_early'
        elif hour < 12:
            time_bucket = 'asian_late'
        elif hour < 16:
            time_bucket = 'london_early'
        elif hour < 20:
            time_bucket = 'london_late'
        else:
            time_bucket = 'ny_late'
            
        # Liquidity bucket
        liq = float(trade_data.get('liquidity', 50))
        if liq < 30:
            liq_bucket = 'low'
        elif liq < 70:
            liq_bucket = 'medium'
        else:
            liq_bucket = 'high'
            
        return MarketContext(
            symbol=trade_data.get('symbol', 'UNKNOWN'),
            regime=trade_data.get('regime', 'unknown'),
            volatility_bucket=vol_bucket,
            trend_bucket=trend_bucket,
            time_bucket=time_bucket,
            liquidity_bucket=liq_bucket
        )
        
    async def _check_duplication(self, context_hash: str) -> Tuple[bool, str]:
        """Check if this market context is a duplicate"""
        
        # Reset daily
        now = datetime.utcnow()
        if (now - self.context_last_reset).total_seconds() > 86400:
            self.seen_contexts.clear()
            self.context_last_reset = now
            
        count = self.seen_contexts.get(context_hash, 0)
        
        if count >= self.MAX_SAME_CONTEXT_PER_DAY:
            return True, f"Context seen {count}x today"
        elif count > 0:
            return True, f"Similar context (#{count+1})"
            
        return False, ""
        
    async def _update_context_tracking(self, context_hash: str):
        """Update context tracking"""
        self.seen_contexts[context_hash] = self.seen_contexts.get(context_hash, 0) + 1
        self.stats.unique_market_contexts = len(self.seen_contexts)
        
    def _calculate_quality_score_v2(self, trade_data: Dict) -> Tuple[float, str]:
        """
        Multi-dimensional quality score
        
        NOT just PnL - considers:
        - Edge at entry
        - Regime stability
        - Liquidity
        - Realized R:R
        - Execution quality
        - Slippage penalty
        """
        scores = {}
        
        # 1. Edge strength (0-25 points)
        edge = float(trade_data.get('edge_score', 0))
        scores['edge'] = min(25, edge * 50)  # 0.5 edge = 25 points
        
        # 2. Regime stability (0-15 points)
        regime = trade_data.get('regime', 'unknown')
        regime_scores = {
            'high_liquidity_trend': 15,
            'trending': 12,
            'accumulation': 10,
            'ranging': 8,
            'distribution': 5,
            'high_volatility': 3,
            'news_spike': 0,  # Never trust news spikes
            'unknown': 5
        }
        scores['regime'] = regime_scores.get(regime, 5)
        
        # 3. Liquidity (0-10 points)
        liquidity = float(trade_data.get('liquidity', 50))
        scores['liquidity'] = min(10, liquidity / 10)
        
        # 4. Realized R:R (0-20 points)
        pnl = float(trade_data.get('pnl_percent', 0))
        sl_pct = float(trade_data.get('stop_loss_percent', 1.0))
        
        if pnl > 0 and sl_pct > 0:
            realized_rr = pnl / sl_pct
            scores['rr'] = min(20, realized_rr * 10)  # 2:1 R:R = 20 points
        elif pnl < 0:
            scores['rr'] = max(0, 10 + pnl * 5)  # Losses reduce from 10
        else:
            scores['rr'] = 5
            
        # 5. Execution quality (0-10 points)
        confidence = float(trade_data.get('confidence', 50))
        xgb_correct = trade_data.get('xgb_signal') == ('buy' if trade_data.get('direction') == 'long' else 'sell')
        scores['execution'] = (confidence / 100) * 5
        if xgb_correct:
            scores['execution'] += 5
            
        # 6. Slippage penalty (0 to -10 points)
        expected_price = float(trade_data.get('expected_price', 0))
        entry_price = float(trade_data.get('entry_price', 0))
        if expected_price > 0 and entry_price > 0:
            slippage = abs(entry_price - expected_price) / expected_price * 100
            scores['slippage'] = max(-10, -slippage * 5)  # 2% slippage = -10
        else:
            scores['slippage'] = 0
            
        # 7. PnL bonus (NOT primary, just bonus for big wins)
        if pnl > 3:
            scores['pnl_bonus'] = min(10, pnl - 3)  # Bonus for 3%+ wins
        elif pnl > 1:
            scores['pnl_bonus'] = (pnl - 1) * 2.5
        else:
            scores['pnl_bonus'] = 0
            
        # Calculate total
        total = sum(scores.values())
        total = max(0, min(100, total))
        
        breakdown = f"E:{scores['edge']:.0f} R:{scores['regime']:.0f} L:{scores['liquidity']:.0f} " \
                   f"RR:{scores['rr']:.0f} X:{scores['execution']:.0f} S:{scores['slippage']:.0f}"
        
        return total, breakdown
        
    def _is_instructive_loss(self, trade_data: Dict) -> bool:
        """Check if a loss is instructive (worth learning from)"""
        pnl = float(trade_data.get('pnl_percent', 0))
        edge = float(trade_data.get('edge_score', 0))
        
        # Instructive: Had good edge but still lost
        if -2.0 < pnl < -0.3 and edge > 0.2:
            return True
            
        # Instructive: Quick stop loss (good risk management)
        duration = int(trade_data.get('duration_seconds', 0))
        if pnl < 0 and duration < 300:  # Lost but exited fast
            return True
            
        return False
        
    def _calculate_training_weight_v2(self, quality_score: float, 
                                       is_duplicate: bool, trade_data: Dict) -> float:
        """Calculate training weight with de-duplication"""
        
        # Base weight from quality score
        base_weight = 0.5 + (quality_score / 100) * 1.0  # 0.5 to 1.5
        
        # Duplicate penalty
        if is_duplicate:
            base_weight *= self.DUPLICATE_WEIGHT_FACTOR
            
        # Bonus for diverse conditions
        regime = trade_data.get('regime', 'unknown')
        if regime in ['high_volatility', 'news_spike', 'distribution']:
            # Rare but valuable learning
            base_weight *= 1.2
            
        # Cap
        return min(2.0, max(0.1, base_weight))
        
    async def _update_user_stats(self, user_id: str, trade: QualityTrade):
        """Update user stats (for UI only, NOT model)"""
        if user_id not in self._user_stats:
            self._user_stats[user_id] = {
                'trades': 0,
                'quality_trades': 0,
                'total_pnl': 0,
                'wins': 0
            }
            
        self._user_stats[user_id]['trades'] += 1
        self._user_stats[user_id]['quality_trades'] += 1
        self._user_stats[user_id]['total_pnl'] += trade.pnl_percent
        if trade.pnl_percent > 0:
            self._user_stats[user_id]['wins'] += 1
            
    async def _store_quality_trade(self, trade: QualityTrade):
        """Store quality trade (WITHOUT user_id in model data)"""
        try:
            # Model data (no user_id)
            model_data = asdict(trade)
            del model_data['_user_id']  # Remove user_id
            
            await self.redis_client.lpush(
                'training:quality_trades_v2',
                json.dumps(model_data)
            )
            await self.redis_client.ltrim('training:quality_trades_v2', 0, 49999)
            
        except Exception as e:
            logger.error(f"Store quality trade error: {e}")
            
    async def _load_quality_trades(self):
        """Load quality trades"""
        try:
            data = await self.redis_client.lrange('training:quality_trades_v2', 0, -1)
            for item in data:
                trade_dict = json.loads(item)
                trade_dict['_user_id'] = ''  # Ensure no user_id
                self.quality_trades.append(QualityTrade(**trade_dict))
        except Exception as e:
            logger.warning(f"Load quality trades error: {e}")
            
    async def _save_quality_trades(self):
        """Already saved incrementally"""
        pass
        
    async def _load_seen_contexts(self):
        """Load seen contexts"""
        try:
            data = await self.redis_client.get('training:seen_contexts')
            if data:
                self.seen_contexts = json.loads(data)
        except:
            pass
            
    async def _save_seen_contexts(self):
        """Save seen contexts"""
        try:
            await self.redis_client.set('training:seen_contexts', json.dumps(self.seen_contexts))
        except:
            pass
            
    async def get_training_dataset(self, max_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training dataset
        
        Returns:
            X: Feature matrix (NO USER_ID)
            y: Labels
            weights: Sample weights
        """
        if not self.quality_trades:
            return np.array([]), np.array([]), np.array([])
            
        trades = self.quality_trades[-max_samples:]
        
        X_list = []
        y_list = []
        weights_list = []
        
        for trade in trades:
            # Features - MARKET DATA ONLY (no user info)
            features = [
                trade.edge_score,
                trade.confidence / 100,
                trade.xgb_confidence / 100,
                trade.sentiment_score,
                trade.volatility / 5,  # Normalize
                trade.liquidity / 100,
                trade.trend_strength,
                # Regime one-hot
                1 if trade.regime == 'trending' else 0,
                1 if trade.regime == 'ranging' else 0,
                1 if trade.regime == 'high_volatility' else 0,
                1 if trade.regime == 'accumulation' else 0,
            ]
            
            # Label based on outcome
            if trade.pnl_percent > 1.0 and trade.realized_rr > 1.5:
                # Strong win with good R:R - correct action
                label = 1 if trade.direction == 'long' else 2
            elif trade.pnl_percent < -1.0:
                # Loss - should have done opposite or held
                label = 0  # Hold would have been better
            else:
                label = 0  # Neutral
                
            X_list.append(features)
            y_list.append(label)
            weights_list.append(trade.training_weight)
            
        return np.array(X_list), np.array(y_list), np.array(weights_list)
        
    async def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'total_trades_received': self.stats.total_trades_received,
            'quality_trades_accepted': self.stats.quality_trades_accepted,
            'trades_rejected_low_quality': self.stats.trades_rejected_low_quality,
            'trades_downweighted_duplicate': self.stats.trades_downweighted_duplicate,
            'unique_market_contexts': len(self.seen_contexts),
            'quality_trades_count': len(self.quality_trades),
            'avg_quality_score': round(self.stats.avg_quality_score, 1),
            'duplicate_rate': round(self.stats.trades_downweighted_duplicate / 
                                   max(1, self.stats.quality_trades_accepted) * 100, 1)
        }
        
    async def get_leaderboard(self) -> List[Dict]:
        """Get user contribution leaderboard (UI only, not model)"""
        leaderboard = []
        for user_id, stats in self._user_stats.items():
            win_rate = (stats['wins'] / max(1, stats['quality_trades'])) * 100
            leaderboard.append({
                'user_id': user_id[:8] + '...',  # Anonymized
                'quality_trades': stats['quality_trades'],
                'win_rate': round(win_rate, 1),
                'avg_pnl': round(stats['total_pnl'] / max(1, stats['quality_trades']), 2)
            })
            
        return sorted(leaderboard, key=lambda x: x['quality_trades'], reverse=True)[:10]


# Global instance
training_data_manager = TrainingDataManager()
