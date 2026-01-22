"""
Edge Estimator - Calculates statistical edge for trading opportunities

Edge = probability of profit * average win - probability of loss * average loss
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EdgeScore:
    """Edge estimation result"""
    edge: float  # Expected value per trade
    confidence: float  # 0-1
    win_probability: float
    avg_win: float
    avg_loss: float
    sample_size: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class EdgeData:
    """Edge calculation result for market scanner"""
    edge: float
    confidence: float
    reasons: List[str]
    warnings: List[str]
    win_probability: float = 0.5  # Default 50%
    risk_reward_ratio: float = 1.0  # Default 1:1
    symbol: str = ""  # Symbol being analyzed
    kelly_fraction: float = 0.0  # Kelly Criterion fraction (0-1)
    recommended_size: str = "normal"  # 'skip', 'small', 'normal', 'large'
    
    def __post_init__(self):
        """Calculate Kelly fraction from win probability and risk/reward"""
        if self.kelly_fraction == 0.0 and self.win_probability > 0:
            # Kelly Criterion: f = (bp - q) / b
            # where b = risk_reward_ratio, p = win_probability, q = 1-p
            p = self.win_probability
            q = 1 - p
            b = max(0.1, self.risk_reward_ratio)  # Avoid division by zero
            
            raw_kelly = (b * p - q) / b if b > 0 else 0
            
            # Clamp to reasonable range (0 to 0.25 = max 25% of bankroll)
            self.kelly_fraction = max(0.0, min(0.25, raw_kelly))
        
        # Determine recommended size based on edge and confidence
        if self.edge <= 0 or self.confidence < 30:
            self.recommended_size = 'skip'
        elif self.edge < 0.15 or self.confidence < 50:
            self.recommended_size = 'small'
        elif self.edge > 0.40 and self.confidence > 70:
            self.recommended_size = 'large'
        else:
            self.recommended_size = 'normal'


class EdgeEstimator:
    """
    Estimates statistical edge for trading opportunities
    
    Uses historical performance to calculate expected value
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self._symbol_stats: Dict[str, Dict] = {}
        self._global_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0
        }
        
    async def initialize(self, redis_client=None):
        """Initialize with Redis client"""
        if redis_client:
            self.redis_client = redis_client
        
        # Try to load stats from Redis
        if self.redis_client:
            try:
                import json
                data = await self.redis_client.get('edge:calibration')
                if data:
                    self._global_stats = json.loads(data)
            except:
                pass
                
        logger.info("Edge Estimator initialized")
        
    async def estimate_edge(self, symbol: str, direction: str, 
                           confidence: float = 0.5, regime: str = 'RANGE') -> EdgeScore:
        """
        Estimate edge for a potential trade
        
        Args:
            symbol: Trading pair
            direction: 'long' or 'short'
            confidence: AI confidence (0-1)
            regime: Current market regime
            
        Returns:
            EdgeScore with expected value calculation
        """
        try:
            # Get symbol-specific stats or use global
            stats = self._symbol_stats.get(symbol, self._global_stats)
            
            total = stats.get('total_trades', 0)
            if total < 10:
                # Not enough data, use confidence as proxy
                win_prob = confidence
                avg_win = 1.0  # Assume 1% average win
                avg_loss = 0.8  # Assume 0.8% average loss
                sample_size = total
            else:
                wins = stats.get('winning_trades', 0)
                win_prob = wins / total
                
                total_profit = stats.get('total_profit', 0)
                total_loss = abs(stats.get('total_loss', 0))
                
                avg_win = total_profit / wins if wins > 0 else 1.0
                avg_loss = total_loss / (total - wins) if (total - wins) > 0 else 0.8
                sample_size = total
            
            # Adjust for regime
            regime_multiplier = {
                'BULL': 1.1 if direction == 'long' else 0.9,
                'BEAR': 0.9 if direction == 'long' else 1.1,
                'RANGE': 1.0,
                'VOLATILE': 0.8,
                'CHOPPY': 0.7
            }.get(regime, 1.0)
            
            # Calculate edge
            win_prob_adjusted = min(0.95, win_prob * regime_multiplier)
            edge = (win_prob_adjusted * avg_win) - ((1 - win_prob_adjusted) * avg_loss)
            
            # Confidence based on sample size
            conf = min(0.95, sample_size / 100 + 0.3)
            
            return EdgeScore(
                edge=edge,
                confidence=conf,
                win_probability=win_prob_adjusted,
                avg_win=avg_win,
                avg_loss=avg_loss,
                sample_size=sample_size
            )
            
        except Exception as e:
            logger.error(f"Edge estimation failed: {e}")
            return EdgeScore(
                edge=0.1,
                confidence=0.3,
                win_probability=0.5,
                avg_win=1.0,
                avg_loss=0.8,
                sample_size=0
            )
    
    async def record_outcome(self, symbol: str, entry_edge: float, won: bool, pnl_percent: float = 0.0):
        """Record trade outcome for future edge calculations"""
        try:
            # Update symbol stats
            if symbol not in self._symbol_stats:
                self._symbol_stats[symbol] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_profit': 0.0,
                    'total_loss': 0.0
                }
            
            stats = self._symbol_stats[symbol]
            stats['total_trades'] += 1
            
            if won:
                stats['winning_trades'] += 1
                stats['total_profit'] += pnl_percent
            else:
                stats['total_loss'] += abs(pnl_percent)
            
            # Update global stats
            self._global_stats['total_trades'] += 1
            if won:
                self._global_stats['winning_trades'] += 1
                self._global_stats['total_profit'] += pnl_percent
            else:
                self._global_stats['total_loss'] += abs(pnl_percent)
                
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
    
    def get_calibration_data(self) -> Dict:
        """Get calibration data for persistence"""
        return {
            'global_stats': self._global_stats,
            'symbol_count': len(self._symbol_stats)
        }

    async def calculate_edge(self, symbol: str, direction: str = 'long',
                            current_price: float = 0.0, volatility: float = 0.0) -> EdgeData:
        """
        Calculate edge for a potential trade (alias for estimate_edge with different signature)
        
        Args:
            symbol: Trading pair
            direction: 'LONG' or 'SHORT' or 'long' or 'short'
            current_price: Current market price (optional)
            volatility: Current volatility measure (optional)
            
        Returns:
            EdgeData compatible object
        """
        # Convert direction format
        dir_lower = direction.lower() if direction else 'long'
        
        # Use estimate_edge internally
        edge_score = await self.estimate_edge(
            symbol=symbol,
            direction=dir_lower,
            confidence=0.5,
            regime='RANGE'
        )
        
        # Return an object with the expected attributes
        # Calculate risk/reward from avg_win and avg_loss
        rr_ratio = edge_score.avg_win / edge_score.avg_loss if edge_score.avg_loss > 0 else 1.0
        
        return EdgeData(
            edge=edge_score.edge,
            confidence=edge_score.confidence,
            reasons=[f"Win prob: {edge_score.win_probability:.1%}", f"Sample: {edge_score.sample_size}"],
            warnings=[] if edge_score.edge > 0 else ["Negative edge"],
            win_probability=edge_score.win_probability,
            risk_reward_ratio=rr_ratio,
            symbol=symbol
        )

    async def shutdown(self):
        """Cleanup on shutdown"""
        try:
            if self.redis_client:
                import json
                await self.redis_client.set('edge:calibration', json.dumps(self._global_stats))
                logger.info("Edge Estimator state saved")
        except Exception as e:
            logger.error(f"Failed to save edge estimator state: {e}")


# Singleton instance
edge_estimator = EdgeEstimator()

