"""
Training Data Manager - Manages data for ML model training

Collects, filters, and prepares trade data for training AI models
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class TrainingDataManager:
    """
    Manages training data for ML models
    
    Features:
    - Collects trade outcomes
    - Filters quality data
    - Prepares batches for training
    - Deduplicates similar trades
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self._data_buffer: List[Dict] = []
        self._max_buffer_size = 1000
        
    async def initialize(self, redis_client=None):
        """Initialize with Redis client"""
        if redis_client:
            self.redis_client = redis_client
        logger.info("Training Data Manager initialized")
        
    async def process_trade(self, trade_data: Dict, user_id: str = "default"):
        """
        Process a completed trade for training
        
        Args:
            trade_data: Trade outcome data
            user_id: User identifier
        """
        try:
            # Add metadata
            trade_data['user_id'] = user_id
            trade_data['processed_at'] = datetime.utcnow().isoformat()
            
            # Add to buffer
            self._data_buffer.append(trade_data)
            
            # Trim buffer if too large
            if len(self._data_buffer) > self._max_buffer_size:
                self._data_buffer = self._data_buffer[-self._max_buffer_size:]
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    key = f"training:trades:{user_id}"
                    await self.redis_client.lpush(key, json.dumps(trade_data))
                    await self.redis_client.ltrim(key, 0, 999)  # Keep last 1000
                except Exception as e:
                    logger.debug(f"Redis store failed: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to process trade: {e}")
    
    async def get_training_batch(self, batch_size: int = 100, 
                                 min_quality: float = 0.5) -> List[Dict]:
        """
        Get a batch of quality trades for training
        
        Args:
            batch_size: Number of trades to return
            min_quality: Minimum quality score (0-1)
            
        Returns:
            List of trade data dictionaries
        """
        try:
            # Filter by quality
            quality_trades = [
                t for t in self._data_buffer
                if self._calculate_quality(t) >= min_quality
            ]
            
            # Return most recent
            return quality_trades[-batch_size:]
            
        except Exception as e:
            logger.error(f"Failed to get training batch: {e}")
            return []
    
    def _calculate_quality(self, trade: Dict) -> float:
        """Calculate quality score for a trade (0-1)"""
        try:
            quality = 0.5  # Base quality
            
            # Higher quality if we have all required fields
            required_fields = ['symbol', 'direction', 'pnl_percent', 'entry_price', 'exit_price']
            for field in required_fields:
                if field in trade:
                    quality += 0.1
            
            # Higher quality for trades with good confidence
            if 'confidence' in trade and trade['confidence'] > 60:
                quality += 0.1
            
            # Lower quality for very short trades (might be errors)
            duration = trade.get('duration_seconds', 0)
            if duration < 60:
                quality -= 0.2
            elif duration > 300:
                quality += 0.1
            
            return min(1.0, max(0.0, quality))
            
        except:
            return 0.5
    
    def get_stats(self) -> Dict:
        """Get training data statistics"""
        return {
            'buffer_size': len(self._data_buffer),
            'max_buffer_size': self._max_buffer_size
        }


# Singleton instance
training_data_manager = TrainingDataManager()

