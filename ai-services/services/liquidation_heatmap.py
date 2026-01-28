"""
Liquidation Heatmap - Tracks and visualizes liquidation levels

Liquidation data helps identify:
- Support/Resistance clusters (where many positions will get liquidated)
- Potential reversal zones (after big liquidation cascades)
- Whale hunting zones (where MM might push price)

Key insights:
- Large liq clusters above = magnet for shorts (they'll try to hunt)
- Large liq clusters below = magnet for longs (they'll try to hunt)
- After big liquidation cascade = often reversal opportunity
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from loguru import logger

import redis.asyncio as redis
from config import settings


@dataclass
class LiquidationLevel:
    """Represents a price level with aggregated liquidation data"""
    price: float
    total_value: float        # Total $ value of liquidations at this level
    long_value: float         # Longs that would be liquidated
    short_value: float        # Shorts that would be liquidated
    estimated_positions: int  # Number of positions
    intensity: str            # 'low', 'medium', 'high', 'extreme'


@dataclass
class LiquidationEvent:
    """Single liquidation event"""
    symbol: str
    side: str                 # 'Buy' (short liq) or 'Sell' (long liq)
    price: float
    qty: float
    value: float              # USD value
    timestamp: datetime


class LiquidationHeatmap:
    """
    Tracks liquidation data and creates heatmap visualization
    """
    
    def __init__(self):
        self.redis_client = None
        self.recent_liquidations: Dict[str, List[LiquidationEvent]] = defaultdict(list)
        self.heatmap_cache: Dict[str, Dict] = {}
        
        # Intensity thresholds (in $)
        self.LOW_THRESHOLD = 100_000
        self.MEDIUM_THRESHOLD = 500_000
        self.HIGH_THRESHOLD = 2_000_000
        self.EXTREME_THRESHOLD = 10_000_000
        
        # Price bucket size (% of price)
        self.BUCKET_PERCENT = 0.25  # 0.25% buckets
        
        self._initialized = False
        
    async def initialize(self):
        """Initialize the tracker"""
        try:
            self.redis_client = await redis.from_url(settings.REDIS_URL)
            await self._load_cache()
            self._initialized = True
            logger.info("Liquidation Heatmap initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Liquidation Heatmap: {e}")
            
    async def shutdown(self):
        """Cleanup"""
        await self._save_cache()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def _load_cache(self):
        """Load cached liquidation data from Redis"""
        try:
            data = await self.redis_client.get('liquidation:cache')
            if data:
                cache = json.loads(data)
                self.heatmap_cache = cache.get('heatmap', {})
                logger.info(f"Loaded liquidation cache: {len(self.heatmap_cache)} symbols")
        except Exception as e:
            logger.debug(f"Could not load liquidation cache: {e}")
            
    async def _save_cache(self):
        """Save liquidation cache to Redis"""
        try:
            cache = {
                'heatmap': self.heatmap_cache,
                'updated': datetime.utcnow().isoformat()
            }
            await self.redis_client.set('liquidation:cache', json.dumps(cache), ex=3600)
        except Exception as e:
            logger.debug(f"Could not save liquidation cache: {e}")
    
    async def fetch_liquidation_data(self, symbol: str, client) -> Dict:
        """
        Fetch and analyze liquidation data for a symbol
        
        Uses multiple data sources:
        1. Estimate liquidation levels based on open interest + leverage
        2. Track recent large liquidation events
        """
        try:
            # Get current ticker for price reference
            ticker_result = await client.get_tickers(symbol=symbol, category="linear")
            if not ticker_result.get('success'):
                return {}
                
            ticker_list = ticker_result.get('data', {}).get('list', [])
            if not ticker_list:
                return {}
                
            ticker = ticker_list[0]
            current_price = float(ticker.get('lastPrice', 0))
            
            # Get instrument info for leverage
            instrument_result = await client.get_instruments_info(symbol=symbol, category="linear")
            max_leverage = 100  # Default
            if instrument_result.get('success'):
                instruments = instrument_result.get('data', {}).get('list', [])
                if instruments:
                    max_leverage = float(instruments[0].get('leverageFilter', {}).get('maxLeverage', 100))
            
            # Get open interest
            oi_result = await client.get_open_interest(symbol=symbol, interval="5min", limit=1)
            current_oi = 0
            if oi_result.get('success'):
                oi_list = oi_result.get('data', {}).get('list', [])
                if oi_list:
                    current_oi = float(oi_list[0].get('openInterest', 0))
            
            # Get long/short ratio for distribution
            ls_result = await client.get_long_short_ratio(symbol=symbol, period="5min", limit=1)
            long_ratio = 0.5
            if ls_result.get('success'):
                ls_list = ls_result.get('data', {}).get('list', [])
                if ls_list:
                    buy_ratio = float(ls_list[0].get('buyRatio', 0.5))
                    long_ratio = buy_ratio
            
            # Estimate liquidation levels
            heatmap = self._estimate_liquidation_levels(
                current_price=current_price,
                open_interest=current_oi,
                long_ratio=long_ratio,
                max_leverage=max_leverage
            )
            
            # Store results
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'open_interest': current_oi,
                'open_interest_usd': current_oi * current_price,
                'long_ratio': long_ratio,
                'short_ratio': 1 - long_ratio,
                'heatmap': heatmap,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache it
            self.heatmap_cache[symbol] = result
            await self._store_heatmap(symbol, result)
            
            return result
            
        except Exception as e:
            logger.debug(f"Liquidation fetch error for {symbol}: {e}")
            return {}
    
    def _estimate_liquidation_levels(
        self,
        current_price: float,
        open_interest: float,
        long_ratio: float,
        max_leverage: float
    ) -> Dict:
        """
        Estimate where liquidations would occur based on OI distribution
        
        Assumptions:
        - Leverage distribution: mostly 10-25x, some higher
        - Entry prices distributed around current price
        """
        if current_price <= 0 or open_interest <= 0:
            return {'levels': [], 'summary': {}}
        
        # Common leverage tiers
        leverage_distribution = [
            (5, 0.05),    # 5% at 5x
            (10, 0.25),   # 25% at 10x
            (20, 0.35),   # 35% at 20x
            (50, 0.20),   # 20% at 50x
            (100, 0.15),  # 15% at 100x
        ]
        
        # Estimate OI in USD
        oi_usd = open_interest * current_price
        long_oi = oi_usd * long_ratio
        short_oi = oi_usd * (1 - long_ratio)
        
        levels = []
        
        # Calculate liquidation levels for each leverage tier
        for leverage, weight in leverage_distribution:
            # Liquidation price for longs (price drops)
            # Liq price = Entry * (1 - 1/leverage - maintenance_margin)
            # Simplified: Liq price ≈ Entry * (1 - 0.9/leverage)
            liq_distance_pct = 0.9 / leverage
            
            # Long liquidations (below current price)
            long_liq_price = current_price * (1 - liq_distance_pct)
            long_value_at_level = long_oi * weight
            
            levels.append({
                'price': round(long_liq_price, 2),
                'type': 'long_liquidation',
                'value': round(long_value_at_level, 0),
                'leverage': leverage,
                'distance_pct': round(-liq_distance_pct * 100, 2),
                'intensity': self._get_intensity(long_value_at_level)
            })
            
            # Short liquidations (above current price)
            short_liq_price = current_price * (1 + liq_distance_pct)
            short_value_at_level = short_oi * weight
            
            levels.append({
                'price': round(short_liq_price, 2),
                'type': 'short_liquidation',
                'value': round(short_value_at_level, 0),
                'leverage': leverage,
                'distance_pct': round(liq_distance_pct * 100, 2),
                'intensity': self._get_intensity(short_value_at_level)
            })
        
        # Sort by price
        levels.sort(key=lambda x: x['price'])
        
        # Calculate summary
        total_long_liqs = sum(l['value'] for l in levels if l['type'] == 'long_liquidation')
        total_short_liqs = sum(l['value'] for l in levels if l['type'] == 'short_liquidation')
        
        # Find biggest cluster
        biggest_long = max((l for l in levels if l['type'] == 'long_liquidation'), 
                          key=lambda x: x['value'], default=None)
        biggest_short = max((l for l in levels if l['type'] == 'short_liquidation'), 
                           key=lambda x: x['value'], default=None)
        
        summary = {
            'total_long_liquidations_usd': round(total_long_liqs, 0),
            'total_short_liquidations_usd': round(total_short_liqs, 0),
            'liq_imbalance': 'more_longs' if total_long_liqs > total_short_liqs * 1.2 else 
                           ('more_shorts' if total_short_liqs > total_long_liqs * 1.2 else 'balanced'),
            'nearest_long_liq': biggest_long['price'] if biggest_long else None,
            'nearest_short_liq': biggest_short['price'] if biggest_short else None,
            'risk_assessment': self._assess_liquidation_risk(levels, current_price)
        }
        
        return {
            'levels': levels,
            'summary': summary
        }
    
    def _get_intensity(self, value: float) -> str:
        """Get intensity level based on USD value"""
        if value >= self.EXTREME_THRESHOLD:
            return 'extreme'
        elif value >= self.HIGH_THRESHOLD:
            return 'high'
        elif value >= self.MEDIUM_THRESHOLD:
            return 'medium'
        else:
            return 'low'
            
    def _assess_liquidation_risk(self, levels: List[Dict], current_price: float) -> str:
        """Assess overall liquidation risk"""
        if not levels:
            return 'unknown'
            
        # Check for nearby extreme levels
        for level in levels:
            distance = abs(level['price'] - current_price) / current_price
            if level['intensity'] == 'extreme' and distance < 0.02:
                return 'extreme_risk_nearby'
            if level['intensity'] == 'high' and distance < 0.01:
                return 'high_risk_nearby'
        
        return 'moderate'
        
    async def _store_heatmap(self, symbol: str, data: Dict):
        """Store heatmap in Redis"""
        try:
            await self.redis_client.hset('liquidation:heatmap', symbol, json.dumps(data))
            await self.redis_client.expire('liquidation:heatmap', 900)  # 15 min
        except Exception as e:
            logger.debug(f"Failed to store heatmap: {e}")
            
    async def get_heatmap(self, symbol: str) -> Optional[Dict]:
        """Get cached heatmap for a symbol"""
        # Check memory cache first
        if symbol in self.heatmap_cache:
            return self.heatmap_cache[symbol]
            
        # Try Redis
        try:
            data = await self.redis_client.hget('liquidation:heatmap', symbol)
            if data:
                return json.loads(data)
        except:
            pass
            
        return None
        
    async def get_liquidation_zones(self, symbol: str, client) -> Dict:
        """
        Get key liquidation zones for trading decisions
        
        Returns simplified zones for entry/exit decisions
        """
        heatmap = await self.fetch_liquidation_data(symbol, client)
        
        if not heatmap or 'heatmap' not in heatmap:
            return {'zones': [], 'recommendation': 'no_data'}
            
        levels = heatmap['heatmap'].get('levels', [])
        current_price = heatmap.get('current_price', 0)
        
        if not levels or current_price <= 0:
            return {'zones': [], 'recommendation': 'no_data'}
        
        # Find key zones
        zones = []
        
        # Resistance zones (short liquidations - price magnets above)
        short_liqs = [l for l in levels if l['type'] == 'short_liquidation' and l['intensity'] in ['high', 'extreme']]
        for liq in short_liqs:
            zones.append({
                'type': 'resistance_magnet',
                'price': liq['price'],
                'value': liq['value'],
                'reason': f"${liq['value']:,.0f} in shorts liquidated here"
            })
            
        # Support zones (long liquidations - price magnets below)
        long_liqs = [l for l in levels if l['type'] == 'long_liquidation' and l['intensity'] in ['high', 'extreme']]
        for liq in long_liqs:
            zones.append({
                'type': 'support_magnet',
                'price': liq['price'],
                'value': liq['value'],
                'reason': f"${liq['value']:,.0f} in longs liquidated here"
            })
        
        # Generate recommendation
        summary = heatmap['heatmap'].get('summary', {})
        imbalance = summary.get('liq_imbalance', 'balanced')
        
        if imbalance == 'more_longs':
            recommendation = "Caution on longs - more long liquidations clustered below"
        elif imbalance == 'more_shorts':
            recommendation = "Caution on shorts - more short liquidations clustered above"
        else:
            recommendation = "Balanced liquidation levels"
            
        return {
            'zones': zones,
            'imbalance': imbalance,
            'recommendation': recommendation,
            'current_price': current_price
        }


# Singleton instance
_liquidation_heatmap: Optional[LiquidationHeatmap] = None

async def get_liquidation_heatmap() -> LiquidationHeatmap:
    """Get or create the liquidation heatmap singleton"""
    global _liquidation_heatmap
    if _liquidation_heatmap is None:
        _liquidation_heatmap = LiquidationHeatmap()
        await _liquidation_heatmap.initialize()
    return _liquidation_heatmap
