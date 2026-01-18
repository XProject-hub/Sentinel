"""
SENTINEL AI - Unified Capital Allocator
Bot decides: Crypto vs TradFi, which asset, how much

KEY CONCEPT:
- ONE total budget
- Bot allocates dynamically based on:
  - Edge score for each opportunity
  - Market regime
  - Correlation between assets
  - Risk limits

NO MORE FIXED %:
- Old: "50% crypto, 50% tradfi"
- New: "100% goes where edge is highest"

BYBIT TRADFI PRODUCTS:
- BTCUSD, ETHUSD (inverse perpetuals)
- Gold (XAUUSD)
- S&P 500 Index (US500)
- Nasdaq 100 (NAS100)
- Oil (OILUSD)
- Forex pairs (limited on Bybit)
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import numpy as np
from loguru import logger
import redis.asyncio as redis

from config import settings


@dataclass
class AllocationTarget:
    """Recommended capital allocation"""
    symbol: str
    asset_class: str  # 'crypto' or 'tradfi'
    allocation_percent: float  # 0-100% of available capital
    position_size_usdt: float
    edge_score: float
    confidence: float
    reason: str


@dataclass
class MarketOpportunity:
    """Trading opportunity for allocation"""
    symbol: str
    asset_class: str
    edge_score: float
    confidence: float
    direction: str  # 'long' or 'short'
    volatility: float
    liquidity: float
    correlation_btc: float  # Correlation with BTC


class CapitalAllocator:
    """
    Dynamic Capital Allocation System
    
    BOT DECIDES:
    - WHERE to allocate (crypto vs tradfi)
    - WHICH assets
    - HOW MUCH per position
    
    Based on:
    - Edge scores (higher edge = more capital)
    - Kelly criterion (optimal sizing)
    - Correlation (diversification)
    - Risk limits (max per position, max total exposure)
    """
    
    # Bybit TradFi symbols
    TRADFI_SYMBOLS = [
        # Indices
        'US500USDT',    # S&P 500
        'NAS100USDT',   # Nasdaq 100
        # Commodities
        'XAUUSDT',      # Gold
        'XAGUSDT',      # Silver
        # Crypto derivatives (inverse)
        'BTCUSD',       # BTC Inverse
        'ETHUSD',       # ETH Inverse
    ]
    
    # Major crypto for comparison
    MAJOR_CRYPTO = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    
    # Allocation limits
    MAX_SINGLE_POSITION = 0.20  # Max 20% in single position
    MAX_ASSET_CLASS = 0.80  # Max 80% in single asset class
    MIN_POSITION_SIZE = 5  # Min $5 per position
    KELLY_FRACTION = 0.25  # Use 25% of Kelly (conservative)
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # Current allocations
        self.current_allocations: Dict[str, float] = {}  # symbol -> % allocated
        self.total_capital = 0.0
        self.available_capital = 0.0
        
        # Opportunity tracking
        self.opportunities: List[MarketOpportunity] = []
        
        # Stats
        self.stats = {
            'allocations_made': 0,
            'crypto_allocation_avg': 0.0,
            'tradfi_allocation_avg': 0.0,
            'best_performing_class': 'none',
            'total_edge_captured': 0.0
        }
        
        # Performance tracking
        self.class_performance: Dict[str, List[float]] = {
            'crypto': [],
            'tradfi': []
        }
        
    async def initialize(self):
        """Initialize capital allocator"""
        logger.info("Initializing Unified Capital Allocator...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.is_running = True
        
        await self._load_state()
        
        logger.info("Capital Allocator ready - Bot decides all allocations")
        
    async def shutdown(self):
        """Cleanup"""
        await self._save_state()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def update_capital(self, total_capital: float, used_capital: float):
        """Update available capital"""
        self.total_capital = total_capital
        self.available_capital = total_capital - used_capital
        
    async def add_opportunity(self, opp: MarketOpportunity):
        """Add trading opportunity for consideration"""
        # Remove old opportunity for same symbol
        self.opportunities = [o for o in self.opportunities if o.symbol != opp.symbol]
        self.opportunities.append(opp)
        
        # Keep top 50 opportunities
        self.opportunities.sort(key=lambda x: x.edge_score, reverse=True)
        self.opportunities = self.opportunities[:50]
        
    async def get_allocation_recommendation(self, 
                                            opportunities: List[MarketOpportunity],
                                            available_capital: float) -> List[AllocationTarget]:
        """
        Get recommended capital allocations
        
        Bot decides:
        - Which opportunities to take
        - How much to allocate to each
        - Respects risk limits and diversification
        """
        if not opportunities or available_capital < self.MIN_POSITION_SIZE:
            return []
            
        self.stats['allocations_made'] += 1
        
        # Sort by edge score
        sorted_opps = sorted(opportunities, key=lambda x: x.edge_score, reverse=True)
        
        allocations = []
        remaining_capital = available_capital
        crypto_allocated = 0.0
        tradfi_allocated = 0.0
        
        for opp in sorted_opps:
            if remaining_capital < self.MIN_POSITION_SIZE:
                break
                
            # Skip low edge
            if opp.edge_score < 0.1:
                continue
                
            # Calculate Kelly-based position size
            kelly_pct = self._calculate_kelly_size(
                edge=opp.edge_score,
                confidence=opp.confidence,
                volatility=opp.volatility
            )
            
            # Apply limits
            max_pct = self.MAX_SINGLE_POSITION
            
            # Check asset class limit
            if opp.asset_class == 'crypto':
                remaining_class_limit = self.MAX_ASSET_CLASS - (crypto_allocated / self.total_capital)
            else:
                remaining_class_limit = self.MAX_ASSET_CLASS - (tradfi_allocated / self.total_capital)
                
            max_pct = min(max_pct, remaining_class_limit)
            
            # Final allocation
            alloc_pct = min(kelly_pct, max_pct)
            position_size = remaining_capital * alloc_pct
            
            if position_size >= self.MIN_POSITION_SIZE:
                allocations.append(AllocationTarget(
                    symbol=opp.symbol,
                    asset_class=opp.asset_class,
                    allocation_percent=round(alloc_pct * 100, 2),
                    position_size_usdt=round(position_size, 2),
                    edge_score=opp.edge_score,
                    confidence=opp.confidence,
                    reason=f"Edge {opp.edge_score:.2f}, Kelly {kelly_pct:.1%}"
                ))
                
                remaining_capital -= position_size
                
                if opp.asset_class == 'crypto':
                    crypto_allocated += position_size
                else:
                    tradfi_allocated += position_size
                    
        # Update stats
        if self.total_capital > 0:
            self.stats['crypto_allocation_avg'] = crypto_allocated / self.total_capital * 100
            self.stats['tradfi_allocation_avg'] = tradfi_allocated / self.total_capital * 100
            
        return allocations
        
    def _calculate_kelly_size(self, edge: float, confidence: float, volatility: float) -> float:
        """
        Calculate position size using Kelly Criterion
        
        Kelly% = (p * b - q) / b
        Where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio
        """
        # Estimate win probability from confidence
        p = min(0.7, 0.5 + (confidence / 100) * 0.2)  # Cap at 70%
        q = 1 - p
        
        # Estimate win/loss ratio from edge
        b = 1 + edge  # e.g., edge 0.3 = 1.3:1 ratio
        
        # Kelly formula
        kelly = (p * b - q) / b
        
        # Apply fraction (conservative)
        kelly_adjusted = kelly * self.KELLY_FRACTION
        
        # Adjust for volatility (reduce size for high volatility)
        volatility_factor = 1 / (1 + volatility * 0.5)
        
        final_size = kelly_adjusted * volatility_factor
        
        # Cap at max single position
        return min(self.MAX_SINGLE_POSITION, max(0, final_size))
        
    async def get_best_asset_class(self) -> str:
        """Determine which asset class is currently performing better"""
        crypto_perf = self.class_performance['crypto']
        tradfi_perf = self.class_performance['tradfi']
        
        if not crypto_perf and not tradfi_perf:
            return 'both'  # No data yet
            
        crypto_avg = np.mean(crypto_perf[-20:]) if crypto_perf else 0
        tradfi_avg = np.mean(tradfi_perf[-20:]) if tradfi_perf else 0
        
        if crypto_avg > tradfi_avg * 1.2:  # 20% better
            return 'crypto'
        elif tradfi_avg > crypto_avg * 1.2:
            return 'tradfi'
        else:
            return 'both'
            
    async def record_trade_result(self, symbol: str, asset_class: str, pnl_percent: float):
        """Record trade result for performance tracking"""
        if asset_class in self.class_performance:
            self.class_performance[asset_class].append(pnl_percent)
            self.class_performance[asset_class] = self.class_performance[asset_class][-100:]
            
        self.stats['total_edge_captured'] += pnl_percent
        self.stats['best_performing_class'] = await self.get_best_asset_class()
        
    async def get_tradfi_opportunities(self) -> List[Dict]:
        """Get current TradFi opportunities from Bybit"""
        # This would be populated by market scanner
        # For now, return structure
        return [
            {'symbol': s, 'asset_class': 'tradfi', 'available': True}
            for s in self.TRADFI_SYMBOLS
        ]
        
    def classify_symbol(self, symbol: str) -> str:
        """Classify symbol as crypto or tradfi"""
        if symbol in self.TRADFI_SYMBOLS:
            return 'tradfi'
        return 'crypto'
        
    async def get_allocation_status(self) -> Dict:
        """Get current allocation status"""
        crypto_total = sum(
            v for k, v in self.current_allocations.items() 
            if self.classify_symbol(k) == 'crypto'
        )
        tradfi_total = sum(
            v for k, v in self.current_allocations.items() 
            if self.classify_symbol(k) == 'tradfi'
        )
        
        return {
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'crypto_allocated': crypto_total,
            'tradfi_allocated': tradfi_total,
            'crypto_pct': round(crypto_total / max(1, self.total_capital) * 100, 1),
            'tradfi_pct': round(tradfi_total / max(1, self.total_capital) * 100, 1),
            'positions_count': len(self.current_allocations),
            'best_performing': self.stats['best_performing_class'],
            'opportunities_available': len(self.opportunities)
        }
        
    async def _load_state(self):
        """Load state from Redis"""
        try:
            data = await self.redis_client.get('capital_allocator:state')
            if data:
                state = json.loads(data)
                self.current_allocations = state.get('allocations', {})
                self.class_performance = state.get('performance', {'crypto': [], 'tradfi': []})
        except:
            pass
            
    async def _save_state(self):
        """Save state to Redis"""
        try:
            state = {
                'allocations': self.current_allocations,
                'performance': self.class_performance
            }
            await self.redis_client.set(
                'capital_allocator:state',
                json.dumps(state),
                ex=86400  # 24h
            )
        except:
            pass
            
    async def get_stats(self) -> Dict:
        """Get allocator statistics"""
        return {
            'allocations_made': self.stats['allocations_made'],
            'crypto_allocation_avg': round(self.stats['crypto_allocation_avg'], 1),
            'tradfi_allocation_avg': round(self.stats['tradfi_allocation_avg'], 1),
            'best_performing_class': self.stats['best_performing_class'],
            'total_edge_captured': round(self.stats['total_edge_captured'], 2),
            'total_capital': self.total_capital,
            'kelly_fraction': self.KELLY_FRACTION,
            'max_single_position': self.MAX_SINGLE_POSITION * 100
        }


# Global instance
capital_allocator = CapitalAllocator()

