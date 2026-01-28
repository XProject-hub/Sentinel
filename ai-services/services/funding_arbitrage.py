"""
Funding Rate Arbitrage - Earn from funding without directional exposure

Strategy:
1. Find coins with high funding rates (either positive or negative)
2. Open opposite positions on spot + perpetual (delta neutral)
3. Collect funding payments every 8 hours

Example:
- BTC funding = +0.1% (longs pay shorts)
- Go SHORT on perp, LONG on spot (1:1)
- Collect 0.1% every 8h = 0.3% daily = ~10% monthly!

Risk-free yield when executed correctly.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

import redis.asyncio as redis
from config import settings


@dataclass
class FundingOpportunity:
    """A funding arbitrage opportunity"""
    symbol: str
    funding_rate: float           # Current funding rate (e.g., 0.0015 = 0.15%)
    funding_rate_annualized: float  # APY estimate
    direction: str                # 'short_perp' or 'long_perp' (which perp position to take)
    spot_action: str              # 'buy_spot' or 'sell_spot'
    estimated_daily_yield: float  # % daily
    estimated_monthly_yield: float
    next_funding_time: datetime
    hours_until_funding: float
    risk_level: str               # 'low', 'medium', 'high'
    recommendation: str
    timestamp: datetime


class FundingArbitrage:
    """
    Detects and tracks funding rate arbitrage opportunities
    """
    
    def __init__(self):
        self.redis_client = None
        self.opportunities: Dict[str, FundingOpportunity] = {}
        self.funding_history: Dict[str, List[Dict]] = {}
        
        # Thresholds
        self.MIN_FUNDING_RATE = 0.0005   # 0.05% minimum to consider
        self.HIGH_FUNDING_RATE = 0.001   # 0.1% = good opportunity
        self.EXTREME_FUNDING_RATE = 0.003  # 0.3% = excellent
        
        # Risk parameters
        self.MAX_HISTORICAL_VOLATILITY = 0.02  # Max 2% funding swings
        
        self._initialized = False
        
    async def initialize(self):
        """Initialize the tracker"""
        try:
            self.redis_client = await redis.from_url(settings.REDIS_URL)
            await self._load_cache()
            self._initialized = True
            logger.info("Funding Arbitrage initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Funding Arbitrage: {e}")
            
    async def shutdown(self):
        """Cleanup"""
        await self._save_cache()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def _load_cache(self):
        """Load cached data from Redis"""
        try:
            data = await self.redis_client.get('funding:arbitrage:cache')
            if data:
                cache = json.loads(data)
                self.funding_history = cache.get('history', {})
                logger.info(f"Loaded funding cache: {len(self.funding_history)} symbols")
        except Exception as e:
            logger.debug(f"Could not load funding cache: {e}")
            
    async def _save_cache(self):
        """Save cache to Redis"""
        try:
            cache = {
                'history': self.funding_history,
                'updated': datetime.utcnow().isoformat()
            }
            await self.redis_client.set('funding:arbitrage:cache', json.dumps(cache), ex=3600)
        except Exception as e:
            logger.debug(f"Could not save funding cache: {e}")
    
    async def scan_opportunities(self, client, symbols: List[str] = None) -> List[FundingOpportunity]:
        """
        Scan for funding arbitrage opportunities
        
        Args:
            client: BybitV5Client instance
            symbols: List of symbols to scan (default: top coins)
            
        Returns:
            List of opportunities sorted by yield
        """
        if not symbols:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
                'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
                'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'NEARUSDT', 'APTUSDT',
                'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'SEIUSDT', 'TIAUSDT'
            ]
        
        opportunities = []
        
        for symbol in symbols:
            try:
                opp = await self._analyze_symbol(symbol, client)
                if opp:
                    opportunities.append(opp)
                    self.opportunities[symbol] = opp
            except Exception as e:
                logger.debug(f"Funding scan error for {symbol}: {e}")
        
        # Sort by estimated daily yield (absolute value, descending)
        opportunities.sort(key=lambda x: abs(x.estimated_daily_yield), reverse=True)
        
        # Store top opportunities in Redis
        await self._store_opportunities(opportunities)
        
        return opportunities
    
    async def _analyze_symbol(self, symbol: str, client) -> Optional[FundingOpportunity]:
        """Analyze a single symbol for funding opportunity"""
        try:
            # Get current ticker (includes funding info)
            ticker_result = await client.get_tickers(symbol=symbol, category="linear")
            if not ticker_result.get('success'):
                return None
                
            ticker_list = ticker_result.get('data', {}).get('list', [])
            if not ticker_list:
                return None
                
            ticker = ticker_list[0]
            
            # Get funding rate info
            funding_rate = float(ticker.get('fundingRate', 0))
            next_funding_time_str = ticker.get('nextFundingTime', '')
            
            # Skip if funding rate too low
            if abs(funding_rate) < self.MIN_FUNDING_RATE:
                return None
            
            # Parse next funding time
            try:
                if next_funding_time_str:
                    next_funding_time = datetime.fromtimestamp(int(next_funding_time_str) / 1000)
                else:
                    # Default: assume next funding in 8 hours
                    next_funding_time = datetime.utcnow() + timedelta(hours=8)
            except:
                next_funding_time = datetime.utcnow() + timedelta(hours=8)
            
            hours_until = max(0, (next_funding_time - datetime.utcnow()).total_seconds() / 3600)
            
            # Get funding history for volatility check
            history_result = await client.get_funding_rate_history(symbol=symbol, limit=24)
            funding_history = []
            if history_result.get('success'):
                history_list = history_result.get('data', {}).get('list', [])
                for item in history_list:
                    funding_history.append(float(item.get('fundingRate', 0)))
            
            # Store history
            self.funding_history[symbol] = funding_history
            
            # Calculate volatility (std dev of recent funding rates)
            funding_volatility = 0
            if len(funding_history) > 3:
                import statistics
                funding_volatility = statistics.stdev(funding_history)
            
            # Determine strategy
            if funding_rate > 0:
                # Positive funding = longs pay shorts
                # Strategy: SHORT perp + LONG spot
                direction = 'short_perp'
                spot_action = 'buy_spot'
            else:
                # Negative funding = shorts pay longs
                # Strategy: LONG perp + SHORT spot (or use margin)
                direction = 'long_perp'
                spot_action = 'sell_spot'
            
            # Calculate yields
            daily_yield = abs(funding_rate) * 3  # 3 funding periods per day
            monthly_yield = daily_yield * 30
            annualized = daily_yield * 365
            
            # Risk assessment
            risk_level = self._assess_risk(abs(funding_rate), funding_volatility, funding_history)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                symbol, funding_rate, daily_yield, risk_level, hours_until
            )
            
            return FundingOpportunity(
                symbol=symbol,
                funding_rate=funding_rate,
                funding_rate_annualized=annualized,
                direction=direction,
                spot_action=spot_action,
                estimated_daily_yield=round(daily_yield * 100, 4),  # As percentage
                estimated_monthly_yield=round(monthly_yield * 100, 2),
                next_funding_time=next_funding_time,
                hours_until_funding=round(hours_until, 2),
                risk_level=risk_level,
                recommendation=recommendation,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.debug(f"Funding analysis error for {symbol}: {e}")
            return None
    
    def _assess_risk(self, funding_rate: float, volatility: float, history: List[float]) -> str:
        """Assess risk level of the opportunity"""
        # Check for sign flips in history (rate changed direction)
        sign_flips = 0
        if len(history) > 1:
            for i in range(1, len(history)):
                if (history[i] > 0) != (history[i-1] > 0):
                    sign_flips += 1
        
        # High volatility = high risk
        if volatility > self.MAX_HISTORICAL_VOLATILITY:
            return 'high'
            
        # Many sign flips = medium risk
        if sign_flips >= 3:
            return 'medium'
            
        # Extremely high funding could flip quickly
        if funding_rate > self.EXTREME_FUNDING_RATE:
            return 'medium'
            
        return 'low'
    
    def _generate_recommendation(
        self, 
        symbol: str, 
        funding_rate: float,
        daily_yield: float,
        risk_level: str,
        hours_until: float
    ) -> str:
        """Generate human-readable recommendation"""
        rate_pct = abs(funding_rate) * 100
        daily_pct = daily_yield * 100
        
        if risk_level == 'high':
            return f"⚠️ High risk - funding volatile. {rate_pct:.3f}% rate but unstable history."
        
        if abs(funding_rate) >= self.EXTREME_FUNDING_RATE:
            direction = "SHORT perp + LONG spot" if funding_rate > 0 else "LONG perp + SHORT spot"
            return f"🔥 EXCELLENT: {rate_pct:.3f}% funding ({daily_pct:.2f}%/day). {direction}. Next funding in {hours_until:.1f}h"
        
        if abs(funding_rate) >= self.HIGH_FUNDING_RATE:
            direction = "SHORT perp" if funding_rate > 0 else "LONG perp"
            return f"✅ GOOD: {rate_pct:.3f}% funding ({daily_pct:.2f}%/day). {direction} + hedge spot."
        
        return f"💡 Moderate: {rate_pct:.3f}% funding. Consider if capital available."
    
    async def _store_opportunities(self, opportunities: List[FundingOpportunity]):
        """Store opportunities in Redis"""
        try:
            data = []
            for opp in opportunities[:20]:  # Top 20
                data.append({
                    'symbol': opp.symbol,
                    'funding_rate': opp.funding_rate,
                    'funding_rate_pct': round(opp.funding_rate * 100, 4),
                    'annualized_pct': round(opp.funding_rate_annualized * 100, 2),
                    'direction': opp.direction,
                    'spot_action': opp.spot_action,
                    'daily_yield_pct': opp.estimated_daily_yield,
                    'monthly_yield_pct': opp.estimated_monthly_yield,
                    'next_funding': opp.next_funding_time.isoformat(),
                    'hours_until': opp.hours_until_funding,
                    'risk_level': opp.risk_level,
                    'recommendation': opp.recommendation,
                    'timestamp': opp.timestamp.isoformat()
                })
            
            await self.redis_client.set('funding:opportunities', json.dumps(data), ex=900)
            
        except Exception as e:
            logger.debug(f"Failed to store opportunities: {e}")
    
    async def get_opportunities(self, limit: int = 10) -> List[Dict]:
        """Get current opportunities from cache"""
        try:
            data = await self.redis_client.get('funding:opportunities')
            if data:
                opportunities = json.loads(data)
                return opportunities[:limit]
        except:
            pass
        
        return []
    
    async def get_best_opportunity(self) -> Optional[Dict]:
        """Get the single best funding opportunity right now"""
        opportunities = await self.get_opportunities(limit=5)
        
        # Filter for low risk only
        low_risk = [o for o in opportunities if o.get('risk_level') == 'low']
        
        if low_risk:
            return low_risk[0]
        elif opportunities:
            return opportunities[0]
        
        return None
    
    def calculate_position_size(
        self,
        capital: float,
        funding_rate: float,
        leverage: float = 1.0
    ) -> Dict:
        """
        Calculate optimal position size for funding arb
        
        Args:
            capital: Available capital in USD
            funding_rate: Current funding rate
            leverage: Leverage to use (1.0 = no leverage, delta neutral)
            
        Returns:
            Position sizing details
        """
        # For delta neutral, split capital equally
        # Half goes to spot, half to perp margin
        spot_allocation = capital / 2
        perp_margin = capital / 2
        
        # With leverage, perp position can be larger
        perp_position_size = perp_margin * leverage
        
        # But spot should match perp for delta neutral
        # So we're limited by spot allocation
        actual_position = min(spot_allocation, perp_position_size)
        
        # Calculate expected return
        funding_per_period = actual_position * abs(funding_rate)
        funding_per_day = funding_per_period * 3
        funding_per_month = funding_per_day * 30
        
        return {
            'capital_required': capital,
            'spot_position': actual_position,
            'perp_position': actual_position,
            'perp_margin_required': actual_position / leverage,
            'funding_per_8h': round(funding_per_period, 2),
            'funding_per_day': round(funding_per_day, 2),
            'funding_per_month': round(funding_per_month, 2),
            'daily_return_pct': round((funding_per_day / capital) * 100, 4),
            'monthly_return_pct': round((funding_per_month / capital) * 100, 2),
            'note': 'Delta neutral - no directional exposure'
        }


# Singleton instance
_funding_arbitrage: Optional[FundingArbitrage] = None

async def get_funding_arbitrage() -> FundingArbitrage:
    """Get or create the funding arbitrage singleton"""
    global _funding_arbitrage
    if _funding_arbitrage is None:
        _funding_arbitrage = FundingArbitrage()
        await _funding_arbitrage.initialize()
    return _funding_arbitrage
