"""
SENTINEL AI - Market Scanner
Scans ALL available trading pairs and ranks opportunities

This is the "eyes" of the system - sees everything, trades selectively.

Key features:
- Scans 500+ Bybit pairs
- Calculates edge for each
- Ranks by opportunity quality
- Returns tradeable opportunities only
- Updates every 30 seconds

Unlike basic bots that trade everything, this FILTERS.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger
import redis.asyncio as redis
import json
import httpx

from config import settings
from services.regime_detector import RegimeDetector, RegimeState
from services.edge_estimator import EdgeEstimator, EdgeScore


@dataclass
class TradingOpportunity:
    """A ranked trading opportunity"""
    symbol: str
    direction: str  # 'long' or 'short'
    
    # Scores
    edge_score: float  # -1 to +1
    confidence: float  # 0-100
    opportunity_score: float  # Combined ranking score
    
    # Edge components
    edge_data: Optional[EdgeScore] = None
    
    # Regime
    regime: str = 'unknown'
    regime_action: str = 'avoid'
    
    # Price data
    current_price: float = 0.0
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    volume_24h: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    liquidity_score: float = 0.0
    
    # Recommendation
    should_trade: bool = False
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Asset type
    is_tradfi: bool = False  # True for stocks, forex, commodities, metals
    asset_category: str = 'crypto'  # 'crypto', 'stocks', 'forex', 'metals', 'commodities', 'indices'
    
    # Timing
    timestamp: str = ""


class MarketScanner:
    """
    Comprehensive Market Scanner
    
    Scans all Bybit pairs and identifies opportunities:
    1. Fetches all available trading pairs
    2. Gets regime for each
    3. Calculates edge for promising ones
    4. Ranks by opportunity quality
    5. Returns list of tradeable opportunities
    
    This is what makes Sentinel different from dumb bots.
    """
    
    def __init__(self):
        self.redis_client = None
        self.http_client = None
        
        # Components
        self.regime_detector: Optional[RegimeDetector] = None
        self.edge_estimator: Optional[EdgeEstimator] = None
        
        # Cached symbol list
        self.all_symbols: List[str] = []
        self.tradfi_symbols: List[str] = []  # TradFi symbols (stocks, forex, metals, commodities)
        self.symbol_info: Dict[str, Dict] = {}
        self.symbol_category: Dict[str, str] = {}  # Maps symbol to category
        
        # Scan results cache
        self.last_scan_time: datetime = None
        self.last_opportunities: List[TradingOpportunity] = []
        
        # Blacklisted symbols (had issues)
        self.blacklist: Set[str] = set()
        
        # Performance tracking
        self.symbol_performance: Dict[str, Dict] = {}
        
        # Scan settings
        self.scan_interval = 30  # seconds
        self.max_parallel_scans = 50  # Concurrent API calls
        self.min_volume_24h = 10000  # $10k minimum volume (lowered to include more pairs)
        self.min_liquidity = 30  # Minimum liquidity score (lowered)
        
    async def initialize(self, regime_detector: RegimeDetector, 
                         edge_estimator: EdgeEstimator):
        """Initialize scanner with required components"""
        logger.info("Initializing Market Scanner (500+ pairs)...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        self.regime_detector = regime_detector
        self.edge_estimator = edge_estimator
        
        # Load crypto symbols
        await self._load_all_symbols()
        
        # Load TradFi symbols (stocks, forex, metals, commodities)
        await self._load_tradfi_symbols()
        
        # Load blacklist
        await self._load_blacklist()
        
        # Load performance data
        await self._load_performance()
        
        # Start background scanner
        asyncio.create_task(self._continuous_scan_loop())
        
        logger.info(f"Market Scanner initialized - Tracking {len(self.all_symbols)} pairs")
        
    async def shutdown(self):
        """Cleanup"""
        await self._save_blacklist()
        await self._save_performance()
        
        if self.http_client:
            await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def _continuous_scan_loop(self):
        """Continuous market scanning"""
        logger.info("Starting continuous market scan loop...")
        
        while True:
            try:
                await self.scan_market()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                await asyncio.sleep(60)
                
    async def scan_market(self) -> List[TradingOpportunity]:
        """
        Main scan function - scans all pairs
        
        Returns list of opportunities sorted by quality
        """
        scan_start = datetime.utcnow()
        opportunities: List[TradingOpportunity] = []
        
        logger.info(f"Scanning {len(self.all_symbols)} pairs...")
        
        # Step 1: Quick filter - get tickers for volume/price filter
        tickers = await self._get_all_tickers()
        
        # Step 2: Filter to viable symbols
        viable_symbols = self._filter_viable_symbols(tickers)
        logger.info(f"Viable symbols after volume filter: {len(viable_symbols)}")
        
        # Step 3: Get regime for all viable symbols (batch)
        regimes = await self._batch_detect_regimes(viable_symbols)
        
        # Step 4: Filter by regime - skip 'avoid' regimes
        tradeable_symbols = [
            s for s in viable_symbols
            if regimes.get(s, {}).get('action', 'avoid') != 'avoid'
        ]
        logger.info(f"Tradeable after regime filter: {len(tradeable_symbols)}")
        
        # Step 5: Calculate edge for tradeable symbols
        for symbol in tradeable_symbols:
            try:
                ticker = tickers.get(symbol, {})
                regime = regimes.get(symbol, {})
                
                # Calculate edge for both directions
                long_edge = await self.edge_estimator.calculate_edge(symbol, 'long')
                short_edge = await self.edge_estimator.calculate_edge(symbol, 'short')
                
                # Pick better direction
                if long_edge.edge > short_edge.edge:
                    direction = 'long'
                    edge_data = long_edge
                else:
                    direction = 'short'
                    edge_data = short_edge
                    
                # Skip if no edge
                if edge_data.edge <= 0:
                    continue
                    
                # Calculate opportunity score (for ranking)
                opp_score = self._calculate_opportunity_score(
                    edge_data, regime, ticker
                )
                
                # Determine if should trade
                should_trade = (
                    edge_data.edge > 0.15 and
                    edge_data.confidence > 50 and
                    edge_data.recommended_size != 'skip' and
                    regime.get('action', 'avoid') != 'avoid'
                )
                
                # Determine if TradFi
                is_tradfi = symbol in self.tradfi_symbols
                asset_category = self.symbol_category.get(symbol, 'crypto')
                
                opportunity = TradingOpportunity(
                    symbol=symbol,
                    direction=direction,
                    edge_score=edge_data.edge,
                    confidence=edge_data.confidence,
                    opportunity_score=opp_score,
                    edge_data=edge_data,
                    regime=regime.get('regime', 'unknown'),
                    regime_action=regime.get('action', 'avoid'),
                    current_price=float(ticker.get('lastPrice', 0)),
                    price_change_1h=float(ticker.get('price1hPcnt', 0)) * 100,
                    price_change_24h=float(ticker.get('price24hPcnt', 0)) * 100,
                    volume_24h=float(ticker.get('volume24h', 0)),
                    volatility=float(regime.get('volatility', 0)),
                    liquidity_score=float(regime.get('liquidity', 50)),
                    should_trade=should_trade,
                    reasons=edge_data.reasons,
                    warnings=edge_data.warnings,
                    is_tradfi=is_tradfi,
                    asset_category=asset_category,
                    timestamp=datetime.utcnow().isoformat()
                )
                
                opportunities.append(opportunity)
                
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
                
        # Sort by opportunity score (highest first)
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        # Cache results
        self.last_scan_time = scan_start
        self.last_opportunities = opportunities
        
        # Store in Redis
        await self._store_scan_results(opportunities)
        
        # Log summary
        tradeable = [o for o in opportunities if o.should_trade]
        logger.info(
            f"Scan complete: {len(opportunities)} with edge, "
            f"{len(tradeable)} tradeable opportunities"
        )
        
        return opportunities
        
    async def get_top_opportunities(self, limit: int = 20) -> List[TradingOpportunity]:
        """Get top N opportunities from last scan"""
        # Use cached results if recent
        if (self.last_scan_time and 
            datetime.utcnow() - self.last_scan_time < timedelta(seconds=self.scan_interval)):
            return self.last_opportunities[:limit]
            
        # Otherwise trigger new scan
        opportunities = await self.scan_market()
        return opportunities[:limit]
        
    async def get_tradeable_opportunities(self) -> List[TradingOpportunity]:
        """Get only opportunities that should be traded"""
        opps = await self.get_top_opportunities(limit=100)
        return [o for o in opps if o.should_trade]
        
    def _calculate_opportunity_score(self, edge: EdgeScore, 
                                      regime: Dict, ticker: Dict) -> float:
        """
        Calculate composite opportunity score
        
        Higher score = better opportunity
        """
        score = 0.0
        
        # Edge contribution (40%)
        score += edge.edge * 40
        
        # Confidence contribution (20%)
        score += (edge.confidence / 100) * 20
        
        # Win probability contribution (15%)
        score += ((edge.win_probability - 50) / 50) * 15
        
        # Risk/Reward contribution (10%)
        rr_score = min(edge.risk_reward_ratio, 3) / 3
        score += rr_score * 10
        
        # Volume contribution (10%) - higher volume = more reliable
        volume = float(ticker.get('volume24h', 0))
        if volume > 10000000:  # $10M+
            score += 10
        elif volume > 1000000:  # $1M+
            score += 7
        elif volume > 100000:  # $100k+
            score += 4
            
        # Regime contribution (5%)
        regime_scores = {
            'high_liquidity_trend': 5,
            'accumulation': 3,
            'range_bound': 2,
            'distribution': 0,
            'high_volatility': -2,
            'low_liquidity': -5,
            'news_event': -5,
            'unknown': -3
        }
        score += regime_scores.get(regime.get('regime', 'unknown'), 0)
        
        # Historical performance bonus/penalty
        perf = self.symbol_performance.get(edge.symbol, {})
        if perf.get('total_trades', 0) > 5:
            win_rate = perf.get('wins', 0) / perf.get('total_trades', 1)
            if win_rate > 0.6:
                score += 5
            elif win_rate < 0.4:
                score -= 5
                
        return max(0, score)
        
    def _filter_viable_symbols(self, tickers: Dict[str, Dict]) -> List[str]:
        """Filter symbols by volume, price, etc."""
        viable = []
        
        for symbol, ticker in tickers.items():
            # Skip blacklisted
            if symbol in self.blacklist:
                continue
                
            # Check if it's a known symbol (crypto or tradfi)
            is_crypto = symbol.endswith('USDT')
            is_tradfi = symbol in self.tradfi_symbols or symbol in self.symbol_category
            
            if not is_crypto and not is_tradfi:
                continue
                
            # Check volume (lower threshold for TradFi)
            volume = float(ticker.get('volume24h', 0))
            min_vol = self.min_volume_24h if is_crypto else 1000  # Lower for TradFi
            if volume < min_vol:
                continue
                
            # Check price exists
            price = float(ticker.get('lastPrice', 0))
            if price <= 0:
                continue
                
            # Skip stablecoins
            if symbol in ['USDCUSDT', 'DAIUSDT', 'TUSDUSDT', 'BUSDUSDT']:
                continue
                
            viable.append(symbol)
            
        # Also add TradFi symbols that might not be in tickers
        for symbol in self.tradfi_symbols:
            if symbol not in viable and symbol not in self.blacklist:
                viable.append(symbol)
            
        return viable
        
    async def _batch_detect_regimes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Detect regimes for multiple symbols in batches"""
        regimes = {}
        
        # Process in batches
        batch_size = self.max_parallel_scans
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Run regime detection concurrently
            tasks = [
                self.regime_detector.detect_regime(symbol)
                for symbol in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, results):
                if isinstance(result, Exception):
                    regimes[symbol] = {
                        'regime': 'unknown',
                        'action': 'avoid',
                        'volatility': 0,
                        'liquidity': 0
                    }
                else:
                    regimes[symbol] = {
                        'regime': result.regime,
                        'action': result.recommended_action,
                        'volatility': result.volatility,
                        'liquidity': result.liquidity_score
                    }
                    
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.5)
                
        return regimes
        
    async def _load_all_symbols(self):
        """Load all available trading pairs from Bybit"""
        try:
            url = "https://api.bybit.com/v5/market/instruments-info"
            params = {'category': 'linear', 'limit': 1000}
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to load symbols: {response.status_code}")
                return
                
            data = response.json()
            instruments = data.get('result', {}).get('list', [])
            
            self.all_symbols = []
            self.symbol_info = {}
            
            for inst in instruments:
                symbol = inst.get('symbol', '')
                if symbol.endswith('USDT'):
                    self.all_symbols.append(symbol)
                    self.symbol_info[symbol] = {
                        'min_qty': float(inst.get('lotSizeFilter', {}).get('minOrderQty', 0.001)),
                        'qty_step': float(inst.get('lotSizeFilter', {}).get('qtyStep', 0.001)),
                        'tick_size': float(inst.get('priceFilter', {}).get('tickSize', 0.01)),
                        'min_notional': float(inst.get('lotSizeFilter', {}).get('minNotionalValue', 5))
                    }
                    
            logger.info(f"Loaded {len(self.all_symbols)} crypto trading pairs")
            
            # Mark all as crypto
            for symbol in self.all_symbols:
                self.symbol_category[symbol] = 'crypto'
            
            # Store in Redis
            await self.redis_client.set(
                'trading:available_symbols',
                ','.join(self.all_symbols)
            )
            
        except Exception as e:
            logger.error(f"Error loading crypto symbols: {e}")
            # Fallback to major pairs
            self.all_symbols = [
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
                'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LINKUSDT'
            ]
            for symbol in self.all_symbols:
                self.symbol_category[symbol] = 'crypto'
    
    async def _load_tradfi_symbols(self):
        """Load TradFi symbols from Bybit (stocks, forex, metals, commodities)"""
        try:
            # Bybit TradFi uses MT5 - try to fetch from their API
            # Note: This might use a different endpoint
            
            # Try the linear category first with different suffixes
            url = "https://api.bybit.com/v5/market/instruments-info"
            
            # TradFi symbols typically have suffixes like:
            # Metals: XAUUSD, XAGUSD
            # Stocks: suffix varies
            # Forex: EURUSD, etc.
            
            # First, try to get all tickers and filter for TradFi-like symbols
            tickers_url = "https://api.bybit.com/v5/market/tickers"
            
            # Try different categories for TradFi
            tradfi_categories = ['linear', 'spot']
            
            for category in tradfi_categories:
                try:
                    response = await self.http_client.get(url, params={'category': category, 'limit': 1000})
                    
                    if response.status_code == 200:
                        data = response.json()
                        instruments = data.get('result', {}).get('list', [])
                        
                        for inst in instruments:
                            symbol = inst.get('symbol', '')
                            
                            # Identify TradFi symbols by their patterns
                            # Metals: XAU (gold), XAG (silver), XPT (platinum)
                            # Forex: EUR, GBP, JPY, AUD pairs without USDT
                            # Stocks: Usually 3-4 letter codes
                            
                            is_tradfi = False
                            asset_category = 'crypto'
                            
                            # Check for metals
                            if any(metal in symbol.upper() for metal in ['XAU', 'XAG', 'XPT', 'XPD']):
                                is_tradfi = True
                                asset_category = 'metals'
                            # Check for forex
                            elif any(fx in symbol.upper() for fx in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']):
                                is_tradfi = True
                                asset_category = 'forex'
                            # Check for commodities
                            elif any(comm in symbol.upper() for comm in ['OIL', 'GAS', 'COCOA', 'COFFEE', 'COTTON', 'COPPER', 'SUGAR']):
                                is_tradfi = True
                                asset_category = 'commodities'
                            # Check for indices
                            elif any(idx in symbol.upper() for idx in ['SPX', 'NDX', 'DJI', 'QQQ']):
                                is_tradfi = True
                                asset_category = 'indices'
                            # Check for stocks (common stock symbols on Bybit)
                            elif symbol.upper() in ['TSLA', 'AAPL', 'GOOG', 'GOOGL', 'META', 'AMZN', 'MSFT', 'NVDA', 
                                                     'MSTR', 'RIOT', 'MARA', 'COIN', 'AMD', 'INTC', 'NFLX',
                                                     'DIS', 'BA', 'JPM', 'GS', 'V', 'MA', 'PYPL']:
                                is_tradfi = True
                                asset_category = 'stocks'
                            
                            if is_tradfi and symbol not in self.tradfi_symbols:
                                self.tradfi_symbols.append(symbol)
                                self.symbol_category[symbol] = asset_category
                                self.symbol_info[symbol] = {
                                    'min_qty': float(inst.get('lotSizeFilter', {}).get('minOrderQty', 0.01)),
                                    'qty_step': float(inst.get('lotSizeFilter', {}).get('qtyStep', 0.01)),
                                    'tick_size': float(inst.get('priceFilter', {}).get('tickSize', 0.01)),
                                    'min_notional': float(inst.get('lotSizeFilter', {}).get('minNotionalValue', 5)),
                                    'is_tradfi': True
                                }
                                
                except Exception as e:
                    logger.debug(f"Error fetching {category} category for TradFi: {e}")
                    
            # Also add known TradFi symbols from Bybit MT5
            # These are the symbols shown in the user's screenshots
            known_tradfi = {
                # Metals
                'XAUUSD+': 'metals', 'XAGUSD': 'metals', 'XPTUSD': 'metals',
                'XAUAUD': 'metals', 'XAUEUR': 'metals', 'XAUJPY': 'metals',
                # Forex
                'EURUSD+': 'forex', 'USOUSD': 'forex', 'GBPUSD': 'forex',
                'USDJPY': 'forex', 'AUDUSD': 'forex',
                # Stocks
                'TSLA': 'stocks', 'AAPL': 'stocks', 'GOOG': 'stocks', 
                'META': 'stocks', 'NVDA': 'stocks', 'MSTR': 'stocks',
                'CRCL': 'stocks', 'RIOT': 'stocks', 'HUT': 'stocks',
                'DFDV': 'stocks', 'APP': 'stocks', 'QQQ': 'indices',
                'TQQQ': 'indices',
                # Commodities  
                'COCOA-C': 'commodities', 'COFFEE-C': 'commodities',
                'COPPER-C': 'commodities', 'COTTON-C': 'commodities',
                'GAS-C': 'commodities', 'GASOIL-C': 'commodities',
            }
            
            for symbol, category in known_tradfi.items():
                if symbol not in self.tradfi_symbols:
                    self.tradfi_symbols.append(symbol)
                    self.symbol_category[symbol] = category
                    if symbol not in self.symbol_info:
                        self.symbol_info[symbol] = {
                            'min_qty': 0.01,
                            'qty_step': 0.01,
                            'tick_size': 0.01,
                            'min_notional': 5,
                            'is_tradfi': True
                        }
            
            logger.info(f"Loaded {len(self.tradfi_symbols)} TradFi symbols (metals, stocks, forex, commodities)")
            
            # Store in Redis
            await self.redis_client.set(
                'trading:tradfi_symbols',
                ','.join(self.tradfi_symbols)
            )
            
            # Store combined list
            all_combined = self.all_symbols + self.tradfi_symbols
            await self.redis_client.set(
                'trading:all_symbols',
                ','.join(all_combined)
            )
            await self.redis_client.set('trading:total_pairs', str(len(all_combined)))
            
        except Exception as e:
            logger.error(f"Error loading TradFi symbols: {e}")
            # TradFi is optional, continue without it
            
    async def _get_all_tickers(self) -> Dict[str, Dict]:
        """Get tickers for all symbols"""
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {'category': 'linear'}
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code != 200:
                return {}
                
            data = response.json()
            tickers = data.get('result', {}).get('list', [])
            
            return {t['symbol']: t for t in tickers}
            
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            return {}
            
    async def _store_scan_results(self, opportunities: List[TradingOpportunity]):
        """Store scan results in Redis"""
        try:
            # Store summary
            summary = {
                'scan_time': datetime.utcnow().isoformat(),
                'total_pairs': len(self.all_symbols),
                'with_edge': len(opportunities),
                'tradeable': len([o for o in opportunities if o.should_trade]),
                'top_opportunities': [
                    {
                        'symbol': o.symbol,
                        'direction': o.direction,
                        'edge': o.edge_score,
                        'confidence': o.confidence,
                        'score': o.opportunity_score,
                        'should_trade': o.should_trade
                    }
                    for o in opportunities[:20]
                ]
            }
            
            await self.redis_client.setex(
                'scanner:results',
                120,  # 2 minute TTL
                json.dumps(summary)
            )
            
        except Exception as e:
            logger.debug(f"Error storing scan results: {e}")
            
    async def blacklist_symbol(self, symbol: str, reason: str):
        """Add symbol to blacklist"""
        self.blacklist.add(symbol)
        await self._save_blacklist()
        logger.warning(f"Blacklisted {symbol}: {reason}")
        
    async def _load_blacklist(self):
        """Load blacklist from Redis"""
        try:
            data = await self.redis_client.get('scanner:blacklist')
            if data:
                self.blacklist = set(json.loads(data))
        except:
            pass
            
    async def _save_blacklist(self):
        """Save blacklist to Redis"""
        try:
            await self.redis_client.set('scanner:blacklist', json.dumps(list(self.blacklist)))
        except:
            pass
            
    async def record_trade_result(self, symbol: str, won: bool, pnl: float):
        """Record trade result for symbol performance tracking"""
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {
                'total_trades': 0,
                'wins': 0,
                'total_pnl': 0.0
            }
            
        self.symbol_performance[symbol]['total_trades'] += 1
        if won:
            self.symbol_performance[symbol]['wins'] += 1
        self.symbol_performance[symbol]['total_pnl'] += pnl
        
        await self._save_performance()
        
    async def _load_performance(self):
        """Load performance data from Redis"""
        try:
            data = await self.redis_client.get('scanner:performance')
            if data:
                self.symbol_performance = json.loads(data)
        except:
            pass
            
    async def _save_performance(self):
        """Save performance data to Redis"""
        try:
            await self.redis_client.set(
                'scanner:performance', 
                json.dumps(self.symbol_performance)
            )
        except:
            pass
            
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get trading info for a symbol"""
        return self.symbol_info.get(symbol, {
            'min_qty': 0.001,
            'qty_step': 0.001,
            'tick_size': 0.01,
            'min_notional': 5
        })
        
    async def get_stats(self) -> Dict:
        """Get scanner statistics"""
        # Count by category
        category_counts = {}
        for symbol, category in self.symbol_category.items():
            category_counts[category] = category_counts.get(category, 0) + 1
            
        return {
            'total_pairs_loaded': len(self.all_symbols) + len(self.tradfi_symbols),
            'crypto_pairs': len(self.all_symbols),
            'tradfi_pairs': len(self.tradfi_symbols),
            'pairs_with_info': len(self.symbol_info),
            'blacklisted': len(self.blacklist),
            'min_volume_24h': self.min_volume_24h,
            'scan_interval': self.scan_interval,
            'category_breakdown': category_counts,
            'all_symbols': self.all_symbols + self.tradfi_symbols,  # Return ALL symbols
            'crypto_symbols': self.all_symbols,
            'tradfi_symbols': self.tradfi_symbols,
            'last_scan': (await self.redis_client.get('scanner:results')) if self.redis_client else None
        }
        
    def get_all_pairs(self) -> List[str]:
        """Get all loaded trading pairs"""
        return self.all_symbols.copy()


# Global instance
market_scanner = MarketScanner()

