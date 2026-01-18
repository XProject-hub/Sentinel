"""
SENTINEL AI - Advanced Regime Detection
Detects market regimes using HMM-inspired approach + XGBoost

Regimes:
- HIGH_LIQUIDITY_TREND: Best for momentum/scalping
- LOW_LIQUIDITY: AVOID or reduce size
- HIGH_VOLATILITY: Wide stops, reduced size
- RANGE_BOUND: Mean reversion
- NEWS_EVENT: Pause or hedge
- ACCUMULATION: Spot opportunities
- DISTRIBUTION: Exit longs

This is THE MOST IMPORTANT component - wrong regime = losses
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import redis.asyncio as redis
import json
import httpx

from config import settings


@dataclass
class RegimeState:
    """Current market regime for a symbol"""
    symbol: str
    regime: str
    confidence: float  # 0-100
    stability: float  # How stable is this regime (0-100)
    duration_minutes: int  # How long in this regime
    volatility: float
    liquidity_score: float
    trend_strength: float
    trend_direction: str  # 'up', 'down', 'neutral'
    volume_profile: str  # 'increasing', 'decreasing', 'stable'
    recommended_action: str  # 'aggressive', 'normal', 'reduced', 'avoid'
    timestamp: str


class RegimeDetector:
    """
    Advanced Market Regime Detection
    
    Uses multiple signals to determine current market state:
    - Volatility clustering
    - Volume profile
    - Trend strength
    - Liquidity indicators
    - Cross-asset correlation
    
    This determines HOW the bot should trade, not WHAT.
    """
    
    REGIMES = [
        'high_liquidity_trend',
        'low_liquidity',
        'high_volatility',
        'range_bound',
        'news_event',
        'accumulation',
        'distribution',
        'unknown'
    ]
    
    def __init__(self):
        self.redis_client = None
        self.http_client = None
        
        # Regime history for stability calculation
        self.regime_history: Dict[str, List[str]] = {}
        
        # Volatility baselines
        self.volatility_baselines: Dict[str, float] = {}
        
        # Volume baselines
        self.volume_baselines: Dict[str, float] = {}
        
        # BTC/ETH as market leaders
        self.market_leaders = ['BTCUSDT', 'ETHUSDT']
        
        # Global market regime
        self.global_regime = 'unknown'
        self.global_confidence = 50.0
        
    async def initialize(self):
        """Initialize regime detector"""
        logger.info("Initializing Regime Detector (HMM-inspired)...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load saved baselines
        await self._load_baselines()
        
        # Calculate initial baselines for major pairs
        await self._calculate_initial_baselines()
        
        logger.info("Regime Detector initialized - Market state awareness active")
        
    async def shutdown(self):
        """Cleanup"""
        await self._save_baselines()
        if self.http_client:
            await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def detect_regime(self, symbol: str) -> RegimeState:
        """
        Detect current market regime for a symbol
        
        This is the CORE function that determines trading behavior
        """
        try:
            # Get market data
            data = await self._fetch_market_data(symbol)
            if not data:
                return self._unknown_regime(symbol)
                
            # Calculate regime indicators
            volatility = self._calculate_volatility(data)
            liquidity = self._calculate_liquidity_score(data)
            trend_strength, trend_direction = self._calculate_trend(data)
            volume_profile = self._analyze_volume_profile(data)
            
            # Get global market context
            global_context = await self._get_global_context()
            
            # Determine regime based on indicators
            regime, confidence = self._classify_regime(
                volatility=volatility,
                liquidity=liquidity,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                volume_profile=volume_profile,
                global_context=global_context,
                symbol=symbol
            )
            
            # Calculate regime stability
            stability = self._calculate_stability(symbol, regime)
            
            # Get duration in current regime
            duration = await self._get_regime_duration(symbol, regime)
            
            # Determine recommended action
            action = self._get_recommended_action(regime, confidence, stability, volatility)
            
            state = RegimeState(
                symbol=symbol,
                regime=regime,
                confidence=confidence,
                stability=stability,
                duration_minutes=duration,
                volatility=volatility,
                liquidity_score=liquidity,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                volume_profile=volume_profile,
                recommended_action=action,
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Store regime
            await self._store_regime(state)
            
            # Update history
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
            self.regime_history[symbol].append(regime)
            self.regime_history[symbol] = self.regime_history[symbol][-20:]  # Keep last 20
            
            return state
            
        except Exception as e:
            logger.error(f"Regime detection error for {symbol}: {e}")
            return self._unknown_regime(symbol)
            
    def _classify_regime(self, volatility: float, liquidity: float,
                         trend_strength: float, trend_direction: str,
                         volume_profile: str, global_context: Dict,
                         symbol: str) -> Tuple[str, float]:
        """
        Classify market regime based on indicators
        
        This is the "HMM-inspired" classification logic
        """
        
        # Get baselines
        vol_baseline = self.volatility_baselines.get(symbol, 2.0)
        
        # Normalized volatility (1.0 = normal)
        vol_ratio = volatility / vol_baseline if vol_baseline > 0 else 1.0
        
        scores = {regime: 0.0 for regime in self.REGIMES}
        
        # === HIGH LIQUIDITY TREND ===
        # Best for trading - clear direction with volume
        if liquidity > 70 and trend_strength > 0.5:
            scores['high_liquidity_trend'] += 40
            if volume_profile == 'increasing':
                scores['high_liquidity_trend'] += 20
            if vol_ratio < 1.5:  # Not too volatile
                scores['high_liquidity_trend'] += 15
                
        # === LOW LIQUIDITY ===
        # Dangerous - wide spreads, slippage
        if liquidity < 40:
            scores['low_liquidity'] += 50
            if volume_profile == 'decreasing':
                scores['low_liquidity'] += 20
                
        # === HIGH VOLATILITY ===
        # Can be profitable but risky
        if vol_ratio > 2.0:
            scores['high_volatility'] += 40
            if vol_ratio > 3.0:
                scores['high_volatility'] += 30
                
        # === RANGE BOUND ===
        # Good for mean reversion
        if trend_strength < 0.3 and vol_ratio < 1.2:
            scores['range_bound'] += 35
            if liquidity > 50:
                scores['range_bound'] += 20
                
        # === NEWS EVENT ===
        # Detected by sudden volume + volatility spike
        if vol_ratio > 2.5 and volume_profile == 'increasing':
            if global_context.get('news_detected', False):
                scores['news_event'] += 60
            else:
                scores['news_event'] += 30
                
        # === ACCUMULATION ===
        # Price stable, volume increasing, often before breakout
        if trend_strength < 0.2 and volume_profile == 'increasing':
            if trend_direction == 'up':
                scores['accumulation'] += 40
                
        # === DISTRIBUTION ===
        # Price stable/up but volume decreasing - smart money exiting
        if trend_direction == 'up' and volume_profile == 'decreasing':
            if trend_strength < 0.3:
                scores['distribution'] += 40
                
        # Apply global market context adjustments
        if global_context.get('global_regime') == 'high_volatility':
            scores['high_volatility'] += 15
        if global_context.get('fear_greed', 50) < 25:
            scores['high_volatility'] += 10
            scores['distribution'] += 10
        if global_context.get('fear_greed', 50) > 75:
            scores['accumulation'] -= 10
            scores['distribution'] += 15
            
        # Find winning regime
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]
        
        # Calculate confidence (normalized)
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = (best_score / total_score) * 100
        else:
            confidence = 50.0
            best_regime = 'unknown'
            
        # Minimum confidence threshold
        if confidence < 40:
            best_regime = 'unknown'
            
        return best_regime, min(95, confidence)
        
    def _calculate_volatility(self, data: Dict) -> float:
        """Calculate current volatility (ATR-based)"""
        closes = data.get('closes', [])
        highs = data.get('highs', [])
        lows = data.get('lows', [])
        
        if len(closes) < 14:
            return 2.0
            
        # True Range
        tr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
            
        # ATR as percentage
        atr = np.mean(tr_values[-14:])
        current_price = closes[-1]
        
        volatility_pct = (atr / current_price) * 100 if current_price > 0 else 2.0
        
        return volatility_pct
        
    def _calculate_liquidity_score(self, data: Dict) -> float:
        """
        Calculate liquidity score (0-100)
        
        Based on:
        - Volume relative to average
        - Spread (if available)
        - Volume consistency
        """
        volumes = data.get('volumes', [])
        
        if len(volumes) < 20:
            return 50.0
            
        # Volume ratio (current vs average)
        avg_volume = np.mean(volumes[-20:])
        recent_volume = np.mean(volumes[-5:])
        
        if avg_volume <= 0:
            return 50.0
            
        volume_ratio = recent_volume / avg_volume
        
        # Volume consistency (lower std = more consistent = better liquidity)
        volume_std = np.std(volumes[-20:]) / avg_volume if avg_volume > 0 else 1.0
        consistency_score = max(0, 100 - volume_std * 50)
        
        # Combine scores
        if volume_ratio > 1.5:
            volume_score = 80
        elif volume_ratio > 1.0:
            volume_score = 60
        elif volume_ratio > 0.5:
            volume_score = 40
        else:
            volume_score = 20
            
        liquidity = (volume_score * 0.6 + consistency_score * 0.4)
        
        return min(100, max(0, liquidity))
        
    def _calculate_trend(self, data: Dict) -> Tuple[float, str]:
        """
        Calculate trend strength and direction
        
        Strength: 0-1 (0 = no trend, 1 = strong trend)
        Direction: 'up', 'down', 'neutral'
        """
        closes = data.get('closes', [])
        
        if len(closes) < 50:
            return 0.0, 'neutral'
            
        # Calculate multiple SMAs
        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        current = closes[-1]
        
        # Trend alignment score
        alignment = 0
        
        if current > sma_10 > sma_20 > sma_50:
            alignment = 1.0  # Perfect uptrend
            direction = 'up'
        elif current < sma_10 < sma_20 < sma_50:
            alignment = 1.0  # Perfect downtrend
            direction = 'down'
        else:
            # Partial alignment
            up_signals = sum([
                current > sma_10,
                sma_10 > sma_20,
                sma_20 > sma_50
            ])
            alignment = abs(up_signals - 1.5) / 1.5  # 0 to 1
            direction = 'up' if up_signals >= 2 else ('down' if up_signals <= 1 else 'neutral')
            
        # ADX-like strength
        price_range = max(closes[-20:]) - min(closes[-20:])
        if price_range > 0:
            directional_move = abs(current - sma_20)
            strength = min(1.0, (directional_move / price_range) * alignment)
        else:
            strength = 0.0
            
        return strength, direction
        
    def _analyze_volume_profile(self, data: Dict) -> str:
        """Analyze volume trend"""
        volumes = data.get('volumes', [])
        
        if len(volumes) < 20:
            return 'stable'
            
        # Compare recent volume to older
        recent = np.mean(volumes[-5:])
        older = np.mean(volumes[-20:-5])
        
        if older <= 0:
            return 'stable'
            
        ratio = recent / older
        
        if ratio > 1.3:
            return 'increasing'
        elif ratio < 0.7:
            return 'decreasing'
        else:
            return 'stable'
            
    async def _get_global_context(self) -> Dict:
        """Get global market context from BTC/ETH"""
        try:
            # Get BTC regime as market leader
            btc_data = await self._fetch_market_data('BTCUSDT')
            
            context = {
                'global_regime': 'unknown',
                'btc_trend': 'neutral',
                'fear_greed': 50,
                'news_detected': False
            }
            
            if btc_data:
                vol = self._calculate_volatility(btc_data)
                trend_str, trend_dir = self._calculate_trend(btc_data)
                
                if vol > 3.0:
                    context['global_regime'] = 'high_volatility'
                elif trend_str > 0.5:
                    context['global_regime'] = 'trending'
                else:
                    context['global_regime'] = 'ranging'
                    
                context['btc_trend'] = trend_dir
                
            # Get Fear & Greed
            try:
                fg_data = await self.redis_client.get('data:fear_greed')
                if fg_data:
                    fg = json.loads(fg_data)
                    context['fear_greed'] = int(fg.get('value', 50))
            except:
                pass
                
            # Check for news events
            try:
                news_data = await self.redis_client.get('data:crypto_news')
                if news_data:
                    news = json.loads(news_data)
                    articles = news.get('articles', [])
                    # Check for high-impact news in last hour
                    recent_count = sum(1 for a in articles[:10] if 'breaking' in a.get('title', '').lower())
                    context['news_detected'] = recent_count > 2
            except:
                pass
                
            return context
            
        except Exception as e:
            logger.debug(f"Global context error: {e}")
            return {'global_regime': 'unknown', 'btc_trend': 'neutral', 'fear_greed': 50}
            
    def _calculate_stability(self, symbol: str, current_regime: str) -> float:
        """Calculate how stable the regime is (prevents flip-flopping)"""
        history = self.regime_history.get(symbol, [])
        
        if len(history) < 5:
            return 50.0
            
        # Count how many of last N were same regime
        same_count = sum(1 for r in history[-10:] if r == current_regime)
        stability = (same_count / min(10, len(history))) * 100
        
        return stability
        
    async def _get_regime_duration(self, symbol: str, regime: str) -> int:
        """Get how long we've been in this regime"""
        try:
            key = f"regime:duration:{symbol}"
            data = await self.redis_client.hgetall(key)
            
            if data:
                stored_regime = data.get(b'regime', b'').decode()
                started = data.get(b'started', b'').decode()
                
                if stored_regime == regime and started:
                    start_time = datetime.fromisoformat(started)
                    duration = (datetime.utcnow() - start_time).total_seconds() / 60
                    return int(duration)
                    
            # New regime - reset timer
            await self.redis_client.hset(key, mapping={
                'regime': regime,
                'started': datetime.utcnow().isoformat()
            })
            return 0
            
        except:
            return 0
            
    def _get_recommended_action(self, regime: str, confidence: float,
                                 stability: float, volatility: float) -> str:
        """
        Get recommended trading action based on regime
        
        This is CRITICAL for risk management
        """
        
        # Regime-based recommendations
        actions = {
            'high_liquidity_trend': 'aggressive',
            'low_liquidity': 'avoid',
            'high_volatility': 'reduced',
            'range_bound': 'normal',
            'news_event': 'avoid',
            'accumulation': 'normal',
            'distribution': 'reduced',
            'unknown': 'hold'  # Allow trading for unknown regime (new symbols)
        }
        
        base_action = actions.get(regime, 'hold')  # Default to hold, not avoid
        
        # Adjust based on confidence
        if confidence < 60:
            if base_action == 'aggressive':
                base_action = 'normal'
            elif base_action == 'normal':
                base_action = 'reduced'
                
        # Adjust based on stability
        if stability < 40:  # Regime is unstable
            if base_action in ['aggressive', 'normal']:
                base_action = 'reduced'
                
        # Extreme volatility override
        if volatility > 5.0:
            base_action = 'reduced' if base_action != 'avoid' else 'avoid'
            
        return base_action
        
    async def _fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch OHLCV data for regime detection"""
        try:
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': '15',  # 15-minute candles
                'limit': 100
            }
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            klines = data.get('result', {}).get('list', [])
            
            if len(klines) < 50:
                return None
                
            # Parse OHLCV
            opens, highs, lows, closes, volumes = [], [], [], [], []
            
            for k in reversed(klines):
                opens.append(float(k[1]))
                highs.append(float(k[2]))
                lows.append(float(k[3]))
                closes.append(float(k[4]))
                volumes.append(float(k[5]))
                
            return {
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'volumes': volumes
            }
            
        except Exception as e:
            logger.debug(f"Market data fetch error: {e}")
            return None
            
    def _unknown_regime(self, symbol: str) -> RegimeState:
        """Return unknown regime state"""
        return RegimeState(
            symbol=symbol,
            regime='unknown',
            confidence=0,
            stability=0,
            duration_minutes=0,
            volatility=0,
            liquidity_score=0,
            trend_strength=0,
            trend_direction='neutral',
            volume_profile='stable',
            recommended_action='hold',  # Allow trading when no regime data yet
            timestamp=datetime.utcnow().isoformat()
        )
        
    async def _store_regime(self, state: RegimeState):
        """Store regime state in Redis"""
        try:
            await self.redis_client.hset(
                f"regime:{state.symbol}",
                mapping={
                    'regime': state.regime,
                    'confidence': str(state.confidence),
                    'stability': str(state.stability),
                    'volatility': str(state.volatility),
                    'liquidity': str(state.liquidity_score),
                    'trend_strength': str(state.trend_strength),
                    'trend_direction': state.trend_direction,
                    'volume_profile': state.volume_profile,
                    'action': state.recommended_action,
                    'timestamp': state.timestamp
                }
            )
            
            # Also store global regime if this is BTC
            if state.symbol == 'BTCUSDT':
                self.global_regime = state.regime
                self.global_confidence = state.confidence
                await self.redis_client.hset('regime:global', mapping={
                    'regime': state.regime,
                    'confidence': str(state.confidence)
                })
                
        except Exception as e:
            logger.debug(f"Store regime error: {e}")
            
    async def _calculate_initial_baselines(self):
        """Calculate volatility/volume baselines for major pairs"""
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
        
        for symbol in major_pairs:
            try:
                data = await self._fetch_market_data(symbol)
                if data:
                    vol = self._calculate_volatility(data)
                    self.volatility_baselines[symbol] = vol
                    
                    avg_vol = np.mean(data.get('volumes', [1])[-20:])
                    self.volume_baselines[symbol] = avg_vol
                    
            except:
                pass
                
        logger.info(f"Calculated baselines for {len(self.volatility_baselines)} pairs")
        
    async def _load_baselines(self):
        """Load saved baselines from Redis"""
        try:
            data = await self.redis_client.get('regime:baselines')
            if data:
                baselines = json.loads(data)
                self.volatility_baselines = baselines.get('volatility', {})
                self.volume_baselines = baselines.get('volume', {})
        except:
            pass
            
    async def _save_baselines(self):
        """Save baselines to Redis"""
        try:
            await self.redis_client.set('regime:baselines', json.dumps({
                'volatility': self.volatility_baselines,
                'volume': self.volume_baselines
            }))
        except:
            pass
            
    async def get_all_regimes(self) -> Dict[str, RegimeState]:
        """Get regime for all tracked symbols"""
        regimes = {}
        
        try:
            keys = await self.redis_client.keys('regime:*')
            
            for key in keys:
                if b':duration:' in key or b':global' in key or b':baselines' in key:
                    continue
                    
                symbol = key.decode().split(':')[1]
                data = await self.redis_client.hgetall(key)
                
                if data:
                    regimes[symbol] = RegimeState(
                        symbol=symbol,
                        regime=data.get(b'regime', b'unknown').decode(),
                        confidence=float(data.get(b'confidence', 0)),
                        stability=float(data.get(b'stability', 0)),
                        duration_minutes=0,
                        volatility=float(data.get(b'volatility', 0)),
                        liquidity_score=float(data.get(b'liquidity', 0)),
                        trend_strength=float(data.get(b'trend_strength', 0)),
                        trend_direction=data.get(b'trend_direction', b'neutral').decode(),
                        volume_profile=data.get(b'volume_profile', b'stable').decode(),
                        recommended_action=data.get(b'action', b'avoid').decode(),
                        timestamp=data.get(b'timestamp', b'').decode()
                    )
                    
        except Exception as e:
            logger.error(f"Get all regimes error: {e}")
            
        return regimes


# Global instance
regime_detector = RegimeDetector()

