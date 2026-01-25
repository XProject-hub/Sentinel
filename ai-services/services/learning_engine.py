"""
SENTINEL AI - Advanced Multi-Source Learning Engine
Learns from: Historical Data, Market Movements, News, Technical Patterns, Trades

This AI continuously improves by:
1. Pre-training on historical market data
2. Learning from real-time market movements
3. Analyzing news sentiment patterns
4. Recognizing technical analysis patterns
5. Learning from every trade outcome
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from loguru import logger
import redis.asyncio as redis
import json
import httpx

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


@dataclass
class MarketPattern:
    """A learned market pattern"""
    pattern_type: str  # 'bullish_reversal', 'bearish_breakdown', etc.
    indicators: Dict[str, float]  # RSI, MACD, etc.
    outcome: str  # 'up', 'down', 'sideways'
    success_rate: float
    occurrences: int


class LearningEngine:
    """
    Advanced Multi-Source Learning Engine.
    
    Learning Sources:
    1. Historical market data (pre-training)
    2. Real-time market movements (continuous learning)
    3. News sentiment (sentiment patterns)
    4. Technical patterns (pattern recognition)
    5. Trade outcomes (reinforcement learning)
    
    This creates a smarter AI that improves over time.
    """
    
    def __init__(self):
        self.redis_client = None
        self.http_client = None
        self.is_running = False
        self._learning_task = None
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.15
        self.min_exploration = 0.02
        self.exploration_decay = 0.998
        
        # Q-values for strategy-regime combinations
        self.strategy_q_values: Dict[str, Dict[str, float]] = {}
        
        # Technical pattern recognition
        self.pattern_memory: Dict[str, MarketPattern] = {}
        
        # News sentiment learning
        self.sentiment_patterns: Dict[str, Dict[str, float]] = {}
        
        # Market state learning
        self.market_states: Dict[str, Dict[str, Any]] = {}
        
        # Definitions
        self.regimes = ['bull_trend', 'bear_trend', 'sideways', 'high_volatility', 'low_volatility', 'breakout', 'reversal']
        self.strategies = ['momentum', 'grid', 'scalping', 'mean_reversion', 'breakout', 'hedge', 'hold', 'aggressive', 'conservative']
        self.patterns = [
            'double_bottom', 'double_top', 'head_shoulders', 'inv_head_shoulders',
            'bull_flag', 'bear_flag', 'ascending_triangle', 'descending_triangle',
            'rsi_oversold', 'rsi_overbought', 'macd_crossover', 'macd_crossunder',
            'support_bounce', 'resistance_reject', 'volume_spike', 'volume_dry'
        ]
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_volume = 0.0  # Track trading volume
        self.max_drawdown = 0.0
        self.learning_iterations = 0
        self.patterns_learned = 0
        self.market_states_learned = 0
        
    async def initialize(self):
        """Initialize learning engine and start background learning"""
        logger.info("Initializing Advanced Learning Engine...")
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load saved learning state
        await self._load_all_learning_data()
        
        # Initialize Q-table if empty
        if not self.strategy_q_values:
            self._initialize_q_table()
            # Save initial Q-values to Redis immediately
            await self._save_all_learning_data()
            logger.info("Initialized and saved Q-table with prior knowledge")
            
        self.is_running = True
        
        # Start background learning tasks
        self._learning_task = asyncio.create_task(self._continuous_learning_loop())
        
        logger.info("Learning Engine initialized - Multi-source AI learning active")
        
    async def shutdown(self):
        """Save state and cleanup"""
        self.is_running = False
        if self._learning_task:
            self._learning_task.cancel()
        await self._save_all_learning_data()
        if self.http_client:
            await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.close()

    def _initialize_q_table(self):
        """Initialize Q-values with prior knowledge"""
        for regime in self.regimes:
            self.strategy_q_values[regime] = {}
            for strategy in self.strategies:
                self.strategy_q_values[regime][strategy] = 0.0
                
        # Prior knowledge from trading experience
        # Bull trends
        self.strategy_q_values['bull_trend']['momentum'] = 0.5
        self.strategy_q_values['bull_trend']['aggressive'] = 0.4
        self.strategy_q_values['bull_trend']['breakout'] = 0.3
        
        # Bear trends
        self.strategy_q_values['bear_trend']['hedge'] = 0.5
        self.strategy_q_values['bear_trend']['conservative'] = 0.4
        self.strategy_q_values['bear_trend']['scalping'] = 0.2
        
        # Sideways markets
        self.strategy_q_values['sideways']['grid'] = 0.5
        self.strategy_q_values['sideways']['mean_reversion'] = 0.4
        self.strategy_q_values['sideways']['scalping'] = 0.3
        
        # High volatility
        self.strategy_q_values['high_volatility']['scalping'] = 0.4
        self.strategy_q_values['high_volatility']['conservative'] = 0.3
        self.strategy_q_values['high_volatility']['hold'] = 0.2
        
        # Low volatility
        self.strategy_q_values['low_volatility']['grid'] = 0.4
        self.strategy_q_values['low_volatility']['mean_reversion'] = 0.3
        
        # Breakout
        self.strategy_q_values['breakout']['breakout'] = 0.6
        self.strategy_q_values['breakout']['momentum'] = 0.5
        self.strategy_q_values['breakout']['aggressive'] = 0.4
        
        # Reversal
        self.strategy_q_values['reversal']['mean_reversion'] = 0.5
        self.strategy_q_values['reversal']['conservative'] = 0.4

    async def _continuous_learning_loop(self):
        """Background task that continuously learns from multiple sources"""
        logger.info("Starting continuous learning loop...")
        
        while self.is_running:
            try:
                # 1. Learn from historical market data (every 30 min)
                if self.learning_iterations % 6 == 0:
                    await self._learn_from_historical_data()
                
                # 2. Learn from current market state (every 5 min)
                await self._learn_from_market_state()
                
                # 3. Learn from news sentiment (every 5 min)
                await self._learn_from_news_sentiment()
                
                # 4. Learn technical patterns (every 5 min)
                await self._learn_technical_patterns()
                
                # 5. Save progress
                self.learning_iterations += 1
                if self.learning_iterations % 12 == 0:  # Every hour
                    await self._save_all_learning_data()
                    
                # Log learning progress
                total_learned = len(self.pattern_memory) + len(self.market_states) + self._count_q_states()
                logger.info(f"Learning iteration {self.learning_iterations}: {total_learned} total states learned")
                
                await asyncio.sleep(300)  # 5 minutes between iterations
                
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                await asyncio.sleep(60)

    async def _learn_from_historical_data(self):
        """Pre-train on historical market data from Redis + Bybit API"""
        try:
            symbols_analyzed = 0
            candles_processed = 0
            
            # === PART 1: Learn from loaded historical data in Redis ===
            if self.redis_client:
                try:
                    # Find all loaded historical datasets
                    historical_keys = await self.redis_client.keys('ai:klines:*')
                    
                    for key in historical_keys[:10]:  # Process up to 10 datasets per iteration
                        try:
                            key_str = key.decode() if isinstance(key, bytes) else key
                            klines_raw = await self.redis_client.get(key_str)
                            
                            if klines_raw:
                                klines_data = json.loads(klines_raw)
                                
                                if isinstance(klines_data, list) and len(klines_data) >= 50:
                                    # Extract symbol from key (ai:klines:BTCUSDT:1h)
                                    parts = key_str.split(':')
                                    symbol = parts[2] if len(parts) > 2 else 'UNKNOWN'
                                    
                                    # Convert to analysis format
                                    # Redis data format: {open_time, open, high, low, close, volume, ...}
                                    closes = [float(k.get('close', 0)) for k in klines_data if k.get('close')]
                                    highs = [float(k.get('high', 0)) for k in klines_data if k.get('high')]
                                    lows = [float(k.get('low', 0)) for k in klines_data if k.get('low')]
                                    volumes = [float(k.get('volume', 0)) for k in klines_data if k.get('volume')]
                                    
                                    if len(closes) >= 50:
                                        await self._analyze_historical_closes(symbol, closes, highs, lows, volumes)
                                        symbols_analyzed += 1
                                        candles_processed += len(closes)
                        except Exception as e:
                            logger.debug(f"Error processing historical key: {e}")
                            continue
                            
                    if symbols_analyzed > 0:
                        logger.info(f"Learned from {symbols_analyzed} historical datasets ({candles_processed:,} candles)")
                        
                except Exception as e:
                    logger.debug(f"Redis historical read error: {e}")
            
            # === PART 2: Also fetch live data from Bybit API ===
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
            
            for symbol in symbols:
                try:
                    # Fetch historical klines
                    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=60&limit=200"
                    response = await self.http_client.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        klines = data.get('result', {}).get('list', [])
                        
                        if len(klines) >= 50:
                            # Analyze historical patterns
                            await self._analyze_historical_klines(symbol, klines)
                            symbols_analyzed += 1
                except Exception as e:
                    logger.debug(f"Bybit API error for {symbol}: {e}")
                    continue
                    
            logger.info(f"Historical learning complete: analyzed {symbols_analyzed} symbols")
            
        except Exception as e:
            logger.error(f"Historical learning error: {e}")
    
    async def _analyze_historical_closes(self, symbol: str, closes: List, highs: List, lows: List, volumes: List):
        """Analyze historical data directly from closes/highs/lows/volumes arrays"""
        try:
            if len(closes) < 50:
                return
                
            # Calculate indicators
            rsi = self._calculate_rsi(closes)
            macd, signal = self._calculate_macd(closes)
            volatility = self._calculate_volatility(closes)
            
            # Detect patterns and learn from outcomes
            patterns_found = 0
            for i in range(30, min(len(closes) - 10, 200)):  # Limit to avoid long processing
                pattern_state = self._detect_pattern_at_index(
                    closes[:i+1], highs[:i+1], lows[:i+1], volumes[:i+1],
                    rsi[:i+1] if i < len(rsi) else rsi,
                    macd[:i+1] if i < len(macd) else macd,
                    signal[:i+1] if i < len(signal) else signal
                )
                
                if pattern_state:
                    # Determine outcome (price movement after pattern)
                    future_price = closes[min(i + 10, len(closes) - 1)]
                    current_price = closes[i]
                    price_change = (future_price - current_price) / current_price * 100
                    
                    outcome = 'up' if price_change > 0.5 else ('down' if price_change < -0.5 else 'sideways')
                    
                    # Update pattern memory
                    await self._update_pattern_memory(pattern_state, outcome)
                    patterns_found += 1
            
            if patterns_found > 0:
                logger.debug(f"Learned {patterns_found} patterns from {symbol} historical data")
                
        except Exception as e:
            logger.debug(f"Historical closes analysis error: {e}")

    async def _analyze_historical_klines(self, symbol: str, klines: List):
        """Analyze historical klines for patterns and outcomes"""
        try:
            # Convert klines to OHLCV
            # Bybit kline format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
            closes = [float(k[4]) for k in reversed(klines)]
            highs = [float(k[2]) for k in reversed(klines)]
            lows = [float(k[3]) for k in reversed(klines)]
            volumes = [float(k[5]) for k in reversed(klines)]
            
            if len(closes) < 50:
                return
                
            # Calculate indicators
            rsi = self._calculate_rsi(closes)
            macd, signal = self._calculate_macd(closes)
            sma_20 = self._calculate_sma(closes, 20)
            sma_50 = self._calculate_sma(closes, 50)
            volatility = self._calculate_volatility(closes)
            
            # Detect patterns and their outcomes
            for i in range(30, len(closes) - 10):
                pattern_state = self._detect_pattern_at_index(
                    closes[:i+1], highs[:i+1], lows[:i+1], volumes[:i+1],
                    rsi[:i+1] if i < len(rsi) else rsi,
                    macd[:i+1] if i < len(macd) else macd,
                    signal[:i+1] if i < len(signal) else signal
                )
                
                if pattern_state:
                    # Calculate what happened next (10 candles forward)
                    future_return = (closes[min(i+10, len(closes)-1)] - closes[i]) / closes[i] * 100
                    
                    outcome = 'up' if future_return > 0.5 else ('down' if future_return < -0.5 else 'sideways')
                    
                    # Update pattern memory
                    await self._update_pattern_memory(pattern_state, outcome, future_return)
                    
        except Exception as e:
            logger.debug(f"Historical analysis error for {symbol}: {e}")

    def _detect_pattern_at_index(self, closes, highs, lows, volumes, rsi, macd, signal) -> Optional[str]:
        """Detect market pattern at a given index"""
        if len(closes) < 20 or len(rsi) < 1:
            return None
            
        current_rsi = rsi[-1] if rsi else 50
        current_macd = macd[-1] if macd else 0
        current_signal = signal[-1] if signal else 0
        
        patterns_detected = []
        
        # RSI patterns
        if current_rsi < 30:
            patterns_detected.append('rsi_oversold')
        elif current_rsi > 70:
            patterns_detected.append('rsi_overbought')
            
        # MACD crossovers
        if len(macd) >= 2 and len(signal) >= 2:
            if macd[-2] < signal[-2] and macd[-1] > signal[-1]:
                patterns_detected.append('macd_crossover')
            elif macd[-2] > signal[-2] and macd[-1] < signal[-1]:
                patterns_detected.append('macd_crossunder')
                
        # Price patterns
        if len(closes) >= 20:
            recent_high = max(closes[-20:])
            recent_low = min(closes[-20:])
            current = closes[-1]
            
            # Support/Resistance
            if current <= recent_low * 1.02:
                patterns_detected.append('support_bounce')
            elif current >= recent_high * 0.98:
                patterns_detected.append('resistance_reject')
                
        # Volume spike
        if len(volumes) >= 20:
            avg_volume = sum(volumes[-20:]) / 20
            if volumes[-1] > avg_volume * 2:
                patterns_detected.append('volume_spike')
            elif volumes[-1] < avg_volume * 0.5:
                patterns_detected.append('volume_dry')
                
        if patterns_detected:
            return '_'.join(sorted(patterns_detected))
        return None

    async def _update_pattern_memory(self, pattern: str, outcome: str, return_pct: float):
        """Update pattern memory with new observation"""
        if pattern not in self.pattern_memory:
            self.pattern_memory[pattern] = {
                'outcomes': {'up': 0, 'down': 0, 'sideways': 0},
                'total_return': 0.0,
                'count': 0
            }
            
        self.pattern_memory[pattern]['outcomes'][outcome] += 1
        self.pattern_memory[pattern]['total_return'] += return_pct
        self.pattern_memory[pattern]['count'] += 1
        self.patterns_learned += 1

    async def _learn_from_market_state(self):
        """Learn from current market conditions"""
        try:
            # Get current market data for major symbols
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            
            for symbol in symbols:
                # Fetch current ticker
                url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
                response = await self.http_client.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    tickers = data.get('result', {}).get('list', [])
                    
                    if tickers:
                        ticker = tickers[0]
                        price_change = float(ticker.get('price24hPcnt', 0)) * 100
                        volume = float(ticker.get('volume24h', 0))
                        
                        # Determine current regime
                        regime = self._determine_regime(price_change, volume)
                        
                        # Store market state
                        state_key = f"{symbol}_{regime}"
                        if state_key not in self.market_states:
                            self.market_states[state_key] = {
                                'observations': 0,
                                'avg_return': 0.0,
                                'regime': regime
                            }
                            
                        self.market_states[state_key]['observations'] += 1
                        self.market_states_learned += 1
                        
        except Exception as e:
            logger.debug(f"Market state learning error: {e}")

    def _determine_regime(self, price_change: float, volume: float) -> str:
        """Determine market regime from indicators"""
        if price_change > 5:
            return 'bull_trend'
        elif price_change < -5:
            return 'bear_trend'
        elif abs(price_change) > 3:
            return 'high_volatility'
        elif abs(price_change) < 1:
            return 'low_volatility' if volume < 1000000000 else 'sideways'
        else:
            return 'sideways'

    async def _learn_from_news_sentiment(self):
        """Learn patterns from news sentiment"""
        try:
            # Get news sentiment from our data aggregator
            news_data = await self.redis_client.get('data:crypto_news')
            
            if news_data:
                news = json.loads(news_data)
                sentiment = news.get('sentiment', {})
                
                overall = sentiment.get('overall', 'neutral')
                bullish_pct = sentiment.get('bullish_percent', 0)
                bearish_pct = sentiment.get('bearish_percent', 0)
                
                # Create sentiment state
                if bullish_pct > 60:
                    sentiment_state = 'very_bullish'
                elif bullish_pct > 40:
                    sentiment_state = 'bullish'
                elif bearish_pct > 60:
                    sentiment_state = 'very_bearish'
                elif bearish_pct > 40:
                    sentiment_state = 'bearish'
                else:
                    sentiment_state = 'neutral'
                    
                # Learn from sentiment
                if sentiment_state not in self.sentiment_patterns:
                    self.sentiment_patterns[sentiment_state] = {
                        'observations': 0,
                        'strategy_outcomes': {}
                    }
                    
                self.sentiment_patterns[sentiment_state]['observations'] += 1
                
                # Update Q-values based on sentiment
                if sentiment_state in ['very_bullish', 'bullish']:
                    self._boost_strategy('bull_trend', 'momentum', 0.01)
                    self._boost_strategy('bull_trend', 'aggressive', 0.01)
                elif sentiment_state in ['very_bearish', 'bearish']:
                    self._boost_strategy('bear_trend', 'hedge', 0.01)
                    self._boost_strategy('bear_trend', 'conservative', 0.01)
                    
        except Exception as e:
            logger.debug(f"News sentiment learning error: {e}")

    def _boost_strategy(self, regime: str, strategy: str, amount: float):
        """Slightly boost a strategy's Q-value"""
        if regime in self.strategy_q_values and strategy in self.strategy_q_values[regime]:
            self.strategy_q_values[regime][strategy] += amount

    async def _learn_technical_patterns(self):
        """Learn from technical analysis patterns"""
        try:
            # Analyze BTC as market leader
            url = "https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=15&limit=100"
            response = await self.http_client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                klines = data.get('result', {}).get('list', [])
                
                if len(klines) >= 50:
                    closes = [float(k[4]) for k in reversed(klines)]
                    
                    # Calculate indicators
                    rsi = self._calculate_rsi(closes)
                    
                    if rsi:
                        current_rsi = rsi[-1]
                        
                        # Learn RSI patterns
                        if current_rsi < 30:
                            # Oversold - momentum often follows
                            self._boost_strategy('reversal', 'mean_reversion', 0.02)
                            self._boost_strategy('reversal', 'momentum', 0.01)
                        elif current_rsi > 70:
                            # Overbought - caution
                            self._boost_strategy('reversal', 'conservative', 0.02)
                            self._boost_strategy('reversal', 'hedge', 0.01)
                            
        except Exception as e:
            logger.debug(f"Technical pattern learning error: {e}")

    # === TECHNICAL INDICATOR CALCULATIONS ===
    
    def _calculate_rsi(self, closes: List[float], period: int = 14) -> List[float]:
        """Calculate RSI"""
        if len(closes) < period + 1:
            return []
            
        rsi_values = []
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
            
            if i >= period:
                avg_gain = sum(gains[-period:]) / period
                avg_loss = sum(losses[-period:]) / period
                
                if avg_loss == 0:
                    rsi_values.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi_values.append(100 - (100 / (1 + rs)))
                    
        return rsi_values

    def _calculate_macd(self, closes: List[float]) -> Tuple[List[float], List[float]]:
        """Calculate MACD and Signal line"""
        if len(closes) < 26:
            return [], []
            
        ema_12 = self._calculate_ema(closes, 12)
        ema_26 = self._calculate_ema(closes, 26)
        
        macd = []
        for i in range(len(ema_26)):
            if i < len(ema_12) - len(ema_26):
                continue
            idx_12 = i + (len(ema_12) - len(ema_26))
            macd.append(ema_12[idx_12] - ema_26[i])
            
        signal = self._calculate_ema(macd, 9) if len(macd) >= 9 else []
        
        return macd, signal

    def _calculate_ema(self, data: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return []
            
        multiplier = 2 / (period + 1)
        ema = [sum(data[:period]) / period]
        
        for price in data[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
            
        return ema

    def _calculate_sma(self, data: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        if len(data) < period:
            return []
        return [sum(data[i:i+period])/period for i in range(len(data)-period+1)]

    def _calculate_volatility(self, closes: List[float], period: int = 20) -> float:
        """Calculate price volatility"""
        if len(closes) < period:
            return 0.0
        returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]
        if not returns:
            return 0.0
        return np.std(returns[-period:])

    # === Q-LEARNING METHODS ===
    
    def calculate_reward(self, trade: TradeOutcome) -> float:
        """Calculate reward for a trade outcome"""
        reward = 0.0
        
        # Profit factor (primary)
        if trade.pnl > 0:
            reward += trade.pnl_percent * 2.0
        else:
            reward += trade.pnl_percent * 3.0
            
        # Risk-adjusted return
        if trade.volatility_at_entry > 0:
            risk_adjusted = trade.pnl_percent / trade.volatility_at_entry
            reward += risk_adjusted * 0.5
            
        # Time efficiency
        hold_hours = trade.hold_time_seconds / 3600
        if trade.pnl > 0 and hold_hours < 1:
            reward += 0.2
        elif trade.pnl < 0 and hold_hours > 6:
            reward -= 0.2
            
        return reward
        
    async def update_from_trade(self, trade: TradeOutcome):
        """Update Q-values from trade outcome"""
        regime = trade.market_regime
        strategy = trade.strategy
        
        if regime not in self.strategy_q_values:
            self.strategy_q_values[regime] = {s: 0.0 for s in self.strategies}
            
        if strategy not in self.strategy_q_values[regime]:
            self.strategy_q_values[regime][strategy] = 0.0
            
        # Calculate reward
        reward = self.calculate_reward(trade)
        
        # Q-learning update
        current_q = self.strategy_q_values[regime][strategy]
        max_future_q = max(self.strategy_q_values[regime].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        
        self.strategy_q_values[regime][strategy] = new_q
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += trade.pnl
        # Track volume (position value)
        self.total_volume += abs(getattr(trade, 'position_value', 0)) or abs(trade.pnl * 100)  # Estimate if not available
        if trade.pnl > 0:
            self.winning_trades += 1
            
        if trade.pnl < 0:
            current_drawdown = abs(trade.pnl_percent)
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                
        # Decay exploration
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
        
        # Save periodically
        if self.total_trades % 5 == 0:
            await self._save_all_learning_data()
            
        logger.info(f"Trade learning: {regime}/{strategy} Q={new_q:.4f} reward={reward:.4f}")
        
        # Store learning event
        await self._store_learning_event({
            'type': 'trade',
            'state': f"{regime}_{strategy}",
            'action': strategy,
            'reward': reward,
            'new_q': new_q,
            'pnl': trade.pnl,
            'timestamp': datetime.utcnow().isoformat()
        })

    async def _store_learning_event(self, event: Dict):
        """Store a learning event in Redis"""
        try:
            await self.redis_client.lpush('ai:learning:history', json.dumps(event))
            await self.redis_client.ltrim('ai:learning:history', 0, 499)
        except Exception as e:
            logger.debug(f"Failed to store learning event: {e}")

    def get_best_strategy(self, regime: str) -> Tuple[str, float]:
        """Get best strategy for a regime using epsilon-greedy"""
        if regime not in self.strategy_q_values:
            return 'hold', 0.5
            
        # Exploration
        if np.random.random() < self.exploration_rate:
            random_strategy = np.random.choice(self.strategies)
            return random_strategy, self.strategy_q_values[regime].get(random_strategy, 0)
            
        # Exploitation
        best = max(self.strategy_q_values[regime].items(), key=lambda x: x[1])
        return best[0], best[1]
        
    def get_strategy_confidence(self, regime: str, strategy: str) -> float:
        """Get confidence level (0-100%)"""
        if regime not in self.strategy_q_values:
            return 50.0
        q_value = self.strategy_q_values[regime].get(strategy, 0)
        confidence = 50 + (q_value * 25)
        return max(0, min(100, confidence))

    def _count_q_states(self) -> int:
        """Count total Q-states with significant learning"""
        count = 0
        for regime_values in self.strategy_q_values.values():
            count += sum(1 for q in regime_values.values() if abs(q) > 0.1)
        return count

    # === DATA PERSISTENCE ===
    
    async def _load_all_learning_data(self):
        """Load all learning data from Redis"""
        try:
            # Q-values
            q_data = await self.redis_client.get('ai:learning:q_values')
            if q_data:
                self.strategy_q_values = json.loads(q_data)
                
            # Pattern memory
            pattern_data = await self.redis_client.get('ai:learning:patterns')
            if pattern_data:
                self.pattern_memory = json.loads(pattern_data)
                
            # Sentiment patterns
            sentiment_data = await self.redis_client.get('ai:learning:sentiment')
            if sentiment_data:
                self.sentiment_patterns = json.loads(sentiment_data)
                
            # Market states
            market_data = await self.redis_client.get('ai:learning:market_states')
            if market_data:
                self.market_states = json.loads(market_data)
                
            # Statistics
            stats = await self.redis_client.hgetall('ai:learning:stats')
            if stats:
                self.total_trades = int(stats.get(b'total_trades', 0))
                self.winning_trades = int(stats.get(b'winning_trades', 0))
                self.total_pnl = float(stats.get(b'total_pnl', 0))
                self.total_volume = float(stats.get(b'total_volume', 0))
                self.max_drawdown = float(stats.get(b'max_drawdown', 0))
                self.learning_iterations = int(stats.get(b'learning_iterations', 0))
                self.patterns_learned = int(stats.get(b'patterns_learned', 0))
                self.market_states_learned = int(stats.get(b'market_states_learned', 0))
                
            logger.info(f"Loaded learning data: {self._count_q_states()} Q-states, {len(self.pattern_memory)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")

    async def _save_all_learning_data(self):
        """Save all learning data to Redis"""
        try:
            # Q-values
            await self.redis_client.set('ai:learning:q_values', json.dumps(self.strategy_q_values))
            
            # Pattern memory
            await self.redis_client.set('ai:learning:patterns', json.dumps(self.pattern_memory))
            
            # Sentiment patterns
            await self.redis_client.set('ai:learning:sentiment', json.dumps(self.sentiment_patterns))
            
            # Market states
            await self.redis_client.set('ai:learning:market_states', json.dumps(self.market_states))
            
            # Statistics
            total_states = self._count_q_states() + len(self.pattern_memory) + len(self.market_states)
            
            await self.redis_client.hset('ai:learning:stats', mapping={
                'total_trades': str(self.total_trades),
                'winning_trades': str(self.winning_trades),
                'win_rate': str(self.winning_trades / max(1, self.total_trades) * 100),
                'total_pnl': str(self.total_pnl),
                'total_volume': str(self.total_volume),
                'max_drawdown': str(self.max_drawdown),
                'learning_iterations': str(self.learning_iterations),
                'patterns_learned': str(self.patterns_learned),
                'market_states_learned': str(self.market_states_learned),
                'total_states_learned': str(total_states),
                'updated_at': datetime.utcnow().isoformat()
            })
            
            # Also store for dashboard
            await self.redis_client.hset('ai:trading:stats', mapping={
                'total_trades': str(self.total_trades),
                'wins': str(self.winning_trades),
                'losses': str(self.total_trades - self.winning_trades),
                'win_rate': str(self.winning_trades / max(1, self.total_trades) * 100),
                'total_profit': str(self.total_pnl),
                'avg_profit': str(self.total_pnl / max(1, self.total_trades)),
                'best_trade': str(self.max_drawdown),  # Will be updated properly
                'worst_trade': str(-self.max_drawdown),
                'total_volume': str(getattr(self, 'total_volume', 0)),
                'updated_at': datetime.utcnow().isoformat()
            })
            
            logger.debug(f"Saved learning data: {total_states} total states")
            
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")

    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        total_states = self._count_q_states() + len(self.pattern_memory) + len(self.market_states)
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades) * 100,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'exploration_rate': self.exploration_rate * 100,
            'learning_iterations': self.learning_iterations,
            'total_states_learned': total_states,
            'q_states': self._count_q_states(),
            'patterns_learned': len(self.pattern_memory),
            'market_states': len(self.market_states),
            'sentiment_states': len(self.sentiment_patterns),
            'best_performing': self._get_best_performing_strategies(),
            'top_patterns': self._get_top_patterns(),
        }

    def _get_best_performing_strategies(self) -> List[Dict]:
        """Get top strategy-regime combinations"""
        all_combos = []
        for regime, strategies in self.strategy_q_values.items():
            for strategy, q_value in strategies.items():
                all_combos.append({
                    'regime': regime,
                    'strategy': strategy,
                    'q_value': q_value,
                    'confidence': self.get_strategy_confidence(regime, strategy)
                })
        return sorted(all_combos, key=lambda x: x['q_value'], reverse=True)[:10]

    def _get_top_patterns(self) -> List[Dict]:
        """Get most reliable learned patterns"""
        patterns = []
        for pattern, data in self.pattern_memory.items():
            if data['count'] >= 5:
                outcomes = data['outcomes']
                total = sum(outcomes.values())
                if total > 0:
                    best_outcome = max(outcomes.items(), key=lambda x: x[1])
                    patterns.append({
                        'pattern': pattern,
                        'best_outcome': best_outcome[0],
                        'success_rate': best_outcome[1] / total * 100,
                        'avg_return': data['total_return'] / data['count'],
                        'occurrences': data['count']
                    })
        return sorted(patterns, key=lambda x: x['success_rate'], reverse=True)[:10]

    async def get_recent_learning_events(self, limit: int = 20) -> List[Dict]:
        """Get recent learning events"""
        try:
            events = await self.redis_client.lrange('ai:learning:history', 0, limit - 1)
            return [json.loads(e) for e in events]
        except:
            return []

    async def update(self, state: Dict, action: str, reward: float, next_state: Dict):
        """Update Q-values from state-action-reward tuple (for external training)"""
        try:
            regime = state.get('regime', 'sideways')
            
            if regime not in self.strategy_q_values:
                self.strategy_q_values[regime] = {s: 0.0 for s in self.strategies}
                
            if action not in self.strategy_q_values[regime]:
                self.strategy_q_values[regime][action] = 0.0
                
            # Q-learning update
            current_q = self.strategy_q_values[regime][action]
            max_future_q = max(self.strategy_q_values.get(regime, {}).values()) if self.strategy_q_values.get(regime) else 0
            
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_future_q - current_q
            )
            
            self.strategy_q_values[regime][action] = new_q
            
            logger.debug(f"Q-Learning update: {regime}/{action} Q={new_q:.4f} reward={reward:.4f}")
            
        except Exception as e:
            logger.error(f"Q-Learning update error: {e}")


# Global instance
learning_engine = LearningEngine()
