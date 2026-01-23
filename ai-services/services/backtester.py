"""
SENTINEL AI - Backtest Engine
Test trading strategies on historical data

Features:
1. Download historical OHLCV data from Bybit
2. Simulate trades with realistic fees and slippage
3. Calculate performance metrics
4. Compare different strategy parameters

This helps validate strategies BEFORE risking real money.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger
import redis.asyncio as redis
import json
import httpx
import numpy as np
from collections import defaultdict

from config import settings


@dataclass
class BacktestTrade:
    """A simulated trade"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    size_usd: float = 100.0
    leverage: int = 1
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""
    fees_paid: float = 0.0


@dataclass
class BacktestResult:
    """Results of a backtest run"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    
    # Capital
    initial_capital: float
    final_capital: float
    
    # Performance
    total_return: float  # Percentage
    total_pnl: float  # USD
    
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk metrics
    max_drawdown: float  # Percentage
    sharpe_ratio: float
    profit_factor: float  # Gross profit / Gross loss
    
    # Average trade
    avg_win: float
    avg_loss: float
    avg_trade_duration: float  # Minutes
    
    # All trades
    trades: List[BacktestTrade] = field(default_factory=list)
    
    # Equity curve
    equity_curve: List[Dict] = field(default_factory=list)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run"""
    symbol: str = "BTCUSDT"
    start_date: str = ""  # YYYY-MM-DD, empty = 30 days ago
    end_date: str = ""  # YYYY-MM-DD, empty = today
    initial_capital: float = 1000.0
    
    # Strategy parameters
    take_profit_percent: float = 2.0
    stop_loss_percent: float = 1.0
    trailing_stop_percent: float = 0.5
    min_profit_to_trail: float = 0.5
    
    # Position sizing
    position_size_percent: float = 10.0  # % of capital per trade
    max_open_positions: int = 1
    leverage: int = 1
    
    # Fees
    maker_fee: float = 0.01  # 0.01%
    taker_fee: float = 0.06  # 0.06%
    slippage: float = 0.05  # 0.05% slippage
    
    # Strategy type
    strategy: str = "trend_following"  # trend_following, mean_reversion, breakout


class Backtester:
    """
    Professional Backtesting Engine
    
    Simulates trading strategies on historical data with:
    - Realistic fees and slippage
    - Multiple strategy types
    - Comprehensive metrics
    - Walk-forward testing support
    """
    
    # Bybit API for historical data
    BASE_URL = "https://api.bybit.com"
    
    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None
        self.redis_client: Optional[redis.Redis] = None
        self._candle_cache: Dict[str, List[Dict]] = {}
    
    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client - creates new one if closed"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client
        
    async def initialize(self):
        """Initialize connections"""
        try:
            self.redis_client = await redis.from_url(settings.REDIS_URL)
            logger.info("Backtester initialized")
        except Exception as e:
            logger.error(f"Backtester init failed: {e}")
    
    async def close(self):
        """Close connections - but don't crash if already closed"""
        try:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()
        except Exception as e:
            logger.warning(f"Error closing backtester HTTP client: {e}")
    
    async def fetch_historical_data(
        self, 
        symbol: str, 
        interval: str = "15",  # 15 minute candles
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Fetch historical OHLCV data from Bybit
        
        Returns list of candles:
        [{'time': datetime, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}, ...]
        """
        try:
            if start_time is None:
                start_time = datetime.utcnow() - timedelta(days=30)
            if end_time is None:
                end_time = datetime.utcnow()
            
            logger.info(f"Backtest: Fetching {symbol} from {start_time} to {end_time}")
            
            all_candles = []
            current_start = start_time
            batch_count = 0
            
            while current_start < end_time:
                batch_count += 1
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "start": int(current_start.timestamp() * 1000),
                    "end": int(end_time.timestamp() * 1000),
                    "limit": min(limit, 1000)
                }
                
                logger.debug(f"Batch {batch_count}: Requesting {symbol} from {current_start}")
                
                http_client = self._get_http_client()
                response = await http_client.get(
                    f"{self.BASE_URL}/v5/market/kline",
                    params=params
                )
                
                logger.debug(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch candles: {response.text}")
                    break
                
                data = response.json()
                logger.debug(f"API retCode: {data.get('retCode')}, retMsg: {data.get('retMsg')}")
                
                if data.get('retCode') != 0:
                    logger.error(f"Bybit API error: {data.get('retMsg')}")
                    break
                
                candles = data.get('result', {}).get('list', [])
                logger.info(f"Batch {batch_count}: Got {len(candles)} candles")
                
                if not candles:
                    logger.warning(f"No candles returned for {symbol}")
                    break
                
                # Bybit returns newest first, we want oldest first
                candles.reverse()
                
                for candle in candles:
                    timestamp = int(candle[0])
                    all_candles.append({
                        'time': datetime.fromtimestamp(timestamp / 1000),
                        'timestamp': timestamp,
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5])
                    })
                
                # Move to next batch - break if we got less than requested
                if len(candles) < min(limit, 1000):
                    logger.info(f"Got all available data ({len(candles)} < {min(limit, 1000)})")
                    break
                    
                # Get the newest candle's timestamp (after reverse, it's the last one)
                last_time = datetime.fromtimestamp(int(candles[-1][0]) / 1000)
                current_start = last_time + timedelta(minutes=int(interval))
                logger.debug(f"Next batch starts at: {current_start}")
                
                await asyncio.sleep(0.1)  # Rate limiting
                
                # Safety limit - max 10 batches (10000 candles)
                if batch_count >= 10:
                    logger.info(f"Reached max batch limit")
                    break
            
            # Remove duplicates and sort
            seen = set()
            unique_candles = []
            for candle in all_candles:
                if candle['timestamp'] not in seen:
                    seen.add(candle['timestamp'])
                    unique_candles.append(candle)
            
            unique_candles.sort(key=lambda x: x['timestamp'])
            
            logger.info(f"Fetched {len(unique_candles)} candles for {symbol}")
            return unique_candles
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    def _calculate_indicators(self, candles: List[Dict]) -> List[Dict]:
        """Calculate technical indicators for each candle"""
        if len(candles) < 50:
            return candles
        
        closes = np.array([c['close'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])
        
        # Simple Moving Averages
        sma_20 = self._sma(closes, 20)
        sma_50 = self._sma(closes, 50)
        
        # Exponential Moving Average
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)
        
        # MACD
        macd = ema_12 - ema_26
        macd_signal = self._ema(macd, 9)
        
        # RSI
        rsi = self._rsi(closes, 14)
        
        # Bollinger Bands
        bb_middle = sma_20
        bb_std = self._rolling_std(closes, 20)
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        # ATR (Average True Range)
        atr = self._atr(highs, lows, closes, 14)
        
        # Add indicators to candles
        for i, candle in enumerate(candles):
            candle['sma_20'] = sma_20[i] if i < len(sma_20) else 0
            candle['sma_50'] = sma_50[i] if i < len(sma_50) else 0
            candle['ema_12'] = ema_12[i] if i < len(ema_12) else 0
            candle['ema_26'] = ema_26[i] if i < len(ema_26) else 0
            candle['macd'] = macd[i] if i < len(macd) else 0
            candle['macd_signal'] = macd_signal[i] if i < len(macd_signal) else 0
            candle['rsi'] = rsi[i] if i < len(rsi) else 50
            candle['bb_upper'] = bb_upper[i] if i < len(bb_upper) else 0
            candle['bb_lower'] = bb_lower[i] if i < len(bb_lower) else 0
            candle['atr'] = atr[i] if i < len(atr) else 0
        
        return candles
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        result = np.full(len(data), np.nan)
        multiplier = 2 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    def _rsi(self, data: np.ndarray, period: int) -> np.ndarray:
        """Relative Strength Index"""
        result = np.full(len(data), 50.0)
        deltas = np.diff(data)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.full(len(data), np.nan)
        avg_loss = np.full(len(data), np.nan)
        
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
        
        for i in range(period, len(data)):
            if avg_loss[i] == 0:
                result[i] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                result[i] = 100 - (100 / (1 + rs))
        
        return result
    
    def _rolling_std(self, data: np.ndarray, period: int) -> np.ndarray:
        """Rolling standard deviation"""
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.std(data[i - period + 1:i + 1])
        return result
    
    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Average True Range"""
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        tr = np.insert(tr, 0, highs[0] - lows[0])
        
        result = np.full(len(tr), np.nan)
        result[period - 1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
        
        return result
    
    def _generate_signals(self, candles: List[Dict], strategy: str) -> List[Dict]:
        """Generate trading signals based on strategy"""
        for i, candle in enumerate(candles):
            candle['signal'] = 'hold'
            candle['signal_strength'] = 0
            
            if i < 50:  # Need enough data for indicators
                continue
            
            if strategy == "trend_following":
                # Trend following: Buy when price crosses above SMA, sell when crosses below
                if candle['close'] > candle['sma_20'] > candle['sma_50']:
                    if candle['rsi'] < 70:  # Not overbought
                        candle['signal'] = 'long'
                        candle['signal_strength'] = min(100, 50 + (candle['rsi'] - 30))
                elif candle['close'] < candle['sma_20'] < candle['sma_50']:
                    if candle['rsi'] > 30:  # Not oversold
                        candle['signal'] = 'short'
                        candle['signal_strength'] = min(100, 50 + (70 - candle['rsi']))
            
            elif strategy == "mean_reversion":
                # Mean reversion: Buy oversold, sell overbought
                if candle['close'] < candle['bb_lower'] and candle['rsi'] < 30:
                    candle['signal'] = 'long'
                    candle['signal_strength'] = min(100, 100 - candle['rsi'])
                elif candle['close'] > candle['bb_upper'] and candle['rsi'] > 70:
                    candle['signal'] = 'short'
                    candle['signal_strength'] = min(100, candle['rsi'])
            
            elif strategy == "breakout":
                # Breakout: Trade when price breaks out of range
                if i >= 20:
                    recent_high = max(c['high'] for c in candles[i-20:i])
                    recent_low = min(c['low'] for c in candles[i-20:i])
                    
                    if candle['close'] > recent_high:
                        candle['signal'] = 'long'
                        candle['signal_strength'] = 70
                    elif candle['close'] < recent_low:
                        candle['signal'] = 'short'
                        candle['signal_strength'] = 70
            
            elif strategy == "macd_crossover":
                # MACD crossover strategy
                if i > 0:
                    prev = candles[i - 1]
                    if candle['macd'] > candle['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                        candle['signal'] = 'long'
                        candle['signal_strength'] = 60
                    elif candle['macd'] < candle['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                        candle['signal'] = 'short'
                        candle['signal_strength'] = 60
        
        return candles
    
    async def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run a complete backtest
        
        Returns BacktestResult with all metrics and trades
        """
        logger.info(f"Starting backtest for {config.symbol} with strategy: {config.strategy}")
        
        # Parse dates - default to 30 days of historical data
        # Empty string or None means use defaults
        if config.start_date and config.start_date.strip():
            start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
        else:
            # Default: 30 days ago
            start_date = datetime.utcnow() - timedelta(days=30)
        
        if config.end_date and config.end_date.strip():
            end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
        else:
            # Default: now
            end_date = datetime.utcnow()
        
        logger.info(f"Backtest date range: {start_date} to {end_date} (config: start='{config.start_date}', end='{config.end_date}')")
        
        # Fetch historical data
        candles = await self.fetch_historical_data(
            config.symbol,
            interval="15",
            start_time=start_date,
            end_time=end_date
        )
        
        if len(candles) < 100:
            raise ValueError(f"Not enough data for backtest. Got {len(candles)} candles, need at least 100.")
        
        # Calculate indicators
        candles = self._calculate_indicators(candles)
        
        # Generate signals
        candles = self._generate_signals(candles, config.strategy)
        
        # Run simulation
        trades, equity_curve = self._simulate_trading(candles, config)
        
        # Calculate metrics
        result = self._calculate_metrics(trades, equity_curve, config, start_date, end_date)
        
        # Store result in Redis
        if self.redis_client:
            await self._store_result(result)
        
        logger.info(f"Backtest complete: {result.total_trades} trades, {result.win_rate:.1f}% win rate, {result.total_return:.2f}% return")
        
        return result
    
    def _simulate_trading(
        self, 
        candles: List[Dict], 
        config: BacktestConfig
    ) -> Tuple[List[BacktestTrade], List[Dict]]:
        """Simulate trading through the candles"""
        trades: List[BacktestTrade] = []
        equity_curve: List[Dict] = []
        
        capital = config.initial_capital
        open_trades: List[BacktestTrade] = []
        
        for i, candle in enumerate(candles):
            current_time = candle['time']
            current_price = candle['close']
            
            # Check exit conditions for open trades
            closed_trades = []
            for trade in open_trades:
                should_close, exit_reason, exit_price = self._check_exit(
                    trade, candle, config
                )
                
                if should_close:
                    # Calculate P&L
                    if trade.direction == 'long':
                        pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100 * trade.leverage
                    else:
                        pnl_percent = ((trade.entry_price - exit_price) / trade.entry_price) * 100 * trade.leverage
                    
                    # Subtract fees
                    total_fees = (config.taker_fee * 2) + config.slippage
                    pnl_percent -= total_fees
                    
                    trade.exit_time = current_time
                    trade.exit_price = exit_price
                    trade.pnl_percent = pnl_percent
                    trade.pnl = trade.size_usd * (pnl_percent / 100)
                    trade.exit_reason = exit_reason
                    trade.fees_paid = trade.size_usd * (total_fees / 100)
                    
                    capital += trade.pnl
                    closed_trades.append(trade)
                    trades.append(trade)
            
            # Remove closed trades
            for trade in closed_trades:
                open_trades.remove(trade)
            
            # Check for new entry signals
            if len(open_trades) < config.max_open_positions:
                if candle['signal'] in ['long', 'short'] and candle['signal_strength'] >= 50:
                    position_size = capital * (config.position_size_percent / 100)
                    
                    # Apply slippage to entry
                    if candle['signal'] == 'long':
                        entry_price = current_price * (1 + config.slippage / 100)
                    else:
                        entry_price = current_price * (1 - config.slippage / 100)
                    
                    new_trade = BacktestTrade(
                        symbol=config.symbol,
                        direction=candle['signal'],
                        entry_time=current_time,
                        entry_price=entry_price,
                        size_usd=position_size,
                        leverage=config.leverage
                    )
                    open_trades.append(new_trade)
            
            # Record equity
            unrealized_pnl = 0
            for trade in open_trades:
                if trade.direction == 'long':
                    unrealized_pnl += trade.size_usd * ((current_price - trade.entry_price) / trade.entry_price)
                else:
                    unrealized_pnl += trade.size_usd * ((trade.entry_price - current_price) / trade.entry_price)
            
            equity_curve.append({
                'time': current_time.isoformat(),
                'equity': capital + unrealized_pnl,
                'capital': capital,
                'open_positions': len(open_trades)
            })
        
        # Close any remaining open trades at last price
        for trade in open_trades:
            last_candle = candles[-1]
            if trade.direction == 'long':
                pnl_percent = ((last_candle['close'] - trade.entry_price) / trade.entry_price) * 100 * trade.leverage
            else:
                pnl_percent = ((trade.entry_price - last_candle['close']) / trade.entry_price) * 100 * trade.leverage
            
            total_fees = (config.taker_fee * 2) + config.slippage
            pnl_percent -= total_fees
            
            trade.exit_time = last_candle['time']
            trade.exit_price = last_candle['close']
            trade.pnl_percent = pnl_percent
            trade.pnl = trade.size_usd * (pnl_percent / 100)
            trade.exit_reason = "End of backtest"
            trade.fees_paid = trade.size_usd * (total_fees / 100)
            
            capital += trade.pnl
            trades.append(trade)
        
        return trades, equity_curve
    
    def _check_exit(
        self, 
        trade: BacktestTrade, 
        candle: Dict, 
        config: BacktestConfig
    ) -> Tuple[bool, str, float]:
        """Check if trade should be exited"""
        current_price = candle['close']
        high = candle['high']
        low = candle['low']
        
        if trade.direction == 'long':
            # Stop loss
            stop_price = trade.entry_price * (1 - config.stop_loss_percent / 100)
            if low <= stop_price:
                return True, "Stop loss", stop_price
            
            # Take profit
            tp_price = trade.entry_price * (1 + config.take_profit_percent / 100)
            if high >= tp_price:
                return True, "Take profit", tp_price
            
            # Trailing stop (if in profit)
            current_pnl = ((current_price - trade.entry_price) / trade.entry_price) * 100
            if current_pnl >= config.min_profit_to_trail:
                # Calculate trailing stop from peak
                peak_price = max(trade.entry_price, high)
                trail_price = peak_price * (1 - config.trailing_stop_percent / 100)
                if low <= trail_price:
                    return True, "Trailing stop", trail_price
        
        else:  # short
            # Stop loss
            stop_price = trade.entry_price * (1 + config.stop_loss_percent / 100)
            if high >= stop_price:
                return True, "Stop loss", stop_price
            
            # Take profit
            tp_price = trade.entry_price * (1 - config.take_profit_percent / 100)
            if low <= tp_price:
                return True, "Take profit", tp_price
            
            # Trailing stop
            current_pnl = ((trade.entry_price - current_price) / trade.entry_price) * 100
            if current_pnl >= config.min_profit_to_trail:
                trough_price = min(trade.entry_price, low)
                trail_price = trough_price * (1 + config.trailing_stop_percent / 100)
                if high >= trail_price:
                    return True, "Trailing stop", trail_price
        
        return False, "", 0
    
    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[Dict],
        config: BacktestConfig,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics"""
        if not trades:
            return BacktestResult(
                strategy_name=config.strategy,
                symbol=config.symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=config.initial_capital,
                final_capital=config.initial_capital,
                total_return=0,
                total_pnl=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                max_drawdown=0,
                sharpe_ratio=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                avg_trade_duration=0,
                trades=[],
                equity_curve=equity_curve
            )
        
        # Basic stats
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        final_capital = config.initial_capital + total_pnl
        total_return = ((final_capital - config.initial_capital) / config.initial_capital) * 100
        
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        
        # Average win/loss
        avg_win = np.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_percent for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Sharpe ratio (simplified)
        returns = [t.pnl_percent for t in trades]
        if len(returns) > 1:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Average trade duration
        durations = []
        for t in trades:
            if t.exit_time and t.entry_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 60
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0
        
        return BacktestResult(
            strategy_name=config.strategy,
            symbol=config.symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=config.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_pnl=total_pnl,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_duration,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        """Calculate maximum drawdown percentage"""
        if not equity_curve:
            return 0
        
        equities = [e['equity'] for e in equity_curve]
        peak = equities[0]
        max_dd = 0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            drawdown = ((peak - equity) / peak) * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    async def _store_result(self, result: BacktestResult):
        """Store backtest result in Redis"""
        try:
            result_data = {
                'strategy': result.strategy_name,
                'symbol': result.symbol,
                'start_date': result.start_date.isoformat(),
                'end_date': result.end_date.isoformat(),
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital,
                'total_return': result.total_return,
                'total_pnl': result.total_pnl,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'profit_factor': result.profit_factor,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss,
                'avg_trade_duration': result.avg_trade_duration,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store latest result
            await self.redis_client.hset(
                f"backtest:latest:{result.symbol}",
                mapping={k: str(v) for k, v in result_data.items()}
            )
            
            # Add to history
            await self.redis_client.lpush(
                f"backtest:history:{result.symbol}",
                json.dumps(result_data)
            )
            await self.redis_client.ltrim(f"backtest:history:{result.symbol}", 0, 99)
            
        except Exception as e:
            logger.error(f"Failed to store backtest result: {e}")


# Global instance
backtester = Backtester()

