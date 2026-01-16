"""
SENTINEL AI - Autonomous Trading System
24/7 Non-stop trading with real money
Learns, trades, compounds profits automatically

SAFE PROFIT MODE Strategy:
- Many small wins (0.5% target) instead of few big wins
- Very fast exit on any loss (-0.3% max)
- Trailing stops lock in profits
- Better 10 trades x €1 profit than 1 trade x -€10 loss
- Exit immediately when price starts falling
- Emergency exit on market crash
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from loguru import logger
import redis.asyncio as redis
import json

from config import settings
from services.bybit_client import BybitV5Client
from services.learning_engine import LearningEngine, TradeOutcome


@dataclass
class TradeSignal:
    """Trading signal from AI analysis"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    strategy: str
    regime: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_percent: float
    reasoning: str


class AutonomousTrader:
    """
    Fully autonomous 24/7 trading system.
    
    - Monitors ALL crypto pairs
    - Makes trading decisions automatically
    - Uses all available capital
    - Compounds profits
    - Learns from every trade
    - Never sleeps
    """
    
    def __init__(self):
        self.is_running = False
        self.redis_client = None
        self.learning_engine: Optional[LearningEngine] = None
        
        # Connected exchange clients per user
        self.user_clients: Dict[str, BybitV5Client] = {}
        
        # ALL major crypto trading pairs
        self.trading_pairs = [
            # Top 10
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'TRXUSDT',
            # Top 20
            'MATICUSDT', 'LINKUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT',
            'APTUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'FILUSDT',
            # Top 30
            'INJUSDT', 'SEIUSDT', 'SUIUSDT', 'TIAUSDT', 'JUPUSDT',
            'STXUSDT', 'IMXUSDT', 'RUNEUSDT', 'AAVEUSDT', 'MKRUSDT',
            # Top 50
            'GRTUSDT', 'FTMUSDT', 'ALGOUSDT', 'SANDUSDT', 'MANAUSDT',
            'AXSUSDT', 'GALAUSDT', 'APEUSDT', 'LDOUSDT', 'CROUSDT',
            'EGLDUSDT', 'FLOWUSDT', 'XTZUSDT', 'EOSUSDT', 'ARUSDT',
            'CFXUSDT', 'MINAUSDT', 'RNDRUSDT', 'AGIXUSDT', 'FETUSDT',
            # DeFi
            'COMPUSDT', 'SNXUSDT', 'CRVUSDT', 'YFIUSDT', 'SUSHIUSDT',
            '1INCHUSDT', 'DYDXUSDT', 'GMXUSDT', 'PENDLEUSDT', 'ENSUSDT',
            # Meme coins
            'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT',
            # Layer 2
            'STRKUSDT', 'ZKUSDT', 'SCROLLUSDT', 'MANTAUSDT', 'BLASTUSDT',
            # AI coins
            'TAOUSDT', 'WLDUSDT', 'OCEANUSDT', 'RNDRAUSDT', 'AKTUSDT',
            # Gaming
            'ILVUSDT', 'MAGICUSDT', 'PRIMAUSDT', 'BEAMUSDT', 'PIXELUSDT',
        ]
        
        # Trading parameters - SAFE PROFIT MODE
        self.min_trade_interval_seconds = 10  # Even faster trading cycles
        self.max_position_percent = 5.0  # Max 5% of portfolio per position (safer)
        self.min_confidence = 50.0  # Lower threshold - trade more often
        self.max_open_positions = 15  # More smaller positions = diversified risk
        
        # SAFE PROFIT THRESHOLDS - Small consistent wins
        self.take_profit_percent = 0.5   # Take profit at +0.5% (small but safe)
        self.stop_loss_percent = 0.3     # Stop loss at -0.3% (exit FAST on loss)
        self.trailing_activate = 0.3     # Activate trailing at +0.3% profit
        self.trailing_stop = 0.15        # Trail by 0.15% (lock in profits)
        
        # Track last trade times
        self.last_trade_time: Dict[str, datetime] = {}
        
        # Active positions
        self.active_positions: Dict[str, Dict] = {}
        
    async def initialize(self, learning_engine: LearningEngine):
        """Initialize autonomous trader"""
        logger.info("Initializing Autonomous Trading System...")
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.learning_engine = learning_engine
        self.is_running = True
        logger.info("Autonomous Trading System initialized - 24/7 trading enabled")
        
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Autonomous Trader...")
        self.is_running = False
        
        # Close all client connections
        for user_id, client in self.user_clients.items():
            try:
                await client.close()
            except:
                pass
                
        if self.redis_client:
            await self.redis_client.close()
            
    async def connect_user(self, user_id: str, api_key: str, api_secret: str, testnet: bool = False):
        """Connect a user's exchange account for autonomous trading"""
        try:
            client = BybitV5Client(api_key, api_secret, testnet)
            result = await client.test_connection()
            
            if result.get('success'):
                self.user_clients[user_id] = client
                logger.info(f"User {user_id} connected for autonomous trading")
                return True
            else:
                await client.close()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect user {user_id}: {e}")
            return False
            
    async def disconnect_user(self, user_id: str):
        """Disconnect user from autonomous trading"""
        if user_id in self.user_clients:
            await self.user_clients[user_id].close()
            del self.user_clients[user_id]
            logger.info(f"User {user_id} disconnected from autonomous trading")
            
    async def run_trading_loop(self):
        """
        Main 24/7 trading loop.
        Runs continuously, analyzing and trading.
        """
        logger.info("Starting 24/7 autonomous trading loop...")
        
        cycle_count = 0
        
        while self.is_running:
            try:
                cycle_count += 1
                cycle_start = datetime.utcnow()
                
                # For each connected user
                for user_id, client in list(self.user_clients.items()):
                    try:
                        await self._process_user_trading(user_id, client)
                    except Exception as e:
                        logger.error(f"Error processing user {user_id}: {e}")
                        
                # Log status every 100 cycles
                if cycle_count % 100 == 0:
                    await self._log_trading_status()
                    
                # Calculate sleep time to maintain cycle rate
                cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time = max(5 - cycle_duration, 1)  # Run every 5 seconds minimum
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)
                
    async def _process_user_trading(self, user_id: str, client: BybitV5Client):
        """Process trading for a single user"""
        
        logger.info(f"Processing trading for user {user_id}...")
        
        # 1. Get current balance
        balance_result = await client.get_wallet_balance()
        if not balance_result.get('success'):
            logger.warning(f"Failed to get balance for {user_id}: {balance_result.get('error')}")
            return
            
        balance_data = balance_result.get('data', {})
        total_equity = 0
        available_usdt = 0
        
        for account in balance_data.get('list', []):
            total_equity = float(account.get('totalEquity', 0))
            for coin in account.get('coin', []):
                if coin.get('coin') == 'USDT':
                    available_usdt = float(coin.get('availableToWithdraw', 0))
                    
        if total_equity < 10:  # Minimum $10 to trade
            return
            
        # 2. Get current positions
        positions_result = await client.get_positions()
        current_positions = []
        
        if positions_result.get('success'):
            for pos in positions_result.get('data', {}).get('list', []):
                size = float(pos.get('size', 0))
                if size > 0:
                    current_positions.append({
                        'symbol': pos.get('symbol'),
                        'side': pos.get('side'),
                        'size': size,
                        'entryPrice': float(pos.get('avgPrice', 0)),
                        'unrealizedPnl': float(pos.get('unrealisedPnl', 0)),
                    })
                    
        # 3. Check and close losing positions
        for position in current_positions:
            await self._check_position_exit(user_id, client, position, total_equity)
            
        # 4. Analyze each trading pair for new opportunities
        if len(current_positions) < self.max_open_positions:
            symbols_analyzed = 0
            signals_found = 0
            
            for symbol in self.trading_pairs[:20]:  # Check top 20 pairs each cycle
                # Skip if we already have position in this symbol
                if any(p['symbol'] == symbol for p in current_positions):
                    continue
                    
                # Skip if traded too recently
                last_trade = self.last_trade_time.get(f"{user_id}:{symbol}")
                if last_trade and (datetime.utcnow() - last_trade).seconds < self.min_trade_interval_seconds:
                    continue
                    
                # Analyze and potentially trade
                signal = await self._analyze_for_trade(symbol, client)
                symbols_analyzed += 1
                
                if signal:
                    logger.info(f"{symbol}: {signal.action.upper()} signal, confidence={signal.confidence:.0f}%, strategy={signal.strategy}")
                    
                    if signal.action != 'hold' and signal.confidence >= self.min_confidence:
                        signals_found += 1
                        logger.info(f"EXECUTING TRADE: {symbol} {signal.action.upper()}")
                        await self._execute_trade(user_id, client, signal, available_usdt, total_equity)
            
            if symbols_analyzed > 0:
                logger.info(f"Analyzed {symbols_analyzed} pairs, found {signals_found} tradeable signals")
                    
    async def _analyze_for_trade(self, symbol: str, client: BybitV5Client) -> Optional[TradeSignal]:
        """Analyze a symbol and generate trading signal"""
        
        try:
            # Get real-time ticker
            ticker_result = await client.get_tickers(symbol=symbol)
            if not ticker_result.get('success'):
                return None
                
            tickers = ticker_result.get('data', {}).get('list', [])
            if not tickers:
                return None
                
            ticker = tickers[0]
            last_price = float(ticker.get('lastPrice', 0))
            price_change_24h = float(ticker.get('price24hPcnt', 0)) * 100
            volume_24h = float(ticker.get('volume24h', 0))
            funding_rate = float(ticker.get('fundingRate', 0)) * 100
            
            # Get market regime from Redis (set by strategy planner)
            regime_data = await self.redis_client.hgetall(f"regime:{symbol}")
            
            if regime_data:
                regime = regime_data.get(b'regime', b'sideways').decode()
                volatility = float(regime_data.get(b'volatility', b'1.5').decode())
                trend = regime_data.get(b'trend', b'sideways').decode()
                rsi = float(regime_data.get(b'rsi', b'50').decode())
            else:
                # Infer from price action
                if price_change_24h > 5:
                    regime = 'bull_trend'
                    trend = 'bullish'
                elif price_change_24h < -5:
                    regime = 'bear_trend'
                    trend = 'bearish'
                else:
                    regime = 'sideways'
                    trend = 'sideways'
                volatility = abs(price_change_24h) / 2
                rsi = 50 + (price_change_24h * 2)  # Rough estimate
                
            # Get best strategy from learning engine
            best_strategy, q_value = self.learning_engine.get_best_strategy(regime)
            confidence = self.learning_engine.get_strategy_confidence(regime, best_strategy)
            
            # Generate signal based on strategy
            action = 'hold'
            reasoning = ""
            
            if best_strategy == 'momentum':
                if trend == 'bullish' and rsi < 70:
                    action = 'buy'
                    reasoning = f"Momentum strategy: Bullish trend, RSI={rsi:.0f}"
                elif trend == 'bearish' and rsi > 30:
                    action = 'sell'
                    reasoning = f"Momentum strategy: Bearish trend, RSI={rsi:.0f}"
                    
            elif best_strategy == 'mean_reversion':
                if rsi < 30:
                    action = 'buy'
                    reasoning = f"Mean reversion: Oversold RSI={rsi:.0f}"
                elif rsi > 70:
                    action = 'sell'
                    reasoning = f"Mean reversion: Overbought RSI={rsi:.0f}"
                    
            elif best_strategy == 'breakout':
                if price_change_24h > 3 and volume_24h > 100000:
                    action = 'buy'
                    reasoning = f"Breakout: +{price_change_24h:.1f}% with high volume"
                elif price_change_24h < -3 and volume_24h > 100000:
                    action = 'sell'
                    reasoning = f"Breakdown: {price_change_24h:.1f}% with high volume"
                    
            elif best_strategy == 'scalping':
                # Quick trades based on micro movements
                if abs(funding_rate) > 0.01:
                    if funding_rate > 0:
                        action = 'sell'  # Too many longs, short
                        reasoning = f"Scalping: High funding rate {funding_rate:.3f}%"
                    else:
                        action = 'buy'  # Too many shorts, long
                        reasoning = f"Scalping: Negative funding {funding_rate:.3f}%"
                        
            elif best_strategy == 'grid':
                # Range trading
                if volatility < 2 and rsi < 40:
                    action = 'buy'
                    reasoning = f"Grid: Low volatility, RSI={rsi:.0f}"
                elif volatility < 2 and rsi > 60:
                    action = 'sell'
                    reasoning = f"Grid: Low volatility, RSI={rsi:.0f}"
                    
            if action == 'hold':
                return None
                
            # Calculate stop loss and take profit - SAFE PROFIT MODE
            # Very tight stops for consistent small gains
            
            if action == 'buy':
                stop_loss = last_price * (1 - self.stop_loss_percent / 100)  # -0.3%
                take_profit = last_price * (1 + self.take_profit_percent / 100)  # +0.5%
            else:
                stop_loss = last_price * (1 + self.stop_loss_percent / 100)  # +0.3%
                take_profit = last_price * (1 - self.take_profit_percent / 100)  # -0.5%
                
            # Position size based on confidence
            position_size_percent = min(
                self.max_position_percent,
                (confidence / 100) * self.max_position_percent
            )
            
            return TradeSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                strategy=best_strategy,
                regime=regime,
                entry_price=last_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_percent=position_size_percent,
                reasoning=reasoning,
            )
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return None
            
    async def _execute_trade(
        self, 
        user_id: str, 
        client: BybitV5Client, 
        signal: TradeSignal,
        available_usdt: float,
        total_equity: float
    ):
        """Execute a trade based on signal"""
        
        try:
            # Calculate position size
            trade_value = total_equity * (signal.position_size_percent / 100)
            trade_value = min(trade_value, available_usdt * 0.95)  # Keep 5% buffer
            
            if trade_value < 10:  # Minimum $10 trade
                return
                
            quantity = trade_value / signal.entry_price
            
            # Round quantity to appropriate precision
            quantity = round(quantity, 4)
            
            # Determine side
            side = 'Buy' if signal.action == 'buy' else 'Sell'
            
            logger.info(f"Executing {side} {signal.symbol}: qty={quantity}, price≈{signal.entry_price}")
            
            # Place order (qty must be string for Bybit API)
            order_result = await client.place_order(
                symbol=signal.symbol,
                side=side,
                order_type='Market',
                qty=str(quantity),
            )
            
            if order_result.get('success'):
                # Record trade
                trade_record = {
                    'user_id': user_id,
                    'symbol': signal.symbol,
                    'side': side.lower(),
                    'quantity': quantity,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'strategy': signal.strategy,
                    'regime': signal.regime,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning,
                    'timestamp': datetime.utcnow().isoformat(),
                }
                
                # Store in Redis
                await self.redis_client.lpush(
                    f"trades:active:{user_id}",
                    json.dumps(trade_record)
                )
                
                # Update last trade time
                self.last_trade_time[f"{user_id}:{signal.symbol}"] = datetime.utcnow()
                
                # Log trade
                await self._log_trade(user_id, trade_record)
                
                logger.info(f"Trade executed: {signal.symbol} {side} @ ${signal.entry_price:.2f}")
                
            else:
                logger.warning(f"Trade failed: {order_result.get('error')}")
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            
    async def _check_position_exit(
        self, 
        user_id: str, 
        client: BybitV5Client, 
        position: Dict,
        total_equity: float
    ):
        """Check if position should be closed"""
        
        symbol = position['symbol']
        unrealized_pnl = position['unrealizedPnl']
        entry_price = position['entryPrice']
        
        # Get current price
        ticker_result = await client.get_tickers(symbol=symbol)
        if not ticker_result.get('success'):
            return
            
        tickers = ticker_result.get('data', {}).get('list', [])
        if not tickers:
            return
            
        current_price = float(tickers[0].get('lastPrice', 0))
        
        # Calculate PnL percentage
        pnl_percent = (unrealized_pnl / total_equity) * 100 if total_equity > 0 else 0
        
        should_close = False
        close_reason = ""
        
        # Calculate price change from entry
        if position['side'].lower() == 'buy':
            price_change_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            price_change_percent = ((entry_price - current_price) / entry_price) * 100
        
        # SAFE PROFIT MODE: Small consistent wins, fast exits on loss
        
        # Take profit: +0.5% or more - CASH OUT!
        if price_change_percent >= self.take_profit_percent:
            should_close = True
            close_reason = f"PROFIT SECURED: +{price_change_percent:.2f}% (target: {self.take_profit_percent}%)"
            
        # Stop loss: -0.3% - EXIT IMMEDIATELY
        elif price_change_percent <= -self.stop_loss_percent:
            should_close = True
            close_reason = f"STOP LOSS: {price_change_percent:.2f}% (limit: -{self.stop_loss_percent}%)"
            
        # Trailing stop: If we hit +0.3% profit and it's dropping, exit
        elif price_change_percent >= self.trailing_activate:
            # We're in profit, but check if price is reversing
            # Store peak profit in Redis to track
            peak_key = f"peak:{position['symbol']}:{position['side']}"
            peak_profit = await self.redis_client.get(peak_key)
            
            if peak_profit:
                peak = float(peak_profit)
                if price_change_percent > peak:
                    await self.redis_client.set(peak_key, str(price_change_percent))
                elif peak - price_change_percent >= self.trailing_stop:
                    # Dropped 0.15% from peak - exit with profit
                    should_close = True
                    close_reason = f"TRAILING STOP: Peak was +{peak:.2f}%, now +{price_change_percent:.2f}%"
                    await self.redis_client.delete(peak_key)
            else:
                await self.redis_client.set(peak_key, str(price_change_percent))
                
        # If position is slightly negative but not at stop loss yet, check trend
        elif price_change_percent < 0:
            # Get quick price trend (is it still falling?)
            # If momentum is negative, exit early before hitting stop loss
            ticker_result = await client.get_tickers(symbol=position['symbol'])
            if ticker_result.get('success'):
                tickers = ticker_result.get('data', {}).get('list', [])
                if tickers:
                    price_24h_change = float(tickers[0].get('price24hPcnt', 0)) * 100
                    # If market is dumping hard, exit immediately
                    if abs(price_24h_change) > 5 and price_change_percent < -0.1:
                        should_close = True
                        close_reason = f"EMERGENCY EXIT: Market crash -{price_24h_change:.1f}%"
                    
        if should_close:
            await self._close_position(user_id, client, position, close_reason)
            
    async def _close_position(
        self, 
        user_id: str, 
        client: BybitV5Client, 
        position: Dict,
        reason: str
    ):
        """Close a position"""
        
        try:
            symbol = position['symbol']
            size = position['size']
            side = 'Sell' if position['side'].lower() == 'buy' else 'Buy'
            
            order_result = await client.place_order(
                symbol=symbol,
                side=side,
                order_type='Market',
                qty=str(size),
                reduce_only=True,
            )
            
            if order_result.get('success'):
                pnl = position['unrealizedPnl']
                
                logger.info(f"Position closed: {symbol} PnL=${pnl:.2f} - {reason}")
                
                # Record for learning
                await self._record_closed_trade(user_id, position, reason)
                
        except Exception as e:
            logger.error(f"Close position error: {e}")
            
    async def _record_closed_trade(self, user_id: str, position: Dict, reason: str):
        """Record closed trade for learning"""
        
        # Get original trade details from Redis
        active_trades = await self.redis_client.lrange(f"trades:active:{user_id}", 0, -1)
        
        original_trade = None
        for trade_json in active_trades:
            trade = json.loads(trade_json)
            if trade.get('symbol') == position['symbol']:
                original_trade = trade
                break
                
        if original_trade:
            # Calculate actual results
            entry_price = position['entryPrice']
            exit_price = float(position.get('markPrice', entry_price))
            pnl = position['unrealizedPnl']
            
            # Calculate hold time
            try:
                entry_time = datetime.fromisoformat(original_trade['timestamp'])
                hold_seconds = int((datetime.utcnow() - entry_time).total_seconds())
            except:
                hold_seconds = 0
                
            # Create trade outcome for learning
            outcome = TradeOutcome(
                symbol=position['symbol'],
                strategy=original_trade.get('strategy', 'unknown'),
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=position['size'],
                side='long' if position['side'].lower() == 'buy' else 'short',
                pnl=pnl,
                pnl_percent=(pnl / entry_price / position['size']) * 100 if position['size'] > 0 else 0,
                hold_time_seconds=hold_seconds,
                market_regime=original_trade.get('regime', 'unknown'),
                volatility_at_entry=1.5,  # Would be stored from original trade
                sentiment_at_entry=0,
                timestamp=datetime.utcnow().isoformat(),
            )
            
            # Update learning engine
            if self.learning_engine:
                await self.learning_engine.update_from_trade(outcome)
                
            # Store completed trade
            await self.redis_client.lpush(
                f"trades:completed:{user_id}",
                json.dumps({
                    **original_trade,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'close_reason': reason,
                    'closed_at': datetime.utcnow().isoformat(),
                })
            )
            await self.redis_client.ltrim(f"trades:completed:{user_id}", 0, 999)
            
    async def _log_trade(self, user_id: str, trade: Dict):
        """Log trade for monitoring"""
        await self.redis_client.lpush(
            'trades:log',
            json.dumps({
                'user_id': user_id,
                **trade,
            })
        )
        await self.redis_client.ltrim('trades:log', 0, 999)
        
    async def _log_trading_status(self):
        """Log current trading status"""
        active_users = len(self.user_clients)
        total_trades = await self.redis_client.llen('trades:log')
        
        logger.info(f"Trading status: {active_users} users, {total_trades} trades logged")
        
        # Store status
        await self.redis_client.hset(
            'trading:status',
            mapping={
                'active_users': str(active_users),
                'is_running': '1' if self.is_running else '0',
                'last_update': datetime.utcnow().isoformat(),
            }
        )


# Global instance
autonomous_trader = AutonomousTrader()

