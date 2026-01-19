"""
SENTINEL AI - Ultimate Autonomous Trading System v2.0

This is the PROFESSIONAL version that integrates:
- RegimeDetector: Knows WHEN to trade
- EdgeEstimator: Knows IF there's an edge
- PositionSizer: Knows HOW MUCH to risk (Kelly)
- MarketScanner: Sees ALL 500+ pairs
- AICoordinator: The brain that combines everything

KEY PRINCIPLES:
1. Trade EVERYTHING on Bybit IF there's edge
2. Dynamic sizing based on confidence
3. Regime-aware strategy selection
4. Continuous learning from outcomes
5. Hard risk limits NEVER exceeded

This is what hedge funds use, not retail bot BS.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from loguru import logger
import redis.asyncio as redis
import json

from config import settings
from services.bybit_client import BybitV5Client
from services.learning_engine import LearningEngine, TradeOutcome
from services.regime_detector import RegimeDetector, RegimeState
from services.edge_estimator import EdgeEstimator, EdgeScore
from services.position_sizer import PositionSizer, PositionSize
from services.market_scanner import MarketScanner, TradingOpportunity

# V3 Advanced ML Components
from services.xgboost_classifier import xgboost_classifier
from services.finbert_sentiment import finbert_sentiment
from services.data_collector import data_collector, TradeRecord
from services.training_data_manager import training_data_manager
from services.crypto_sentiment import crypto_sentiment
from services.price_predictor import price_predictor
from services.capital_allocator import capital_allocator, MarketOpportunity


def safe_float(val, default=0.0):
    """Safely convert value to float"""
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


@dataclass
class ActivePosition:
    """Track active position with all metadata"""
    symbol: str
    side: str  # 'Buy' or 'Sell'
    size: float
    entry_price: float
    entry_time: datetime
    
    # Edge data at entry
    entry_edge: float
    entry_confidence: float
    entry_regime: str
    
    # Tracking
    peak_price: float  # Highest seen (for longs)
    trough_price: float  # Lowest seen (for shorts)
    peak_pnl_percent: float
    
    # Exit strategy
    stop_loss_price: float
    take_profit_price: float
    trailing_active: bool = False
    
    # Sizing
    position_value: float = 0.0
    kelly_fraction: float = 0.0


class AutonomousTraderV2:
    """
    ULTIMATE Autonomous Trading System
    
    This is THE BEST possible implementation:
    - Scans ALL pairs on Bybit
    - Uses edge-based position sizing
    - Respects regime signals
    - Has hard risk limits
    - Learns from every trade
    
    NOT for the faint of heart.
    """
    
    def __init__(self):
        self.is_running = False
        self.redis_client = None
        
        # Core components (injected)
        self.regime_detector: Optional[RegimeDetector] = None
        self.edge_estimator: Optional[EdgeEstimator] = None
        self.position_sizer: Optional[PositionSizer] = None
        self.market_scanner: Optional[MarketScanner] = None
        self.learning_engine: Optional[LearningEngine] = None
        self.ai_coordinator = None  # Legacy support
        
        # Connected exchange clients per user
        self.user_clients: Dict[str, BybitV5Client] = {}
        
        # Active positions with full metadata
        self.active_positions: Dict[str, Dict[str, ActivePosition]] = {}  # user_id -> symbol -> position
        
        # === CONFIGURATION ===
        
        # Trading frequency - FAST for responsive exits
        self.scan_interval = 30  # Seconds between full scans
        self.position_check_interval = 1  # Check positions EVERY SECOND for fast exits
        
        # Entry filters (defaults, will be overridden by settings)
        self.min_edge = 0.15  # Minimum edge to consider trade
        self.min_confidence = 60  # Minimum confidence
        
        # Exit strategy
        self.min_profit_to_trail = 0.8  # % profit before trailing
        self.trail_from_peak = 1.0  # Trail by 1% from peak
        self.emergency_stop_loss = 1.5  # Hard stop at -1.5%
        self.take_profit = 3.0  # Take profit at +3%
        
        # Risk limits (0 = unlimited positions)
        self.max_open_positions = 0  # Unlimited by default
        self.max_exposure_percent = 100  # 100% = can use entire budget
        self.max_daily_drawdown = 3.0
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'opportunities_scanned': 0,
            'trades_rejected_low_edge': 0,
            'trades_rejected_regime': 0,
            'trades_rejected_risk': 0
        }
        
    async def initialize(
        self,
        regime_detector: RegimeDetector,
        edge_estimator: EdgeEstimator,
        position_sizer: PositionSizer,
        market_scanner: MarketScanner,
        learning_engine: LearningEngine,
        ai_coordinator = None
    ):
        """Initialize with all components"""
        logger.info("Initializing Ultimate Autonomous Trader v2.0...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        
        self.regime_detector = regime_detector
        self.edge_estimator = edge_estimator
        self.position_sizer = position_sizer
        self.market_scanner = market_scanner
        self.learning_engine = learning_engine
        self.ai_coordinator = ai_coordinator
        
        self.is_running = True
        
        # Load settings
        await self._load_settings()
        
        # Load stats
        await self._load_stats()
        
        logger.info("Ultimate Autonomous Trader v2.0 initialized!")
        logger.info(f"Components: RegimeDetector={self.regime_detector is not None}, "
                   f"EdgeEstimator={self.edge_estimator is not None}, "
                   f"PositionSizer={self.position_sizer is not None}, "
                   f"MarketScanner={self.market_scanner is not None}")
        
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Ultimate Autonomous Trader...")
        self.is_running = False
        
        await self._save_stats()
        
        for user_id, client in self.user_clients.items():
            try:
                await client.close()
            except:
                pass
                
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def connect_user(self, user_id: str, api_key: str, api_secret: str, testnet: bool = False) -> bool:
        """Connect user's exchange account"""
        try:
            client = BybitV5Client(api_key, api_secret, testnet)
            result = await client.test_connection()
            
            if result.get('success'):
                self.user_clients[user_id] = client
                self.active_positions[user_id] = {}
                logger.info(f"User {user_id} connected for Ultimate trading")
                return True
            else:
                await client.close()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect user {user_id}: {e}")
            return False
            
    async def disconnect_user(self, user_id: str):
        """Disconnect user"""
        if user_id in self.user_clients:
            await self.user_clients[user_id].close()
            del self.user_clients[user_id]
            if user_id in self.active_positions:
                del self.active_positions[user_id]
            logger.info(f"User {user_id} disconnected")
            
    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("=" * 60)
        logger.info("🚀 TRADING LOOP STARTING!")
        logger.info(f"📊 Settings: TP={self.take_profit}%, SL={self.emergency_stop_loss}%")
        logger.info("=" * 60)
        
        cycle = 0
        
        while self.is_running:
            try:
                cycle += 1
                cycle_start = datetime.utcnow()
                
                # Log EVERY cycle for debugging
                connected_users = len(self.user_clients)
                total_positions = sum(len(p) for p in self.active_positions.values())
                
                # Log every 10 cycles (~10 seconds) with clear visibility
                if cycle % 10 == 0 or cycle <= 5:
                    logger.info(f"🔄 Cycle {cycle} | Users: {connected_users} | Pos: {total_positions} | TP={self.take_profit}% SL={self.emergency_stop_loss}%")
                    # Also write to Redis for dashboard console
                    await self._log_to_console(f"Cycle {cycle}: Checking {total_positions} positions (TP={self.take_profit}%, SL={self.emergency_stop_loss}%)")
                
                # Process each connected user
                if not self.user_clients:
                    if cycle % 10 == 0:
                        logger.warning("⚠️ NO USERS CONNECTED - waiting for user to connect via dashboard...")
                        
                for user_id, client in list(self.user_clients.items()):
                    try:
                        await self._process_user(user_id, client)
                    except Exception as e:
                        logger.error(f"Error processing user {user_id}: {e}")
                        
                # Reload settings periodically (every ~30 seconds)
                if cycle % 10 == 0:
                    await self._load_settings()
                
                # Log status periodically
                if cycle % 100 == 0:
                    await self._log_status()
                    
                # Calculate sleep time
                elapsed = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time = max(self.position_check_interval - elapsed, 1)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(30)
                
    async def _process_user(self, user_id: str, client: BybitV5Client):
        """Process trading for one user"""
        
        # 1. Get wallet balance
        wallet = await self._get_wallet(client)
        if wallet['total_equity'] < 20:  # Min $20
            return
            
        # 2. Sync positions from exchange
        await self._sync_positions(user_id, client)
        
        # 3. Check existing positions for exit - BATCH fetch all tickers first for SPEED
        positions_list = list(self.active_positions.get(user_id, {}).items())
        if positions_list:
            logger.info(f"⚡ Fast-checking {len(positions_list)} positions (TP={self.take_profit}%, SL={self.emergency_stop_loss}%)")
            
            # BATCH: Get all tickers in ONE API call
            all_tickers = await self._get_all_tickers(client)
            
            # Check each position with pre-fetched prices
            for symbol, position in positions_list:
                await self._check_position_exit_fast(user_id, client, position, wallet, all_tickers)
            
        # 4. Look for new opportunities (if room)
        # max_open_positions = 0 means unlimited
        num_positions = len(self.active_positions.get(user_id, {}))
        can_open_more = self.max_open_positions == 0 or num_positions < self.max_open_positions
        
        if can_open_more:
            await self._find_opportunities(user_id, client, wallet)
            
    async def _get_wallet(self, client: BybitV5Client) -> Dict:
        """Get wallet balance"""
        result = await client.get_wallet_balance()
        
        if not result.get('success'):
            return {'total_equity': 0, 'available': 0}
            
        data = result.get('data', {})
        total_equity = 0
        available = 0
        
        for account in data.get('list', []):
            total_equity = safe_float(account.get('totalEquity'))
            available = safe_float(account.get('totalAvailableBalance'))
            
            for coin in account.get('coin', []):
                if coin.get('coin') == 'USDT':
                    avail_withdraw = safe_float(coin.get('availableToWithdraw'))
                    available = max(available, avail_withdraw)
                    break
                    
        # Store for position sizer
        await self.redis_client.set('wallet:equity', str(total_equity))
        
        return {
            'total_equity': total_equity,
            'available': available
        }
        
    async def _sync_positions(self, user_id: str, client: BybitV5Client):
        """Sync active positions with exchange"""
        result = await client.get_positions()
        
        if not result.get('success'):
            logger.warning(f"Failed to get positions: {result}")
            return
            
        exchange_positions = set()
        positions_list = result.get('data', {}).get('list', [])
        logger.debug(f"📋 Exchange returned {len(positions_list)} positions")
        
        for pos in positions_list:
            size = safe_float(pos.get('size'))
            if size > 0:
                symbol = pos.get('symbol')
                exchange_positions.add(symbol)
                
                # If not tracked, add it
                if user_id not in self.active_positions:
                    self.active_positions[user_id] = {}
                    
                if symbol not in self.active_positions[user_id]:
                    # New position detected (maybe from manual trade)
                    entry_price = safe_float(pos.get('avgPrice'))
                    mark_price = safe_float(pos.get('markPrice', entry_price))
                    position_value = size * entry_price
                    side = pos.get('side', 'Buy')
                    
                    # Calculate current P&L to set proper peak tracking
                    if side == 'Buy':
                        current_pnl = ((mark_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                        peak_price = max(entry_price, mark_price)
                        trough_price = min(entry_price, mark_price)
                    else:
                        current_pnl = ((entry_price - mark_price) / entry_price) * 100 if entry_price > 0 else 0
                        peak_price = min(entry_price, mark_price)  # For shorts, lower is better
                        trough_price = max(entry_price, mark_price)
                    
                    # Peak P&L is max of 0 and current (we don't know historical peak)
                    peak_pnl = max(0, current_pnl)
                    
                    # Calculate position value
                    position_value = size * entry_price
                    
                    self.active_positions[user_id][symbol] = ActivePosition(
                        symbol=symbol,
                        side=side,
                        size=size,
                        entry_price=entry_price,
                        entry_time=datetime.utcnow(),
                        entry_edge=0.0,
                        entry_confidence=0.0,
                        entry_regime='unknown',
                        peak_price=peak_price,
                        trough_price=trough_price,
                        peak_pnl_percent=peak_pnl,
                        stop_loss_price=entry_price * (1 - self.emergency_stop_loss / 100) if side == 'Buy' else entry_price * (1 + self.emergency_stop_loss / 100),
                        take_profit_price=entry_price * (1 + self.take_profit / 100) if side == 'Buy' else entry_price * (1 - self.take_profit / 100),
                        trailing_active=current_pnl >= self.min_profit_to_trail,  # Already in profit?
                        position_value=position_value
                    )
                    # Also register in position_sizer so it knows about this position
                    await self.position_sizer.register_position(symbol, position_value)
                    logger.info(f"Synced position: {symbol} | Entry: ${entry_price:.4f} | Mark: ${mark_price:.4f} | P&L: {current_pnl:.2f}%")
                else:
                    # Update size if changed
                    self.active_positions[user_id][symbol].size = size
                    
        # Remove closed positions
        if user_id in self.active_positions:
            for symbol in list(self.active_positions[user_id].keys()):
                if symbol not in exchange_positions:
                    del self.active_positions[user_id][symbol]
                    # Also remove from position_sizer
                    await self.position_sizer.close_position(symbol, 0)
                    logger.info(f"Position {symbol} closed externally, removed from tracker")
                    
    async def _check_position_exit(self, user_id: str, client: BybitV5Client,
                                    position: ActivePosition, wallet: Dict):
        """Check if position should be exited"""
        try:
            # Get current price
            ticker = await self._get_ticker(client, position.symbol)
            if not ticker:
                logger.warning(f"⚠️ No ticker for {position.symbol} - cannot check exit")
                return
                
            current_price = ticker['last_price']
            if current_price <= 0:
                logger.warning(f"⚠️ Invalid price {current_price} for {position.symbol}")
                return
            
            # Calculate P&L
            if position.side == 'Buy':
                pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                # Update peak
                if current_price > position.peak_price:
                    position.peak_price = current_price
                    position.peak_pnl_percent = pnl_percent
            else:  # Short
                pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
                # Update trough
                if current_price < position.trough_price:
                    position.trough_price = current_price
                    position.peak_pnl_percent = pnl_percent
                    
            # === EXIT LOGIC ===
            should_exit = False
            exit_reason = ""
            
            # Log current state for EVERY position check - CRITICAL FOR DEBUGGING
            logger.info(f"🔎 CHECK {position.symbol}: Side={position.side}, Price=${current_price:.6f}, Entry=${position.entry_price:.6f}, P&L={pnl_percent:+.2f}%, TP={self.take_profit}%, SL=-{self.emergency_stop_loss}%")
            
            if pnl_percent >= self.take_profit:
                logger.info(f"✅ {position.symbol}: P&L={pnl_percent:+.2f}% >= TP={self.take_profit}% - TRIGGERING TAKE PROFIT!")
            elif pnl_percent <= -self.emergency_stop_loss:
                logger.info(f"❌ {position.symbol}: P&L={pnl_percent:+.2f}% <= SL=-{self.emergency_stop_loss}% - TRIGGERING STOP LOSS!")
            elif pnl_percent >= self.take_profit * 0.7:
                logger.info(f"⏳ {position.symbol}: P&L={pnl_percent:+.2f}% approaching TP={self.take_profit}%")
            
            # 1. STOP LOSS
            if pnl_percent <= -self.emergency_stop_loss:
                should_exit = True
                exit_reason = f"Stop loss hit ({pnl_percent:.2f}%)"
                logger.info(f"🛑 STOP LOSS: {position.symbol} at {pnl_percent:+.2f}%")
                
            # 2. TAKE PROFIT - Exit when profit reaches target
            elif pnl_percent >= self.take_profit:
                should_exit = True
                exit_reason = f"Take profit reached ({pnl_percent:.2f}% >= {self.take_profit}%)"
                logger.info(f"🎯 TAKE PROFIT: {position.symbol} at {pnl_percent:+.2f}% >= TP {self.take_profit}% - SELLING NOW!")
                
            # 3. TRAILING STOP
            # Activates when profit >= min_profit_to_trail (e.g., 0.8%)
            # But only EXITS if we've been above take_profit OR profit stays acceptable
            elif pnl_percent >= self.min_profit_to_trail:
                position.trailing_active = True
                
                # Calculate drop from peak
                if position.side == 'Buy':
                    drop_from_peak = ((position.peak_price - current_price) / position.peak_price) * 100
                else:
                    drop_from_peak = ((current_price - position.trough_price) / position.trough_price) * 100
                    
                # Only exit via trailing if drop is significant AND either:
                # 1. Peak profit was at/above take profit (we hit TP, now trailing down)
                # 2. Current profit is still at least min_profit_to_trail (lock in some profit)
                if drop_from_peak >= self.trail_from_peak:
                    if position.peak_pnl_percent >= self.take_profit:
                        # We reached TP at peak, now trailing to lock profits
                        should_exit = True
                        exit_reason = f"Trailing from TP (peak: {position.peak_pnl_percent:.2f}%, now: {pnl_percent:.2f}%)"
                    elif pnl_percent >= self.min_profit_to_trail:
                        # Didn't reach TP but still in profit - lock it in
                        should_exit = True
                        exit_reason = f"Trailing stop (peak: {position.peak_pnl_percent:.2f}%, now: {pnl_percent:.2f}%)"
                    
            # 4. REGIME CHANGED TO AVOID
            if not should_exit:
                regime = await self.regime_detector.detect_regime(position.symbol)
                if regime.recommended_action == 'avoid' and pnl_percent > 0:
                    should_exit = True
                    exit_reason = f"Regime changed to avoid (locking {pnl_percent:.2f}% profit)"
                    
            # === EXECUTE EXIT ===
            if should_exit:
                logger.info(f"💰 CLOSING {position.symbol}: {exit_reason}")
                await self._close_position(user_id, client, position, pnl_percent, exit_reason)
            else:
                # Log why we're NOT exiting if position has significant P&L
                if abs(pnl_percent) > 1.0:
                    logger.info(f"⏸️ HOLD {position.symbol}: P&L={pnl_percent:+.2f}% (TP={self.take_profit}%, SL=-{self.emergency_stop_loss}%) - No exit trigger")
                
        except Exception as e:
            logger.error(f"Exit check error for {position.symbol}: {e}")
            
    async def _close_position(self, user_id: str, client: BybitV5Client,
                              position: ActivePosition, pnl_percent: float, reason: str):
        """Close a position"""
        try:
            logger.info(f"💰 EXECUTING CLOSE: {position.symbol} | Side: {position.side} | Size: {position.size} | Reason: {reason}")
            
            # Determine close side (opposite of position)
            close_side = 'Sell' if position.side == 'Buy' else 'Buy'
            
            # FIXED: qty must be string, add category parameter
            result = await client.place_order(
                category="linear",  # Crypto perpetuals
                symbol=position.symbol,
                side=close_side,
                order_type='Market',
                qty=str(position.size),  # Must be string!
                reduce_only=True
            )
            
            logger.info(f"📤 Order result for {position.symbol}: {result}")
            
            if result.get('success'):
                won = pnl_percent > 0
                pnl_value = position.position_value * (pnl_percent / 100)
                
                logger.info(f"CLOSED {position.symbol}: {reason} | P&L: {pnl_percent:+.2f}% (${pnl_value:+.2f})")
                
                # Update stats
                self.stats['total_trades'] += 1
                if won:
                    self.stats['winning_trades'] += 1
                self.stats['total_pnl'] += pnl_value
                
                # Record in position sizer
                await self.position_sizer.close_position(position.symbol, pnl_value)
                
                # Record in edge estimator for calibration
                await self.edge_estimator.record_outcome(position.symbol, position.entry_edge, won)
                
                # Record in market scanner
                await self.market_scanner.record_trade_result(position.symbol, won, pnl_value)
                
                # Record in learning engine
                if self.learning_engine:
                    exit_price = position.entry_price * (1 + pnl_percent/100)
                    outcome = TradeOutcome(
                        symbol=position.symbol,
                        strategy='edge_based',
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        quantity=position.size,
                        side='long' if position.side == 'Buy' else 'short',
                        pnl=pnl_value,
                        pnl_percent=pnl_percent,
                        hold_time_seconds=int((datetime.utcnow() - position.entry_time).total_seconds()),
                        market_regime=position.entry_regime,
                        volatility_at_entry=0.0,  # Not tracked
                        sentiment_at_entry=0.0,   # Not tracked
                        timestamp=datetime.utcnow().isoformat()
                    )
                    await self.learning_engine.update_from_trade(outcome)
                    
                # Remove from active positions
                if user_id in self.active_positions:
                    if position.symbol in self.active_positions[user_id]:
                        del self.active_positions[user_id][position.symbol]
                        
                # Store trade for dashboard and data collection (V3)
                await self._store_trade_event(position.symbol, 'closed', pnl_percent, reason, position)
                
            else:
                logger.error(f"Failed to close {position.symbol}: {result}")
                
        except Exception as e:
            logger.error(f"Close position error: {e}")
            
    async def _find_opportunities(self, user_id: str, client: BybitV5Client, wallet: Dict):
        """Find and execute new trading opportunities"""
        try:
            # Get opportunities from scanner
            opportunities = await self.market_scanner.get_tradeable_opportunities()
            
            self.stats['opportunities_scanned'] += len(opportunities)
            
            for opp in opportunities:
                # Skip if we have max positions (0 = unlimited)
                num_positions = len(self.active_positions.get(user_id, {}))
                if self.max_open_positions > 0 and num_positions >= self.max_open_positions:
                    break
                    
                # Skip if already in position
                if opp.symbol in self.active_positions.get(user_id, {}):
                    continue
                    
                # Validate the opportunity
                should_trade, reason = await self._validate_opportunity(opp, wallet)
                
                if should_trade:
                    await self._execute_trade(user_id, client, opp, wallet)
                else:
                    logger.debug(f"Skipped {opp.symbol}: {reason}")
                    
        except Exception as e:
            logger.error(f"Find opportunities error: {e}")
            
    async def _validate_opportunity(self, opp: TradingOpportunity, wallet: Dict) -> Tuple[bool, str]:
        """
        SUPERIOR validation using ALL AI models
        
        Checks:
        1. Basic edge/confidence
        2. XGBoost ML classification
        3. CryptoBERT sentiment (crypto-specific)
        4. Price predictor (multi-timeframe)
        5. Capital allocator (unified budget)
        6. Regime detection
        7. Position sizing
        """
        
        # 1. Edge check
        if opp.edge_score < self.min_edge:
            self.stats['trades_rejected_low_edge'] += 1
            return False, f"Edge too low ({opp.edge_score:.2f} < {self.min_edge})"
            
        # 2. Confidence check
        if opp.confidence < self.min_confidence:
            self.stats['trades_rejected_low_edge'] += 1
            return False, f"Confidence too low ({opp.confidence:.1f} < {self.min_confidence})"
            
        # 3. XGBoost ML Classification
        try:
            feature_data = await self.redis_client.get(f"features:{opp.symbol}")
            if feature_data:
                features = json.loads(feature_data)
                features['symbol'] = opp.symbol
                
                xgb_result = await xgboost_classifier.classify(features)
                
                expected_signal = 'buy' if opp.direction == 'long' else 'sell'
                if xgb_result.signal != expected_signal and xgb_result.confidence > 60:
                    self.stats['trades_rejected_low_edge'] += 1
                    return False, f"XGBoost disagrees ({xgb_result.signal} vs {expected_signal})"
                    
                if xgb_result.signal == expected_signal and xgb_result.confidence < 50:
                    self.stats['trades_rejected_low_edge'] += 1
                    return False, f"XGBoost confidence too low ({xgb_result.confidence:.1f}%)"
        except Exception as e:
            logger.debug(f"XGBoost validation skipped: {e}")
            
        # 4. CryptoBERT Sentiment (SUPERIOR - crypto-specific)
        try:
            symbol_sentiment = await crypto_sentiment.get_symbol_sentiment(opp.symbol)
            if symbol_sentiment and symbol_sentiment.get('sample_count', 0) > 3:
                sentiment_score = symbol_sentiment.get('score', 0)
                
                # Block if sentiment strongly disagrees
                if opp.direction == 'long' and sentiment_score < -0.4:
                    self.stats['trades_rejected_regime'] += 1
                    return False, f"CryptoBERT bearish for {opp.symbol} ({sentiment_score:.2f})"
                elif opp.direction == 'short' and sentiment_score > 0.4:
                    self.stats['trades_rejected_regime'] += 1
                    return False, f"CryptoBERT bullish for {opp.symbol} ({sentiment_score:.2f})"
        except Exception as e:
            logger.debug(f"CryptoBERT check skipped: {e}")
            
        # 5. Price Predictor (multi-timeframe consensus)
        try:
            price_signal = await price_predictor.get_trading_signal(opp.symbol)
            if price_signal and price_signal.get('confidence', 0) > 30:
                pred_signal = price_signal.get('signal', 'neutral')
                
                # Must agree with direction
                if opp.direction == 'long' and pred_signal == 'short':
                    if price_signal.get('strength', 0) > 0.5:
                        self.stats['trades_rejected_low_edge'] += 1
                        return False, f"Price predictor says SHORT (strength: {price_signal['strength']:.2f})"
                elif opp.direction == 'short' and pred_signal == 'long':
                    if price_signal.get('strength', 0) > 0.5:
                        self.stats['trades_rejected_low_edge'] += 1
                        return False, f"Price predictor says LONG (strength: {price_signal['strength']:.2f})"
                        
                # Boost edge if predictor strongly agrees
                if (opp.direction == 'long' and pred_signal == 'long') or \
                   (opp.direction == 'short' and pred_signal == 'short'):
                    opp.edge_score = min(1.0, opp.edge_score * (1 + price_signal.get('strength', 0) * 0.5))
        except Exception as e:
            logger.debug(f"Price predictor check skipped: {e}")
            
        # 6. Capital Allocator - Add as opportunity for unified allocation
        try:
            asset_class = capital_allocator.classify_symbol(opp.symbol)
            market_opp = MarketOpportunity(
                symbol=opp.symbol,
                asset_class=asset_class,
                edge_score=opp.edge_score,
                confidence=opp.confidence,
                direction=opp.direction,
                volatility=getattr(opp, 'volatility', 1.5),
                liquidity=getattr(opp, 'liquidity', 50),
                correlation_btc=getattr(opp, 'correlation_btc', 0.5)
            )
            await capital_allocator.add_opportunity(market_opp)
        except Exception as e:
            logger.debug(f"Capital allocator skipped: {e}")
            
        # 7. Regime check
        if opp.regime_action == 'avoid':
            self.stats['trades_rejected_regime'] += 1
            return False, f"Regime recommends avoid"
            
        # 8. Risk check via position sizer
        if not opp.edge_data:
            return False, "No edge data"
            
        position_size = await self.position_sizer.calculate_position_size(
            symbol=opp.symbol,
            direction=opp.direction,
            edge_score=opp.edge_score,
            win_probability=opp.edge_data.win_probability,
            risk_reward=opp.edge_data.risk_reward_ratio,
            kelly_fraction=opp.edge_data.kelly_fraction,
            regime_action=opp.regime_action,
            current_price=opp.current_price,
            wallet_balance=wallet['total_equity']
        )
        
        if not position_size.is_within_limits:
            self.stats['trades_rejected_risk'] += 1
            return False, position_size.limit_reason
            
        # Store position size for execution
        opp.edge_data._position_size = position_size
        
        return True, "SUPERIOR: Passed ALL AI checks ✅"
        
    async def _execute_trade(self, user_id: str, client: BybitV5Client,
                             opp: TradingOpportunity, wallet: Dict):
        """Execute a trade"""
        try:
            edge_data = opp.edge_data
            pos_size: PositionSize = getattr(edge_data, '_position_size', None)
            
            if not pos_size or pos_size.position_value_usdt < 5:
                logger.debug(f"Position too small for {opp.symbol}")
                return
                
            # Get symbol info for quantity precision
            symbol_info = self.market_scanner.get_symbol_info(opp.symbol)
            min_qty = symbol_info.get('min_qty', 0.001)
            qty_step = symbol_info.get('qty_step', 0.001)
            
            # Calculate quantity
            qty_raw = pos_size.position_value_usdt / opp.current_price
            
            # Round to step
            qty_raw = max(min_qty, round(qty_raw / qty_step) * qty_step)
            
            # Determine decimal precision from qty_step
            # e.g., qty_step=0.001 -> 3 decimals, qty_step=1 -> 0 decimals
            if qty_step >= 1:
                qty_decimals = 0
            else:
                qty_decimals = len(str(qty_step).split('.')[-1].rstrip('0'))
            
            # Format quantity as string with correct precision (Bybit requires string)
            if qty_decimals == 0:
                qty_str = str(int(qty_raw))
            else:
                qty_str = f"{qty_raw:.{qty_decimals}f}"
            
            qty = qty_raw  # Keep float for internal tracking
            
            # Determine side
            side = 'Buy' if opp.direction == 'long' else 'Sell'
            
            logger.info(f"EXECUTING: {side} {opp.symbol} | Qty: {qty_str} | "
                       f"Edge: {opp.edge_score:.2f} | Confidence: {opp.confidence:.1f}%")
            
            result = await client.place_order(
                symbol=opp.symbol,
                side=side,
                order_type='Market',
                qty=qty_str  # Pass as string!
            )
            
            if result.get('success'):
                logger.info(f"ORDER SUCCESS: {opp.symbol} {side} {qty}")
                
                # Create position tracking
                if user_id not in self.active_positions:
                    self.active_positions[user_id] = {}
                    
                # Calculate stops
                entry_price = opp.current_price
                if side == 'Buy':
                    stop_loss = entry_price * (1 - self.emergency_stop_loss / 100)
                    take_profit = entry_price * (1 + self.take_profit / 100)
                else:
                    stop_loss = entry_price * (1 + self.emergency_stop_loss / 100)
                    take_profit = entry_price * (1 - self.take_profit / 100)
                    
                self.active_positions[user_id][opp.symbol] = ActivePosition(
                    symbol=opp.symbol,
                    side=side,
                    size=qty,
                    entry_price=entry_price,
                    entry_time=datetime.utcnow(),
                    entry_edge=opp.edge_score,
                    entry_confidence=opp.confidence,
                    entry_regime=opp.regime,
                    peak_price=entry_price,
                    trough_price=entry_price,
                    peak_pnl_percent=0.0,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    trailing_active=False,
                    position_value=pos_size.position_value_usdt,
                    kelly_fraction=pos_size.kelly_fraction
                )
                
                # Register in position sizer
                await self.position_sizer.register_position(opp.symbol, pos_size.position_value_usdt)
                
                # Store trade event
                await self._store_trade_event(opp.symbol, 'opened', 0, opp.direction)
                
            else:
                logger.error(f"ORDER FAILED: {opp.symbol} - {result}")
                
        except Exception as e:
            logger.error(f"Execute trade error: {e}")
            
    async def _get_ticker(self, client: BybitV5Client, symbol: str) -> Optional[Dict]:
        """Get current ticker for a symbol"""
        try:
            # FIXED: Use get_tickers (plural) not get_ticker
            result = await client.get_tickers(category="linear", symbol=symbol)
            if result.get('success'):
                data = result.get('data', {}).get('list', [])
                if data:
                    return {
                        'last_price': safe_float(data[0].get('lastPrice')),
                        'bid': safe_float(data[0].get('bid1Price')),
                        'ask': safe_float(data[0].get('ask1Price'))
                    }
            else:
                logger.debug(f"Ticker fetch failed for {symbol}: {result}")
        except Exception as e:
            logger.debug(f"Ticker exception for {symbol}: {e}")
        return None
        
    async def _get_all_tickers(self, client: BybitV5Client) -> Dict[str, float]:
        """Get ALL tickers in ONE API call - much faster than individual calls"""
        try:
            result = await client.get_tickers(category="linear")
            if result.get('success'):
                tickers = {}
                for item in result.get('data', {}).get('list', []):
                    symbol = item.get('symbol')
                    price = safe_float(item.get('lastPrice'))
                    if symbol and price > 0:
                        tickers[symbol] = price
                logger.debug(f"⚡ Bulk fetched {len(tickers)} tickers in 1 API call")
                return tickers
        except Exception as e:
            logger.error(f"Bulk ticker fetch error: {e}")
        return {}
        
    async def _check_position_exit_fast(self, user_id: str, client: BybitV5Client,
                                         position: ActivePosition, wallet: Dict,
                                         all_tickers: Dict[str, float]):
        """FAST position exit check using pre-fetched ticker prices"""
        try:
            # Get price from pre-fetched tickers
            current_price = all_tickers.get(position.symbol)
            if not current_price or current_price <= 0:
                logger.warning(f"⚠️ No price for {position.symbol}")
                return
            
            # Calculate P&L
            if position.side == 'Buy':
                pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                if current_price > position.peak_price:
                    position.peak_price = current_price
                    position.peak_pnl_percent = pnl_percent
            else:  # Short
                pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
                if current_price < position.trough_price:
                    position.trough_price = current_price
                    position.peak_pnl_percent = pnl_percent
                    
            # === FAST EXIT LOGIC ===
            should_exit = False
            exit_reason = ""
            
            # Only log positions near thresholds to reduce spam
            if pnl_percent >= self.take_profit:
                logger.info(f"✅ {position.symbol}: P&L={pnl_percent:+.2f}% >= TP={self.take_profit}% - SELLING!")
                await self._log_to_console(f"✅ SELLING {position.symbol}: +{pnl_percent:.2f}% (Take Profit)")
                should_exit = True
                exit_reason = f"Take profit ({pnl_percent:.2f}%)"
            elif pnl_percent <= -self.emergency_stop_loss:
                logger.info(f"❌ {position.symbol}: P&L={pnl_percent:+.2f}% <= SL=-{self.emergency_stop_loss}% - SELLING!")
                await self._log_to_console(f"❌ SELLING {position.symbol}: {pnl_percent:.2f}% (Stop Loss)")
                should_exit = True
                exit_reason = f"Stop loss ({pnl_percent:.2f}%)"
            elif pnl_percent >= self.take_profit * 0.8:
                logger.info(f"⏳ {position.symbol}: {pnl_percent:+.2f}% near TP")
            elif pnl_percent <= -self.emergency_stop_loss * 0.8:
                logger.info(f"⚠️ {position.symbol}: {pnl_percent:+.2f}% near SL")
                
            # Trailing stop logic (simplified)
            if not should_exit and pnl_percent >= self.min_profit_to_trail:
                position.trailing_active = True
                if position.side == 'Buy':
                    drop = ((position.peak_price - current_price) / position.peak_price) * 100
                else:
                    drop = ((current_price - position.trough_price) / position.trough_price) * 100
                    
                if drop >= self.trail_from_peak and position.peak_pnl_percent >= self.take_profit:
                    should_exit = True
                    exit_reason = f"Trailing stop (peak: {position.peak_pnl_percent:.2f}%)"
                    logger.info(f"📉 {position.symbol}: Trailing exit at {pnl_percent:+.2f}%")
                    
            # === EXECUTE EXIT ===
            if should_exit:
                await self._close_position(user_id, client, position, pnl_percent, exit_reason)
                
        except Exception as e:
            logger.error(f"Fast exit check error for {position.symbol}: {e}")
            
    async def _log_to_console(self, message: str):
        """Log message to Redis for dashboard real-time console"""
        try:
            if self.redis_client:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": message
                }
                await self.redis_client.lpush('bot:console:logs', json.dumps(log_entry))
                await self.redis_client.ltrim('bot:console:logs', 0, 99)  # Keep last 100
        except Exception:
            pass  # Don't break trading for console logging
        
    async def _store_trade_event(self, symbol: str, action: str, pnl: float, reason: str,
                                  position: Optional[ActivePosition] = None):
        """Store trade event for dashboard and data collection"""
        try:
            timestamp = datetime.utcnow().isoformat()
            
            event = {
                'symbol': symbol,
                'action': action,
                'pnl_percent': pnl,
                'reason': reason,
                'timestamp': timestamp
            }
            
            # Push to Redis list for dashboard
            await self.redis_client.lpush('trading:events', json.dumps(event))
            await self.redis_client.ltrim('trading:events', 0, 99)  # Keep last 100
            
            # V3: Record to data collector for ML training
            if position:
                trade_record = TradeRecord(
                    timestamp=timestamp,
                    trade_id=f"{symbol}_{int(datetime.utcnow().timestamp())}",
                    symbol=symbol,
                    action=action,
                    direction='long' if position.side == 'Buy' else 'short',
                    confidence=position.entry_confidence,
                    edge_score=position.entry_edge,
                    technical_edge=0,  # Could expand
                    momentum_edge=0,
                    volume_edge=0,
                    sentiment_edge=0,
                    regime=position.entry_regime,
                    regime_action='normal',
                    entry_price=position.entry_price,
                    quantity=position.size,
                    position_value=position.position_value,
                    leverage=1,
                    stop_loss=position.stop_loss_price,
                    take_profit=position.take_profit_price,
                    kelly_fraction=position.kelly_fraction,
                    exit_price=position.peak_price if pnl > 0 else position.trough_price,
                    exit_time=timestamp,
                    pnl_percent=pnl,
                    pnl_value=position.position_value * pnl / 100,
                    duration_seconds=int((datetime.utcnow() - position.entry_time).total_seconds()),
                    exit_reason=reason,
                    won=pnl > 0
                )
                
                await data_collector.record_trade(trade_record)
                
                # V3: Quality filter for multi-user learning (professional)
                # Includes market context for de-duplication + multi-dimensional scoring
                quality_data = {
                    'trade_id': trade_record.trade_id,
                    'symbol': symbol,
                    'direction': trade_record.direction,
                    'entry_price': position.entry_price,
                    'exit_price': trade_record.exit_price,
                    'expected_price': position.entry_price,  # Could track expected vs actual
                    'quantity': position.size,
                    'position_value': position.position_value,
                    
                    # AI signals at entry
                    'edge_score': position.entry_edge,
                    'confidence': position.entry_confidence,
                    'regime': position.entry_regime,
                    'xgb_signal': 'buy' if position.side == 'Buy' else 'sell',
                    'xgb_confidence': position.entry_confidence,
                    'sentiment_score': getattr(position, 'entry_sentiment', 0),
                    
                    # Market context (for de-duplication)
                    'volatility': getattr(position, 'entry_volatility', 1.5),
                    'liquidity': getattr(position, 'entry_liquidity', 50),
                    'trend_strength': getattr(position, 'entry_trend', 0),
                    
                    # Outcome
                    'pnl_percent': pnl,
                    'pnl_value': trade_record.pnl_value,
                    'stop_loss_percent': self.emergency_stop_loss,
                    'duration_seconds': trade_record.duration_seconds,
                    'exit_reason': reason
                }
                
                # Quality filter: user is source, not signal
                await training_data_manager.process_trade(quality_data, user_id="default")
                
                logger.debug(f"Trade recorded for ML training: {symbol} {action} {pnl:.2f}%")
            
        except Exception as e:
            logger.debug(f"Trade event store error: {e}")
            
    async def _load_settings(self):
        """Load settings from Redis"""
        try:
            data = await self.redis_client.hgetall('bot:settings')
            if data:
                parsed = {
                    k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v 
                    for k, v in data.items()
                }
                
                # Exit strategy
                self.emergency_stop_loss = float(parsed.get('stopLossPercent', self.emergency_stop_loss))
                self.take_profit = float(parsed.get('takeProfitPercent', self.take_profit))
                self.trail_from_peak = float(parsed.get('trailingStopPercent', self.trail_from_peak))
                self.min_profit_to_trail = float(parsed.get('minProfitToTrail', self.min_profit_to_trail))
                
                # Entry filters
                self.min_confidence = float(parsed.get('minConfidence', self.min_confidence))
                self.min_edge = float(parsed.get('minEdge', self.min_edge))
                
                # Risk limits (0 = unlimited)
                self.max_open_positions = int(float(parsed.get('maxOpenPositions', self.max_open_positions)))
                self.max_exposure_percent = float(parsed.get('maxTotalExposure', self.max_exposure_percent))
                self.max_daily_drawdown = float(parsed.get('maxDailyDrawdown', self.max_daily_drawdown))
                
                # Only log on first load or when settings actually change
                new_settings_str = f"SL={self.emergency_stop_loss}%,TP={self.take_profit}%,Trail={self.trail_from_peak}%"
                if not hasattr(self, '_last_settings_str') or self._last_settings_str != new_settings_str:
                    logger.info(f"⚙️ Settings: SL={self.emergency_stop_loss}%, TP={self.take_profit}%, "
                               f"Trail={self.trail_from_peak}%, MinConf={self.min_confidence}%, "
                               f"MinEdge={self.min_edge}, MaxPos={'Unlimited' if self.max_open_positions == 0 else self.max_open_positions}")
                    self._last_settings_str = new_settings_str
        except Exception as e:
            logger.debug(f"Load settings error: {e}")
            
    async def _load_stats(self):
        """Load stats from Redis"""
        try:
            data = await self.redis_client.get('trader:stats')
            if data:
                self.stats = json.loads(data)
        except:
            pass
            
    async def _save_stats(self):
        """Save stats to Redis"""
        try:
            await self.redis_client.set('trader:stats', json.dumps(self.stats))
        except:
            pass
            
    async def _log_status(self):
        """Log current trading status"""
        total_positions = sum(len(p) for p in self.active_positions.values())
        win_rate = (self.stats['winning_trades'] / self.stats['total_trades'] * 100) if self.stats['total_trades'] > 0 else 0
        
        logger.info(
            f"STATUS | Positions: {total_positions} | Trades: {self.stats['total_trades']} | "
            f"Win Rate: {win_rate:.1f}% | P&L: ${self.stats['total_pnl']:.2f}"
        )
        
        await self._save_stats()
        
    async def get_status(self) -> Dict:
        """Get current trader status for API"""
        total_positions = sum(len(p) for p in self.active_positions.values())
        win_rate = (self.stats['winning_trades'] / self.stats['total_trades'] * 100) if self.stats['total_trades'] > 0 else 0
        
        return {
            'is_running': self.is_running,
            'connected_users': len(self.user_clients),
            'active_positions': total_positions,
            'stats': {
                'total_trades': self.stats['total_trades'],
                'winning_trades': self.stats['winning_trades'],
                'win_rate': round(win_rate, 1),
                'total_pnl': round(self.stats['total_pnl'], 2),
                'opportunities_scanned': self.stats['opportunities_scanned'],
                'trades_rejected': {
                    'low_edge': self.stats['trades_rejected_low_edge'],
                    'regime': self.stats['trades_rejected_regime'],
                    'risk': self.stats['trades_rejected_risk']
                }
            },
            'settings': {
                'min_edge': self.min_edge,
                'min_confidence': self.min_confidence,
                'stop_loss': self.emergency_stop_loss,
                'take_profit': self.take_profit,
                'trailing': self.trail_from_peak,
                'max_positions': self.max_open_positions
            }
        }


# Global instance (will be initialized in main.py)
autonomous_trader_v2 = AutonomousTraderV2()

