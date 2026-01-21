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
from typing import Dict, List, Optional, Any, Tuple, Set
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
from services.whale_tracker import whale_tracker


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
    leverage: int = 1  # Leverage used for this position
    
    # Smart exit features (MICRO PROFIT)
    breakeven_active: bool = False  # SL moved to entry price
    partial_exit_done: bool = False  # 50% already taken
    original_size: float = 0.0  # Track original size for partial exits


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
        
        # Paused users - still connected but not opening NEW positions
        self.paused_users: Set[str] = set()
        
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
        
        # Smart exit (MICRO PROFIT mode)
        self.breakeven_trigger = 0.25  # Move SL to entry at +0.25%
        self.partial_exit_trigger = 0.30  # TP1: Take 50% at +0.30%
        self.partial_exit_percent = 50  # How much to close (50%)
        self.use_smart_exit = False  # Enable breakeven + partial exits
        self.momentum_threshold = 0.02  # Minimum momentum % required
        self._ticker_momentum = {}  # Cache for fast momentum checks
        
        # Time stop (MICRO PROFIT mode)
        self.time_stop_minutes = 4  # Close after 4 minutes
        self.time_stop_min_pnl = 0.15  # Only if PnL < +0.15%
        self.use_time_stop = False  # Enable time stop
        
        # Risk limits (0 = unlimited positions)
        self.max_open_positions = 0  # Unlimited by default
        self.max_exposure_percent = 100  # 100% = can use entire budget
        self.max_daily_drawdown = 3.0
        
        # Leverage mode: '1x', '2x', '3x', '5x', '10x', 'auto'
        self.leverage_mode = 'auto'
        
        # AI Model toggles
        self.use_dynamic_sizing = True
        self.use_regime_detection = True
        self.use_edge_estimation = True
        self.use_crypto_bert = True
        self.use_xgboost_classifier = True
        self.use_price_predictor = True
        
        # Risk mode tracking
        self.risk_mode = "normal"
        
        # COOLDOWN: Prevent reopening same symbol immediately after close
        self._cooldown_symbols: Dict[str, datetime] = {}
        self.cooldown_seconds = 60  # Wait 60 seconds before reopening same symbol
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'opportunities_scanned': 0,
            'trades_rejected_low_edge': 0,
            'trades_rejected_regime': 0,
            'trades_rejected_risk': 0,
            'trades_rejected_no_momentum': 0
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
        
        # Load paused users from Redis
        await self._load_paused_users()
        
        # Initialize whale tracker
        try:
            await whale_tracker.initialize()
            await whale_tracker.start()
            logger.info("Whale tracker started")
        except Exception as e:
            logger.warning(f"Whale tracker failed to start: {e}")
        
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
        """Pause trading for user - keeps connection but stops NEW positions"""
        await self.pause_trading(user_id)
        logger.info(f"User {user_id} trading paused (existing positions will continue)")
    
    async def pause_trading(self, user_id: str):
        """Pause trading - stop opening NEW positions, but keep monitoring existing ones"""
        self.paused_users.add(user_id)
        # Store in Redis for persistence
        if self.redis_client:
            await self.redis_client.set(f'trading:paused:{user_id}', '1')
        logger.info(f"🛑 Trading PAUSED for {user_id} - no new positions will be opened")
    
    async def resume_trading(self, user_id: str):
        """Resume trading - allow opening new positions again"""
        self.paused_users.discard(user_id)
        # Remove from Redis
        if self.redis_client:
            await self.redis_client.delete(f'trading:paused:{user_id}')
        logger.info(f" Trading RESUMED for {user_id} - new positions will be opened")
    
    def is_paused(self, user_id: str) -> bool:
        """Check if trading is paused for user"""
        return user_id in self.paused_users
    
    async def force_disconnect_user(self, user_id: str):
        """Fully disconnect user - closes connection and removes data"""
        if user_id in self.user_clients:
            await self.user_clients[user_id].close()
            del self.user_clients[user_id]
            if user_id in self.active_positions:
                del self.active_positions[user_id]
            self.paused_users.discard(user_id)
            logger.info(f"User {user_id} fully disconnected")
            
    async def run_trading_loop(self):
        """
        Main trading loop - BULLETPROOF VERSION
        This loop MUST NEVER stop - it's the heart of the trading bot!
        """
        logger.info("=" * 60)
        logger.info(" TRADING LOOP STARTING - BULLETPROOF MODE!")
        logger.info(f"Settings: TP={self.take_profit}%, SL={self.emergency_stop_loss}%")
        logger.info(f"Trail from peak: {self.trail_from_peak}%, Min profit to trail: {self.min_profit_to_trail}%")
        logger.info("=" * 60)
        
        cycle = 0
        consecutive_errors = 0
        
        while self.is_running:
            cycle_start = datetime.utcnow()
            
            try:
                cycle += 1
                
                # Log EVERY cycle for debugging
                connected_users = len(self.user_clients)
                total_positions = sum(len(p) for p in self.active_positions.values())
                
                # Get and store current regime (for dashboard)
                current_regime = "Unknown"
                try:
                    if self.regime_detector:
                        btc_regime = await self.regime_detector.detect_regime("BTCUSDT")
                        current_regime = btc_regime.regime if btc_regime else "Unknown"
                        # Store for dashboard console
                        if self.redis_client:
                            await self.redis_client.set('bot:current_regime', current_regime)
                except:
                    pass
                
                # Log to file every 5 cycles
                mode_icons = {'lock_profit': 'LOCK', 'micro_profit': 'MICRO', 'safe': 'SAFE', 'aggressive': 'AGG', 'normal': 'NORM'}
                mode_str = mode_icons.get(self.risk_mode, 'NORM')
                
                if cycle % 5 == 0 or cycle <= 10:
                    logger.info(f"Cycle {cycle} | {mode_str} | Users: {connected_users} | Pos: {total_positions} | Trail={self.trail_from_peak}%")
                
                # Console log every 30 cycles (1 minute) to reduce spam
                if cycle % 30 == 0 or cycle == 1:
                    tp_str = f"TP={self.take_profit}%" if self.take_profit > 0 else "TP=OFF"
                    console_msg = f"[{mode_str}] Positions: {total_positions} | {tp_str} | SL={self.emergency_stop_loss}% | Trail={self.trail_from_peak}% | Regime: {current_regime}"
                    try:
                        await self._log_to_console(console_msg, "INFO")
                    except:
                        pass  # Never fail on console logging
                
                # Process users - with overall timeout
                if not self.user_clients:
                    if cycle % 5 == 0:
                        logger.warning(" NO USERS CONNECTED - waiting for dashboard connection...")
                else:
                    for user_id, client in list(self.user_clients.items()):
                        try:
                            # 30 second timeout for entire user processing
                            await asyncio.wait_for(
                                self._process_user(user_id, client),
                                timeout=30.0
                            )
                            consecutive_errors = 0  # Reset on success
                        except asyncio.TimeoutError:
                            logger.warning(f" Processing user {user_id} timed out after 30s - continuing")
                        except Exception as e:
                            consecutive_errors += 1
                            logger.error(f"Error processing user {user_id}: {e}")
                            if consecutive_errors >= 5:
                                logger.error(f"🚨 {consecutive_errors} consecutive errors! Sleeping 10s...")
                                await asyncio.sleep(10)
                                consecutive_errors = 0
                        
                # Reload settings every 5 cycles
                if cycle % 5 == 0:
                    try:
                        await asyncio.wait_for(self._load_settings(), timeout=5.0)
                    except:
                        pass  # Never fail on settings reload
                
                # Log status every 100 cycles
                if cycle % 100 == 0:
                    try:
                        await self._log_status()
                    except:
                        pass
                    
                # Calculate sleep time - 0.5s for LOCK_PROFIT, 1s normal
                elapsed = (datetime.utcnow() - cycle_start).total_seconds()
                is_lock_profit = self.trail_from_peak <= 0.1
                check_interval = 0.5 if is_lock_profit else self.position_check_interval
                sleep_time = max(check_interval - elapsed, 0.1)  # Min 0.1s sleep
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.warning(" Trading loop CANCELLED - shutting down gracefully")
                break
            except Exception as e:
                # This should NEVER happen - but if it does, log and continue!
                logger.error(f"🚨 CRITICAL LOOP ERROR (cycle {cycle}): {e}")
                import traceback
                logger.error(traceback.format_exc())
                await asyncio.sleep(2)  # Brief sleep then continue
                # NEVER break - keep the loop running!
                
    async def _process_user(self, user_id: str, client: BybitV5Client):
        """
        Process trading for one user - BULLETPROOF VERSION
        This method MUST NEVER crash - all operations have timeouts and error handling
        """
        try:
            # 1. Get wallet balance (5s timeout)
            try:
                wallet = await asyncio.wait_for(self._get_wallet(client), timeout=5.0)
                if wallet['total_equity'] < 20:
                    return
            except asyncio.TimeoutError:
                logger.warning(f"[STEP 1] Wallet fetch timed out for {user_id}")
                return
            except Exception as e:
                logger.error(f"[STEP 1] Wallet error: {e}")
                return
                
            # 2. Sync positions from exchange (10s timeout)
            try:
                await asyncio.wait_for(self._sync_positions(user_id, client), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(f"[STEP 2] Position sync timed out")
            except Exception as e:
                logger.error(f"[STEP 2] Sync error: {e}")
            
            # 3. Check existing positions for exit - CRITICAL FOR LOCK_PROFIT!
            positions_list = list(self.active_positions.get(user_id, {}).items())
            if positions_list:
                logger.info(f"Fast-checking {len(positions_list)} positions (TP={self.take_profit}%, SL={self.emergency_stop_loss}%)")
                
                # BATCH: Get all tickers in ONE API call (5s timeout)
                try:
                    all_tickers = await asyncio.wait_for(self._get_all_tickers(client), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"[STEP 3a] Ticker fetch timed out - using individual fetches")
                    all_tickers = {}
                except Exception as e:
                    logger.error(f"[STEP 3a] Ticker error: {e}")
                    all_tickers = {}
                
                # Check each position - each with its own error handling
                for symbol, position in positions_list:
                    try:
                        await asyncio.wait_for(
                            self._check_position_exit_fast(user_id, client, position, wallet, all_tickers),
                            timeout=3.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"[STEP 3b] Check timed out for {symbol}")
                    except Exception as pos_error:
                        logger.error(f"[STEP 3b] Error checking {symbol}: {pos_error}")
                
            # 4. Look for new opportunities (if room) - 10s timeout
            num_positions = len(self.active_positions.get(user_id, {}))
            can_open_more = self.max_open_positions == 0 or num_positions < self.max_open_positions
            
            if can_open_more:
                try:
                    await asyncio.wait_for(
                        self._find_opportunities(user_id, client, wallet),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[STEP 4] Opportunity search timed out after 10s")
                except Exception as opp_error:
                    logger.error(f"[STEP 4] Opportunity error: {opp_error}")
                    
        except Exception as e:
            # This should NEVER happen but just in case
            logger.error(f"[PROCESS USER] UNEXPECTED ERROR for {user_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
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
        logger.debug(f" Exchange returned {len(positions_list)} positions")
        
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
        
        # IMPORTANT: Sync position sizer with exchange to remove any stale positions
        # This ensures position_sizer.open_positions matches actual exchange state
        positions_data = {}
        if user_id in self.active_positions:
            for symbol, pos in self.active_positions[user_id].items():
                positions_data[symbol] = pos.position_value
        
        await self.position_sizer.sync_with_exchange(exchange_positions, positions_data)
                    
    async def _check_position_exit(self, user_id: str, client: BybitV5Client,
                                    position: ActivePosition, wallet: Dict):
        """Check if position should be exited"""
        try:
            # Get current price
            ticker = await self._get_ticker(client, position.symbol)
            if not ticker:
                logger.warning(f" No ticker for {position.symbol} - cannot check exit")
                return
                
            current_price = ticker['last_price']
            if current_price <= 0:
                logger.warning(f" Invalid price {current_price} for {position.symbol}")
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
            tp_status = f"TP={self.take_profit}%" if self.take_profit > 0 else "TP=OFF"
            logger.info(f"🔎 CHECK {position.symbol}: Side={position.side}, Price=${current_price:.6f}, Entry=${position.entry_price:.6f}, P&L={pnl_percent:+.2f}%, {tp_status}, SL=-{self.emergency_stop_loss}%")
            
            if self.take_profit > 0 and pnl_percent >= self.take_profit:
                logger.info(f" {position.symbol}: P&L={pnl_percent:+.2f}% >= TP={self.take_profit}% - TRIGGERING TAKE PROFIT!")
            elif pnl_percent <= -self.emergency_stop_loss:
                logger.info(f" {position.symbol}: P&L={pnl_percent:+.2f}% <= SL=-{self.emergency_stop_loss}% - TRIGGERING STOP LOSS!")
            elif self.take_profit > 0 and pnl_percent >= self.take_profit * 0.7:
                logger.info(f" {position.symbol}: P&L={pnl_percent:+.2f}% approaching TP={self.take_profit}%")
            
            # 1. STOP LOSS
            if pnl_percent <= -self.emergency_stop_loss:
                should_exit = True
                exit_reason = f"Stop loss hit ({pnl_percent:.2f}%)"
                logger.info(f"🛑 STOP LOSS: {position.symbol} at {pnl_percent:+.2f}%")
                
            # 2. TAKE PROFIT - Exit when profit reaches target (only if TP is enabled)
            elif self.take_profit > 0 and pnl_percent >= self.take_profit:
                should_exit = True
                exit_reason = f"Take profit reached ({pnl_percent:.2f}% >= {self.take_profit}%)"
                logger.info(f" TAKE PROFIT: {position.symbol} at {pnl_percent:+.2f}% >= TP {self.take_profit}% - SELLING NOW!")
                
            # 3. TRAILING STOP - LOCK PROFIT MODE
            # For LOCK PROFIT: trail_from_peak=0.05%, min_profit_to_trail=0.01%
            # Activates as soon as we're in ANY profit, exits on drop from peak
            
            is_lock_profit_mode = self.trail_from_peak <= 0.1  # Ultra-tight = LOCK PROFIT
            
            # Calculate drop from peak (always, for logging)
            if position.side == 'Buy':
                drop_from_peak = ((position.peak_price - current_price) / position.peak_price) * 100
            else:
                drop_from_peak = ((current_price - position.trough_price) / position.trough_price) * 100
            
            # LOCK PROFIT MODE: More aggressive - activate trailing earlier
            if is_lock_profit_mode:
                # Activate trailing as soon as we've EVER been in profit
                if position.peak_pnl_percent >= self.min_profit_to_trail:
                    position.trailing_active = True
                    
                    # Log EVERY check in lock profit mode
                    logger.info(f"📈 LOCK_PROFIT {position.symbol}: Peak={position.peak_pnl_percent:+.3f}%, Now={pnl_percent:+.3f}%, Drop={drop_from_peak:.3f}%, Trigger={self.trail_from_peak:.3f}%")
                    
                    # EXIT if drop from peak exceeds threshold
                    # Even if current P&L is 0 or slightly negative!
                    if drop_from_peak >= self.trail_from_peak:
                        should_exit = True
                        exit_reason = f"LOCK PROFIT (peak: {position.peak_pnl_percent:.2f}%, drop: {drop_from_peak:.3f}%)"
                        logger.info(f"LOCK PROFIT SELL: {position.symbol} | Peak was {position.peak_pnl_percent:+.2f}%, dropped {drop_from_peak:.3f}% >= {self.trail_from_peak:.3f}%")
            else:
                # NORMAL MODE: Standard trailing stop
                # Activate trailing when profit reaches min_profit_to_trail (e.g., 0.5%)
                if pnl_percent >= self.min_profit_to_trail:
                    position.trailing_active = True
                
                # Log trailing status when approaching trigger
                if position.trailing_active and drop_from_peak >= self.trail_from_peak * 0.5:
                    logger.info(f"📈 TRAILING {position.symbol}: Peak={position.peak_pnl_percent:+.3f}%, Now={pnl_percent:+.3f}%, Drop={drop_from_peak:.3f}%")
                
                # Once trailing is active, sell when price drops from peak
                if position.trailing_active and drop_from_peak >= self.trail_from_peak:
                    # Only sell if we're still in profit (or minimal loss due to spread)
                    if pnl_percent >= -0.05:
                        should_exit = True
                        exit_reason = f"Trailing stop (peak: +{position.peak_pnl_percent:.2f}%, dropped {drop_from_peak:.2f}%)"
                        logger.info(f"📉 TRAILING SELL {position.symbol}: Peak=+{position.peak_pnl_percent:.2f}%, Now={pnl_percent:+.2f}%")
                    
            # 4. REGIME CHANGED TO AVOID
            if not should_exit:
                regime = await self.regime_detector.detect_regime(position.symbol)
                if regime.recommended_action == 'avoid' and pnl_percent > 0:
                    should_exit = True
                    exit_reason = f"Regime changed to avoid (locking {pnl_percent:.2f}% profit)"
                    
            # === EXECUTE EXIT ===
            if should_exit:
                logger.info(f" CLOSING {position.symbol}: {exit_reason}")
                await self._close_position(user_id, client, position, pnl_percent, exit_reason)
            else:
                # Log why we're NOT exiting if position has significant P&L
                if abs(pnl_percent) > 1.0:
                    logger.info(f" HOLD {position.symbol}: P&L={pnl_percent:+.2f}% (TP={self.take_profit}%, SL=-{self.emergency_stop_loss}%) - No exit trigger")
                
        except Exception as e:
            logger.error(f"Exit check error for {position.symbol}: {e}")
            
    async def _close_position(self, user_id: str, client: BybitV5Client,
                              position: ActivePosition, pnl_percent: float, reason: str):
        """Close a position"""
        try:
            logger.info(f" EXECUTING CLOSE: {position.symbol} | Side: {position.side} | Size: {position.size} | Reason: {reason}")
            
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
                # Calculate GROSS P&L
                gross_pnl = position.position_value * (pnl_percent / 100)
                
                # Calculate trading fees (Bybit taker fee: 0.055% entry + 0.055% exit = 0.11% total)
                # Fee is charged on position VALUE (with leverage)
                total_fees = position.position_value * 0.0011  # 0.11% round-trip
                
                # NET P&L = Gross - Fees
                pnl_value = gross_pnl - total_fees
                net_pnl_percent = (pnl_value / position.position_value) * 100 if position.position_value > 0 else 0
                
                won = pnl_value > 0  # Win/loss based on NET, not gross!
                
                logger.info(f"CLOSED {position.symbol}: {reason} | Gross: {pnl_percent:+.2f}% | Fees: ${total_fees:.2f} | NET: {net_pnl_percent:+.2f}% (${pnl_value:+.2f})")
                
                # Console log for dashboard - show NET P&L (after fees)
                await self._log_to_console(
                    f"CLOSED {position.symbol}: {net_pnl_percent:+.2f}% (${pnl_value:+.2f} NET) | {reason}",
                    "TRADE"
                )
                
                # Update stats with NET values
                self.stats['total_trades'] += 1
                if won:
                    self.stats['winning_trades'] += 1
                self.stats['total_pnl'] += pnl_value  # NET P&L
                
                # Record in position sizer (NET P&L)
                await self.position_sizer.close_position(position.symbol, pnl_value)
                
                # Record in edge estimator for calibration
                await self.edge_estimator.record_outcome(position.symbol, position.entry_edge, won)
                
                # Record in market scanner
                await self.market_scanner.record_trade_result(position.symbol, won, pnl_value)
                
                # Record in learning engine with NET values
                if self.learning_engine:
                    exit_price = position.entry_price * (1 + pnl_percent/100)
                    outcome = TradeOutcome(
                        symbol=position.symbol,
                        strategy='edge_based',
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        quantity=position.size,
                        side='long' if position.side == 'Buy' else 'short',
                        pnl=pnl_value,  # NET P&L
                        pnl_percent=net_pnl_percent,  # NET %
                        hold_time_seconds=int((datetime.utcnow() - position.entry_time).total_seconds()),
                        market_regime=position.entry_regime,
                        volatility_at_entry=0.0,
                        sentiment_at_entry=0.0,
                        timestamp=datetime.utcnow().isoformat()
                    )
                    await self.learning_engine.update_from_trade(outcome)
                    
                # Remove from active positions
                if user_id in self.active_positions:
                    if position.symbol in self.active_positions[user_id]:
                        del self.active_positions[user_id][position.symbol]
                
                # ADD COOLDOWN: Prevent reopening this symbol for 60 seconds
                self._cooldown_symbols[position.symbol] = datetime.utcnow()
                logger.debug(f"{position.symbol} on cooldown for {self.cooldown_seconds}s")
                        
                # Store trade for dashboard with NET P&L
                await self._store_trade_event(position.symbol, 'closed', net_pnl_percent, reason, position, pnl_value)
                
            else:
                logger.error(f"Failed to close {position.symbol}: {result}")
                
        except Exception as e:
            logger.error(f"Close position error: {e}")
            
    async def _partial_close_position(self, user_id: str, client: BybitV5Client,
                                       position: ActivePosition, pnl_percent: float,
                                       close_percent: float = 50.0):
        """Partially close a position (scale-out exit)"""
        try:
            # Calculate how much to close
            close_size = position.size * (close_percent / 100.0)
            
            # Get symbol info for minimum qty
            symbol_info = await self._get_symbol_info(position.symbol, client)
            min_qty = float(symbol_info.get('minOrderQty', 1)) if symbol_info else 1
            qty_step = float(symbol_info.get('qtyStep', 0.01)) if symbol_info else 0.01
            
            # Round to qty step
            close_size = round(close_size / qty_step) * qty_step
            
            if close_size < min_qty:
                logger.warning(f" Partial close size {close_size} < min {min_qty}, skipping partial exit")
                return False
                
            remaining_size = position.size - close_size
            
            logger.info(f" PARTIAL CLOSE: {position.symbol} | Closing: {close_size} ({close_percent}%) | Remaining: {remaining_size}")
            
            # Determine close side (opposite of position)
            close_side = 'Sell' if position.side == 'Buy' else 'Buy'
            
            result = await client.place_order(
                category="linear",
                symbol=position.symbol,
                side=close_side,
                order_type='Market',
                qty=str(close_size),
                reduce_only=True
            )
            
            if result.get('success'):
                pnl_value = (position.position_value * close_percent / 100) * (pnl_percent / 100)
                
                logger.info(f" PARTIAL CLOSED {position.symbol}: {close_percent}% at +{pnl_percent:.2f}% (${pnl_value:+.2f})")
                
                # Update position size (remaining)
                position.size = remaining_size
                position.position_value = position.position_value * (remaining_size / (remaining_size + close_size))
                
                # Update stats
                self.stats['total_trades'] += 0.5  # Count as half trade
                if pnl_percent > 0:
                    self.stats['winning_trades'] += 0.5
                self.stats['total_pnl'] += pnl_value
                
                # Log to console
                await self._log_to_console(f"PARTIAL: {position.symbol} +{pnl_percent:.2f}% (took {close_percent}%)", "TRADE")
                
                return True
            else:
                logger.error(f" Partial close failed for {position.symbol}: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Partial close error: {e}")
            return False
    
    async def _get_symbol_info(self, symbol: str, client: BybitV5Client) -> Dict:
        """Get symbol info from market scanner"""
        try:
            if self.market_scanner:
                info = self.market_scanner.get_symbol_info(symbol)
                return {
                    'minOrderQty': info.get('min_qty', 1),
                    'qtyStep': info.get('qty_step', 0.01)
                }
        except Exception:
            pass
        return {'minOrderQty': 1, 'qtyStep': 0.01}
            
    async def _find_breakouts(self, user_id: str, client: BybitV5Client, wallet: Dict) -> List[TradingOpportunity]:
        """
        BREAKOUT DETECTOR - Find coins with MASSIVE moves (+5% or more)
        
        This catches opportunities like HANA +10% that normal filters would miss.
        When a coin explodes, we want IN - no overthinking!
        
        Rules:
        - +5% to +10% = Strong breakout, enter with normal size
        - +10% to +20% = Mega breakout, enter with caution (might be late)
        - +20%+ = FOMO territory, skip or small size
        - Negative breakouts work too for shorts
        """
        breakouts = []
        
        try:
            # Get ALL tickers from Bybit
            tickers_result = await client.get_tickers()
            if not tickers_result.get('success'):
                return breakouts
            
            tickers = tickers_result.get('data', {}).get('list', [])
            
            # Minimum volume filter ($500K to ensure liquidity)
            min_volume = 500000
            
            for ticker in tickers:
                symbol = ticker.get('symbol', '')
                if not symbol.endswith('USDT') or symbol in ['USDCUSDT', 'USDTUSDT', 'DAIUSDT']:
                    continue
                
                # Skip if already in position
                if symbol in self.active_positions.get(user_id, {}):
                    continue
                
                # Skip if on cooldown
                if symbol in self._cooldown_symbols:
                    continue
                
                price_change = float(ticker.get('price24hPcnt', 0)) * 100  # Convert to %
                volume = float(ticker.get('turnover24h', 0))
                last_price = float(ticker.get('lastPrice', 0))
                
                if volume < min_volume or last_price <= 0:
                    continue
                
                # BREAKOUT DETECTION - Now handles ALL big moves
                is_breakout = False
                direction = None
                breakout_strength = 0
                size_multiplier = 1.0  # Reduce size for extreme moves
                
                # Bullish breakout (+5% and up)
                if price_change >= 5:
                    is_breakout = True
                    direction = 'long'
                    
                    # Tiered approach: bigger move = more caution
                    if price_change >= 50:
                        # EXTREME breakout (+50%+) - very risky, small size
                        breakout_strength = 60
                        size_multiplier = 0.3  # Only 30% of normal size
                        logger.warning(f" EXTREME breakout {symbol} +{price_change:.1f}% - using 30% size")
                    elif price_change >= 25:
                        # BIG breakout (+25-50%) - risky, reduced size
                        breakout_strength = 75
                        size_multiplier = 0.5  # 50% of normal size
                    else:
                        # NORMAL breakout (+5-25%) - ideal
                        breakout_strength = min(100, price_change * 10)
                        size_multiplier = 1.0
                    
                # Bearish breakout (-5% and down)
                elif price_change <= -5:
                    is_breakout = True
                    direction = 'short'
                    
                    abs_change = abs(price_change)
                    if abs_change >= 50:
                        breakout_strength = 60
                        size_multiplier = 0.3
                        logger.warning(f" EXTREME dump {symbol} {price_change:.1f}% - using 30% size")
                    elif abs_change >= 25:
                        breakout_strength = 75
                        size_multiplier = 0.5
                    else:
                        breakout_strength = min(100, abs_change * 10)
                        size_multiplier = 1.0
                
                if is_breakout:
                    # Create opportunity with HIGH edge/confidence to bypass normal filters
                    opp = TradingOpportunity(
                        symbol=symbol,
                        direction=direction,
                        edge_score=0.8,  # High edge to pass filters
                        confidence=breakout_strength,
                        opportunity_score=breakout_strength * size_multiplier,  # Adjust score by risk
                        current_price=last_price,
                        price_change_24h=price_change,
                        volume_24h=volume,
                        should_trade=True,
                        reasons=[f" BREAKOUT: {price_change:+.1f}% move (size: {size_multiplier*100:.0f}%)"],
                        timestamp=datetime.utcnow().isoformat()
                    )
                    # Store size multiplier for position sizing
                    opp.size_multiplier = size_multiplier
                    breakouts.append(opp)
                    logger.info(f" BREAKOUT: {symbol} {price_change:+.1f}% | Vol: ${volume/1000000:.1f}M | Size: {size_multiplier*100:.0f}%")
            
            # Sort by opportunity score (balances size and strength)
            breakouts.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            # Log if we found breakouts
            if breakouts:
                await self._log_to_console(f"{len(breakouts)} BREAKOUTS detected! Best: {breakouts[0].symbol} {breakouts[0].price_change_24h:+.1f}%", "SIGNAL")
            
        except Exception as e:
            logger.error(f"Breakout detection error: {e}")
        
        # Return more breakouts for trading (up to 10)
        return breakouts[:10]
    
    async def _find_opportunities(self, user_id: str, client: BybitV5Client, wallet: Dict):
        """Find and execute new trading opportunities"""
        try:
            # Check if trading is paused for this user
            if self.is_paused(user_id):
                logger.debug(f" Trading paused for {user_id} - skipping opportunity search")
                return
            
            # === BREAKOUT DETECTION FIRST ===
            # This catches big moves that normal filters would reject
            breakouts = await self._find_breakouts(user_id, client, wallet)
            
            for opp in breakouts:
                num_positions = len(self.active_positions.get(user_id, {}))
                
                # Check if max positions reached
                if self.max_open_positions > 0 and num_positions >= self.max_open_positions:
                    logger.info(f" BREAKOUT {opp.symbol} skipped - max positions ({self.max_open_positions}) reached")
                    await self._log_to_console(f"{opp.symbol} {opp.price_change_24h:+.1f}% skipped - max positions", "WARNING")
                    break
                
                # Check if already in this position
                if opp.symbol in self.active_positions.get(user_id, {}):
                    logger.debug(f" BREAKOUT {opp.symbol} skipped - already in position")
                    continue
                
                # Breakouts skip most validation - they're already high-conviction
                logger.info(f" BREAKOUT TRADE: {opp.symbol} | {opp.price_change_24h:+.1f}%")
                await self._log_to_console(f"EXECUTING BREAKOUT: {opp.symbol} {opp.price_change_24h:+.1f}%", "TRADE")
                try:
                    await self._execute_trade(user_id, client, opp, wallet)
                    logger.info(f" BREAKOUT TRADE SUBMITTED: {opp.symbol}")
                except Exception as e:
                    logger.error(f" BREAKOUT TRADE FAILED: {opp.symbol} - {e}")
                    await self._log_to_console(f"FAILED: {opp.symbol} - {str(e)[:50]}", "ERROR")
            
            # === NORMAL OPPORTUNITY SCAN ===
            # Get opportunities from scanner
            opportunities = await self.market_scanner.get_tradeable_opportunities()
            
            self.stats['opportunities_scanned'] += len(opportunities)
            
            # Log opportunity search status and reset reject counter
            self._reject_count = 0
            
            # Only log to console every 30 seconds (15 cycles) to reduce spam
            scan_log_interval = getattr(self, '_scan_log_counter', 0)
            should_log_scan = scan_log_interval % 15 == 0
            self._scan_log_counter = scan_log_interval + 1
            
            if opportunities:
                logger.info(f"🔍 Found {len(opportunities)} potential opportunities")
                # Console log only every 30 seconds
                if should_log_scan:
                    top_opps = opportunities[:3]
                    opp_str = ", ".join([f"{o.symbol}({o.edge_score:.2f})" for o in top_opps])
                    await self._log_to_console(f"{len(opportunities)} opportunities | Best: {opp_str}", "SIGNAL")
            else:
                logger.debug("🔍 No opportunities found in this scan")
                if should_log_scan:
                    await self._log_to_console("Scanning... waiting for signals", "INFO")
            
            for opp in opportunities:
                # Skip if we have max positions (0 = unlimited)
                num_positions = len(self.active_positions.get(user_id, {}))
                if self.max_open_positions > 0 and num_positions >= self.max_open_positions:
                    break
                    
                # Skip if already in position
                if opp.symbol in self.active_positions.get(user_id, {}):
                    continue
                
                # Skip if symbol is on cooldown (recently closed)
                if opp.symbol in self._cooldown_symbols:
                    cooldown_time = self._cooldown_symbols[opp.symbol]
                    elapsed = (datetime.utcnow() - cooldown_time).total_seconds()
                    if elapsed < self.cooldown_seconds:
                        logger.debug(f" {opp.symbol} on cooldown ({int(self.cooldown_seconds - elapsed)}s remaining)")
                        continue
                    else:
                        # Cooldown expired, remove from dict
                        del self._cooldown_symbols[opp.symbol]
                    
                # Validate the opportunity
                should_trade, reason = await self._validate_opportunity(opp, wallet, client)
                
                if should_trade:
                    logger.info(f" OPENING TRADE: {opp.symbol} | Edge={opp.edge_score:.2f} | Conf={opp.confidence:.0f}%")
                    await self._execute_trade(user_id, client, opp, wallet)
                else:
                    # Log first 3 rejections per cycle to avoid spam
                    if not hasattr(self, '_reject_count') or self._reject_count < 3:
                        logger.info(f" Rejected {opp.symbol}: {reason}")
                        self._reject_count = getattr(self, '_reject_count', 0) + 1
                    
        except Exception as e:
            logger.error(f"Find opportunities error: {e}")
    
    async def _check_long_short_ratio(self, symbol: str, direction: str, client: BybitV5Client) -> Tuple[bool, float, str]:
        """
        LONG/SHORT RATIO ANALYSIS - CRITICAL for sentiment!
        
        Bybit API: /v5/market/account-ratio
        
        Ratio interpretation:
        - Ratio > 1.5: Too many longs = potential reversal DOWN
        - Ratio 1.0-1.5: Healthy long bias = bullish
        - Ratio 0.7-1.0: Healthy short bias = bearish
        - Ratio < 0.7: Too many shorts = potential reversal UP
        
        Returns: (supports_trade, ratio, reasoning)
        """
        try:
            ls_data = await client.get_long_short_ratio(symbol, period="1h")
            
            if ls_data.get('retCode') != 0:
                return True, 1.0, "L/S ratio unavailable"
            
            ls_list = ls_data.get('result', {}).get('list', [])
            if not ls_list:
                return True, 1.0, "No L/S data"
            
            # buyRatio is the long percentage
            buy_ratio = float(ls_list[0].get('buyRatio', 0.5))
            sell_ratio = float(ls_list[0].get('sellRatio', 0.5))
            
            # Calculate actual ratio
            ls_ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1.0
            
            reasoning = ""
            supports = True
            
            if ls_ratio > 2.0:
                # Way too many longs - expect crash
                if direction == 'long':
                    supports = False
                    reasoning = f"DANGER: L/S ratio {ls_ratio:.2f} - too many longs, expect dump"
                else:
                    reasoning = f"GOOD: L/S ratio {ls_ratio:.2f} - crowded long, good for short"
                    
            elif ls_ratio > 1.5:
                # Many longs - be cautious
                if direction == 'long':
                    reasoning = f"CAUTION: L/S ratio {ls_ratio:.2f} - many longs, late entry"
                else:
                    reasoning = f"OK: L/S ratio {ls_ratio:.2f} - short against crowd"
                    
            elif ls_ratio < 0.5:
                # Way too many shorts - expect squeeze
                if direction == 'short':
                    supports = False
                    reasoning = f"DANGER: L/S ratio {ls_ratio:.2f} - too many shorts, expect squeeze"
                else:
                    reasoning = f"GOOD: L/S ratio {ls_ratio:.2f} - crowded short, good for long"
                    
            elif ls_ratio < 0.7:
                # Many shorts - be cautious for shorts
                if direction == 'short':
                    reasoning = f"CAUTION: L/S ratio {ls_ratio:.2f} - many shorts, late entry"
                else:
                    reasoning = f"OK: L/S ratio {ls_ratio:.2f} - long against crowd"
            else:
                # Balanced - neutral
                reasoning = f"NEUTRAL: L/S ratio {ls_ratio:.2f} - balanced market"
            
            # Cache for dashboard display
            await self.redis_client.hset(f"ls_ratio:{symbol}", "ratio", str(ls_ratio))
            await self.redis_client.hset(f"ls_ratio:{symbol}", "timestamp", datetime.utcnow().isoformat())
            
            return supports, ls_ratio, reasoning
            
        except Exception as e:
            logger.debug(f"L/S ratio check error for {symbol}: {e}")
            return True, 1.0, "L/S check failed"
    
    async def _check_funding_rate(self, symbol: str, direction: str, client: BybitV5Client) -> Tuple[bool, float, str]:
        """
        FUNDING RATE ANALYSIS
        
        Funding rates indicate market sentiment:
        - Positive funding (>0.01%): Longs pay shorts = Market is overleveraged long = BEARISH signal
        - Negative funding (<-0.01%): Shorts pay longs = Market is overleveraged short = BULLISH signal
        - Extreme funding (>0.05%): Potential reversal incoming
        
        Returns: (is_favorable, funding_rate, reasoning)
        - is_favorable: True if funding supports the trade direction
        - funding_rate: Current funding rate as percentage
        - reasoning: Human-readable explanation
        """
        try:
            # Get funding rate from Bybit
            funding_data = await client.get_funding_rate(symbol)
            
            if funding_data.get('retCode') != 0:
                return True, 0, "Funding data unavailable"
            
            funding_list = funding_data.get('result', {}).get('list', [])
            if not funding_list:
                return True, 0, "No funding data"
            
            funding_rate = float(funding_list[0].get('fundingRate', 0)) * 100  # Convert to percentage
            
            reasoning = ""
            is_favorable = True
            
            # Analyze funding rate
            if funding_rate > 0.05:
                # Extremely high positive funding - market very long, expect reversal down
                if direction == 'long':
                    is_favorable = False
                    reasoning = f"EXTREME positive funding ({funding_rate:.3f}%) - market overleveraged long, risky for longs"
                else:
                    reasoning = f"EXTREME positive funding ({funding_rate:.3f}%) - shorts being paid, good for shorts"
                    
            elif funding_rate > 0.01:
                # High positive funding - longs paying shorts
                if direction == 'long':
                    # Slightly unfavorable for longs but not blocking
                    reasoning = f"Positive funding ({funding_rate:.3f}%) - longs paying, slight headwind"
                else:
                    reasoning = f"Positive funding ({funding_rate:.3f}%) - getting paid to short"
                    
            elif funding_rate < -0.05:
                # Extremely negative funding - market very short, expect reversal up
                if direction == 'short':
                    is_favorable = False
                    reasoning = f"EXTREME negative funding ({funding_rate:.3f}%) - market overleveraged short, risky for shorts"
                else:
                    reasoning = f"EXTREME negative funding ({funding_rate:.3f}%) - longs being paid, good for longs"
                    
            elif funding_rate < -0.01:
                # Negative funding - shorts paying longs
                if direction == 'short':
                    reasoning = f"Negative funding ({funding_rate:.3f}%) - shorts paying, slight headwind"
                else:
                    reasoning = f"Negative funding ({funding_rate:.3f}%) - getting paid to go long"
                    
            else:
                # Neutral funding
                reasoning = f"Neutral funding ({funding_rate:.3f}%)"
            
            logger.debug(f" Funding {symbol}: {funding_rate:.4f}% | {direction} | {reasoning}")
            
            return is_favorable, funding_rate, reasoning
            
        except Exception as e:
            logger.debug(f"Funding rate check failed for {symbol}: {e}")
            return True, 0, f"Funding check error: {e}"
    
    def _check_momentum_fast(self, symbol: str) -> Tuple[bool, Optional[float]]:
        """
        FAST MOMENTUM CHECK using cached ticker data
        
        Uses pre-fetched bulk ticker data instead of individual API calls.
        This is INSTANT instead of ~200ms per symbol!
        
        Returns: (has_momentum, momentum_score)
        - has_momentum: True if price is rising
        - momentum_score: Percentage change (can be None if no data)
        """
        # Get cached momentum data (populated by _get_all_tickers)
        momentum_cache = getattr(self, '_ticker_momentum', {})
        
        if symbol not in momentum_cache:
            # No cached data - allow trade but skip threshold check
            return True, None
        
        data = momentum_cache[symbol]
        momentum_score = data.get('momentum', 0)
        is_rising = data.get('is_rising', False)
        pct_24h = data.get('pct_24h', 0)
        
        # Has momentum if:
        # - 24h change is positive, OR
        # - Recent momentum > 0
        has_momentum = is_rising or momentum_score > 0
        
        logger.debug(f" Fast Momentum {symbol}: 24h={pct_24h:.2f}%, momentum={momentum_score:.3f}%, rising={is_rising}")
        
        return has_momentum, momentum_score
    
    async def _check_momentum(self, symbol: str, client: BybitV5Client) -> Tuple[bool, Optional[float]]:
        """
        MOMENTUM FILTER - now uses FAST cached data!
        
        Falls back to individual API call only if cache miss.
        """
        # Try fast check first (uses cached bulk ticker data)
        has_momentum, momentum_score = self._check_momentum_fast(symbol)
        
        if momentum_score is not None:
            return has_momentum, momentum_score
        
        # Fallback: Individual API call (slow, only if cache miss)
        try:
            kline_data = await client.get_kline(symbol, interval="1", category="linear", limit=6)
            
            if not kline_data.get('result', {}).get('list'):
                return True, None
                
            candles = kline_data['result']['list']
            
            if len(candles) < 5:
                return True, None
            
            green_candles = 0
            total_change = 0
            
            for i in range(min(5, len(candles))):
                candle = candles[i]
                open_price = float(candle[1])
                close_price = float(candle[4])
                
                if close_price > open_price:
                    green_candles += 1
                    
                change = (close_price - open_price) / open_price * 100
                total_change += change
            
            momentum_score = total_change / 5
            has_momentum = green_candles >= 3 or total_change > 0.05
            
            logger.debug(f" Momentum {symbol}: {green_candles}/5 green, avg={momentum_score:.4f}%")
            
            return has_momentum, momentum_score
            
        except Exception as e:
            logger.debug(f"Momentum check failed for {symbol}: {e}")
            return True, None
    
    async def _analyze_news_sentiment(self, symbol: str) -> Tuple[float, str]:
        """
        NEWS SENTIMENT ANALYSIS
        
        Reads cached news and determines if sentiment supports the trade.
        Returns: (sentiment_score: -1 to +1, reason: str)
        
        - Positive score (>0.2): Bullish news
        - Negative score (<-0.2): Bearish news
        - Neutral (-0.2 to +0.2): No clear direction
        """
        try:
            # Get cached news from Redis
            news_raw = await self.redis_client.get("market:news:cache")
            if not news_raw:
                return 0, "No news data"
            
            news_list = json.loads(news_raw)
            if not news_list:
                return 0, "Empty news"
            
            # Extract base symbol (BTCUSDT -> BTC)
            base_symbol = symbol.replace('USDT', '').replace('PERP', '')
            
            # Find news related to this symbol
            symbol_news = []
            bullish_count = 0
            bearish_count = 0
            
            for news in news_list:
                title = news.get('title', '').upper()
                sentiment = news.get('sentiment', 'neutral')
                
                # Check if news mentions this symbol or its common names
                if base_symbol in title or (base_symbol == 'BTC' and 'BITCOIN' in title) or (base_symbol == 'ETH' and 'ETHEREUM' in title):
                    symbol_news.append(news)
                    if sentiment == 'bullish':
                        bullish_count += 1
                    elif sentiment == 'bearish':
                        bearish_count += 1
            
            if not symbol_news:
                # Check general market sentiment
                for news in news_list:
                    sentiment = news.get('sentiment', 'neutral')
                    if sentiment == 'bullish':
                        bullish_count += 1
                    elif sentiment == 'bearish':
                        bearish_count += 1
                
                total = bullish_count + bearish_count
                if total > 0:
                    market_sentiment = (bullish_count - bearish_count) / total
                    return market_sentiment * 0.5, f"Market sentiment: {bullish_count}B/{bearish_count}Be"
                return 0, "No relevant news"
            
            # Calculate sentiment score
            total = bullish_count + bearish_count
            if total > 0:
                sentiment_score = (bullish_count - bearish_count) / total
                return sentiment_score, f"{base_symbol} news: {bullish_count} bullish, {bearish_count} bearish"
            
            return 0, "Neutral news"
            
        except Exception as e:
            logger.debug(f"News sentiment analysis error: {e}")
            return 0, "Analysis error"
            
    async def _validate_opportunity(self, opp: TradingOpportunity, wallet: Dict, client: BybitV5Client = None) -> Tuple[bool, str]:
        """
        SUPERIOR validation using ALL AI models
        
        Checks:
        1. Basic edge/confidence
        2. NEWS SENTIMENT (reads and understands news!)
        3. MOMENTUM FILTER (LOCK PROFIT only!)
        4. XGBoost ML classification
        5. CryptoBERT sentiment (crypto-specific)
        6. Price predictor (multi-timeframe)
        7. Capital allocator (unified budget)
        8. Regime detection
        9. Position sizing
        """
        
        # 0. BREAKOUT BYPASS - If this is a breakout trade, skip most filters
        is_breakout = "BREAKOUT" in str(opp.reasons)
        if is_breakout:
            logger.info(f" Breakout trade {opp.symbol} - bypassing strict filters")
            # Only check basic risk limits for breakouts
            total_equity = float(wallet.get('totalEquity', 0))
            if total_equity < 10:
                return False, "Insufficient equity"
            return True, "Breakout trade approved"
        
        # 1. Edge check
        if opp.edge_score < self.min_edge:
            self.stats['trades_rejected_low_edge'] += 1
            return False, f"Edge too low ({opp.edge_score:.2f} < {self.min_edge})"
            
        # 2. Confidence check
        if opp.confidence < self.min_confidence:
            self.stats['trades_rejected_low_edge'] += 1
            return False, f"Confidence too low ({opp.confidence:.1f} < {self.min_confidence})"
        
        # 2.5 NEWS SENTIMENT ANALYSIS - AI reads and understands news!
        try:
            news_sentiment, news_reason = await self._analyze_news_sentiment(opp.symbol)
            
            # Strong negative news blocks long trades
            if news_sentiment < -0.5 and opp.direction == 'long':
                self.stats['trades_rejected_regime'] += 1
                return False, f"News strongly bearish: {news_reason}"
            
            # Strong positive news blocks short trades  
            if news_sentiment > 0.5 and opp.direction == 'short':
                self.stats['trades_rejected_regime'] += 1
                return False, f"News strongly bullish: {news_reason}"
            
            # Boost confidence if news aligns with direction
            if (news_sentiment > 0.3 and opp.direction == 'long') or (news_sentiment < -0.3 and opp.direction == 'short'):
                opp.confidence = min(100, opp.confidence + 10)
                logger.debug(f"📰 News supports {opp.symbol} {opp.direction}: {news_reason}")
                
        except Exception as e:
            logger.debug(f"News sentiment check skipped: {e}")
        
        # 3. MOMENTUM FILTER - For LOCK PROFIT and MICRO PROFIT strategies
        # This increases win rate by only buying when price is already rising
        if self.risk_mode in ["lock_profit", "micro_profit"] and client:
            has_momentum, momentum_score = await self._check_momentum(opp.symbol, client)
            
            # MICRO PROFIT with threshold=0 disables momentum filter entirely
            if self.risk_mode == "micro_profit" and self.momentum_threshold <= 0:
                # Momentum filter disabled - skip all checks
                logger.debug(f" Momentum filter DISABLED for {opp.symbol} (threshold=0)")
            else:
                # Standard momentum check: need 3/5 green candles or positive total change
                if not has_momentum:
                    self.stats['trades_rejected_no_momentum'] += 1
                    mode_name = "MICRO PROFIT" if self.risk_mode == "micro_profit" else "LOCK PROFIT"
                    return False, f"No momentum for {mode_name} ({opp.symbol} not rising)"
                
                # MICRO PROFIT with threshold>0 needs momentum_score above threshold
                # Skip if momentum_score is None (no data available)
                if self.risk_mode == "micro_profit" and self.momentum_threshold > 0 and momentum_score is not None:
                    if momentum_score < self.momentum_threshold:
                        self.stats['trades_rejected_no_momentum'] += 1
                        return False, f"MICRO PROFIT needs stronger momentum ({momentum_score:.3f}% < {self.momentum_threshold}%)"
                
                # Log positive momentum
                if momentum_score is not None:
                    logger.debug(f"Momentum OK for {opp.symbol}: score={momentum_score:.4f}%")
                else:
                    logger.debug(f"Momentum data unavailable for {opp.symbol}, skipping threshold check")
        
        # 3.5 FUNDING RATE ANALYSIS - Check if funding supports the trade
        if client:
            funding_favorable, funding_rate, funding_reason = await self._check_funding_rate(opp.symbol, opp.direction, client)
            
            # Block trade if funding is extremely unfavorable (>0.05% against us)
            if not funding_favorable:
                self.stats['trades_rejected_regime'] += 1
                return False, f"Funding unfavorable: {funding_reason}"
            
            # Store funding rate for position tracking
            opp.funding_rate = funding_rate
            opp.funding_reason = funding_reason
        
        # 3.55 LONG/SHORT RATIO - Critical sentiment indicator!
        if client:
            try:
                ls_supports, ls_ratio, ls_reason = await self._check_long_short_ratio(opp.symbol, opp.direction, client)
                
                # Block if extreme L/S ratio against us
                if not ls_supports:
                    self.stats['trades_rejected_regime'] += 1
                    return False, f"L/S ratio unfavorable: {ls_reason}"
                
                # Log if notable
                if "GOOD" in ls_reason or "CAUTION" in ls_reason:
                    logger.debug(f" L/S ratio for {opp.symbol}: {ls_reason}")
                    
            except Exception as e:
                logger.debug(f"L/S ratio check skipped: {e}")
        
        # 3.6 WHALE TRACKING - Check if whale activity supports the trade
        try:
            whale_supports, whale_reason = await whale_tracker.check_whale_support(opp.symbol, opp.direction)
            if not whale_supports:
                self.stats['trades_rejected_regime'] += 1
                return False, f"Whale activity against trade: {whale_reason}"
            
            # Log whale support
            if 'supports' in whale_reason.lower():
                logger.debug(f"🐋 Whale support for {opp.symbol}: {whale_reason}")
        except Exception as e:
            logger.debug(f"Whale check skipped for {opp.symbol}: {e}")
            
        # 4. XGBoost ML Classification
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
        
        # 7.5 MICRO PROFIT EXTRA VALIDATION - Must have strong agreement
        if self.risk_mode == "micro_profit":
            # For MICRO PROFIT, edge must be higher to ensure quality trades
            if opp.edge_score < 0.10:  # At least 10% edge
                return False, f" MICRO PROFIT needs stronger edge ({opp.edge_score:.2f} < 0.10)"
            
            # Check confidence is high enough
            if opp.confidence < 65:  # Slightly stricter
                return False, f" MICRO PROFIT needs higher confidence ({opp.confidence:.0f}% < 65%)"
            
            logger.info(f" MICRO PROFIT approved: {opp.symbol} edge={opp.edge_score:.2f} conf={opp.confidence:.0f}%")
            
        # 8. Risk check via position sizer
        if not opp.edge_data:
            return False, "No edge data"
        
        # Calculate position size (dynamic or fixed)
        if self.use_dynamic_sizing:
            # Dynamic sizing using Kelly Criterion
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
        else:
            # Fixed sizing - use maxPositionPercent of wallet
            position_size = await self.position_sizer.calculate_position_size(
                symbol=opp.symbol,
                direction=opp.direction,
                edge_score=opp.edge_score,
                win_probability=50,  # Neutral - no edge adjustment
                risk_reward=1.5,     # Standard R/R
                kelly_fraction=0.0,  # No Kelly adjustment
                regime_action='normal',
                current_price=opp.current_price,
                wallet_balance=wallet['total_equity'],
                force_fixed=True  # Force fixed percentage sizing
            )
        
        if not position_size.is_within_limits:
            self.stats['trades_rejected_risk'] += 1
            return False, position_size.limit_reason
            
        # Store position size for execution
        opp.edge_data._position_size = position_size
        
        return True, "SUPERIOR: Passed ALL AI checks "
        
    async def _execute_trade(self, user_id: str, client: BybitV5Client,
                             opp: TradingOpportunity, wallet: Dict):
        """Execute a trade"""
        try:
            edge_data = opp.edge_data
            pos_size: PositionSize = getattr(edge_data, '_position_size', None) if edge_data else None
            
            # === BREAKOUT TRADES: Calculate position size manually ===
            is_breakout = "BREAKOUT" in str(opp.reasons)
            
            if is_breakout or not pos_size:
                # Calculate position value based on settings
                # Support both snake_case (internal) and camelCase (API) keys
                total_equity = float(wallet.get('total_equity', wallet.get('totalEquity', 0)))
                if total_equity < 10:
                    logger.warning(f"Insufficient equity for {opp.symbol}: ${total_equity:.2f}")
                    return
                
                # Use max_position_percent from settings (default 5%)
                max_pos_pct = getattr(self, 'max_position_percent', 5)
                base_position_value = total_equity * (max_pos_pct / 100)
                
                # Apply size multiplier for extreme breakouts
                size_multiplier = getattr(opp, 'size_multiplier', 1.0)
                position_value = base_position_value * size_multiplier
                
                # Minimum $10 position
                position_value = max(10, min(position_value, total_equity * 0.2))  # Max 20% per trade
                
                # Determine leverage based on settings
                leverage_mode = getattr(self, 'leverage_mode', 'auto')
                if leverage_mode == 'auto':
                    # Auto: lower leverage for extreme moves
                    if abs(opp.price_change_24h) > 50:
                        leverage = 2
                    elif abs(opp.price_change_24h) > 25:
                        leverage = 3
                    else:
                        leverage = 5
                else:
                    leverage = int(leverage_mode.replace('x', ''))
                
                logger.info(f" BREAKOUT sizing: ${position_value:.0f} | {leverage}x | Size mult: {size_multiplier:.0%}")
            else:
                if pos_size.position_value_usdt < 5:
                    logger.debug(f"Position too small for {opp.symbol}")
                    return
                position_value = pos_size.position_value_usdt
                leverage = pos_size.recommended_leverage
                
            # Get symbol info for quantity precision
            symbol_info = self.market_scanner.get_symbol_info(opp.symbol)
            min_qty = symbol_info.get('min_qty', 0.001)
            qty_step = symbol_info.get('qty_step', 0.001)
            
            # Calculate quantity
            qty_raw = position_value / opp.current_price
            
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
            
            # Leverage already set above for breakouts or from pos_size
            
            # Set leverage on Bybit before placing order
            try:
                leverage_result = await client.set_leverage(
                    symbol=opp.symbol,
                    leverage=str(leverage),
                    category="linear"
                )
                if leverage_result.get('success'):
                    logger.debug(f"Set leverage to {leverage}x for {opp.symbol}")
                else:
                    # If can't set leverage (e.g., has open position), continue with current leverage
                    logger.debug(f"Could not set leverage for {opp.symbol}: {leverage_result.get('error', 'unknown')}")
            except Exception as lev_err:
                logger.debug(f"Leverage set error for {opp.symbol}: {lev_err}")
            
            logger.info(f"EXECUTING: {side} {opp.symbol} | Qty: {qty_str} | Leverage: {leverage}x | "
                       f"Edge: {opp.edge_score:.2f} | Confidence: {opp.confidence:.1f}%")
            
            result = await client.place_order(
                symbol=opp.symbol,
                side=side,
                order_type='Market',
                qty=qty_str  # Pass as string!
            )
            
            if result.get('success'):
                logger.info(f"ORDER SUCCESS: {opp.symbol} {side} {qty}")
                
                # Log to console for dashboard
                leverage_display = f" | {leverage}x" if leverage > 1 else ""
                await self._log_to_console(
                    f"OPENED {opp.symbol} {side} | ${position_value:.0f}{leverage_display} | Edge: {opp.edge_score:.2f} | Conf: {opp.confidence:.0f}%",
                    "TRADE"
                )
                
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
                    position_value=position_value,
                    kelly_fraction=pos_size.kelly_fraction if pos_size else 0.5,
                    leverage=leverage
                )
                
                # Register in position sizer
                await self.position_sizer.register_position(opp.symbol, position_value)
                
                # Store trade event
                await self._store_trade_event(opp.symbol, 'opened', 0, opp.direction)
                
            else:
                error_msg = result.get('error', result.get('message', str(result)))
                logger.error(f"ORDER FAILED: {opp.symbol} - {error_msg}")
                await self._log_to_console(f"ORDER FAILED: {opp.symbol} - {error_msg[:50]}", "ERROR")
                
        except Exception as e:
            logger.error(f"Execute trade error for {opp.symbol}: {e}")
            await self._log_to_console(f"TRADE ERROR: {opp.symbol} - {str(e)[:50]}", "ERROR")
            
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
        """Get ALL tickers in ONE API call - much faster than individual calls
        
        Also caches momentum data (price24hPcnt) for instant momentum checks!
        """
        try:
            result = await client.get_tickers(category="linear")
            if result.get('success'):
                tickers = {}
                momentum_data = {}  # Cache momentum for instant checks
                
                for item in result.get('data', {}).get('list', []):
                    symbol = item.get('symbol')
                    price = safe_float(item.get('lastPrice'))
                    
                    if symbol and price > 0:
                        tickers[symbol] = price
                        
                        # Cache momentum data: price24hPcnt is 24h % change
                        # Also calculate short-term momentum from price vs prevPrice24h
                        price_24h_pct = safe_float(item.get('price24hPcnt', 0)) * 100  # Convert to %
                        prev_price = safe_float(item.get('prevPrice24h', 0))
                        
                        # Calculate recent momentum: current vs 24h ago price
                        if prev_price > 0:
                            recent_momentum = ((price - prev_price) / prev_price) * 100
                        else:
                            recent_momentum = price_24h_pct
                        
                        momentum_data[symbol] = {
                            'pct_24h': price_24h_pct,
                            'momentum': recent_momentum,
                            'is_rising': price_24h_pct > 0
                        }
                
                # Store momentum cache for instant access in _check_momentum_fast
                self._ticker_momentum = momentum_data
                
                logger.debug(f" Bulk fetched {len(tickers)} tickers in 1 API call")
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
                logger.warning(f" No price for {position.symbol}")
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
                    
            # === SMART EXIT LOGIC (MICRO PROFIT) ===
            should_partial_exit = False
            
            if self.use_smart_exit and self.risk_mode == 'micro_profit':
                # 1. BREAKEVEN LOGIC: Move SL to 0% when profit reaches threshold
                if not position.breakeven_active and pnl_percent >= self.breakeven_trigger:
                    position.breakeven_active = True
                    logger.info(f" BREAKEVEN ACTIVATED: {position.symbol} at +{pnl_percent:.2f}% (trigger: +{self.breakeven_trigger}%)")
                
                # 2. BREAKEVEN EXIT: If breakeven is active and price drops to 0%, exit
                if position.breakeven_active and pnl_percent <= 0.05 and pnl_percent >= -0.05:
                    # Price returned to entry - exit at breakeven
                    logger.info(f" BREAKEVEN EXIT: {position.symbol} P&L={pnl_percent:+.2f}% (was up +{position.peak_pnl_percent:.2f}%)")
                    await self._close_position(user_id, client, position, pnl_percent, 
                        f" BREAKEVEN (was +{position.peak_pnl_percent:.2f}%, now {pnl_percent:+.2f}%)")
                    return  # Exit early
                
                # 3. PARTIAL EXIT: Take 50% profit at trigger level
                if not position.partial_exit_done and pnl_percent >= self.partial_exit_trigger:
                    position.partial_exit_done = True
                    # Execute partial close in background
                    success = await self._partial_close_position(
                        user_id, client, position, pnl_percent, self.partial_exit_percent
                    )
                    if success:
                        logger.info(f" PARTIAL EXIT COMPLETE: {position.symbol} at +{pnl_percent:.2f}%")
                    # Continue with remaining position - don't return
            
            # === TIME STOP LOGIC (MICRO PROFIT) ===
            # Close "dead" trades before they go negative
            if self.use_time_stop and self.risk_mode == 'micro_profit':
                trade_duration = (datetime.utcnow() - position.entry_time).total_seconds() / 60  # minutes
                
                if trade_duration >= self.time_stop_minutes and pnl_percent < self.time_stop_min_pnl:
                    logger.info(f" TIME STOP: {position.symbol} after {trade_duration:.1f}min, P&L={pnl_percent:+.2f}% < +{self.time_stop_min_pnl}%")
                    await self._close_position(user_id, client, position, pnl_percent, 
                        f" TIME STOP ({trade_duration:.1f}min, +{pnl_percent:.2f}%)")
                    return  # Exit early
            
            # === FAST EXIT LOGIC ===
            should_exit = False
            exit_reason = ""
            
            # Only log positions near thresholds to reduce spam
            # Take Profit only triggers if TP > 0 (enabled)
            if self.take_profit > 0 and pnl_percent >= self.take_profit:
                logger.info(f" {position.symbol}: P&L={pnl_percent:+.2f}% >= TP={self.take_profit}% - SELLING!")
                await self._log_to_console(f"SOLD {position.symbol}: +{pnl_percent:.2f}% (Take Profit)", "TRADE")
                should_exit = True
                exit_reason = f"Take profit ({pnl_percent:.2f}%)"
            elif pnl_percent <= -self.emergency_stop_loss:
                logger.info(f" {position.symbol}: P&L={pnl_percent:+.2f}% <= SL=-{self.emergency_stop_loss}% - SELLING!")
                await self._log_to_console(f"SOLD {position.symbol}: {pnl_percent:.2f}% (Stop Loss)", "TRADE")
                should_exit = True
                exit_reason = f"Stop loss ({pnl_percent:.2f}%)"
            elif self.take_profit > 0 and pnl_percent >= self.take_profit * 0.8:
                logger.info(f" {position.symbol}: {pnl_percent:+.2f}% near TP")
            elif pnl_percent <= -self.emergency_stop_loss * 0.8:
                logger.info(f" {position.symbol}: {pnl_percent:+.2f}% near SL")
                
            # TRAILING STOP / LOCK PROFIT LOGIC
            is_lock_profit_mode = self.trail_from_peak <= 0.1  # Ultra-tight = LOCK PROFIT
            
            # Calculate drop from peak
            if position.side == 'Buy':
                drop_from_peak = ((position.peak_price - current_price) / position.peak_price) * 100
            else:
                drop_from_peak = ((current_price - position.trough_price) / position.trough_price) * 100
            
            if not should_exit:
                if is_lock_profit_mode:
                    # === LOCK PROFIT MODE ===
                    # Activate as soon as we've EVER been in profit (peak >= 0.01%)
                    if position.peak_pnl_percent >= self.min_profit_to_trail:
                        position.trailing_active = True
                        
                        # Log EVERY check for debugging
                        logger.info(f" LOCK_PROFIT {position.symbol}: Peak={position.peak_pnl_percent:+.3f}%, Now={pnl_percent:+.3f}%, Drop={drop_from_peak:.3f}%, Trigger={self.trail_from_peak:.3f}%")
                        
                        # EXIT if dropped from peak by threshold
                        # BUT ONLY IF current P&L is still positive (or break-even)!
                        # This ensures we ALWAYS lock profit, never lock loss
                        if drop_from_peak >= self.trail_from_peak and pnl_percent >= -0.02:
                            # Only sell if we're still in profit or at worst break-even (-0.02% for fees)
                            if pnl_percent >= 0:
                                should_exit = True
                                exit_reason = f" LOCK PROFIT (peak: {position.peak_pnl_percent:.2f}%, now: {pnl_percent:+.2f}%)"
                                logger.info(f" LOCK PROFIT SELL: {position.symbol} | Peak={position.peak_pnl_percent:+.2f}%, Now={pnl_percent:+.2f}%  PROFIT!")
                            else:
                                # Price dropped too fast, don't sell at loss - wait for recovery
                                logger.debug(f" {position.symbol}: Dropped but P&L negative ({pnl_percent:+.2f}%), waiting for recovery")
                else:
                    # === NORMAL TRAILING MODE ===
                    # Activate trailing when profit reaches min_profit_to_trail (e.g., 0.5%)
                    if pnl_percent >= self.min_profit_to_trail:
                        position.trailing_active = True
                    
                    # Once trailing is active, sell when price drops from peak
                    if position.trailing_active and drop_from_peak >= self.trail_from_peak:
                        # Only sell if we're still in profit (or minimal loss)
                        if pnl_percent >= -0.05:  # Allow tiny loss due to spread
                            should_exit = True
                            exit_reason = f"Trailing stop (peak: +{position.peak_pnl_percent:.2f}%, dropped {drop_from_peak:.2f}%)"
                            logger.info(f"📉 TRAILING SELL {position.symbol}: Peak=+{position.peak_pnl_percent:.2f}%, Now={pnl_percent:+.2f}%, Drop={drop_from_peak:.2f}% >= {self.trail_from_peak}%")
                        else:
                            logger.debug(f" {position.symbol}: Trailing triggered but P&L too negative ({pnl_percent:+.2f}%), holding")
                    
            # === EXECUTE EXIT ===
            if should_exit:
                await self._close_position(user_id, client, position, pnl_percent, exit_reason)
                
        except Exception as e:
            logger.error(f"Fast exit check error for {position.symbol}: {e}")
            
    async def _log_to_console(self, message: str, level: str = "INFO"):
        """Log message to Redis for dashboard real-time console"""
        try:
            if self.redis_client:
                log_entry = {
                    "time": datetime.utcnow().isoformat(),
                    "level": level,
                    "message": message
                }
                await self.redis_client.lpush('bot:console:logs', json.dumps(log_entry))
                await self.redis_client.ltrim('bot:console:logs', 0, 99)  # Keep last 100
        except Exception:
            pass  # Don't break trading for console logging
        
    async def _store_trade_event(self, symbol: str, action: str, pnl: float, reason: str,
                                  position: Optional[ActivePosition] = None, 
                                  net_pnl_value: Optional[float] = None):
        """Store trade event for dashboard and data collection
        
        Args:
            pnl: NET P&L percentage (after fees)
            net_pnl_value: NET P&L value in USD (after fees)
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # Use provided net_pnl_value or calculate from percentage
            if net_pnl_value is not None:
                pnl_value = net_pnl_value
            elif position and position.position_value:
                pnl_value = position.position_value * (pnl / 100)
            else:
                pnl_value = 0
            
            event = {
                'symbol': symbol,
                'action': action,
                'pnl_percent': round(pnl, 2),  # NET %
                'pnl_value': round(pnl_value, 2),  # NET value
                'reason': reason,
                'timestamp': timestamp,
                'side': position.side if position else None,
                'entry_price': position.entry_price if position else None,
                'value': position.position_value if position else None
            }
            
            # Push to Redis list for dashboard
            await self.redis_client.lpush('trading:events', json.dumps(event))
            await self.redis_client.ltrim('trading:events', 0, 99)  # Keep last 100
            
            # Also store completed trades separately for activity panel (NET P&L)
            if action == 'closed':
                await self.redis_client.lpush('trades:completed:default', json.dumps({
                    'symbol': symbol,
                    'pnl': round(pnl_value, 2),  # NET value
                    'pnl_percent': round(pnl, 2),  # NET %
                    'close_reason': reason,
                    'closed_time': timestamp
                }))
                await self.redis_client.ltrim('trades:completed:default', 0, 49)  # Keep last 50
            
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
                    leverage=position.leverage,
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
                
                # Get risk mode from settings (or detect from parameters for backwards compatibility)
                saved_risk_mode = parsed.get('riskMode', 'normal')
                if saved_risk_mode in ['lock_profit', 'micro_profit', 'safe', 'aggressive']:
                    self.risk_mode = saved_risk_mode
                else:
                    # Backwards compatibility: detect from trailing settings
                    is_lock_profit = self.trail_from_peak <= 0.1 and self.min_profit_to_trail <= 0.05
                    self.risk_mode = "lock_profit" if is_lock_profit else "normal"
                
                # Smart exit settings (breakeven + partial exits)
                self.breakeven_trigger = float(parsed.get('breakevenTrigger', 0.3))
                self.partial_exit_trigger = float(parsed.get('partialExitTrigger', 0.4))
                self.partial_exit_percent = float(parsed.get('partialExitPercent', 50))
                self.momentum_threshold = float(parsed.get('momentumThreshold', 0.05))
                
                # Enable smart exit for MICRO PROFIT mode automatically
                self.use_smart_exit = self.risk_mode == 'micro_profit' or parsed.get('useSmartExit', 'false').lower() == 'true'
                
                # Time stop settings (MICRO PROFIT)
                self.time_stop_minutes = float(parsed.get('timeStopMinutes', 4))
                self.time_stop_min_pnl = float(parsed.get('timeStopMinPnl', 0.15))
                self.use_time_stop = self.risk_mode == 'micro_profit'  # Auto-enable for MICRO PROFIT
                
                # AI Model toggles
                self.use_dynamic_sizing = parsed.get('useDynamicSizing', 'true').lower() == 'true'
                self.use_regime_detection = parsed.get('useRegimeDetection', 'true').lower() == 'true'
                self.use_edge_estimation = parsed.get('useEdgeEstimation', 'true').lower() == 'true'
                self.use_crypto_bert = parsed.get('useCryptoBert', 'true').lower() == 'true'
                self.use_xgboost_classifier = parsed.get('useXgboostClassifier', 'true').lower() == 'true'
                self.use_price_predictor = parsed.get('usePricePredictor', 'true').lower() == 'true'
                self.use_whale_detection = parsed.get('useWhaleDetection', 'true').lower() == 'true'
                self.use_funding_rate = parsed.get('useFundingRate', 'true').lower() == 'true'
                self.use_pattern_recognition = parsed.get('usePatternRecognition', 'true').lower() == 'true'
                self.use_q_learning = parsed.get('useQLearning', 'true').lower() == 'true'
                
                # Position sizing
                self.max_position_percent = float(parsed.get('maxPositionPercent', 5))
                self.kelly_multiplier = float(parsed.get('kellyMultiplier', 0.5))
                
                # Leverage mode
                self.leverage_mode = parsed.get('leverageMode', 'auto')
                if self.leverage_mode not in ['1x', '2x', '3x', '5x', '10x', 'auto']:
                    self.leverage_mode = 'auto'
                
                # Sync ALL settings to position sizer (HOT RELOAD)
                if self.position_sizer:
                    self.position_sizer.leverage_mode = self.leverage_mode
                    self.position_sizer.MAX_OPEN_POSITIONS = self.max_open_positions
                    self.position_sizer.MAX_TOTAL_EXPOSURE = self.max_exposure_percent / 100  # Convert to fraction
                    self.position_sizer.MAX_DAILY_DRAWDOWN = self.max_daily_drawdown / 100  # Convert to fraction
                    self.position_sizer.max_position_percent = self.max_position_percent
                    self.position_sizer.kelly_multiplier = self.kelly_multiplier
                    self.position_sizer.use_dynamic_sizing = self.use_dynamic_sizing
                
                # Mode display names
                mode_names = {
                    'lock_profit': ' LOCK PROFIT',
                    'micro_profit': ' MICRO PROFIT',
                    'safe': ' SAFE',
                    'aggressive': ' AGGRESSIVE',
                    'normal': ' NORMAL'
                }
                mode_name = mode_names.get(self.risk_mode, ' NORMAL')
                
                # Update market scanner's risk mode for optimized scanning
                if self.market_scanner:
                    self.market_scanner.set_risk_mode(self.risk_mode)
                
                # Only log on first load or when settings actually change
                tp_display = f"{self.take_profit}%" if self.take_profit > 0 else "OFF"
                leverage_display = self.leverage_mode.upper()
                new_settings_str = f"Mode={self.risk_mode},SL={self.emergency_stop_loss}%,TP={tp_display},Trail={self.trail_from_peak}%,MinTrail={self.min_profit_to_trail}%,Lev={self.leverage_mode}"
                if not hasattr(self, '_last_settings_str') or self._last_settings_str != new_settings_str:
                    logger.info(f" Settings [{mode_name}]: SL={self.emergency_stop_loss}%, TP={tp_display}, "
                               f"Trail={self.trail_from_peak}%, MinProfitToTrail={self.min_profit_to_trail}%, "
                               f"MinConf={self.min_confidence}%, MinEdge={self.min_edge}, "
                               f"MaxPos={'Unlimited' if self.max_open_positions == 0 else self.max_open_positions}, "
                               f"Leverage={leverage_display}")
                    if self.take_profit == 0:
                        logger.info(f" TRAILING ONLY MODE: No Take Profit limit, sells only when trailing stop triggers")
                    if self.risk_mode == 'lock_profit':
                        logger.info(f" LOCK PROFIT MODE: Sells on {self.trail_from_peak}% drop from peak")
                    elif self.risk_mode == 'micro_profit':
                        logger.info(f" MICRO PROFIT MODE: Quick +{self.take_profit}% profits, -{self.emergency_stop_loss}% stop loss, HIGH confidence required")
                    self._last_settings_str = new_settings_str
        except Exception as e:
            logger.debug(f"Load settings error: {e}")
            
    async def _load_stats(self):
        """Load stats from Redis, preserving any new keys"""
        try:
            data = await self.redis_client.get('trader:stats')
            if data:
                loaded_stats = json.loads(data)
                # Merge loaded stats with current (keeping new keys with defaults)
                for key, value in loaded_stats.items():
                    if key in self.stats:
                        self.stats[key] = value
                # Ensure new keys exist
                if 'trades_rejected_no_momentum' not in self.stats:
                    self.stats['trades_rejected_no_momentum'] = 0
        except:
            pass
    
    async def _load_paused_users(self):
        """Load paused users from Redis"""
        try:
            keys = await self.redis_client.keys('trading:paused:*')
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                user_id = key_str.replace('trading:paused:', '')
                self.paused_users.add(user_id)
            if self.paused_users:
                logger.info(f" Loaded {len(self.paused_users)} paused users: {self.paused_users}")
        except Exception as e:
            logger.warning(f"Failed to load paused users: {e}")
            
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

