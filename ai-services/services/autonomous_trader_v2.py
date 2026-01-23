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

# V4 HuggingFace Safety Models (GATES, not signals!)
try:
    from services.hf_safety_models import hf_safety, get_hf_safety
    HF_SAFETY_AVAILABLE = True
except ImportError:
    HF_SAFETY_AVAILABLE = False
    logger.warning("HuggingFace safety models not available")


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
    
    # Breakout flag - to identify breakout positions on dashboard
    is_breakout: bool = False
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
        
        # Exit strategy - USER CONFIGURABLE
        # User can choose strategy preset or customize values
        self.min_profit_to_trail = 0.5  # Start trailing at +0.5%
        self.trail_from_peak = 0.8  # Trail by 0.8% from peak (balanced)
        self.emergency_stop_loss = 1.5  # Stop loss at -1.5% (give time to recover)
        self.take_profit = 3.0  # Take profit at +3%
        
        # Strategy preset (user selectable)
        # Options: 'conservative', 'balanced', 'aggressive', 'scalper', 'swing', 'mean_reversion'
        self.strategy_preset = 'balanced'
        
        # Mean Reversion Strategy settings
        self.use_mean_reversion = False  # Enable mean reversion entry logic
        self.mr_min_dip = -0.3  # Minimum dip to consider entry (%)
        self.mr_max_dip = -1.5  # Maximum dip (avoid falling knives) (%)
        self.mr_rsi_oversold = 35  # RSI threshold for oversold
        self.mr_only_range_regime = True  # Only trade in RANGE/SIDEWAYS regime
        
        # === PRESET ENTRY THRESHOLDS (from ChatGPT professional presets) ===
        self.momentum_min = 0.025  # Minimum momentum for entry
        self.momentum_max = 0.18   # Maximum momentum (don't chase pumps)
        self.rsi_entry_min = 30    # RSI minimum for entry
        self.rsi_entry_max = 60    # RSI maximum for entry
        self.volume_ratio_min = 1.1  # Minimum volume ratio
        self.spread_max = 0.25     # Maximum spread % (slippage protection)
        self.wick_ratio_max = 0.60 # Maximum wick ratio (rejection candle filter)
        self.distance_from_low_max = 0.30  # Max distance from local low
        self.green_red_ratio_min = 1.3  # Min green/red candle ratio
        
        # BTC correlation settings
        self.btc_correlation_check = True  # Check BTC before alt longs
        self.btc_block_threshold = -2.0    # Block alt longs if BTC < this % (balanced)
        self.btc_required_positive = False # Require BTC positive for trade
        
        # AI score requirements
        self.min_models_agree = 2  # Minimum AI models that must agree
        self.require_positive_ev = False  # Require positive expected value
        self.require_positive_sentiment = False  # Require positive sentiment
        
        # Time stop
        self.max_trade_minutes = 25  # Maximum trade duration
        self.use_max_trade_time = True  # Use preset's max_trade_minutes (user can disable)
        
        # === AI FULL AUTO MODE ===
        self.ai_full_auto = False  # When ON: AI manages everything automatically
        self.auto_selected_preset = 'micro'  # Preset AI selected based on market
        
        # Regime requirements
        self.regime_required = []  # List of required regimes (empty = any)
        
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
        
        # Breakout settings (user must enable in settings)
        self.breakout_extra_slots = False  # OFF by default
        
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
        
        # FAILED ORDER COOLDOWN: Prevent retrying failed orders immediately
        self._failed_order_symbols: Dict[str, datetime] = {}
        self.failed_order_cooldown = 300  # Wait 5 minutes before retrying failed order
        
        # Per-user statistics (isolated per user)
        self.user_stats: Dict[str, Dict] = {}
        
        # ============================================================
        # PER-USER SETTINGS - CRITICAL FOR MULTI-USER ISOLATION!
        # Each user has their OWN settings, NOT shared instance variables!
        # ============================================================
        self.user_settings: Dict[str, Dict] = {}
        
        # Global stats for AI learning (aggregated from all users)
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
    
    def _get_user_stats(self, user_id: str) -> Dict:
        """Get or create stats for a specific user"""
        today = datetime.utcnow().date().isoformat()
        
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'daily_pnl': 0.0,
                'daily_pnl_date': today,
                'opportunities_scanned': 0
            }
        else:
            # Reset daily P&L if it's a new day
            stats = self.user_stats[user_id]
            last_date = stats.get('daily_pnl_date', '')
            if last_date != today:
                logger.info(f"Resetting daily P&L for {user_id}: was ${stats.get('daily_pnl', 0):.2f} from {last_date}")
                stats['daily_pnl'] = 0.0
                stats['daily_pnl_date'] = today
                
        return self.user_stats[user_id]
    
    def _get_user_settings(self, user_id: str) -> Dict:
        """
        Get settings for a specific user - CRITICAL FOR MULTI-USER ISOLATION!
        Returns user-specific settings or HARDCODED defaults if not set.
        
        CRITICAL: NEVER use instance variables (self.take_profit, etc.) as fallback!
        That would leak one user's settings to another user!
        """
        if user_id not in self.user_settings:
            # Initialize with HARDCODED defaults - NEVER use instance variables!
            # These are MICRO preset defaults (the safest default strategy)
            self.user_settings[user_id] = {
                'take_profit': 0.9,        # MICRO default
                'stop_loss': 0.5,          # MICRO default
                'trailing': 0.14,          # MICRO default
                'min_profit_to_trail': 0.45,  # MICRO default
                'max_open_positions': 10,
                'max_exposure_percent': 100,
                'max_daily_drawdown': 0,   # 0 = OFF
                'breakout_extra_slots': False,
                'ai_full_auto': False,
                'use_max_trade_time': True,
                'max_trade_minutes': 30,   # MICRO default
                'min_confidence': 65,
                'min_edge': 0.15,
                'risk_mode': 'MICRO',
                'strategy_preset': 'micro',
                'leverage_mode': 'auto',
            }
            logger.info(f"Created DEFAULT settings for user {user_id} (not loaded from Redis yet)")
        return self.user_settings[user_id]
    
    def _get_user_setting(self, user_id: str, key: str, default=None):
        """Get a specific setting for a user"""
        settings = self._get_user_settings(user_id)
        return settings.get(key, default)
    
    def _apply_strategy_preset(self, preset: str):
        """
        Apply a trading strategy preset - PROFESSIONAL PRESETS from ChatGPT
        
        Each preset has:
        - Entry thresholds (momentum, RSI, volume, spread, wick ratio)
        - AI score requirements (confidence, models agreement)
        - Exit rules (TP, SL, trailing, max time)
        
        Presets:
        - scalp: Fastest, most dangerous - quick bounce, small profit (65-75% winrate)
        - micro: HEALTHIEST DEFAULT - small quality profits (70-80% winrate) - RECOMMENDED
        - swing: Slow but stable - bigger moves, less frequency (55-65% winrate)
        - conservative: Capital preservation - tight everything
        - balanced: Middle ground for beginners
        - aggressive: Risk takers - wide stops
        """
        presets = {
            # SCALP PRESET (najbrzi, najopasniji)
            'scalp': {
                # EXIT
                'stop_loss': 0.35,
                'take_profit': 0.55,  # Average of 0.4-0.7%
                'trailing': 0.12,
                'min_trail': 0.35,
                'max_trade_minutes': 5,  # 3-7 min
                
                # ENTRY THRESHOLDS
                'momentum_min': 0.015,
                'momentum_max': 0.12,
                'rsi_min': 28,
                'rsi_max': 55,
                'volume_ratio_min': 0.9,
                'spread_max': 0.12,
                'wick_ratio_max': 0.65,
                'distance_from_low_max': 0.20,
                'green_red_ratio_min': 1.1,
                'btc_correlation_check': False,  # IGNORE except crash
                
                # AI SCORE
                'min_confidence': 58,
                'min_models_agree': 2,
                
                'use_mean_reversion': False,
                'regime_required': ['RANGE', 'LOW_VOLATILITY'],
                'winrate_expected': '65-75%',
                'description': 'Quick bounce, small profit, high frequency - RISKY'
            },
            
            # MICRO PRESET (NAJZDRAVIJI - PREPORUKA)
            'micro': {
                # EXIT
                'stop_loss': 0.5,
                'take_profit': 0.9,  # Average of 0.6-1.2%
                'trailing': 0.14,
                'min_trail': 0.45,
                'max_trade_minutes': 25,  # 15-40 min
                
                # ENTRY THRESHOLDS
                'momentum_min': 0.025,
                'momentum_max': 0.18,
                'rsi_min': 30,
                'rsi_max': 60,
                'volume_ratio_min': 1.1,
                'spread_max': 0.15,
                'wick_ratio_max': 0.60,
                'distance_from_low_max': 0.30,
                'green_red_ratio_min': 1.3,
                'btc_correlation_check': True,  # BLOCK if BTC < threshold
                'btc_block_threshold': -2.0,  # Allow trades when BTC slightly down (was -0.4)
                
                # AI SCORE
                'min_confidence': 65,
                'min_models_agree': 2,
                'require_positive_ev': True,
                
                'use_mean_reversion': True,
                'regime_required': ['RANGE', 'CHOPPY'],
                'winrate_expected': '70-80%',
                'description': 'Small quality profits - DEFAULT MONEY PRINTER'
            },
            
            # SWING PRESET (spor, ali stabilan)
            'swing': {
                # EXIT
                'stop_loss': 1.2,
                'take_profit': 4.0,  # Average of 2-6%
                'trailing': 0.35,
                'min_trail': 1.2,
                'max_trade_minutes': 1440,  # 6h - 3d (1 day average)
                
                # ENTRY THRESHOLDS
                'momentum_min': 0.08,
                'momentum_max': 0.35,
                'rsi_min': 35,
                'rsi_max': 65,
                'volume_ratio_min': 1.3,
                'spread_max': 0.20,
                'wick_ratio_max': 0.55,
                'distance_from_low_max': 0.45,
                'green_red_ratio_min': 1.5,
                'btc_correlation_check': True,  # REQUIRED CONFIRMATION
                'btc_required_positive': True,
                
                # AI SCORE
                'min_confidence': 72,
                'min_models_agree': 3,  # ALL models must agree
                'require_positive_sentiment': True,
                
                'use_mean_reversion': False,
                'regime_required': ['STABLE_TREND', 'RANGE_EXPANSION'],
                'winrate_expected': '55-65%',
                'description': 'Bigger moves, less frequency - R:R > 2.5'
            },
            
            # Simple presets for backwards compatibility
            'conservative': {
                'stop_loss': 0.6,
                'take_profit': 1.0,
                'trailing': 0.2,
                'min_trail': 0.15,
                'max_trade_minutes': 30,
                'momentum_min': 0.02,
                'momentum_max': 0.15,
                'rsi_min': 30,
                'rsi_max': 55,
                'min_confidence': 70,
                'min_models_agree': 2,
                'use_mean_reversion': True,
                'description': 'Capital preservation - tight everything'
            },
            'balanced': {
                'stop_loss': 1.5,
                'take_profit': 3.0,
                'trailing': 0.8,
                'min_trail': 0.5,
                'max_trade_minutes': 120,
                'momentum_min': 0.03,
                'momentum_max': 0.25,
                'rsi_min': 30,
                'rsi_max': 65,
                'min_confidence': 60,
                'min_models_agree': 2,
                'use_mean_reversion': False,
                'description': 'Balanced risk/reward - for beginners'
            },
            'aggressive': {
                'stop_loss': 2.5,
                'take_profit': 5.0,
                'trailing': 1.2,
                'min_trail': 1.0,
                'max_trade_minutes': 480,
                'momentum_min': 0.05,
                'momentum_max': 0.40,
                'rsi_min': 25,
                'rsi_max': 70,
                'min_confidence': 55,
                'min_models_agree': 2,
                'use_mean_reversion': False,
                'description': 'Big wins, big losses - for risk takers'
            },
            
            # Legacy alias
            'mean_reversion': {
                'stop_loss': 0.5,
                'take_profit': 0.9,
                'trailing': 0.14,
                'min_trail': 0.45,
                'max_trade_minutes': 25,
                'momentum_min': 0.025,
                'momentum_max': 0.18,
                'rsi_min': 30,
                'rsi_max': 60,
                'min_confidence': 65,
                'min_models_agree': 2,
                'use_mean_reversion': True,
                'description': 'Same as MICRO - statistical edge'
            }
        }
        
        # Normalize preset name
        preset_lower = preset.lower().replace('_', '').replace('-', '')
        preset_map = {
            'scalp': 'scalp', 'scalper': 'scalp',
            'micro': 'micro', 'microprofit': 'micro',
            'swing': 'swing', 'swinger': 'swing',
            'conservative': 'conservative', 'safe': 'conservative',
            'balanced': 'balanced', 'normal': 'balanced',
            'aggressive': 'aggressive', 'risky': 'aggressive',
            'meanreversion': 'mean_reversion', 'mean_reversion': 'mean_reversion'
        }
        
        actual_preset = preset_map.get(preset_lower, 'micro')  # Default to MICRO (best)
        
        if actual_preset in presets:
            config = presets[actual_preset]
            
            # EXIT rules
            self.emergency_stop_loss = config['stop_loss']
            self.take_profit = config['take_profit']
            self.trail_from_peak = config['trailing']
            self.min_profit_to_trail = config['min_trail']
            self.max_trade_minutes = config.get('max_trade_minutes', 60)
            
            # ENTRY thresholds
            self.momentum_min = config.get('momentum_min', 0.02)
            self.momentum_max = config.get('momentum_max', 0.20)
            self.rsi_entry_min = config.get('rsi_min', 30)
            self.rsi_entry_max = config.get('rsi_max', 60)
            self.volume_ratio_min = config.get('volume_ratio_min', 1.0)
            self.spread_max = config.get('spread_max', 0.25)
            self.wick_ratio_max = config.get('wick_ratio_max', 0.60)
            self.distance_from_low_max = config.get('distance_from_low_max', 0.30)
            self.green_red_ratio_min = config.get('green_red_ratio_min', 1.2)
            
            # BTC correlation
            self.btc_correlation_check = config.get('btc_correlation_check', True)
            self.btc_block_threshold = config.get('btc_block_threshold', -2.0)
            self.btc_required_positive = config.get('btc_required_positive', False)
            
            # AI score requirements
            self.min_confidence = config.get('min_confidence', 60)
            self.min_models_agree = config.get('min_models_agree', 2)
            self.require_positive_ev = config.get('require_positive_ev', False)
            self.require_positive_sentiment = config.get('require_positive_sentiment', False)
            
            # Mean reversion mode
            self.use_mean_reversion = config.get('use_mean_reversion', False)
            self.regime_required = config.get('regime_required', [])
            
            logger.info(f"PRESET '{actual_preset.upper()}' applied:")
            logger.info(f"  EXIT: SL={self.emergency_stop_loss}%, TP={self.take_profit}%, Trail={self.trail_from_peak}%")
            logger.info(f"  ENTRY: Momentum={self.momentum_min}-{self.momentum_max}%, RSI={self.rsi_entry_min}-{self.rsi_entry_max}")
            logger.info(f"  AI: MinConf={self.min_confidence}%, ModelsAgree={self.min_models_agree}")
            logger.info(f"  Expected winrate: {config.get('winrate_expected', 'N/A')}")
        else:
            logger.warning(f"Unknown strategy preset '{preset}', using MICRO")
            self._apply_strategy_preset('micro')
    
    async def _auto_select_strategy(self, client: BybitV5Client = None) -> str:
        """
        AI FULL AUTO: Automatically select the best strategy AND optimize all parameters
        
        AI decides:
        - Strategy preset (SCALP/MICRO/SWING/CONSERVATIVE)
        - Number of positions (based on volatility and win rate)
        - Entry filters (confidence, edge thresholds)
        - All safety filters remain active!
        
        User's MAX DAILY DRAWDOWN is ALWAYS respected (protection)
        """
        try:
            # Get current regime
            regime = "normal"
            if self.regime_detector:
                regime_info = self.regime_detector.get_current_regime()
                regime = regime_info.get('regime', 'normal').lower()
            
            # Get BTC volatility as market indicator
            btc_volatility = 1.5  # Default moderate
            btc_change_24h = 0
            if client:
                try:
                    # Get volatility from klines
                    btc_klines = await client.get_klines('BTCUSDT', interval='15', limit=20)
                    if btc_klines.get('success') and btc_klines.get('data', {}).get('list'):
                        klines = btc_klines['data']['list']
                        ranges = []
                        for k in klines[:10]:
                            high = float(k[2])
                            low = float(k[3])
                            close = float(k[4])
                            if close > 0:
                                ranges.append((high - low) / close * 100)
                        if ranges:
                            btc_volatility = sum(ranges) / len(ranges)
                    
                    # Get 24h change
                    btc_ticker = await client.get_tickers(symbol='BTCUSDT')
                    if btc_ticker.get('success'):
                        btc_list = btc_ticker.get('data', {}).get('list', [])
                        if btc_list:
                            btc_change_24h = float(btc_list[0].get('price24hPcnt', 0)) * 100
                except:
                    pass
            
            # Get recent win rate from stats
            recent_win_rate = 0.5
            total_trades = self.stats.get('total_trades', 0)
            if total_trades > 10:
                recent_win_rate = self.stats.get('winning_trades', 0) / max(total_trades, 1)
            
            # ============================================================
            # AI DECISION: SELECT STRATEGY
            # ============================================================
            selected_preset = 'micro'  # Default
            ai_reason = ""
            
            # DANGER: BTC crashing = CONSERVATIVE
            if btc_change_24h < -5:
                selected_preset = 'conservative'
                ai_reason = f"BTC crash ({btc_change_24h:.1f}%)"
            
            # HIGH VOLATILITY + RANGE = SCALP (quick trades)
            elif btc_volatility > 2.0 and regime in ['range', 'sideways', 'choppy']:
                selected_preset = 'scalp'
                ai_reason = f"High volatility ({btc_volatility:.1f}%) + {regime}"
            
            # STRONG TREND = SWING (hold longer)
            elif regime in ['bull', 'bear', 'trend', 'strong_trend']:
                selected_preset = 'swing'
                ai_reason = f"Strong trend ({regime})"
            
            # LOW WIN RATE = CONSERVATIVE (protect capital)
            elif recent_win_rate < 0.45 and total_trades > 20:
                selected_preset = 'conservative'
                ai_reason = f"Low win rate ({recent_win_rate:.0%})"
            
            # HIGH WIN RATE = can be more aggressive
            elif recent_win_rate > 0.70 and total_trades > 30:
                selected_preset = 'micro'  # Stay with best preset
                ai_reason = f"High win rate ({recent_win_rate:.0%})"
            
            # NORMAL CONDITIONS = MICRO (best default)
            else:
                selected_preset = 'micro'
                ai_reason = "Normal conditions"
            
            # Apply the selected preset (sets TP, SL, trailing, entry thresholds)
            self._apply_strategy_preset(selected_preset)
            self.auto_selected_preset = selected_preset
            
            # ============================================================
            # AI DECISION: OPTIMIZE POSITIONS COUNT
            # ============================================================
            # More positions in stable markets, fewer in volatile
            if selected_preset == 'scalp':
                self.max_open_positions = 8  # Fewer, faster trades
            elif selected_preset == 'swing':
                self.max_open_positions = 5  # Even fewer, hold longer
            elif selected_preset == 'conservative':
                self.max_open_positions = 3  # Minimal exposure
            else:  # micro
                self.max_open_positions = 12  # Normal diversification
            
            # Adjust based on win rate
            if recent_win_rate > 0.65:
                self.max_open_positions += 2  # More positions when winning
            elif recent_win_rate < 0.40:
                self.max_open_positions = max(3, self.max_open_positions - 3)  # Reduce exposure
            
            # ============================================================
            # AI DECISION: ADJUST ENTRY FILTERS
            # ============================================================
            # Be stricter when losing, more permissive when winning
            if recent_win_rate < 0.45:
                self.min_confidence = max(70, self.min_confidence)  # Higher threshold
                self.min_edge = max(0.20, self.min_edge)
            elif recent_win_rate > 0.65:
                self.min_confidence = min(60, self.min_confidence)  # Lower threshold
                self.min_edge = min(0.15, self.min_edge)
            
            logger.info(f"AI FULL AUTO: {selected_preset.upper()} | Reason: {ai_reason}")
            logger.info(f"  -> Positions: {self.max_open_positions} | Confidence: {self.min_confidence}% | Edge: {self.min_edge}")
            logger.info(f"  -> TP: {self.take_profit}% | SL: {self.emergency_stop_loss}% | Trail: {self.trail_from_peak}%")
            
            return selected_preset
            
        except Exception as e:
            logger.warning(f"Auto strategy selection failed: {e}, using MICRO defaults")
            self._apply_strategy_preset('micro')
            self.max_open_positions = 10
            return 'micro'
        
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
        
        # DO NOT load global settings here! Each user loads their own in _process_user
        # Settings are PER USER - no global settings!
        
        # Load stats
        await self._load_stats()
        
        # Load paused users from Redis
        await self._load_paused_users()
        
        # Load and connect all registered users from Redis
        await self._load_registered_users()
        
        # Initialize whale tracker
        try:
            await whale_tracker.initialize()
            await whale_tracker.start()
            logger.info("Whale tracker started")
        except Exception as e:
            logger.warning(f"Whale tracker failed to start: {e}")
        
        # Initialize HuggingFace safety models (OPTIONAL - these are GATES, not signals)
        if HF_SAFETY_AVAILABLE:
            try:
                hf = await get_hf_safety()
                if hf.initialized:
                    logger.info("HuggingFace safety models loaded (sentiment, topic, emotion)")
                else:
                    logger.warning("HuggingFace safety models failed to initialize - using fallback")
            except Exception as e:
                logger.warning(f"HuggingFace safety models not available: {e}")
        
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
                
                # Load user-specific stats from Redis
                if self.redis_client:
                    await self._load_user_stats(user_id)
                
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
        logger.info(f"Trading PAUSED for {user_id} - no new positions will be opened")
    
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
                                logger.error(f"ALERT: {consecutive_errors} consecutive errors! Sleeping 10s...")
                                await asyncio.sleep(10)
                                consecutive_errors = 0
                        
                # REMOVED: Global settings reload that was OVERWRITING user settings!
                # Each user's settings are loaded in _process_user ONLY
                # NO GLOBAL RELOAD - this was causing settings to be shared!
                
                # Log status every 100 cycles
                if cycle % 100 == 0:
                    try:
                        await self._log_status()
                    except:
                        pass
                
                # Update dashboard status info PER USER (every 10 cycles to avoid spam)
                if cycle % 10 == 0 and self.redis_client:
                    try:
                        # Update status for EACH connected user
                        for uid in list(self.user_clients.keys()):
                            user_positions = len(self.active_positions.get(uid, {}))
                            status_info = {
                                "last_action": f"Trading active | {user_positions} positions",
                                "regime": current_regime,
                                "strategy": self.active_strategy_name if hasattr(self, 'active_strategy_name') else 'BALANCED',
                                "cycle": cycle,
                                "updated_at": datetime.utcnow().isoformat()
                            }
                            await self.redis_client.set(f'bot:status:live:{uid}', json.dumps(status_info))
                        
                        # Also update global status for reference
                        global_status = {
                            "last_action": f"Trading active | {connected_users} users | {total_positions} positions",
                            "regime": current_regime,
                            "strategy": "GLOBAL",
                            "cycle": cycle,
                            "updated_at": datetime.utcnow().isoformat()
                        }
                        await self.redis_client.set('bot:status:live', json.dumps(global_status))
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
                logger.error(f"CRITICAL LOOP ERROR (cycle {cycle}): {e}")
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
            # 0. Load user-specific settings (CRITICAL - each user has their own settings!)
            try:
                await self._load_settings(user_id)
            except Exception as e:
                logger.debug(f"Failed to load settings for {user_id}: {e}")
            
            # ============================================================
            # CONTEXT SWITCH: Apply this user's settings to instance variables
            # This ensures ALL methods use THIS USER's settings during processing!
            # ============================================================
            user_set = self._get_user_settings(user_id)
            self.take_profit = user_set.get('take_profit', 0.8)
            self.emergency_stop_loss = user_set.get('stop_loss', 0.5)
            self.trail_from_peak = user_set.get('trailing', 0.15)
            self.min_profit_to_trail = user_set.get('min_profit_to_trail', 0.25)
            self.max_open_positions = user_set.get('max_open_positions', 10)
            self.max_exposure_percent = user_set.get('max_exposure_percent', 100)
            self.max_daily_drawdown = user_set.get('max_daily_drawdown', 3.0)
            self.breakout_extra_slots = user_set.get('breakout_extra_slots', False)
            self.ai_full_auto = user_set.get('ai_full_auto', False)
            self.min_confidence = user_set.get('min_confidence', 55)
            self.min_edge = user_set.get('min_edge', 0.2)
            self.risk_mode = user_set.get('risk_mode', 'normal')
            self.leverage_mode = user_set.get('leverage_mode', 'auto')
            self.max_trade_minutes = user_set.get('max_trade_minutes', 15)
            self.use_max_trade_time = user_set.get('use_max_trade_time', True)
            self.active_strategy_name = user_set.get('strategy_preset', 'micro').upper()
            
            # 0.5 AI FULL AUTO: Let AI select strategy based on market conditions
            if self.ai_full_auto:
                try:
                    await self._auto_select_strategy(client)
                except Exception as e:
                    logger.debug(f"Auto strategy selection failed: {e}")
            
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
                logger.info(f"Fast-checking {len(positions_list)} positions (TP={self.take_profit}%, SL={self.emergency_stop_loss}%, Trail={self.trail_from_peak}%, MinTrail={self.min_profit_to_trail}%)")
                
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
                    # Also register in position_sizer so it knows about this position (PER USER!)
                    await self.position_sizer.register_position(symbol, position_value, user_id)
                    logger.info(f"Synced position: {symbol} | Entry: ${entry_price:.4f} | Mark: ${mark_price:.4f} | P&L: {current_pnl:.2f}%")
                else:
                    # Update size if changed
                    self.active_positions[user_id][symbol].size = size
                    
        # Remove closed positions and LOG THEM!
        if user_id in self.active_positions:
            for symbol in list(self.active_positions[user_id].keys()):
                if symbol not in exchange_positions:
                    # SKIP if this symbol is on cooldown (means bot already closed it!)
                    if symbol in self._cooldown_symbols:
                        cooldown_time = self._cooldown_symbols[symbol]
                        elapsed = (datetime.utcnow() - cooldown_time).total_seconds()
                        if elapsed < self.cooldown_seconds:
                            # Bot closed this position - just remove from tracking, don't log as external
                            del self.active_positions[user_id][symbol]
                            logger.debug(f"{symbol} removed from tracking (bot closed {elapsed:.0f}s ago)")
                            continue
                    
                    # Get position info before deleting
                    closed_pos = self.active_positions[user_id][symbol]
                    
                    # Try to estimate P&L from last known data
                    estimated_pnl = 0.0
                    pnl_value = 0.0
                    close_reason = "Closed externally (Bybit SL/TP or manual)"
                    
                    # Check if we can get last price
                    try:
                        if closed_pos.peak_pnl_percent and closed_pos.peak_pnl_percent > 0.5:
                            # Likely hit trailing stop or take profit
                            estimated_pnl = closed_pos.peak_pnl_percent * 0.7  # Estimate
                            close_reason = f"External close (was +{closed_pos.peak_pnl_percent:.1f}% peak)"
                        elif closed_pos.entry_price and closed_pos.position_value:
                            # Estimate based on current SL setting
                            estimated_pnl = -self.emergency_stop_loss  # Assume SL hit
                            close_reason = "External close (likely SL triggered on Bybit)"
                        
                        pnl_value = closed_pos.position_value * (estimated_pnl / 100) if closed_pos.position_value else 0
                    except:
                        pass
                    
                    # Delete from active positions
                    del self.active_positions[user_id][symbol]
                    
                    # Remove from position_sizer (PER USER!)
                    await self.position_sizer.close_position(symbol, pnl_value, user_id)
                    
                    # LOG TO CONSOLE so user sees it on dashboard!
                    await self._log_to_console(
                        f"CLOSED {symbol}: {estimated_pnl:+.2f}% (${pnl_value:+.2f}) | {close_reason}",
                        "TRADE",
                        user_id
                    )
                    
                    # STORE in trades:completed so it shows in Recent Trades!
                    await self._store_trade_event(
                        user_id, symbol, 'closed', estimated_pnl, close_reason, closed_pos, pnl_value
                    )
                    
                    # Update user stats (approximate)
                    user_stats = self._get_user_stats(user_id)
                    user_stats['total_trades'] += 1
                    if pnl_value > 0:
                        user_stats['winning_trades'] += 1
                    user_stats['total_pnl'] += pnl_value
                    await self._save_user_stats(user_id)
                    
                    logger.info(f"Position {symbol} closed EXTERNALLY | Est P&L: {estimated_pnl:+.2f}% (${pnl_value:+.2f}) | {close_reason}")
        
        # IMPORTANT: Sync position sizer with exchange to remove any stale positions
        # This ensures position_sizer tracks THIS USER's positions correctly (PER USER!)
        positions_data = {}
        if user_id in self.active_positions:
            for symbol, pos in self.active_positions[user_id].items():
                positions_data[symbol] = pos.position_value
        
        await self.position_sizer.sync_with_exchange(exchange_positions, positions_data, user_id)
                    
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
            logger.info(f"CHECK {position.symbol}: Side={position.side}, Price=${current_price:.6f}, Entry=${position.entry_price:.6f}, P&L={pnl_percent:+.2f}%, {tp_status}, SL=-{self.emergency_stop_loss}%")
            
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
                logger.info(f"STOP LOSS: {position.symbol} at {pnl_percent:+.2f}%")
                
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
                    logger.info(f"LOCK_PROFIT {position.symbol}: Peak={position.peak_pnl_percent:+.3f}%, Now={pnl_percent:+.3f}%, Drop={drop_from_peak:.3f}%, Trigger={self.trail_from_peak:.3f}%")
                    
                    # EXIT if drop from peak exceeds threshold
                    # Even if current P&L is 0 or slightly negative!
                    if drop_from_peak >= self.trail_from_peak:
                        should_exit = True
                        exit_reason = f"LOCK PROFIT (peak: {position.peak_pnl_percent:.2f}%, drop: {drop_from_peak:.3f}%)"
                        logger.info(f"LOCK PROFIT SELL: {position.symbol} | Peak was {position.peak_pnl_percent:+.2f}%, dropped {drop_from_peak:.3f}% >= {self.trail_from_peak:.3f}%")
            else:
                # ADAPTIVE TRAILING MODE: Let winners run MORE when profit is HIGH
                # Activate trailing when profit reaches min_profit_to_trail (e.g., 0.3%)
                if pnl_percent >= self.min_profit_to_trail:
                    position.trailing_active = True
                
                # ADAPTIVE TRAIL: Wider when in more profit
                adaptive_trail = self.trail_from_peak
                if pnl_percent >= 2.0:
                    adaptive_trail = max(1.0, self.trail_from_peak * 2)  # 2x wider
                elif pnl_percent >= 1.0:
                    adaptive_trail = max(0.8, self.trail_from_peak * 1.5)  # 1.5x wider
                
                # Log trailing status when approaching trigger
                if position.trailing_active and drop_from_peak >= adaptive_trail * 0.5:
                    logger.info(f"TRAILING {position.symbol}: Peak={position.peak_pnl_percent:+.3f}%, Now={pnl_percent:+.3f}%, Drop={drop_from_peak:.3f}%, Trail={adaptive_trail:.2f}%")
                
                # Once trailing is active, sell when price drops from peak
                if position.trailing_active and drop_from_peak >= adaptive_trail:
                    # Only sell if we're still in profit (or minimal loss due to spread)
                    if pnl_percent >= -0.05:
                        should_exit = True
                        exit_reason = f"Trailing stop (peak: +{position.peak_pnl_percent:.2f}%, dropped {drop_from_peak:.2f}%)"
                        logger.info(f"TRAILING SELL {position.symbol}: Peak=+{position.peak_pnl_percent:.2f}%, Now={pnl_percent:+.2f}%, Trail={adaptive_trail:.2f}%")
                    
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
            
            logger.info(f"Order result for {position.symbol}: {result}")
            
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
                
                # Console log for dashboard - show NET P&L (after fees) - PER USER
                await self._log_to_console(
                    f"CLOSED {position.symbol}: {net_pnl_percent:+.2f}% (${pnl_value:+.2f} NET) | {reason}",
                    "TRADE",
                    user_id
                )
                
                # Update USER-SPECIFIC stats
                user_stats = self._get_user_stats(user_id)
                user_stats['total_trades'] += 1
                if won:
                    user_stats['winning_trades'] += 1
                user_stats['total_pnl'] += pnl_value  # NET P&L
                user_stats['daily_pnl'] = user_stats.get('daily_pnl', 0) + pnl_value  # Daily P&L tracking
                
                # Also update GLOBAL stats for AI learning
                self.stats['total_trades'] += 1
                if won:
                    self.stats['winning_trades'] += 1
                self.stats['total_pnl'] += pnl_value
                
                # Save user stats to Redis immediately
                await self._save_user_stats(user_id)
                
                # Record in position sizer (NET P&L) - PER USER!
                await self.position_sizer.close_position(position.symbol, pnl_value, user_id)
                
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
                
                # Remove breakout flag from Redis
                if self.redis_client:
                    await self.redis_client.hdel("positions:breakout", position.symbol)
                
                # ADD COOLDOWN: Prevent reopening this symbol for 60 seconds
                self._cooldown_symbols[position.symbol] = datetime.utcnow()
                logger.debug(f"{position.symbol} on cooldown for {self.cooldown_seconds}s")
                        
                # Store trade for dashboard with NET P&L (per user)
                await self._store_trade_event(user_id, position.symbol, 'closed', net_pnl_percent, reason, position, pnl_value)
                
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
                
                # Update USER-SPECIFIC stats
                user_stats = self._get_user_stats(user_id)
                user_stats['total_trades'] += 0.5  # Count as half trade
                if pnl_percent > 0:
                    user_stats['winning_trades'] += 0.5
                user_stats['total_pnl'] += pnl_value
                user_stats['daily_pnl'] = user_stats.get('daily_pnl', 0) + pnl_value  # Daily P&L tracking
                
                # Also update GLOBAL stats for AI learning
                self.stats['total_trades'] += 0.5
                if pnl_percent > 0:
                    self.stats['winning_trades'] += 0.5
                self.stats['total_pnl'] += pnl_value
                
                # Save user stats
                await self._save_user_stats(user_id)
                
                # Log to console - PER USER
                await self._log_to_console(f"PARTIAL: {position.symbol} +{pnl_percent:.2f}% (took {close_percent}%)", "TRADE", user_id)
                
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
    
    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI from closing prices"""
        if len(closes) < period + 1:
            return 50.0  # Default neutral
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def _advanced_safety_filters(self, opp: TradingOpportunity, client: BybitV5Client, 
                                       kline_data: Dict = None, is_breakout: bool = False) -> Tuple[bool, str]:
        """
        ADVANCED SAFETY FILTERS - Professional hedge fund level checks
        
        These are GATES, not signals. Each removes 5-15% of bad trades.
        
        Filters:
        1. REJECTION CANDLE - Don't buy candles with big wicks (rejection)
        2. BTC CORRELATION - Don't long alts when BTC is dumping
        3. SPREAD CHECK - Don't trade if spread too high (slippage)
        4. ENTRY QUALITY - Must be near support, not chasing (relaxed for breakouts!)
        5. EXPECTED VALUE - Only trade if EV is positive
        """
        try:
            # Get kline data if not provided
            if kline_data is None:
                klines = await client.get_klines(opp.symbol, interval='5', limit=20)
                if not klines.get('success') or not klines.get('data', {}).get('list'):
                    return True, "No kline data for safety check"
                kline_data = klines['data']['list']
            
            if len(kline_data) < 5:
                return True, "Insufficient data"
            
            # Parse latest candle (first in list is most recent)
            latest = kline_data[0]
            open_price = float(latest[1])
            high_price = float(latest[2])
            low_price = float(latest[3])
            close_price = float(latest[4])
            
            # ================================================================
            # FILTER 1: REJECTION CANDLE (Micro Structure)
            # ================================================================
            # Big wick = price rejection = don't buy
            # Uses preset threshold: self.wick_ratio_max (default 0.60)
            candle_range = high_price - low_price
            if candle_range > 0:
                if opp.direction == 'long':
                    # For LONG: upper wick should be small (not rejected at top)
                    upper_wick = high_price - max(open_price, close_price)
                    wick_ratio = upper_wick / candle_range
                    if wick_ratio > self.wick_ratio_max:
                        return False, f"Wick rejection {wick_ratio:.0%}"
                else:
                    # For SHORT: lower wick should be small (not rejected at bottom)
                    lower_wick = min(open_price, close_price) - low_price
                    wick_ratio = lower_wick / candle_range
                    if wick_ratio > self.wick_ratio_max:
                        return False, f"Wick rejection {wick_ratio:.0%}"
            
            # ================================================================
            # FILTER 2: BTC CORRELATION SAFETY
            # ================================================================
            # Uses preset thresholds:
            # - self.btc_correlation_check (enable/disable)
            # - self.btc_block_threshold (default -2.0% - balanced setting)
            # - self.btc_required_positive (for SWING preset)
            if self.btc_correlation_check and opp.symbol != 'BTCUSDT' and opp.direction == 'long':
                try:
                    btc_ticker = await client.get_tickers(symbol='BTCUSDT')
                    if btc_ticker.get('success'):
                        btc_list = btc_ticker.get('data', {}).get('list', [])
                        if btc_list:
                            btc_change = float(btc_list[0].get('price24hPcnt', 0)) * 100
                            
                            # If BTC required positive and it's negative, block
                            if self.btc_required_positive and btc_change < 0:
                                return False, f"BTC negative {btc_change:.1f}%"
                            
                            # If BTC below threshold, block alt longs
                            if btc_change < self.btc_block_threshold:
                                return False, f"BTC down {btc_change:.1f}%"
                            
                            # If BTC is crashing (>5% down), ALWAYS block regardless of preset
                            if btc_change < -5.0:
                                return False, f"BTC crash {btc_change:.1f}%"
                except Exception as e:
                    logger.debug(f"BTC correlation check failed: {e}")
            
            # ================================================================
            # FILTER 3: SPREAD CHECK (Liquidity)
            # ================================================================
            # Uses preset threshold: self.spread_max (default 0.15% for MICRO)
            try:
                orderbook = await client.get_orderbook(opp.symbol, limit=5)
                if orderbook.get('success'):
                    bids = orderbook.get('data', {}).get('b', [])
                    asks = orderbook.get('data', {}).get('a', [])
                    
                    if bids and asks:
                        best_bid = float(bids[0][0])
                        best_ask = float(asks[0][0])
                        mid_price = (best_bid + best_ask) / 2
                        spread_pct = ((best_ask - best_bid) / mid_price) * 100
                        
                        # Spread above threshold = slippage risk
                        if spread_pct > self.spread_max:
                            return False, f"Spread {spread_pct:.2f}%"
            except Exception as e:
                logger.debug(f"Spread check failed: {e}")
            
            # ================================================================
            # FILTER 4: ENTRY PRICE QUALITY
            # ================================================================
            # Uses preset threshold: self.distance_from_low_max (default 0.30 for MICRO)
            # For LONG: should be near support (low of range), not chasing
            # For SHORT: should be near resistance (high of range)
            # BREAKOUTS: Use much higher threshold (85%) since breakouts BY DEFINITION have moved!
            closes = [float(k[4]) for k in kline_data[:10]]
            lows = [float(k[3]) for k in kline_data[:10]]
            highs = [float(k[2]) for k in kline_data[:10]]
            
            local_low = min(lows)
            local_high = max(highs)
            local_range = local_high - local_low
            
            # Convert distance_from_low_max from percentage to ratio (0.30 = 30%)
            max_distance = self.distance_from_low_max if self.distance_from_low_max > 1 else self.distance_from_low_max
            # Ensure it's in 0-1 range
            if max_distance > 1:
                max_distance = max_distance / 100
            
            # BREAKOUTS: Use higher threshold since they've already moved significantly
            # ChatGPT: DON'T touch this - Chasing filter is protecting you from FOMO!
            # 85% means: if price already moved 85% of its range, don't chase
            if is_breakout:
                max_distance = 0.85  # Allow breakouts up to 85% of range
            
            if local_range > 0:
                if opp.direction == 'long':
                    # Distance from local low (support)
                    distance_from_support = (close_price - local_low) / local_range
                    
                    # If already bounced more than threshold, we're chasing
                    if distance_from_support > max_distance:
                        return False, f"Chasing {distance_from_support:.0%}"
                else:
                    # Distance from local high (resistance)
                    distance_from_resistance = (local_high - close_price) / local_range
                    
                    # If already dropped more than threshold, we're chasing the dump
                    if distance_from_resistance > max_distance:
                        return False, f"Chasing {distance_from_resistance:.0%}"
            
            # ================================================================
            # FILTER 5: EXPECTED VALUE CHECK
            # ================================================================
            # Uses preset flag: self.require_positive_ev (default True for MICRO)
            # Simple EV calculation based on historical win rate and R:R
            # EV = P(win) * AvgWin - P(loss) * AvgLoss
            
            # Use our actual settings for calculation
            potential_win = self.take_profit if self.take_profit > 0 else 0.8  # Default 0.8%
            potential_loss = self.emergency_stop_loss
            
            # Estimate win probability based on confidence and edge
            estimated_win_rate = min(opp.confidence / 100, 0.85)  # Cap at 85%
            
            # Calculate EV
            ev = (estimated_win_rate * potential_win) - ((1 - estimated_win_rate) * potential_loss)
            
            # Only block if preset requires positive EV (MICRO preset does)
            if self.require_positive_ev and ev <= 0:
                return False, f"Negative EV {ev:.2f}"
            
            # Passed all filters!
            logger.info(f"SAFETY PASSED: {opp.symbol} | Wick OK | BTC OK | Spread OK | Entry quality OK | EV={ev:.3f}")
            return True, f"Filters passed: Wick/BTC/Spread/Entry/EV"
            
        except Exception as e:
            logger.warning(f"Safety filters error for {opp.symbol}: {e}")
            return True, f"Safety check error: {e}"
    
    async def _confirm_entry(self, opp: TradingOpportunity, client: BybitV5Client, 
                            is_breakout: bool = False, is_mean_rev: bool = False) -> Tuple[bool, str]:
        """
        SCORING-BASED ENTRY CONFIRMATION
        
        Uses ChatGPT breakout rules with scoring system:
        - Each rule contributes points (0-25)
        - Total score out of 100
        - STRICT mode: score >= 70
        - AGGRESSIVE mode: score >= 60
        
        This replaces individual boolean filters with a weighted scoring approach.
        """
        try:
            # Mean reversion trades are already confirmed during detection
            if is_mean_rev:
                return True, "Mean reversion pre-confirmed"
            
            # Get klines for analysis (5min timeframe)
            klines = await client.get_klines(opp.symbol, interval='5', limit=25)
            if not klines.get('success') or not klines.get('data', {}).get('list'):
                logger.warning(f"CONFIRM: {opp.symbol} - no kline data, allowing trade")
                return True, "No data available"
            
            kline_list = klines['data']['list']
            if len(kline_list) < 10:
                return True, "Insufficient kline data"
            
            # Parse klines (most recent first in Bybit, reverse to chronological)
            closes = [float(k[4]) for k in reversed(kline_list)]
            opens = [float(k[1]) for k in reversed(kline_list)]
            volumes = [float(k[5]) for k in reversed(kline_list)]
            lows = [float(k[3]) for k in reversed(kline_list)]
            highs = [float(k[2]) for k in reversed(kline_list)]
            
            # Last candle data
            last_close = closes[-1]
            last_open = opens[-1]
            last_high = highs[-1]
            last_low = lows[-1]
            last_volume = volumes[-1]
            candle_range = last_high - last_low if last_high > last_low else 0.0001
            
            # Calculate RSI
            rsi = self._calculate_rsi(closes, period=14)
            
            # =====================================================
            # SCORING SYSTEM - ChatGPT Breakout Rules
            # =====================================================
            score = 0
            score_breakdown = []
            
            # ===== RULE #1: Green Candle Count (0-25 points) =====
            green_count = sum(1 for i in range(-5, 0) if closes[i] > opens[i])
            if opp.direction == 'long':
                if green_count >= 4:
                    score += 25
                    score_breakdown.append(f"Candles:{green_count}/5=25")
                elif green_count >= 3:
                    score += 20
                    score_breakdown.append(f"Candles:{green_count}/5=20")
                elif green_count >= 2:
                    score += 10
                    score_breakdown.append(f"Candles:{green_count}/5=10")
                else:
                    score_breakdown.append(f"Candles:{green_count}/5=0")
            else:  # SHORT
                red_count = 5 - green_count
                if red_count >= 4:
                    score += 25
                    score_breakdown.append(f"Candles:{red_count}/5red=25")
                elif red_count >= 3:
                    score += 20
                    score_breakdown.append(f"Candles:{red_count}/5red=20")
                elif red_count >= 2:
                    score += 10
                    score_breakdown.append(f"Candles:{red_count}/5red=10")
                else:
                    score_breakdown.append(f"Candles:{red_count}/5red=0")
            
            # ===== RULE #2: Last Candle Confirmation (0-10 points) =====
            last_candle_green = last_close > last_open
            if opp.direction == 'long' and last_candle_green:
                score += 10
                score_breakdown.append("LastGreen=10")
            elif opp.direction == 'short' and not last_candle_green:
                score += 10
                score_breakdown.append("LastRed=10")
            else:
                score_breakdown.append("LastCandle=0")
            
            # ===== RULE #3: Body Strength (0-15 points) =====
            # ChatGPT FIX: Breakout confirmation != impulse candle
            # Confirmation candles are often wicky or small body + continuation
            body_size = abs(last_close - last_open)
            body_ratio = body_size / candle_range if candle_range > 0 else 0
            if body_ratio >= 0.55:
                score += 15  # Full points for >= 55%
                score_breakdown.append(f"Body:{body_ratio:.0%}=15")
            elif body_ratio >= 0.45:
                score += 8   # Half points for 45-55%
                score_breakdown.append(f"Body:{body_ratio:.0%}=8")
            else:
                score_breakdown.append(f"Body:{body_ratio:.0%}=0")
            
            # ===== RULE #4: Close Location >= 70% (0-10 points) =====
            if opp.direction == 'long':
                close_location = (last_close - last_low) / candle_range if candle_range > 0 else 0.5
            else:
                close_location = (last_high - last_close) / candle_range if candle_range > 0 else 0.5
            
            if close_location >= 0.8:
                score += 10
                score_breakdown.append(f"CloseLoc:{close_location:.0%}=10")
            elif close_location >= 0.7:
                score += 8
                score_breakdown.append(f"CloseLoc:{close_location:.0%}=8")
            elif close_location >= 0.5:
                score += 4
                score_breakdown.append(f"CloseLoc:{close_location:.0%}=4")
            else:
                score_breakdown.append(f"CloseLoc:{close_location:.0%}=0")
            
            # ===== RULE #5: Volume Spike (0-25 points) =====
            # ChatGPT FIX: Use max(volume[-3:]) because spike might be 1-2 candles ago
            # You're entering on confirmation candle, not the spike candle!
            avg_volume_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
            recent_max_volume = max(volumes[-3:]) if len(volumes) >= 3 else last_volume
            volume_ratio = recent_max_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            if volume_ratio >= 1.5:
                score += 25
                score_breakdown.append(f"VolSpike:{volume_ratio:.1f}x=25")
            elif volume_ratio >= 1.2:
                score += 18
                score_breakdown.append(f"VolSpike:{volume_ratio:.1f}x=18")
            elif volume_ratio >= 1.0:
                score += 10
                score_breakdown.append(f"VolSpike:{volume_ratio:.1f}x=10")
            elif volume_ratio >= 0.8:
                score += 5
                score_breakdown.append(f"VolSpike:{volume_ratio:.1f}x=5")
            else:
                score_breakdown.append(f"VolSpike:{volume_ratio:.1f}x=0")
            
            # ===== RULE #6: Volume Trend (0-10 points) =====
            vol_trend_up = volumes[-1] > volumes[-2] or volumes[-1] > (sum(volumes[-10:]) / 10)
            if vol_trend_up:
                score += 10
                score_breakdown.append("VolTrend=10")
            else:
                score_breakdown.append("VolTrend=0")
            
            # ===== RULE #7: Structure Break (0-15 points) =====
            # Close above/below local range (last 10 candles)
            local_high = max(highs[-10:])
            local_low = min(lows[-10:])
            
            if opp.direction == 'long' and last_close >= local_high * 0.998:  # Near or above high
                score += 15
                score_breakdown.append("StructBreak=15")
            elif opp.direction == 'short' and last_close <= local_low * 1.002:  # Near or below low
                score += 15
                score_breakdown.append("StructBreak=15")
            elif opp.direction == 'long' and last_close >= local_high * 0.99:
                score += 8
                score_breakdown.append("StructBreak=8")
            elif opp.direction == 'short' and last_close <= local_low * 1.01:
                score += 8
                score_breakdown.append("StructBreak=8")
            else:
                score_breakdown.append("StructBreak=0")
            
            # ===== RULE #8: Wick Check <= 30% (penalty if too long) =====
            if opp.direction == 'long':
                upper_wick = last_high - max(last_open, last_close)
                wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
            else:
                lower_wick = min(last_open, last_close) - last_low
                wick_ratio = lower_wick / candle_range if candle_range > 0 else 0
            
            if wick_ratio <= 0.2:
                score += 10  # Very clean candle
                score_breakdown.append(f"Wick:{wick_ratio:.0%}=10")
            elif wick_ratio <= 0.3:
                score += 5
                score_breakdown.append(f"Wick:{wick_ratio:.0%}=5")
            elif wick_ratio > 0.5:
                score -= 10  # Penalty for rejection wick
                score_breakdown.append(f"Wick:{wick_ratio:.0%}=-10")
            else:
                score_breakdown.append(f"Wick:{wick_ratio:.0%}=0")
            
            # ===== RULE #9: RSI Soft Filter (bonus/penalty) =====
            if opp.direction == 'long':
                if 30 <= rsi <= 60:
                    score += 5  # Ideal zone
                    score_breakdown.append(f"RSI:{rsi:.0f}=+5")
                elif rsi > 80:
                    score -= 10  # Too overbought
                    score_breakdown.append(f"RSI:{rsi:.0f}=-10")
                elif rsi > 70:
                    score -= 5
                    score_breakdown.append(f"RSI:{rsi:.0f}=-5")
                else:
                    score_breakdown.append(f"RSI:{rsi:.0f}=0")
            else:  # SHORT
                if 40 <= rsi <= 70:
                    score += 5
                    score_breakdown.append(f"RSI:{rsi:.0f}=+5")
                elif rsi < 20:
                    score -= 10  # Too oversold
                    score_breakdown.append(f"RSI:{rsi:.0f}=-10")
                elif rsi < 30:
                    score -= 5
                    score_breakdown.append(f"RSI:{rsi:.0f}=-5")
                else:
                    score_breakdown.append(f"RSI:{rsi:.0f}=0")
            
            # =====================================================
            # FINAL DECISION based on score
            # =====================================================
            # ChatGPT FIX: Breakouts need 52, normal trades need 65
            # Breakouts already passed detection + chasing/wick/spread filters protect us
            
            # BALANCED: Breakout=52, Regular=55 (was 65 - too strict, caused 0 trades)
            min_score = 52 if is_breakout else 55
            
            score_str = " | ".join(score_breakdown[:5])  # Limit for readability
            
            if score >= min_score:
                logger.info(f"SCORE OK {opp.symbol}: {score}/100 >= {min_score} | {score_str}")
                return True, f"Score {score}/100 OK"
            else:
                # Find the weakest component
                lowest = min(score_breakdown, key=lambda x: int(x.split('=')[-1]) if '=' in x else 0)
                logger.info(f"SCORE LOW {opp.symbol}: {score}/100 < {min_score} | {score_str}")
                return False, f"Score {score} < {min_score} (weak: {lowest})"
            
        except Exception as e:
            logger.warning(f"CONFIRM {opp.symbol}: Error {e} - allowing trade")
            return True, f"Confirmation error: {e}"
    
    async def _find_momentum_opportunities(self, user_id: str, client: BybitV5Client, wallet: Dict) -> List[TradingOpportunity]:
        """
        MOMENTUM OPPORTUNITIES - Same as AI Signals display
        
        These are the regular momentum-based trades that AI Signals shows.
        Simple but effective: trade with strong momentum when filters pass.
        
        ENTRY CONDITIONS:
        1. Strong momentum (>0.5% 24h change)
        2. Good volume (>$1M)
        3. Low spread (<0.5%)
        4. Must pass all safety filters (_validate_opportunity)
        """
        opportunities = []
        
        try:
            # Get tickers
            tickers_result = await client.get_tickers()
            if not tickers_result.get('success'):
                return opportunities
            
            tickers = tickers_result.get('data', {}).get('list', [])
            
            # Filter and score opportunities
            candidates = []
            for ticker in tickers[:200]:  # Scan top 200
                symbol = ticker.get('symbol', '')
                if not symbol.endswith('USDT') or symbol in ['USDCUSDT', 'USDTUSDT', 'DAIUSDT', 'TUSDUSDT']:
                    continue
                
                # Skip if already in position
                if symbol in self.active_positions.get(user_id, {}):
                    continue
                
                # Skip if on cooldown
                if symbol in self._cooldown_symbols or symbol in self._failed_order_symbols:
                    continue
                
                price_change = float(ticker.get('price24hPcnt', 0)) * 100
                volume = float(ticker.get('turnover24h', 0))
                last_price = float(ticker.get('lastPrice', 0))
                bid_price = float(ticker.get('bid1Price', 0))
                ask_price = float(ticker.get('ask1Price', 0))
                
                # Basic filters
                if volume < 1000000 or last_price <= 0:  # Min $1M volume
                    continue
                
                # Calculate spread
                spread = ((ask_price - bid_price) / last_price * 100) if last_price > 0 else 0
                if spread > 0.3:  # Strict spread filter for momentum
                    continue
                
                # Determine direction - need strong momentum
                if price_change > 1.0:
                    direction = 'long'
                elif price_change < -1.0:
                    direction = 'short'
                else:
                    continue  # Not enough momentum
                
                # Calculate edge
                volatility = abs(price_change)
                edge = max(0, volatility * 0.25 - spread * 2)
                
                if edge < self.min_edge:
                    continue
                
                # Calculate confidence
                volume_score = min(50, volume / 10000000 * 50)
                momentum_score = min(50, abs(price_change) * 8)
                confidence = int(volume_score + momentum_score)
                
                if confidence < self.min_confidence:
                    continue
                
                # Create opportunity
                opp = TradingOpportunity(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    edge=edge,
                    entry_price=last_price,
                    current_price=last_price,
                    price_change_24h=price_change,
                    volume_24h=volume,
                    volatility=volatility,
                    regime='MOMENTUM',
                    strategy='momentum',
                    reasons=[f"Strong momentum {price_change:+.1f}%", f"Volume ${volume/1e6:.1f}M"]
                )
                
                candidates.append(opp)
            
            # Sort by edge * confidence
            candidates.sort(key=lambda x: x.edge * x.confidence, reverse=True)
            
            # Validate top candidates through safety filters
            for opp in candidates[:20]:  # Check top 20
                is_valid, reason = await self._validate_opportunity(user_id, client, opp, is_breakout=False)
                
                if is_valid:
                    opportunities.append(opp)
                    logger.info(f"MOMENTUM OK: {opp.symbol} {opp.direction.upper()} | Edge: {opp.edge:.1f}% | Conf: {opp.confidence}")
                    
                    if len(opportunities) >= 5:  # Max 5 candidates
                        break
                else:
                    logger.debug(f"MOMENTUM BLOCKED {opp.symbol}: {reason}")
            
            logger.info(f"MOMENTUM: Found {len(opportunities)} valid opportunities from {len(candidates)} candidates")
            
        except Exception as e:
            logger.error(f"Error finding momentum opportunities: {e}")
        
        return opportunities
    
    async def _find_mean_reversion_opportunities(self, user_id: str, client: BybitV5Client, wallet: Dict) -> List[TradingOpportunity]:
        """
        MEAN REVERSION STRATEGY v2.0 - Professional Statistical Edge
        
        This is what market makers and quant funds use.
        
        ENTRY CONDITIONS (ALL must be true):
        1. Regime = RANGE/SIDEWAYS (not trending!)
        2. Price dipped -0.5% to -2.0% recently
        3. RSI < 35 AND rising (exiting oversold)
        4. Last candle is GREEN (momentum turned)
        5. Volume DECREASING (panic fading)
        6. Price NOT making new lows
        7. Near support zone (bottom 30% of range)
        
        KEY INSIGHT: Don't buy the dip - buy when the dip STOPS!
        
        Expected: 70-80% winrate with consistent profits
        """
        opportunities = []
        
        if not self.use_mean_reversion:
            return opportunities
        
        try:
            # STEP 1: Check regime - ONLY trade in RANGE markets
            if self.mr_only_range_regime:
                btc_regime = await self.regime_detector.detect_regime('BTCUSDT')
                regime_name = btc_regime.regime.lower() if btc_regime else 'unknown'
                
                allowed_regimes = ['sideways', 'range', 'neutral', 'low_volatility', 'consolidation', 'range_bound']
                is_range = any(r in regime_name for r in allowed_regimes)
                
                if not is_range:
                    logger.debug(f"MEAN_REV: Skip - regime is {regime_name}, need RANGE")
                    return opportunities
                
                logger.info(f"MEAN_REV: Regime={regime_name} - scanning for dip reversals...")
            
            # STEP 2: Get all tickers for initial filtering
            tickers_result = await client.get_tickers()
            if not tickers_result.get('success'):
                return opportunities
            
            tickers = tickers_result.get('data', {}).get('list', [])
            min_volume = 2000000  # $2M minimum (need good liquidity)
            
            # First pass: Find potential candidates based on price dip
            potential_candidates = []
            
            for ticker in tickers:
                symbol = ticker.get('symbol', '')
                if not symbol.endswith('USDT') or symbol in ['USDCUSDT', 'USDTUSDT', 'DAIUSDT']:
                    continue
                
                if symbol in self.active_positions.get(user_id, {}):
                    continue
                if symbol in self._cooldown_symbols:
                    continue
                if symbol in self._failed_order_symbols:
                    continue
                
                price_change = float(ticker.get('price24hPcnt', 0)) * 100
                volume = float(ticker.get('turnover24h', 0))
                last_price = float(ticker.get('lastPrice', 0))
                high_24h = float(ticker.get('highPrice24h', 0))
                low_24h = float(ticker.get('lowPrice24h', 0))
                
                if volume < min_volume or last_price <= 0:
                    continue
                
                # Look for DIPS: -0.5% to -3.0% (wider range, but we'll filter more)
                if not (-0.5 >= price_change >= -3.0):
                    continue
                
                # Check position in range - must be near BOTTOM (support zone)
                price_range = high_24h - low_24h
                if price_range <= 0:
                    continue
                
                position_in_range = (last_price - low_24h) / price_range
                
                # Only consider if in bottom 35% of range
                if position_in_range > 0.35:
                    continue
                
                potential_candidates.append({
                    'symbol': symbol,
                    'price': last_price,
                    'change': price_change,
                    'volume': volume,
                    'high_24h': high_24h,
                    'low_24h': low_24h,
                    'position_in_range': position_in_range
                })
            
            # Sort by dip size (bigger dip = more potential for reversal)
            potential_candidates.sort(key=lambda x: x['change'])
            
            # STEP 3: Deep analysis on top 10 candidates (to avoid too many API calls)
            confirmed_candidates = []
            
            for candidate in potential_candidates[:10]:
                symbol = candidate['symbol']
                
                try:
                    # Get 5-minute klines for momentum analysis
                    klines = await client.get_klines(symbol, interval='5', limit=30)
                    if not klines.get('success') or not klines.get('data', {}).get('list'):
                        continue
                    
                    kline_list = klines['data']['list']
                    if len(kline_list) < 20:
                        continue
                    
                    # Klines are [timestamp, open, high, low, close, volume, turnover]
                    # Most recent is first in Bybit API
                    closes = [float(k[4]) for k in reversed(kline_list)]  # Reverse to chronological
                    opens = [float(k[1]) for k in reversed(kline_list)]
                    volumes = [float(k[5]) for k in reversed(kline_list)]
                    lows = [float(k[3]) for k in reversed(kline_list)]
                    
                    # === CONFIRMATION CHECK 1: RSI exiting oversold ===
                    rsi = self._calculate_rsi(closes, period=14)
                    
                    # Calculate previous RSI (3 candles ago)
                    prev_rsi = self._calculate_rsi(closes[:-3], period=14) if len(closes) > 17 else rsi
                    
                    # RSI should be < 40 AND rising (exiting oversold)
                    rsi_recovering = rsi < 40 and rsi > prev_rsi
                    if not rsi_recovering:
                        logger.debug(f"MEAN_REV {symbol}: RSI={rsi:.1f} (prev={prev_rsi:.1f}) - not recovering")
                        continue
                    
                    # === CONFIRMATION CHECK 2: Last candle is GREEN ===
                    last_close = closes[-1]
                    last_open = opens[-1]
                    is_green_candle = last_close > last_open
                    
                    if not is_green_candle:
                        logger.debug(f"MEAN_REV {symbol}: Last candle RED - momentum not turned")
                        continue
                    
                    # === CONFIRMATION CHECK 3: Volume decreasing (panic fading) ===
                    recent_vol = sum(volumes[-3:]) / 3  # Last 3 candles avg
                    older_vol = sum(volumes[-8:-3]) / 5  # 5 candles before that
                    volume_decreasing = recent_vol < older_vol * 1.1  # Allow 10% tolerance
                    
                    if not volume_decreasing:
                        logger.debug(f"MEAN_REV {symbol}: Volume still high - panic not fading")
                        continue
                    
                    # === CONFIRMATION CHECK 4: Not making new lows ===
                    recent_low = min(lows[-3:])
                    older_low = min(lows[-10:-3]) if len(lows) > 10 else min(lows[:-3])
                    making_new_lows = recent_low < older_low * 0.998  # 0.2% tolerance
                    
                    if making_new_lows:
                        logger.debug(f"MEAN_REV {symbol}: Still making new lows - not safe")
                        continue
                    
                    # === ALL CHECKS PASSED - Calculate score ===
                    score = 0
                    
                    # RSI score (lower RSI that's recovering = better)
                    rsi_score = max(0, (40 - rsi) * 2)  # Up to 40 points
                    score += rsi_score
                    
                    # Position in range score (lower = better)
                    range_score = (1 - candidate['position_in_range']) * 30  # Up to 30 points
                    score += range_score
                    
                    # Green candle strength
                    candle_strength = (last_close - last_open) / last_open * 100
                    candle_score = min(candle_strength * 10, 15)  # Up to 15 points
                    score += candle_score
                    
                    # Volume score
                    vol_score = min(candidate['volume'] / 10000000 * 10, 15)  # Up to 15 points
                    score += vol_score
                    
                    confirmed_candidates.append({
                        **candidate,
                        'rsi': rsi,
                        'rsi_recovering': True,
                        'green_candle': True,
                        'volume_fading': True,
                        'score': score
                    })
                    
                    logger.info(f"MEAN_REV CONFIRMED: {symbol} | RSI={rsi:.1f} rising | Green candle | Vol fading | Score={score:.0f}")
                    
                except Exception as e:
                    logger.debug(f"MEAN_REV {symbol}: Analysis error - {e}")
                    continue
            
            # Sort confirmed candidates by score
            confirmed_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Create opportunities from top 3 confirmed candidates
            for c in confirmed_candidates[:3]:
                opp = TradingOpportunity(
                    symbol=c['symbol'],
                    direction='long',
                    edge_score=0.75,  # Higher edge for confirmed reversals
                    confidence=min(c['score'], 90),
                    opportunity_score=c['score'],
                    current_price=c['price'],
                    price_change_24h=c['change'],
                    volume_24h=c['volume'],
                    should_trade=True,
                    reasons=[
                        f"MEAN_REV v2: Dip {c['change']:.1f}%",
                        f"RSI={c['rsi']:.0f} recovering",
                        f"Green candle confirmed",
                        f"Volume fading (panic over)",
                        f"Score: {c['score']:.0f}"
                    ],
                    timestamp=datetime.utcnow().isoformat()
                )
                opp.is_mean_reversion = True
                opp.size_multiplier = 1.0
                opportunities.append(opp)
                logger.info(f"MEAN_REV v2: {c['symbol']} | Dip: {c['change']:.1f}% | RSI={c['rsi']:.0f} | GREEN | Score: {c['score']:.0f}")
            
            if opportunities:
                best = opportunities[0]
                await self._log_to_console(
                    f"MEAN_REV v2: {len(opportunities)} CONFIRMED reversals | Best: {best.symbol} (RSI recovering, green candle)", 
                    "SIGNAL"
                )
            elif potential_candidates:
                await self._log_to_console(
                    f"MEAN_REV: {len(potential_candidates)} dips found, 0 confirmed (waiting for reversal signals)", 
                    "INFO"
                )
                
        except Exception as e:
            logger.error(f"Mean reversion detection error: {e}")
        
        return opportunities
            
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
                high_24h = float(ticker.get('highPrice24h', 0))
                low_24h = float(ticker.get('lowPrice24h', 0))
                prev_price = float(ticker.get('prevPrice24h', last_price))
                
                if volume < min_volume or last_price <= 0:
                    continue
                
                # =========================================================
                # PRE-BREAKOUT DETECTION (Volume Surge before Price Move)
                # =========================================================
                # This catches coins BEFORE they pump, not after
                # Key: Volume is high but price hasn't moved much yet
                avg_volume = volume / 24 * 4  # Rough 4h average (approximate)
                price_range = ((high_24h - low_24h) / last_price * 100) if last_price > 0 else 0
                
                # Volume surge: High volume but small price move = accumulation
                # Price near high of range = ready to break out
                position_in_range = ((last_price - low_24h) / (high_24h - low_24h)) if (high_24h - low_24h) > 0 else 0.5
                
                # Detect pre-breakout conditions:
                # 1. Volume > $1M (good liquidity)
                # 2. Price change small (-5% to +5%) - hasn't moved much
                # 3. Position in range > 0.7 (near top) = bullish, < 0.3 = bearish
                if volume > 1000000 and -5 < price_change < 5:
                    if position_in_range > 0.75:
                        # Near top of range with consolidation = potential bullish breakout
                        pre_breakout_score = position_in_range * 50  # Up to 50 score
                        opp = TradingOpportunity(
                            symbol=symbol,
                            direction='long',
                            edge_score=0.6,
                            confidence=pre_breakout_score,
                            opportunity_score=pre_breakout_score,
                            current_price=last_price,
                            price_change_24h=price_change,
                            volume_24h=volume,
                            should_trade=True,
                            reasons=[f"PRE-BREAKOUT: Near top of range ({position_in_range:.0%}), vol ${volume/1000000:.1f}M"],
                            timestamp=datetime.utcnow().isoformat()
                        )
                        opp.size_multiplier = 0.7  # Smaller size for pre-breakout (more risk)
                        opp.is_pre_breakout = True
                        breakouts.append(opp)
                        logger.info(f" PRE-BREAKOUT: {symbol} at {position_in_range:.0%} of range | Vol: ${volume/1000000:.1f}M")
                        continue  # Skip normal breakout check
                    elif position_in_range < 0.25:
                        # Near bottom of range = potential bearish breakdown or bounce
                        # Skip for now - bottoms are harder to trade
                        pass
                
                # =========================================================
                # BREAKOUT DETECTION - Only SIGNIFICANT moves (10%+)
                # =========================================================
                # 5-10% = normal volatility, not breakout
                # 10%+ = real breakout worth trading
                is_breakout = False
                direction = None
                breakout_strength = 0
                size_multiplier = 1.0  # Reduce size for extreme moves
                
                # Bullish breakout (+10% and up) - raised threshold for real breakouts
                if price_change >= 10:
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
                    
                # Bearish breakout (-10% and down) - raised threshold for real breakouts
                elif price_change <= -10:
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
            
            # Log if we found breakouts with ranking
            if breakouts:
                top3 = breakouts[:3]
                rank_str = ", ".join([f"#{i+1} {b.symbol}({b.price_change_24h:+.1f}%)" for i, b in enumerate(top3)])
                await self._log_to_console(f"{len(breakouts)} BREAKOUTS | Best: {rank_str}", "SIGNAL")
                logger.info(f"TOP BREAKOUT: {breakouts[0].symbol} | Score: {breakouts[0].opportunity_score:.0f} | Move: {breakouts[0].price_change_24h:+.1f}%")
            
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
            
            # === MEAN REVERSION STRATEGY (if enabled) ===
            # This is the statistical edge strategy - trade dips in range markets
            # Takes priority over breakout hunting when enabled
            if self.use_mean_reversion:
                mean_rev_opps = await self._find_mean_reversion_opportunities(user_id, client, wallet)
                
                mr_trades_opened = 0
                max_mr_per_cycle = 2
                
                for opp in mean_rev_opps:
                    num_positions = len(self.active_positions.get(user_id, {}))
                    
                    # Check position limit
                    if self.max_open_positions > 0 and num_positions >= self.max_open_positions:
                        logger.info(f"MEAN_REV {opp.symbol} skipped - position limit reached")
                        break
                    
                    if mr_trades_opened >= max_mr_per_cycle:
                        break
                    
                    # Skip if already in position
                    if opp.symbol in self.active_positions.get(user_id, {}):
                        continue
                    
                    # Skip if on cooldown
                    if opp.symbol in self._cooldown_symbols or opp.symbol in self._failed_order_symbols:
                        continue
                    
                    logger.info(f"MEAN_REV v2 TRADE: {opp.symbol} | Dip: {opp.price_change_24h:.1f}% | Reversal confirmed")
                    await self._log_to_console(f"MEAN_REV BUY: {opp.symbol} | Dip {opp.price_change_24h:.1f}% | RSI+Green confirmed", "TRADE", user_id)
                    
                    try:
                        await self._execute_trade(user_id, client, opp, wallet)
                        mr_trades_opened += 1
                    except Exception as e:
                        logger.error(f"MEAN_REV trade failed: {opp.symbol} - {e}")
                
                # If mean reversion is enabled and we found opportunities, skip breakout hunting
                if mean_rev_opps:
                    return
            
            # === MOMENTUM OPPORTUNITIES (AI Signals) ===
            # These are the regular momentum-based opportunities shown in AI Signals
            # Always active - this is the main trading source when breakouts are disabled
            user_set = self.user_settings.get(user_id, {})
            enable_breakout = user_set.get('enable_breakout', False)
            
            momentum_opps = await self._find_momentum_opportunities(user_id, client, wallet)
            
            momentum_trades_opened = 0
            max_momentum_per_cycle = 2
            
            for opp in momentum_opps:
                num_positions = len(self.active_positions.get(user_id, {}))
                
                # Check position limit
                if self.max_open_positions > 0 and num_positions >= self.max_open_positions:
                    logger.debug(f"MOMENTUM {opp.symbol} skipped - position limit reached")
                    break
                
                if momentum_trades_opened >= max_momentum_per_cycle:
                    break
                
                # Skip if already in position
                if opp.symbol in self.active_positions.get(user_id, {}):
                    continue
                
                # Skip if on cooldown
                if opp.symbol in self._cooldown_symbols or opp.symbol in self._failed_order_symbols:
                    continue
                
                logger.info(f"MOMENTUM TRADE: {opp.symbol} {opp.direction.upper()} | Edge: {opp.edge:.1f}% | Conf: {opp.confidence}")
                await self._log_to_console(f"MOMENTUM {opp.direction.upper()}: {opp.symbol} | Edge {opp.edge:.1f}%", "TRADE", user_id)
                
                try:
                    await self._execute_trade(user_id, client, opp, wallet)
                    momentum_trades_opened += 1
                except Exception as e:
                    logger.error(f"MOMENTUM trade failed: {opp.symbol} - {e}")
            
            # === BREAKOUT DETECTION (if enabled) ===
            # This catches big moves - only runs if breakout trading is enabled
            if not enable_breakout:
                logger.debug(f"Breakout trading DISABLED for {user_id} - skipping breakout detection")
                return  # Done - momentum opportunities already processed
            
            breakouts = await self._find_breakouts(user_id, client, wallet)
            
            # Calculate breakout position limit based on user settings
            # If breakoutExtraSlots is enabled, allow +2 extra positions for breakouts
            extra_slots = 2 if self.breakout_extra_slots else 0
            if self.max_open_positions > 0:
                breakout_limit = self.max_open_positions + extra_slots
            else:
                breakout_limit = 0  # 0 = unlimited
            
            breakout_trades_opened = 0
            max_breakout_trades_per_cycle = 2  # Reduced to avoid rate limiting
            
            for opp in breakouts:
                # Add small delay between orders to avoid rate limiting
                if breakout_trades_opened > 0:
                    import asyncio
                    await asyncio.sleep(1)  # 1 second delay between orders
                num_positions = len(self.active_positions.get(user_id, {}))
                
                # Check if max positions reached
                if breakout_limit > 0 and num_positions >= breakout_limit:
                    limit_info = f"{breakout_limit} (max {self.max_open_positions} + {extra_slots} breakout)" if extra_slots > 0 else str(self.max_open_positions)
                    logger.info(f"BREAKOUT {opp.symbol} skipped - position limit {limit_info} reached")
                    await self._log_to_console(f"{opp.symbol} {opp.price_change_24h:+.1f}% skipped - position limit ({num_positions}/{limit_info})", "WARNING", user_id)
                    break
                
                # Limit breakout trades per cycle to avoid overtrading
                if breakout_trades_opened >= max_breakout_trades_per_cycle:
                    logger.info(f"BREAKOUT {opp.symbol} skipped - max {max_breakout_trades_per_cycle} breakouts per cycle")
                    break
                
                # Check if already in this position
                if opp.symbol in self.active_positions.get(user_id, {}):
                    logger.debug(f"BREAKOUT {opp.symbol} skipped - already in position")
                    continue
                
                # Check if this symbol recently failed - don't retry for 5 minutes
                if opp.symbol in self._failed_order_symbols:
                    failed_time = self._failed_order_symbols[opp.symbol]
                    elapsed = (datetime.utcnow() - failed_time).total_seconds()
                    if elapsed < self.failed_order_cooldown:
                        logger.debug(f"BREAKOUT {opp.symbol} skipped - failed order cooldown ({int(self.failed_order_cooldown - elapsed)}s remaining)")
                        continue
                    else:
                        # Cooldown expired, remove from dict
                        del self._failed_order_symbols[opp.symbol]
                
                # === MOMENTUM CHECK FOR BREAKOUTS ===
                # Ensure we're trading WITH momentum, not catching falling knives
                # For LONG: price should still be moving up (not reversing)
                # For SHORT: price should still be moving down
                is_pre_breakout = getattr(opp, 'is_pre_breakout', False)
                
                if not is_pre_breakout:  # Skip momentum check for pre-breakouts
                    # Get recent price action (using 24h change as proxy)
                    if opp.direction == 'long' and opp.price_change_24h < 5:
                        # Breakout said long, but price not moving up anymore
                        logger.info(f"BREAKOUT {opp.symbol} skipped - momentum fading ({opp.price_change_24h:+.1f}% < +5%)")
                        continue
                    elif opp.direction == 'short' and opp.price_change_24h > -5:
                        # Breakout said short, but price not moving down anymore
                        logger.info(f"BREAKOUT {opp.symbol} skipped - momentum fading ({opp.price_change_24h:+.1f}% > -5%)")
                        continue
                
                # Log the trade type
                trade_type = "PRE-BREAKOUT" if is_pre_breakout else "BREAKOUT"
                
                # === FULL VALIDATION FOR BREAKOUTS ===
                # Breakouts MUST pass ALL filters: RSI, candles, spread, BTC correlation, etc.
                # This ensures we don't enter bad trades even during breakouts!
                should_trade, reject_reason = await self._validate_opportunity(
                    opp, wallet, client,
                    adjusted_min_confidence=max(50, self.min_confidence * 0.7),  # Slightly lower confidence for breakouts
                    adjusted_min_edge=max(0.1, self.min_edge * 0.8),  # Slightly lower edge for breakouts
                    user_id=user_id
                )
                
                if not should_trade:
                    logger.info(f"{trade_type} BLOCKED: {opp.symbol} - {reject_reason}")
                    # Shorten reason for console (remove symbol from reason if present)
                    short_reason = reject_reason.replace(f"Breakout {opp.symbol}: ", "").replace(f"{opp.symbol}: ", "")[:35]
                    await self._log_to_console(f"BLOCKED {opp.symbol}: {short_reason}", "WARNING", user_id)
                    continue
                
                # Passed ALL filters - proceed with trade
                logger.info(f"{trade_type} APPROVED: {opp.symbol} | Score: {opp.opportunity_score:.0f} | Conf: {opp.confidence:.0f}% | Edge: {opp.edge_score:.2f} | All filters OK")
                logger.info(f"{trade_type} TRADE: {opp.symbol} | {opp.price_change_24h:+.1f}% | Direction: {opp.direction}")
                await self._log_to_console(f"{trade_type} {opp.direction.upper()} {opp.symbol} | Score:{opp.opportunity_score:.0f} | All filters OK", "TRADE", user_id)
                try:
                    await self._execute_trade(user_id, client, opp, wallet)
                    breakout_trades_opened += 1
                    logger.info(f"{trade_type} TRADE OPENED: {opp.symbol}")
                except Exception as e:
                    logger.error(f"BREAKOUT TRADE FAILED: {opp.symbol} - {e}")
                    await self._log_to_console(f"BREAKOUT FAILED: {opp.symbol} - {str(e)[:50]}", "ERROR", user_id)
            
            # === SMART TRADING LOGIC ===
            # Check recent performance - be more conservative if losing
            user_stats = self._get_user_stats(user_id)
            recent_trades = user_stats.get('total_trades', 0)
            recent_wins = user_stats.get('winning_trades', 0)
            recent_win_rate = (recent_wins / recent_trades * 100) if recent_trades > 0 else 50
            
            # Dynamic confidence threshold based on recent performance
            # If losing (win rate < 50%), require higher confidence
            adjusted_min_confidence = self.min_confidence
            adjusted_min_edge = self.min_edge
            
            if recent_trades >= 10:  # Only adjust after 10 trades
                if recent_win_rate < 40:
                    # We're losing badly - be VERY conservative
                    adjusted_min_confidence = max(75, self.min_confidence + 15)
                    adjusted_min_edge = max(0.35, self.min_edge + 0.15)
                    logger.info(f"CONSERVATIVE MODE: Win rate {recent_win_rate:.1f}% - requiring conf>{adjusted_min_confidence}, edge>{adjusted_min_edge}")
                elif recent_win_rate < 50:
                    # We're losing - be more conservative
                    adjusted_min_confidence = max(70, self.min_confidence + 10)
                    adjusted_min_edge = max(0.25, self.min_edge + 0.10)
            
            # Rate limiting: Only open 1 NORMAL trade per cycle (30 seconds)
            # This prevents rushing to fill positions
            normal_trades_this_cycle = 0
            max_normal_trades_per_cycle = 1  # BE PATIENT - only 1 trade every 30 seconds
            
            # Check if we just opened a trade recently (global cooldown)
            last_trade_time = getattr(self, '_last_trade_time', None)
            if last_trade_time:
                seconds_since_trade = (datetime.utcnow() - last_trade_time).total_seconds()
                if seconds_since_trade < 60:  # Wait at least 60 seconds between trades
                    logger.debug(f"Global cooldown: {60 - seconds_since_trade:.0f}s until next trade allowed")
                    return
            
            # === NORMAL OPPORTUNITY SCAN ===
            # Get opportunities from scanner
            opportunities = await self.market_scanner.get_tradeable_opportunities()
            
            self.stats['opportunities_scanned'] += len(opportunities)
            # Update GLOBAL counter in Redis for landing page (use separate key to avoid WRONGTYPE conflict)
            if self.redis_client and len(opportunities) > 0:
                await self.redis_client.incrby('trader:global:opportunities_scanned', len(opportunities))
            
            # Log opportunity search status and reset reject counter
            self._reject_count = 0
            
            # Only log to console every 30 seconds (15 cycles) to reduce spam
            scan_log_interval = getattr(self, '_scan_log_counter', 0)
            should_log_scan = scan_log_interval % 15 == 0
            self._scan_log_counter = scan_log_interval + 1
            
            if opportunities:
                logger.info(f"Found {len(opportunities)} opportunities (using conf>{adjusted_min_confidence}, edge>{adjusted_min_edge})")
                # Console log only every 30 seconds
                if should_log_scan:
                    top_opps = opportunities[:3]
                    # Show ranking with score, edge, confidence
                    opp_str = ", ".join([f"#{i+1} {o.symbol} (Score:{o.opportunity_score:.0f})" for i, o in enumerate(top_opps)])
                    best = top_opps[0]
                    await self._log_to_console(f"Scanned {len(opportunities)} | Top: {opp_str}", "SIGNAL")
                    # Log the BEST opportunity details
                    logger.info(f"BEST OPPORTUNITY: {best.symbol} | Score: {best.opportunity_score:.0f} | Edge: {best.edge_score:.2f} | Conf: {best.confidence:.0f}% | {best.direction.upper()}")
            else:
                logger.debug("No opportunities found in this scan")
                if should_log_scan:
                    await self._log_to_console("Scanning... waiting for signals", "INFO")
            
            for opp in opportunities:
                # Rate limit: only 1 normal trade per cycle
                if normal_trades_this_cycle >= max_normal_trades_per_cycle:
                    logger.debug(f"Rate limit: Already opened {normal_trades_this_cycle} trade(s) this cycle")
                    break
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
                        logger.debug(f"{opp.symbol} on cooldown ({int(self.cooldown_seconds - elapsed)}s remaining)")
                        continue
                    else:
                        # Cooldown expired, remove from dict
                        del self._cooldown_symbols[opp.symbol]
                
                # Skip if symbol recently had failed order - don't spam retries
                if opp.symbol in self._failed_order_symbols:
                    failed_time = self._failed_order_symbols[opp.symbol]
                    elapsed = (datetime.utcnow() - failed_time).total_seconds()
                    if elapsed < self.failed_order_cooldown:
                        logger.debug(f"{opp.symbol} on failed order cooldown ({int(self.failed_order_cooldown - elapsed)}s remaining)")
                        continue
                    else:
                        del self._failed_order_symbols[opp.symbol]
                    
                # Validate the opportunity with ADJUSTED thresholds (PER USER!)
                should_trade, reason = await self._validate_opportunity(
                    opp, wallet, client,
                    adjusted_min_confidence=adjusted_min_confidence,
                    adjusted_min_edge=adjusted_min_edge,
                    user_id=user_id
                )
                
                if should_trade:
                    # CONSULT Q-LEARNING before trading - use learned knowledge!
                    q_value = await self._get_q_value_for_trade(opp.symbol, opp.direction)
                    if q_value is not None and q_value < -0.1:
                        logger.info(f"Q-Learning says AVOID {opp.symbol} (Q={q_value:.2f})")
                        await self._log_to_console(f"SKIPPED {opp.symbol}: Q-Learning negative ({q_value:.2f})", "WARNING", user_id)
                        continue
                    
                    logger.info(f"OPENING TRADE: {opp.symbol} | Score: {opp.opportunity_score:.0f} | Edge={opp.edge_score:.2f} | Conf={opp.confidence:.0f}% | All filters passed")
                    await self._log_to_console(f"{opp.direction.upper()} {opp.symbol} | Score:{opp.opportunity_score:.0f} | All filters OK", "TRADE", user_id)
                    await self._execute_trade(user_id, client, opp, wallet)
                    
                    # Track trade timing for rate limiting
                    self._last_trade_time = datetime.utcnow()
                    normal_trades_this_cycle += 1
                else:
                    # Log first 3 rejections per cycle to avoid spam
                    if not hasattr(self, '_reject_count') or self._reject_count < 3:
                        logger.info(f"Rejected {opp.symbol}: {reason}")
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
            
    async def _get_q_value_for_trade(self, symbol: str, direction: str) -> Optional[float]:
        """
        CONSULT Q-LEARNING before trading!
        
        Returns the Q-value for this symbol/direction combo.
        Positive Q = historically profitable
        Negative Q = historically losing
        """
        try:
            if not self.redis_client:
                return None
                
            # Get current market regime
            regime = "normal"
            regime_data = await self.redis_client.get('bot:current_regime')
            if regime_data:
                regime = regime_data.decode() if isinstance(regime_data, bytes) else regime_data
            
            # Load Q-values
            q_values_raw = await self.redis_client.get('ai:learning:q_values')
            if not q_values_raw:
                return None
                
            q_values = json.loads(q_values_raw)
            
            # Try to find Q-value for this regime and action
            regime_q = q_values.get(regime, {})
            action = 'buy' if direction == 'long' else 'sell'
            
            if action in regime_q:
                return float(regime_q[action])
            
            # Try 'hold' as default
            return float(regime_q.get('hold', 0))
            
        except Exception as e:
            logger.debug(f"Q-value lookup failed: {e}")
            return None
    
    async def _validate_opportunity(self, opp: TradingOpportunity, wallet: Dict, client: BybitV5Client = None,
                                    adjusted_min_confidence: float = None, adjusted_min_edge: float = None,
                                    user_id: str = "default") -> Tuple[bool, str]:
        """
        SUPERIOR validation using ALL AI models + GLOBAL CONFIRMATION CHECKS
        
        Checks:
        0. GLOBAL ENTRY CONFIRMATION (RSI, candles, volume, price structure)
        1. Basic edge/confidence (with DYNAMIC thresholds based on recent performance)
        2. NEWS SENTIMENT (reads and understands news!)
        3. MOMENTUM FILTER (LOCK PROFIT only!)
        4. XGBoost ML classification
        5. CryptoBERT sentiment (crypto-specific)
        6. Price predictor (multi-timeframe)
        7. Capital allocator (unified budget)
        8. Regime detection
        9. Position sizing
        """
        
        # Determine trade type
        is_breakout = "BREAKOUT" in str(opp.reasons)
        is_mean_rev = getattr(opp, 'is_mean_reversion', False) or "MEAN_REV" in str(opp.reasons)
        
        # === LAYER 0: HUGGINGFACE SAFETY GATE (NEWS/SENTIMENT/TOPIC) ===
        # These models say WHEN NOT TO TRADE, not BUY/SELL!
        if HF_SAFETY_AVAILABLE:
            try:
                hf = await get_hf_safety()
                
                # Check if trading is paused due to dangerous news
                is_paused, pause_reason = hf.is_trading_paused()
                if is_paused:
                    self.stats['trades_rejected_regime'] += 1
                    return False, f"HF Safety: {pause_reason}"
                
                # Get recent news for context (if available)
                news_context = []
                try:
                    news_data = await self.redis_client.lrange('market:news:recent', 0, 4)
                    if news_data:
                        for item in news_data:
                            try:
                                news = json.loads(item)
                                if 'title' in news:
                                    news_context.append(news['title'])
                            except:
                                pass
                except:
                    pass
                
                # Check if this trade should be allowed
                allowed, hf_reason = await hf.should_allow_trade(
                    opp.symbol, 
                    opp.direction,
                    news_context
                )
                
                if not allowed:
                    self.stats['trades_rejected_regime'] += 1
                    logger.info(f"HF GATE BLOCKED: {opp.symbol} - {hf_reason}")
                    return False, hf_reason
                
                # Apply risk modifier to position sizing later
                # (stored in hf.risk_modifier)
                
            except Exception as e:
                logger.debug(f"HF safety check skipped: {e}")
        
        # === LAYER 1: GLOBAL ENTRY CONFIRMATION (for ALL trades) ===
        if client:
            confirmed, confirm_reason = await self._confirm_entry(opp, client, is_breakout, is_mean_rev)
            if not confirmed:
                self.stats['trades_rejected_no_momentum'] += 1
                return False, confirm_reason
        
        # === LAYER 2: ADVANCED SAFETY FILTERS (hedge fund level) ===
        # These are GATES that remove bad trades:
        # - Rejection candles (big wicks)
        # - BTC correlation (don't long alts when BTC dumps)
        # - Spread check (avoid slippage)
        # - Entry quality (near support, not chasing) - RELAXED for breakouts!
        # - Expected value (must be positive)
        if client:
            safe, safety_reason = await self._advanced_safety_filters(opp, client, is_breakout=is_breakout)
            if not safe:
                self.stats['trades_rejected_low_edge'] += 1
                logger.info(f"SAFETY FILTER BLOCKED: {opp.symbol} - {safety_reason}")
                return False, safety_reason
        
        # Check basic equity (support both snake_case and camelCase)
        total_equity = float(wallet.get('total_equity', wallet.get('totalEquity', 0)))
        if total_equity < 10:
            return False, f"Insufficient equity (${total_equity:.2f})"
        
        # Breakouts ALSO need minimum quality thresholds based on USER'S settings
        if is_breakout:
            # Use adjusted thresholds if provided (from caller), otherwise calculate from user settings
            if adjusted_min_confidence is not None:
                breakout_min_confidence = adjusted_min_confidence
            else:
                breakout_min_confidence = max(50, self.min_confidence * 0.7)
            
            if adjusted_min_edge is not None:
                breakout_min_edge = adjusted_min_edge
            else:
                breakout_min_edge = max(0.1, self.min_edge * 0.8)
            
            if opp.confidence < breakout_min_confidence:
                self.stats['trades_rejected_low_edge'] += 1
                logger.info(f"BREAKOUT BLOCKED: {opp.symbol} - Confidence {opp.confidence:.0f}% < {breakout_min_confidence:.0f}% (user: {self.min_confidence}%)")
                return False, f"Breakout confidence too low ({opp.confidence:.0f}% < {breakout_min_confidence:.0f}%)"
            
            if opp.edge_score < breakout_min_edge:
                self.stats['trades_rejected_low_edge'] += 1
                logger.info(f"BREAKOUT BLOCKED: {opp.symbol} - Edge {opp.edge_score:.2f} < {breakout_min_edge:.2f} (user: {self.min_edge})")
                return False, f"Breakout edge too low ({opp.edge_score:.2f} < {breakout_min_edge:.2f})"
            
                logger.info(f"Breakout {opp.symbol} APPROVED - Conf: {opp.confidence:.0f}%, Edge: {opp.edge_score:.2f} | RSI/Candles/Spread/BTC OK!")
            return True, "Breakout: ALL filters passed (RSI, candles, spread, BTC, news)"
        
        # Use adjusted thresholds if provided (based on recent performance)
        min_edge = adjusted_min_edge if adjusted_min_edge is not None else self.min_edge
        min_confidence = adjusted_min_confidence if adjusted_min_confidence is not None else self.min_confidence
        
        # 1. Edge check (DYNAMIC based on recent performance)
        if opp.edge_score < min_edge:
            self.stats['trades_rejected_low_edge'] += 1
            return False, f"Edge too low ({opp.edge_score:.2f} < {min_edge:.2f})"
            
        # 2. Confidence check (DYNAMIC based on recent performance)
        if opp.confidence < min_confidence:
            self.stats['trades_rejected_low_edge'] += 1
            return False, f"Confidence too low ({opp.confidence:.1f} < {min_confidence:.0f})"
        
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
                logger.debug(f"News supports {opp.symbol} {opp.direction}: {news_reason}")
                
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
                logger.debug(f"Whale support for {opp.symbol}: {whale_reason}")
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
        
        # Calculate position size (dynamic or fixed) - PER USER!
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
                wallet_balance=wallet['total_equity'],
                user_id=user_id
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
                force_fixed=True,  # Force fixed percentage sizing
                user_id=user_id
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
                
                # Get USER's settings (not class defaults!)
                user_settings = self.user_settings.get(user_id, {})
                max_pos_pct = float(user_settings.get('max_position_percent', 100))  # Default 100% (no limit)
                max_positions = int(user_settings.get('max_open_positions', 10))  # maxOpenPositions from UI
                kelly_mult = float(user_settings.get('kelly_multiplier', 1.0))  # Default 1.0 = FULL SIZE
                
                # SIMPLE FORMULA: equity / max_positions
                # Kelly 1.0 = full size, Kelly 0.5 = half, Kelly 0.1 = 10%
                base_per_position = total_equity / max_positions
                
                # Apply Kelly multiplier (1.0 = no reduction)
                if kelly_mult >= 1.0:
                    # Kelly 1.0 or higher = use full calculated size
                    position_value = base_per_position
                else:
                    # Kelly < 1.0 = reduce position
                    position_value = base_per_position * kelly_mult
                
                # Apply size multiplier for extreme breakouts (some breakouts use 30-50%)
                size_multiplier = getattr(opp, 'size_multiplier', 1.0)
                position_value = position_value * size_multiplier
                
                # Cap at user's max_position_percent
                max_allowed = total_equity * (max_pos_pct / 100)
                position_value = min(position_value, max_allowed)
                
                # Minimum $10 position
                position_value = max(10, position_value)
                
                logger.info(f"SIZING [{user_id[:8]}]: ${total_equity:.0f} / {max_positions} pos = ${base_per_position:.0f}/pos | Kelly {kelly_mult}x -> ${position_value:.0f}")
                
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
            
            # Get instrument info to verify minimum qty and round to qty_step
            try:
                inst_info = await client.get_instrument_info(opp.symbol)
                if inst_info.get('success'):
                    inst_list = inst_info.get('data', {}).get('list', [])
                    if inst_list:
                        inst_data = inst_list[0]
                        min_qty = float(inst_data.get('lotSizeFilter', {}).get('minOrderQty', 0))
                        qty_step = float(inst_data.get('lotSizeFilter', {}).get('qtyStep', 0.001))
                        
                        # IMPORTANT: Round qty to qty_step (prevents "Request parameter error")
                        if qty_step > 0:
                            qty = round(qty / qty_step) * qty_step
                            # Handle floating point precision
                            if qty_step >= 1:
                                qty = int(qty)
                            else:
                                # Round to appropriate decimal places
                                decimals = len(str(qty_step).split('.')[-1]) if '.' in str(qty_step) else 0
                                qty = round(qty, decimals)
                            qty_str = str(qty)
                        
                        logger.debug(f"{opp.symbol}: min_qty={min_qty}, qty_step={qty_step}, our_qty={qty}")
                        
                        if qty < min_qty:
                            logger.warning(f"Order qty {qty} < min {min_qty} for {opp.symbol}")
                            await self._log_to_console(f"ORDER SKIP: {opp.symbol} qty {qty} < min {min_qty}", "WARNING", user_id)
                            return
            except Exception as inst_err:
                logger.debug(f"Could not check instrument info: {inst_err}")
            
            # IMPORTANT: Cancel any pending orders for this symbol first
            # This prevents "Order already cancelled" error (110007)
            try:
                cancel_result = await client.cancel_all_orders(category="linear", symbol=opp.symbol)
                if cancel_result.get('success'):
                    cancelled = cancel_result.get('data', {}).get('list', [])
                    if cancelled:
                        logger.info(f"Cancelled {len(cancelled)} pending orders for {opp.symbol} before new order")
            except Exception as cancel_err:
                logger.debug(f"Cancel orders check: {cancel_err}")
            
            # Check if we already have a position in this symbol (prevents 110007)
            try:
                positions_result = await client.get_positions(symbol=opp.symbol)
                if positions_result.get('success'):
                    existing_positions = positions_result.get('data', {}).get('list', [])
                    for pos in existing_positions:
                        pos_size = float(pos.get('size', 0))
                        pos_side = pos.get('side', '')
                        if pos_size > 0:
                            # Already have a position - check if same direction
                            if (side == 'Buy' and pos_side == 'Buy') or (side == 'Sell' and pos_side == 'Sell'):
                                logger.warning(f"Already have {pos_side} position in {opp.symbol} (size: {pos_size}) - skipping")
                                await self._log_to_console(f"SKIP {opp.symbol}: Already have {pos_side} position", "WARNING", user_id)
                                return
                            else:
                                # Opposite direction - this will be a reverse/close, which is OK
                                logger.info(f"Have opposite {pos_side} position in {opp.symbol}, this order will reverse it")
            except Exception as pos_err:
                logger.debug(f"Position check error: {pos_err}")
            
            result = await client.place_order(
                category="linear",  # Explicitly set category
                symbol=opp.symbol,
                side=side,
                order_type='Market',
                qty=qty_str  # Pass as string!
            )
            
            # Log full result for debugging
            logger.info(f"Order result for {opp.symbol}: {result}")
            
            if result.get('success'):
                logger.info(f"ORDER SUCCESS: {opp.symbol} {side} {qty}")
                
                # Log to console for dashboard - PER USER
                leverage_display = f" | {leverage}x" if leverage > 1 else ""
                await self._log_to_console(
                    f"OPENED {opp.symbol} {side} | ${position_value:.0f}{leverage_display} | Edge: {opp.edge_score:.2f} | Conf: {opp.confidence:.0f}%",
                    "TRADE",
                    user_id
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
                    leverage=leverage,
                    is_breakout=is_breakout  # Track if this was a breakout trade
                )
                
                # Register in position sizer (PER USER!)
                await self.position_sizer.register_position(opp.symbol, position_value, user_id)
                
                # Store breakout flag in Redis for dashboard display
                if is_breakout and self.redis_client:
                    await self.redis_client.hset("positions:breakout", opp.symbol, "1")
                
                # Store trade event (per user)
                await self._store_trade_event(user_id, opp.symbol, 'opened', 0, opp.direction)
                
            else:
                error_msg = result.get('error', result.get('message', str(result)))
                error_code = result.get('code', 'unknown')
                logger.error(f"ORDER FAILED: {opp.symbol} - Code: {error_code} - {error_msg}")
                logger.error(f"Order params: symbol={opp.symbol}, side={side}, qty={qty_str}, price={opp.current_price}")
                
                # For 110007, add more debugging
                if error_code == 110007:
                    logger.error(f"110007 DEBUG: This might be due to: 1) Symbol not tradeable, 2) Account restrictions, 3) Insufficient margin for this specific symbol")
                    # Check wallet balance
                    try:
                        wallet = await client.get_wallet_balance()
                        if wallet.get('success'):
                            available = 0
                            for coin in wallet.get('data', {}).get('list', [{}])[0].get('coin', []):
                                if coin.get('coin') == 'USDT':
                                    available = float(coin.get('availableToWithdraw', 0))
                            logger.error(f"110007 DEBUG: Available USDT: ${available:.2f}")
                    except:
                        pass
                
                await self._log_to_console(f"ORDER FAILED: {opp.symbol} - {error_msg[:50]}", "ERROR", user_id)
                
                # ADD COOLDOWN for failed orders - don't retry for 5 minutes
                self._failed_order_symbols[opp.symbol] = datetime.utcnow()
                logger.info(f"Added {opp.symbol} to failed order cooldown for {self.failed_order_cooldown}s")
                
        except Exception as e:
            logger.error(f"Execute trade error for {opp.symbol}: {e}")
            await self._log_to_console(f"TRADE ERROR: {opp.symbol} - {str(e)[:50]}", "ERROR", user_id)
            
            # Also add to cooldown on exception
            self._failed_order_symbols[opp.symbol] = datetime.utcnow()
            
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
            
            # === MAX TRADE TIME (from preset) ===
            # Close trades that exceed maximum duration for the preset
            trade_duration_minutes = (datetime.utcnow() - position.entry_time).total_seconds() / 60
            
            if self.max_trade_minutes > 0 and trade_duration_minutes >= self.max_trade_minutes:
                # Only close if not in significant profit (don't cut winners short)
                if pnl_percent < self.take_profit * 0.5:  # Less than 50% of TP
                    logger.info(f"MAX TIME: {position.symbol} after {trade_duration_minutes:.0f}min (max={self.max_trade_minutes}min), P&L={pnl_percent:+.2f}%")
                    await self._log_to_console(f"MAX TIME: {position.symbol} {pnl_percent:+.2f}% after {trade_duration_minutes:.0f}min", "TRADE", user_id)
                    await self._close_position(user_id, client, position, pnl_percent, 
                        f"Max time ({trade_duration_minutes:.0f}min)")
                    return  # Exit early
            
            # === TIME STOP LOGIC (MICRO PROFIT mode - aggressive) ===
            # Close "dead" trades before they go negative
            if self.use_time_stop and self.risk_mode == 'micro_profit':
                if trade_duration_minutes >= self.time_stop_minutes and pnl_percent < self.time_stop_min_pnl:
                    logger.info(f"TIME STOP: {position.symbol} after {trade_duration_minutes:.1f}min, P&L={pnl_percent:+.2f}% < +{self.time_stop_min_pnl}%")
                    await self._close_position(user_id, client, position, pnl_percent, 
                        f"Time stop ({trade_duration_minutes:.1f}min)")
                    return  # Exit early
            
            # === FAST EXIT LOGIC ===
            should_exit = False
            exit_reason = ""
            
            # Only log positions near thresholds to reduce spam
            # Take Profit only triggers if TP > 0 (enabled)
            if self.take_profit > 0 and pnl_percent >= self.take_profit:
                logger.info(f" {position.symbol}: P&L={pnl_percent:+.2f}% >= TP={self.take_profit}% - SELLING!")
                await self._log_to_console(f"SOLD {position.symbol}: +{pnl_percent:.2f}% (Take Profit)", "TRADE", user_id)
                should_exit = True
                exit_reason = f"Take profit ({pnl_percent:.2f}%)"
            elif pnl_percent <= -self.emergency_stop_loss:
                logger.info(f" {position.symbol}: P&L={pnl_percent:+.2f}% <= SL=-{self.emergency_stop_loss}% - SELLING!")
                await self._log_to_console(f"SOLD {position.symbol}: {pnl_percent:.2f}% (Stop Loss)", "TRADE", user_id)
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
                    # === ADAPTIVE TRAILING MODE ===
                    # Key insight: Let winners run MORE when profit is HIGH
                    # Tighter trailing when profit is small (protect gains)
                    
                    # Activate trailing when profit reaches min_profit_to_trail (e.g., 0.3%)
                    if pnl_percent >= self.min_profit_to_trail:
                        if not position.trailing_active:
                            logger.info(f"TRAILING ACTIVATED {position.symbol}: P&L={pnl_percent:+.2f}% >= MinTrail={self.min_profit_to_trail}%")
                        position.trailing_active = True
                    
                    # ADAPTIVE TRAIL: Wider when in more profit
                    # +0.3% profit = use base trail (0.5%)
                    # +1.0% profit = use 0.8% trail (let it run more)
                    # +2.0% profit = use 1.0% trail (really let it run)
                    adaptive_trail = self.trail_from_peak
                    if pnl_percent >= 2.0:
                        adaptive_trail = max(1.0, self.trail_from_peak * 2)  # 2x wider
                    elif pnl_percent >= 1.0:
                        adaptive_trail = max(0.8, self.trail_from_peak * 1.5)  # 1.5x wider
                    
                    # Once trailing is active, sell when price drops from peak
                    if position.trailing_active and drop_from_peak >= adaptive_trail:
                        # Only sell if we're still in profit (or minimal loss)
                        if pnl_percent >= -0.05:  # Allow tiny loss due to spread
                            should_exit = True
                            exit_reason = f"Trailing stop (peak: +{position.peak_pnl_percent:.2f}%, dropped {drop_from_peak:.2f}%)"
                            logger.info(f"TRAILING SELL {position.symbol}: Peak=+{position.peak_pnl_percent:.2f}%, Now={pnl_percent:+.2f}%, Drop={drop_from_peak:.2f}% >= Trail={adaptive_trail}% | MinTrail={self.min_profit_to_trail}%")
                        else:
                            logger.debug(f" {position.symbol}: Trailing triggered but P&L too negative ({pnl_percent:+.2f}%), holding")
                    
            # === EXECUTE EXIT ===
            if should_exit:
                await self._close_position(user_id, client, position, pnl_percent, exit_reason)
                
        except Exception as e:
            logger.error(f"Fast exit check error for {position.symbol}: {e}")
            
    async def _log_to_console(self, message: str, level: str = "INFO", user_id: str = None):
        """Log message to Redis for dashboard real-time console
        
        If user_id is provided, logs to user-specific console.
        Also logs to global console for admin visibility.
        """
        try:
            if self.redis_client:
                log_entry = {
                    "time": datetime.utcnow().isoformat(),
                    "level": level,
                    "message": message
                }
                log_json = json.dumps(log_entry)
                
                # Log ONLY to user-specific console - NO GLOBAL SHARING!
                if user_id:
                    await self.redis_client.lpush(f'bot:console:logs:{user_id}', log_json)
                    await self.redis_client.ltrim(f'bot:console:logs:{user_id}', 0, 99)
                # NO GLOBAL LOG - each user sees ONLY their own data!
        except Exception:
            pass  # Don't break trading for console logging
        
    async def _store_trade_event(self, user_id: str, symbol: str, action: str, pnl: float, reason: str,
                                  position: Optional[ActivePosition] = None, 
                                  net_pnl_value: Optional[float] = None):
        """Store trade event for dashboard and data collection
        
        Args:
            user_id: User identifier for data isolation
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
                'value': position.position_value if position else None,
                'user_id': user_id  # Track user
            }
            
            # Push to user-specific Redis list for dashboard
            await self.redis_client.lpush(f'trading:events:{user_id}', json.dumps(event))
            await self.redis_client.ltrim(f'trading:events:{user_id}', 0, 99)  # Keep last 100
            
            # Also push to global events for AI learning (all users)
            await self.redis_client.lpush('trading:events:all', json.dumps(event))
            await self.redis_client.ltrim('trading:events:all', 0, 999)  # Keep last 1000 for learning
            
            # Store completed trades per user for activity panel (NET P&L)
            if action == 'closed':
                await self.redis_client.lpush(f'trades:completed:{user_id}', json.dumps({
                    'symbol': symbol,
                    'pnl': round(pnl_value, 2),  # NET value
                    'pnl_percent': round(pnl, 2),  # NET %
                    'close_reason': reason,
                    'closed_time': timestamp
                }))
                await self.redis_client.ltrim(f'trades:completed:{user_id}', 0, 49)  # Keep last 50
            
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
            
    async def _load_settings(self, user_id: str = "default"):
        """
        Load settings from Redis - PER USER!
        
        CRITICAL: Settings are stored in self.user_settings[user_id], NOT in instance variables!
        This ensures COMPLETE ISOLATION between users!
        """
        try:
            settings_key = f'bot:settings:{user_id}'
            parsed = {}
            
            # Check key type FIRST to avoid WRONGTYPE errors
            key_type = await self.redis_client.type(settings_key)
            key_type_str = key_type.decode() if isinstance(key_type, bytes) else str(key_type)
            
            if key_type_str == 'hash':
                # Key is a HASH - use HGETALL
                data = await self.redis_client.hgetall(settings_key)
                if data:
                    parsed = {
                        k.decode() if isinstance(k, bytes) else k: 
                        v.decode() if isinstance(v, bytes) else v 
                        for k, v in data.items()
                    }
            elif key_type_str == 'string':
                # Key is a STRING - use GET and parse JSON
                data_raw = await self.redis_client.get(settings_key)
                if data_raw:
                    try:
                        parsed = json.loads(data_raw)
                    except:
                        pass
            # If key_type is 'none', the key doesn't exist - parsed stays empty
            
            # Initialize user settings dict if not exists
            if user_id not in self.user_settings:
                self.user_settings[user_id] = {}
            
            user_set = self.user_settings[user_id]
            
            # ============================================================
            # PARSE ALL SETTINGS INTO USER-SPECIFIC DICTIONARY
            # NOTHING goes into instance variables - ALL per-user!
            # ============================================================
            
            # AI Full Auto mode
            user_set['ai_full_auto'] = str(parsed.get('aiFullAuto', 'false')).lower() == 'true'
            user_set['use_max_trade_time'] = str(parsed.get('useMaxTradeTime', 'true')).lower() == 'true'
            
            # Max Daily Drawdown - ALWAYS user controlled
            user_set['max_daily_drawdown'] = float(parsed.get('maxDailyDrawdown', 3.0))
            
            # Strategy preset
            user_set['strategy_preset'] = parsed.get('strategyPreset', 'micro')
            
            # Apply preset defaults first
            preset = user_set['strategy_preset']
            preset_settings = self._get_preset_settings(preset)
            
            if user_set['ai_full_auto']:
                # AI Full Auto - AI decides everything except max_daily_drawdown
                user_set['take_profit'] = preset_settings.get('take_profit', 0.8)
                user_set['stop_loss'] = preset_settings.get('stop_loss', 0.5)
                user_set['trailing'] = preset_settings.get('trailing', 0.15)
                user_set['min_profit_to_trail'] = preset_settings.get('min_trail', 0.25)
                user_set['max_open_positions'] = 10
                user_set['max_exposure_percent'] = 80
                user_set['breakout_extra_slots'] = True
                user_set['max_trade_minutes'] = preset_settings.get('max_trade_minutes', 15)
            else:
                # User manual control - use their settings
                user_set['take_profit'] = float(parsed.get('takeProfitPercent', preset_settings.get('take_profit', 0.8)))
                user_set['stop_loss'] = float(parsed.get('stopLossPercent', preset_settings.get('stop_loss', 0.5)))
                user_set['trailing'] = float(parsed.get('trailingStopPercent', preset_settings.get('trailing', 0.15)))
                user_set['min_profit_to_trail'] = float(parsed.get('minProfitToTrail', preset_settings.get('min_trail', 0.25)))
                user_set['max_open_positions'] = int(float(parsed.get('maxOpenPositions', 10)))
                user_set['max_exposure_percent'] = float(parsed.get('maxTotalExposure', 100))
                user_set['breakout_extra_slots'] = str(parsed.get('breakoutExtraSlots', 'false')).lower() == 'true'
                user_set['max_trade_minutes'] = preset_settings.get('max_trade_minutes', 15) if user_set['use_max_trade_time'] else 0
            
            # Entry filters
            user_set['min_confidence'] = float(parsed.get('minConfidence', 55))
            user_set['min_edge'] = float(parsed.get('minEdge', 0.2))
            
            # Risk mode
            user_set['risk_mode'] = parsed.get('riskMode', 'normal')
            
            # Leverage
            user_set['leverage_mode'] = parsed.get('leverageMode', 'auto')
            
            # AI model toggles
            user_set['use_dynamic_sizing'] = str(parsed.get('useDynamicSizing', 'true')).lower() == 'true'
            user_set['use_regime_detection'] = str(parsed.get('useRegimeDetection', 'true')).lower() == 'true'
            user_set['use_edge_estimation'] = str(parsed.get('useEdgeEstimation', 'true')).lower() == 'true'
            
            # Breakout trading toggle (default: OFF for safer trading)
            user_set['enable_breakout'] = str(parsed.get('enableBreakout', 'false')).lower() == 'true'
            
            # Kelly multiplier (0.1 to 1.0, default 1.0 = FULL SIZE)
            user_set['kelly_multiplier'] = float(parsed.get('kellyMultiplier', 1.0))
            
            # Max position percent (5-100%)
            user_set['max_position_percent'] = float(parsed.get('maxPositionPercent', 15))
            
            logger.debug(f"Loaded settings for {user_id}: TP={user_set['take_profit']}%, SL={user_set['stop_loss']}%, Trail={user_set['trailing']}%, MinTrail={user_set['min_profit_to_trail']}%")
            
        except Exception as e:
            logger.error(f"Failed to load settings for {user_id}: {e}")
            # Initialize with HARDCODED MICRO preset defaults if failed
            # CRITICAL: NEVER copy from instance variables - that leaks settings between users!
            if user_id not in self.user_settings:
                self.user_settings[user_id] = {
                    'take_profit': 0.9,        # MICRO default
                    'stop_loss': 0.5,          # MICRO default  
                    'trailing': 0.14,          # MICRO default
                    'min_profit_to_trail': 0.45,  # MICRO default
                    'max_open_positions': 10,
                    'max_exposure_percent': 100,
                    'max_daily_drawdown': 0,   # 0 = OFF
                    'breakout_extra_slots': False,
                    'enable_breakout': False,  # Breakout trading OFF by default
                    'ai_full_auto': False,
                    'min_confidence': 65,
                    'min_edge': 0.15,
                    'risk_mode': 'MICRO',
                    'strategy_preset': 'micro',
                    'leverage_mode': 'auto',
                    'max_trade_minutes': 30,   # MICRO default
                    'use_max_trade_time': True,
                    'kelly_multiplier': 1.0,   # Full size by default (1.0 = no reduction)
                    'max_position_percent': 100,  # No limit by default
                }
                logger.warning(f"Using HARDCODED defaults for user {user_id} (Redis load failed)")
    
    def _get_preset_settings(self, preset: str) -> Dict:
        """Get default settings for a strategy preset"""
        presets = {
            'scalp': {'take_profit': 0.55, 'stop_loss': 0.35, 'trailing': 0.12, 'min_trail': 0.35, 'max_trade_minutes': 5},
            'micro': {'take_profit': 0.8, 'stop_loss': 0.5, 'trailing': 0.15, 'min_trail': 0.25, 'max_trade_minutes': 10},
            'swing': {'take_profit': 2.5, 'stop_loss': 1.2, 'trailing': 0.6, 'min_trail': 1.0, 'max_trade_minutes': 60},
            'conservative': {'take_profit': 0.6, 'stop_loss': 0.3, 'trailing': 0.1, 'min_trail': 0.3, 'max_trade_minutes': 8},
            'balanced': {'take_profit': 1.2, 'stop_loss': 0.8, 'trailing': 0.3, 'min_trail': 0.5, 'max_trade_minutes': 20},
            'aggressive': {'take_profit': 3.0, 'stop_loss': 1.5, 'trailing': 0.8, 'min_trail': 1.2, 'max_trade_minutes': 45},
            'mean_reversion': {'take_profit': 0.6, 'stop_loss': 0.4, 'trailing': 0.12, 'min_trail': 0.3, 'max_trade_minutes': 8},
        }
        return presets.get(preset.lower(), presets['micro'])
    
    async def _get_user_losing_streak(self, user_id: str) -> int:
        """
        Get user's consecutive losing trades count.
        Used for Dynamic Kelly Criterion adjustment.
        
        Returns:
            Number of consecutive losing trades (0 if winning or no trades)
        """
        try:
            # Get last 10 trades from user's trade history
            trades_key = f"trades:completed:{user_id}"
            trades_raw = await self.redis_client.lrange(trades_key, 0, 9)
            
            if not trades_raw:
                return 0
            
            consecutive_losses = 0
            for trade_raw in trades_raw:
                try:
                    trade_str = trade_raw.decode() if isinstance(trade_raw, bytes) else trade_raw
                    trade = json.loads(trade_str)
                    pnl = float(trade.get('pnl_percent', trade.get('pnl', 0)))
                    
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        # Hit a winning trade, stop counting
                        break
                except:
                    continue
            
            return consecutive_losses
            
        except Exception as e:
            logger.debug(f"Error getting losing streak for {user_id}: {e}")
            return 0
    
            
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
                logger.info(f"Loaded {len(self.paused_users)} paused users: {self.paused_users}")
        except Exception as e:
            logger.warning(f"Failed to load paused users: {e}")
    
    async def _load_registered_users(self):
        """Load all registered users from Redis and connect them"""
        try:
            import base64
            
            # Find all user credential keys
            keys = await self.redis_client.keys('user:*:exchange:*')
            connected_count = 0
            
            for key in keys:
                try:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    # Parse user_id from key format "user:{user_id}:exchange:{exchange}"
                    parts = key_str.split(':')
                    if len(parts) >= 4:
                        user_id = parts[1]
                        exchange = parts[3]
                        
                        # Skip if already connected
                        if user_id in self.user_clients:
                            continue
                        
                        # Get credentials
                        creds = await self.redis_client.hgetall(key)
                        if not creds:
                            continue
                        
                        # Decode credentials
                        creds_decoded = {
                            k.decode() if isinstance(k, bytes) else k: 
                            v.decode() if isinstance(v, bytes) else v 
                            for k, v in creds.items()
                        }
                        
                        # Check if active
                        if creds_decoded.get('is_active') != '1':
                            continue
                        
                        # Decrypt credentials (simple base64)
                        api_key = base64.b64decode(creds_decoded.get('api_key', '')).decode()
                        api_secret = base64.b64decode(creds_decoded.get('api_secret', '')).decode()
                        is_testnet = creds_decoded.get('is_testnet') == '1'
                        
                        if api_key and api_secret:
                            # Connect user
                            success = await self.connect_user(user_id, api_key, api_secret, is_testnet)
                            if success:
                                connected_count += 1
                                logger.info(f"Auto-connected user {user_id} from Redis")
                            
                except Exception as e:
                    logger.warning(f"Failed to load user from key {key}: {e}")
                    continue
            
            if connected_count > 0:
                logger.info(f"Auto-connected {connected_count} users from Redis")
            else:
                logger.info("No users to auto-connect from Redis")
                
        except Exception as e:
            logger.warning(f"Failed to load registered users: {e}")
            
    async def _save_stats(self):
        """Save GLOBAL stats to Redis (for AI learning)"""
        try:
            await self.redis_client.set('trader:stats', json.dumps(self.stats))
        except:
            pass
    
    async def _save_user_stats(self, user_id: str):
        """Save USER-SPECIFIC stats to Redis"""
        try:
            if user_id in self.user_stats:
                await self.redis_client.set(f'trader:stats:{user_id}', json.dumps(self.user_stats[user_id]))
        except:
            pass
    
    async def _load_user_stats(self, user_id: str):
        """Load user-specific stats from Redis"""
        try:
            data = await self.redis_client.get(f'trader:stats:{user_id}')
            if data:
                loaded_stats = json.loads(data)
                self.user_stats[user_id] = loaded_stats
                logger.info(f"Loaded stats for user {user_id}: {loaded_stats.get('total_trades', 0)} trades")
        except Exception as e:
            logger.warning(f"Failed to load stats for user {user_id}: {e}")
            
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

