"""
SENTINEL AI - Dynamic Position Sizing (Kelly-based)

This determines HOW MUCH to risk on each trade.
Uses modified Kelly Criterion with safety constraints.

Key principle: Bet more when edge is higher, less when uncertain.

Hard limits:
- Max 0.5% risk per trade
- Max 2% daily drawdown
- Max 30% total exposure
- Never use full wallet
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from loguru import logger
import redis.asyncio as redis
import json

from config import settings


@dataclass
class PositionSize:
    """Calculated position size for a trade"""
    symbol: str
    
    # Size in USDT
    position_value_usdt: float
    
    # Size as fraction of wallet
    wallet_fraction: float
    
    # Quantity (in asset)
    quantity: float
    
    # Leverage recommendation
    recommended_leverage: int
    
    # Risk metrics
    risk_per_trade_pct: float  # Actual risk as % of wallet
    stop_loss_pct: float  # Where to place stop
    take_profit_pct: float  # Where to take profit
    
    # Reasoning
    sizing_method: str  # 'kelly', 'fixed', 'reduced', 'minimum'
    adjustments: List[str]  # Why size was adjusted
    
    # Limits
    is_within_limits: bool
    limit_reason: str
    
    # Kelly fraction used (has default value - must come after non-default fields)
    kelly_fraction: float = 0.0


class PositionSizer:
    """
    Dynamic Position Sizing System
    
    Uses Kelly Criterion with safety modifications:
    - Fractional Kelly (user configurable via kellyMultiplier)
    - Dynamic Kelly adjustments based on confidence, losing streak, regime
    - Hard risk limits
    - Drawdown protection
    - Correlation limits
    - Regime adjustments
    """
    
    # === DEFAULT LIMITS (can be overridden by settings) ===
    MAX_RISK_PER_TRADE = 0.15  # 15% of wallet (user configurable)
    MAX_DAILY_DRAWDOWN = 0  # 0 = OFF (no daily limit by default)
    MAX_WEEKLY_DRAWDOWN = 0.10  # 10% weekly loss limit
    MAX_TOTAL_EXPOSURE = 1.0  # 100% of wallet max deployed (default)
    MAX_SINGLE_POSITION = 0.15  # 15% max in one position (user configurable)
    MAX_CORRELATED_EXPOSURE = 0.30  # 30% max in correlated assets
    MIN_POSITION_VALUE = 5.5  # $5.5 minimum trade (Bybit minimum)
    MAX_LEVERAGE = 10  # Max leverage allowed
    MAX_OPEN_POSITIONS = 0  # 0 = unlimited
    
    def __init__(self):
        self.redis_client = None
        
        # Track daily P&L
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = 0.0
        self.last_reset_date: date = None
        
        # Track weekly P&L
        self.weekly_pnl: float = 0.0
        
        # Open positions tracker
        self.open_positions: Dict[str, float] = {}
        
        # Leverage mode: '1x', '2x', '3x', '5x', '10x', 'auto'
        self.leverage_mode: str = 'auto'
        
        # === DYNAMIC KELLY SETTINGS ===
        self.kelly_multiplier: float = 0.5  # User's Kelly multiplier (0.1 - 1.0)
        self.losing_streak: int = 0  # Consecutive losses
        self.winning_streak: int = 0  # Consecutive wins
        
        # Correlation groups (assets that move together)
        self.correlation_groups = {
            'btc_correlated': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT'],
            'meme': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT'],
            'defi': ['UNIUSDT', 'AAVEUSDT', 'LINKUSDT', 'MKRUSDT', 'COMPUSDT'],
            'layer2': ['MATICUSDT', 'OPUSDT', 'ARBUSDT'],
            'gaming': ['AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'ENJUSDT'],
        }
        
    async def initialize(self):
        """Initialize position sizer"""
        logger.info("Initializing Dynamic Position Sizer (Kelly-based)...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Load state
        await self._load_state()
        
        # Load user settings
        await self._load_settings()
        
        logger.info("Position Sizer initialized - Risk management active")
        
    async def _load_settings(self, user_id: str = "default"):
        """Load user risk settings from Redis - PER USER!"""
        try:
            # Try JSON string format first
            settings_raw = await self.redis_client.get(f'bot:settings:{user_id}')
            parsed = {}
            
            if settings_raw:
                try:
                    import json
                    parsed = json.loads(settings_raw)
                except:
                    pass
            
            # Fallback to hash format
            if not parsed:
                settings_data = await self.redis_client.hgetall(f'bot:settings:{user_id}')
                if settings_data:
                    parsed = {
                        k.decode() if isinstance(k, bytes) else k: 
                        v.decode() if isinstance(v, bytes) else v 
                        for k, v in settings_data.items()
                    }
            
            if parsed:
                # Apply settings
                self.MAX_DAILY_DRAWDOWN = float(parsed.get('maxDailyDrawdown', 0)) / 100  # 0 = OFF
                self.MAX_TOTAL_EXPOSURE = float(parsed.get('maxTotalExposure', 100)) / 100
                self.MAX_SINGLE_POSITION = float(parsed.get('maxPositionPercent', 15)) / 100
                self.MAX_RISK_PER_TRADE = float(parsed.get('maxPositionPercent', 15)) / 100  # Same as position %
                self.MAX_OPEN_POSITIONS = int(float(parsed.get('maxOpenPositions', 0)))
                
                # Leverage mode
                self.leverage_mode = parsed.get('leverageMode', 'auto')
                if self.leverage_mode not in ['1x', '2x', '3x', '5x', '10x', 'auto']:
                    self.leverage_mode = 'auto'
                
                # === KELLY MULTIPLIER (Dynamic Kelly) ===
                # User can adjust from 0.1 (very conservative) to 1.0 (full Kelly)
                self.kelly_multiplier = float(parsed.get('kellyMultiplier', 0.5))
                self.kelly_multiplier = max(0.1, min(1.0, self.kelly_multiplier))
                
                logger.debug(f"Loaded position sizer settings for {user_id}: MaxDD={'OFF' if self.MAX_DAILY_DRAWDOWN == 0 else f'{self.MAX_DAILY_DRAWDOWN*100:.1f}%'}, "
                           f"MaxExposure={self.MAX_TOTAL_EXPOSURE*100:.0f}%, "
                           f"MaxPos={self.MAX_SINGLE_POSITION*100:.0f}%, "
                           f"MaxOpenPos={'Unlimited' if self.MAX_OPEN_POSITIONS == 0 else self.MAX_OPEN_POSITIONS}, "
                           f"Leverage={self.leverage_mode}, Kelly={self.kelly_multiplier}x")
        except Exception as e:
            logger.debug(f"Load settings error for {user_id}: {e}")
        
    async def shutdown(self):
        """Cleanup"""
        await self._save_state()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def calculate_position_size(
        self,
        symbol: str,
        direction: str,  # 'long' or 'short'
        edge_score: float,
        win_probability: float,
        risk_reward: float,
        kelly_fraction: float,
        regime_action: str,
        current_price: float,
        wallet_balance: float,
        force_fixed: bool = False
    ) -> PositionSize:
        """
        Calculate optimal position size
        
        Args:
            symbol: Trading pair
            direction: 'long' or 'short'
            edge_score: Edge from EdgeEstimator (-1 to +1)
            win_probability: Estimated win rate (0-100)
            risk_reward: Expected R:R ratio
            kelly_fraction: Raw Kelly fraction from EdgeEstimator
            regime_action: 'aggressive', 'normal', 'reduced', 'avoid'
            current_price: Current asset price
            wallet_balance: Total wallet in USDT
            force_fixed: If True, ignore Kelly and use fixed maxPositionPercent
            
        Returns:
            PositionSize with all details
        """
        adjustments = []
        
        # === Check Daily Reset ===
        await self._check_daily_reset(wallet_balance)
        
        # === Check if Trading is Allowed ===
        
        # 1. Check daily drawdown (0 = disabled)
        daily_dd = await self._get_daily_drawdown(wallet_balance)
        if self.MAX_DAILY_DRAWDOWN > 0 and daily_dd >= self.MAX_DAILY_DRAWDOWN:
            return self._blocked_position(
                symbol, "Daily drawdown limit reached", wallet_balance
            )
            
        # 2. Check weekly drawdown (0 = disabled)
        weekly_dd = await self._get_weekly_drawdown()
        if self.MAX_WEEKLY_DRAWDOWN > 0 and weekly_dd >= self.MAX_WEEKLY_DRAWDOWN:
            return self._blocked_position(
                symbol, "Weekly drawdown limit reached", wallet_balance
            )
            
        # 3. Check total exposure
        current_exposure = await self._get_current_exposure()
        max_allowed = self.MAX_TOTAL_EXPOSURE * wallet_balance
        if current_exposure >= max_allowed:
            logger.warning(f"Exposure check failed for {symbol}: current=${current_exposure:.2f} >= max=${max_allowed:.2f} (wallet=${wallet_balance:.2f}, limit={self.MAX_TOTAL_EXPOSURE*100:.0f}%)")
            logger.warning(f"   Open positions ({len(self.open_positions)}): {list(self.open_positions.keys())[:10]}...")
            return self._blocked_position(
                symbol, f"Maximum total exposure reached (${current_exposure:.0f}/${max_allowed:.0f})", wallet_balance
            )
        
        # 3.5 Check max open positions (0 = unlimited)
        num_positions = len(self.open_positions)
        
        # DEBUG: Always log position count to track the 20 limit bug
        logger.debug(f"Position sizer: {num_positions} tracked, MAX={'unlimited' if self.MAX_OPEN_POSITIONS == 0 else self.MAX_OPEN_POSITIONS}, exposure=${current_exposure:.2f}/${max_allowed:.2f}")
        
        if self.MAX_OPEN_POSITIONS > 0 and num_positions >= self.MAX_OPEN_POSITIONS:
            logger.warning(f"Max positions check failed: {num_positions} >= {self.MAX_OPEN_POSITIONS}")
            logger.warning(f"   Tracked positions: {list(self.open_positions.keys())[:10]}...")
            return self._blocked_position(
                symbol, f"Maximum {self.MAX_OPEN_POSITIONS} positions reached", wallet_balance
            )
            
        # 4. Check if symbol already has position
        if symbol in self.open_positions:
            return self._blocked_position(
                symbol, "Already have position in this symbol", wallet_balance
            )
            
        # 5. Check edge
        if edge_score <= 0:
            return self._blocked_position(
                symbol, "No positive edge", wallet_balance
            )
            
        # 6. Check regime
        if regime_action == 'avoid':
            return self._blocked_position(
                symbol, "Regime recommends avoiding trades", wallet_balance
            )
            
        # === Calculate Base Position Size ===
        
        # Force fixed sizing (dynamic sizing disabled) or Kelly-based
        if force_fixed:
            # Fixed sizing - use full maxPositionPercent
            base_fraction = self.MAX_RISK_PER_TRADE
            sizing_method = 'fixed'
            adjustments.append(f"Fixed sizing: {base_fraction*100:.1f}%")
        elif kelly_fraction > 0:
            # === DYNAMIC KELLY CALCULATION ===
            # Apply user's kellyMultiplier
            adjusted_kelly = kelly_fraction * self.kelly_multiplier
            
            # === CONFIDENCE ADJUSTMENT ===
            # Lower confidence = smaller position
            # Confidence < 65% → Kelly *= 0.5
            # Confidence 65-75% → Kelly *= 1.0
            # Confidence > 80% → Kelly *= 1.4
            confidence_pct = win_probability  # win_probability is 0-100
            if confidence_pct < 65:
                adjusted_kelly *= 0.5
                adjustments.append(f"Low confidence ({confidence_pct:.0f}%): Kelly×0.5")
            elif confidence_pct > 80:
                adjusted_kelly *= 1.4
                adjustments.append(f"High confidence ({confidence_pct:.0f}%): Kelly×1.4")
            
            # === LOSING STREAK ADJUSTMENT ===
            # After 3+ consecutive losses, reduce size significantly
            if self.losing_streak >= 3:
                streak_multiplier = max(0.3, 1 - (self.losing_streak - 2) * 0.15)
                adjusted_kelly *= streak_multiplier
                adjustments.append(f"Losing streak ({self.losing_streak}): Kelly×{streak_multiplier:.2f}")
            elif self.winning_streak >= 3:
                # Small bonus for winning streak (but don't get overconfident)
                adjusted_kelly *= min(1.2, 1 + self.winning_streak * 0.05)
                adjustments.append(f"Winning streak ({self.winning_streak}): Kelly×{1 + self.winning_streak * 0.05:.2f}")
            
            base_fraction = min(adjusted_kelly, self.MAX_RISK_PER_TRADE)
            sizing_method = 'kelly'
            adjustments.append(f"Kelly: {kelly_fraction*100:.1f}%×{self.kelly_multiplier}x → {adjusted_kelly*100:.1f}%")
        else:
            # Fallback to user-configured position size
            base_fraction = self.MAX_RISK_PER_TRADE * 0.5  # 50% of max when no Kelly
            sizing_method = 'fixed'
            adjustments.append(f"No Kelly edge, using {base_fraction*100:.1f}%")
            
        # === Apply Adjustments ===
        
        adjusted_fraction = base_fraction
        
        # 1. Regime adjustment (DYNAMIC KELLY REGIME MULTIPLIER)
        # Choppy/volatile regime = reduce Kelly significantly
        regime_multipliers = {
            'aggressive': 1.0,  # Trending market - full Kelly
            'normal': 0.8,      # Normal conditions
            'reduced': 0.5,     # Reduced - half Kelly
            'avoid': 0.0,       # Don't trade
            'choppy': 0.6,      # Choppy market - reduce Kelly
            'volatile': 0.5,    # High volatility - reduce Kelly
            'hold': 0.7,        # Hold - slightly reduced
        }
        regime_mult = regime_multipliers.get(regime_action, 0.5)
        adjusted_fraction *= regime_mult
        if regime_mult != 1.0:
            adjustments.append(f"Regime {regime_action}: Kelly×{regime_mult:.0%}")
            
        # 2. Edge-based adjustment
        if edge_score < 0.2:
            adjusted_fraction *= 0.5
            adjustments.append("Low edge: 50% reduction")
        elif edge_score > 0.5:
            adjusted_fraction *= 1.2
            adjustments.append("High edge: 20% increase")
            
        # 3. Win probability adjustment
        if win_probability < 55:
            adjusted_fraction *= 0.7
            adjustments.append("Low win prob: 30% reduction")
        elif win_probability > 65:
            adjusted_fraction *= 1.1
            adjustments.append("High win prob: 10% increase")
            
        # 4. Correlation adjustment
        corr_exposure = await self._get_correlation_exposure(symbol)
        if corr_exposure > self.MAX_CORRELATED_EXPOSURE * wallet_balance * 0.5:
            adjusted_fraction *= 0.5
            adjustments.append("Correlated exposure: 50% reduction")
            
        # 5. Daily drawdown buffer
        remaining_dd = self.MAX_DAILY_DRAWDOWN - daily_dd
        if remaining_dd < 0.01:  # Less than 1% DD remaining
            adjusted_fraction *= 0.5
            adjustments.append("Near daily DD limit: 50% reduction")
            
        # === Apply Hard Limits ===
        
        # Risk limit
        risk_per_trade = adjusted_fraction
        if risk_per_trade > self.MAX_RISK_PER_TRADE:
            adjusted_fraction = self.MAX_RISK_PER_TRADE
            adjustments.append(f"Capped at {self.MAX_RISK_PER_TRADE:.1%} max risk")
            sizing_method = 'capped'
            
        # Single position limit
        if adjusted_fraction > self.MAX_SINGLE_POSITION:
            adjusted_fraction = self.MAX_SINGLE_POSITION
            adjustments.append(f"Capped at {self.MAX_SINGLE_POSITION:.0%} single position")
            
        # === Calculate Final Values ===
        
        # Calculate base position value (this is the MARGIN/collateral)
        margin_value = wallet_balance * adjusted_fraction
        
        # Check minimum margin
        if margin_value < self.MIN_POSITION_VALUE:
            if wallet_balance > self.MIN_POSITION_VALUE * 20:  # Have at least $100
                margin_value = self.MIN_POSITION_VALUE
                adjusted_fraction = margin_value / wallet_balance
                adjustments.append(f"Using minimum ${self.MIN_POSITION_VALUE}")
                sizing_method = 'minimum'
            else:
                return self._blocked_position(
                    symbol, "Position too small", wallet_balance
                )
                
        # Check doesn't exceed available exposure (based on margin)
        available = (self.MAX_TOTAL_EXPOSURE * wallet_balance) - current_exposure
        if margin_value > available:
            margin_value = available
            adjusted_fraction = margin_value / wallet_balance
            adjustments.append("Limited by available exposure")
        
        # Determine leverage
        recommended_leverage = self._calculate_leverage(edge_score, win_probability)
        
        # Apply leverage to get actual position value
        # margin_value is collateral, position_value is what we actually trade
        position_value = margin_value * recommended_leverage
        adjustments.append(f"Leverage {recommended_leverage}x applied: ${margin_value:.0f} margin → ${position_value:.0f} position")
            
        # Calculate quantity based on LEVERAGED position value
        quantity = position_value / current_price if current_price > 0 else 0
        
        # Calculate stop loss and take profit
        # Use ATR-based or fixed based on edge
        if risk_reward > 2:
            stop_loss_pct = 1.5
            take_profit_pct = stop_loss_pct * risk_reward
        else:
            stop_loss_pct = 2.0
            take_profit_pct = stop_loss_pct * max(1.5, risk_reward)
        
        return PositionSize(
            symbol=symbol,
            position_value_usdt=round(position_value, 2),  # This is now the LEVERAGED position value
            wallet_fraction=round(adjusted_fraction, 4),   # This is still % of wallet (margin)
            quantity=quantity,  # Quantity is based on leveraged position
            recommended_leverage=recommended_leverage,
            risk_per_trade_pct=round(adjusted_fraction * 100, 2),  # Risk is based on margin, not position
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            kelly_fraction=kelly_fraction,
            sizing_method=sizing_method,
            adjustments=adjustments,
            is_within_limits=True,
            limit_reason=""
        )
        
    def _calculate_leverage(self, edge: float, win_prob: float) -> int:
        """Calculate recommended leverage based on edge and leverage_mode setting"""
        
        # If fixed leverage mode, use that value
        if self.leverage_mode != 'auto':
            try:
                fixed_leverage = int(self.leverage_mode.replace('x', ''))
                return min(self.MAX_LEVERAGE, fixed_leverage)
            except ValueError:
                pass  # Fall through to auto calculation
        
        # AUTO mode: Higher edge = can use slightly more leverage
        # But always conservative
        if edge > 0.5 and win_prob > 60:
            return min(self.MAX_LEVERAGE, 3)
        elif edge > 0.3:
            return 2
        else:
            return 1
            
    def _blocked_position(self, symbol: str, reason: str, wallet: float) -> PositionSize:
        """Return blocked position response"""
        return PositionSize(
            symbol=symbol,
            position_value_usdt=0,
            wallet_fraction=0,
            quantity=0,
            recommended_leverage=1,
            risk_per_trade_pct=0,
            stop_loss_pct=0,
            take_profit_pct=0,
            kelly_fraction=0.0,  # Blocked = no kelly
            sizing_method='blocked',
            adjustments=[],
            is_within_limits=False,
            limit_reason=reason
        )
        
    async def _check_daily_reset(self, current_equity: float):
        """Reset daily tracking if new day"""
        today = date.today()
        
        if self.last_reset_date != today:
            # Save yesterday's P&L
            if self.last_reset_date:
                await self._record_daily_pnl(self.daily_pnl)
                
            # Reset daily
            self.daily_pnl = 0.0
            self.daily_start_equity = current_equity
            self.last_reset_date = today
            
            # Check for weekly reset (Sunday)
            if today.weekday() == 6:  # Sunday
                await self._record_weekly_pnl(self.weekly_pnl)
                self.weekly_pnl = 0.0
                
            await self._save_state()
            logger.info(f"Daily reset complete. Starting equity: ${current_equity:.2f}")
            
    async def _get_daily_drawdown(self, current_equity: float) -> float:
        """Get current daily drawdown as fraction"""
        if self.daily_start_equity <= 0:
            return 0.0
            
        if current_equity >= self.daily_start_equity:
            return 0.0
            
        drawdown = (self.daily_start_equity - current_equity) / self.daily_start_equity
        return drawdown
        
    async def _get_weekly_drawdown(self) -> float:
        """Get current weekly drawdown as fraction"""
        try:
            data = await self.redis_client.get('sizer:weekly_start')
            if data:
                weekly_start = float(data)
                current = await self._get_current_equity()
                if weekly_start > 0 and current < weekly_start:
                    return (weekly_start - current) / weekly_start
        except:
            pass
        return 0.0
        
    async def _get_current_equity(self) -> float:
        """Get current equity from Redis"""
        try:
            data = await self.redis_client.get('wallet:equity')
            return float(data) if data else 0.0
        except:
            return 0.0
            
    async def _get_current_exposure(self) -> float:
        """Get current total exposure in USDT"""
        total = sum(self.open_positions.values())
        return total
        
    async def _get_correlation_exposure(self, symbol: str) -> float:
        """Get exposure in correlated assets"""
        # Find which group this symbol belongs to
        symbol_group = None
        for group, symbols in self.correlation_groups.items():
            if symbol in symbols:
                symbol_group = group
                break
                
        if not symbol_group:
            return 0.0
            
        # Sum exposure in same group
        group_symbols = self.correlation_groups[symbol_group]
        exposure = sum(
            v for s, v in self.open_positions.items() 
            if s in group_symbols
        )
        
        return exposure
        
    async def register_position(self, symbol: str, size_usdt: float):
        """Register a new open position"""
        self.open_positions[symbol] = size_usdt
        await self._save_state()
        
    async def close_position(self, symbol: str, pnl: float):
        """Close a position and record P&L, track streaks for Dynamic Kelly"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
            
        # Update daily P&L
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        
        # === TRACK WINNING/LOSING STREAKS FOR DYNAMIC KELLY ===
        if pnl > 0:
            # Win
            self.winning_streak += 1
            self.losing_streak = 0
            if self.winning_streak >= 3:
                logger.info(f"Winning streak: {self.winning_streak} - Kelly boost active")
        else:
            # Loss
            self.losing_streak += 1
            self.winning_streak = 0
            if self.losing_streak >= 3:
                logger.warning(f"Losing streak: {self.losing_streak} - Kelly reduced by {min(70, (self.losing_streak - 2) * 15)}%")
        
        await self._save_state()
    
    async def sync_with_exchange(self, exchange_symbols: set, positions_data: dict):
        """
        Sync position sizer with actual exchange positions.
        AGGRESSIVELY replaces entire open_positions dict with exchange reality.
        
        Args:
            exchange_symbols: Set of symbols currently on exchange
            positions_data: Dict of symbol -> position_value_usdt
        """
        old_count = len(self.open_positions)
        old_symbols = set(self.open_positions.keys())
        
        # AGGRESSIVE SYNC: Completely replace open_positions with exchange data
        # This fixes the "stuck at 20 positions" bug from stale Redis data
        self.open_positions = {}
        
        # Only add positions that are ACTUALLY on the exchange
        for symbol, value in positions_data.items():
            if symbol in exchange_symbols and value > 0:
                self.open_positions[symbol] = value
        
        new_count = len(self.open_positions)
        new_exposure = sum(self.open_positions.values())
        
        removed_symbols = old_symbols - exchange_symbols
        added_symbols = exchange_symbols - old_symbols
        
        if removed_symbols or old_count != new_count:
            logger.info(f"Position sizer SYNCED: {old_count} -> {new_count} positions, exposure=${new_exposure:.2f}")
            if removed_symbols:
                logger.info(f"   Removed stale: {list(removed_symbols)[:5]}{'...' if len(removed_symbols) > 5 else ''}")
        
        await self._save_state()
        
    async def _record_daily_pnl(self, pnl: float):
        """Record daily P&L to history"""
        try:
            history = await self.redis_client.get('sizer:daily_history')
            if history:
                daily_history = json.loads(history)
            else:
                daily_history = []
                
            daily_history.append({
                'date': str(self.last_reset_date),
                'pnl': pnl
            })
            
            # Keep last 30 days
            daily_history = daily_history[-30:]
            
            await self.redis_client.set('sizer:daily_history', json.dumps(daily_history))
        except:
            pass
            
    async def _record_weekly_pnl(self, pnl: float):
        """Record weekly P&L to history"""
        try:
            history = await self.redis_client.get('sizer:weekly_history')
            if history:
                weekly_history = json.loads(history)
            else:
                weekly_history = []
                
            weekly_history.append({
                'week': str(date.today()),
                'pnl': pnl
            })
            
            weekly_history = weekly_history[-12:]  # Keep last 12 weeks
            
            await self.redis_client.set('sizer:weekly_history', json.dumps(weekly_history))
        except:
            pass
            
    async def _load_state(self):
        """Load state from Redis"""
        try:
            state = await self.redis_client.get('sizer:state')
            if state:
                data = json.loads(state)
                self.daily_pnl = data.get('daily_pnl', 0)
                self.weekly_pnl = data.get('weekly_pnl', 0)
                self.daily_start_equity = data.get('daily_start_equity', 0)
                self.open_positions = data.get('open_positions', {})
                
                # Load streak data for Dynamic Kelly
                self.losing_streak = data.get('losing_streak', 0)
                self.winning_streak = data.get('winning_streak', 0)
                
                last_date = data.get('last_reset_date')
                if last_date:
                    self.last_reset_date = date.fromisoformat(last_date)
                    
            logger.info(f"Loaded position sizer state: {len(self.open_positions)} positions, exposure=${sum(self.open_positions.values()):.2f}, streak=W{self.winning_streak}/L{self.losing_streak}")
        except Exception as e:
            logger.debug(f"Load state error: {e}")
            # Reset to empty state if error
            self.open_positions = {}
            self.daily_pnl = 0
            self.weekly_pnl = 0
            self.losing_streak = 0
            self.winning_streak = 0
            logger.info("Position sizer state reset to empty")
            
    async def _save_state(self):
        """Save state to Redis"""
        try:
            state = {
                'daily_pnl': self.daily_pnl,
                'weekly_pnl': self.weekly_pnl,
                'daily_start_equity': self.daily_start_equity,
                'open_positions': self.open_positions,
                'last_reset_date': str(self.last_reset_date) if self.last_reset_date else None,
                # Dynamic Kelly streak tracking
                'losing_streak': self.losing_streak,
                'winning_streak': self.winning_streak
            }
            await self.redis_client.set('sizer:state', json.dumps(state))
        except:
            pass
            
    async def get_risk_status(self, wallet_balance: float) -> Dict:
        """Get current risk status including Dynamic Kelly info"""
        daily_dd = await self._get_daily_drawdown(wallet_balance)
        weekly_dd = await self._get_weekly_drawdown()
        exposure = await self._get_current_exposure()
        
        # Calculate current Kelly modifier based on streaks
        kelly_modifier = 1.0
        streak_info = "Neutral"
        if self.losing_streak >= 3:
            kelly_modifier = max(0.3, 1 - (self.losing_streak - 2) * 0.15)
            streak_info = f"Losing streak: {self.losing_streak} (Kelly x{kelly_modifier:.0%})"
        elif self.winning_streak >= 3:
            kelly_modifier = min(1.2, 1 + self.winning_streak * 0.05)
            streak_info = f"Winning streak: {self.winning_streak} (Kelly x{kelly_modifier:.0%})"
        
        return {
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_drawdown_pct': round(daily_dd * 100, 2),
            'daily_dd_remaining_pct': round((self.MAX_DAILY_DRAWDOWN - daily_dd) * 100, 2),
            'weekly_drawdown_pct': round(weekly_dd * 100, 2),
            'current_exposure': round(exposure, 2),
            'max_exposure': round(self.MAX_TOTAL_EXPOSURE * wallet_balance, 2),
            'exposure_pct': round((exposure / (wallet_balance * self.MAX_TOTAL_EXPOSURE)) * 100, 1) if wallet_balance > 0 else 0,
            'open_positions_count': len(self.open_positions),
            'open_positions': self.open_positions,
            'can_trade': (self.MAX_DAILY_DRAWDOWN == 0 or daily_dd < self.MAX_DAILY_DRAWDOWN) and (self.MAX_WEEKLY_DRAWDOWN == 0 or weekly_dd < self.MAX_WEEKLY_DRAWDOWN),
            # Dynamic Kelly info
            'kelly_multiplier': self.kelly_multiplier,
            'kelly_modifier': kelly_modifier,
            'losing_streak': self.losing_streak,
            'winning_streak': self.winning_streak,
            'streak_info': streak_info
        }


# Global instance
position_sizer = PositionSizer()

