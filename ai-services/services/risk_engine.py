"""
SENTINEL AI - Risk Management Engine
Capital protection and risk control
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
import redis.asyncio as redis
import json

from config import settings


class RiskStatus(Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    STOPPED = "STOPPED"


class RiskEventType(Enum):
    DAILY_LOSS_WARNING = "daily_loss_warning"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    TRADE_LOSS_LIMIT = "trade_loss_limit"
    EXPOSURE_WARNING = "exposure_warning"
    VOLATILITY_SPIKE = "volatility_spike"
    EMERGENCY_STOP = "emergency_stop"
    COOLDOWN_STARTED = "cooldown_started"
    COOLDOWN_ENDED = "cooldown_ended"


class RiskEngine:
    """
    Risk Management Engine - The Capital Bodyguard
    
    Enforces:
    - Max loss per trade
    - Max loss per day
    - Max exposure per asset
    - Auto cooldown after losses
    - Emergency exit
    - Global kill switch
    
    Philosophy: Better to not profit than to lose.
    """
    
    def __init__(self):
        self.redis_client = None
        self.active_alerts: Dict[str, List[Dict]] = {}
        self.user_cooldowns: Dict[str, datetime] = {}
        self.emergency_stopped_users: set = set()
        
    async def initialize(self):
        """Initialize risk engine"""
        logger.info("Initializing Risk Engine...")
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        logger.info("Risk Engine initialized - Capital Protection Active")
        
    async def evaluate_trade(
        self,
        user_id: str,
        trade_request: Dict[str, Any],
        user_balance: float,
        user_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a trade request against risk rules.
        Returns approval status and any adjustments.
        """
        
        # Check if user is emergency stopped
        if user_id in self.emergency_stopped_users:
            return {
                'approved': False,
                'reason': 'emergency_stop_active',
                'message': 'Trading is halted. Emergency stop is active.',
            }
            
        # Check if user is in cooldown
        if user_id in self.user_cooldowns:
            cooldown_until = self.user_cooldowns[user_id]
            if datetime.utcnow() < cooldown_until:
                remaining = (cooldown_until - datetime.utcnow()).seconds // 60
                return {
                    'approved': False,
                    'reason': 'cooldown_active',
                    'message': f'Cooldown active. Resume in {remaining} minutes.',
                    'cooldown_until': cooldown_until.isoformat(),
                }
            else:
                del self.user_cooldowns[user_id]
                
        # Get risk settings
        max_loss_per_trade = user_settings.get('max_loss_per_trade', 2.0)
        max_loss_per_day = user_settings.get('max_loss_per_day', 5.0)
        max_exposure = user_settings.get('max_exposure_percent', 30.0)
        max_positions = user_settings.get('max_positions', 5)
        
        # Calculate trade risk
        trade_size = trade_request.get('quantity', 0) * trade_request.get('price', 0)
        trade_risk_percent = (trade_size / user_balance) * 100 if user_balance > 0 else 100
        
        # Check individual trade risk
        if trade_risk_percent > max_loss_per_trade * 2:  # Position size check
            return {
                'approved': False,
                'reason': 'position_too_large',
                'message': f'Position size {trade_risk_percent:.1f}% exceeds limit.',
                'suggested_size': (max_loss_per_trade * 2 / 100) * user_balance / trade_request.get('price', 1),
            }
            
        # Get today's P&L
        today_pnl = await self._get_today_pnl(user_id)
        today_loss_percent = abs(min(0, today_pnl)) / user_balance * 100 if user_balance > 0 else 0
        
        # Check daily loss limit
        if today_loss_percent >= max_loss_per_day:
            await self._trigger_cooldown(user_id, user_settings)
            return {
                'approved': False,
                'reason': 'daily_loss_limit',
                'message': f'Daily loss limit reached ({today_loss_percent:.1f}%).',
            }
            
        # Warning if approaching limit
        if today_loss_percent >= max_loss_per_day * 0.7:
            await self._add_alert(user_id, RiskEventType.DAILY_LOSS_WARNING, {
                'current_loss': today_loss_percent,
                'limit': max_loss_per_day,
            })
            
        # Check current exposure
        current_exposure = await self._get_current_exposure(user_id)
        
        if current_exposure + trade_risk_percent > max_exposure:
            return {
                'approved': False,
                'reason': 'exposure_limit',
                'message': f'Would exceed exposure limit ({current_exposure + trade_risk_percent:.1f}% > {max_exposure}%).',
                'current_exposure': current_exposure,
            }
            
        # Check position count
        current_positions = await self._get_position_count(user_id)
        if current_positions >= max_positions:
            return {
                'approved': False,
                'reason': 'max_positions',
                'message': f'Maximum positions reached ({current_positions}/{max_positions}).',
            }
            
        # Calculate optimal stop loss
        suggested_stop_loss = self._calculate_stop_loss(
            trade_request,
            user_balance,
            max_loss_per_trade
        )
        
        return {
            'approved': True,
            'risk_score': self._calculate_risk_score(trade_request, user_balance, today_loss_percent),
            'suggested_stop_loss': suggested_stop_loss,
            'max_position_size': (max_exposure - current_exposure) / 100 * user_balance,
            'remaining_daily_risk': max_loss_per_day - today_loss_percent,
        }
        
    async def post_trade_analysis(
        self,
        user_id: str,
        trade_result: Dict[str, Any],
        user_balance: float,
        user_settings: Dict[str, Any]
    ):
        """
        Analyze completed trade and update risk state.
        Trigger cooldown or alerts if needed.
        """
        
        pnl = trade_result.get('pnl', 0)
        pnl_percent = (pnl / user_balance) * 100 if user_balance > 0 else 0
        
        # Record trade result
        await self.redis_client.lpush(
            f"risk:trades:{user_id}",
            json.dumps({
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'timestamp': datetime.utcnow().isoformat(),
            })
        )
        await self.redis_client.ltrim(f"risk:trades:{user_id}", 0, 99)
        
        # Check if single trade exceeded limit
        max_loss_per_trade = user_settings.get('max_loss_per_trade', 2.0)
        if abs(pnl_percent) > max_loss_per_trade and pnl < 0:
            await self._add_alert(user_id, RiskEventType.TRADE_LOSS_LIMIT, {
                'trade_loss': abs(pnl_percent),
                'limit': max_loss_per_trade,
            })
            
        # Check consecutive losses
        consecutive_losses = await self._check_consecutive_losses(user_id)
        if consecutive_losses >= 3:
            await self._trigger_cooldown(user_id, user_settings)
            await self._add_alert(user_id, RiskEventType.COOLDOWN_STARTED, {
                'reason': 'consecutive_losses',
                'count': consecutive_losses,
            })
            
    async def check_market_conditions(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check market conditions for systemic risk.
        May trigger global caution mode.
        """
        
        warnings = []
        
        for symbol, data in market_data.items():
            indicators = data.get('indicators', {})
            volatility = float(indicators.get('volatility_percent', 0))
            
            # Extreme volatility warning
            if volatility > 5.0:
                warnings.append({
                    'symbol': symbol,
                    'type': 'extreme_volatility',
                    'value': volatility,
                    'recommendation': 'reduce_exposure',
                })
                
        return {
            'market_safe': len(warnings) == 0,
            'warnings': warnings,
            'recommendation': 'hold' if warnings else 'normal',
        }
        
    async def emergency_stop(self, user_id: str, reason: str):
        """
        Activate emergency stop for a user.
        Halts all trading immediately.
        """
        
        self.emergency_stopped_users.add(user_id)
        
        await self._add_alert(user_id, RiskEventType.EMERGENCY_STOP, {
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat(),
        })
        
        # Store in Redis for persistence
        await self.redis_client.set(f"risk:emergency_stop:{user_id}", "1")
        
        logger.warning(f"Emergency stop activated for user {user_id}: {reason}")
        
    async def resume_trading(self, user_id: str):
        """Resume trading after emergency stop"""
        
        self.emergency_stopped_users.discard(user_id)
        await self.redis_client.delete(f"risk:emergency_stop:{user_id}")
        
        logger.info(f"Trading resumed for user {user_id}")
        
    async def get_risk_status(self, user_id: str, user_balance: float, user_settings: Dict) -> Dict[str, Any]:
        """Get comprehensive risk status for a user"""
        
        # Check emergency stop
        if user_id in self.emergency_stopped_users:
            return {
                'status': RiskStatus.STOPPED.value,
                'message': 'Emergency stop active',
                'can_trade': False,
            }
            
        # Check cooldown
        if user_id in self.user_cooldowns:
            cooldown_until = self.user_cooldowns[user_id]
            if datetime.utcnow() < cooldown_until:
                return {
                    'status': RiskStatus.STOPPED.value,
                    'message': 'Cooldown active',
                    'can_trade': False,
                    'cooldown_until': cooldown_until.isoformat(),
                }
                
        today_pnl = await self._get_today_pnl(user_id)
        today_loss_percent = abs(min(0, today_pnl)) / user_balance * 100 if user_balance > 0 else 0
        max_loss_per_day = user_settings.get('max_loss_per_day', 5.0)
        
        current_exposure = await self._get_current_exposure(user_id)
        max_exposure = user_settings.get('max_exposure_percent', 30.0)
        
        # Determine status
        if today_loss_percent >= max_loss_per_day:
            status = RiskStatus.STOPPED
        elif today_loss_percent >= max_loss_per_day * 0.8:
            status = RiskStatus.CRITICAL
        elif today_loss_percent >= max_loss_per_day * 0.6:
            status = RiskStatus.WARNING
        elif today_loss_percent >= max_loss_per_day * 0.4 or current_exposure >= max_exposure * 0.8:
            status = RiskStatus.CAUTION
        else:
            status = RiskStatus.SAFE
            
        return {
            'status': status.value,
            'can_trade': status not in [RiskStatus.STOPPED, RiskStatus.CRITICAL],
            'today_loss_percent': round(today_loss_percent, 2),
            'max_loss_percent': max_loss_per_day,
            'current_exposure': round(current_exposure, 2),
            'max_exposure': max_exposure,
            'remaining_risk_budget': round(max_loss_per_day - today_loss_percent, 2),
            'alerts_count': len(self.active_alerts.get(user_id, [])),
        }
        
    async def get_active_alerts_count(self) -> int:
        """Get total active alerts count"""
        return sum(len(alerts) for alerts in self.active_alerts.values())
        
    # Private helper methods
    
    async def _get_today_pnl(self, user_id: str) -> float:
        """Get user's P&L for today"""
        today = datetime.utcnow().date().isoformat()
        pnl = await self.redis_client.get(f"risk:daily_pnl:{user_id}:{today}")
        return float(pnl) if pnl else 0.0
        
    async def _get_current_exposure(self, user_id: str) -> float:
        """Get user's current position exposure"""
        exposure = await self.redis_client.get(f"risk:exposure:{user_id}")
        return float(exposure) if exposure else 0.0
        
    async def _get_position_count(self, user_id: str) -> int:
        """Get user's current position count"""
        count = await self.redis_client.get(f"risk:position_count:{user_id}")
        return int(count) if count else 0
        
    async def _check_consecutive_losses(self, user_id: str) -> int:
        """Check consecutive losing trades"""
        trades = await self.redis_client.lrange(f"risk:trades:{user_id}", 0, 9)
        
        consecutive = 0
        for trade in trades:
            data = json.loads(trade)
            if data.get('pnl', 0) < 0:
                consecutive += 1
            else:
                break
                
        return consecutive
        
    async def _trigger_cooldown(self, user_id: str, user_settings: Dict):
        """Trigger trading cooldown for user"""
        cooldown_minutes = user_settings.get('cooldown_after_loss_minutes', 30)
        cooldown_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
        self.user_cooldowns[user_id] = cooldown_until
        
        await self.redis_client.setex(
            f"risk:cooldown:{user_id}",
            cooldown_minutes * 60,
            cooldown_until.isoformat()
        )
        
    async def _add_alert(self, user_id: str, event_type: RiskEventType, data: Dict):
        """Add risk alert"""
        alert = {
            'type': event_type.value,
            'data': data,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        if user_id not in self.active_alerts:
            self.active_alerts[user_id] = []
        self.active_alerts[user_id].append(alert)
        
        # Keep only last 20 alerts per user
        self.active_alerts[user_id] = self.active_alerts[user_id][-20:]
        
        # Store in Redis
        await self.redis_client.lpush(f"risk:alerts:{user_id}", json.dumps(alert))
        await self.redis_client.ltrim(f"risk:alerts:{user_id}", 0, 49)
        
    def _calculate_stop_loss(
        self,
        trade_request: Dict,
        user_balance: float,
        max_loss_percent: float
    ) -> float:
        """Calculate optimal stop loss price"""
        
        price = trade_request.get('price', 0)
        quantity = trade_request.get('quantity', 0)
        side = trade_request.get('side', 'buy')
        
        max_loss_usd = user_balance * (max_loss_percent / 100)
        
        if quantity > 0:
            max_price_drop = max_loss_usd / quantity
            
            if side == 'buy':
                return price - max_price_drop
            else:
                return price + max_price_drop
                
        return price * 0.98  # Default 2% stop loss
        
    def _calculate_risk_score(
        self,
        trade_request: Dict,
        user_balance: float,
        current_daily_loss: float
    ) -> float:
        """Calculate risk score for a trade (0-100)"""
        
        trade_size = trade_request.get('quantity', 0) * trade_request.get('price', 0)
        position_risk = (trade_size / user_balance) * 100 if user_balance > 0 else 100
        
        # Higher score = higher risk
        score = position_risk * 2 + current_daily_loss * 3
        
        return min(100, max(0, score))

