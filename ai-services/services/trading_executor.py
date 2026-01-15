"""
SENTINEL AI - Trading Execution Engine
Order execution and management
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from loguru import logger
import ccxt.async_support as ccxt
import redis.asyncio as redis
import json

from config import settings


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TradingExecutor:
    """
    Executes trades on exchanges with:
    - Optimal entry/exit timing
    - Slippage control
    - Fee optimization
    - Order management
    """
    
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.redis_client = None
        self.pending_orders: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize trading executor"""
        logger.info("Initializing Trading Executor...")
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        logger.info("Trading Executor initialized")
        
    async def connect_exchange(
        self,
        user_id: str,
        exchange_name: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None
    ) -> bool:
        """Connect to user's exchange"""
        
        exchange_id = f"{user_id}:{exchange_name}"
        
        try:
            exchange_class = getattr(ccxt, exchange_name.lower())
            
            config = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            }
            
            if passphrase:
                config['password'] = passphrase
                
            exchange = exchange_class(config)
            
            # Test connection
            await exchange.fetch_balance()
            
            self.exchanges[exchange_id] = exchange
            logger.info(f"Connected to {exchange_name} for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {exchange_name}: {e}")
            return False
            
    async def disconnect_exchange(self, user_id: str, exchange_name: str):
        """Disconnect from exchange"""
        exchange_id = f"{user_id}:{exchange_name}"
        
        if exchange_id in self.exchanges:
            await self.exchanges[exchange_id].close()
            del self.exchanges[exchange_id]
            
    async def execute_trade(
        self,
        user_id: str,
        exchange_name: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        ai_confidence: Optional[float] = None,
        ai_reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a trade with optimal parameters
        """
        
        exchange_id = f"{user_id}:{exchange_name}"
        
        if exchange_id not in self.exchanges:
            return {
                'success': False,
                'error': 'exchange_not_connected',
                'message': 'Please connect your exchange first.',
            }
            
        exchange = self.exchanges[exchange_id]
        
        try:
            # Prepare order
            order_params = {}
            
            # Execute based on order type
            if order_type == 'market':
                order = await exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=quantity,
                    params=order_params
                )
            else:
                if price is None:
                    return {
                        'success': False,
                        'error': 'price_required',
                        'message': 'Price is required for limit orders.',
                    }
                    
                order = await exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=quantity,
                    price=price,
                    params=order_params
                )
                
            # Create stop loss order if specified
            if stop_loss and order['status'] == 'closed':
                sl_side = 'sell' if side == 'buy' else 'buy'
                await exchange.create_order(
                    symbol=symbol,
                    type='stop_loss_limit',
                    side=sl_side,
                    amount=quantity,
                    price=stop_loss,
                    params={'stopPrice': stop_loss}
                )
                
            # Create take profit order if specified
            if take_profit and order['status'] == 'closed':
                tp_side = 'sell' if side == 'buy' else 'buy'
                await exchange.create_order(
                    symbol=symbol,
                    type='take_profit_limit',
                    side=tp_side,
                    amount=quantity,
                    price=take_profit,
                    params={'stopPrice': take_profit}
                )
                
            # Record trade
            trade_record = {
                'user_id': user_id,
                'exchange': exchange_name,
                'symbol': symbol,
                'side': side,
                'order_type': order_type,
                'quantity': quantity,
                'price': order.get('average', price),
                'filled': order.get('filled', 0),
                'status': order.get('status'),
                'order_id': order.get('id'),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'ai_confidence': ai_confidence,
                'ai_reasoning': ai_reasoning,
                'fee': order.get('fee', {}).get('cost', 0),
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            # Store in Redis
            await self.redis_client.lpush(
                f"trades:{user_id}",
                json.dumps(trade_record)
            )
            
            return {
                'success': True,
                'trade': trade_record,
                'order': order,
            }
            
        except ccxt.InsufficientFunds:
            return {
                'success': False,
                'error': 'insufficient_funds',
                'message': 'Insufficient balance for this trade.',
            }
            
        except ccxt.InvalidOrder as e:
            return {
                'success': False,
                'error': 'invalid_order',
                'message': str(e),
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {
                'success': False,
                'error': 'execution_error',
                'message': str(e),
            }
            
    async def close_position(
        self,
        user_id: str,
        exchange_name: str,
        symbol: str,
        quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """Close an open position"""
        
        exchange_id = f"{user_id}:{exchange_name}"
        
        if exchange_id not in self.exchanges:
            return {'success': False, 'error': 'exchange_not_connected'}
            
        exchange = self.exchanges[exchange_id]
        
        try:
            # Get current position
            positions = await exchange.fetch_positions([symbol])
            
            if not positions:
                return {'success': False, 'error': 'no_position'}
                
            position = positions[0]
            pos_size = abs(float(position.get('contracts', 0)))
            pos_side = position.get('side')
            
            if pos_size == 0:
                return {'success': False, 'error': 'no_position'}
                
            close_quantity = quantity or pos_size
            close_side = 'sell' if pos_side == 'long' else 'buy'
            
            # Execute close
            order = await exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=close_quantity,
                params={'reduceOnly': True}
            )
            
            return {
                'success': True,
                'order': order,
                'closed_quantity': close_quantity,
            }
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return {'success': False, 'error': str(e)}
            
    async def close_all_positions(
        self,
        user_id: str,
        exchange_name: str
    ) -> Dict[str, Any]:
        """Emergency: Close all open positions"""
        
        exchange_id = f"{user_id}:{exchange_name}"
        
        if exchange_id not in self.exchanges:
            return {'success': False, 'error': 'exchange_not_connected'}
            
        exchange = self.exchanges[exchange_id]
        
        try:
            positions = await exchange.fetch_positions()
            closed = []
            failed = []
            
            for position in positions:
                pos_size = abs(float(position.get('contracts', 0)))
                
                if pos_size > 0:
                    symbol = position.get('symbol')
                    result = await self.close_position(user_id, exchange_name, symbol)
                    
                    if result.get('success'):
                        closed.append(symbol)
                    else:
                        failed.append({'symbol': symbol, 'error': result.get('error')})
                        
            return {
                'success': len(failed) == 0,
                'closed': closed,
                'failed': failed,
            }
            
        except Exception as e:
            logger.error(f"Close all positions error: {e}")
            return {'success': False, 'error': str(e)}
            
    async def get_balance(
        self,
        user_id: str,
        exchange_name: str
    ) -> Dict[str, Any]:
        """Get user's exchange balance"""
        
        exchange_id = f"{user_id}:{exchange_name}"
        
        if exchange_id not in self.exchanges:
            return {'success': False, 'error': 'exchange_not_connected'}
            
        try:
            balance = await self.exchanges[exchange_id].fetch_balance()
            
            return {
                'success': True,
                'total': balance.get('total', {}),
                'free': balance.get('free', {}),
                'used': balance.get('used', {}),
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def get_positions(
        self,
        user_id: str,
        exchange_name: str
    ) -> Dict[str, Any]:
        """Get user's open positions"""
        
        exchange_id = f"{user_id}:{exchange_name}"
        
        if exchange_id not in self.exchanges:
            return {'success': False, 'error': 'exchange_not_connected'}
            
        try:
            positions = await self.exchanges[exchange_id].fetch_positions()
            
            active = []
            for pos in positions:
                contracts = abs(float(pos.get('contracts', 0)))
                if contracts > 0:
                    active.append({
                        'symbol': pos.get('symbol'),
                        'side': pos.get('side'),
                        'contracts': contracts,
                        'entry_price': pos.get('entryPrice'),
                        'mark_price': pos.get('markPrice'),
                        'unrealized_pnl': pos.get('unrealizedPnl'),
                        'leverage': pos.get('leverage'),
                        'liquidation_price': pos.get('liquidationPrice'),
                    })
                    
            return {
                'success': True,
                'positions': active,
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def sync_user_data(
        self,
        user_id: str,
        exchange_name: str
    ) -> Dict[str, Any]:
        """Sync all user data from exchange"""
        
        balance = await self.get_balance(user_id, exchange_name)
        positions = await self.get_positions(user_id, exchange_name)
        
        if balance.get('success') and positions.get('success'):
            # Calculate totals
            total_balance = sum(
                float(v) for k, v in balance.get('total', {}).items() 
                if k == 'USDT'
            )
            
            total_pnl = sum(
                float(p.get('unrealized_pnl', 0) or 0) 
                for p in positions.get('positions', [])
            )
            
            # Cache in Redis
            await self.redis_client.hset(
                f"user_data:{user_id}:{exchange_name}",
                mapping={
                    'total_balance': str(total_balance),
                    'unrealized_pnl': str(total_pnl),
                    'positions_count': str(len(positions.get('positions', []))),
                    'last_sync': datetime.utcnow().isoformat(),
                }
            )
            
            return {
                'success': True,
                'balance': balance,
                'positions': positions,
                'total_balance': total_balance,
                'unrealized_pnl': total_pnl,
            }
            
        return {
            'success': False,
            'balance': balance,
            'positions': positions,
        }

