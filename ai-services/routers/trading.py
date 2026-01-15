"""Trading Execution API Routes"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class ConnectExchangeRequest(BaseModel):
    user_id: str
    exchange: str
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None


class ExecuteTradeRequest(BaseModel):
    user_id: str
    exchange: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    ai_confidence: Optional[float] = None
    ai_reasoning: Optional[str] = None


class ClosePositionRequest(BaseModel):
    user_id: str
    exchange: str
    symbol: str
    quantity: Optional[float] = None


@router.post("/connect")
async def connect_exchange(request: ConnectExchangeRequest):
    """Connect user's exchange API"""
    from main import trading_executor
    
    success = await trading_executor.connect_exchange(
        request.user_id,
        request.exchange,
        request.api_key,
        request.api_secret,
        request.passphrase
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to connect exchange")
        
    return {"success": True, "message": f"Connected to {request.exchange}"}


@router.post("/execute")
async def execute_trade(request: ExecuteTradeRequest):
    """Execute a trade"""
    from main import trading_executor
    
    result = await trading_executor.execute_trade(
        request.user_id,
        request.exchange,
        request.symbol,
        request.side,
        request.order_type,
        request.quantity,
        request.price,
        request.stop_loss,
        request.take_profit,
        request.ai_confidence,
        request.ai_reasoning
    )
    
    return result


@router.post("/close")
async def close_position(request: ClosePositionRequest):
    """Close a position"""
    from main import trading_executor
    
    result = await trading_executor.close_position(
        request.user_id,
        request.exchange,
        request.symbol,
        request.quantity
    )
    
    return result


@router.post("/close-all/{user_id}/{exchange}")
async def close_all_positions(user_id: str, exchange: str):
    """Emergency: Close all positions"""
    from main import trading_executor
    
    result = await trading_executor.close_all_positions(user_id, exchange)
    return result


@router.get("/balance/{user_id}/{exchange}")
async def get_balance(user_id: str, exchange: str):
    """Get user's exchange balance"""
    from main import trading_executor
    
    result = await trading_executor.get_balance(user_id, exchange)
    return result


@router.get("/positions/{user_id}/{exchange}")
async def get_positions(user_id: str, exchange: str):
    """Get user's open positions"""
    from main import trading_executor
    
    result = await trading_executor.get_positions(user_id, exchange)
    return result


@router.post("/sync/{user_id}/{exchange}")
async def sync_user_data(user_id: str, exchange: str):
    """Sync all user data from exchange"""
    from main import trading_executor
    
    result = await trading_executor.sync_user_data(user_id, exchange)
    return result

