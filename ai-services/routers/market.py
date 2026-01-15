"""Market Intelligence API Routes"""

from fastapi import APIRouter, HTTPException
from typing import Optional

router = APIRouter()


@router.get("/data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    from main import market_intelligence
    
    data = await market_intelligence.get_symbol_state(symbol)
    if not data:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return {"success": True, "data": data}


@router.get("/indicators/{symbol}")
async def get_indicators(symbol: str):
    """Get technical indicators for a symbol"""
    from main import market_intelligence
    
    state = await market_intelligence.get_symbol_state(symbol)
    indicators = state.get("indicators", {})
    return {"success": True, "indicators": indicators}


@router.get("/orderbook/{symbol}")
async def get_orderbook(symbol: str):
    """Get order book data for a symbol"""
    from main import market_intelligence
    
    state = await market_intelligence.get_symbol_state(symbol)
    orderbook = state.get("orderbook", {})
    return {"success": True, "orderbook": orderbook}


@router.get("/overview")
async def get_market_overview():
    """Get overview of all tracked markets"""
    from main import market_intelligence
    
    state = await market_intelligence.get_current_state()
    return {"success": True, "markets": state}

