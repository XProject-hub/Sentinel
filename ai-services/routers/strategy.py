"""Strategy Planning API Routes"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class StrategyRequest(BaseModel):
    user_id: str
    symbol: str
    risk_tolerance: str = "medium"


@router.get("/regime/{symbol}")
async def get_market_regime(symbol: str):
    """Get current market regime for a symbol"""
    from main import market_intelligence, sentiment_analyzer, strategy_planner
    
    market_state = await market_intelligence.get_symbol_state(symbol)
    sentiment = await sentiment_analyzer.get_asset_sentiment(symbol.replace("USDT", ""))
    
    regime = await strategy_planner.detect_regime(
        {symbol: market_state}, 
        {symbol.replace("USDT", ""): sentiment}
    )
    
    return {"success": True, "regime": regime}


@router.post("/recommend")
async def get_strategy_recommendation(request: StrategyRequest):
    """Get strategy recommendation for a user/symbol"""
    from main import strategy_planner
    
    recommendation = await strategy_planner.get_recommended_strategy(
        request.user_id,
        request.symbol,
        request.risk_tolerance
    )
    
    return {"success": True, "recommendation": recommendation}


@router.get("/available")
async def get_available_strategies():
    """Get list of available AI strategies"""
    return {
        "success": True,
        "strategies": [
            {
                "id": "momentum",
                "name": "Momentum Surge",
                "description": "Captures strong directional moves with tight risk management",
                "risk_level": "medium",
                "best_for": ["trending", "bull_market"]
            },
            {
                "id": "grid",
                "name": "Grid Master",
                "description": "Range-bound trading with dynamic grid placement",
                "risk_level": "low",
                "best_for": ["sideways", "low_volatility"]
            },
            {
                "id": "breakout",
                "name": "Breakout Hunter",
                "description": "Identifies and trades key level breakouts",
                "risk_level": "high",
                "best_for": ["breakout", "high_volatility"]
            },
            {
                "id": "mean_reversion",
                "name": "Mean Reversion",
                "description": "Profits from price returning to statistical means",
                "risk_level": "medium",
                "best_for": ["sideways", "ranging"]
            },
            {
                "id": "scalping",
                "name": "Scalp Pro",
                "description": "High-frequency small profit captures",
                "risk_level": "high",
                "best_for": ["high_liquidity", "volatile"]
            }
        ]
    }

