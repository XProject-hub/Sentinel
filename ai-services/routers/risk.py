"""Risk Management API Routes"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

router = APIRouter()


class RiskCheckRequest(BaseModel):
    user_id: str
    trade_request: Dict
    user_balance: float
    user_settings: Dict


class EmergencyStopRequest(BaseModel):
    user_id: str
    reason: str


@router.post("/evaluate")
async def evaluate_trade_risk(request: RiskCheckRequest):
    """Evaluate risk for a proposed trade"""
    from main import risk_engine
    
    result = await risk_engine.evaluate_trade(
        request.user_id,
        request.trade_request,
        request.user_balance,
        request.user_settings
    )
    
    return {"success": True, "evaluation": result}


@router.get("/status/{user_id}")
async def get_risk_status(user_id: str, balance: float = 10000, max_loss: float = 5.0):
    """Get current risk status for a user"""
    from main import risk_engine
    
    status = await risk_engine.get_risk_status(
        user_id,
        balance,
        {"max_loss_per_day": max_loss}
    )
    
    return {"success": True, "status": status}


@router.post("/emergency-stop")
async def trigger_emergency_stop(request: EmergencyStopRequest):
    """Trigger emergency stop for a user"""
    from main import risk_engine
    
    await risk_engine.emergency_stop(request.user_id, request.reason)
    return {"success": True, "message": "Emergency stop activated"}


@router.post("/resume/{user_id}")
async def resume_trading(user_id: str):
    """Resume trading after emergency stop"""
    from main import risk_engine
    
    await risk_engine.resume_trading(user_id)
    return {"success": True, "message": "Trading resumed"}


@router.get("/market-check")
async def check_market_conditions():
    """Check current market conditions for systemic risk"""
    from main import risk_engine, market_intelligence
    
    market_data = await market_intelligence.get_current_state()
    conditions = await risk_engine.check_market_conditions(market_data)
    
    return {"success": True, "conditions": conditions}

