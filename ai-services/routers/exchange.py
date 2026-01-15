"""
Exchange Connection API Routes
Real Bybit V5 API integration
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import redis.asyncio as redis
import json
from loguru import logger

from services.bybit_client import BybitV5Client
from config import settings

router = APIRouter()

# Store for active exchange connections
exchange_connections = {}


class ExchangeCredentials(BaseModel):
    exchange: str
    apiKey: str
    apiSecret: str
    testnet: bool = False


class TestConnectionRequest(BaseModel):
    exchange: str
    apiKey: str
    apiSecret: str


@router.post("/test")
async def test_exchange_connection(request: TestConnectionRequest):
    """Test exchange API connection"""
    
    if request.exchange.lower() != "bybit":
        return {"success": False, "error": "Only Bybit is currently supported"}
        
    try:
        client = BybitV5Client(
            api_key=request.apiKey,
            api_secret=request.apiSecret,
            testnet=False
        )
        
        result = await client.test_connection()
        await client.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/connect")
async def connect_exchange(credentials: ExchangeCredentials):
    """Connect and store exchange credentials"""
    
    if credentials.exchange.lower() != "bybit":
        return {"success": False, "error": "Only Bybit is currently supported"}
        
    try:
        # Test connection first
        client = BybitV5Client(
            api_key=credentials.apiKey,
            api_secret=credentials.apiSecret,
            testnet=credentials.testnet
        )
        
        result = await client.test_connection()
        
        if not result.get("success"):
            await client.close()
            return result
            
        # Store connection (in production, encrypt and store in database)
        # For now, store in memory
        exchange_connections["default"] = client
        
        return {
            "success": True,
            "message": "Exchange connected successfully"
        }
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return {"success": False, "error": str(e)}


@router.get("/balance")
async def get_balance():
    """Get real wallet balance from connected exchange"""
    
    if "default" not in exchange_connections:
        return {"success": False, "error": "No exchange connected"}
        
    client = exchange_connections["default"]
    result = await client.get_wallet_balance()
    
    if not result.get("success"):
        return result
        
    # Parse balance data
    data = result.get("data", {})
    coins = []
    total_equity = 0
    
    for account in data.get("list", []):
        total_equity = float(account.get("totalEquity", 0))
        for coin in account.get("coin", []):
            if float(coin.get("walletBalance", 0)) > 0:
                coins.append({
                    "coin": coin.get("coin"),
                    "balance": float(coin.get("walletBalance", 0)),
                    "equity": float(coin.get("equity", 0)),
                    "usdValue": float(coin.get("usdValue", 0)),
                    "unrealizedPnl": float(coin.get("unrealisedPnl", 0))
                })
                
    return {
        "success": True,
        "data": {
            "totalEquity": total_equity,
            "coins": coins
        }
    }


@router.get("/positions")
async def get_positions():
    """Get real open positions from connected exchange"""
    
    if "default" not in exchange_connections:
        return {"success": False, "error": "No exchange connected"}
        
    client = exchange_connections["default"]
    result = await client.get_positions()
    
    if not result.get("success"):
        return result
        
    # Parse positions
    positions = []
    data = result.get("data", {})
    
    for pos in data.get("list", []):
        size = float(pos.get("size", 0))
        if size > 0:
            positions.append({
                "symbol": pos.get("symbol"),
                "side": pos.get("side"),
                "size": size,
                "entryPrice": float(pos.get("avgPrice", 0)),
                "markPrice": float(pos.get("markPrice", 0)),
                "unrealizedPnl": float(pos.get("unrealisedPnl", 0)),
                "leverage": pos.get("leverage"),
                "liquidationPrice": float(pos.get("liqPrice", 0)) if pos.get("liqPrice") else None,
                "takeProfit": pos.get("takeProfit"),
                "stopLoss": pos.get("stopLoss"),
            })
            
    return {
        "success": True,
        "data": {"positions": positions}
    }


@router.get("/pnl")
async def get_pnl_history():
    """Get real closed PnL history"""
    
    if "default" not in exchange_connections:
        return {"success": False, "error": "No exchange connected"}
        
    client = exchange_connections["default"]
    result = await client.get_pnl()
    
    if not result.get("success"):
        return result
        
    # Parse PnL history
    pnl_list = []
    data = result.get("data", {})
    
    for pnl in data.get("list", []):
        pnl_list.append({
            "symbol": pnl.get("symbol"),
            "side": pnl.get("side"),
            "qty": float(pnl.get("qty", 0)),
            "entryPrice": float(pnl.get("avgEntryPrice", 0)),
            "exitPrice": float(pnl.get("avgExitPrice", 0)),
            "closedPnl": float(pnl.get("closedPnl", 0)),
            "createdTime": pnl.get("createdTime"),
            "updatedTime": pnl.get("updatedTime"),
        })
        
    # Calculate totals
    total_pnl = sum(p["closedPnl"] for p in pnl_list)
    winning = len([p for p in pnl_list if p["closedPnl"] > 0])
    losing = len([p for p in pnl_list if p["closedPnl"] < 0])
    
    return {
        "success": True,
        "data": {
            "trades": pnl_list,
            "totalPnl": total_pnl,
            "winningTrades": winning,
            "losingTrades": losing,
            "winRate": (winning / len(pnl_list) * 100) if pnl_list else 0
        }
    }


@router.get("/ticker/{symbol}")
async def get_ticker(symbol: str):
    """Get real-time ticker for a symbol"""
    
    if "default" not in exchange_connections:
        # Use public endpoint without auth
        client = BybitV5Client("", "", testnet=False)
    else:
        client = exchange_connections["default"]
        
    result = await client.get_tickers(symbol=symbol)
    
    if not result.get("success"):
        return result
        
    data = result.get("data", {})
    tickers = data.get("list", [])
    
    if not tickers:
        return {"success": False, "error": "Symbol not found"}
        
    ticker = tickers[0]
    
    return {
        "success": True,
        "data": {
            "symbol": ticker.get("symbol"),
            "lastPrice": float(ticker.get("lastPrice", 0)),
            "price24hPcnt": float(ticker.get("price24hPcnt", 0)) * 100,
            "highPrice24h": float(ticker.get("highPrice24h", 0)),
            "lowPrice24h": float(ticker.get("lowPrice24h", 0)),
            "volume24h": float(ticker.get("volume24h", 0)),
            "turnover24h": float(ticker.get("turnover24h", 0)),
            "bid": float(ticker.get("bid1Price", 0)),
            "ask": float(ticker.get("ask1Price", 0)),
            "fundingRate": float(ticker.get("fundingRate", 0)) * 100,
            "openInterest": float(ticker.get("openInterest", 0)),
        }
    }


@router.get("/orderbook/{symbol}")
async def get_orderbook(symbol: str, limit: int = 25):
    """Get real-time orderbook"""
    
    client = BybitV5Client("", "", testnet=False)
    result = await client.get_orderbook(symbol=symbol, limit=limit)
    await client.close()
    
    if not result.get("success"):
        return result
        
    data = result.get("data", {})
    
    return {
        "success": True,
        "data": {
            "symbol": symbol,
            "bids": [[float(b[0]), float(b[1])] for b in data.get("b", [])],
            "asks": [[float(a[0]), float(a[1])] for a in data.get("a", [])],
            "timestamp": data.get("ts")
        }
    }


@router.get("/status")
async def get_connection_status():
    """Check if exchange is connected"""
    
    connected = "default" in exchange_connections
    
    return {
        "connected": connected,
        "exchange": "bybit" if connected else None,
        "serverIp": "109.104.154.183"
    }

