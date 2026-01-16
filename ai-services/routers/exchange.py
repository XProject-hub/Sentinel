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
    """Get real wallet balance from connected exchange - checks ALL account types"""
    
    if "default" not in exchange_connections:
        return {"success": False, "error": "No exchange connected"}
        
    client = exchange_connections["default"]
    
    coins = []
    total_equity = 0
    account_type_found = None
    
    # Try UNIFIED account first (derivatives + spot combined)
    result = await client.get_wallet_balance(account_type="UNIFIED")
    if result.get("success"):
        data = result.get("data", {})
        for account in data.get("list", []):
            eq = float(account.get("totalEquity", 0))
            if eq > 0:
                total_equity = eq
                account_type_found = "UNIFIED"
                for coin in account.get("coin", []):
                    if float(coin.get("walletBalance", 0)) > 0:
                        coins.append({
                            "coin": coin.get("coin"),
                            "balance": float(coin.get("walletBalance", 0)),
                            "equity": float(coin.get("equity", 0)),
                            "usdValue": float(coin.get("usdValue", 0)),
                            "unrealizedPnl": float(coin.get("unrealisedPnl", 0))
                        })
    
    # Try SPOT account if UNIFIED is empty
    if total_equity == 0:
        result = await client.get_wallet_balance(account_type="SPOT")
        if result.get("success"):
            data = result.get("data", {})
            for account in data.get("list", []):
                for coin in account.get("coin", []):
                    balance = float(coin.get("walletBalance", 0))
                    if balance > 0:
                        usd_value = float(coin.get("usdValue", 0))
                        total_equity += usd_value
                        account_type_found = "SPOT"
                        coins.append({
                            "coin": coin.get("coin"),
                            "balance": balance,
                            "equity": balance,
                            "usdValue": usd_value,
                            "unrealizedPnl": 0
                        })
    
    # Try CONTRACT account
    if total_equity == 0:
        result = await client.get_wallet_balance(account_type="CONTRACT")
        if result.get("success"):
            data = result.get("data", {})
            for account in data.get("list", []):
                eq = float(account.get("totalEquity", 0))
                if eq > 0:
                    total_equity = eq
                    account_type_found = "CONTRACT"
                    for coin in account.get("coin", []):
                        if float(coin.get("walletBalance", 0)) > 0:
                            coins.append({
                                "coin": coin.get("coin"),
                                "balance": float(coin.get("walletBalance", 0)),
                                "equity": float(coin.get("equity", 0)),
                                "usdValue": float(coin.get("usdValue", 0)),
                                "unrealizedPnl": float(coin.get("unrealisedPnl", 0))
                            })

    # Try FUND account (for holdings)
    if total_equity == 0:
        result = await client.get_wallet_balance(account_type="FUND")
        if result.get("success"):
            data = result.get("data", {})
            for account in data.get("list", []):
                for coin in account.get("coin", []):
                    balance = float(coin.get("walletBalance", 0))
                    if balance > 0:
                        usd_value = float(coin.get("usdValue", 0))
                        total_equity += usd_value
                        account_type_found = "FUND"
                        coins.append({
                            "coin": coin.get("coin"),
                            "balance": balance,
                            "equity": balance,
                            "usdValue": usd_value,
                            "unrealizedPnl": 0
                        })
    
    logger.info(f"Balance fetched: €{total_equity:.2f} from {account_type_found} account, {len(coins)} coins")
                
    return {
        "success": True,
        "data": {
            "totalEquity": total_equity,
            "coins": coins,
            "accountType": account_type_found
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


# ========================================
# AUTONOMOUS TRADING ENDPOINTS
# ========================================

from services.autonomous_trader import autonomous_trader


class EnableTradingRequest(BaseModel):
    user_id: str = "default"
    api_key: str
    api_secret: str


@router.post("/trading/enable")
async def enable_autonomous_trading(request: EnableTradingRequest):
    """Enable 24/7 autonomous trading for a user"""
    
    success = await autonomous_trader.connect_user(
        user_id=request.user_id,
        api_key=request.api_key,
        api_secret=request.api_secret,
    )
    
    if success:
        return {
            "success": True,
            "message": "Autonomous trading enabled. AI will trade 24/7 using your funds.",
            "warning": "REAL MONEY WILL BE USED FOR TRADING"
        }
    else:
        return {
            "success": False,
            "error": "Failed to enable autonomous trading. Check API credentials."
        }


@router.post("/trading/disable")
async def disable_autonomous_trading(user_id: str = "default"):
    """Disable autonomous trading for a user"""
    
    await autonomous_trader.disconnect_user(user_id)
    
    return {
        "success": True,
        "message": "Autonomous trading disabled"
    }


@router.get("/trading/status")
async def get_trading_status(user_id: str = "default"):
    """Get autonomous trading status"""
    
    is_trading = user_id in autonomous_trader.user_clients
    
    # Get recent trades
    trades = []
    if autonomous_trader.redis_client:
        trade_data = await autonomous_trader.redis_client.lrange(f"trades:completed:{user_id}", 0, 9)
        trades = [json.loads(t) for t in trade_data]
        
    return {
        "success": True,
        "data": {
            "is_autonomous_trading": is_trading,
            "trading_pairs": autonomous_trader.trading_pairs if is_trading else [],
            "max_positions": autonomous_trader.max_open_positions,
            "min_confidence": autonomous_trader.min_confidence,
            "recent_trades": trades,
        }
    }


@router.get("/trading/log")
async def get_trading_log(limit: int = 50):
    """Get trading activity log"""
    
    if not autonomous_trader.redis_client:
        return {"success": False, "error": "Not initialized"}
        
    trades = await autonomous_trader.redis_client.lrange('trades:log', 0, limit - 1)
    
    return {
        "success": True,
        "data": [json.loads(t) for t in trades]
    }


@router.get("/trading/activity")
async def get_live_activity(user_id: str = "default"):
    """Get LIVE bot activity - what is the bot doing right now?"""
    
    activity = {
        "is_running": autonomous_trader.is_running,
        "is_user_connected": user_id in autonomous_trader.user_clients,
        "total_pairs_monitoring": len(autonomous_trader.trading_pairs),
        "active_trades": [],
        "recent_completed": [],
        "bot_actions": [],
        "current_analysis": None,
    }
    
    if not autonomous_trader.redis_client:
        return {"success": True, "data": activity}
    
    try:
        # Get active trades for this user
        active_trades = await autonomous_trader.redis_client.lrange(f"trades:active:{user_id}", 0, -1)
        activity["active_trades"] = [json.loads(t) for t in active_trades]
        
        # Get recent completed trades
        completed = await autonomous_trader.redis_client.lrange(f"trades:completed:{user_id}", 0, 9)
        activity["recent_completed"] = [json.loads(t) for t in completed]
        
        # Get last bot actions/decisions
        bot_log = await autonomous_trader.redis_client.lrange('trades:log', 0, 19)
        activity["bot_actions"] = [json.loads(t) for t in bot_log]
        
        # Get current market analysis if available
        analysis = await autonomous_trader.redis_client.hgetall('market:current_analysis')
        if analysis:
            activity["current_analysis"] = {
                k.decode(): v.decode() for k, v in analysis.items()
            }
            
        # Get trading stats
        trading_status = await autonomous_trader.redis_client.hgetall('trading:status')
        if trading_status:
            activity["trading_stats"] = {
                k.decode(): v.decode() for k, v in trading_status.items()
            }
            
    except Exception as e:
        logger.error(f"Error getting activity: {e}")
        
    return {"success": True, "data": activity}


@router.get("/trading/pairs")
async def get_trading_pairs():
    """Get all crypto pairs the bot monitors"""
    
    return {
        "success": True,
        "data": {
            "pairs": autonomous_trader.trading_pairs,
            "total": len(autonomous_trader.trading_pairs),
            "categories": {
                "top10": autonomous_trader.trading_pairs[:10],
                "defi": [p for p in autonomous_trader.trading_pairs if p in ['COMPUSDT', 'SNXUSDT', 'CRVUSDT', 'YFIUSDT', 'SUSHIUSDT', '1INCHUSDT', 'DYDXUSDT', 'GMXUSDT', 'PENDLEUSDT', 'ENSUSDT']],
                "meme": [p for p in autonomous_trader.trading_pairs if p in ['SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT']],
                "ai": [p for p in autonomous_trader.trading_pairs if p in ['TAOUSDT', 'WLDUSDT', 'OCEANUSDT', 'RNDRAUSDT', 'AKTUSDT']],
            }
        }
    }

