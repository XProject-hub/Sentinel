"""
Exchange Connection API Routes
Real Bybit V5 API integration
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import redis.asyncio as redis
import json
import hashlib
import base64
from loguru import logger

from services.bybit_client import BybitV5Client
from config import settings

router = APIRouter()

# Store for active exchange connections
exchange_connections = {}

# Redis client for persistent storage
_redis_client = None

async def get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = await redis.from_url(settings.REDIS_URL)
    return _redis_client


def simple_encrypt(text: str, key: str = "sentinel_secret_key_2026") -> str:
    """Simple obfuscation for API credentials (use proper encryption in production)"""
    combined = text + key
    encoded = base64.b64encode(text.encode()).decode()
    return encoded


def simple_decrypt(encoded: str, key: str = "sentinel_secret_key_2026") -> str:
    """Simple de-obfuscation"""
    try:
        decoded = base64.b64decode(encoded.encode()).decode()
        return decoded
    except:
        return ""


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
    """Connect and store exchange credentials - PERSISTED to Redis"""
    
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
            
        # Store connection in memory
        exchange_connections["default"] = client
        
        # PERSIST credentials to Redis (encrypted)
        r = await get_redis()
        await r.hset("exchange:credentials:default", mapping={
            "exchange": credentials.exchange,
            "api_key": simple_encrypt(credentials.apiKey),
            "api_secret": simple_encrypt(credentials.apiSecret),
            "testnet": "1" if credentials.testnet else "0",
            "connected_at": str(json.dumps({"ts": "now"})),
        })
        
        logger.info("Exchange credentials saved to Redis - will persist across restarts")
        
        return {
            "success": True,
            "message": "Exchange connected successfully - credentials saved"
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
    all_results = {}  # For debugging
    
    # Try UNIFIED account first (derivatives + spot combined)
    logger.info("Fetching balance from UNIFIED account...")
    result = await client.get_wallet_balance(account_type="UNIFIED")
    all_results["UNIFIED"] = result
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
    
    # Note: Most Bybit accounts now only support UNIFIED
    # SPOT, CONTRACT, FUND are legacy account types
    # If UNIFIED returns 0, the user needs to transfer funds to Unified Trading wallet
    
    logger.info(f"Balance fetched: €{total_equity:.2f} from {account_type_found} account, {len(coins)} coins")
    
    # If still 0, log all results for debugging
    if total_equity == 0:
        logger.warning(f"No balance found in any account type. Raw results: {all_results}")
                
    return {
        "success": True,
        "data": {
            "totalEquity": total_equity,
            "coins": coins,
            "accountType": account_type_found,
            "debug": all_results if total_equity == 0 else None
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
    """Check if exchange is connected - auto-reconnect from Redis if needed"""
    
    # Check if connected in memory
    if "default" in exchange_connections:
        return {
            "connected": True,
            "exchange": "bybit",
            "serverIp": "109.104.154.183"
        }
    
    # Try to auto-reconnect from Redis
    try:
        r = await get_redis()
        creds = await r.hgetall("exchange:credentials:default")
        
        if creds:
            api_key = simple_decrypt(creds.get(b"api_key", b"").decode())
            api_secret = simple_decrypt(creds.get(b"api_secret", b"").decode())
            testnet = creds.get(b"testnet", b"0").decode() == "1"
            
            if api_key and api_secret:
                # Try to reconnect
                client = BybitV5Client(api_key, api_secret, testnet)
                result = await client.test_connection()
                
                if result.get("success"):
                    exchange_connections["default"] = client
                    logger.info("Auto-reconnected to exchange from saved credentials")
                    return {
                        "connected": True,
                        "exchange": "bybit",
                        "serverIp": "109.104.154.183",
                        "auto_reconnected": True
                    }
                else:
                    await client.close()
    except Exception as e:
        logger.error(f"Auto-reconnect failed: {e}")
    
    return {
        "connected": False,
        "exchange": None,
        "serverIp": "109.104.154.183"
    }


# ========================================
# AUTONOMOUS TRADING ENDPOINTS
# ========================================

from services.autonomous_trader import autonomous_trader
from services.autonomous_trader_v2 import autonomous_trader_v2

# Use V2 by default
USE_V2_TRADER = True

def get_trader():
    """Get the active trader instance"""
    return autonomous_trader_v2 if USE_V2_TRADER else autonomous_trader


class EnableTradingRequest(BaseModel):
    user_id: str = "default"
    api_key: str
    api_secret: str


@router.post("/trading/enable")
async def enable_autonomous_trading(request: EnableTradingRequest):
    """Enable 24/7 autonomous trading for a user - PERSISTED"""
    
    trader = get_trader()
    success = await trader.connect_user(
        user_id=request.user_id,
        api_key=request.api_key,
        api_secret=request.api_secret,
    )
    
    if success:
        # Save trading state to Redis
        r = await get_redis()
        await r.hset(f"trading:enabled:{request.user_id}", mapping={
            "enabled": "1",
            "api_key": simple_encrypt(request.api_key),
            "api_secret": simple_encrypt(request.api_secret),
        })
        logger.info(f"Trading enabled for {request.user_id} - state saved to Redis")
        
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


@router.post("/trading/auto-reconnect")
async def auto_reconnect_trading():
    """Auto-reconnect all users who had trading enabled - called on startup"""
    
    reconnected = []
    
    try:
        r = await get_redis()
        
        # Find all users with trading enabled
        keys_raw = await r.keys("trading:enabled:*")
        keys = list(keys_raw) if keys_raw else []
        
        for key in keys:
            user_id = key.decode().split(":")[-1]
            data = await r.hgetall(key)
            
            if data.get(b"enabled", b"0").decode() == "1":
                api_key = simple_decrypt(data.get(b"api_key", b"").decode())
                api_secret = simple_decrypt(data.get(b"api_secret", b"").decode())
                
                if api_key and api_secret:
                    trader = get_trader()
                    success = await trader.connect_user(
                        user_id=user_id,
                        api_key=api_key,
                        api_secret=api_secret,
                    )
                    if success:
                        reconnected.append(user_id)
                        logger.info(f"Auto-reconnected trading for user: {user_id}")
                        
    except Exception as e:
        logger.error(f"Auto-reconnect error: {e}")
        
    return {
        "success": True,
        "reconnected_users": reconnected,
        "count": len(reconnected)
    }


@router.post("/trading/disable")
async def disable_autonomous_trading(user_id: str = "default"):
    """Disable autonomous trading for a user"""
    trader = get_trader()
    await trader.disconnect_user(user_id)
    
    return {
        "success": True,
        "message": "Autonomous trading disabled"
    }


@router.get("/trading/status")
async def get_trading_status(user_id: str = "default"):
    """Get autonomous trading status"""
    trader = get_trader()
    is_trading = user_id in trader.user_clients
    
    # Get recent trades
    trades = []
    if hasattr(trader, 'redis_client') and trader.redis_client:
        trade_data = await trader.redis_client.lrange(f"trades:completed:{user_id}", 0, 9)
        trades = [json.loads(t) for t in trade_data]
    
    # Get trading pairs (different attribute in v2)
    trading_pairs = getattr(trader, 'trading_pairs', [])
    max_positions = getattr(trader, 'max_open_positions', 10)
    min_conf = getattr(trader, 'min_confidence', 55)
        
    return {
        "success": True,
        "data": {
            "is_autonomous_trading": is_trading,
            "trading_pairs": trading_pairs if is_trading else [],
            "max_positions": max_positions,
            "min_confidence": min_conf,
            "recent_trades": trades,
            "version": "v2" if USE_V2_TRADER else "v1"
        }
    }


@router.get("/trading/log")
async def get_trading_log(limit: int = 50):
    """Get trading activity log"""
    trader = get_trader()
    if not hasattr(trader, 'redis_client') or not trader.redis_client:
        return {"success": False, "error": "Not initialized"}
        
    trades = await trader.redis_client.lrange('trades:log', 0, limit - 1)
    
    return {
        "success": True,
        "data": [json.loads(t) for t in trades]
    }


@router.get("/trading/activity")
async def get_live_activity(user_id: str = "default"):
    """Get LIVE bot activity - what is the bot doing right now?"""
    
    trader = get_trader()
    trading_pairs = getattr(trader, 'trading_pairs', [])
    
    activity = {
        "is_running": trader.is_running,
        "is_user_connected": user_id in trader.user_clients,
        "total_pairs_monitoring": len(trading_pairs),
        "active_trades": [],
        "recent_completed": [],
        "bot_actions": [],
        "current_analysis": None,
        "version": "v2" if USE_V2_TRADER else "v1"
    }
    
    if not hasattr(trader, 'redis_client') or not trader.redis_client:
        return {"success": True, "data": activity}
    
    try:
        # Get active trades for this user
        active_trades = await trader.redis_client.lrange(f"trades:active:{user_id}", 0, -1)
        activity["active_trades"] = [json.loads(t) for t in active_trades]
        
        # Get recent completed trades
        completed = await trader.redis_client.lrange(f"trades:completed:{user_id}", 0, 9)
        activity["recent_completed"] = [json.loads(t) for t in completed]
        
        # Get last bot actions/decisions (from new v2 events list)
        bot_log = await trader.redis_client.lrange('trading:events', 0, 19)
        activity["bot_actions"] = [json.loads(t) for t in bot_log]
        
        # Get current market analysis if available
        analysis = await trader.redis_client.hgetall('market:current_analysis')
        if analysis:
            activity["current_analysis"] = {
                k.decode(): v.decode() for k, v in analysis.items()
            }
            
        # Get trading stats
        trading_status = await trader.redis_client.hgetall('trading:status')
        if trading_status:
            activity["trading_stats"] = {
                k.decode(): v.decode() for k, v in trading_status.items()
            }
        
        # V2 specific: Get trader stats
        if USE_V2_TRADER:
            stats = await trader.get_status()
            activity["trader_stats"] = stats.get('stats', {})
            
    except Exception as e:
        logger.error(f"Error getting activity: {e}")
        
    return {"success": True, "data": activity}


@router.get("/trading/pairs")
async def get_trading_pairs():
    """Get all crypto pairs the bot monitors"""
    
    return {
    trader = get_trader()
    trading_pairs = getattr(trader, 'trading_pairs', [])
    
    return {
        "success": True,
        "data": {
            "pairs": trading_pairs,
            "total": len(trading_pairs),
            "version": "v2" if USE_V2_TRADER else "v1",
            "categories": {
                "top10": trading_pairs[:10] if trading_pairs else [],
                "defi": [p for p in trading_pairs if p in ['COMPUSDT', 'SNXUSDT', 'CRVUSDT', 'YFIUSDT', 'SUSHIUSDT', '1INCHUSDT', 'DYDXUSDT', 'GMXUSDT', 'PENDLEUSDT', 'ENSUSDT']],
                "meme": [p for p in trading_pairs if p in ['SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT']],
                "ai": [p for p in trading_pairs if p in ['TAOUSDT', 'WLDUSDT', 'OCEANUSDT', 'RNDRAUSDT', 'AKTUSDT']],
            }
        }
    }


# ============================================
# BOT SETTINGS
# ============================================

@router.get("/settings")
async def get_settings():
    """Get current bot settings"""
    
    r = await redis.from_url(settings.REDIS_URL)
    
    # Default settings
    defaults = {
        "riskMode": "normal",
        "takeProfitPercent": 3.0,
        "stopLossPercent": 1.5,
        "trailingStopPercent": 1.0,
        "minProfitToTrail": 0.8,
        "minConfidence": 60,
        "minEdge": 0.15,
        "maxPositionPercent": 5,
        "maxOpenPositions": 0,  # 0 = unlimited
        "maxDailyDrawdown": 3,
        "maxTotalExposure": 50,
        "cryptoBudget": 100,
        "tradFiBudget": 0,
        "enableCrypto": True,
        "enableTradFi": False,
        "useAiSignals": True,
        "learnFromTrades": True,
        "useRegimeDetection": True,
        "useEdgeEstimation": True,
        "useDynamicSizing": True,
    }
    
    try:
        settings_data = await r.hgetall('bot:settings')
        
        if settings_data:
            parsed = {
                k.decode() if isinstance(k, bytes) else k: 
                v.decode() if isinstance(v, bytes) else v 
                for k, v in settings_data.items()
            }
            
            # Convert string values to proper types
            return {
                "success": True,
                "data": {
                    "riskMode": parsed.get('riskMode', defaults['riskMode']),
                    "takeProfitPercent": float(parsed.get('takeProfitPercent', defaults['takeProfitPercent'])),
                    "stopLossPercent": float(parsed.get('stopLossPercent', defaults['stopLossPercent'])),
                    "trailingStopPercent": float(parsed.get('trailingStopPercent', defaults['trailingStopPercent'])),
                    "minProfitToTrail": float(parsed.get('minProfitToTrail', defaults['minProfitToTrail'])),
                    "minConfidence": int(float(parsed.get('minConfidence', defaults['minConfidence']))),
                    "minEdge": float(parsed.get('minEdge', defaults['minEdge'])),
                    "maxPositionPercent": int(float(parsed.get('maxPositionPercent', defaults['maxPositionPercent']))),
                    "maxOpenPositions": int(float(parsed.get('maxOpenPositions', defaults['maxOpenPositions']))),
                    "maxDailyDrawdown": float(parsed.get('maxDailyDrawdown', defaults['maxDailyDrawdown'])),
                    "maxTotalExposure": float(parsed.get('maxTotalExposure', defaults['maxTotalExposure'])),
                    "cryptoBudget": float(parsed.get('cryptoBudget', defaults['cryptoBudget'])),
                    "tradFiBudget": float(parsed.get('tradFiBudget', defaults['tradFiBudget'])),
                    "enableCrypto": parsed.get('enableCrypto', 'true') == 'true',
                    "enableTradFi": parsed.get('enableTradFi', 'false') == 'true',
                    "useAiSignals": parsed.get('useAiSignals', 'true') == 'true',
                    "learnFromTrades": parsed.get('learnFromTrades', 'true') == 'true',
                    "useRegimeDetection": parsed.get('useRegimeDetection', 'true') == 'true',
                    "useEdgeEstimation": parsed.get('useEdgeEstimation', 'true') == 'true',
                    "useDynamicSizing": parsed.get('useDynamicSizing', 'true') == 'true',
                }
            }
        else:
            # Return defaults
            return {
                "success": True,
                "data": defaults
            }
    finally:
        await r.aclose()


@router.post("/settings")
async def save_settings(request: Request):
    """Save bot settings and apply them to trader"""
    
    body = await request.json()
    
    r = await redis.from_url(settings.REDIS_URL)
    
    try:
        # Save to Redis - all settings
        settings_to_save = {
            # Risk mode
            'riskMode': str(body.get('riskMode', 'normal')),
            
            # Trading parameters
            'takeProfitPercent': str(body.get('takeProfitPercent', 3.0)),
            'stopLossPercent': str(body.get('stopLossPercent', 1.5)),
            'trailingStopPercent': str(body.get('trailingStopPercent', 1.0)),
            'minProfitToTrail': str(body.get('minProfitToTrail', 0.8)),
            'minConfidence': str(body.get('minConfidence', 60)),
            'minEdge': str(body.get('minEdge', 0.15)),
            
            # Risk management
            'maxPositionPercent': str(body.get('maxPositionPercent', 5)),
            'maxOpenPositions': str(body.get('maxOpenPositions', 0)),  # 0 = unlimited
            'maxDailyDrawdown': str(body.get('maxDailyDrawdown', 3)),
            'maxTotalExposure': str(body.get('maxTotalExposure', 50)),
            
            # Budget allocation
            'cryptoBudget': str(body.get('cryptoBudget', 100)),
            'tradFiBudget': str(body.get('tradFiBudget', 0)),
            'enableCrypto': str(body.get('enableCrypto', True)).lower(),
            'enableTradFi': str(body.get('enableTradFi', False)).lower(),
            
            # AI features
            'useAiSignals': str(body.get('useAiSignals', True)).lower(),
            'learnFromTrades': str(body.get('learnFromTrades', True)).lower(),
            'useRegimeDetection': str(body.get('useRegimeDetection', True)).lower(),
            'useEdgeEstimation': str(body.get('useEdgeEstimation', True)).lower(),
            'useDynamicSizing': str(body.get('useDynamicSizing', True)).lower(),
        }
        
        await r.hset('bot:settings', mapping=settings_to_save)
        
        # Apply settings to autonomous trader (both v1 and v2)
        trader = get_trader()
        
        # Exit strategy
        if hasattr(trader, 'emergency_stop_loss'):
            trader.emergency_stop_loss = float(body.get('stopLossPercent', 1.5))
        if hasattr(trader, 'take_profit'):
            trader.take_profit = float(body.get('takeProfitPercent', 3.0))
        if hasattr(trader, 'trail_from_peak'):
            trader.trail_from_peak = float(body.get('trailingStopPercent', 1.0))
        if hasattr(trader, 'min_profit_to_trail'):
            trader.min_profit_to_trail = float(body.get('minProfitToTrail', 0.8))
            
        # Entry filters
        if hasattr(trader, 'min_confidence'):
            trader.min_confidence = float(body.get('minConfidence', 60))
        if hasattr(trader, 'min_edge'):
            trader.min_edge = float(body.get('minEdge', 0.15))
            
        # Risk limits
        if hasattr(trader, 'max_open_positions'):
            trader.max_open_positions = int(body.get('maxOpenPositions', 0))
        if hasattr(trader, 'max_exposure_percent'):
            trader.max_exposure_percent = float(body.get('maxTotalExposure', 50))
        
        logger.info(f"Bot settings updated: {body.get('riskMode')} mode, "
                   f"TP={body.get('takeProfitPercent')}%, SL={body.get('stopLossPercent')}%")
        
        return {"success": True, "message": "Settings saved and applied"}
        
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return {"success": False, "error": str(e)}
    finally:
        await r.close()


@router.post("/sell-all")
async def sell_all_positions():
    """Emergency: Close ALL open positions immediately"""
    
    if "default" not in exchange_connections:
        return {"success": False, "error": "No exchange connected"}
    
    client = exchange_connections["default"]
    
    try:
        # Get all open positions
        positions_result = await client.get_positions()
        
        if not positions_result.get('success'):
            return {"success": False, "error": "Failed to get positions"}
        
        positions = positions_result.get('data', {}).get('list', [])
        closed_count = 0
        total_pnl = 0.0
        errors = []
        
        for pos in positions:
            size = float(pos.get('size', 0))
            if size <= 0:
                continue
                
            symbol = pos.get('symbol')
            side = pos.get('side')
            unrealized_pnl = float(pos.get('unrealisedPnl', 0))
            
            # Close position
            close_side = 'Sell' if side == 'Buy' else 'Buy'
            
            order_result = await client.place_order(
                symbol=symbol,
                side=close_side,
                order_type='Market',
                qty=str(size),
                reduce_only=True
            )
            
            if order_result.get('success'):
                closed_count += 1
                total_pnl += unrealized_pnl
                logger.info(f"SELL ALL: Closed {symbol} {side} size={size} PnL={unrealized_pnl}")
            else:
                errors.append(f"{symbol}: {order_result.get('error')}")
                logger.error(f"SELL ALL: Failed to close {symbol}: {order_result.get('error')}")
        
        # Clear peak tracking
        r = await redis.from_url(settings.REDIS_URL)
        keys = await r.keys('peak:*')
        if keys:
            await r.delete(*keys)
        await r.close()
        
        return {
            "success": True,
            "data": {
                "closedCount": closed_count,
                "totalPnl": total_pnl,
                "errors": errors if errors else None
            }
        }
        
    except Exception as e:
        logger.error(f"Sell all error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/close-position/{symbol}")
async def close_single_position(symbol: str):
    """Close a specific position by symbol"""
    
    if "default" not in exchange_connections:
        return {"success": False, "error": "No exchange connected"}
    
    client = exchange_connections["default"]
    
    try:
        # Get current positions
        positions_result = await client.get_positions()
        
        if not positions_result.get('success'):
            return {"success": False, "error": "Failed to get positions"}
        
        positions = positions_result.get('data', {}).get('list', [])
        
        # Find the specific position
        target_position = None
        for pos in positions:
            if pos.get('symbol') == symbol and float(pos.get('size', 0)) > 0:
                target_position = pos
                break
        
        if not target_position:
            return {"success": False, "error": f"No open position found for {symbol}"}
        
        size = float(target_position.get('size', 0))
        side = target_position.get('side')
        unrealized_pnl = float(target_position.get('unrealisedPnl', 0))
        
        # Close position (opposite side)
        close_side = 'Sell' if side == 'Buy' else 'Buy'
        
        order_result = await client.place_order(
            symbol=symbol,
            side=close_side,
            order_type='Market',
            qty=str(size),
            reduce_only=True
        )
        
        if order_result.get('success'):
            logger.info(f"MANUAL CLOSE: {symbol} {side} size={size} PnL=€{unrealized_pnl:.2f}")
            
            # Clear peak tracking for this symbol
            r = await redis.from_url(settings.REDIS_URL)
            await r.delete(f'peak:default:{symbol}')
            await r.close()
            
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "pnl": unrealized_pnl,
                    "message": f"Position closed successfully"
                }
            }
        else:
            return {"success": False, "error": order_result.get('error', 'Failed to close position')}
        
    except Exception as e:
        logger.error(f"Close position error for {symbol}: {e}")
        return {"success": False, "error": str(e)}

