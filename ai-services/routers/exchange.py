"""
Exchange Connection API Routes
Real Bybit V5 API integration
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import redis.asyncio as redis
import json
import hashlib
import base64
from loguru import logger

from services.bybit_client import BybitV5Client
from services.binance_client import BinanceClient
from config import settings

# Type alias for exchange clients
ExchangeClient = BybitV5Client | BinanceClient

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
    region: Optional[str] = None  # For Bybit regional endpoints: EU, NL, TR, KZ, GE, AE


class UserCredentialsRequest(BaseModel):
    """Request model for user-specific exchange credentials"""
    user_id: str
    exchange: str
    api_key: str
    api_secret: str
    is_testnet: bool = False
    is_active: bool = True
    region: Optional[str] = None  # For Bybit regional endpoints: EU, NL, TR, KZ, GE, AE


class VerifyCredentialsRequest(BaseModel):
    """Request model for verifying exchange credentials"""
    exchange: str
    api_key: str
    api_secret: str
    is_testnet: bool = False
    region: Optional[str] = None  # For Bybit regional endpoints: EU, NL, TR, KZ, GE, AE


# Store for user-specific exchange connections
user_exchange_connections = {}


@router.post("/verify-credentials")
async def verify_credentials(request: VerifyCredentialsRequest):
    """Verify exchange API credentials without storing them"""
    
    exchange = request.exchange.lower()
    
    if exchange not in ["bybit", "binance"]:
        return {
            "valid": False,
            "error": "Only Bybit and Binance are currently supported"
        }
    
    try:
        # Create appropriate client based on exchange
        if exchange == "binance":
            client = BinanceClient(
                api_key=request.api_key,
                api_secret=request.api_secret,
                testnet=request.is_testnet
            )
        else:  # bybit (default)
            # Support regional endpoints (EU, NL, TR, etc.)
            region = getattr(request, 'region', None)
            client = BybitV5Client(
                api_key=request.api_key,
                api_secret=request.api_secret,
                testnet=request.is_testnet,
                region=region
            )
            if region:
                logger.info(f"Using Bybit region: {region}")
        
        # Test connection
        result = await client.test_connection()
        
        if not result.get("success"):
            await client.close()
            return {
                "valid": False,
                "error": result.get("error", "Connection failed")
            }
        
        # Get balance to verify permissions
        balance = None
        if exchange == "binance":
            # Try spot balance first
            balance_result = await client.get_spot_balance()
            if balance_result.get("success") and balance_result.get("data"):
                # Binance returns array of all non-zero balances
                total_usd_value = 0.0
                assets = []
                
                # Get BTC price for conversion
                btc_price = 0
                try:
                    btc_ticker = await client.get_ticker("BTCUSDT")
                    if btc_ticker.get("success") and btc_ticker.get("data"):
                        btc_price = float(btc_ticker["data"].get("lastPrice", 0))
                except:
                    pass
                
                for asset in balance_result["data"]:
                    asset_name = asset.get("asset", "")
                    free = float(asset.get("free", 0))
                    locked = float(asset.get("locked", 0))
                    total = free + locked
                    
                    if total > 0:
                        # Calculate USD value
                        usd_value = 0
                        if asset_name in ["USDT", "USDC", "BUSD", "FDUSD"]:
                            usd_value = total
                        elif asset_name == "BTC" and btc_price > 0:
                            usd_value = total * btc_price
                        
                        total_usd_value += usd_value
                        assets.append({
                            "asset": asset_name,
                            "balance": total,
                            "usdValue": usd_value
                        })
                
                balance = {
                    "total_equity": total_usd_value,
                    "assets": assets,
                    "currency": "USD"
                }
        else:  # bybit
            balance_result = await client.get_wallet_balance(account_type="UNIFIED")
            if balance_result.get("result", {}).get("list"):
                account = balance_result["result"]["list"][0]
                balance = {
                    "total_equity": float(account.get("totalEquity", 0)),
                    "currency": "USDT"
                }
        
        await client.close()
        
        return {
            "valid": True,
            "permissions": ["read", "trade"],
            "balance": balance
        }
        
    except Exception as e:
        logger.error(f"Credential verification failed: {e}")
        return {
            "valid": False,
            "error": str(e)
        }


@router.post("/set-credentials")
async def set_user_credentials(request: UserCredentialsRequest):
    """Set exchange credentials for a specific user"""
    
    try:
        r = await get_redis()
        exchange = request.exchange.lower()
        
        # Create appropriate client based on exchange
        if exchange == "binance":
            client = BinanceClient(
                api_key=request.api_key,
                api_secret=request.api_secret,
                testnet=request.is_testnet
            )
        else:  # bybit (default)
            # Support regional endpoints (EU, NL, TR, etc.)
            region = getattr(request, 'region', None) or None
            client = BybitV5Client(
                api_key=request.api_key,
                api_secret=request.api_secret,
                testnet=request.is_testnet,
                region=region
            )
            if region:
                logger.info(f"Using Bybit region: {region} for user {request.user_id}")
        
        # Test connection to detect account type (especially for Binance)
        test_result = await client.test_connection()
        account_type = test_result.get("account_type", "spot")  # Default to spot
        has_spot = test_result.get("has_spot", True)
        has_futures = test_result.get("has_futures", False)
        
        # Store encrypted credentials in Redis with account type info
        key = f"user:{request.user_id}:exchange:{request.exchange}"
        region_to_store = getattr(request, 'region', '') or ''
        await r.hset(key, mapping={
            "api_key": simple_encrypt(request.api_key),
            "api_secret": simple_encrypt(request.api_secret),
            "is_testnet": "1" if request.is_testnet else "0",
            "is_active": "1" if request.is_active else "0",
            "account_type": account_type,  # "spot" or "futures"
            "has_spot": "1" if has_spot else "0",
            "has_futures": "1" if has_futures else "0",
            "region": region_to_store,  # Bybit region: EU, NL, TR, etc.
            "updated_at": datetime.utcnow().isoformat(),
        })
        
        logger.info(f"Detected {exchange} account_type={account_type} for user {request.user_id}")
        
        # If active, store connection for API calls
        if request.is_active:
            user_exchange_connections[f"{request.user_id}:{request.exchange}"] = client
            
            # Connect user to autonomous trader (but DON'T start trading automatically for NEW users)
            # Connect to autonomous trader (supports both Bybit and Binance)
            # Import here to avoid circular imports at module load time
            from services.autonomous_trader_v2 import autonomous_trader_v2
            
            # Check if this is a NEW user (never traded before) or existing user
            is_new_user = not await r.exists(f'trading:started:{request.user_id}')
            
            # Connect user to autonomous trader (now supports both Bybit and Binance!)
            connected = await autonomous_trader_v2.connect_user(
                user_id=request.user_id,
                api_key=request.api_key,
                api_secret=request.api_secret,
                testnet=request.is_testnet,
                exchange=exchange
            )
            
            if not connected:
                logger.warning(f"User {request.user_id} connected to {exchange} but autonomous trader failed")
            
            if connected:
                if is_new_user:
                    # NEW USER: Pause by default - must click "Start" to begin
                    await autonomous_trader_v2.pause_trading(request.user_id)
                    logger.info(f"NEW user {request.user_id} connected (PAUSED - must click Start)")
                else:
                    # EXISTING USER: Check if they were paused before
                    was_paused = await r.exists(f'trading:paused:{request.user_id}')
                    if was_paused:
                        logger.info(f"User {request.user_id} reconnected (was PAUSED)")
                    else:
                        logger.info(f"User {request.user_id} reconnected (ACTIVE - continues trading)")
            else:
                logger.warning(f"User {request.user_id} credentials saved but autonomous trader connection failed")
        
        logger.info(f"Credentials set for user {request.user_id} on {request.exchange}")
        
        return {"success": True, "message": "Credentials saved"}
        
    except Exception as e:
        logger.error(f"Failed to set credentials: {e}")
        return {"success": False, "error": str(e)}


@router.delete("/remove-credentials")
async def remove_user_credentials(user_id: str, exchange: str):
    """Remove exchange credentials for a specific user"""
    
    try:
        r = await get_redis()
        
        # Remove from Redis
        key = f"user:{user_id}:exchange:{exchange}"
        await r.delete(key)
        
        # Close and remove connection
        conn_key = f"{user_id}:{exchange}"
        if conn_key in user_exchange_connections:
            await user_exchange_connections[conn_key].close()
            del user_exchange_connections[conn_key]
        
        logger.info(f"Credentials removed for user {user_id} on {exchange}")
        
        return {"success": True, "message": "Credentials removed"}
        
    except Exception as e:
        logger.error(f"Failed to remove credentials: {e}")
        return {"success": False, "error": str(e)}


async def get_user_client(user_id: str, exchange: str = "bybit") -> Optional[ExchangeClient]:
    """Get or create exchange client for a specific user"""
    
    conn_key = f"{user_id}:{exchange}"
    exchange = exchange.lower()
    
    # Return existing connection if available
    # Try to load from Redis first to check account_type
    try:
        r = await get_redis()
        api_key = None
        api_secret = None
        is_testnet = False
        data = None
        
        # 1. Try new format: user:{user_id}:exchange:{exchange}
        key = f"user:{user_id}:exchange:{exchange}"
        data = await r.hgetall(key)
        
        if data:
            is_active = data.get(b"is_active", b"0").decode() == "1"
            if is_active:
                api_key = simple_decrypt(data.get(b"api_key", b"").decode())
                api_secret = simple_decrypt(data.get(b"api_secret", b"").decode())
                is_testnet = data.get(b"is_testnet", b"0").decode() == "1"
        
        # 2. Try legacy format: exchange:credentials:default (for admin/default user)
        if not api_key and user_id == "default":
            legacy_data = await r.hgetall("exchange:credentials:default")
            if legacy_data:
                api_key = simple_decrypt(legacy_data.get(b"api_key", b"").decode())
                api_secret = simple_decrypt(legacy_data.get(b"api_secret", b"").decode())
                is_testnet = legacy_data.get(b"testnet", b"0").decode() == "1"
                logger.debug(f"Using legacy credentials for default user")
        
        if not api_key or not api_secret:
            return None
        
        # For Binance, determine expected account_type from Redis
        expected_account_type = "spot"
        if exchange == "binance" and data:
            account_type_raw = data.get(b"account_type", b"spot").decode()
            has_futures = data.get(b"has_futures", b"0").decode() == "1"
            if has_futures:
                expected_account_type = "futures"
            else:
                expected_account_type = account_type_raw
        
        # Check if cached client exists AND has correct account_type
        if conn_key in user_exchange_connections:
            cached_client = user_exchange_connections[conn_key]
            if exchange == "binance":
                cached_account_type = getattr(cached_client, 'account_type', 'spot')
                if cached_account_type != expected_account_type:
                    # Cache is stale - account_type changed! Invalidate cache.
                    logger.warning(f"Binance client cache invalidated for {user_id}: was {cached_account_type}, now {expected_account_type}")
                    del user_exchange_connections[conn_key]
                else:
                    return cached_client
            else:
                return cached_client
        
        # Create appropriate client based on exchange
        if exchange == "binance":
            # Use the account_type we already determined
            account_type = expected_account_type
            
            client = BinanceClient(
                api_key=api_key,
                api_secret=api_secret,
                testnet=is_testnet,
                account_type=account_type
            )
            
            # For Futures, verify connection works
            if account_type == "futures":
                test_result = await client.test_connection()
                if test_result.get("success"):
                    # Update account_type from test result if different
                    detected_type = test_result.get("account_type", account_type)
                    if detected_type != account_type:
                        client.set_account_type(detected_type)
                        logger.info(f"BinanceClient account_type updated to {detected_type}")
                else:
                    logger.warning(f"Binance connection test failed: {test_result.get('error')}")
                    
            logger.info(f"Created BinanceClient for {user_id} with account_type={client.account_type}")
        else:  # bybit (default)
            # Get region from Redis (for Bybit EU, NL, etc.)
            region = data.get(b"region", b"").decode() if data else ""
            region = region if region else None
            
            client = BybitV5Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=is_testnet,
                region=region
            )
            if region:
                logger.info(f"Created BybitClient for {user_id} with region={region}")
        
        user_exchange_connections[conn_key] = client
        return client
        
    except Exception as e:
        logger.error(f"Failed to get user client: {e}")
        return None


async def detect_user_exchange(user_id: str) -> Optional[str]:
    """Detect which exchange the user is connected to (binance or bybit)"""
    try:
        r = await get_redis()
        
        # Check Binance first
        binance_key = f"user:{user_id}:exchange:binance"
        binance_data = await r.hgetall(binance_key)
        if binance_data and binance_data.get(b"is_active", b"0").decode() == "1":
            return "binance"
        
        # Check Bybit
        bybit_key = f"user:{user_id}:exchange:bybit"
        bybit_data = await r.hgetall(bybit_key)
        if bybit_data and bybit_data.get(b"is_active", b"0").decode() == "1":
            return "bybit"
        
        # Legacy check for default user
        if user_id == "default":
            legacy_data = await r.hgetall("exchange:credentials:default")
            if legacy_data:
                return "bybit"
        
        return None
    except Exception as e:
        logger.error(f"Failed to detect user exchange: {e}")
        return None


@router.post("/test")
async def test_exchange_connection(request: TestConnectionRequest):
    """Test exchange API connection"""
    
    exchange = request.exchange.lower()
    if exchange not in ["bybit", "binance"]:
        return {"success": False, "error": "Only Bybit and Binance are currently supported"}
        
    try:
        # Create appropriate client based on exchange
        if exchange == "binance":
            client = BinanceClient(
                api_key=request.apiKey,
                api_secret=request.apiSecret,
                testnet=False
            )
        else:  # bybit (default)
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
    
    exchange = credentials.exchange.lower()
    if exchange not in ["bybit", "binance"]:
        return {"success": False, "error": "Only Bybit and Binance are currently supported"}
        
    try:
        # Test connection first - create appropriate client based on exchange
        if exchange == "binance":
            client = BinanceClient(
                api_key=credentials.apiKey,
                api_secret=credentials.apiSecret,
                testnet=credentials.testnet
            )
        else:  # bybit (default)
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
        user_exchange_connections[f"default:{exchange}"] = client
        
        # PERSIST credentials to Redis (encrypted) - BOTH formats for compatibility
        r = await get_redis()
        
        # Legacy format (for backwards compatibility) - only for bybit
        if exchange == "bybit":
            await r.hset("exchange:credentials:default", mapping={
                "exchange": credentials.exchange,
                "api_key": simple_encrypt(credentials.apiKey),
                "api_secret": simple_encrypt(credentials.apiSecret),
                "testnet": "1" if credentials.testnet else "0",
                "connected_at": str(json.dumps({"ts": "now"})),
            })
        
        # New format (for get_user_client)
        # For Binance, also store the detected account_type (spot/futures)
        account_type = result.get("account_type", "spot")  # Default to spot if not detected
        has_spot = result.get("has_spot", True)
        has_futures = result.get("has_futures", False)
        
        await r.hset(f"user:default:exchange:{exchange}", mapping={
            "api_key": simple_encrypt(credentials.apiKey),
            "api_secret": simple_encrypt(credentials.apiSecret),
            "is_testnet": "1" if credentials.testnet else "0",
            "is_active": "1",
            "account_type": account_type,  # "spot" or "futures"
            "has_spot": "1" if has_spot else "0",
            "has_futures": "1" if has_futures else "0",
            "updated_at": datetime.utcnow().isoformat(),
        })
        
        logger.info(f"{exchange.capitalize()} credentials saved to Redis (account_type={account_type}) - will persist across restarts")
        
        return {
            "success": True,
            "message": f"{exchange.capitalize()} connected successfully - credentials saved",
            "account_type": account_type,
            "has_futures": has_futures
        }
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return {"success": False, "error": str(e)}


@router.get("/user/{user_id}/balance")
async def get_user_balance(user_id: str, exchange: str = "auto"):
    """Get wallet balance for a specific user"""
    
    # Auto-detect exchange if not specified
    if exchange == "auto":
        exchange = await detect_user_exchange(user_id) or "bybit"
    
    client = await get_user_client(user_id, exchange)
    if not client:
        return {"success": False, "error": "No exchange connected for this user"}
    
    try:
        coins = []
        total_equity = 0
        
        if exchange == "binance":
            # BINANCE: Get spot balance
            result = await client.get_spot_balance()
            if not result.get("success"):
                return {"success": False, "error": "Failed to fetch balance"}
            
            # Get BTC price for conversion
            btc_price = 0
            try:
                btc_ticker = await client.get_ticker("BTCUSDT")
                if btc_ticker.get("success") and btc_ticker.get("data"):
                    btc_price = float(btc_ticker["data"].get("lastPrice", 0))
            except:
                pass
            
            for asset in result.get("data", []):
                asset_name = asset.get("asset", "")
                free = float(asset.get("free", 0))
                locked = float(asset.get("locked", 0))
                total = free + locked
                
                if total > 0:
                    usd_value = 0
                    if asset_name in ["USDT", "USDC", "BUSD", "FDUSD"]:
                        usd_value = total
                    elif asset_name == "BTC" and btc_price > 0:
                        usd_value = total * btc_price
                    
                    total_equity += usd_value
                    coins.append({
                        "coin": asset_name,
                        "balance": total,
                        "equity": total,
                        "usdValue": usd_value,
                        "unrealizedPnl": 0
                    })
        else:
            # BYBIT: Use unified account
            result = await client.get_wallet_balance(account_type="UNIFIED")
            
            if not result.get("success"):
                return {"success": False, "error": "Failed to fetch balance"}
            
            data = result.get("data", {})
            for account in data.get("list", []):
                eq = float(account.get("totalEquity", 0))
                if eq > 0:
                    total_equity = eq
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
                "coins": coins,
                "exchange": exchange
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get user balance: {e}")
        return {"success": False, "error": str(e)}


@router.get("/user/{user_id}/positions")
async def get_user_positions(user_id: str, exchange: str = "bybit"):
    """Get open positions for a specific user"""
    
    client = await get_user_client(user_id, exchange)
    if not client:
        return {"success": False, "error": "No exchange connected for this user"}
    
    try:
        result = await client.get_positions()
        
        if not result.get("success"):
            return {"success": False, "error": "Failed to fetch positions"}
        
        positions = []
        data = result.get("data", {})
        
        for pos in data.get("list", []):
            size = float(pos.get("size", 0))
            if size > 0:
                mark_price = float(pos.get("markPrice", 0))
                position_value = float(pos.get("positionValue", 0))
                if position_value == 0 and mark_price > 0:
                    position_value = size * mark_price
                
                # Gross P&L (without fees)
                unrealized_pnl_gross = float(pos.get("unrealisedPnl", 0))
                
                # Calculate exit fee (taker: 0.055%)
                estimated_exit_fee = position_value * 0.00055
                
                # NET P&L = Gross - Exit Fee
                estimated_net_pnl = unrealized_pnl_gross - estimated_exit_fee
                
                positions.append({
                    "symbol": pos.get("symbol"),
                    "side": pos.get("side"),
                    "size": size,
                    "entryPrice": float(pos.get("avgPrice", 0)),
                    "markPrice": mark_price,
                    "positionValue": position_value,
                    "unrealisedPnl": unrealized_pnl_gross,  # Gross
                    "estimatedNetPnl": round(estimated_net_pnl, 4),  # NET (after fees)
                    "estimatedExitFee": round(estimated_exit_fee, 4),
                    "leverage": pos.get("leverage"),
                })
        
        return {
            "success": True,
            "positions": positions,
        }
        
    except Exception as e:
        logger.error(f"Failed to get user positions: {e}")
        return {"success": False, "error": str(e)}


@router.post("/sync")
async def sync_user_exchange(user_id: str, exchange: str = "bybit"):
    """Sync exchange data for a specific user"""
    
    client = await get_user_client(user_id, exchange)
    if not client:
        return {"success": False, "error": "No exchange connected for this user"}
    
    try:
        # Get balance
        balance_result = await client.get_wallet_balance(account_type="UNIFIED")
        
        # Get positions
        positions_result = await client.get_positions()
        
        # Store in Redis for user
        r = await get_redis()
        
        if balance_result.get("success"):
            await r.set(
                f"user:{user_id}:balance",
                json.dumps(balance_result.get("data", {})),
                ex=60  # Cache for 60 seconds
            )
        
        if positions_result.get("success"):
            await r.set(
                f"user:{user_id}:positions",
                json.dumps(positions_result.get("data", {})),
                ex=60
            )
        
        return {
            "success": True,
            "message": "Sync completed",
            "balance": balance_result.get("data") if balance_result.get("success") else None,
            "positions": positions_result.get("data") if positions_result.get("success") else None,
        }
        
    except Exception as e:
        logger.error(f"Sync failed for user {user_id}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/balance")
async def get_balance(user_id: str = "default"):
    """Get real wallet balance from connected exchange - checks ALL account types"""
    
    coins = []
    total_equity = 0
    account_type_found = None
    all_results = {}  # For debugging
    
    # Detect which exchange the user is connected to
    exchange_type = await detect_user_exchange(user_id)
    
    if not exchange_type:
        return {"success": False, "error": "No exchange connected for this user"}
    
    # Get user-specific client
    client = await get_user_client(user_id, exchange_type)
    
    if not client:
        return {"success": False, "error": f"Failed to connect to {exchange_type}"}
    
    # Handle based on exchange type
    if exchange_type == "binance":
        # BINANCE: Check account type (SPOT or Futures)
        # First try Futures if user has Futures account
        account_type = getattr(client, 'account_type', 'spot')
        
        if account_type == "futures":
            logger.info(f"Fetching balance from Binance FUTURES for user {user_id}...")
            result = await client.get_wallet_balance()
            all_results["FUTURES"] = result
            
            if result.get("success") and result.get("data"):
                account_type_found = "futures"
                data = result.get("data", [])
                
                # Binance Futures /fapi/v2/balance returns array directly, not object with "assets" key
                # Handle both formats for safety
                assets_list = data if isinstance(data, list) else data.get("assets", [])
                
                logger.debug(f"Binance FUTURES balance data type: {type(data)}, count: {len(assets_list) if assets_list else 0}")
                
                for asset in assets_list:
                    # Binance uses "balance" or "walletBalance" depending on endpoint
                    wallet_balance = float(asset.get("walletBalance", asset.get("balance", 0)))
                    unrealized_pnl = float(asset.get("unrealizedProfit", asset.get("crossUnPnl", 0)))
                    available = float(asset.get("availableBalance", asset.get("crossWalletBalance", wallet_balance)))
                    
                    if wallet_balance > 0 or unrealized_pnl != 0:
                        asset_name = asset.get("asset", "USDT")
                        coins.append({
                            "coin": asset_name,
                            "balance": wallet_balance,
                            "equity": wallet_balance + unrealized_pnl,
                            "usdValue": wallet_balance + unrealized_pnl,  # Futures is in USDT
                            "unrealizedPnl": unrealized_pnl,
                            "free": available,
                            "locked": wallet_balance - available if wallet_balance > available else 0
                        })
                        total_equity += wallet_balance + unrealized_pnl
                
                logger.info(f"Binance FUTURES balance fetched: ${total_equity:.2f} total, {len(coins)} assets")
        
        # Fallback to SPOT or if Futures failed
        if not account_type_found or account_type == "spot":
            logger.info(f"Fetching balance from Binance SPOT for user {user_id}...")
            result = await client.get_spot_balance()
            all_results["SPOT"] = result
        
        if result.get("success") and result.get("data") and not account_type_found:
            account_type_found = "spot"
            # Get current prices to calculate USD value
            # For now, fetch BTC and other major asset prices
            btc_price = 0
            eth_price = 0
            try:
                btc_ticker = await client.get_ticker("BTCUSDT")
                if btc_ticker.get("success") and btc_ticker.get("data"):
                    btc_price = float(btc_ticker["data"].get("lastPrice", 0))
                eth_ticker = await client.get_ticker("ETHUSDT")
                if eth_ticker.get("success") and eth_ticker.get("data"):
                    eth_price = float(eth_ticker["data"].get("lastPrice", 0))
            except Exception as e:
                logger.warning(f"Failed to get prices: {e}")
            
            for asset in result["data"]:
                asset_name = asset.get("asset", "")
                free = float(asset.get("free", 0))
                locked = float(asset.get("locked", 0))
                total_balance = free + locked
                
                if total_balance > 0:
                    # Calculate USD value
                    usd_value = 0
                    if asset_name == "USDT":
                        usd_value = total_balance
                    elif asset_name == "BTC" and btc_price > 0:
                        usd_value = total_balance * btc_price
                    elif asset_name == "ETH" and eth_price > 0:
                        usd_value = total_balance * eth_price
                    elif asset_name in ["USDC", "BUSD", "FDUSD"]:
                        usd_value = total_balance  # Stablecoins
                    else:
                        # Try to get price for other assets
                        try:
                            ticker = await client.get_ticker(f"{asset_name}USDT")
                            if ticker.get("success") and ticker.get("data"):
                                price = float(ticker["data"].get("lastPrice", 0))
                                usd_value = total_balance * price
                        except:
                            pass
                    
                    coins.append({
                        "coin": asset_name,
                        "balance": total_balance,
                        "equity": total_balance,
                        "usdValue": usd_value,
                        "unrealizedPnl": 0,
                        "free": free,
                        "locked": locked
                    })
                    total_equity += usd_value
            
            logger.info(f"Binance balance fetched: ${total_equity:.2f} total, {len(coins)} assets")
    else:
        # BYBIT: Use unified account
        logger.info(f"Fetching balance from Bybit UNIFIED for user {user_id}...")
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
        
        logger.info(f"Bybit balance fetched: {total_equity:.2f} USDT from {account_type_found} account, {len(coins)} coins")
    
    # If still 0, log all results for debugging
    if total_equity == 0:
        logger.warning(f"No balance found in any account type. Raw results: {all_results}")
    
    # Get USDT/EUR conversion rate (approximate)
    usdt_to_eur = 0.92  # TODO: Fetch real rate
    
    # Calculate available balance
    available_balance_usdt = 0
    if exchange_type == "binance":
        # For Binance, available = total equity (all assets converted to USD)
        available_balance_usdt = total_equity
    else:
        # For Bybit, use USDT coin balance
        usdt_coin = next((c for c in coins if c.get("coin") == "USDT"), None)
        if usdt_coin:
            available_balance_usdt = usdt_coin.get("balance", 0) - usdt_coin.get("unrealizedPnl", 0)
    
    # Calculate unrealized PnL
    total_unrealized_pnl = sum(c.get("unrealizedPnl", 0) for c in coins)
    
    # Get daily/weekly P&L from USER-SPECIFIC Redis stats
    try:
        r = await redis.from_url(settings.REDIS_URL)
        
        # Try user-specific stats first
        stats_raw = await r.get(f'trader:stats:{user_id}')
        sizer_raw = await r.get(f'sizer:state:{user_id}')
        
        # For admin/default user: migrate from global stats if user-specific doesn't exist
        if not stats_raw and user_id == 'default':
            global_stats = await r.get('trader:stats')
            if global_stats:
                # Migrate admin data to user-specific key
                await r.set(f'trader:stats:{user_id}', global_stats)
                stats_raw = global_stats
                logger.info(f"Migrated global stats to trader:stats:{user_id}")
        
        if not sizer_raw and user_id == 'default':
            global_sizer = await r.get('sizer:state')
            if global_sizer:
                await r.set(f'sizer:state:{user_id}', global_sizer)
                sizer_raw = global_sizer
        
        stats = json.loads(stats_raw) if stats_raw else {}
        sizer = json.loads(sizer_raw) if sizer_raw else {}
        await r.close()
        
        total_pnl = float(stats.get('total_pnl', 0))
        # Daily P&L: prefer stats (updated on each trade), fallback to sizer
        daily_pnl = float(stats.get('daily_pnl', 0))
        if daily_pnl == 0:
            daily_pnl = float(sizer.get('daily_pnl', 0))
        weekly_pnl = float(sizer.get('weekly_pnl', 0))
        
        # Log for debugging
        logger.debug(f"Balance P&L for {user_id}: daily=${daily_pnl:.2f}, total=${total_pnl:.2f}, date={stats.get('daily_pnl_date', 'N/A')}")
    except Exception as e:
        logger.warning(f"Failed to load P&L stats for {user_id}: {e}")
        total_pnl = 0
        daily_pnl = 0
        weekly_pnl = 0
                
    return {
        "success": True,
        "data": {
            "totalEquity": total_equity * usdt_to_eur,
            "totalEquityUSDT": total_equity,
            "availableBalance": available_balance_usdt * usdt_to_eur,
            "availableBalanceUSDT": available_balance_usdt,
            "totalPnL": total_pnl,
            "dailyPnL": daily_pnl,
            "weeklyPnL": weekly_pnl,
            "unrealizedPnL": total_unrealized_pnl * usdt_to_eur,
            "coins": coins,
            "accountType": account_type_found,
            "exchange": exchange_type,
            "currency": "USDT",
            "conversionRate": usdt_to_eur,
            "debug": all_results if total_equity == 0 else None
        }
    }


@router.get("/signals")
async def get_ai_signals(user_id: str = "default", limit: int = 5):
    """Get top AI trading signals - opportunities the bot is considering (PER USER)"""
    
    # Get user-specific client or fallback to any connected client
    client = exchange_connections.get(user_id) or exchange_connections.get("default")
    if not client:
        # Try to get any connected client
        if exchange_connections:
            client = list(exchange_connections.values())[0]
        else:
            return {"signals": [], "error": "No exchange connected"}
    
    signals = []
    
    try:
        # Use USER-SPECIFIC settings
        take_profit = 2.0
        stop_loss = 1.5
        min_edge = 0.25
        
        try:
            r = await get_redis()
            # Check key type FIRST to avoid WRONGTYPE errors
            settings_key = f"bot:settings:{user_id}"
            key_type = await r.type(settings_key)
            key_type_str = key_type.decode() if isinstance(key_type, bytes) else str(key_type)
            
            settings = {}
            if key_type_str == 'hash':
                data = await r.hgetall(settings_key)
                if data:
                    settings = {
                        k.decode() if isinstance(k, bytes) else k: 
                        v.decode() if isinstance(v, bytes) else v 
                        for k, v in data.items()
                    }
            elif key_type_str == 'string':
                settings_raw = await r.get(settings_key)
                if settings_raw:
                    try:
                        settings = json.loads(settings_raw)
                    except:
                        pass
            
            if settings:
                take_profit = float(settings.get("takeProfitPercent", take_profit))
                stop_loss = float(settings.get("stopLossPercent", stop_loss))
                min_edge = float(settings.get("minEdge", min_edge))
        except Exception:
            pass  # Use default values
        
        # Get tickers for analysis
        tickers_result = await client.get_tickers()
        if not tickers_result.get('success'):
            return {"signals": []}
        
        tickers = tickers_result.get('data', {}).get('list', [])
        
        # Score and filter opportunities
        opportunities = []
        for ticker in tickers[:100]:  # Scan top 100 by volume
            symbol = ticker.get('symbol', '')
            if not symbol.endswith('USDT') or symbol in ['USDCUSDT', 'USDTUSDT']:
                continue
            
            price_change = float(ticker.get('price24hPcnt', 0)) * 100
            volume = float(ticker.get('turnover24h', 0))
            last_price = float(ticker.get('lastPrice', 0))
            bid_price = float(ticker.get('bid1Price', 0))
            ask_price = float(ticker.get('ask1Price', 0))
            
            if volume < 1000000 or last_price <= 0:  # Min $1M volume
                continue
            
            # Calculate spread
            spread = ((ask_price - bid_price) / last_price * 100) if last_price > 0 else 0
            if spread > 0.5:  # Skip high spread pairs
                continue
            
            # Determine direction based on momentum
            direction = 'LONG' if price_change > 0.5 else 'SHORT' if price_change < -0.5 else None
            if not direction:
                continue
            
            # Calculate edge (simplified)
            volatility = abs(price_change)
            edge = max(0, volatility * 0.3 - spread * 2)  # Simple edge calculation
            
            if edge < min_edge:
                continue
            
            # Calculate confidence based on volume and momentum
            volume_score = min(100, volume / 10000000 * 50)  # Up to 50 from volume
            momentum_score = min(50, abs(price_change) * 10)  # Up to 50 from momentum
            confidence = int(volume_score + momentum_score)
            
            # Calculate entry, TP, SL
            entry_price = last_price
            if direction == 'LONG':
                target_price = entry_price * (1 + take_profit / 100)
                stop_loss_price = entry_price * (1 - stop_loss / 100)
            else:
                target_price = entry_price * (1 - take_profit / 100)
                stop_loss_price = entry_price * (1 + stop_loss / 100)
            
            # Get funding rate if available (skip - causes Redis errors)
            funding_rate = None
            
            opportunities.append({
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "entry_price": entry_price,
                "target_price": target_price,
                "stop_loss": stop_loss_price,
                "edge": edge,
                "reason": f"{'Strong bullish' if direction == 'LONG' else 'Strong bearish'} momentum ({price_change:+.1f}%)",
                "funding_rate": funding_rate,
                "volume_24h": volume,
                "price_change_24h": price_change
            })
        
        # Sort by confidence and take top N
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        signals = opportunities[:limit]
        
    except Exception as e:
        logger.error(f"Error getting AI signals: {e}")
    
    return {"signals": signals}


@router.get("/intelligence")
async def get_ai_intelligence(user_id: str = "default"):
    """
    Get AI Intelligence status - shows what AI is doing in REAL TIME
    PER USER - each user sees THEIR OWN data!
    
    Returns:
    - news_sentiment: Current market sentiment from news
    - breakouts_detected: Number of breakouts found
    - breakout_alerts: List of current breakouts
    - pairs_analyzed: How many pairs AI scanned
    - last_action: What AI did last
    - strategy_mode: Current trading strategy
    """
    intelligence = {
        "news_sentiment": "neutral",
        "breakouts_detected": 0,
        "breakout_alerts": [],
        "pairs_analyzed": 0,
        "last_action": "Scanning market...",
        "strategy_mode": "NORMAL",
        "active_preset": "BALANCED"
    }
    
    try:
        r = await get_redis()
        # Get USER-SPECIFIC settings for strategy mode
        # Check key type FIRST to avoid WRONGTYPE errors
        settings_key = f"bot:settings:{user_id}"
        key_type = await r.type(settings_key)
        key_type_str = key_type.decode() if isinstance(key_type, bytes) else str(key_type)
        
        settings = {}
        if key_type_str == 'hash':
            # Key is a HASH - use HGETALL
            data = await r.hgetall(settings_key)
            if data:
                settings = {
                    k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v 
                    for k, v in data.items()
                }
        elif key_type_str == 'string':
            # Key is a STRING - use GET and parse JSON
            settings_raw = await r.get(settings_key)
            if settings_raw:
                try:
                    settings = json.loads(settings_raw)
                except:
                    pass
        
        if settings:
            # Check if AI Full Auto mode
            if str(settings.get("aiFullAuto", "false")).lower() == "true":
                intelligence["strategy_mode"] = "AI AUTO"
            else:
                # Use strategyPreset if set, otherwise riskMode
                preset = settings.get("strategyPreset", "").upper()
                risk_mode = settings.get("riskMode", "NORMAL").upper()
                intelligence["strategy_mode"] = preset if preset else risk_mode
            intelligence["active_preset"] = settings.get("strategyPreset", "BALANCED").upper()
        
        # Get trader stats for pairs analyzed - use global counter key
        try:
            pairs_scanned = await r.get("trader:global:opportunities_scanned")
            if pairs_scanned:
                intelligence["pairs_analyzed"] = int(pairs_scanned)
            else:
                # Fallback: try to get from local stats
                intelligence["pairs_analyzed"] = 559  # Default pairs count
        except Exception as e:
            logger.debug(f"Could not get pairs_analyzed: {e}")
            intelligence["pairs_analyzed"] = 559
        
        # Get live status from trading loop (USER-SPECIFIC)
        try:
            live_status_raw = await r.get(f"bot:status:live:{user_id}")
            if not live_status_raw:
                # Fallback to global status
                live_status_raw = await r.get("bot:status:live")
            if live_status_raw:
                live_status = json.loads(live_status_raw)
                intelligence["last_action"] = live_status.get("last_action", "Trading active...")
                if live_status.get("strategy"):
                    intelligence["strategy_mode"] = live_status.get("strategy")
        except Exception as e:
            logger.debug(f"Could not get live status: {e}")
        
        # Also try USER-SPECIFIC console logs for more specific actions
        try:
            console_logs_raw = await r.lrange(f"bot:console:logs:{user_id}", 0, 5)
            if console_logs_raw:
                # Find the most recent TRADE or SIGNAL action
                for log_raw in console_logs_raw:
                    try:
                        log = json.loads(log_raw)
                        msg = log.get("message", "")
                        level = log.get("level", "")
                        if level in ["TRADE", "SIGNAL"] or "OPENED" in msg or "CLOSED" in msg or "BREAKOUT" in msg:
                            intelligence["last_action"] = msg[:100]
                            break
                    except:
                        continue
        except Exception:
            pass
        
        # Get news sentiment from cache
        try:
            news_raw = await r.get("market:news:cache")
            if news_raw:
                news_list = json.loads(news_raw)
                bullish = sum(1 for n in news_list if n.get('sentiment') == 'bullish')
                bearish = sum(1 for n in news_list if n.get('sentiment') == 'bearish')
                total = bullish + bearish
                if total > 0:
                    score = (bullish - bearish) / total
                    if score > 0.3:
                        intelligence["news_sentiment"] = "bullish"
                    elif score < -0.3:
                        intelligence["news_sentiment"] = "bearish"
                    else:
                        intelligence["news_sentiment"] = "neutral"
        except Exception:
            pass  # Ignore Redis errors for news
        
        # Scan for BREAKOUTS in real-time
        if "default" in exchange_connections:
            client = exchange_connections["default"]
            tickers_result = await client.get_tickers()
            
            if tickers_result.get('success'):
                tickers = tickers_result.get('data', {}).get('list', [])
                breakouts = []
                
                for ticker in tickers:
                    symbol = ticker.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    
                    price_change = float(ticker.get('price24hPcnt', 0)) * 100
                    volume = float(ticker.get('turnover24h', 0))
                    
                    # Breakout: +5% or more with decent volume
                    if abs(price_change) >= 5 and volume >= 500000:
                        breakouts.append({
                            "symbol": symbol,
                            "change": round(price_change, 2),
                            "volume": round(volume / 1000000, 2),  # In millions
                            "time": datetime.utcnow().isoformat()
                        })
                
                # Sort by absolute change
                breakouts.sort(key=lambda x: abs(x['change']), reverse=True)
                intelligence["breakouts_detected"] = len(breakouts)
                intelligence["breakout_alerts"] = breakouts[:10]  # Top 10
        
    except Exception as e:
        logger.error(f"Error getting AI intelligence: {e}")
    
    return intelligence


@router.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """
    Get comprehensive market data for a symbol including:
    - Long/Short Ratio
    - Open Interest
    - Fee Rates
    - Recent Trades summary
    
    This is the FULL picture for decision making!
    """
    if "default" not in exchange_connections:
        return {"error": "No exchange connected"}
    
    client = exchange_connections["default"]
    data = {
        "symbol": symbol,
        "long_short_ratio": None,
        "open_interest": None,
        "fee_rate": None,
        "funding_rate": None,
        "recent_trade_bias": None
    }
    
    try:
        # Long/Short Ratio
        ls_result = await client.get_long_short_ratio(symbol)
        if ls_result.get('retCode') == 0:
            ls_list = ls_result.get('result', {}).get('list', [])
            if ls_list:
                buy_ratio = float(ls_list[0].get('buyRatio', 0.5))
                sell_ratio = float(ls_list[0].get('sellRatio', 0.5))
                data['long_short_ratio'] = {
                    'buy_ratio': round(buy_ratio * 100, 1),
                    'sell_ratio': round(sell_ratio * 100, 1),
                    'ratio': round(buy_ratio / sell_ratio, 2) if sell_ratio > 0 else 1.0,
                    'sentiment': 'bullish' if buy_ratio > sell_ratio else 'bearish'
                }
        
        # Open Interest
        oi_result = await client.get_open_interest(symbol)
        if oi_result.get('retCode') == 0:
            oi_list = oi_result.get('result', {}).get('list', [])
            if oi_list:
                oi_value = float(oi_list[0].get('openInterest', 0))
                data['open_interest'] = {
                    'value': oi_value,
                    'value_formatted': f"${oi_value/1000000:.1f}M" if oi_value > 1000000 else f"${oi_value/1000:.0f}K"
                }
        
        # Fee Rate
        fee_result = await client.get_fee_rate(symbol)
        if fee_result.get('retCode') == 0:
            fee_list = fee_result.get('result', {}).get('list', [])
            if fee_list:
                taker = float(fee_list[0].get('takerFeeRate', 0.0006))
                maker = float(fee_list[0].get('makerFeeRate', 0.0001))
                data['fee_rate'] = {
                    'taker': round(taker * 100, 4),  # As percentage
                    'maker': round(maker * 100, 4),
                    'taker_display': f"{taker * 100:.3f}%",
                    'maker_display': f"{maker * 100:.3f}%"
                }
        
        # Funding Rate
        funding_result = await client.get_funding_rate(symbol)
        if funding_result.get('retCode') == 0:
            funding_list = funding_result.get('result', {}).get('list', [])
            if funding_list:
                rate = float(funding_list[0].get('fundingRate', 0))
                data['funding_rate'] = {
                    'rate': round(rate * 100, 4),
                    'display': f"{rate * 100:+.4f}%",
                    'direction': 'longs pay shorts' if rate > 0 else 'shorts pay longs'
                }
        
        # Recent Trades Analysis (buy vs sell volume)
        trades_result = await client.get_recent_trades(symbol, limit=60)
        if trades_result.get('retCode') == 0:
            trades = trades_result.get('result', {}).get('list', [])
            buy_volume = sum(float(t.get('size', 0)) for t in trades if t.get('side') == 'Buy')
            sell_volume = sum(float(t.get('size', 0)) for t in trades if t.get('side') == 'Sell')
            total = buy_volume + sell_volume
            if total > 0:
                data['recent_trade_bias'] = {
                    'buy_percent': round(buy_volume / total * 100, 1),
                    'sell_percent': round(sell_volume / total * 100, 1),
                    'bias': 'buying' if buy_volume > sell_volume else 'selling'
                }
    
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
    
    return data


@router.get("/fee-rates")
async def get_fee_rates():
    """Get your trading fee rates"""
    if "default" not in exchange_connections:
        return {"error": "No exchange connected"}
    
    client = exchange_connections["default"]
    
    try:
        result = await client.get_fee_rate(category="linear")
        if result.get('retCode') == 0:
            fees = result.get('result', {}).get('list', [])
            # Get default fee (usually first entry or BTCUSDT)
            if fees:
                return {
                    "taker_fee": f"{float(fees[0].get('takerFeeRate', 0)) * 100:.3f}%",
                    "maker_fee": f"{float(fees[0].get('makerFeeRate', 0)) * 100:.3f}%",
                    "raw": fees[:5]  # First 5 for reference
                }
    except Exception as e:
        logger.error(f"Error getting fee rates: {e}")
    
    return {"taker_fee": "0.060%", "maker_fee": "0.010%", "error": "Could not fetch real rates"}


@router.get("/positions")
async def get_positions(user_id: str = "default"):
    """Get real open positions from connected exchange"""
    
    # Detect which exchange the user is connected to
    exchange_type = await detect_user_exchange(user_id)
    
    if not exchange_type:
        return {"success": False, "error": "No exchange connected for this user"}
    
    # Get user-specific client
    client = await get_user_client(user_id, exchange_type)
    
    if not client:
        return {"success": False, "error": f"Failed to connect to {exchange_type}"}
    
    # For Binance SPOT: No futures positions, return empty (spot holdings are in /balance)
    if exchange_type == "binance":
        # For now, Binance spot doesn't have "positions" like futures
        # The balance endpoint shows all holdings
        return {
            "success": True,
            "data": {"positions": []},
            "message": "Binance Spot connected - holdings shown in balance. Futures trading coming soon."
        }
        
    result = await client.get_positions()
    
    if not result.get("success"):
        return result
    
    # Get breakout positions and trade mode info from Redis
    breakout_symbols = set()
    trade_mode_info = {}  # symbol -> {mode, mode_display, is_spot}
    try:
        r = await get_redis()
        breakout_data = await r.hgetall("positions:breakout")
        for symbol, _ in breakout_data.items():
            if isinstance(symbol, bytes):
                breakout_symbols.add(symbol.decode())
            else:
                breakout_symbols.add(symbol)
        
        # Get trade mode info for positions
        mode_data = await r.hgetall(f"positions:trade_mode:{user_id}")
        for symbol, mode_json in mode_data.items():
            try:
                symbol_str = symbol.decode() if isinstance(symbol, bytes) else symbol
                mode_json_str = mode_json.decode() if isinstance(mode_json, bytes) else mode_json
                trade_mode_info[symbol_str] = json.loads(mode_json_str)
            except:
                pass
    except Exception as e:
        logger.warning(f"Failed to get position metadata: {e}")
        
    # Parse positions
    positions = []
    data = result.get("data", {})
    
    for pos in data.get("list", []):
        size = float(pos.get("size", 0))
        if size > 0:
            symbol = pos.get("symbol")
            mark_price = float(pos.get("markPrice", 0))
            entry_price = float(pos.get("avgPrice", 0))
            # Calculate position value: size * mark price
            position_value = float(pos.get("positionValue", 0))
            if position_value == 0 and mark_price > 0:
                position_value = size * mark_price
            
            # Get unrealized P&L (gross, before fees)
            unrealized_pnl_gross = float(pos.get("unrealisedPnl", 0))
            
            # Calculate estimated exit fee (taker: 0.055% of position value)
            # Entry fee was already paid, exit fee will be paid when closing
            estimated_exit_fee = position_value * 0.00055
            
            # Estimated NET P&L = Gross - Exit Fee
            # Note: Entry fee was already deducted when position was opened
            estimated_net_pnl = unrealized_pnl_gross - estimated_exit_fee
            
            # Get trade mode info for this symbol
            mode_info = trade_mode_info.get(symbol, {})
            
            positions.append({
                "symbol": symbol,
                "side": pos.get("side"),
                "size": size,
                "entryPrice": entry_price,
                "markPrice": mark_price,
                "positionValue": position_value,
                "unrealisedPnl": unrealized_pnl_gross,  # Gross P&L (before exit fee)
                "estimatedNetPnl": round(estimated_net_pnl, 4),  # NET P&L (after estimated exit fee)
                "estimatedExitFee": round(estimated_exit_fee, 4),  # Estimated exit fee
                "leverage": pos.get("leverage"),
                "liquidationPrice": float(pos.get("liqPrice", 0)) if pos.get("liqPrice") else None,
                "takeProfit": pos.get("takeProfit"),
                "stopLoss": pos.get("stopLoss"),
                "createdTime": pos.get("createdTime"),
                "updatedTime": pos.get("updatedTime"),
                "isBreakout": symbol in breakout_symbols,  # Flag for breakout positions
                # Trade mode info (AI selected)
                "tradeMode": mode_info.get("mode", "futures_long"),
                "tradeModeDisplay": mode_info.get("mode_display", f"Futures {pos.get('leverage', 1)}x"),
                "isSpot": mode_info.get("is_spot", False),
                "exchange": mode_info.get("exchange", "bybit"),
            })
    
    # Also fetch SPOT positions from Redis (they don't appear in futures API)
    try:
        spot_positions = await r.hgetall(f"positions:spot:{user_id}")
        for symbol_bytes, data_bytes in spot_positions.items():
            try:
                symbol = symbol_bytes.decode() if isinstance(symbol_bytes, bytes) else symbol_bytes
                data_str = data_bytes.decode() if isinstance(data_bytes, bytes) else data_bytes
                spot_data = json.loads(data_str)
                
                # Get current price for this symbol to calculate P&L
                current_price = spot_data.get("entry_price", 0)
                try:
                    # Try to get current price from tickers
                    client_for_ticker = await get_user_client(user_id, "bybit")
                    if client_for_ticker:
                        ticker_result = await client_for_ticker.get_tickers(symbol=symbol, category="spot")
                        if ticker_result.get("success"):
                            ticker_list = ticker_result.get("data", {}).get("list", [])
                            if ticker_list:
                                current_price = float(ticker_list[0].get("lastPrice", current_price))
                except:
                    pass  # Use entry price if ticker fails
                
                entry_price = float(spot_data.get("entry_price", 0))
                size = float(spot_data.get("size", 0))
                position_value = size * current_price
                
                # Calculate P&L
                pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                unrealized_pnl = size * (current_price - entry_price)
                
                # SPOT has lower fees (0.1% round trip typically)
                estimated_exit_fee = position_value * 0.001
                estimated_net_pnl = unrealized_pnl - estimated_exit_fee
                
                positions.append({
                    "symbol": symbol,
                    "side": spot_data.get("side", "Buy"),
                    "size": size,
                    "entryPrice": entry_price,
                    "markPrice": current_price,
                    "positionValue": position_value,
                    "unrealisedPnl": round(unrealized_pnl, 4),
                    "estimatedNetPnl": round(estimated_net_pnl, 4),
                    "estimatedExitFee": round(estimated_exit_fee, 4),
                    "leverage": 1,
                    "liquidationPrice": None,  # No liquidation for SPOT
                    "takeProfit": spot_data.get("take_profit_price"),
                    "stopLoss": spot_data.get("stop_loss_price"),
                    "createdTime": spot_data.get("entry_time"),
                    "updatedTime": None,
                    "isBreakout": symbol in breakout_symbols,
                    "tradeMode": "spot",
                    "tradeModeDisplay": spot_data.get("trade_mode_display", "SPOT (no leverage)"),
                    "isSpot": True,
                    "exchange": "bybit",
                    # Extra info for SPOT display
                    "pnlPercent": round(pnl_percent, 2),
                })
                logger.debug(f"Added SPOT position to response: {symbol} | PnL: {pnl_percent:.2f}%")
            except Exception as e:
                logger.warning(f"Failed to parse SPOT position {symbol_bytes}: {e}")
    except Exception as e:
        logger.warning(f"Failed to fetch SPOT positions: {e}")
    
    # Sort by createdTime (oldest first) for stable display order
    positions.sort(key=lambda x: x.get("createdTime") or "0")
            
    return {
        "success": True,
        "data": {"positions": positions}
    }


@router.get("/trade-history/{symbol}")
async def get_trade_history(symbol: str, user_id: str = "default", limit: int = 50):
    """Get trade history for a specific symbol
    
    Returns all trade events (opens and closes) for a symbol, 
    useful for showing history when user clicks on a position.
    """
    try:
        r = await get_redis()
        
        # Fetch per-symbol history from Redis
        history_key = f'trade_history:{user_id}:{symbol}'
        raw_history = await r.lrange(history_key, 0, limit - 1)
        
        history = []
        for item in raw_history:
            try:
                data = json.loads(item.decode() if isinstance(item, bytes) else item)
                history.append(data)
            except Exception as e:
                logger.warning(f"Failed to parse history item: {e}")
        
        # Also get completed trades for this symbol (for historical P&L)
        completed_key = f'trades:completed:{user_id}'
        raw_completed = await r.lrange(completed_key, 0, 99)
        
        completed_trades = []
        for item in raw_completed:
            try:
                data = json.loads(item.decode() if isinstance(item, bytes) else item)
                if data.get('symbol') == symbol:
                    completed_trades.append(data)
            except:
                pass
        
        # Calculate stats for this symbol
        total_pnl = sum(t.get('pnl', 0) for t in completed_trades)
        total_trades = len(completed_trades)
        wins = sum(1 for t in completed_trades if t.get('pnl', 0) > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "history": history,  # All trade events (opens/closes)
                "completedTrades": completed_trades,  # Just closed trades
                "stats": {
                    "totalTrades": total_trades,
                    "totalPnl": round(total_pnl, 2),
                    "winRate": round(win_rate, 1),
                    "wins": wins,
                    "losses": total_trades - wins
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get trade history for {symbol}: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/klines/{symbol}")
async def get_klines(symbol: str, interval: str = "5", limit: int = 30, user_id: str = "default"):
    """Get kline/candlestick data for sparkline charts"""
    
    client = await get_user_client(user_id, "bybit")
    if not client:
        return {"success": False, "error": "No exchange connected"}
    
    try:
        result = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        if not result.get("success"):
            return {"success": False, "error": "Failed to fetch klines"}
        
        kline_list = result.get("data", {}).get("list", [])
        
        # Parse klines: [timestamp, open, high, low, close, volume, turnover]
        # Reverse to chronological order (Bybit returns newest first)
        prices = []
        for kline in reversed(kline_list):
            prices.append({
                "time": int(kline[0]),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            })
        
        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "interval": interval,
                "prices": prices
            }
        }
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/klines-batch")
async def get_klines_batch(symbols: str, interval: str = "5", limit: int = 20, user_id: str = "default"):
    """Get klines for multiple symbols at once (comma-separated)"""
    
    client = await get_user_client(user_id, "bybit")
    if not client:
        return {"success": False, "error": "No exchange connected"}
    
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    results = {}
    
    for symbol in symbol_list[:20]:  # Limit to 20 symbols
        try:
            result = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            if result.get("success"):
                kline_list = result.get("data", {}).get("list", [])
                prices = []
                for kline in reversed(kline_list):
                    prices.append({
                        "time": int(kline[0]),
                        "close": float(kline[4])  # Just close price for sparklines
                    })
                results[symbol] = prices
        except Exception as e:
            logger.warning(f"Failed to get klines for {symbol}: {e}")
            results[symbol] = []
    
    return {
        "success": True,
        "data": results
    }


@router.get("/pnl")
async def get_pnl_history(days: int = 7, limit: int = 200):
    """
    Get real closed PnL history for last N days.
    Rolling window: shows stats for last 7 days by default.
    Max 200 trades per request (Bybit limit).
    """
    
    if "default" not in exchange_connections:
        return {"success": False, "error": "No exchange connected"}
        
    client = exchange_connections["default"]
    result = await client.get_pnl(limit=limit, days=days)
    
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
            "winRate": (winning / len(pnl_list) * 100) if pnl_list else 0,
            "period": f"Last {days} days",
            "tradeCount": len(pnl_list)
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
    """Pause autonomous trading - stops opening NEW positions, monitors existing ones"""
    trader = get_trader()
    await trader.pause_trading(user_id)
    
    return {
        "success": True,
        "message": "Trading paused - no new positions will be opened, existing positions are still monitored"
    }


@router.post("/trading/resume")
async def resume_autonomous_trading(user_id: str = "default"):
    """Resume autonomous trading - starts opening new positions again"""
    trader = get_trader()
    
    # If user is not connected, try to connect them first
    if user_id not in trader.user_clients:
        logger.info(f"User {user_id} not connected, attempting to connect...")
        
        r = await get_redis()
        api_key = None
        api_secret = None
        is_testnet = False
        
        # Try multiple credential sources in order:
        # 1. User-specific credentials (new format)
        creds = await r.hgetall(f"user:{user_id}:exchange:bybit")
        if creds:
            api_key = simple_decrypt(creds.get(b"api_key", b"").decode())
            api_secret = simple_decrypt(creds.get(b"api_secret", b"").decode())
            is_testnet = creds.get(b"is_testnet", b"0").decode() == "1"
            logger.info(f"Found credentials for user:{user_id}:exchange:bybit")
        
        # 2. Try user:default:exchange:bybit (admin might use this)
        if not api_key:
            creds = await r.hgetall("user:default:exchange:bybit")
            if creds:
                api_key = simple_decrypt(creds.get(b"api_key", b"").decode())
                api_secret = simple_decrypt(creds.get(b"api_secret", b"").decode())
                is_testnet = creds.get(b"is_testnet", b"0").decode() == "1"
                logger.info("Found credentials for user:default:exchange:bybit")
        
        # 3. Try legacy format (exchange:credentials:default)
        if not api_key:
            legacy_creds = await r.hgetall("exchange:credentials:default")
            if legacy_creds:
                api_key = simple_decrypt(legacy_creds.get(b"api_key", b"").decode())
                api_secret = simple_decrypt(legacy_creds.get(b"api_secret", b"").decode())
                is_testnet = legacy_creds.get(b"testnet", b"0").decode() == "1"
                logger.info("Found credentials in legacy exchange:credentials:default")
        
        # 4. Try trading:enabled:{user_id} (old auto-reconnect format)
        if not api_key:
            enabled_creds = await r.hgetall(f"trading:enabled:{user_id}")
            if enabled_creds:
                api_key = simple_decrypt(enabled_creds.get(b"api_key", b"").decode())
                api_secret = simple_decrypt(enabled_creds.get(b"api_secret", b"").decode())
                logger.info(f"Found credentials in trading:enabled:{user_id}")
        
        if api_key and api_secret:
            connected = await trader.connect_user(user_id, api_key, api_secret, is_testnet)
            if connected:
                logger.info(f"User {user_id} auto-connected on resume")
            else:
                return {"success": False, "error": "Failed to connect - check API credentials"}
        else:
            logger.error(f"No credentials found for user {user_id}")
            return {"success": False, "error": "No credentials found - please connect exchange first"}
    
    # Now resume trading
    await trader.resume_trading(user_id)
    
    # Mark that this user has started trading at least once
    r = await get_redis()
    await r.set(f'trading:started:{user_id}', '1')
    
    return {
        "success": True,
        "message": "Trading resumed - new positions will be opened"
    }


@router.get("/trading/status")
async def get_trading_status(user_id: str = "default"):
    """Get autonomous trading status"""
    trader = get_trader()
    r = await get_redis()
    
    # Check if user SHOULD be trading (started before and not paused)
    has_started = await r.exists(f'trading:started:{user_id}')
    is_paused_in_redis = await r.exists(f'trading:paused:{user_id}')
    
    # User is currently connected to trader
    is_connected = user_id in trader.user_clients
    
    # User SHOULD be actively trading if: started before AND not paused
    should_be_trading = has_started and not is_paused_in_redis
    
    # If user should be trading but isn't connected, try to auto-reconnect
    if should_be_trading and not is_connected:
        logger.info(f"User {user_id} should be trading but not connected - attempting auto-reconnect")
        try:
            # Try to get credentials and reconnect
            api_key = None
            api_secret = None
            is_testnet = False
            
            # Check various credential sources
            user_creds = await r.hgetall(f'user:{user_id}:exchange:bybit')
            if user_creds:
                api_key = simple_decrypt(user_creds.get(b'api_key', b'').decode())
                api_secret = simple_decrypt(user_creds.get(b'api_secret', b'').decode())
                is_testnet = user_creds.get(b'testnet', b'0').decode() == '1'
            
            if not api_key:
                enabled_creds = await r.hgetall(f'trading:enabled:{user_id}')
                if enabled_creds:
                    api_key = simple_decrypt(enabled_creds.get(b'api_key', b'').decode())
                    api_secret = simple_decrypt(enabled_creds.get(b'api_secret', b'').decode())
            
            if api_key and api_secret:
                connected = await trader.connect_user(user_id, api_key, api_secret, is_testnet)
                if connected:
                    is_connected = True
                    logger.info(f"Auto-reconnected user {user_id} on status check")
        except Exception as e:
            logger.error(f"Failed to auto-reconnect {user_id}: {e}")
    
    # Final state
    is_paused = trader.is_paused(user_id) if hasattr(trader, 'is_paused') else is_paused_in_redis
    is_actively_trading = is_connected and not is_paused
    
    # Get recent trades
    trades = []
    if hasattr(trader, 'redis_client') and trader.redis_client:
        trade_data = await trader.redis_client.lrange(f"trades:completed:{user_id}", 0, 9)
        trades = [json.loads(t) for t in trade_data]
    
    # Get trading pairs (different attribute in v2)
    trading_pairs = getattr(trader, 'trading_pairs', [])
    max_positions = getattr(trader, 'max_open_positions', 0)  # 0 = unlimited
    min_conf = getattr(trader, 'min_confidence', 55)
    risk_mode = getattr(trader, 'risk_mode', 'normal')
    trail_percent = getattr(trader, 'trail_from_peak', 1.0)
    
    # Get total pairs from market scanner
    total_pairs = 0
    if hasattr(trader, 'market_scanner') and trader.market_scanner:
        total_pairs = len(getattr(trader.market_scanner, 'all_symbols', []))
    
    # Fallback: check Redis for stored count
    if total_pairs == 0 and hasattr(trader, 'redis_client') and trader.redis_client:
        try:
            symbols_str = await trader.redis_client.get('trading:available_symbols')
            if symbols_str:
                total_pairs = len(symbols_str.split(','))
        except:
            pass
        
    # Get current regime from Redis
    current_regime = "analyzing"
    ai_confidence = 0
    pairs_scanned = total_pairs
    
    if hasattr(trader, 'redis_client') and trader.redis_client:
        try:
            regime_data = await trader.redis_client.get('bot:current_regime')
            if regime_data:
                current_regime = regime_data.decode() if isinstance(regime_data, bytes) else regime_data
            
            # Get AI confidence from stats
            stats_data = await trader.redis_client.get('trader:stats')
            if stats_data:
                stats = json.loads(stats_data)
                total_trades = stats.get('total_trades', 0)
                winning = stats.get('winning_trades', 0)
                if total_trades > 10:
                    ai_confidence = int((winning / total_trades) * 100)
                else:
                    ai_confidence = min_conf
            
            # Get opportunities scanned
            opp_count = await trader.redis_client.get('bot:opportunities_scanned')
            if opp_count:
                pairs_scanned = int(opp_count)
        except:
            pass
    
    return {
        "success": True,
        "data": {
            "is_autonomous_trading": is_actively_trading,
            "is_connected": is_connected,
            "is_paused": is_paused,
            "should_be_trading": should_be_trading,  # For debugging
            "trading_pairs": trading_pairs if is_connected else [],
            "total_pairs": total_pairs,
            "pairs_scanned": pairs_scanned,
            "max_positions": max_positions,
            "min_confidence": min_conf,
            "ai_confidence": ai_confidence,
            "current_regime": current_regime,
            "risk_mode": risk_mode,
            "trailing_stop_percent": trail_percent,
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
    
    # Get total pairs from market scanner
    total_pairs = 0
    if hasattr(trader, 'market_scanner') and trader.market_scanner:
        total_pairs = len(getattr(trader.market_scanner, 'all_symbols', []))
    
    # Fallback: check Redis for stored count
    if total_pairs == 0 and hasattr(trader, 'redis_client') and trader.redis_client:
        try:
            symbols_str = await trader.redis_client.get('trading:available_symbols')
            if symbols_str:
                total_pairs = len(symbols_str.split(','))
        except:
            pass
    
    activity = {
        "is_running": trader.is_running,
        "is_user_connected": user_id in trader.user_clients,
        "total_pairs_monitoring": total_pairs if total_pairs > 0 else len(trading_pairs),
        "active_trades": [],
        "recent_completed": [],
        "bot_actions": [],
        "current_analysis": None,
        "version": "v2" if USE_V2_TRADER else "v1"
    }
    
    if not hasattr(trader, 'redis_client') or not trader.redis_client:
        return {"success": True, "data": activity}
    
    try:
        # V2: Get active trades directly from trader's active_positions
        if USE_V2_TRADER and hasattr(trader, 'active_positions'):
            user_positions = trader.active_positions.get(user_id, {})
            for symbol, pos in user_positions.items():
                pnl_percent = 0
                # Try to calculate current P&L
                try:
                    ticker_data = getattr(trader, '_last_ticker_data', {}).get(symbol, {})
                    current_price = float(ticker_data.get('lastPrice', pos.entry_price))
                    if pos.side == 'Buy':
                        pnl_percent = ((current_price - pos.entry_price) / pos.entry_price) * 100
                    else:
                        pnl_percent = ((pos.entry_price - current_price) / pos.entry_price) * 100
                except:
                    pass
                    
                activity["active_trades"].append({
                    "symbol": symbol,
                    "side": pos.side.lower(),
                    "entry_price": pos.entry_price,
                    "size": pos.size,
                    "value": pos.position_value,
                    "pnl_percent": round(pnl_percent, 2),
                    "confidence": pos.entry_confidence,
                    "edge": pos.entry_edge,
                    "regime": pos.entry_regime,
                    "trailing_active": pos.trailing_active,
                    "peak_pnl": round(pos.peak_pnl_percent, 2),
                    "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                    "strategy": f"Edge: {pos.entry_edge:.2f}"
                })
        else:
            # Fallback to Redis
            active_trades = await trader.redis_client.lrange(f"trades:active:{user_id}", 0, -1)
            activity["active_trades"] = [json.loads(t) for t in active_trades]
        
        # Get recent completed trades from Redis
        completed = await trader.redis_client.lrange(f"trades:completed:{user_id}", 0, 9)
        activity["recent_completed"] = [json.loads(t) for t in completed]
        
        # Fallback: get from trading:events if no completed trades found
        if not activity["recent_completed"]:
            events = await trader.redis_client.lrange('trading:events', 0, 49)
            for event in events:
                try:
                    e = json.loads(event)
                    if e.get('action') == 'closed':
                        activity["recent_completed"].append({
                            "symbol": e.get('symbol'),
                            "pnl": e.get('pnl_value', 0),
                            "pnl_percent": e.get('pnl_percent', 0),
                            "close_reason": e.get('reason', 'Unknown'),
                            "closed_time": e.get('timestamp')
                        })
                except:
                    pass
            activity["recent_completed"] = activity["recent_completed"][:10]
        
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
        
        # V2 specific: Get trader stats and current settings
        if USE_V2_TRADER:
            stats = await trader.get_status()
            activity["trader_stats"] = stats.get('stats', {})
            
            # Add current settings for display
            activity["current_settings"] = {
                "risk_mode": getattr(trader, 'risk_mode', 'normal'),
                "take_profit": getattr(trader, 'take_profit', 3.0),
                "stop_loss": getattr(trader, 'emergency_stop_loss', 1.5),
                "trailing_stop": getattr(trader, 'trail_from_peak', 0.14),
                "min_profit_to_trail": getattr(trader, 'min_profit_to_trail', 0.5),
                "max_positions": getattr(trader, 'max_open_positions', 0)
            }
            
    except Exception as e:
        logger.error(f"Error getting activity: {e}")
        
    return {"success": True, "data": activity}


@router.get("/trading/console")
async def get_realtime_console(user_id: str = "default"):
    """Get REAL-TIME console output - what bot is doing RIGHT NOW"""
    
    trader = get_trader()
    
    console = {
        "timestamp": datetime.utcnow().isoformat(),
        "logs": [],
        "current_action": None,
        "scanning": {
            "pairs_scanned": 0,
            "opportunities_found": 0,
            "last_scan_time": None
        },
        "decisions": [],
        "ai_models": {
            "regime": None,
            "sentiment": None,
            "edge_score": None
        }
    }
    
    try:
        if not hasattr(trader, 'redis_client') or not trader.redis_client:
            return {"success": True, "data": console}
        
        # Get ONLY user-specific log entries - NO FALLBACK TO GLOBAL!
        log_entries = await trader.redis_client.lrange(f'bot:console:logs:{user_id}', 0, 49)
        
        # NO FALLBACK - each user sees ONLY their own logs!
        console["logs"] = [json.loads(l) for l in log_entries] if log_entries else []
        
        # Get current action
        current_action = await trader.redis_client.get('bot:current_action')
        console["current_action"] = current_action.decode() if current_action else "Monitoring markets..."
        
        # Get scanning stats
        scan_stats = await trader.redis_client.hgetall('bot:scan_stats')
        if scan_stats:
            console["scanning"] = {
                k.decode(): v.decode() for k, v in scan_stats.items()
            }
        
        # Fallback: get pairs count from market_scanner if not in Redis
        if console["scanning"].get("pairs_scanned", 0) == 0:
            if hasattr(trader, 'market_scanner') and trader.market_scanner:
                console["scanning"]["pairs_scanned"] = len(getattr(trader.market_scanner, 'all_symbols', []))
                console["scanning"]["last_scan_time"] = datetime.utcnow().isoformat()
            else:
                # Try Redis stored symbols
                symbols_str = await trader.redis_client.get('trading:available_symbols')
                if symbols_str:
                    console["scanning"]["pairs_scanned"] = len(symbols_str.split(','))
        
        # Get recent decisions
        decisions = await trader.redis_client.lrange('bot:decisions', 0, 9)
        console["decisions"] = [json.loads(d) for d in decisions] if decisions else []
        
        # Get AI model status
        regime = await trader.redis_client.get('bot:current_regime')
        console["ai_models"]["regime"] = regime.decode() if regime else "Unknown"
        
        sentiment_data = await trader.redis_client.get('market:sentiment:cryptobert')
        if sentiment_data:
            try:
                sent = json.loads(sentiment_data)
                console["ai_models"]["sentiment"] = sent.get('overall_score', 0)
            except:
                console["ai_models"]["sentiment"] = 0
        
        # Get trader stats from v2 if available
        if USE_V2_TRADER:
            stats = await trader.get_status()
            console["trader_stats"] = stats.get('stats', {})
            
    except Exception as e:
        logger.error(f"Error getting console: {e}")
        console["logs"].append({
            "time": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "message": f"Console error: {str(e)}"
        })
        
    return {"success": True, "data": console}


@router.get("/trading/pairs")
async def get_trading_pairs():
    """Get ALL crypto pairs from Bybit that bot can trade"""
    import httpx
    
    try:
        # Get ALL pairs directly from Bybit API
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://api.bybit.com/v5/market/instruments-info",
                params={'category': 'linear', 'limit': 1000}
            )
            
            if response.status_code == 200:
                data = response.json()
                instruments = data.get('result', {}).get('list', [])
                
                # Get ALL USDT pairs
                all_pairs = [inst['symbol'] for inst in instruments if inst['symbol'].endswith('USDT')]
                
                # Categorize
                defi_pairs = ['AAVEUSDT', 'MKRUSDT', 'UNIUSDT', 'CRVUSDT', 'SNXUSDT', 'COMPUSDT', 
                             '1INCHUSDT', 'YFIUSDT', 'SUSHIUSDT', 'DYDXUSDT', 'GMXUSDT', 'PENDLEUSDT']
                meme_pairs = ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT', 
                             'BOMEUSDT', 'MEMEUSDT', 'NOTUSDT', 'NEIROUSDT']
                ai_pairs = ['FETUSDT', 'AGIXUSDT', 'OCEANUSDT', 'RNDRUSDT', 'TAOUSDT', 'WLDUSDT', 
                           'AKTUSDT', 'ARKMUSDT', 'AIUSDT']
                layer1 = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOTUSDT', 
                         'ATOMUSDT', 'NEARUSDT', 'SUIUSDT', 'APTUSDT', 'SEIUSDT', 'TIAUSDT']
                layer2 = ['ARBUSDT', 'OPUSDT', 'MATICUSDT', 'STRKUSDT', 'MANTAUSDT', 'ZKUSDT', 
                         'BLASTUSDT', 'SCROLLUSDT', 'LINEAUSDT', 'METISUSDT']
                
                return {
                    "success": True,
                    "data": {
                        "pairs": all_pairs,
                        "total": len(all_pairs),
                        "version": "v2" if USE_V2_TRADER else "v1",
                        "source": "bybit_api_live",
                        "categories": {
                            "layer1": [p for p in all_pairs if p in layer1],
                            "layer2": [p for p in all_pairs if p in layer2],
                            "defi": [p for p in all_pairs if p in defi_pairs],
                            "meme": [p for p in all_pairs if p in meme_pairs],
                            "ai": [p for p in all_pairs if p in ai_pairs],
                            "top10_by_volume": all_pairs[:10]  # Usually sorted by volume
                        }
                    }
                }
    except Exception as e:
        logger.error(f"Error getting pairs from Bybit: {e}")
    
    # Fallback to trader's list
    trader = get_trader()
    trading_pairs = getattr(trader, 'trading_pairs', [])
    
    return {
        "success": True,
        "data": {
            "pairs": trading_pairs,
            "total": len(trading_pairs),
            "version": "v2" if USE_V2_TRADER else "v1",
            "source": "fallback",
            "categories": {}
        }
    }


# ============================================
# BOT SETTINGS
# ============================================

@router.get("/settings")
async def get_settings(user_id: str = "default"):
    """Get current bot settings for specific user"""
    
    r = await redis.from_url(settings.REDIS_URL)
    
    # Settings key is PER-USER (not global!)
    settings_key = f'bot:settings:{user_id}'
    logger.info(f"Loading settings for user: {user_id} from key: {settings_key}")
    
    # Default settings
    defaults = {
        # === AI FULL AUTO MODE ===
        "aiFullAuto": False,  # When ON: AI manages everything (strategy, positions, risk)
        "useMaxTradeTime": True,  # Use preset's max_trade_minutes (can disable if aiFullAuto is OFF)
        
        "riskMode": "micro",  # Default to MICRO (best preset)
        "strategyPreset": "micro",  # scalp, micro, swing, conservative, balanced, aggressive
        "takeProfitPercent": 0.9,  # MICRO default
        "stopLossPercent": 0.5,  # MICRO default
        "trailingStopPercent": 0.14,  # MICRO default
        "minProfitToTrail": 0.45,  # MICRO default
        "minConfidence": 65,
        "minEdge": 0.15,
        "maxPositionPercent": 5,
        "maxOpenPositions": 0,  # 0 = unlimited (AI decides if aiFullAuto)
        "maxDailyDrawdown": 0,  # 0 = OFF (no daily limit)
        "maxTotalExposure": 100,  # 100% = can use entire budget
        "leverageMode": "auto",  # 1x, 2x, 3x, 5x, 10x, auto
        "cryptoBudget": 100,
        "tradFiBudget": 0,
        "enableCrypto": True,
        "enableTradFi": False,
        "useAiSignals": True,
        "learnFromTrades": True,
        "useRegimeDetection": True,
        "useEdgeEstimation": True,
        "useDynamicSizing": True,
        # V3 AI Models
        "useCryptoBert": True,
        "useXgboostClassifier": True,
        "usePricePredictor": True,
        "useWhaleDetection": True,
        "useFundingRate": True,
        "usePatternRecognition": True,
        "useQLearning": True,
        # Smart exit (MICRO PROFIT)
        "breakevenTrigger": 0.3,
        "partialExitTrigger": 0.4,
        "partialExitPercent": 50,
        "useSmartExit": False,
        "momentumThreshold": 0.05,
        # Kelly Criterion
        "kellyEnabled": False,  # OFF = equal sizing per trade
        "kellyMultiplier": 0.5,
        # Breakout settings
        "enableBreakout": False,  # Enable/disable breakout trading
        "breakoutExtraSlots": False,  # Allow +2 positions for breakouts
    }
    
    try:
        settings_data = await r.hgetall(settings_key)
        
        if settings_data:
            parsed = {
                k.decode() if isinstance(k, bytes) else k: 
                v.decode() if isinstance(v, bytes) else v 
                for k, v in settings_data.items()
            }
            logger.info(f"Loaded {len(parsed)} settings for user {user_id}")
            
            # Convert string values to proper types
            return {
                "success": True,
                "data": {
                    # AI Full Auto Mode
                    "aiFullAuto": parsed.get('aiFullAuto', 'false') == 'true',
                    "useMaxTradeTime": parsed.get('useMaxTradeTime', 'true') == 'true',
                    
                    "strategyPreset": parsed.get('strategyPreset', defaults['strategyPreset']),
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
                    "leverageMode": parsed.get('leverageMode', defaults['leverageMode']),
                    "cryptoBudget": float(parsed.get('cryptoBudget', defaults['cryptoBudget'])),
                    "tradFiBudget": float(parsed.get('tradFiBudget', defaults['tradFiBudget'])),
                    "enableCrypto": parsed.get('enableCrypto', 'true') == 'true',
                    "enableTradFi": parsed.get('enableTradFi', 'false') == 'true',
                    "useAiSignals": parsed.get('useAiSignals', 'true') == 'true',
                    "learnFromTrades": parsed.get('learnFromTrades', 'true') == 'true',
                    "useRegimeDetection": parsed.get('useRegimeDetection', 'true') == 'true',
                    "useEdgeEstimation": parsed.get('useEdgeEstimation', 'true') == 'true',
                    "useDynamicSizing": parsed.get('useDynamicSizing', 'true') == 'true',
                    # V3 AI Models
                    "useCryptoBert": parsed.get('useCryptoBert', 'true') == 'true',
                    "useXgboostClassifier": parsed.get('useXgboostClassifier', 'true') == 'true',
                    "usePricePredictor": parsed.get('usePricePredictor', 'true') == 'true',
                    "useWhaleDetection": parsed.get('useWhaleDetection', 'true') == 'true',
                    "useFundingRate": parsed.get('useFundingRate', 'true') == 'true',
                    "usePatternRecognition": parsed.get('usePatternRecognition', 'true') == 'true',
                    "useQLearning": parsed.get('useQLearning', 'true') == 'true',
                    # Smart exit (MICRO PROFIT)
                    "breakevenTrigger": float(parsed.get('breakevenTrigger', defaults['breakevenTrigger'])),
                    "partialExitTrigger": float(parsed.get('partialExitTrigger', defaults['partialExitTrigger'])),
                    "partialExitPercent": float(parsed.get('partialExitPercent', defaults['partialExitPercent'])),
                    "useSmartExit": parsed.get('useSmartExit', 'false') == 'true',
                    "momentumThreshold": float(parsed.get('momentumThreshold', defaults['momentumThreshold'])),
                    # Kelly Criterion
                    "kellyEnabled": parsed.get('kellyEnabled', 'false') == 'true',
                    "kellyMultiplier": float(parsed.get('kellyMultiplier', 0.5)),
                    # Breakout settings
                    "enableBreakout": parsed.get('enableBreakout', 'false') == 'true',
                    "breakoutExtraSlots": parsed.get('breakoutExtraSlots', 'false') == 'true',
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
async def save_settings(request: Request, user_id: str = "default"):
    """Save bot settings for specific user and apply them to trader"""
    
    body = await request.json()
    
    # Settings key is PER-USER (not global!)
    settings_key = f'bot:settings:{user_id}'
    logger.info(f"Saving settings for user: {user_id} to key: {settings_key}")
    
    r = await redis.from_url(settings.REDIS_URL)
    
    try:
        # Save to Redis - all settings
        settings_to_save = {
            # === AI FULL AUTO MODE ===
            'aiFullAuto': str(body.get('aiFullAuto', False)).lower(),
            'useMaxTradeTime': str(body.get('useMaxTradeTime', True)).lower(),
            
            # Strategy preset (applies defaults for selected strategy)
            'strategyPreset': str(body.get('strategyPreset', 'micro')),
            
            # Risk mode
            'riskMode': str(body.get('riskMode', 'micro')),
            
            # Trading parameters (can override preset values)
            'takeProfitPercent': str(body.get('takeProfitPercent', 3.0)),
            'stopLossPercent': str(body.get('stopLossPercent', 1.5)),
            'trailingStopPercent': str(body.get('trailingStopPercent', 0.8)),
            'minProfitToTrail': str(body.get('minProfitToTrail', 0.5)),
            'minConfidence': str(body.get('minConfidence', 60)),
            'minEdge': str(body.get('minEdge', 0.15)),
            
            # Risk management
            'maxPositionPercent': str(body.get('maxPositionPercent', 5)),
            'maxOpenPositions': str(body.get('maxOpenPositions', 0)),  # 0 = unlimited
            'maxDailyDrawdown': str(body.get('maxDailyDrawdown', 3)),
            'maxTotalExposure': str(body.get('maxTotalExposure', 100)),  # 100% = can use entire budget
            
            # Leverage
            'leverageMode': str(body.get('leverageMode', 'auto')),  # 1x, 2x, 3x, 5x, 10x, auto
            
            # Budget allocation
            'cryptoBudget': str(body.get('cryptoBudget', 100)),
            'tradFiBudget': str(body.get('tradFiBudget', 0)),
            'enableCrypto': str(body.get('enableCrypto', True)).lower(),
            'enableTradFi': str(body.get('enableTradFi', False)).lower(),
            
            # AI features (accept both old and new field names)
            'useAiSignals': str(body.get('useAiSignals', True)).lower(),
            'learnFromTrades': str(body.get('learnFromTrades', True)).lower(),
            'useRegimeDetection': str(body.get('enableRegimeDetection', body.get('useRegimeDetection', True))).lower(),
            'useEdgeEstimation': str(body.get('enableEdgeEstimation', body.get('useEdgeEstimation', True))).lower(),
            'useDynamicSizing': str(body.get('enableDynamicSizing', body.get('useDynamicSizing', True))).lower(),
            
            # V3 AI Models
            'useCryptoBert': str(body.get('useSentimentAnalysis', body.get('useCryptoBert', True))).lower(),
            'useXgboostClassifier': str(body.get('useXGBoost', body.get('useXgboostClassifier', True))).lower(),
            'usePricePredictor': str(body.get('enablePricePrediction', body.get('usePricePredictor', True))).lower(),
            'useWhaleDetection': str(body.get('useWhaleDetection', True)).lower(),
            'useFundingRate': str(body.get('useFundingRate', True)).lower(),
            'usePatternRecognition': str(body.get('usePatternRecognition', True)).lower(),
            'useQLearning': str(body.get('useQLearning', True)).lower(),
            
            # Smart exit (MICRO PROFIT)
            'breakevenTrigger': str(body.get('breakevenTrigger', 0.3)),
            'partialExitTrigger': str(body.get('partialExitTrigger', 0.4)),
            'partialExitPercent': str(body.get('partialExitPercent', 50)),
            'useSmartExit': str(body.get('useSmartExit', False)).lower(),
            'momentumThreshold': str(body.get('momentumThreshold', 0.05)),
            
            # Kelly Criterion
            'kellyEnabled': str(body.get('kellyEnabled', False)).lower(),
            'kellyMultiplier': str(body.get('kellyMultiplier', 0.5)),
            
            # Breakout settings
            'enableBreakout': str(body.get('enableBreakout', False)).lower(),
            'breakoutExtraSlots': str(body.get('breakoutExtraSlots', False)).lower(),
        }
        
        # Save to user-specific key
        await r.hset(settings_key, mapping=settings_to_save)
        logger.info(f"Saved {len(settings_to_save)} settings for user {user_id}")
        
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
        
        # Also update position sizer settings
        if hasattr(trader, 'position_sizer') and trader.position_sizer:
            trader.position_sizer.max_position_percent = float(body.get('maxPositionPercent', 5))
            trader.position_sizer.max_exposure_percent = float(body.get('maxTotalExposure', 100))
            trader.position_sizer.max_daily_drawdown = float(body.get('maxDailyDrawdown', 3))
            trader.position_sizer.max_open_positions = int(body.get('maxOpenPositions', 0))
        
        # Update leverage mode
        leverage_mode = body.get('leverageMode', 'auto')
        if hasattr(trader, 'leverage_mode'):
            trader.leverage_mode = leverage_mode
        if hasattr(trader, 'position_sizer') and trader.position_sizer:
            trader.position_sizer.leverage_mode = leverage_mode
        
        logger.info("=" * 50)
        logger.info(f"SETTINGS UPDATED INSTANTLY!")
        logger.info(f"   Mode: {body.get('riskMode')} | Leverage: {leverage_mode}")
        logger.info(f"   TP: {body.get('takeProfitPercent')}% | SL: {body.get('stopLossPercent')}%")
        logger.info(f"   Min Edge: {body.get('minEdge')} | Min Conf: {body.get('minConfidence')}%")
        logger.info(f"   Max Positions: {body.get('maxOpenPositions')} | Max Exposure: {body.get('maxTotalExposure')}%")
        logger.info("=" * 50)
        
        return {"success": True, "message": "Settings saved and applied instantly"}
        
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
async def close_single_position(symbol: str, user_id: str = "default"):
    """Close a specific position by symbol (FUTURES or SPOT)"""
    
    # Get user-specific client - NO FALLBACK TO OTHER USERS!
    client = await get_user_client(user_id, "bybit")
    
    # NO FALLBACK! Each user MUST have their own connection!
    if not client:
        return {"success": False, "error": "No exchange connected for this user"}
    
    try:
        r = await get_redis()
        
        # === FIRST: Check if it's a SPOT position (stored in Redis) ===
        spot_data = await r.hget(f"positions:spot:{user_id}", symbol)
        
        if spot_data:
            # This is a SPOT position - need to SELL the coins
            try:
                spot_info = json.loads(spot_data.decode() if isinstance(spot_data, bytes) else spot_data)
                size = float(spot_info.get('size', 0))
                entry_price = float(spot_info.get('entry_price', 0))
                
                if size <= 0:
                    return {"success": False, "error": f"Invalid SPOT position size for {symbol}"}
                
                logger.info(f"CLOSING SPOT POSITION: {symbol} | Size: {size} | Entry: ${entry_price}")
                
                # Get current price for P&L calculation
                current_price = entry_price
                try:
                    ticker_result = await client.get_tickers(symbol=symbol, category="spot")
                    if ticker_result.get("success"):
                        ticker_list = ticker_result.get("data", {}).get("list", [])
                        if ticker_list:
                            current_price = float(ticker_list[0].get("lastPrice", entry_price))
                except:
                    pass
                
                # Calculate P&L
                pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                pnl_value = size * (current_price - entry_price)
                
                # Place SPOT sell order
                order_result = await client.place_spot_order(
                    symbol=symbol,
                    side='Sell',
                    order_type='Market',
                    qty=str(size)
                )
                
                if order_result.get('success'):
                    logger.info(f"SPOT CLOSE SUCCESS: {symbol} sold {size} coins | PnL: {pnl_percent:+.2f}% (${pnl_value:+.2f})")
                    
                    # === STORE IN RECENT TRADES (so manual close counts!) ===
                    try:
                        trade_data = {
                            'symbol': symbol,
                            'pnl': round(pnl_value, 2),
                            'pnl_percent': round(pnl_percent, 2),
                            'close_reason': 'Manual close (SPOT)',
                            'closed_time': datetime.utcnow().isoformat()
                        }
                        await r.lpush(f'trades:completed:{user_id}', json.dumps(trade_data))
                        await r.ltrim(f'trades:completed:{user_id}', 0, 49)
                        
                        # Also store in symbol history
                        history_entry = {
                            'action': 'closed',
                            'timestamp': datetime.utcnow().isoformat(),
                            'side': 'Sell',
                            'entry_price': entry_price,
                            'size': size,
                            'position_value': size * entry_price,
                            'leverage': 1,
                            'is_spot': True,
                            'reason': 'Manual close',
                            'pnl_percent': round(pnl_percent, 2),
                            'pnl_value': round(pnl_value, 2),
                        }
                        await r.lpush(f'trade_history:{user_id}:{symbol}', json.dumps(history_entry))
                        await r.ltrim(f'trade_history:{user_id}:{symbol}', 0, 99)
                        
                        # === UPDATE TRADER STATS (Total P&L, Win Rate, Daily P&L) ===
                        stats_key = f'trader:stats:{user_id}'
                        stats_raw = await r.get(stats_key)
                        if stats_raw:
                            stats = json.loads(stats_raw)
                        else:
                            stats = {'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0, 'daily_pnl': 0}
                        
                        stats['total_trades'] = int(stats.get('total_trades', 0)) + 1
                        stats['total_pnl'] = float(stats.get('total_pnl', 0)) + pnl_value
                        stats['daily_pnl'] = float(stats.get('daily_pnl', 0)) + pnl_value
                        if pnl_value > 0:
                            stats['winning_trades'] = int(stats.get('winning_trades', 0)) + 1
                        
                        await r.set(stats_key, json.dumps(stats))
                        logger.info(f"Updated stats: total_pnl={stats['total_pnl']:.2f}, daily_pnl={stats['daily_pnl']:.2f}")
                        
                        logger.info(f"Manual SPOT close recorded: {symbol} {pnl_percent:+.2f}% -> trades:completed:{user_id}")
                    except Exception as store_err:
                        logger.warning(f"Failed to store manual close trade: {store_err}")
                    
                    # Remove from Redis tracking
                    await r.hdel(f"positions:spot:{user_id}", symbol)
                    await r.hdel(f"positions:trade_mode:{user_id}", symbol)
                    await r.delete(f'peak:{user_id}:{symbol}')
                    
                    return {
                        "success": True,
                        "data": {
                            "symbol": symbol,
                            "side": "Sell",
                            "size": size,
                            "pnl": pnl_value,
                            "pnl_percent": pnl_percent,
                            "type": "SPOT",
                            "message": f"SPOT position closed: {pnl_percent:+.2f}%"
                        }
                    }
                else:
                    error_msg = order_result.get('error', 'Failed to sell SPOT coins')
                    logger.error(f"SPOT CLOSE FAILED: {symbol} - {error_msg}")
                    return {"success": False, "error": error_msg}
                    
            except Exception as spot_err:
                logger.error(f"Error closing SPOT position {symbol}: {spot_err}")
                return {"success": False, "error": f"SPOT close error: {str(spot_err)}"}
        
        # === SECOND: Check FUTURES positions ===
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
            return {"success": False, "error": f"No open position found for {symbol} (checked both SPOT and FUTURES)"}
        
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
            
            # === STORE IN RECENT TRADES (so manual close counts!) ===
            try:
                # Get entry price from position for P&L percent calc
                entry_price = float(target_position.get('avgPrice', 0)) or float(target_position.get('entryPrice', 0))
                mark_price = float(target_position.get('markPrice', entry_price))
                pnl_percent = ((mark_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                if side == 'Sell':  # Short position
                    pnl_percent = -pnl_percent
                
                trade_data = {
                    'symbol': symbol,
                    'pnl': round(unrealized_pnl, 2),
                    'pnl_percent': round(pnl_percent, 2),
                    'close_reason': 'Manual close (FUTURES)',
                    'closed_time': datetime.utcnow().isoformat()
                }
                await r.lpush(f'trades:completed:{user_id}', json.dumps(trade_data))
                await r.ltrim(f'trades:completed:{user_id}', 0, 49)
                
                # Also store in symbol history
                history_entry = {
                    'action': 'closed',
                    'timestamp': datetime.utcnow().isoformat(),
                    'side': close_side,
                    'entry_price': entry_price,
                    'size': size,
                    'position_value': float(target_position.get('positionValue', size * entry_price)),
                    'leverage': float(target_position.get('leverage', 1)),
                    'is_spot': False,
                    'reason': 'Manual close',
                    'pnl_percent': round(pnl_percent, 2),
                    'pnl_value': round(unrealized_pnl, 2),
                }
                await r.lpush(f'trade_history:{user_id}:{symbol}', json.dumps(history_entry))
                await r.ltrim(f'trade_history:{user_id}:{symbol}', 0, 99)
                
                # === UPDATE TRADER STATS (Total P&L, Win Rate, Daily P&L) ===
                stats_key = f'trader:stats:{user_id}'
                stats_raw = await r.get(stats_key)
                if stats_raw:
                    stats = json.loads(stats_raw)
                else:
                    stats = {'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0, 'daily_pnl': 0}
                
                stats['total_trades'] = int(stats.get('total_trades', 0)) + 1
                stats['total_pnl'] = float(stats.get('total_pnl', 0)) + unrealized_pnl
                stats['daily_pnl'] = float(stats.get('daily_pnl', 0)) + unrealized_pnl
                if unrealized_pnl > 0:
                    stats['winning_trades'] = int(stats.get('winning_trades', 0)) + 1
                
                await r.set(stats_key, json.dumps(stats))
                logger.info(f"Updated stats: total_pnl={stats['total_pnl']:.2f}, daily_pnl={stats['daily_pnl']:.2f}")
                
                logger.info(f"Manual FUTURES close recorded: {symbol} {pnl_percent:+.2f}% -> trades:completed:{user_id}")
            except Exception as store_err:
                logger.warning(f"Failed to store manual close trade: {store_err}")
            
            # Clear tracking data
            await r.delete(f'peak:{user_id}:{symbol}')
            await r.hdel(f"positions:trade_mode:{user_id}", symbol)
            
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "pnl": unrealized_pnl,
                    "type": "FUTURES",
                    "message": f"Position closed successfully"
                }
            }
        else:
            return {"success": False, "error": order_result.get('error', 'Failed to close position')}
        
    except Exception as e:
        logger.error(f"Close position error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/close-all-positions")
async def close_all_positions(user_id: str = "default"):
    """Close ALL open positions at once"""
    
    # Get user-specific client - NO FALLBACK TO OTHER USERS!
    client = await get_user_client(user_id, "bybit")
    
    # NO FALLBACK! Each user MUST have their own connection!
    if not client:
        return {"success": False, "error": "No exchange connected for this user"}
    
    try:
        # Get current positions
        positions_result = await client.get_positions()
        
        if not positions_result.get('success'):
            return {"success": False, "error": "Failed to get positions"}
        
        positions = positions_result.get('data', {}).get('list', [])
        open_positions = [p for p in positions if float(p.get('size', 0)) > 0]
        
        if not open_positions:
            return {"success": True, "closed": 0, "message": "No open positions to close"}
        
        closed = 0
        errors = []
        total_pnl = 0
        
        r = await redis.from_url(settings.REDIS_URL)
        
        for pos in open_positions:
            symbol = pos.get('symbol')
            size = float(pos.get('size', 0))
            side = pos.get('side')
            unrealized_pnl = float(pos.get('unrealisedPnl', 0))
            
            # Close position (opposite side)
            close_side = 'Sell' if side == 'Buy' else 'Buy'
            
            try:
                order_result = await client.place_order(
                    symbol=symbol,
                    side=close_side,
                    order_type='Market',
                    qty=str(size),
                    reduce_only=True
                )
                
                if order_result.get('success'):
                    closed += 1
                    total_pnl += unrealized_pnl
                    # Clear peak tracking
                    await r.delete(f'peak:default:{symbol}')
                    logger.info(f"SELL ALL: Closed {symbol} {side} size={size} PnL={unrealized_pnl:.2f}")
                else:
                    errors.append(f"{symbol}: {order_result.get('error', 'Unknown error')}")
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
        
        await r.close()
        
        return {
            "success": True,
            "closed": closed,
            "total_pnl": total_pnl,
            "errors": errors if errors else None,
            "message": f"Closed {closed}/{len(open_positions)} positions"
        }
        
    except Exception as e:
        logger.error(f"Close all positions error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/learning/status")
async def get_learning_status():
    """Get learning progress for all AI models - reads REAL data from Redis"""
    
    r = await get_redis()
    
    learning_status = {
        "models": {},
        "overall_progress": 0,
        "recommendations": [],
        "expert_level": "Beginner"
    }
    
    try:
        # 1. TRADER STATS (Primary learning source)
        trader_stats_raw = await r.get('trader:stats')
        total_trades = 0
        winning = 0
        total_pnl = 0
        
        if trader_stats_raw:
            trader_stats = json.loads(trader_stats_raw)
            total_trades = int(float(trader_stats.get('total_trades', 0)))
            winning = int(float(trader_stats.get('winning_trades', 0)))
            total_pnl = float(trader_stats.get('total_pnl', 0))
            max_drawdown = float(trader_stats.get('max_drawdown', 0))
            losing = total_trades - winning
            win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
            
            # EXPERT requires 5000+ trades with 55%+ win rate
            is_expert = total_trades >= 5000 and win_rate >= 55
            is_learning = total_trades >= 100
            trade_progress = min(100, (total_trades / 5000) * 100)  # 5000 trades = 100%
            
            learning_status["models"]["trader"] = {
                "name": "Trade Executor",
                "total_trades": total_trades,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": round(win_rate, 1),
                "total_pnl": round(total_pnl, 2),
                "max_drawdown": round(max_drawdown, 2),
                "progress": round(trade_progress, 1),
                "status": "expert" if is_expert else "learning" if is_learning else "beginner",
                "needed_for_expert": max(0, 5000 - total_trades),
                "description": f"{winning}W / {losing}L ({win_rate:.1f}% win rate)"
            }
        
        # 2. Q-LEARNING STRATEGIES (Read from ai:learning:q_values)
        q_values_raw = await r.get('ai:learning:q_values')
        q_states = 0
        strategies_learned = 0
        
        if q_values_raw:
            try:
                q_values = json.loads(q_values_raw)
                # Count total Q-states across all regimes
                for regime, strategies in q_values.items():
                    if isinstance(strategies, dict):
                        q_states += len([v for v in strategies.values() if abs(v) > 0.1])
                        strategies_learned += len(strategies)
            except:
                pass
        
        # Also try learning:q_values as fallback
        if q_states == 0:
            q_values_raw2 = await r.get('learning:q_values')
            if q_values_raw2:
                try:
                    q_values = json.loads(q_values_raw2)
                    q_states = len(q_values) if isinstance(q_values, dict) else 0
                except:
                    pass
        
        # EXPERT requires 500+ Q-states
        q_progress = min(100, (q_states / 500) * 100)  # 500 Q-states = 100%
        learning_status["models"]["learning_engine"] = {
            "name": "Strategy Learner (Q-Learning)",
            "q_states": q_states,
            "strategies_learned": strategies_learned,
            "progress": round(q_progress, 1),
            "status": "expert" if q_states >= 500 else "learning" if q_states >= 20 else "beginner",
            "needed_for_expert": max(0, 500 - q_states),
            "description": f"{q_states} Q-states, {strategies_learned} strategies"
        }
        
        # 3. PATTERN RECOGNITION
        patterns_raw = await r.get('ai:learning:patterns')
        patterns_count = 0
        if patterns_raw:
            try:
                patterns = json.loads(patterns_raw)
                patterns_count = len(patterns) if isinstance(patterns, dict) else 0
            except:
                pass
        
        # EXPERT requires 200+ patterns
        pattern_progress = min(100, (patterns_count / 200) * 100)  # 200 patterns = 100%
        learning_status["models"]["patterns"] = {
            "name": "Pattern Recognition",
            "patterns_learned": patterns_count,
            "progress": round(pattern_progress, 1),
            "status": "expert" if patterns_count >= 200 else "learning" if patterns_count >= 20 else "beginner",
            "needed_for_expert": max(0, 200 - patterns_count),
            "description": f"{patterns_count} patterns identified"
        }
        
        # 4. MARKET STATE ANALYZER
        market_states_raw = await r.get('ai:learning:market_states')
        market_states = 0
        if market_states_raw:
            try:
                ms = json.loads(market_states_raw)
                market_states = len(ms) if isinstance(ms, dict) else 0
            except:
                pass
        
        # EXPERT requires 50+ market states
        market_progress = min(100, (market_states / 50) * 100)  # 50 market states = 100%
        learning_status["models"]["market_states"] = {
            "name": "Market State Analyzer",
            "states_learned": market_states,
            "progress": round(market_progress, 1),
            "status": "expert" if market_states >= 50 else "learning" if market_states >= 5 else "beginner",
            "needed_for_expert": max(0, 50 - market_states),
            "description": f"{market_states} market states learned"
        }
        
        # 5. SENTIMENT ANALYZER
        sentiment_raw = await r.get('ai:learning:sentiment')
        sentiment_states = 0
        if sentiment_raw:
            try:
                s = json.loads(sentiment_raw)
                sentiment_states = len(s) if isinstance(s, dict) else 0
            except:
                pass
        
        # EXPERT requires 30+ sentiment patterns
        sentiment_progress = min(100, (sentiment_states / 30) * 100)  # 30 sentiment states = 100%
        learning_status["models"]["sentiment"] = {
            "name": "Sentiment Analyzer",
            "states_learned": sentiment_states,
            "progress": round(sentiment_progress, 1),
            "status": "expert" if sentiment_states >= 30 else "learning" if sentiment_states >= 3 else "beginner",
            "needed_for_expert": max(0, 30 - sentiment_states),
            "description": f"{sentiment_states} sentiment patterns learned"
        }
        
        # 6. REGIME DETECTOR
        regime_keys = await r.keys('regime:*')
        regime_count = len([k for k in regime_keys if b':duration:' not in k]) if regime_keys else 0
        
        # EXPERT requires 500+ symbols analyzed
        regime_progress = min(100, (regime_count / 500) * 100)  # 500 symbols = 100%
        learning_status["models"]["regime_detector"] = {
            "name": "Regime Detector",
            "symbols_analyzed": regime_count,
            "progress": round(regime_progress, 1),
            "status": "expert" if regime_count >= 500 else "learning" if regime_count >= 50 else "beginner",
            "needed_for_expert": max(0, 500 - regime_count),
            "description": f"{regime_count} symbols analyzed"
        }
        
        # 7. COMPLETED TRADES (Learning History)
        completed_count = await r.llen('trades:completed:default')
        
        # EXPERT requires 500+ completed trades in history
        history_progress = min(100, (completed_count / 500) * 100)  # 500 trades = 100%
        learning_status["models"]["trade_history"] = {
            "name": "Trade History",
            "completed_trades": completed_count,
            "progress": round(history_progress, 1),
            "status": "expert" if completed_count >= 500 else "learning" if completed_count >= 50 else "beginner",
            "needed_for_expert": max(0, 500 - completed_count),
            "description": f"{completed_count} trades recorded"
        }
        
        # Calculate overall progress (weighted average)
        progresses = [m.get('progress', 0) for m in learning_status["models"].values()]
        overall = round(sum(progresses) / len(progresses), 1) if progresses else 0
        learning_status["overall_progress"] = overall
        
        # Determine expert level
        if overall >= 80:
            learning_status["expert_level"] = "Expert Trader"
        elif overall >= 60:
            learning_status["expert_level"] = "Advanced Learner"
        elif overall >= 40:
            learning_status["expert_level"] = "Intermediate"
        elif overall >= 20:
            learning_status["expert_level"] = "Learning"
        else:
            learning_status["expert_level"] = "Beginner"
        
        # Summary stats with accurate numbers
        learning_status["summary"] = {
            "total_trades": total_trades,
            "winning_trades": winning,
            "losing_trades": total_trades - winning if total_trades else 0,
            "win_rate": round((winning / total_trades * 100), 1) if total_trades > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "q_states": q_states,
            "patterns": patterns_count,
            "market_states": market_states,
            "sentiment_states": sentiment_states,
            "regime_symbols": regime_count,
            "completed_history": completed_count,
            "total_states_learned": q_states + patterns_count + market_states + sentiment_states
        }
        
        # Generate helpful recommendations
        if total_trades < 1000:
            learning_status["recommendations"].append(f"Continue trading to improve learning ({total_trades}/5000 for expert)")
        if q_states < 100:
            learning_status["recommendations"].append(f"Q-Learning building knowledge ({q_states}/500 states for expert)")
        if patterns_count < 50:
            learning_status["recommendations"].append(f"Pattern recognition developing ({patterns_count}/200 patterns for expert)")
        # Count expert models
        expert_models = sum(1 for m in learning_status["models"].values() if m.get('status') == 'expert')
        learning_models = sum(1 for m in learning_status["models"].values() if m.get('status') == 'learning')
        total_models = len(learning_status['models'])
        
        learning_status["model_summary"] = f"{expert_models} expert, {learning_models} learning, {total_models - expert_models - learning_models} beginner"
        
    except Exception as e:
        logger.error(f"Learning status error: {e}")
        learning_status["error"] = str(e)
        
    return {"success": True, "data": learning_status}


@router.get("/trades/history")
async def get_trade_history(user_id: str = "default", limit: int = 10):
    """Get recent completed trades history"""
    r = await redis.from_url(settings.REDIS_URL)
    
    try:
        trades = []
        
        # Get completed trades from Redis
        trades_data = await r.lrange(f"trades:completed:{user_id}", 0, limit - 1)
        
        # For admin/default: migrate from global if user-specific is empty
        if not trades_data and user_id == 'default':
            global_trades = await r.lrange('trades:completed:default', 0, limit - 1)
            if not global_trades:
                # Try old key format without user_id suffix
                global_trades = await r.lrange('trading:events', 0, limit - 1)
            trades_data = global_trades
        
        for trade_json in trades_data:
            try:
                if isinstance(trade_json, bytes):
                    trade_json = trade_json.decode()
                trade = json.loads(trade_json)
                trades.append({
                    "symbol": trade.get("symbol", "UNKNOWN"),
                    "pnl": float(trade.get("pnl", 0)),
                    "pnl_percent": float(trade.get("pnl_percent", 0)),
                    "close_reason": trade.get("close_reason", "Unknown"),
                    "closed_time": trade.get("closed_time", datetime.utcnow().isoformat()),
                    "side": trade.get("side", ""),
                    "entry_price": trade.get("entry_price"),
                    "exit_price": trade.get("exit_price")
                })
            except:
                continue
        
        return {
            "success": True,
            "data": {
                "trades": trades,
                "total": len(trades)
            }
        }
    finally:
        await r.aclose()


@router.get("/trading/stats")
async def get_trading_stats(user_id: str = "default"):
    """Get comprehensive trading statistics FOR A SPECIFIC USER"""
    r = await redis.from_url(settings.REDIS_URL)
    
    try:
        # Get USER-SPECIFIC stats
        stats_data = await r.get(f"trader:stats:{user_id}")
        
        # For admin/default: migrate from global if user-specific doesn't exist
        if not stats_data and user_id == 'default':
            global_stats = await r.get('trader:stats')
            if global_stats:
                await r.set(f'trader:stats:{user_id}', global_stats)
                stats_data = global_stats
                logger.info(f"Migrated global stats to trader:stats:{user_id}")
        
        stats = {}
        if stats_data:
            try:
                stats = json.loads(stats_data.decode() if isinstance(stats_data, bytes) else stats_data)
            except:
                pass
        
        total_trades = int(stats.get("total_trades", 0))
        winning_trades = int(stats.get("winning_trades", 0))
        total_pnl = float(stats.get("total_pnl", 0))
        
        # Calculate derived stats
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Get completed trades for more stats
        trades_data = await r.lrange(f"trades:completed:{user_id}", 0, 99)
        
        profits = []
        losses = []
        best_trade = 0
        worst_trade = 0
        current_streak = 0
        last_result = None
        
        for trade_json in trades_data:
            try:
                if isinstance(trade_json, bytes):
                    trade_json = trade_json.decode()
                trade = json.loads(trade_json)
                pnl = float(trade.get("pnl", 0))
                
                if pnl > 0:
                    profits.append(pnl)
                    if pnl > best_trade:
                        best_trade = pnl
                    # Streak tracking
                    if last_result == 'win':
                        current_streak += 1
                    else:
                        current_streak = 1
                    last_result = 'win'
                elif pnl < 0:
                    losses.append(pnl)
                    if pnl < worst_trade:
                        worst_trade = pnl
                    # Streak tracking
                    if last_result == 'loss':
                        current_streak -= 1
                    else:
                        current_streak = -1
                    last_result = 'loss'
            except:
                continue
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        profit_factor = (sum(profits) / abs(sum(losses))) if losses and sum(losses) != 0 else 0
        
        # Get opportunities_scanned from GLOBAL stats (it's system-wide, not per-user)
        opportunities_scanned = 0
        global_stats_raw = await r.get('trader:stats')
        if global_stats_raw:
            try:
                global_stats = json.loads(global_stats_raw.decode() if isinstance(global_stats_raw, bytes) else global_stats_raw)
                opportunities_scanned = int(global_stats.get("opportunities_scanned", 0))
            except:
                pass
        
        return {
            "success": True,
            "data": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 1),
                "total_pnl": round(total_pnl, 2),
                "max_drawdown": float(stats.get("max_drawdown", 0)),
                "opportunities_scanned": opportunities_scanned,  # From global stats
                "avg_profit": round(avg_profit, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "current_streak": current_streak,
                "best_trade": round(best_trade, 2),
                "worst_trade": round(worst_trade, 2)
            }
        }
    finally:
        await r.aclose()

