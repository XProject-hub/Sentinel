"""
SENTINEL AI - Bybit V5 API Client
Real-time trading with Bybit V5 API
https://bybit-exchange.github.io/docs/v5/intro
"""

import asyncio
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import httpx
from loguru import logger


class BybitV5Client:
    """
    Bybit V5 API Client for real trading
    - Uses latest V5 unified API
    - Supports spot, linear, inverse perpetuals
    - Real-time market data
    - Order execution
    """
    
    BASE_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"
    
    def __init__(
        self, 
        api_key: str, 
        api_secret: str, 
        testnet: bool = False
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self.recv_window = 5000
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
    def _generate_signature(self, timestamp: str, params: str) -> str:
        """Generate HMAC SHA256 signature for V5 API"""
        param_str = f"{timestamp}{self.api_key}{self.recv_window}{params}"
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    async def _get_server_time(self) -> int:
        """Get Bybit server time to sync timestamps"""
        try:
            response = await self.http_client.get(f"{self.base_url}/v5/market/time")
            data = response.json()
            if data.get("retCode") == 0:
                server_time = int(data.get("result", {}).get("timeSecond", 0)) * 1000
                if server_time:
                    return server_time
        except:
            pass
        return int(time.time() * 1000)
    
    def _get_headers(self, params: str = "", server_time: Optional[int] = None) -> Dict[str, str]:
        """Get authenticated headers"""
        timestamp = str(server_time or int(time.time() * 1000))
        signature = self._generate_signature(timestamp, params)
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "Content-Type": "application/json"
        }
        
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        auth: bool = False
    ) -> Dict:
        """Make API request"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            # Get server time for authenticated requests to avoid timestamp issues
            server_time = None
            if auth:
                server_time = await self._get_server_time()
            
            if method == "GET":
                param_str = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
                headers = self._get_headers(param_str, server_time) if auth else {}
                
                if params:
                    url = f"{url}?{param_str}"
                    
                response = await self.http_client.get(url, headers=headers)
            else:
                import json
                param_str = json.dumps(params) if params else ""
                headers = self._get_headers(param_str, server_time) if auth else {}
                response = await self.http_client.post(url, headers=headers, json=params)
                
            data = response.json()
            
            if data.get("retCode") != 0:
                error_msg = data.get('retMsg', 'Unknown error')
                error_code = data.get('retCode')
                logger.error(f"Bybit API error [{error_code}]: {error_msg}")
                
                # Translate common errors
                if error_code == 10003:
                    return {"success": False, "error": "Invalid API key"}
                elif error_code == 10004:
                    return {"success": False, "error": "Invalid signature - check API secret"}
                elif error_code == 10005:
                    return {"success": False, "error": "Permission denied - check API permissions"}
                elif error_code == 10006:
                    return {"success": False, "error": "IP not whitelisted on Bybit"}
                elif error_code == 10010:
                    return {"success": False, "error": "IP not in whitelist - add 109.104.154.183 to Bybit API"}
                elif "ip" in error_msg.lower():
                    return {"success": False, "error": f"IP issue: {error_msg}. Whitelist: 109.104.154.183"}
                    
                return {"success": False, "error": error_msg}
                
            return {"success": True, "data": data.get("result", {})}
            
        except Exception as e:
            logger.error(f"Bybit request error: {e}")
            return {"success": False, "error": str(e)}
            
    # ============================================
    # MARKET DATA (Public)
    # ============================================
    
    async def get_tickers(self, category: str = "linear", symbol: Optional[str] = None) -> Dict:
        """Get real-time tickers"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/market/tickers", params)
        
    async def get_orderbook(self, symbol: str, category: str = "linear", limit: int = 25) -> Dict:
        """Get orderbook depth"""
        params = {
            "category": category,
            "symbol": symbol,
            "limit": limit
        }
        return await self._request("GET", "/v5/market/orderbook", params)
        
    async def get_kline(
        self, 
        symbol: str, 
        interval: str = "60",  # 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W
        category: str = "linear",
        limit: int = 200
    ) -> Dict:
        """Get kline/candlestick data"""
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return await self._request("GET", "/v5/market/kline", params)
        
    async def get_funding_rate(self, symbol: str, category: str = "linear") -> Dict:
        """Get current funding rate"""
        params = {
            "category": category,
            "symbol": symbol
        }
        return await self._request("GET", "/v5/market/funding/history", params)
        
    async def get_open_interest(self, symbol: str, category: str = "linear") -> Dict:
        """Get open interest"""
        params = {
            "category": category,
            "symbol": symbol,
            "intervalTime": "5min",
            "limit": 1
        }
        return await self._request("GET", "/v5/market/open-interest", params)
        
    # ============================================
    # ACCOUNT (Private - requires auth)
    # ============================================
    
    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
        """Get wallet balance"""
        params = {"accountType": account_type}
        return await self._request("GET", "/v5/account/wallet-balance", params, auth=True)
        
    async def get_positions(self, category: str = "linear", symbol: Optional[str] = None, settle_coin: str = "USDT") -> Dict:
        """Get open positions"""
        params = {"category": category, "settleCoin": settle_coin}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/position/list", params, auth=True)
        
    async def get_pnl(self, category: str = "linear", limit: int = 50) -> Dict:
        """Get closed PnL history"""
        params = {
            "category": category,
            "limit": limit
        }
        return await self._request("GET", "/v5/position/closed-pnl", params, auth=True)
        
    # ============================================
    # TRADING (Private - requires auth)
    # ============================================
    
    async def place_order(
        self,
        symbol: str,
        side: str,  # Buy, Sell
        order_type: str,  # Market, Limit
        qty: str,
        category: str = "linear",
        price: Optional[str] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
    ) -> Dict:
        """Place an order"""
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
            "closeOnTrigger": close_on_trigger
        }
        
        if price and order_type == "Limit":
            params["price"] = price
            
        if take_profit:
            params["takeProfit"] = take_profit
            
        if stop_loss:
            params["stopLoss"] = stop_loss
            
        return await self._request("POST", "/v5/order/create", params, auth=True)
        
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        category: str = "linear"
    ) -> Dict:
        """Cancel an order"""
        params = {
            "category": category,
            "symbol": symbol
        }
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
            
        return await self._request("POST", "/v5/order/cancel", params, auth=True)
        
    async def cancel_all_orders(self, category: str = "linear", symbol: Optional[str] = None) -> Dict:
        """Cancel all orders"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return await self._request("POST", "/v5/order/cancel-all", params, auth=True)
        
    async def get_open_orders(self, category: str = "linear", symbol: Optional[str] = None) -> Dict:
        """Get open orders"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/order/realtime", params, auth=True)
        
    async def get_order_history(
        self, 
        category: str = "linear", 
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> Dict:
        """Get order history"""
        params = {
            "category": category,
            "limit": limit
        }
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/order/history", params, auth=True)
        
    # ============================================
    # POSITION MANAGEMENT
    # ============================================
    
    async def set_leverage(self, symbol: str, leverage: str, category: str = "linear") -> Dict:
        """Set position leverage"""
        params = {
            "category": category,
            "symbol": symbol,
            "buyLeverage": leverage,
            "sellLeverage": leverage
        }
        return await self._request("POST", "/v5/position/set-leverage", params, auth=True)
        
    async def set_trading_stop(
        self,
        symbol: str,
        category: str = "linear",
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        position_idx: int = 0
    ) -> Dict:
        """Set take profit and stop loss for position"""
        params = {
            "category": category,
            "symbol": symbol,
            "positionIdx": position_idx
        }
        if take_profit:
            params["takeProfit"] = take_profit
        if stop_loss:
            params["stopLoss"] = stop_loss
            
        return await self._request("POST", "/v5/position/trading-stop", params, auth=True)
        
    # ============================================
    # CONNECTION TEST
    # ============================================
    
    async def test_connection(self) -> Dict:
        """Test API connection and authentication"""
        try:
            logger.info(f"Testing Bybit connection with API key: {self.api_key[:8]}...")
            
            # First test with a simple public endpoint
            public_test = await self.get_tickers(symbol="BTCUSDT")
            if not public_test.get("success"):
                logger.error(f"Public API test failed: {public_test.get('error')}")
                return {
                    "success": False,
                    "error": f"Network error: {public_test.get('error')}"
                }
            
            # Then test authenticated endpoint
            result = await self.get_wallet_balance(account_type="UNIFIED")
            
            if result.get("success"):
                logger.info("Bybit connection test successful (UNIFIED)")
                return {
                    "success": True,
                    "message": "Connection successful",
                    "data": result.get("data")
                }
            
            # Try SPOT account
            result = await self.get_wallet_balance(account_type="SPOT")
            if result.get("success"):
                logger.info("Bybit connection test successful (SPOT)")
                return {
                    "success": True,
                    "message": "Connection successful",
                    "data": result.get("data")
                }
                
            # Try FUND account
            result = await self.get_wallet_balance(account_type="FUND")
            if result.get("success"):
                logger.info("Bybit connection test successful (FUND)")
                return {
                    "success": True,
                    "message": "Connection successful",
                    "data": result.get("data")
                }
            
            logger.error(f"Bybit auth failed: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Authentication failed - check API key and IP whitelist")
            }
                
        except Exception as e:
            logger.error(f"Bybit connection test exception: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()


# ============================================
# REAL-TIME WEBSOCKET CLIENT
# ============================================

class BybitWebSocket:
    """
    Bybit V5 WebSocket for real-time data
    """
    
    PUBLIC_URL = "wss://stream.bybit.com/v5/public/linear"
    PRIVATE_URL = "wss://stream.bybit.com/v5/private"
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws = None
        self.callbacks = {}
        
    def _generate_auth_signature(self) -> Dict:
        """Generate WebSocket auth signature"""
        expires = int(time.time() * 1000) + 10000
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            f"GET/realtime{expires}".encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "op": "auth",
            "args": [self.api_key, expires, signature]
        }
        
    async def subscribe_ticker(self, symbol: str, callback):
        """Subscribe to real-time ticker"""
        self.callbacks[f"tickers.{symbol}"] = callback
        # Implementation would use websockets library
        
    async def subscribe_trades(self, symbol: str, callback):
        """Subscribe to real-time trades"""
        self.callbacks[f"publicTrade.{symbol}"] = callback
        
    async def subscribe_orderbook(self, symbol: str, callback, depth: int = 25):
        """Subscribe to real-time orderbook"""
        self.callbacks[f"orderbook.{depth}.{symbol}"] = callback
        
    async def subscribe_positions(self, callback):
        """Subscribe to position updates (private)"""
        self.callbacks["position"] = callback
        
    async def subscribe_orders(self, callback):
        """Subscribe to order updates (private)"""
        self.callbacks["order"] = callback

