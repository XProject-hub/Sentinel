"""
SENTINEL AI - Binance API Client
Real-time trading with Binance API
https://binance-docs.github.io/apidocs/futures/en/
"""

import asyncio
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urlencode
import httpx
from loguru import logger


class BinanceClient:
    """
    Binance API Client for real trading
    - Supports USDT-M Futures (linear perpetuals)
    - Real-time market data
    - Order execution
    
    Documentation: https://binance-docs.github.io/apidocs/futures/en/
    """
    
    # Main endpoints - USDT-M Futures
    BASE_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://demo-fapi.binance.com"
    
    # Spot endpoints (for reference)
    SPOT_URL = "https://api.binance.com"
    SPOT_TESTNET_URL = "https://testnet.binance.vision"
    
    def __init__(
        self, 
        api_key: str, 
        api_secret: str, 
        testnet: bool = False,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Use futures endpoint by default (matching Bybit's focus on perpetuals)
        if testnet:
            self.base_url = self.TESTNET_URL
        else:
            self.base_url = self.BASE_URL
            
        self.recv_window = 5000
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.request_counter = 0
        
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for Binance API"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    async def _get_server_time(self) -> int:
        """Get Binance server time to sync timestamps"""
        try:
            response = await self.http_client.get(f"{self.base_url}/fapi/v1/time")
            data = response.json()
            if "serverTime" in data:
                return data["serverTime"]
        except:
            pass
        return int(time.time() * 1000)
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for Binance API requests
        """
        return {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
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
        params = params or {}
        
        try:
            headers = self._get_headers() if auth else {}
            
            if auth:
                # Get server time for authenticated requests
                server_time = await self._get_server_time()
                params["timestamp"] = server_time
                params["recvWindow"] = self.recv_window
                
                # Create query string and signature
                query_string = urlencode(params)
                signature = self._generate_signature(query_string)
                params["signature"] = signature
            
            if method == "GET":
                if params:
                    query_string = urlencode(params)
                    url = f"{url}?{query_string}"
                response = await self.http_client.get(url, headers=headers)
            else:  # POST, DELETE
                response = await self.http_client.request(
                    method,
                    url,
                    headers=headers,
                    data=params
                )
            
            # Handle empty or non-JSON responses
            if not response.text or response.text.strip() == "":
                logger.error(f"Binance returned empty response for {endpoint}")
                return {"success": False, "error": "Empty response from Binance API"}
            
            try:
                data = response.json()
            except Exception as json_err:
                logger.error(f"Binance JSON parse error: {json_err}, Response: {response.text[:200]}")
                return {"success": False, "error": f"Invalid response from Binance: {str(json_err)}"}
            
            # Binance returns error in different format
            if "code" in data and data["code"] != 200:
                error_code = data.get("code")
                error_msg = data.get("msg", "Unknown error")
                logger.error(f"Binance API error [{error_code}]: {error_msg}")
                
                # Comprehensive error code handling
                ERROR_MESSAGES = {
                    -1000: "Unknown error",
                    -1001: "Disconnected - internal server error",
                    -1002: "Unauthorized - API key not authorized",
                    -1003: "Too many requests - rate limit exceeded",
                    -1006: "Unexpected response - contact support",
                    -1007: "Timeout - request took too long",
                    -1010: "ERROR_MSG_RECEIVED",
                    -1013: "Invalid quantity",
                    -1014: "Unknown order composition",
                    -1015: "Too many orders",
                    -1016: "Service shutting down",
                    -1020: "Operation not supported",
                    -1021: "Invalid timestamp - sync your clock",
                    -1022: "Invalid signature - check your API secret",
                    -1100: "Illegal characters in parameter",
                    -1101: "Too many parameters",
                    -1102: "Mandatory parameter missing",
                    -1103: "Unknown parameter",
                    -1104: "Unread parameters",
                    -1105: "Parameter empty",
                    -1106: "Parameter not required",
                    -1111: "Precision is over maximum",
                    -1112: "No trading window",
                    -1114: "TimeInForce not required",
                    -1115: "Invalid timeInForce",
                    -1116: "Invalid order type",
                    -1117: "Invalid side",
                    -1118: "Empty new client order ID",
                    -1119: "Empty original client order ID",
                    -1120: "Invalid interval",
                    -1121: "Invalid symbol",
                    -1125: "Invalid listen key",
                    -1127: "Lookup interval too big",
                    -1128: "Combination of parameters invalid",
                    -1130: "Invalid data sent",
                    -2010: "NEW_ORDER_REJECTED",
                    -2011: "CANCEL_REJECTED",
                    -2013: "Order does not exist",
                    -2014: "API key format invalid",
                    -2015: "Invalid API key, IP, or permissions",
                    -2016: "No trading window",
                    -2018: "Balance is insufficient",
                    -2019: "Margin is insufficient",
                    -2020: "Unable to fill order",
                    -2021: "Order would trigger immediately",
                    -2022: "ReduceOnly order rejected",
                    -2024: "Position side mismatch",
                    -2025: "Reach max open orders limit",
                    -2026: "Reached max position count limit",
                    -4000: "Invalid order status",
                    -4001: "Price less than 0",
                    -4002: "Price greater than max price",
                    -4003: "Quantity less than 0",
                    -4004: "Quantity less than min quantity",
                    -4005: "Quantity greater than max quantity",
                    -4006: "Stop price less than 0",
                    -4007: "Stop price greater than max price",
                    -4008: "Tick size precision error",
                    -4009: "Price precision error",
                    -4010: "Quantity precision error",
                    -4014: "Price too high",
                    -4015: "Client order ID too long",
                    -4016: "Price lower than mark price multiplier down",
                    -4017: "Price higher than mark price multiplier up",
                    -4028: "Timestamp earlier than order time",
                    -4029: "Timestamp later than order time",
                }
                
                if error_code in ERROR_MESSAGES:
                    return {"success": False, "error": ERROR_MESSAGES[error_code], "code": error_code}
                    
                return {"success": False, "error": error_msg, "code": error_code}
                
            return {"success": True, "data": data}
            
        except Exception as e:
            logger.error(f"Binance request error: {e}")
            return {"success": False, "error": str(e)}
            
    # ============================================
    # MARKET DATA (Public)
    # ============================================
    
    async def get_tickers(self, symbol: Optional[str] = None) -> Dict:
        """Get real-time tickers"""
        params = {}
        if symbol:
            params["symbol"] = symbol
            endpoint = "/fapi/v1/ticker/24hr"
        else:
            endpoint = "/fapi/v1/ticker/24hr"
        return await self._request("GET", endpoint, params)
        
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get orderbook depth"""
        params = {
            "symbol": symbol,
            "limit": min(limit, 1000)  # Binance max is 1000
        }
        return await self._request("GET", "/fapi/v1/depth", params)
    
    async def get_klines(self, symbol: str, interval: str = "5m", limit: int = 100) -> Dict:
        """
        Get kline/candlestick data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval. 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
            limit: Limit for data size (max 1500)
            
        Returns:
            Dict with kline data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500)
        }
        return await self._request("GET", "/fapi/v1/klines", params)
        
    async def get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate"""
        params = {"symbol": symbol}
        return await self._request("GET", "/fapi/v1/fundingRate", params)
        
    async def get_open_interest(self, symbol: str) -> Dict:
        """Get open interest"""
        params = {"symbol": symbol}
        return await self._request("GET", "/fapi/v1/openInterest", params)
    
    async def get_long_short_ratio(self, symbol: str, period: str = "1h") -> Dict:
        """
        Get Long/Short Ratio - for sentiment analysis
        
        Period: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
        """
        params = {
            "symbol": symbol,
            "period": period,
            "limit": 1
        }
        return await self._request("GET", "/futures/data/globalLongShortAccountRatio", params)
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get Recent Trades - useful for momentum detection
        """
        params = {
            "symbol": symbol,
            "limit": min(limit, 1000)
        }
        return await self._request("GET", "/fapi/v1/trades", params)
    
    async def get_all_symbols(self) -> List[str]:
        """
        Get ALL available trading symbols from Binance Futures
        """
        try:
            response = await self._request("GET", "/fapi/v1/exchangeInfo", {})
            
            if response.get("success") and response.get("data"):
                symbols = []
                for item in response["data"].get("symbols", []):
                    symbol = item.get("symbol", "")
                    status = item.get("status", "")
                    # Only get active USDT trading pairs
                    if status == "TRADING" and symbol.endswith("USDT"):
                        symbols.append(symbol)
                
                logger.info(f"Fetched {len(symbols)} active USDT-M futures trading pairs from Binance")
                return symbols
            return []
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return []
    
    async def get_instrument_info(self, symbol: str) -> Dict:
        """Get detailed instrument info (min qty, tick size, etc.)"""
        response = await self._request("GET", "/fapi/v1/exchangeInfo", {})
        if response.get("success") and response.get("data"):
            for item in response["data"].get("symbols", []):
                if item.get("symbol") == symbol:
                    return {"success": True, "data": item}
        return {"success": False, "error": "Symbol not found"}
        
    # ============================================
    # ACCOUNT (Private - requires auth)
    # ============================================
    
    async def get_wallet_balance(self) -> Dict:
        """Get wallet balance (futures account)"""
        return await self._request("GET", "/fapi/v2/balance", {}, auth=True)
    
    async def get_account_info(self) -> Dict:
        """Get account information including positions"""
        return await self._request("GET", "/fapi/v2/account", {}, auth=True)
        
    async def get_positions(self, symbol: Optional[str] = None) -> Dict:
        """Get open positions"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v2/positionRisk", params, auth=True)
        
    async def get_pnl(self, symbol: Optional[str] = None, limit: int = 100) -> Dict:
        """Get income history (PnL)"""
        params = {"limit": min(limit, 1000)}
        if symbol:
            params["symbol"] = symbol
        params["incomeType"] = "REALIZED_PNL"
        return await self._request("GET", "/fapi/v1/income", params, auth=True)
        
    # ============================================
    # TRADING (Private - requires auth)
    # ============================================
    
    async def place_order(
        self,
        symbol: str,
        side: str,  # BUY, SELL
        order_type: str,  # MARKET, LIMIT, STOP, TAKE_PROFIT, etc.
        qty: str,
        price: Optional[str] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        stop_price: Optional[str] = None,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        position_side: str = "BOTH",  # BOTH for one-way, LONG/SHORT for hedge
    ) -> Dict:
        """Place an order"""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": qty,
            "positionSide": position_side,
        }
        
        if reduce_only:
            params["reduceOnly"] = "true"
        
        if order_type.upper() == "LIMIT":
            params["timeInForce"] = time_in_force
            if price:
                params["price"] = price
                
        if stop_price:
            params["stopPrice"] = stop_price
            
        # Note: Binance uses separate endpoints for TP/SL orders
        # For basic orders, TP/SL need to be placed as separate orders
            
        return await self._request("POST", "/fapi/v1/order", params, auth=True)
        
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        """Cancel an order"""
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
            
        return await self._request("DELETE", "/fapi/v1/order", params, auth=True)
        
    async def cancel_all_orders(self, symbol: str) -> Dict:
        """Cancel all orders for a symbol"""
        params = {"symbol": symbol}
        return await self._request("DELETE", "/fapi/v1/allOpenOrders", params, auth=True)
        
    async def get_open_orders(self, symbol: Optional[str] = None) -> Dict:
        """Get open orders"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v1/openOrders", params, auth=True)
        
    async def get_order_history(
        self, 
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> Dict:
        """Get order history"""
        params = {"limit": min(limit, 1000)}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v1/allOrders", params, auth=True)
        
    # ============================================
    # POSITION MANAGEMENT
    # ============================================
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set position leverage"""
        params = {
            "symbol": symbol,
            "leverage": leverage
        }
        return await self._request("POST", "/fapi/v1/leverage", params, auth=True)
    
    async def switch_position_mode(self, dual_side: bool = False) -> Dict:
        """
        Switch position mode between One-Way and Hedge
        dual_side: True = Hedge Mode, False = One-Way Mode
        """
        params = {
            "dualSidePosition": "true" if dual_side else "false"
        }
        return await self._request("POST", "/fapi/v1/positionSide/dual", params, auth=True)
    
    async def switch_margin_mode(self, symbol: str, margin_type: str = "CROSSED") -> Dict:
        """
        Switch margin mode between Cross and Isolated
        margin_type: CROSSED or ISOLATED
        """
        params = {
            "symbol": symbol,
            "marginType": margin_type.upper()
        }
        return await self._request("POST", "/fapi/v1/marginType", params, auth=True)
        
    async def set_isolated_margin(
        self,
        symbol: str,
        amount: str,
        position_side: str = "BOTH",
        type_: int = 1  # 1 = add, 2 = reduce
    ) -> Dict:
        """Modify isolated margin"""
        params = {
            "symbol": symbol,
            "positionSide": position_side,
            "amount": amount,
            "type": type_
        }
        return await self._request("POST", "/fapi/v1/positionMargin", params, auth=True)
        
    # ============================================
    # CONNECTION TEST
    # ============================================
    
    async def test_connection(self) -> Dict:
        """Test API connection and authentication - tries SPOT first, then FUTURES"""
        try:
            logger.info(f"Testing Binance connection with API key: {self.api_key[:8]}...")
            
            # First test with a simple public endpoint (Spot API)
            try:
                public_url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
                response = await self.http_client.get(public_url)
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": "Cannot reach Binance API - check network"
                    }
            except Exception as net_err:
                return {
                    "success": False,
                    "error": f"Network error: {str(net_err)}"
                }
            
            # Try SPOT account first (more common)
            spot_result = await self.get_spot_balance()
            if spot_result.get("success"):
                logger.info("Binance SPOT connection test successful")
                return {
                    "success": True,
                    "message": "Connection successful (Spot)",
                    "account_type": "spot",
                    "data": spot_result.get("data")
                }
            
            # Try FUTURES account
            futures_result = await self.get_wallet_balance()
            if futures_result.get("success"):
                logger.info("Binance FUTURES connection test successful")
                return {
                    "success": True,
                    "message": "Connection successful (Futures)",
                    "account_type": "futures",
                    "data": futures_result.get("data")
                }
            
            # Both failed - provide helpful error message
            spot_error = spot_result.get("error", "")
            futures_error = futures_result.get("error", "")
            
            if "IP" in spot_error or "IP" in futures_error:
                error_msg = f"IP not whitelisted. Add server IP to your Binance API key settings."
            elif "-2015" in str(spot_result.get("code", "")) or "-2015" in str(futures_result.get("code", "")):
                error_msg = "Invalid API key, IP, or permissions. Check your API key settings on Binance."
            else:
                error_msg = f"Spot: {spot_error} | Futures: {futures_error}"
            
            logger.error(f"Binance auth failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
                
        except Exception as e:
            logger.error(f"Binance connection test exception: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ============================================
    # SPOT TRADING
    # ============================================
    
    async def _spot_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        auth: bool = False
    ) -> Dict:
        """Make Spot API request (different base URL)"""
        # Spot uses api.binance.com instead of fapi.binance.com
        if self.testnet:
            spot_url = "https://testnet.binance.vision"
        else:
            spot_url = "https://api.binance.com"
            
        url = f"{spot_url}{endpoint}"
        params = params or {}
        
        try:
            headers = self._get_headers() if auth else {}
            
            if auth:
                server_time = await self._get_server_time()
                params["timestamp"] = server_time
                params["recvWindow"] = self.recv_window
                
                query_string = urlencode(params)
                signature = self._generate_signature(query_string)
                params["signature"] = signature
            
            if method == "GET":
                if params:
                    query_string = urlencode(params)
                    url = f"{url}?{query_string}"
                response = await self.http_client.get(url, headers=headers)
            else:
                response = await self.http_client.request(
                    method,
                    url,
                    headers=headers,
                    data=params
                )
            
            if not response.text or response.text.strip() == "":
                return {"success": False, "error": "Empty response from Binance Spot API"}
            
            try:
                data = response.json()
            except Exception as json_err:
                return {"success": False, "error": f"Invalid response: {str(json_err)}"}
            
            if "code" in data and data["code"] != 200:
                return {"success": False, "error": data.get("msg", "Unknown error"), "code": data["code"]}
                
            return {"success": True, "data": data}
            
        except Exception as e:
            logger.error(f"Binance Spot request error: {e}")
            return {"success": False, "error": str(e)}
    
    async def place_spot_order(
        self,
        symbol: str,
        side: str,  # BUY, SELL
        order_type: str,  # MARKET, LIMIT
        quantity: Optional[str] = None,
        quote_quantity: Optional[str] = None,  # For MARKET BUY with USDT amount
        price: Optional[str] = None,
        time_in_force: str = "GTC",
    ) -> Dict:
        """
        Place a SPOT order (no leverage, actual asset purchase)
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            order_type: 'MARKET' or 'LIMIT'
            quantity: Quantity of base asset (e.g., BTC)
            quote_quantity: For MARKET BUY - amount of quote asset (e.g., USDT) to spend
            price: Price for limit orders
            time_in_force: Order time in force
            
        Returns:
            Dict with order result
        """
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
        }
        
        if quantity:
            params["quantity"] = quantity
        elif quote_quantity and order_type.upper() == "MARKET" and side.upper() == "BUY":
            params["quoteOrderQty"] = quote_quantity
        
        if price and order_type.upper() == "LIMIT":
            params["price"] = price
            params["timeInForce"] = time_in_force
        
        logger.info(f"Placing Binance SPOT {side} order: {symbol}")
        return await self._spot_request("POST", "/api/v3/order", params, auth=True)
    
    async def get_spot_balance(self) -> Dict:
        """Get spot account balance"""
        result = await self._spot_request("GET", "/api/v3/account", {}, auth=True)
        if result.get("success") and result.get("data"):
            balances = result["data"].get("balances", [])
            # Filter to only show non-zero balances
            non_zero = [b for b in balances if float(b.get("free", 0)) > 0 or float(b.get("locked", 0)) > 0]
            return {"success": True, "data": non_zero}
        return result
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get spot ticker for a specific symbol (e.g. BTCUSDT)"""
        params = {"symbol": symbol}
        result = await self._spot_request("GET", "/api/v3/ticker/price", params)
        if result.get("success") and result.get("data"):
            return {
                "success": True,
                "data": {
                    "symbol": result["data"].get("symbol"),
                    "lastPrice": result["data"].get("price")
                }
            }
        return result
    
    async def get_spot_orders(self, symbol: Optional[str] = None) -> Dict:
        """Get open spot orders"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._spot_request("GET", "/api/v3/openOrders", params, auth=True)
    
    async def cancel_spot_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict:
        """Cancel a spot order"""
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
            
        return await self._spot_request("DELETE", "/api/v3/order", params, auth=True)
    
    async def get_spot_asset_info(self, symbol: str) -> Dict:
        """Get spot asset balance for a specific coin"""
        result = await self.get_spot_balance()
        if result.get("success") and result.get("data"):
            base_asset = symbol.replace("USDT", "").replace("USDC", "")
            for balance in result["data"]:
                if balance.get("asset") == base_asset:
                    return {
                        "success": True,
                        "data": {
                            "coin": balance.get("asset"),
                            "available": float(balance.get("free", 0)),
                            "locked": float(balance.get("locked", 0)),
                            "total": float(balance.get("free", 0)) + float(balance.get("locked", 0))
                        }
                    }
        return {"success": False, "error": "Asset not found"}
            
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()


# ============================================
# REAL-TIME WEBSOCKET CLIENT
# ============================================

class BinanceWebSocket:
    """
    Binance Futures WebSocket for real-time data
    """
    
    PUBLIC_URL = "wss://fstream.binance.com/ws"
    TESTNET_URL = "wss://demo-fstream.binance.com/ws"
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = self.TESTNET_URL if testnet else self.PUBLIC_URL
        self.ws = None
        self.callbacks = {}
        
    async def subscribe_ticker(self, symbol: str, callback):
        """Subscribe to real-time ticker"""
        stream = f"{symbol.lower()}@ticker"
        self.callbacks[stream] = callback
        
    async def subscribe_trades(self, symbol: str, callback):
        """Subscribe to real-time trades"""
        stream = f"{symbol.lower()}@aggTrade"
        self.callbacks[stream] = callback
        
    async def subscribe_orderbook(self, symbol: str, callback, depth: int = 20):
        """Subscribe to real-time orderbook"""
        stream = f"{symbol.lower()}@depth{depth}@100ms"
        self.callbacks[stream] = callback
        
    async def subscribe_klines(self, symbol: str, interval: str, callback):
        """Subscribe to real-time klines"""
        stream = f"{symbol.lower()}@kline_{interval}"
        self.callbacks[stream] = callback
