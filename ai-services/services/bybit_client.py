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
    
    Documentation: https://bybit-exchange.github.io/docs/v5/guide
    """
    
    # Main endpoints
    BASE_URL = "https://api.bybit.com"
    BACKUP_URL = "https://api.bytick.com"  # Backup endpoint
    TESTNET_URL = "https://api-testnet.bybit.com"
    
    # Regional endpoints for better latency
    REGIONAL_ENDPOINTS = {
        "NL": "https://api.bybit.nl",          # Netherlands
        "TR": "https://api.bybit-tr.com",       # Turkey
        "KZ": "https://api.bybit.kz",           # Kazakhstan
        "GE": "https://api.bybitgeorgia.ge",    # Georgia
        "AE": "https://api.bybit.ae",           # UAE
        "EU": "https://api.bybit.eu",           # EEA (European Economic Area)
        "HK": "https://api.byhkbit.com",        # Hong Kong (P2P only)
    }
    
    def __init__(
        self, 
        api_key: str, 
        api_secret: str, 
        testnet: bool = False,
        region: str = None  # Optional: "NL", "TR", "KZ", "GE", "AE", "EU"
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Select endpoint based on region or use default
        if testnet:
            self.base_url = self.TESTNET_URL
        elif region and region.upper() in self.REGIONAL_ENDPOINTS:
            self.base_url = self.REGIONAL_ENDPOINTS[region.upper()]
            logger.info(f"Using regional Bybit endpoint: {self.base_url}")
        else:
            self.base_url = self.BASE_URL
            
        self.backup_url = self.BACKUP_URL
        self.recv_window = 5000
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.request_counter = 0  # For unique request IDs
        
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
        """
        Get authenticated headers as per Bybit V5 API documentation.
        https://bybit-exchange.github.io/docs/v5/guide
        """
        timestamp = str(server_time or int(time.time() * 1000))
        signature = self._generate_signature(timestamp, params)
        
        # Increment request counter for unique request ID
        self.request_counter += 1
        request_id = f"sentinel-{timestamp}-{self.request_counter}"
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "Content-Type": "application/json",
            "cdn-request-id": request_id,  # For network diagnostics (per Bybit docs)
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
                # IMPORTANT: Use compact JSON (no spaces) for signature - must match body exactly
                param_str = json.dumps(params, separators=(',', ':')) if params else ""
                headers = self._get_headers(param_str, server_time) if auth else {}
                # Send the exact same string we used for signature
                response = await self.http_client.post(
                    url, 
                    headers=headers, 
                    content=param_str,  # Use content instead of json to ensure exact match
                )
                
            data = response.json()
            
            if data.get("retCode") != 0:
                error_msg = data.get('retMsg', 'Unknown error')
                error_code = data.get('retCode')
                logger.error(f"Bybit API error [{error_code}]: {error_msg}")
                
                # Comprehensive error code handling (from Bybit docs)
                ERROR_MESSAGES = {
                    10001: "Request parameter error - check parameters",
                    10002: "Request timestamp expired - sync server time",
                    10003: "Invalid API key",
                    10004: "Invalid signature - check API secret",
                    10005: "Permission denied - check API permissions",
                    10006: "IP not whitelisted on Bybit",
                    10007: "Access denied - API key not valid for this endpoint",
                    10008: "Invalid sign type",
                    10009: "Invalid Timestamp",
                    10010: "IP not in whitelist - add server IP to Bybit API settings",
                    10014: "Invalid API key format",
                    10016: "Server error - try again",
                    10017: "API not available for this path",
                    10018: "Unsupported request",
                    10024: "KYC required",
                    10025: "Cross margin not enabled",
                    10027: "Banned API key",
                    10028: "API key expired",
                    10029: "Duplicate request",
                    33004: "API key not enabled for trading",
                    110001: "Order does not exist",
                    110003: "Order already filled",
                    110004: "Insufficient balance",
                    110007: "Order already cancelled",
                    110012: "Insufficient available balance",
                    110013: "Cannot cancel order - already filled or cancelled",
                    110017: "Reduce only order rejected",
                    110018: "User ID invalid",
                    110043: "Set leverage not allowed - has open position",
                    110044: "Insufficient available margin",
                    110071: "Order price/qty precision error",
                    140003: "Order cost not available",
                    140004: "Position is in cross margin mode",
                    140013: "Due to risk limit, cannot set leverage so high",
                }
                
                if error_code in ERROR_MESSAGES:
                    return {"success": False, "error": ERROR_MESSAGES[error_code], "code": error_code}
                elif "ip" in error_msg.lower():
                    return {"success": False, "error": f"IP issue: {error_msg}. Whitelist your server IP on Bybit.", "code": error_code}
                    
                return {"success": False, "error": error_msg, "code": error_code}
                
            return {"success": True, "data": data.get("result", {}), "time": data.get("time")}
            
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
    
    async def get_long_short_ratio(self, symbol: str, category: str = "linear", period: str = "1h") -> Dict:
        """
        Get Long/Short Ratio - CRITICAL for sentiment analysis!
        
        Period: 5min, 15min, 30min, 1h, 4h, 1d
        
        Returns ratio of long vs short positions - tells us market sentiment:
        - Ratio > 1: More longs than shorts (bullish sentiment)
        - Ratio < 1: More shorts than longs (bearish sentiment)
        - Extreme ratios (>2 or <0.5) often signal reversal
        
        https://bybit-exchange.github.io/docs/v5/market/long-short-ratio
        """
        params = {
            "category": category,
            "symbol": symbol,
            "period": period,
            "limit": 1
        }
        return await self._request("GET", "/v5/market/account-ratio", params)
    
    async def get_fee_rate(self, symbol: str = None, category: str = "linear") -> Dict:
        """
        Get Trading Fee Rate - for calculating REAL profit after fees
        
        https://bybit-exchange.github.io/docs/v5/account/fee-rate
        
        Returns:
        - takerFeeRate: Fee when taking liquidity (market orders)
        - makerFeeRate: Fee when providing liquidity (limit orders)
        
        Example: takerFeeRate: 0.0006 = 0.06%
        On $1000 trade = $0.60 fee per side ($1.20 round trip)
        """
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/account/fee-rate", params, auth=True)
    
    async def get_historical_volatility(self, category: str = "option", base_coin: str = "BTC", period: int = 7) -> Dict:
        """
        Get Historical Volatility
        
        https://bybit-exchange.github.io/docs/v5/market/historical-volatility
        """
        params = {
            "category": category,
            "baseCoin": base_coin,
            "period": period
        }
        return await self._request("GET", "/v5/market/historical-volatility", params)
    
    async def get_recent_trades(self, symbol: str, category: str = "linear", limit: int = 60) -> Dict:
        """
        Get Recent Trades - useful for momentum detection
        
        Shows actual executed trades, not just orderbook
        
        https://bybit-exchange.github.io/docs/v5/market/recent-trade
        """
        params = {
            "category": category,
            "symbol": symbol,
            "limit": limit
        }
        return await self._request("GET", "/v5/market/recent-trade", params)
    
    async def get_risk_limit(self, symbol: str, category: str = "linear") -> Dict:
        """
        Get Risk Limit info for a symbol
        
        https://bybit-exchange.github.io/docs/v5/market/risk-limit
        """
        params = {
            "category": category,
            "symbol": symbol
        }
        return await self._request("GET", "/v5/market/risk-limit", params)
    
    async def get_insurance_fund(self, coin: str = "USDT") -> Dict:
        """
        Get Insurance Fund data
        
        https://bybit-exchange.github.io/docs/v5/market/insurance
        """
        params = {"coin": coin}
        return await self._request("GET", "/v5/market/insurance", params)
    
    async def get_delivery_price(self, category: str = "linear", symbol: str = None) -> Dict:
        """
        Get Delivery Price
        
        https://bybit-exchange.github.io/docs/v5/market/delivery-price
        """
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/market/delivery-price", params)
    
    async def get_execution_list(self, category: str = "linear", symbol: str = None, limit: int = 100) -> Dict:
        """
        Get Trade Execution List - your actual filled trades
        
        https://bybit-exchange.github.io/docs/v5/order/execution
        """
        params = {
            "category": category,
            "limit": limit
        }
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/execution/list", params, auth=True)
    
    async def get_transaction_log(self, category: str = "linear", limit: int = 50) -> Dict:
        """
        Get Transaction Log - all wallet movements (trading, funding, etc.)
        
        https://bybit-exchange.github.io/docs/v5/account/transaction-log
        """
        params = {
            "accountType": "UNIFIED",
            "category": category,
            "limit": limit
        }
        return await self._request("GET", "/v5/account/transaction-log", params, auth=True)
    
    async def get_all_symbols(self, category: str = "linear") -> List[str]:
        """
        Get ALL available trading symbols from Bybit
        Categories: linear (USDT perpetual), inverse, spot
        """
        try:
            params = {"category": category}
            response = await self._request("GET", "/v5/market/instruments-info", params)
            
            if response.get("success") and response.get("data"):
                symbols = []
                for item in response["data"].get("list", []):
                    symbol = item.get("symbol", "")
                    status = item.get("status", "")
                    # Only get active trading pairs
                    if status == "Trading" and symbol:
                        # For linear, only get USDT pairs
                        if category == "linear" and symbol.endswith("USDT"):
                            symbols.append(symbol)
                        elif category != "linear":
                            symbols.append(symbol)
                
                logger.info(f"Fetched {len(symbols)} active {category} trading pairs from Bybit")
                return symbols
            return []
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return []
    
    async def get_instrument_info(self, symbol: str, category: str = "linear") -> Dict:
        """Get detailed instrument info (min qty, tick size, etc.)"""
        params = {
            "category": category,
            "symbol": symbol
        }
        return await self._request("GET", "/v5/market/instruments-info", params)
        
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
        
    async def get_pnl(self, category: str = "linear", limit: int = 200, days: int = 7) -> Dict:
        """Get closed PnL history for last N days (max 200 trades)"""
        import time
        
        # Calculate start time (7 days ago by default)
        end_time = int(time.time() * 1000)  # Now in milliseconds
        start_time = end_time - (days * 24 * 60 * 60 * 1000)  # N days ago
        
        params = {
            "category": category,
            "limit": min(limit, 200),  # Bybit max is 200
            "startTime": str(start_time),
            "endTime": str(end_time)
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

