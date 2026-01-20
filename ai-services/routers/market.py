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


@router.get("/whale-alerts")
async def get_whale_alerts():
    """Get recent whale activity alerts"""
    try:
        from services.whale_tracker import whale_tracker
        import redis.asyncio as redis
        from config import settings
        import json
        
        # Get alerts from Redis
        r = await redis.from_url(settings.REDIS_URL)
        
        all_alerts = []
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
        
        for symbol in symbols:
            try:
                alerts_data = await r.lrange(f"whale:alerts:{symbol}", 0, 4)
                for alert_json in alerts_data:
                    alert = json.loads(alert_json)
                    all_alerts.append(alert)
            except:
                pass
        
        # Sort by timestamp (newest first)
        all_alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        await r.aclose()
        
        return {"success": True, "alerts": all_alerts[:15]}
        
    except Exception as e:
        return {"success": True, "alerts": [], "error": str(e)}


@router.get("/funding-rates")
async def get_funding_rates():
    """Get current funding rates for perpetual futures"""
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 
                      'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT']
            
            rates = {}
            
            for symbol in symbols:
                try:
                    response = await client.get(
                        f"https://api.bybit.com/v5/market/funding/history",
                        params={"category": "linear", "symbol": symbol, "limit": 1}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('retCode') == 0:
                            funding_list = data.get('result', {}).get('list', [])
                            if funding_list:
                                rates[symbol] = {
                                    'symbol': symbol,
                                    'funding_rate': float(funding_list[0].get('fundingRate', 0)),
                                    'funding_rate_timestamp': funding_list[0].get('fundingRateTimestamp')
                                }
                except:
                    pass
            
            return {"success": True, "rates": rates}
            
    except Exception as e:
        return {"success": True, "rates": {}, "error": str(e)}

