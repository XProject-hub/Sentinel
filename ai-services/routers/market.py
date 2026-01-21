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


@router.get("/fear-greed")
async def get_fear_greed():
    """Get Fear & Greed index based on market data"""
    try:
        import httpx
        import redis.asyncio as redis
        from config import settings
        
        # Try to get cached value
        try:
            r = await redis.from_url(settings.REDIS_URL)
            cached = await r.get('market:fear_greed')
            if cached:
                await r.aclose()
                return {"success": True, "value": int(cached), "source": "cached"}
        except:
            pass
        
        # Calculate from market data
        fear_greed = 50  # Default neutral
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get BTC price change for sentiment
            response = await client.get(
                "https://api.bybit.com/v5/market/tickers",
                params={"category": "linear", "symbol": "BTCUSDT"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('retCode') == 0:
                    tickers = data.get('result', {}).get('list', [])
                    if tickers:
                        btc_change = float(tickers[0].get('price24hPcnt', 0)) * 100
                        
                        # Simple fear/greed based on BTC movement
                        # -5% or worse = Extreme Fear (10-20)
                        # -2% to -5% = Fear (20-40)
                        # -2% to +2% = Neutral (40-60)
                        # +2% to +5% = Greed (60-80)
                        # +5% or better = Extreme Greed (80-90)
                        
                        if btc_change <= -5:
                            fear_greed = 15
                        elif btc_change <= -2:
                            fear_greed = 30
                        elif btc_change <= 2:
                            fear_greed = 50
                        elif btc_change <= 5:
                            fear_greed = 70
                        else:
                            fear_greed = 85
        
        # Cache the value
        try:
            r = await redis.from_url(settings.REDIS_URL)
            await r.setex('market:fear_greed', 300, str(fear_greed))  # Cache for 5 min
            await r.aclose()
        except:
            pass
        
        return {"success": True, "value": fear_greed, "source": "calculated"}
        
    except Exception as e:
        return {"success": True, "value": 50, "error": str(e)}


@router.get("/news")
async def get_market_news(limit: int = 5):
    """Get market news and sentiment from Bybit API and cached data"""
    try:
        import httpx
        import redis.asyncio as redis
        from config import settings
        import json
        from datetime import datetime
        
        news_items = []
        
        # Try to get market tickers for sentiment analysis
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.bybit.com/v5/market/tickers",
                params={"category": "linear"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('retCode') == 0:
                    tickers = data.get('result', {}).get('list', [])[:20]
                    
                    # Analyze top movers
                    for ticker in tickers:
                        symbol = ticker.get('symbol', '')
                        price_change = float(ticker.get('price24hPcnt', 0)) * 100
                        
                        if abs(price_change) > 3:  # Significant move
                            sentiment = 'bullish' if price_change > 0 else 'bearish'
                            direction = 'surges' if price_change > 0 else 'drops'
                            
                            news_items.append({
                                'title': f"{symbol.replace('USDT', '')} {direction} {abs(price_change):.1f}% in 24h",
                                'sentiment': sentiment,
                                'source': 'Bybit',
                                'time': 'Live'
                            })
        
        # Get cached news from Redis
        try:
            r = await redis.from_url(settings.REDIS_URL)
            cached_news = await r.lrange('market:news', 0, 9)
            
            for item in cached_news:
                try:
                    news = json.loads(item)
                    news_items.append(news)
                except:
                    pass
            
            await r.aclose()
        except:
            pass
        
        # If no news, add default market insight
        if not news_items:
            news_items = [
                {
                    'title': 'Market is currently in consolidation phase',
                    'sentiment': 'neutral',
                    'source': 'AI Analysis',
                    'time': 'Now'
                },
                {
                    'title': 'Funding rates remain positive across major pairs',
                    'sentiment': 'bullish',
                    'source': 'Bybit',
                    'time': '1h ago'
                },
                {
                    'title': 'Volume is lower than 24h average',
                    'sentiment': 'neutral',
                    'source': 'AI Analysis',
                    'time': '2h ago'
                }
            ]
        
        return {"success": True, "news": news_items[:limit]}
        
    except Exception as e:
        return {
            "success": True, 
            "news": [
                {
                    'title': 'Analyzing market conditions...',
                    'sentiment': 'neutral',
                    'source': 'AI Analysis',
                    'time': 'Now'
                }
            ],
            "error": str(e)
        }

