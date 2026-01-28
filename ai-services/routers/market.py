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


@router.get("/open-interest/{symbol}")
async def get_open_interest(symbol: str):
    """Get Open Interest analysis for a symbol"""
    try:
        from services.open_interest_tracker import get_oi_tracker
        from services.bybit_client import BybitV5Client
        
        # Create a temporary client for the API call
        client = BybitV5Client()
        
        oi_tracker = await get_oi_tracker()
        analysis = await oi_tracker.update_oi_data(symbol, client)
        
        if analysis:
            return {
                "success": True,
                "data": {
                    "symbol": analysis.symbol,
                    "signal": analysis.signal.value,
                    "oi_change_pct": analysis.oi_change_pct,
                    "price_change_pct": analysis.price_change_pct,
                    "oi_trend": analysis.oi_trend,
                    "confidence": analysis.confidence,
                    "recommendation": analysis.recommendation,
                    "reasoning": analysis.reasoning,
                    "timestamp": analysis.timestamp.isoformat()
                }
            }
        else:
            return {"success": False, "error": "Could not fetch OI data"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/open-interest-signals")
async def get_oi_signals(limit: int = 10, signal_type: str = None):
    """
    Get top Open Interest signals across all symbols
    
    Args:
        limit: Number of signals to return (default 10)
        signal_type: Filter by signal type (strong_bullish, strong_bearish, accumulation, etc.)
    """
    try:
        from services.open_interest_tracker import get_oi_tracker, OISignal
        
        oi_tracker = await get_oi_tracker()
        
        # Parse signal type filter if provided
        filter_signal = None
        if signal_type:
            try:
                filter_signal = OISignal(signal_type)
            except ValueError:
                pass
        
        signals = await oi_tracker.get_top_signals(limit=limit, signal_type=filter_signal)
        
        return {
            "success": True,
            "signals": signals,
            "count": len(signals)
        }
        
    except Exception as e:
        return {"success": True, "signals": [], "error": str(e)}


@router.get("/open-interest-bulk")
async def get_oi_bulk(symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT"):
    """
    Get OI analysis for multiple symbols at once
    
    Args:
        symbols: Comma-separated list of symbols
    """
    try:
        from services.open_interest_tracker import get_oi_tracker
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        oi_tracker = await get_oi_tracker()
        
        symbol_list = [s.strip().upper() for s in symbols.split(',')][:20]  # Max 20
        
        results = {}
        for symbol in symbol_list:
            try:
                analysis = await oi_tracker.update_oi_data(symbol, client)
                if analysis:
                    results[symbol] = {
                        "signal": analysis.signal.value,
                        "oi_change_pct": analysis.oi_change_pct,
                        "price_change_pct": analysis.price_change_pct,
                        "oi_trend": analysis.oi_trend,
                        "confidence": analysis.confidence,
                        "recommendation": analysis.recommendation,
                        "reasoning": analysis.reasoning
                    }
            except:
                pass
                
        return {
            "success": True,
            "data": results,
            "count": len(results)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/fear-greed")
async def get_fear_greed():
    """Get REAL Fear & Greed Index from Alternative.me API"""
    try:
        import httpx
        import redis.asyncio as redis
        from config import settings
        
        # Try to get cached value (cache for 10 min)
        try:
            r = await redis.from_url(settings.REDIS_URL)
            cached = await r.get('market:fear_greed')
            if cached:
                await r.aclose()
                return {"success": True, "value": int(cached), "source": "cached"}
        except:
            pass
        
        fear_greed = 50  # Default neutral
        source = "default"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # TRY 1: Alternative.me API (the REAL Fear & Greed Index)
            try:
                response = await client.get("https://api.alternative.me/fng/")
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data') and len(data['data']) > 0:
                        fear_greed = int(data['data'][0].get('value', 50))
                        source = "alternative.me"
            except:
                pass
            
            # TRY 2: Calculate from BTC if Alternative.me fails
            if source == "default":
                try:
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
                                
                                # Calculate F/G from BTC movement
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
                                source = "btc_calculated"
                except:
                    pass
        
        # Cache the value for 10 minutes
        try:
            r = await redis.from_url(settings.REDIS_URL)
            await r.setex('market:fear_greed', 600, str(fear_greed))
            await r.aclose()
        except:
            pass
        
        return {"success": True, "value": fear_greed, "source": source}
        
    except Exception as e:
        return {"success": True, "value": 50, "error": str(e)}


@router.get("/news")
async def get_market_news(limit: int = 10):
    """Get market news and sentiment from Bybit API and market analysis"""
    try:
        import httpx
        import redis.asyncio as redis
        from config import settings
        import json
        from datetime import datetime
        
        news_items = []
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get all tickers for comprehensive analysis
            response = await client.get(
                "https://api.bybit.com/v5/market/tickers",
                params={"category": "linear"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('retCode') == 0:
                    tickers = data.get('result', {}).get('list', [])
                    
                    # Sort by price change to get top movers
                    sorted_by_change = sorted(tickers, key=lambda x: abs(float(x.get('price24hPcnt', 0))), reverse=True)
                    
                    # Top gainers
                    gainers = [t for t in sorted_by_change if float(t.get('price24hPcnt', 0)) > 0][:5]
                    for ticker in gainers:
                        symbol = ticker.get('symbol', '')
                        price_change = float(ticker.get('price24hPcnt', 0)) * 100
                        if price_change > 1.5:
                            news_items.append({
                                'title': f"{symbol.replace('USDT', '')} surges +{price_change:.1f}% in 24h",
                                'sentiment': 'bullish',
                                'source': 'Bybit Market',
                                'time': 'Live'
                            })
                    
                    # Top losers
                    losers = [t for t in sorted_by_change if float(t.get('price24hPcnt', 0)) < 0][:5]
                    for ticker in losers:
                        symbol = ticker.get('symbol', '')
                        price_change = float(ticker.get('price24hPcnt', 0)) * 100
                        if price_change < -1.5:
                            news_items.append({
                                'title': f"{symbol.replace('USDT', '')} drops {price_change:.1f}% in 24h",
                                'sentiment': 'bearish',
                                'source': 'Bybit Market',
                                'time': 'Live'
                            })
                    
                    # Volume leaders
                    sorted_by_volume = sorted(tickers, key=lambda x: float(x.get('turnover24h', 0)), reverse=True)[:3]
                    for ticker in sorted_by_volume:
                        symbol = ticker.get('symbol', '')
                        volume = float(ticker.get('turnover24h', 0))
                        if volume > 100000000:  # > 100M volume
                            news_items.append({
                                'title': f"{symbol.replace('USDT', '')} sees ${volume/1000000:.0f}M in 24h volume",
                                'sentiment': 'neutral',
                                'source': 'Volume Alert',
                                'time': 'Live'
                            })
            
            # Get funding rates for major pairs
            major_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            for symbol in major_pairs:
                try:
                    fund_response = await client.get(
                        "https://api.bybit.com/v5/market/funding/history",
                        params={"category": "linear", "symbol": symbol, "limit": 1}
                    )
                    if fund_response.status_code == 200:
                        fund_data = fund_response.json()
                        if fund_data.get('retCode') == 0:
                            funding_list = fund_data.get('result', {}).get('list', [])
                            if funding_list:
                                rate = float(funding_list[0].get('fundingRate', 0)) * 100
                                if abs(rate) > 0.01:
                                    sentiment = 'bullish' if rate > 0 else 'bearish'
                                    direction = 'positive' if rate > 0 else 'negative'
                                    news_items.append({
                                        'title': f"{symbol.replace('USDT', '')} funding rate {direction}: {rate:.3f}%",
                                        'sentiment': sentiment,
                                        'source': 'Funding Rate',
                                        'time': 'Live'
                                    })
                except:
                    pass
        
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
        
        # If still no news, add default insights
        if len(news_items) < 3:
            default_news = [
                {'title': 'Market consolidating near key levels', 'sentiment': 'neutral', 'source': 'AI Analysis', 'time': 'Now'},
                {'title': 'Crypto market cap stable at current levels', 'sentiment': 'neutral', 'source': 'Market Overview', 'time': '1h ago'},
                {'title': 'Institutional interest remains strong', 'sentiment': 'bullish', 'source': 'Analysis', 'time': '2h ago'},
            ]
            news_items.extend(default_news)
        
        return {"success": True, "news": news_items[:limit]}
        
    except Exception as e:
        return {
            "success": True, 
            "news": [
                {'title': 'Analyzing market conditions...', 'sentiment': 'neutral', 'source': 'AI Analysis', 'time': 'Now'},
                {'title': 'Loading market data...', 'sentiment': 'neutral', 'source': 'System', 'time': 'Now'}
            ],
            "error": str(e)
        }

