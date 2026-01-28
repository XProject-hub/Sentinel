"""Market Intelligence API Routes"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime

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


# ============================================
# LIQUIDATION HEATMAP ENDPOINTS
# ============================================

@router.get("/liquidation-heatmap/{symbol}")
async def get_liquidation_heatmap(symbol: str):
    """
    Get liquidation heatmap for a symbol
    
    Shows where liquidations would occur based on OI and leverage distribution.
    Useful for identifying support/resistance magnets.
    """
    try:
        from services.liquidation_heatmap import get_liquidation_heatmap
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        heatmap = await get_liquidation_heatmap()
        
        data = await heatmap.fetch_liquidation_data(symbol, client)
        
        if data:
            return {"success": True, "data": data}
        else:
            return {"success": False, "error": "Could not fetch liquidation data"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/liquidation-zones/{symbol}")
async def get_liquidation_zones(symbol: str):
    """
    Get simplified liquidation zones for trading decisions
    
    Returns key support/resistance magnets where large liquidations would occur.
    """
    try:
        from services.liquidation_heatmap import get_liquidation_heatmap
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        heatmap = await get_liquidation_heatmap()
        
        zones = await heatmap.get_liquidation_zones(symbol, client)
        
        return {"success": True, "data": zones}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/liquidation-bulk")
async def get_liquidation_bulk(symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT"):
    """
    Get liquidation summaries for multiple symbols
    
    Args:
        symbols: Comma-separated list of symbols
    """
    try:
        from services.liquidation_heatmap import get_liquidation_heatmap
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        heatmap = await get_liquidation_heatmap()
        
        symbol_list = [s.strip().upper() for s in symbols.split(',')][:10]
        
        results = {}
        for symbol in symbol_list:
            try:
                data = await heatmap.fetch_liquidation_data(symbol, client)
                if data and 'heatmap' in data:
                    summary = data['heatmap'].get('summary', {})
                    results[symbol] = {
                        'current_price': data.get('current_price'),
                        'open_interest_usd': data.get('open_interest_usd'),
                        'long_ratio': data.get('long_ratio'),
                        'short_ratio': data.get('short_ratio'),
                        'total_long_liqs': summary.get('total_long_liquidations_usd'),
                        'total_short_liqs': summary.get('total_short_liquidations_usd'),
                        'liq_imbalance': summary.get('liq_imbalance'),
                        'risk_assessment': summary.get('risk_assessment')
                    }
            except:
                pass
                
        return {"success": True, "data": results, "count": len(results)}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================
# LONG/SHORT RATIO ENDPOINTS
# ============================================

@router.get("/long-short-ratio/{symbol}")
async def get_long_short_ratio(symbol: str, period: str = "5min", limit: int = 24):
    """
    Get long/short ratio history for a symbol
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        period: '5min', '15min', '30min', '1h', '4h', '1d'
        limit: Number of data points (max 50)
    """
    try:
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        result = await client.get_long_short_ratio(symbol=symbol, period=period, limit=min(limit, 50))
        
        if not result.get('success'):
            return {"success": False, "error": "Failed to fetch data"}
            
        ls_list = result.get('data', {}).get('list', [])
        
        data = []
        for item in ls_list:
            data.append({
                'timestamp': item.get('timestamp'),
                'buy_ratio': float(item.get('buyRatio', 0.5)),
                'sell_ratio': float(item.get('sellRatio', 0.5)),
                'long_pct': round(float(item.get('buyRatio', 0.5)) * 100, 2),
                'short_pct': round(float(item.get('sellRatio', 0.5)) * 100, 2)
            })
        
        # Calculate current sentiment
        if data:
            current = data[0]
            sentiment = 'neutral'
            if current['long_pct'] > 55:
                sentiment = 'crowded_long'
            elif current['long_pct'] > 52:
                sentiment = 'slightly_long'
            elif current['short_pct'] > 55:
                sentiment = 'crowded_short'
            elif current['short_pct'] > 52:
                sentiment = 'slightly_short'
        else:
            sentiment = 'unknown'
            current = {'long_pct': 50, 'short_pct': 50}
        
        return {
            "success": True,
            "symbol": symbol,
            "current": current,
            "sentiment": sentiment,
            "history": data,
            "insight": f"Market is {sentiment.replace('_', ' ')} - {'contrarian short may be favorable' if sentiment == 'crowded_long' else 'contrarian long may be favorable' if sentiment == 'crowded_short' else 'balanced positioning'}"
        }
            
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/long-short-bulk")
async def get_ls_ratio_bulk(symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT"):
    """
    Get current long/short ratios for multiple symbols
    """
    from loguru import logger
    try:
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        symbol_list = [s.strip().upper() for s in symbols.split(',')][:15]
        
        results = {}
        errors = []
        
        for symbol in symbol_list:
            try:
                result = await client.get_long_short_ratio(symbol=symbol, period="5min", limit=1)
                
                if result.get('success'):
                    ls_list = result.get('data', {}).get('list', [])
                    if ls_list:
                        item = ls_list[0]
                        long_pct = round(float(item.get('buyRatio', 0.5)) * 100, 2)
                        short_pct = round(float(item.get('sellRatio', 0.5)) * 100, 2)
                        
                        results[symbol] = {
                            'long_pct': long_pct,
                            'short_pct': short_pct,
                            'sentiment': 'crowded_long' if long_pct > 55 else ('crowded_short' if short_pct > 55 else 'balanced')
                        }
                    else:
                        # No data for this symbol - use default 50/50
                        results[symbol] = {
                            'long_pct': 50.0,
                            'short_pct': 50.0,
                            'sentiment': 'balanced'
                        }
                        errors.append(f"{symbol}: no data")
                else:
                    # API returned error - use default
                    results[symbol] = {
                        'long_pct': 50.0,
                        'short_pct': 50.0,
                        'sentiment': 'balanced'
                    }
                    errors.append(f"{symbol}: {result.get('error', 'API error')}")
            except Exception as e:
                # Exception - use default
                results[symbol] = {
                    'long_pct': 50.0,
                    'short_pct': 50.0,
                    'sentiment': 'balanced'
                }
                errors.append(f"{symbol}: {str(e)}")
        
        if errors:
            logger.debug(f"L/S Ratio errors: {errors}")
                
        return {"success": True, "data": results, "count": len(results), "errors": errors if errors else None}
        
    except Exception as e:
        logger.error(f"L/S Ratio bulk error: {e}")
        return {"success": False, "error": str(e)}


# ============================================
# FUNDING ARBITRAGE ENDPOINTS
# ============================================

@router.get("/funding-arbitrage")
async def get_funding_arbitrage_opportunities(limit: int = 10):
    """
    Get current funding rate arbitrage opportunities
    
    Returns coins with high funding rates suitable for delta-neutral yield farming.
    """
    try:
        from services.funding_arbitrage import get_funding_arbitrage
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        arb = await get_funding_arbitrage()
        
        # Scan for opportunities
        opportunities = await arb.scan_opportunities(client)
        
        # Format response
        data = []
        for opp in opportunities[:limit]:
            data.append({
                'symbol': opp.symbol,
                'funding_rate': opp.funding_rate,
                'funding_rate_pct': round(opp.funding_rate * 100, 4),
                'direction': opp.direction,
                'spot_action': opp.spot_action,
                'daily_yield_pct': opp.estimated_daily_yield,
                'monthly_yield_pct': opp.estimated_monthly_yield,
                'annualized_pct': round(opp.funding_rate_annualized * 100, 2),
                'hours_until_funding': opp.hours_until_funding,
                'risk_level': opp.risk_level,
                'recommendation': opp.recommendation
            })
        
        return {
            "success": True,
            "opportunities": data,
            "count": len(data),
            "explanation": "Delta-neutral strategy: Open opposite positions on spot + perp to collect funding without directional risk."
        }
            
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/funding-arbitrage/best")
async def get_best_funding_opportunity():
    """Get the single best funding arbitrage opportunity right now"""
    try:
        from services.funding_arbitrage import get_funding_arbitrage
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        arb = await get_funding_arbitrage()
        
        # Scan and get best
        await arb.scan_opportunities(client)
        best = await arb.get_best_opportunity()
        
        if best:
            return {
                "success": True,
                "opportunity": best,
                "action_plan": f"1. {best.get('spot_action', '').replace('_', ' ').title()} {best.get('symbol', '')}\n"
                             f"2. Open {best.get('direction', '').replace('_', ' ')} position\n"
                             f"3. Collect {best.get('funding_rate_pct', 0):.3f}% every 8 hours"
            }
        else:
            return {"success": True, "opportunity": None, "message": "No good opportunities at the moment"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/funding-arbitrage/calculate")
async def calculate_funding_position(symbol: str, capital: float = 1000, leverage: float = 1.0):
    """
    Calculate position size and expected returns for funding arbitrage
    
    Args:
        symbol: Trading pair
        capital: Capital to allocate (USD)
        leverage: Leverage to use (1.0 = no leverage)
    """
    try:
        from services.funding_arbitrage import get_funding_arbitrage
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        arb = await get_funding_arbitrage()
        
        # Get current funding rate
        ticker_result = await client.get_tickers(symbol=symbol, category="linear")
        if not ticker_result.get('success'):
            return {"success": False, "error": "Could not fetch ticker"}
            
        ticker_list = ticker_result.get('data', {}).get('list', [])
        if not ticker_list:
            return {"success": False, "error": "Symbol not found"}
            
        funding_rate = float(ticker_list[0].get('fundingRate', 0))
        
        # Calculate position
        calc = arb.calculate_position_size(capital, funding_rate, leverage)
        
        direction = "SHORT perp + LONG spot" if funding_rate > 0 else "LONG perp + SHORT spot"
        
        return {
            "success": True,
            "symbol": symbol,
            "funding_rate": funding_rate,
            "funding_rate_pct": round(funding_rate * 100, 4),
            "direction": direction,
            "calculation": calc
        }
            
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================
# COMBINED MARKET INTELLIGENCE
# ============================================

@router.get("/intelligence/{symbol}")
async def get_full_market_intelligence(symbol: str):
    """
    Get complete market intelligence for a symbol
    
    Combines: OI analysis, liquidation heatmap, L/S ratio, funding rate
    """
    try:
        from services.open_interest_tracker import get_oi_tracker
        from services.liquidation_heatmap import get_liquidation_heatmap
        from services.bybit_client import BybitV5Client
        
        client = BybitV5Client()
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Open Interest
        try:
            oi_tracker = await get_oi_tracker()
            oi_analysis = await oi_tracker.update_oi_data(symbol, client)
            if oi_analysis:
                result['open_interest'] = {
                    'signal': oi_analysis.signal.value,
                    'oi_change_pct': oi_analysis.oi_change_pct,
                    'price_change_pct': oi_analysis.price_change_pct,
                    'recommendation': oi_analysis.recommendation,
                    'reasoning': oi_analysis.reasoning
                }
        except:
            result['open_interest'] = None
            
        # Liquidation Heatmap
        try:
            liq_heatmap = await get_liquidation_heatmap()
            liq_data = await liq_heatmap.fetch_liquidation_data(symbol, client)
            if liq_data and 'heatmap' in liq_data:
                result['liquidation'] = {
                    'current_price': liq_data.get('current_price'),
                    'open_interest_usd': liq_data.get('open_interest_usd'),
                    'summary': liq_data['heatmap'].get('summary')
                }
        except:
            result['liquidation'] = None
            
        # Long/Short Ratio
        try:
            ls_result = await client.get_long_short_ratio(symbol=symbol, period="5min", limit=1)
            if ls_result.get('success'):
                ls_list = ls_result.get('data', {}).get('list', [])
                if ls_list:
                    item = ls_list[0]
                    long_pct = round(float(item.get('buyRatio', 0.5)) * 100, 2)
                    result['long_short_ratio'] = {
                        'long_pct': long_pct,
                        'short_pct': 100 - long_pct,
                        'sentiment': 'crowded_long' if long_pct > 55 else ('crowded_short' if long_pct < 45 else 'balanced')
                    }
        except:
            result['long_short_ratio'] = None
            
        # Funding Rate
        try:
            ticker_result = await client.get_tickers(symbol=symbol, category="linear")
            if ticker_result.get('success'):
                ticker_list = ticker_result.get('data', {}).get('list', [])
                if ticker_list:
                    ticker = ticker_list[0]
                    funding = float(ticker.get('fundingRate', 0))
                    result['funding'] = {
                        'rate': funding,
                        'rate_pct': round(funding * 100, 4),
                        'annualized_pct': round(funding * 3 * 365 * 100, 2),
                        'sentiment': 'longs_pay' if funding > 0 else ('shorts_pay' if funding < 0 else 'neutral')
                    }
        except:
            result['funding'] = None
        
        return {"success": True, "data": result}
            
    except Exception as e:
        return {"success": False, "error": str(e)}
