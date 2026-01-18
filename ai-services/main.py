"""
SENTINEL AI - Ultimate AI Trading Platform
Professional Hedge-Fund Level Autonomous Trading System

Components:
- RegimeDetector: HMM-inspired market state detection
- EdgeEstimator: Statistical edge calculation
- PositionSizer: Kelly-based dynamic sizing
- MarketScanner: Scans ALL 500+ Bybit pairs
- LearningEngine: Continuous self-improvement
- AICoordinator: Combines all AI models
- AutonomousTraderV2: The ultimate trading system

This is what a REAL trading system looks like.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from loguru import logger

from config import settings

# === CORE SERVICES ===
from services.market_intelligence import MarketIntelligenceService
from services.sentiment_analyzer import SentimentAnalyzer
from services.strategy_planner import StrategyPlanner
from services.risk_engine import RiskEngine
from services.trading_executor import TradingExecutor
from services.websocket_manager import WebSocketManager
from services.data_aggregator import DataAggregator
from services.learning_engine import LearningEngine

# === ADVANCED AI COMPONENTS ===
from services.regime_detector import RegimeDetector
from services.edge_estimator import EdgeEstimator
from services.position_sizer import PositionSizer
from services.market_scanner import MarketScanner

# === TRADERS ===
from services.autonomous_trader import autonomous_trader  # Legacy v1
from services.autonomous_trader_v2 import autonomous_trader_v2  # Ultimate v2

# === ROUTERS ===
from routers import market, sentiment, strategy, risk, trading, exchange, admin, data

# Initialize services
market_intelligence = MarketIntelligenceService()
sentiment_analyzer = SentimentAnalyzer()
strategy_planner = StrategyPlanner()
risk_engine = RiskEngine()
trading_executor = TradingExecutor()
ws_manager = WebSocketManager()
data_aggregator = DataAggregator()
learning_engine = LearningEngine()

# Initialize advanced AI components
regime_detector = RegimeDetector()
edge_estimator = EdgeEstimator()
position_sizer = PositionSizer()
market_scanner = MarketScanner()

# Use V2 by default
USE_V2_TRADER = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("=" * 60)
    logger.info("SENTINEL AI - Ultimate Trading Platform Starting...")
    logger.info("=" * 60)
    
    # === Initialize Core Services ===
    logger.info("Initializing core services...")
    await market_intelligence.initialize()
    await sentiment_analyzer.initialize()
    await strategy_planner.initialize()
    await risk_engine.initialize()
    await data_aggregator.initialize()
    await learning_engine.initialize()
    
    # === Initialize Advanced AI Components ===
    logger.info("Initializing advanced AI components...")
    await regime_detector.initialize()
    await edge_estimator.initialize()
    await position_sizer.initialize()
    await market_scanner.initialize(regime_detector, edge_estimator)
    
    # === Initialize Trader ===
    if USE_V2_TRADER:
        logger.info("Initializing Ultimate Autonomous Trader v2.0...")
        await autonomous_trader_v2.initialize(
            regime_detector=regime_detector,
            edge_estimator=edge_estimator,
            position_sizer=position_sizer,
            market_scanner=market_scanner,
            learning_engine=learning_engine
        )
    else:
        logger.info("Initializing Legacy Autonomous Trader v1...")
        await autonomous_trader.initialize(learning_engine)
    
    # === Start Background Tasks ===
    logger.info("Starting background tasks...")
    asyncio.create_task(market_intelligence.start_data_collection())
    asyncio.create_task(sentiment_analyzer.start_news_monitoring())
    asyncio.create_task(data_aggregator.start_collection())
    asyncio.create_task(run_main_loop())
    
    # Start the appropriate trader
    if USE_V2_TRADER:
        asyncio.create_task(autonomous_trader_v2.run_trading_loop())
    else:
        asyncio.create_task(autonomous_trader.run_trading_loop())
    
    # Auto-reconnect users
    asyncio.create_task(auto_reconnect_on_startup())
    
    logger.info("=" * 60)
    logger.info("SENTINEL AI Services Ready - Ultimate Trading Active")
    logger.info("=" * 60)
    
    yield
    
    # Cleanup on shutdown
    logger.info("SENTINEL AI Services Shutting Down...")
    await market_intelligence.shutdown()
    await sentiment_analyzer.shutdown()
    await data_aggregator.shutdown()
    await learning_engine.shutdown()
    await regime_detector.shutdown()
    await edge_estimator.shutdown()
    await position_sizer.shutdown()
    await market_scanner.shutdown()
    
    if USE_V2_TRADER:
        await autonomous_trader_v2.shutdown()
    else:
        await autonomous_trader.shutdown()


async def auto_reconnect_on_startup():
    """Auto-reconnect all users with saved trading credentials"""
    await asyncio.sleep(5)  # Wait for services to fully initialize
    
    try:
        import redis.asyncio as aioredis
        import base64
        r = await aioredis.from_url(settings.REDIS_URL)
        reconnected = 0
        
        # Check exchange:credentials:default
        creds = await r.hgetall("exchange:credentials:default")
        if creds:
            api_key_enc = creds.get(b"api_key", b"").decode() if creds.get(b"api_key") else ""
            api_secret_enc = creds.get(b"api_secret", b"").decode() if creds.get(b"api_secret") else ""
            
            if api_key_enc and api_secret_enc:
                try:
                    api_key = base64.b64decode(api_key_enc.encode()).decode()
                    api_secret = base64.b64decode(api_secret_enc.encode()).decode()
                    
                    if api_key and api_secret:
                        from services.bybit_client import BybitV5Client
                        from routers.exchange import exchange_connections
                        
                        client = BybitV5Client(api_key, api_secret)
                        result = await client.test_connection()
                        
                        if result.get("success"):
                            exchange_connections["default"] = client
                            
                            # Connect to the appropriate trader
                            if USE_V2_TRADER:
                                success = await autonomous_trader_v2.connect_user(
                                    user_id="default",
                                    api_key=api_key,
                                    api_secret=api_secret,
                                )
                            else:
                                success = await autonomous_trader.connect_user(
                                    user_id="default",
                                    api_key=api_key,
                                    api_secret=api_secret,
                                )
                                
                            if success:
                                reconnected += 1
                                logger.info("Auto-reconnected exchange from saved credentials!")
                        else:
                            logger.warning(f"Auto-reconnect failed: {result.get('error')}")
                except Exception as e:
                    logger.error(f"Failed to decode credentials: {e}")
        
        await r.aclose()
        
        if reconnected > 0:
            logger.info(f"Auto-reconnected {reconnected} users for autonomous trading")
        else:
            logger.info("No users to auto-reconnect (new install or no saved credentials)")
            
    except Exception as e:
        logger.error(f"Auto-reconnect on startup failed: {e}")


async def run_main_loop():
    """Main AI processing loop"""
    while True:
        try:
            # Get current market state
            market_state = await market_intelligence.get_current_state()
            
            # Get sentiment data
            sentiment_data = await sentiment_analyzer.get_current_sentiment()
            
            # Detect market regime
            regime = await strategy_planner.detect_regime(market_state, sentiment_data)
            
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(10)


# Create FastAPI app
app = FastAPI(
    title="SENTINEL AI - Ultimate Trading Platform",
    description="Professional Hedge-Fund Level Autonomous Trading System",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(market.router, prefix="/ai/market", tags=["Market Intelligence"])
app.include_router(sentiment.router, prefix="/ai/sentiment", tags=["Sentiment Analysis"])
app.include_router(strategy.router, prefix="/ai/strategy", tags=["Strategy Planning"])
app.include_router(risk.router, prefix="/ai/risk", tags=["Risk Management"])
app.include_router(trading.router, prefix="/ai/trading", tags=["Trading"])
app.include_router(exchange.router, prefix="/ai/exchange", tags=["Exchange Connection"])
app.include_router(admin.router, prefix="/ai/admin", tags=["Admin"])
app.include_router(data.router, tags=["Data & News"])


# ============================================
# WEBSOCKET ENDPOINTS
# ============================================

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Real-time WebSocket connection for user dashboard"""
    await ws_manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await ws_manager.receive(websocket)
            
            if data.get("type") == "ping":
                await ws_manager.send(user_id, {"type": "pong"})
            
    except WebSocketDisconnect:
        ws_manager.disconnect(user_id)
        logger.info(f"User {user_id} disconnected")


# ============================================
# HEALTH & STATUS ENDPOINTS
# ============================================

@app.get("/ai/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "trader_version": "v2" if USE_V2_TRADER else "v1",
        "services": {
            "market_intelligence": market_intelligence.is_running,
            "sentiment_analyzer": sentiment_analyzer.is_running,
            "regime_detector": regime_detector is not None,
            "edge_estimator": edge_estimator is not None,
            "position_sizer": position_sizer is not None,
            "market_scanner": market_scanner is not None,
        }
    }


@app.get("/ai/status")
async def get_system_status():
    """Get overall AI system status"""
    trader_status = await autonomous_trader_v2.get_status() if USE_V2_TRADER else {}
    
    return {
        "active": True,
        "trader_version": "v2" if USE_V2_TRADER else "v1",
        "market_data_age_seconds": await market_intelligence.get_data_age(),
        "sentiment_data_age_seconds": await sentiment_analyzer.get_data_age(),
        "trader": trader_status
    }


@app.get("/ai/regime/{symbol}")
async def get_market_regime(symbol: str):
    """Get current market regime for a symbol using advanced detector"""
    regime = await regime_detector.detect_regime(symbol)
    
    return {
        "symbol": symbol,
        "regime": regime.regime,
        "confidence": regime.confidence,
        "stability": regime.stability,
        "volatility": regime.volatility,
        "liquidity_score": regime.liquidity_score,
        "trend_strength": regime.trend_strength,
        "trend_direction": regime.trend_direction,
        "volume_profile": regime.volume_profile,
        "recommended_action": regime.recommended_action,
        "duration_minutes": regime.duration_minutes,
        "timestamp": regime.timestamp
    }


@app.get("/ai/edge/{symbol}")
async def get_edge_score(symbol: str, direction: str = "long"):
    """Get edge score for a trading opportunity"""
    edge = await edge_estimator.calculate_edge(symbol, direction)
    
    return {
        "symbol": symbol,
        "direction": direction,
        "edge": edge.edge,
        "confidence": edge.confidence,
        "components": {
            "technical": edge.technical_edge,
            "regime": edge.regime_edge,
            "momentum": edge.momentum_edge,
            "volume": edge.volume_edge,
            "correlation": edge.correlation_edge,
            "sentiment": edge.sentiment_edge
        },
        "expected_return": edge.expected_return,
        "risk_reward_ratio": edge.risk_reward_ratio,
        "win_probability": edge.win_probability,
        "kelly_fraction": edge.kelly_fraction,
        "recommended_size": edge.recommended_size,
        "reasons": edge.reasons,
        "warnings": edge.warnings,
        "timestamp": edge.timestamp
    }


@app.get("/ai/opportunities")
async def get_trading_opportunities(limit: int = 20):
    """Get top trading opportunities from scanner"""
    opportunities = await market_scanner.get_top_opportunities(limit)
    
    return {
        "success": True,
        "count": len(opportunities),
        "opportunities": [
            {
                "symbol": o.symbol,
                "direction": o.direction,
                "edge_score": o.edge_score,
                "confidence": o.confidence,
                "opportunity_score": o.opportunity_score,
                "regime": o.regime,
                "regime_action": o.regime_action,
                "current_price": o.current_price,
                "price_change_24h": o.price_change_24h,
                "volume_24h": o.volume_24h,
                "should_trade": o.should_trade,
                "reasons": o.reasons[:5],
                "warnings": o.warnings
            }
            for o in opportunities
        ]
    }


@app.get("/ai/risk-status")
async def get_risk_status():
    """Get current risk management status"""
    try:
        import redis.asyncio as aioredis
        r = await aioredis.from_url(settings.REDIS_URL)
        equity = await r.get('wallet:equity')
        wallet_balance = float(equity) if equity else 0
        await r.aclose()
        
        status = await position_sizer.get_risk_status(wallet_balance)
        return {"success": True, "data": status}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/ai/insight")
async def get_ai_insight():
    """Get AI-generated market insight and confidence"""
    try:
        # Get current market state
        market_state = await market_intelligence.get_current_state()
        sentiment_data = await sentiment_analyzer.get_current_sentiment()
        aggregated_data = await data_aggregator.get_aggregated_data()
        
        # Detect regime for BTC (primary indicator)
        btc_regime = await regime_detector.detect_regime('BTCUSDT')
        
        # Calculate overall AI confidence
        confidence = btc_regime.confidence / 100
        
        # Get AI insight text
        insight = await data_aggregator.get_market_insight()
        
        # Determine risk status
        if btc_regime.volatility > 4.0:
            risk_status = 'CAUTION'
        elif btc_regime.volatility > 2.5:
            risk_status = 'ELEVATED'
        else:
            risk_status = 'SAFE'
            
        # Get Fear & Greed
        fng = aggregated_data.get('fear_greed', {})
        
        return {
            "success": True,
            "data": {
                "confidence": round(btc_regime.confidence, 1),
                "confidence_label": "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low",
                "risk_status": risk_status,
                "insight": insight,
                "regime": btc_regime.regime,
                "volatility": round(btc_regime.volatility, 2),
                "trend": btc_regime.trend_direction,
                "trend_strength": round(btc_regime.trend_strength, 2),
                "fear_greed_index": int(fng.get('value', 50)),
                "fear_greed_label": fng.get('classification', 'Neutral'),
                "recommended_action": btc_regime.recommended_action,
                "timestamp": btc_regime.timestamp
            }
        }
    except Exception as e:
        logger.error(f"AI insight error: {e}")
        return {
            "success": True,
            "data": {
                "confidence": 50.0,
                "confidence_label": "Medium",
                "risk_status": "SAFE",
                "insight": "Initializing market analysis...",
                "regime": "initializing",
                "volatility": 0,
                "trend": "neutral",
                "trend_strength": 0,
                "fear_greed_index": 50,
                "fear_greed_label": "Neutral",
                "recommended_action": "hold",
                "timestamp": None
            }
        }


@app.get("/ai/aggregated-data")
async def get_all_aggregated_data():
    """Get all aggregated market data"""
    data = await data_aggregator.get_aggregated_data()
    return {"success": True, "data": data}


@app.get("/ai/learning/stats")
async def get_learning_statistics():
    """Get AI learning statistics"""
    stats = await learning_engine.get_learning_stats()
    return {"success": True, "data": stats}


@app.get("/ai/learning/events")
async def get_learning_events(limit: int = 20):
    """Get recent AI learning events"""
    events = await learning_engine.get_recent_learning_events(limit)
    return {"success": True, "data": events}


@app.get("/ai/learning/strategy/{regime}")
async def get_best_strategy_for_regime(regime: str):
    """Get AI-recommended strategy for a market regime"""
    strategy, q_value = learning_engine.get_best_strategy(regime)
    confidence = learning_engine.get_strategy_confidence(regime, strategy)
    
    return {
        "success": True,
        "data": {
            "regime": regime,
            "recommended_strategy": strategy,
            "confidence": confidence,
            "q_value": q_value,
            "exploration_rate": learning_engine.exploration_rate * 100,
        }
    }


@app.get("/ai/trader/status")
async def get_trader_status():
    """Get ultimate trader status"""
    if USE_V2_TRADER:
        status = await autonomous_trader_v2.get_status()
    else:
        status = {
            "is_running": autonomous_trader.is_running,
            "connected_users": len(autonomous_trader.user_clients)
        }
    return {"success": True, "data": status}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
