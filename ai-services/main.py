"""
SENTINEL AI - Core AI Services
Market Intelligence, Sentiment Analysis, Strategy Planning, Risk Management
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from loguru import logger

from config import settings
from services.market_intelligence import MarketIntelligenceService
from services.sentiment_analyzer import SentimentAnalyzer
from services.strategy_planner import StrategyPlanner
from services.risk_engine import RiskEngine
from services.trading_executor import TradingExecutor
from services.websocket_manager import WebSocketManager
from routers import market, sentiment, strategy, risk, trading, exchange, admin

# Initialize services
market_intelligence = MarketIntelligenceService()
sentiment_analyzer = SentimentAnalyzer()
strategy_planner = StrategyPlanner()
risk_engine = RiskEngine()
trading_executor = TradingExecutor()
ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("SENTINEL AI Services Starting...")
    
    # Initialize all services
    await market_intelligence.initialize()
    await sentiment_analyzer.initialize()
    await strategy_planner.initialize()
    await risk_engine.initialize()
    
    # Start background tasks
    asyncio.create_task(market_intelligence.start_data_collection())
    asyncio.create_task(sentiment_analyzer.start_news_monitoring())
    asyncio.create_task(run_main_loop())
    
    logger.info("SENTINEL AI Services Ready")
    
    yield
    
    # Cleanup
    logger.info("SENTINEL AI Services Shutting Down...")
    await market_intelligence.shutdown()
    await sentiment_analyzer.shutdown()


async def run_main_loop():
    """Main AI processing loop - runs every few seconds"""
    while True:
        try:
            # Get current market state
            market_state = await market_intelligence.get_current_state()
            
            # Get sentiment data
            sentiment_data = await sentiment_analyzer.get_current_sentiment()
            
            # Detect market regime
            regime = await strategy_planner.detect_regime(market_state, sentiment_data)
            
            # For each active user, run strategy planning
            # This would be triggered per-user in production
            
            await asyncio.sleep(5)  # Run every 5 seconds
            
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(10)


# Create FastAPI app
app = FastAPI(
    title="SENTINEL AI Services",
    description="Autonomous Market Intelligence & Trading System",
    version="1.0.0",
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


# ============================================
# WEBSOCKET ENDPOINTS
# ============================================

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Real-time WebSocket connection for user dashboard"""
    await ws_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Send real-time updates to connected users
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
        "services": {
            "market_intelligence": market_intelligence.is_running,
            "sentiment_analyzer": sentiment_analyzer.is_running,
            "strategy_planner": True,
            "risk_engine": True,
        }
    }


@app.get("/ai/status")
async def get_system_status():
    """Get overall AI system status"""
    return {
        "active": True,
        "market_data_age_seconds": await market_intelligence.get_data_age(),
        "sentiment_data_age_seconds": await sentiment_analyzer.get_data_age(),
        "active_strategies": await strategy_planner.get_active_count(),
        "risk_alerts": await risk_engine.get_active_alerts_count(),
    }


@app.get("/ai/regime/{symbol}")
async def get_market_regime(symbol: str):
    """Get current market regime for a symbol"""
    market_state = await market_intelligence.get_symbol_state(symbol)
    sentiment = await sentiment_analyzer.get_asset_sentiment(symbol.replace("USDT", ""))
    regime = await strategy_planner.detect_regime(market_state, sentiment)
    
    return {
        "symbol": symbol,
        "regime": regime["regime"],
        "confidence": regime["confidence"],
        "volatility": regime["volatility"],
        "trend_strength": regime["trend_strength"],
        "recommended_strategy": regime["recommended_strategy"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

