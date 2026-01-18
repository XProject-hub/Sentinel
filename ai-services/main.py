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
from services.data_aggregator import DataAggregator
from services.learning_engine import LearningEngine
from services.autonomous_trader import autonomous_trader
from services.ai_coordinator import ai_coordinator
from services.price_predictor import price_predictor
from services.pattern_recognition import pattern_recognition
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("SENTINEL AI Services Starting...")
    
    # Initialize all services
    await market_intelligence.initialize()
    await sentiment_analyzer.initialize()
    await strategy_planner.initialize()
    await risk_engine.initialize()
    await data_aggregator.initialize()
    await learning_engine.initialize()
    
    # Initialize AI Coordinator (connects all AI models)
    await ai_coordinator.initialize(learning_engine)
    
    await autonomous_trader.initialize(learning_engine, ai_coordinator)
    
    # Start background tasks
    asyncio.create_task(market_intelligence.start_data_collection())
    asyncio.create_task(sentiment_analyzer.start_news_monitoring())
    asyncio.create_task(data_aggregator.start_collection())
    asyncio.create_task(run_main_loop())
    asyncio.create_task(autonomous_trader.run_trading_loop())  # 24/7 TRADING
    
    # Auto-reconnect users who had trading enabled before restart
    asyncio.create_task(auto_reconnect_on_startup())
    
    logger.info("SENTINEL AI Services Ready")
    
    yield  # THIS IS REQUIRED FOR LIFESPAN!
    
    # Cleanup on shutdown
    logger.info("SENTINEL AI Services Shutting Down...")
    await market_intelligence.shutdown()
    await sentiment_analyzer.shutdown()
    await data_aggregator.shutdown()
    await learning_engine.shutdown()
    await ai_coordinator.shutdown()
    await autonomous_trader.shutdown()


async def auto_reconnect_on_startup():
    """Auto-reconnect all users with saved trading credentials"""
    await asyncio.sleep(5)  # Wait for services to fully initialize
    
    try:
        import redis.asyncio as aioredis
        import base64
        r = await aioredis.from_url(settings.REDIS_URL)
        reconnected = 0
        
        # Method 1: Check exchange:credentials:default (main storage)
        creds = await r.hgetall("exchange:credentials:default")
        if creds:
            api_key_enc = creds.get(b"api_key", b"").decode() if creds.get(b"api_key") else ""
            api_secret_enc = creds.get(b"api_secret", b"").decode() if creds.get(b"api_secret") else ""
            
            if api_key_enc and api_secret_enc:
                try:
                    # Decrypt (simple base64)
                    api_key = base64.b64decode(api_key_enc.encode()).decode()
                    api_secret = base64.b64decode(api_secret_enc.encode()).decode()
                    
                    if api_key and api_secret:
                        # Import and connect
                        from services.bybit_client import BybitV5Client
                        from routers.exchange import exchange_connections
                        
                        client = BybitV5Client(api_key, api_secret)
                        result = await client.test_connection()
                        
                        if result.get("success"):
                            exchange_connections["default"] = client
                            
                            # Also connect to autonomous trader
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
        
        # Method 2: Check trading:enabled:* keys (legacy)
        keys_raw = await r.keys("trading:enabled:*")
        keys = list(keys_raw) if keys_raw else []
        
        for key in keys:
            user_id = key.decode().split(":")[-1]
            data = await r.hgetall(key)
            
            if data.get(b"enabled", b"0").decode() == "1":
                api_key_enc = data.get(b"api_key", b"").decode()
                api_secret_enc = data.get(b"api_secret", b"").decode()
                
                try:
                    api_key = base64.b64decode(api_key_enc.encode()).decode()
                    api_secret = base64.b64decode(api_secret_enc.encode()).decode()
                    
                    if api_key and api_secret:
                        success = await autonomous_trader.connect_user(
                            user_id=user_id,
                            api_key=api_key,
                            api_secret=api_secret,
                        )
                        if success:
                            reconnected += 1
                            logger.info(f"Auto-reconnected trading for user: {user_id}")
                except Exception as e:
                    logger.error(f"Failed to reconnect user {user_id}: {e}")
                    
        await r.close()
        
        if reconnected > 0:
            logger.info(f"Auto-reconnected {reconnected} users for autonomous trading")
        else:
            logger.info("No users to auto-reconnect (new install or no saved credentials)")
            
    except Exception as e:
        logger.error(f"Auto-reconnect on startup failed: {e}")


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


@app.get("/ai/insight")
async def get_ai_insight():
    """Get AI-generated market insight and confidence"""
    try:
        # Get current market state
        market_state = await market_intelligence.get_current_state()
        sentiment_data = await sentiment_analyzer.get_current_sentiment()
        aggregated_data = await data_aggregator.get_aggregated_data()
        
        # Detect regime for BTC (primary indicator)
        btc_state = market_state.get('BTCUSDT', {})
        btc_sentiment = sentiment_data.get('BTC', {})
        regime = await strategy_planner.detect_regime({'BTCUSDT': btc_state}, sentiment_data)
        
        # Calculate overall AI confidence
        confidence = regime.get('confidence', 0.5)
        
        # Adjust confidence based on data freshness
        market_age = await market_intelligence.get_data_age()
        sentiment_age = await sentiment_analyzer.get_data_age()
        
        if market_age > 60:  # Data older than 60 seconds
            confidence *= 0.9
        if sentiment_age > 600:  # Sentiment older than 10 minutes
            confidence *= 0.95
            
        # Get AI insight text
        insight = await data_aggregator.get_market_insight()
        
        # Determine risk status
        volatility = regime.get('volatility', 1.5)
        if volatility > 4.0:
            risk_status = 'CAUTION'
        elif volatility > 2.5:
            risk_status = 'ELEVATED'
        else:
            risk_status = 'SAFE'
            
        # Get Fear & Greed
        fng = aggregated_data.get('fear_greed', {})
        
        return {
            "success": True,
            "data": {
                "confidence": round(confidence * 100, 1),
                "confidence_label": "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low",
                "risk_status": risk_status,
                "insight": insight,
                "regime": regime.get('regime', 'sideways'),
                "volatility": round(volatility, 2),
                "trend": regime.get('trend', 'sideways'),
                "fear_greed_index": int(fng.get('value', 50)),
                "fear_greed_label": fng.get('classification', 'Neutral'),
                "recommended_action": regime.get('recommended_strategy', 'hold'),
                "timestamp": regime.get('timestamp')
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
                "fear_greed_index": 50,
                "fear_greed_label": "Neutral",
                "recommended_action": "hold",
                "timestamp": None
            }
        }


@app.get("/ai/aggregated-data")
async def get_all_aggregated_data():
    """Get all aggregated market data (whale alerts, on-chain, etc.)"""
    data = await data_aggregator.get_aggregated_data()
    return {
        "success": True,
        "data": data
    }


@app.get("/ai/learning/stats")
async def get_learning_statistics():
    """Get AI learning statistics and performance"""
    stats = await learning_engine.get_learning_stats()
    return {
        "success": True,
        "data": stats
    }


@app.get("/ai/learning/events")
async def get_learning_events(limit: int = 20):
    """Get recent AI learning events"""
    events = await learning_engine.get_recent_learning_events(limit)
    return {
        "success": True,
        "data": events
    }


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


@app.get("/ai/coordinator/stats")
async def get_ai_coordinator_stats():
    """Get AI Coordinator (brain) statistics - all models combined"""
    stats = await ai_coordinator.get_coordinator_stats()
    return {
        "success": True,
        "data": stats
    }


@app.get("/ai/prediction/{symbol}")
async def get_price_prediction(symbol: str):
    """Get LSTM price prediction for a symbol"""
    prediction = await price_predictor.predict_price(symbol)
    
    if prediction:
        return {
            "success": True,
            "data": {
                "symbol": prediction.symbol,
                "current_price": prediction.current_price,
                "predicted_1h": prediction.predicted_price_1h,
                "predicted_4h": prediction.predicted_price_4h,
                "predicted_24h": prediction.predicted_price_24h,
                "direction": prediction.direction,
                "confidence": prediction.confidence,
                "predicted_change": prediction.predicted_change_percent,
                "model_agreement": prediction.model_agreement,
                "timestamp": prediction.timestamp
            }
        }
    else:
        return {
            "success": False,
            "error": "Unable to generate prediction"
        }


@app.get("/ai/patterns/{symbol}")
async def get_patterns(symbol: str):
    """Get detected chart patterns for a symbol"""
    patterns = await pattern_recognition.analyze_patterns(symbol)
    
    return {
        "success": True,
        "data": {
            "symbol": symbol,
            "patterns_found": len(patterns),
            "patterns": [
                {
                    "name": p.pattern_name,
                    "type": p.pattern_type,
                    "strength": p.strength,
                    "confidence": p.confidence,
                    "timeframe": p.timeframe,
                    "description": p.description,
                    "target": p.target_price,
                    "stop_loss": p.stop_loss,
                    "risk_reward": p.risk_reward
                }
                for p in patterns
            ]
        }
    }


@app.get("/ai/decision/{symbol}")
async def get_ai_decision(symbol: str):
    """Get unified AI trading decision for a symbol"""
    # Get market data
    market_state = await market_intelligence.get_symbol_state(symbol)
    
    market_data = {
        'current_price': market_state.get('price', 0),
        'rsi': market_state.get('rsi', 50),
        'price_change_24h': market_state.get('change_24h', 0),
        'funding_rate': market_state.get('funding_rate', 0),
        'regime': market_state.get('regime', 'sideways')
    }
    
    decision = await ai_coordinator.get_ai_decision(symbol, market_data)
    
    if decision:
        return {
            "success": True,
            "data": {
                "symbol": decision.symbol,
                "action": decision.action,
                "confidence": decision.confidence,
                "scores": {
                    "lstm": decision.lstm_score,
                    "patterns": decision.pattern_score,
                    "sentiment": decision.sentiment_score,
                    "technical": decision.technical_score,
                    "learning": decision.learning_score
                },
                "prediction": {
                    "direction": decision.predicted_direction,
                    "change_percent": decision.predicted_change
                },
                "trade_params": {
                    "entry": decision.suggested_entry,
                    "stop_loss": decision.suggested_stop_loss,
                    "take_profit": decision.suggested_take_profit,
                    "position_size": decision.suggested_position_size,
                    "risk_reward": decision.risk_reward_ratio
                },
                "reasons": decision.reasons,
                "patterns": decision.patterns_detected,
                "model_agreement": decision.model_agreement,
                "timestamp": decision.timestamp
            }
        }
    else:
        return {
            "success": False,
            "error": "Unable to generate decision"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

