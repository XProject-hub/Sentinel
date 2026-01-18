"""
SENTINEL AI - Ultimate AI Trading Platform
Professional Hedge-Fund Level Autonomous Trading System

=== V3.0 Architecture ===

CORE COMPONENTS:
- DataCollector: Stores EVERYTHING for replay and training
- XGBoostClassifier: Fast ML edge classification
- FinBERTSentiment: Pre-trained financial NLP
- ModelTrainer: Periodic training (not non-stop)
- ClusterManager: Multi-server load balancing

ADVANCED AI:
- RegimeDetector: HMM-inspired market state detection
- EdgeEstimator: Statistical edge calculation
- PositionSizer: Kelly-based dynamic sizing
- MarketScanner: Scans ALL 500+ Bybit pairs
- LearningEngine: Continuous self-improvement

ARCHITECTURE:
- 24/7 Analysis & Trading
- Periodic Training (every 6-12h)
- Controlled Learning
- Multi-server support

CPU Allocation (24 cores):
- 8 cores: Live trading + inference
- 8 cores: Scanning + feature generation
- 6 cores: Background training
- 2 cores: OS / logging / safety

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

# === NEW V3 COMPONENTS ===
from services.data_collector import data_collector
from services.xgboost_classifier import xgboost_classifier
from services.finbert_sentiment import finbert_sentiment
from services.model_trainer import model_trainer
from services.cluster_manager import cluster_manager
from services.training_data_manager import training_data_manager
from services.crypto_sentiment import crypto_sentiment
from services.price_predictor import price_predictor
from services.capital_allocator import capital_allocator

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

# Enable cluster mode (for multi-server)
CLUSTER_MODE = False  # Set to True when running multiple servers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("=" * 70)
    logger.info("   SENTINEL AI v3.0 - Ultimate AI Trading Platform")
    logger.info("   Professional Hedge-Fund Level Autonomous Trading System")
    logger.info("=" * 70)
    
    # === Phase 1: Core Services ===
    logger.info("[1/6] Initializing core services...")
    await market_intelligence.initialize()
    await sentiment_analyzer.initialize()
    await strategy_planner.initialize()
    await risk_engine.initialize()
    await data_aggregator.initialize()
    await learning_engine.initialize()
    
    # === Phase 2: V3 Components (Data, ML) ===
    logger.info("[2/6] Initializing V3 ML components...")
    await data_collector.initialize()
    await training_data_manager.initialize()  # Quality filter for multi-user learning
    await xgboost_classifier.initialize()
    await finbert_sentiment.initialize()
    await crypto_sentiment.initialize()  # CryptoBERT - superior crypto sentiment
    await price_predictor.initialize()  # Multi-model price prediction
    await capital_allocator.initialize()  # Unified budget allocation
    await model_trainer.initialize()
    
    # === Phase 3: Cluster (if enabled) ===
    if CLUSTER_MODE:
        logger.info("[3/6] Initializing cluster manager...")
        await cluster_manager.initialize()
        
        # Set callback for symbol assignment
        cluster_manager.on_symbols_assigned = on_cluster_symbols_assigned
    else:
        logger.info("[3/6] Cluster mode disabled (single server)")
    
    # === Phase 4: Advanced AI Components ===
    logger.info("[4/6] Initializing advanced AI components...")
    await regime_detector.initialize()
    await edge_estimator.initialize()
    await position_sizer.initialize()
    await market_scanner.initialize(regime_detector, edge_estimator)
    
    # === Phase 5: Trader ===
    if USE_V2_TRADER:
        logger.info("[5/6] Initializing Ultimate Autonomous Trader v2.0...")
        await autonomous_trader_v2.initialize(
            regime_detector=regime_detector,
            edge_estimator=edge_estimator,
            position_sizer=position_sizer,
            market_scanner=market_scanner,
            learning_engine=learning_engine
        )
    else:
        logger.info("[5/6] Initializing Legacy Autonomous Trader v1...")
        await autonomous_trader.initialize(learning_engine)
    
    # === Phase 6: Start Background Tasks ===
    logger.info("[6/6] Starting background tasks...")
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
    
    logger.info("=" * 70)
    logger.info("   SENTINEL AI v3.0 READY")
    logger.info("   - ML Models: XGBoost, FinBERT, LSTM")
    logger.info("   - Training: Periodic (every 6-12h)")
    logger.info("   - Data Collection: Active (storing everything)")
    logger.info("   - Cluster Mode: " + ("ENABLED" if CLUSTER_MODE else "DISABLED"))
    logger.info("=" * 70)
    
    yield
    
    # === Cleanup on shutdown ===
    logger.info("SENTINEL AI Services Shutting Down...")
    
    # Shutdown in reverse order
    if USE_V2_TRADER:
        await autonomous_trader_v2.shutdown()
    else:
        await autonomous_trader.shutdown()
        
    await market_scanner.shutdown()
    await position_sizer.shutdown()
    await edge_estimator.shutdown()
    await regime_detector.shutdown()
    
    if CLUSTER_MODE:
        await cluster_manager.shutdown()
        
    await model_trainer.shutdown()
    await finbert_sentiment.shutdown()
    await crypto_sentiment.shutdown()
    await price_predictor.shutdown()
    await capital_allocator.shutdown()
    await xgboost_classifier.shutdown()
    await training_data_manager.shutdown()
    await data_collector.shutdown()
    
    await learning_engine.shutdown()
    await data_aggregator.shutdown()
    await sentiment_analyzer.shutdown()
    await market_intelligence.shutdown()
    
    logger.info("SENTINEL AI Shutdown Complete")


async def on_cluster_symbols_assigned(symbols: list):
    """Called when cluster assigns symbols to this node"""
    logger.info(f"Cluster assigned {len(symbols)} symbols to this node")
    # Update market scanner with assigned symbols
    market_scanner.assigned_symbols = symbols


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


# ============================================
# V3 ENDPOINTS - Advanced ML & Cluster
# ============================================

@app.get("/ai/xgboost/classify")
async def xgboost_classify(symbol: str):
    """Classify trade signal using XGBoost"""
    try:
        # Get features for symbol
        import redis.asyncio as aioredis
        r = await aioredis.from_url(settings.REDIS_URL)
        
        feature_data = await r.get(f"features:{symbol}")
        if not feature_data:
            return {"success": False, "error": "No feature data available"}
            
        import json
        features = json.loads(feature_data)
        features['symbol'] = symbol
        
        result = await xgboost_classifier.classify(features)
        
        await r.aclose()
        
        return {
            "success": True,
            "data": {
                "symbol": result.symbol,
                "signal": result.signal,
                "confidence": result.confidence,
                "probabilities": {
                    "buy": result.buy_prob,
                    "sell": result.sell_prob,
                    "hold": result.hold_prob
                },
                "top_features": result.top_features,
                "model_version": result.model_version
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/ai/xgboost/stats")
async def xgboost_stats():
    """Get XGBoost classifier statistics"""
    stats = await xgboost_classifier.get_stats()
    return {"success": True, "data": stats}


@app.get("/ai/finbert/analyze")
async def finbert_analyze(text: str):
    """Analyze text sentiment using FinBERT"""
    result = await finbert_sentiment.analyze(text)
    
    return {
        "success": True,
        "data": {
            "label": result.label,
            "confidence": result.confidence,
            "sentiment_score": result.sentiment_score,
            "trading_signal": result.trading_signal,
            "impact_level": result.impact_level,
            "probabilities": {
                "positive": result.positive_prob,
                "negative": result.negative_prob,
                "neutral": result.neutral_prob
            }
        }
    }


@app.get("/ai/finbert/market-sentiment")
async def finbert_market_sentiment():
    """Get aggregated market sentiment from FinBERT"""
    sentiment = await finbert_sentiment.get_market_sentiment()
    
    if sentiment:
        return {
            "success": True,
            "data": {
                "overall_sentiment": sentiment.overall_sentiment,
                "overall_label": sentiment.overall_label,
                "confidence": sentiment.confidence,
                "bullish_count": sentiment.bullish_count,
                "bearish_count": sentiment.bearish_count,
                "neutral_count": sentiment.neutral_count,
                "total_analyzed": sentiment.total_analyzed,
                "top_bullish": sentiment.top_bullish,
                "top_bearish": sentiment.top_bearish,
                "timestamp": sentiment.timestamp
            }
        }
    else:
        return {"success": False, "error": "No sentiment data available"}


@app.get("/ai/finbert/stats")
async def finbert_stats():
    """Get FinBERT analyzer statistics"""
    stats = await finbert_sentiment.get_stats()
    return {"success": True, "data": stats}


@app.get("/ai/data-collector/stats")
async def data_collector_stats():
    """Get data collection statistics"""
    stats = await data_collector.get_stats()
    return {"success": True, "data": stats}


@app.get("/ai/trainer/status")
async def trainer_status():
    """Get model trainer status"""
    status = await model_trainer.get_status()
    return {"success": True, "data": status}


@app.post("/ai/trainer/trigger/{job_id}")
async def trigger_training(job_id: str):
    """Manually trigger a training job"""
    success = await model_trainer.trigger_training(job_id)
    return {"success": success}


@app.get("/ai/cluster/status")
async def cluster_status():
    """Get cluster status (multi-server)"""
    if not CLUSTER_MODE:
        return {
            "success": True,
            "data": {
                "cluster_mode": False,
                "message": "Cluster mode is disabled. Set CLUSTER_MODE=True to enable."
            }
        }
        
    status = await cluster_manager.get_cluster_status()
    return {"success": True, "data": status}


@app.get("/ai/models/summary")
async def models_summary():
    """Get summary of all AI models"""
    xgb_stats = await xgboost_classifier.get_stats()
    finbert_stats = await finbert_sentiment.get_stats()
    trainer_status = await model_trainer.get_status()
    learning_stats = await learning_engine.get_learning_stats()
    training_stats = await training_data_manager.get_stats()
    
    return {
        "success": True,
        "data": {
            "xgboost": {
                "available": xgb_stats.get('is_available', False),
                "version": xgb_stats.get('model_version', 'N/A'),
                "accuracy": xgb_stats.get('training_accuracy', 0),
                "last_trained": xgb_stats.get('last_trained')
            },
            "finbert": {
                "available": finbert_stats.get('is_available', False),
                "device": finbert_stats.get('device', 'N/A'),
                "texts_analyzed": finbert_stats.get('texts_analyzed', 0),
                "avg_inference_ms": finbert_stats.get('avg_inference_time_ms', 0)
            },
            "learning_engine": {
                "total_trades": learning_stats.get('total_trades', 0),
                "win_rate": learning_stats.get('win_rate', 0),
                "q_values_learned": learning_stats.get('q_values_count', 0),
                "exploration_rate": learning_stats.get('exploration_rate', 0)
            },
            "trainer": {
                "active_training": trainer_status.get('active_training'),
                "jobs_defined": len(trainer_status.get('jobs', {})),
                "recent_trainings": len(trainer_status.get('recent_history', []))
            },
            "data_collection": {
                "snapshots": (await data_collector.get_stats()).get('snapshots_collected', 0),
                "trades_recorded": (await data_collector.get_stats()).get('trades_recorded', 0),
                "features_generated": (await data_collector.get_stats()).get('features_generated', 0)
            },
            "multi_user_learning": {
                "total_users": training_stats.get('total_users', 0),
                "quality_trades": training_stats.get('quality_trades_count', 0),
                "avg_quality_score": training_stats.get('avg_quality_score', 0),
                "trades_rejected": training_stats.get('trades_rejected_low_quality', 0)
            }
        }
    }


@app.get("/ai/training/stats")
async def training_data_stats():
    """Get training data statistics"""
    stats = await training_data_manager.get_stats()
    return {"success": True, "data": stats}


@app.get("/ai/training/leaderboard")
async def training_leaderboard():
    """Get multi-user contribution leaderboard"""
    leaderboard = await training_data_manager.get_leaderboard()
    return {
        "success": True,
        "data": {
            "leaderboard": leaderboard,
            "message": "Top contributors to AI learning. More quality trades = higher rank."
        }
    }


# ========== NEW SUPERIOR AI ENDPOINTS ==========

@app.get("/ai/crypto-sentiment/market")
async def get_crypto_market_sentiment():
    """Get overall crypto market sentiment using CryptoBERT"""
    sentiment = await crypto_sentiment.get_market_sentiment()
    return {"success": True, "data": sentiment}


@app.post("/ai/crypto-sentiment/analyze")
async def analyze_crypto_text(request: Request):
    """Analyze crypto-specific text sentiment"""
    body = await request.json()
    text = body.get('text', '')
    symbol = body.get('symbol')
    
    result = await crypto_sentiment.analyze_text(text, symbol)
    return {
        "success": True,
        "data": {
            "sentiment": result.sentiment,
            "score": result.score,
            "confidence": result.confidence,
            "crypto_specific": result.crypto_specific,
            "model": result.model_used
        }
    }


@app.get("/ai/crypto-sentiment/symbol/{symbol}")
async def get_symbol_sentiment(symbol: str):
    """Get sentiment for specific symbol"""
    data = await crypto_sentiment.get_symbol_sentiment(symbol)
    return {"success": True, "data": data}


@app.get("/ai/crypto-sentiment/stats")
async def crypto_sentiment_stats():
    """Get CryptoBERT stats"""
    stats = await crypto_sentiment.get_stats()
    return {"success": True, "data": stats}


@app.get("/ai/price-predictor/predict/{symbol}")
async def predict_price(symbol: str):
    """Get price predictions for symbol"""
    prediction = await price_predictor.predict(symbol)
    return {
        "success": True,
        "data": {
            "symbol": prediction.symbol,
            "current_price": prediction.current_price,
            "predictions": {
                "5m": {"price": prediction.prediction_5m, "prob_up": prediction.prob_up_5m},
                "15m": {"price": prediction.prediction_15m, "prob_up": prediction.prob_up_15m},
                "1h": {"price": prediction.prediction_1h, "prob_up": prediction.prob_up_1h},
                "4h": {"price": prediction.prediction_4h, "prob_up": prediction.prob_up_4h}
            },
            "confidence": prediction.confidence,
            "model": prediction.model_used
        }
    }


@app.get("/ai/price-predictor/signal/{symbol}")
async def get_trading_signal(symbol: str):
    """Get trading signal based on price prediction"""
    signal = await price_predictor.get_trading_signal(symbol)
    return {"success": True, "data": signal}


@app.get("/ai/price-predictor/stats")
async def price_predictor_stats():
    """Get price predictor accuracy stats"""
    stats = await price_predictor.get_stats()
    return {"success": True, "data": stats}


@app.get("/ai/capital-allocator/status")
async def allocation_status():
    """Get current capital allocation status"""
    status = await capital_allocator.get_allocation_status()
    return {"success": True, "data": status}


@app.get("/ai/capital-allocator/tradfi")
async def get_tradfi_opportunities():
    """Get available TradFi opportunities"""
    opps = await capital_allocator.get_tradfi_opportunities()
    return {"success": True, "data": opps}


@app.get("/ai/capital-allocator/stats")
async def allocator_stats():
    """Get capital allocator statistics"""
    stats = await capital_allocator.get_stats()
    return {"success": True, "data": stats}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
