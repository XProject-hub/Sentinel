"""
SENTINEL AI - Data & News API Router
Real-time crypto news and learning data endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from loguru import logger

router = APIRouter(prefix="/ai/data", tags=["data"])


@router.get("/news")
async def get_crypto_news(limit: int = 20) -> Dict[str, Any]:
    """
    Get real-time crypto news from multiple sources.
    Sources: CryptoCompare, CoinPaprika, CoinGecko, Reddit
    """
    try:
        from services.data_aggregator import DataAggregator
        import redis.asyncio as redis
        from config import settings
        import json
        
        redis_client = await redis.from_url(settings.REDIS_URL)
        
        news_data = await redis_client.get('data:crypto_news')
        await redis_client.close()
        
        if news_data:
            data = json.loads(news_data)
            articles = data.get('articles', [])[:limit]
            return {
                'success': True,
                'articles': articles,
                'sentiment': data.get('sentiment', {}),
                'timestamp': data.get('timestamp'),
                'count': len(articles)
            }
            
        return {
            'success': True,
            'articles': [],
            'sentiment': {'overall': 'neutral', 'bullish_percent': 0, 'bearish_percent': 0},
            'timestamp': None,
            'count': 0,
            'message': 'News collection in progress...'
        }
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning")
async def get_learning_stats() -> Dict[str, Any]:
    """
    Get comprehensive AI learning statistics.
    Shows what the AI has learned from:
    - Historical market data
    - Real-time market movements  
    - News sentiment
    - Technical patterns
    - Trade outcomes
    """
    try:
        import redis.asyncio as redis
        from config import settings
        import json
        
        redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Get Q-values (strategy learning)
        q_values_raw = await redis_client.get('ai:learning:q_values')
        q_values = json.loads(q_values_raw) if q_values_raw else {}
        
        # Get pattern memory
        patterns_raw = await redis_client.get('ai:learning:patterns')
        patterns = json.loads(patterns_raw) if patterns_raw else {}
        
        # Get market states
        market_states_raw = await redis_client.get('ai:learning:market_states')
        market_states = json.loads(market_states_raw) if market_states_raw else {}
        
        # Get sentiment patterns
        sentiment_raw = await redis_client.get('ai:learning:sentiment')
        sentiment_patterns = json.loads(sentiment_raw) if sentiment_raw else {}
        
        # Get learning history
        history_raw = await redis_client.lrange('ai:learning:history', 0, 49)
        history = [json.loads(h) for h in history_raw] if history_raw else []
        
        # Get comprehensive stats
        stats_raw = await redis_client.hgetall('ai:learning:stats')
        stats = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in stats_raw.items()
        } if stats_raw else {}
        
        # Get trade statistics
        trade_stats_raw = await redis_client.hgetall('ai:trading:stats')
        trade_stats = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in trade_stats_raw.items()
        } if trade_stats_raw else {}
        
        # Calculate Q-state count
        q_state_count = sum(
            1 for regime_values in q_values.values() 
            if isinstance(regime_values, dict)
            for q in regime_values.values() 
            if abs(q) > 0.1
        )
        
        # Total learned states
        total_states = q_state_count + len(patterns) + len(market_states) + len(sentiment_patterns)
        
        # Extract best strategies
        best_strategies = {}
        for regime, actions in q_values.items():
            if isinstance(actions, dict) and actions:
                best_action = max(actions.items(), key=lambda x: x[1])
                if best_action[1] > 0.2:
                    best_strategies[regime] = {
                        'action': best_action[0],
                        'confidence': round(min(best_action[1] * 50 + 50, 100), 1)
                    }
        
        # Extract top patterns
        top_patterns = []
        for pattern, data in patterns.items():
            if isinstance(data, dict) and data.get('count', 0) >= 3:
                outcomes = data.get('outcomes', {})
                total = sum(outcomes.values()) if outcomes else 0
                if total > 0:
                    best_outcome = max(outcomes.items(), key=lambda x: x[1])
                    top_patterns.append({
                        'pattern': pattern,
                        'outcome': best_outcome[0],
                        'success_rate': round(best_outcome[1] / total * 100, 1),
                        'occurrences': data.get('count', 0)
                    })
        top_patterns = sorted(top_patterns, key=lambda x: x['success_rate'], reverse=True)[:5]
        
        await redis_client.close()
        
        # Calculate learning progress (target: 100 states)
        learning_progress = min(100, round(total_states / 50 * 100, 1))
        
        return {
            'success': True,
            'learning': {
                'total_states_learned': total_states,
                'q_states': q_state_count,
                'patterns_learned': len(patterns),
                'market_states': len(market_states),
                'sentiment_states': len(sentiment_patterns),
                'learning_progress': learning_progress,
                'learning_iterations': int(stats.get('learning_iterations', 0)),
                'best_strategies': best_strategies,
                'top_patterns': top_patterns,
            },
            'history': history[:20],
            'stats': {
                'total_trades': int(trade_stats.get('total_trades', 0)),
                'wins': int(trade_stats.get('wins', 0)),
                'losses': int(trade_stats.get('losses', 0)),
                'win_rate': float(trade_stats.get('win_rate', 0)),
                'total_profit': float(trade_stats.get('total_profit', 0)),
                'avg_profit_per_trade': float(trade_stats.get('avg_profit', 0)),
                'best_trade': float(trade_stats.get('best_trade', 0)),
                'worst_trade': float(trade_stats.get('worst_trade', 0)),
            },
            'sources': {
                'historical_data': True,
                'market_movements': True,
                'news_sentiment': True,
                'technical_patterns': True,
                'trade_outcomes': True
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching learning stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aggregated")
async def get_aggregated_data() -> Dict[str, Any]:
    """
    Get all aggregated market data.
    Includes: whale alerts, liquidations, fear & greed, on-chain metrics, news sentiment
    """
    try:
        import redis.asyncio as redis
        from config import settings
        import json
        
        redis_client = await redis.from_url(settings.REDIS_URL)
        
        data = {}
        
        # Whale alerts
        whale_data = await redis_client.get('data:whale_alerts')
        if whale_data:
            data['whale_alerts'] = json.loads(whale_data)
            
        # Whale sentiment
        whale_sentiment = await redis_client.hgetall('data:whale_sentiment')
        if whale_sentiment:
            data['whale_sentiment'] = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in whale_sentiment.items()
            }
            
        # Liquidations
        liquidations = await redis_client.hgetall('data:liquidations')
        if liquidations:
            data['liquidations'] = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in liquidations.items()
            }
            
        # Fear & Greed
        fng = await redis_client.hgetall('data:fear_greed')
        if fng:
            data['fear_greed'] = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in fng.items()
            }
            
        # On-chain
        onchain = await redis_client.hgetall('data:onchain')
        if onchain:
            data['onchain'] = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in onchain.items()
            }
            
        # News sentiment (summary only)
        news_data = await redis_client.get('data:crypto_news')
        if news_data:
            news = json.loads(news_data)
            data['news_sentiment'] = news.get('sentiment', {})
            
        await redis_client.close()
        
        return {
            'success': True,
            'data': data,
            'timestamp': None
        }
        
    except Exception as e:
        logger.error(f"Error fetching aggregated data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

