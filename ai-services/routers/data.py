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
    Get AI learning statistics and Q-values.
    Shows what the AI has learned from trading.
    """
    try:
        import redis.asyncio as redis
        from config import settings
        import json
        
        redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Get Q-values (what AI learned)
        q_values_raw = await redis_client.get('ai:learning:q_values')
        q_values = json.loads(q_values_raw) if q_values_raw else {}
        
        # Get learning history
        history_raw = await redis_client.lrange('ai:learning:history', 0, 49)
        history = [json.loads(h) for h in history_raw] if history_raw else []
        
        # Get trade statistics
        stats_raw = await redis_client.hgetall('ai:trading:stats')
        stats = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in stats_raw.items()
        } if stats_raw else {}
        
        # Calculate learning metrics
        total_states = len(q_values)
        learned_actions = sum(1 for v in q_values.values() if isinstance(v, dict) and any(abs(val) > 0.1 for val in v.values()))
        
        # Extract best learned strategies per market condition
        best_strategies = {}
        for state_key, actions in q_values.items():
            if isinstance(actions, dict):
                best_action = max(actions.items(), key=lambda x: x[1]) if actions else (None, 0)
                if best_action[1] > 0.5:  # Significant positive learning
                    best_strategies[state_key] = {
                        'action': best_action[0],
                        'confidence': round(best_action[1], 2)
                    }
        
        await redis_client.close()
        
        return {
            'success': True,
            'learning': {
                'total_states_learned': total_states,
                'active_learning_states': learned_actions,
                'learning_progress': min(100, round(total_states / 100 * 100, 1)),  # Progress to 100 states
                'best_strategies': dict(list(best_strategies.items())[:10]),  # Top 10 strategies
            },
            'history': history[:20],  # Last 20 learning events
            'stats': {
                'total_trades': int(stats.get('total_trades', 0)),
                'wins': int(stats.get('wins', 0)),
                'losses': int(stats.get('losses', 0)),
                'win_rate': float(stats.get('win_rate', 0)),
                'total_profit': float(stats.get('total_profit', 0)),
                'avg_profit_per_trade': float(stats.get('avg_profit', 0)),
                'best_trade': float(stats.get('best_trade', 0)),
                'worst_trade': float(stats.get('worst_trade', 0)),
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

