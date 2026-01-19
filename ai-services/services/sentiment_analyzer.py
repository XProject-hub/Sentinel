"""
SENTINEL AI - Sentiment Analysis Service
NLP-based news and social sentiment analysis
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from loguru import logger
import httpx
import redis.asyncio as redis
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

from config import settings


class SentimentAnalyzer:
    """
    Analyzes sentiment from multiple sources:
    - Crypto news portals
    - Breaking news
    - Exchange announcements
    - Social media sentiment
    - On-chain metrics
    
    Uses:
    - Transformer-based NLP models
    - VADER sentiment analysis
    - Custom crypto sentiment scoring
    """
    
    def __init__(self):
        self.is_running = False
        self.redis_client = None
        self.sentiment_pipeline = None
        self.vader_analyzer = None
        self.http_client = None
        self.last_update = datetime.utcnow()
        self._tasks = []
        
        # News sources
        self.news_sources = [
            'cryptopanic',
            'newsapi',
        ]
        
        # Crypto-specific keywords for impact scoring
        self.high_impact_keywords = [
            'etf', 'sec', 'regulation', 'ban', 'hack', 'exploit',
            'fed', 'rate', 'inflation', 'cpi', 'fomc',
            'bankruptcy', 'insolvency', 'liquidation',
            'partnership', 'adoption', 'institutional',
            'halving', 'upgrade', 'fork',
        ]
        
        self.bullish_keywords = [
            'approved', 'adoption', 'partnership', 'investment', 'bullish',
            'rally', 'surge', 'breakout', 'institutional', 'accumulation',
        ]
        
        self.bearish_keywords = [
            'rejected', 'ban', 'hack', 'crash', 'bearish', 'dump',
            'sell-off', 'liquidation', 'bankruptcy', 'investigation',
        ]
        
    async def initialize(self):
        """Initialize NLP models and connections"""
        logger.info("Initializing Sentiment Analyzer...")
        
        # Initialize Redis
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize VADER (rule-based, fast)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformer model (more accurate but slower)
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=settings.SENTIMENT_MODEL,
                tokenizer=settings.SENTIMENT_MODEL,
                device=-1  # CPU
            )
            logger.info("Transformer sentiment model loaded")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}. Using VADER only.")
            
        self.is_running = True
        logger.info("Sentiment Analyzer initialized")
        
    async def shutdown(self):
        """Cleanup resources"""
        self.is_running = False
        
        for task in self._tasks:
            task.cancel()
            
        if self.http_client:
            await self.http_client.aclose()
            
        if self.redis_client:
            await self.redis_client.close()
            
    async def start_news_monitoring(self):
        """Start news collection and analysis"""
        self._tasks = [
            asyncio.create_task(self._collect_crypto_news()),
            asyncio.create_task(self._aggregate_sentiment()),
        ]
        
    async def _collect_crypto_news(self):
        """Collect news from crypto news sources"""
        while self.is_running:
            try:
                # CryptoPanic API
                if settings.CRYPTOPANIC_API_KEY:
                    await self._fetch_cryptopanic_news()
                    
                # NewsAPI
                if settings.NEWSAPI_KEY:
                    await self._fetch_newsapi_news()
                    
                self.last_update = datetime.utcnow()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"News collection error: {e}")
                await asyncio.sleep(60)
                
    async def _fetch_cryptopanic_news(self):
        """Fetch news from CryptoPanic"""
        try:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={settings.CRYPTOPANIC_API_KEY}&filter=rising"
            response = await self.http_client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('results', [])[:20]:
                    await self._process_news_item({
                        'source': 'cryptopanic',
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'published_at': item.get('published_at', ''),
                        'currencies': [c['code'] for c in item.get('currencies', [])],
                    })
                    
        except Exception as e:
            logger.error(f"CryptoPanic fetch error: {e}")
            
    async def _fetch_newsapi_news(self):
        """Fetch crypto news from NewsAPI"""
        try:
            url = f"https://newsapi.org/v2/everything?q=bitcoin+OR+ethereum+OR+crypto&sortBy=publishedAt&apiKey={settings.NEWSAPI_KEY}"
            response = await self.http_client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('articles', [])[:20]:
                    await self._process_news_item({
                        'source': 'newsapi',
                        'title': item.get('title', ''),
                        'content': item.get('description', ''),
                        'url': item.get('url', ''),
                        'published_at': item.get('publishedAt', ''),
                    })
                    
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            
    async def _process_news_item(self, item: Dict):
        """Process and analyze a news item"""
        title = item.get('title', '')
        content = item.get('content', '')
        text = f"{title} {content}".strip()
        
        if not text:
            return
            
        # Analyze sentiment
        sentiment_score = await self._analyze_text_sentiment(text)
        impact_level = self._calculate_impact_level(text)
        related_assets = self._extract_related_assets(text, item.get('currencies', []))
        
        news_data = {
            'source': item.get('source'),
            'title': title,
            'url': item.get('url'),
            'published_at': item.get('published_at'),
            'sentiment_score': sentiment_score,
            'impact_level': impact_level,
            'related_assets': related_assets,
            'processed_at': datetime.utcnow().isoformat(),
        }
        
        # Store in Redis (last 100 news items)
        await self.redis_client.lpush('news:items', json.dumps(news_data))
        await self.redis_client.ltrim('news:items', 0, 99)
        
        # Update asset-specific sentiment
        for asset in related_assets:
            await self._update_asset_sentiment(asset, sentiment_score, impact_level)
            
    async def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using multiple methods"""
        scores = []
        
        # VADER sentiment (fast, rule-based)
        vader_score = self.vader_analyzer.polarity_scores(text)['compound']
        scores.append(vader_score)
        
        # Transformer sentiment (if available)
        if self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(text[:512])[0]
                label = result['label'].lower()
                confidence = result['score']
                
                if 'positive' in label:
                    transformer_score = confidence
                elif 'negative' in label:
                    transformer_score = -confidence
                else:
                    transformer_score = 0
                    
                scores.append(transformer_score)
            except Exception as e:
                logger.debug(f"Transformer analysis error: {e}")
                
        # Keyword-based adjustment
        keyword_adjustment = self._keyword_sentiment_adjustment(text)
        scores.append(keyword_adjustment * 0.5)  # Weight keywords less
        
        # Average all scores
        final_score = np.mean(scores)
        return max(-1.0, min(1.0, final_score))
        
    def _keyword_sentiment_adjustment(self, text: str) -> float:
        """Adjust sentiment based on crypto-specific keywords"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in self.bearish_keywords if kw in text_lower)
        
        if bullish_count + bearish_count == 0:
            return 0.0
            
        return (bullish_count - bearish_count) / (bullish_count + bearish_count)
        
    def _calculate_impact_level(self, text: str) -> str:
        """Calculate news impact level"""
        text_lower = text.lower()
        
        high_impact_count = sum(1 for kw in self.high_impact_keywords if kw in text_lower)
        
        if high_impact_count >= 3:
            return 'critical'
        elif high_impact_count >= 2:
            return 'high'
        elif high_impact_count >= 1:
            return 'medium'
        else:
            return 'low'
            
    def _extract_related_assets(self, text: str, currencies: List[str]) -> List[str]:
        """Extract related crypto assets from text"""
        assets = set(currencies)
        
        # Common asset mentions
        asset_mapping = {
            'bitcoin': 'BTC', 'btc': 'BTC',
            'ethereum': 'ETH', 'eth': 'ETH',
            'solana': 'SOL', 'sol': 'SOL',
            'ripple': 'XRP', 'xrp': 'XRP',
            'binance': 'BNB', 'bnb': 'BNB',
        }
        
        text_lower = text.lower()
        for keyword, asset in asset_mapping.items():
            if keyword in text_lower:
                assets.add(asset)
                
        return list(assets) if assets else ['BTC', 'ETH']  # Default to BTC/ETH
        
    async def _update_asset_sentiment(self, asset: str, sentiment: float, impact: str):
        """Update sentiment aggregate for an asset"""
        impact_multiplier = {'critical': 2.0, 'high': 1.5, 'medium': 1.0, 'low': 0.5}
        weighted_sentiment = sentiment * impact_multiplier.get(impact, 1.0)
        
        # Add to sentiment history
        await self.redis_client.lpush(
            f"sentiment:history:{asset}",
            json.dumps({
                'score': weighted_sentiment,
                'timestamp': datetime.utcnow().isoformat()
            })
        )
        await self.redis_client.ltrim(f"sentiment:history:{asset}", 0, 99)
        
    async def _aggregate_sentiment(self):
        """Aggregate sentiment scores periodically"""
        while self.is_running:
            try:
                assets = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
                
                for asset in assets:
                    history = await self.redis_client.lrange(f"sentiment:history:{asset}", 0, 49)
                    
                    if not history:
                        continue
                        
                    scores = []
                    for item in history:
                        data = json.loads(item)
                        scores.append(data['score'])
                        
                    # Calculate aggregate metrics
                    avg_sentiment = np.mean(scores)
                    recent_sentiment = np.mean(scores[:10]) if len(scores) >= 10 else avg_sentiment
                    sentiment_momentum = recent_sentiment - avg_sentiment
                    
                    aggregate = {
                        'asset': asset,
                        'sentiment_score': round(avg_sentiment, 4),
                        'recent_sentiment': round(recent_sentiment, 4),
                        'sentiment_momentum': round(sentiment_momentum, 4),
                        'data_points': len(scores),
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    
                    await self.redis_client.hset(
                        f"sentiment:aggregate:{asset}",
                        mapping={k: str(v) for k, v in aggregate.items()}
                    )
                    
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Sentiment aggregation error: {e}")
                await asyncio.sleep(30)
                
    async def get_current_sentiment(self) -> Dict[str, Any]:
        """Get aggregated sentiment for all tracked assets"""
        sentiment = {}
        assets = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
        
        for asset in assets:
            data = await self.redis_client.hgetall(f"sentiment:aggregate:{asset}")
            if data:
                sentiment[asset] = {
                    k.decode() if isinstance(k, bytes) else k: 
                    float(v.decode() if isinstance(v, bytes) else v) 
                    if k not in ['asset', 'timestamp'] else (v.decode() if isinstance(v, bytes) else v)
                    for k, v in data.items()
                }
                
        return sentiment
        
    async def get_asset_sentiment(self, asset: str) -> Dict[str, Any]:
        """Get sentiment for a specific asset"""
        data = await self.redis_client.hgetall(f"sentiment:aggregate:{asset.upper()}")
        
        if not data:
            return {
                'asset': asset.upper(),
                'sentiment_score': 0.0,
                'recent_sentiment': 0.0,
                'sentiment_momentum': 0.0,
                'data_points': 0,
            }
            
        return {
            k.decode() if isinstance(k, bytes) else k: 
            v.decode() if isinstance(v, bytes) else v
            for k, v in data.items()
        }
        
    async def get_recent_news(self, limit: int = 10) -> List[Dict]:
        """Get recent processed news items"""
        items = await self.redis_client.lrange('news:items', 0, limit - 1)
        return [json.loads(item) for item in items]
        
    async def get_data_age(self) -> int:
        """Get age of most recent data in seconds"""
        return int((datetime.utcnow() - self.last_update).total_seconds())

