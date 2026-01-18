"""
SENTINEL AI - FinBERT Sentiment Analyzer
Pre-trained financial NLP model for news sentiment

Uses FinBERT - a BERT model fine-tuned on financial texts:
- Trained on 10K+ financial news articles
- Understands financial terminology
- Returns: positive, negative, neutral

Why FinBERT instead of rule-based:
- Understands context ("not good" = negative)
- Handles sarcasm and negation
- Pre-trained, no need for custom training
- Industry standard for financial NLP

Model: yiyanghkust/finbert-tone (Hugging Face)
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from loguru import logger
import redis.asyncio as redis
import httpx

from config import settings

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning("Transformers/PyTorch not installed - using fallback sentiment")


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    timestamp: str
    
    # FinBERT output
    label: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0-100
    
    # Probabilities
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    
    # Trading signal
    sentiment_score: float  # -1 to 1
    trading_signal: str  # 'bullish', 'bearish', 'neutral'
    impact_level: str  # 'high', 'medium', 'low'


@dataclass
class MarketSentiment:
    """Aggregated market sentiment"""
    timestamp: str
    
    # Overall
    overall_sentiment: float  # -1 to 1
    overall_label: str
    confidence: float
    
    # By category
    news_sentiment: float
    social_sentiment: float
    
    # Metrics
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_analyzed: int
    
    # Top headlines
    top_bullish: List[str]
    top_bearish: List[str]


class FinBERTSentiment:
    """
    FinBERT-based Financial Sentiment Analyzer
    
    Analyzes:
    - Crypto news headlines
    - Press releases
    - Social media posts
    - Exchange announcements
    
    Pre-trained model - no GPU required for inference
    (though GPU would speed it up)
    """
    
    # Keywords for impact level
    HIGH_IMPACT_KEYWORDS = [
        'sec', 'regulation', 'ban', 'hack', 'exploit', 'bankruptcy',
        'etf', 'halving', 'fed', 'interest rate', 'lawsuit', 'fraud',
        'partnership', 'major', 'billion', 'million', 'institutional'
    ]
    
    MEDIUM_IMPACT_KEYWORDS = [
        'update', 'launch', 'upgrade', 'network', 'fork', 'listing',
        'whale', 'accumulation', 'sell-off', 'rally', 'dump', 'pump'
    ]
    
    def __init__(self):
        self.redis_client = None
        self.http_client = None
        self.is_running = False
        
        # Model
        self.model = None
        self.tokenizer = None
        self.device = 'cpu'
        
        # Cache
        self.sentiment_cache: Dict[str, SentimentResult] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Batch processing
        self.batch_size = 8  # Process 8 texts at once
        self.queue: List[Tuple[str, asyncio.Future]] = []
        
        # Stats
        self.stats = {
            'texts_analyzed': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'avg_inference_time_ms': 0
        }
        
    async def initialize(self):
        """Initialize FinBERT model"""
        logger.info("Initializing FinBERT Sentiment Analyzer...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        if FINBERT_AVAILABLE:
            await self._load_model()
        else:
            logger.warning("FinBERT not available - using rule-based fallback")
            
        self.is_running = True
        
        # Start batch processing loop
        asyncio.create_task(self._batch_processing_loop())
        
        # Start news collection loop
        asyncio.create_task(self._news_collection_loop())
        
        logger.info("FinBERT Sentiment Analyzer initialized")
        
    async def shutdown(self):
        """Cleanup"""
        self.is_running = False
        if self.http_client:
            await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def _load_model(self):
        """Load FinBERT model from Hugging Face"""
        try:
            logger.info("Loading FinBERT model (this may take a minute)...")
            
            # Use FinBERT for financial sentiment
            model_name = "yiyanghkust/finbert-tone"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to device
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.model = self.model.cuda()
                logger.info("FinBERT loaded on GPU")
            else:
                self.device = 'cpu'
                logger.info("FinBERT loaded on CPU")
                
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.model = None
            
    async def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text
        
        Args:
            text: News headline or article text
            
        Returns:
            SentimentResult with label, confidence, and trading signal
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Check cache
        cache_key = hash(text[:100])  # Hash first 100 chars
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            # Check if still valid
            cached_time = datetime.fromisoformat(cached.timestamp)
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl:
                return cached
                
        if not FINBERT_AVAILABLE or self.model is None:
            return self._fallback_analyze(text, timestamp)
            
        try:
            import time
            start = time.time()
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            if self.device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                
            # FinBERT labels: 0=neutral, 1=positive, 2=negative
            neutral_prob = probs[0].item()
            positive_prob = probs[1].item()
            negative_prob = probs[2].item()
            
            # Determine label
            max_prob = max(neutral_prob, positive_prob, negative_prob)
            if max_prob == positive_prob:
                label = 'positive'
            elif max_prob == negative_prob:
                label = 'negative'
            else:
                label = 'neutral'
                
            confidence = max_prob * 100
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = positive_prob - negative_prob
            
            # Trading signal
            if sentiment_score > 0.3:
                trading_signal = 'bullish'
            elif sentiment_score < -0.3:
                trading_signal = 'bearish'
            else:
                trading_signal = 'neutral'
                
            # Impact level
            impact_level = self._determine_impact(text)
            
            # Update stats
            elapsed_ms = (time.time() - start) * 1000
            self.stats['texts_analyzed'] += 1
            self.stats[f'{label}_count'] += 1
            self.stats['avg_inference_time_ms'] = (
                (self.stats['avg_inference_time_ms'] * (self.stats['texts_analyzed'] - 1) + elapsed_ms) 
                / self.stats['texts_analyzed']
            )
            
            result = SentimentResult(
                text=text[:200],  # Truncate for storage
                timestamp=timestamp,
                label=label,
                confidence=confidence,
                positive_prob=positive_prob,
                negative_prob=negative_prob,
                neutral_prob=neutral_prob,
                sentiment_score=sentiment_score,
                trading_signal=trading_signal,
                impact_level=impact_level
            )
            
            # Cache
            self.sentiment_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            return self._fallback_analyze(text, timestamp)
            
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts efficiently"""
        if not texts:
            return []
            
        if not FINBERT_AVAILABLE or self.model is None:
            return [self._fallback_analyze(t, datetime.utcnow().isoformat()) for t in texts]
            
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
        return results
        
    async def _process_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Process a batch of texts"""
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            if self.device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                
            results = []
            for i, text in enumerate(texts):
                neutral_prob = probs[i][0].item()
                positive_prob = probs[i][1].item()
                negative_prob = probs[i][2].item()
                
                max_prob = max(neutral_prob, positive_prob, negative_prob)
                if max_prob == positive_prob:
                    label = 'positive'
                elif max_prob == negative_prob:
                    label = 'negative'
                else:
                    label = 'neutral'
                    
                sentiment_score = positive_prob - negative_prob
                
                if sentiment_score > 0.3:
                    trading_signal = 'bullish'
                elif sentiment_score < -0.3:
                    trading_signal = 'bearish'
                else:
                    trading_signal = 'neutral'
                    
                results.append(SentimentResult(
                    text=text[:200],
                    timestamp=timestamp,
                    label=label,
                    confidence=max_prob * 100,
                    positive_prob=positive_prob,
                    negative_prob=negative_prob,
                    neutral_prob=neutral_prob,
                    sentiment_score=sentiment_score,
                    trading_signal=trading_signal,
                    impact_level=self._determine_impact(text)
                ))
                
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [self._fallback_analyze(t, timestamp) for t in texts]
            
    def _fallback_analyze(self, text: str, timestamp: str) -> SentimentResult:
        """Rule-based fallback when FinBERT not available"""
        text_lower = text.lower()
        
        # Sentiment word lists
        positive_words = [
            'bullish', 'surge', 'rally', 'gain', 'soar', 'jump', 'rise',
            'breakout', 'profit', 'growth', 'adoption', 'partnership',
            'approval', 'upgrade', 'success', 'milestone', 'record',
            'boom', 'moon', 'pump', 'buy', 'long'
        ]
        
        negative_words = [
            'bearish', 'crash', 'drop', 'fall', 'plunge', 'sink', 'dump',
            'sell-off', 'loss', 'decline', 'reject', 'ban', 'hack',
            'scam', 'fraud', 'lawsuit', 'warning', 'risk', 'fear',
            'concern', 'worry', 'short', 'liquidation'
        ]
        
        # Count matches
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count + 1  # +1 to avoid division by zero
        
        positive_prob = positive_count / total
        negative_prob = negative_count / total
        neutral_prob = 1 - positive_prob - negative_prob
        
        if positive_prob > negative_prob and positive_prob > 0.2:
            label = 'positive'
            trading_signal = 'bullish'
        elif negative_prob > positive_prob and negative_prob > 0.2:
            label = 'negative'
            trading_signal = 'bearish'
        else:
            label = 'neutral'
            trading_signal = 'neutral'
            
        confidence = max(positive_prob, negative_prob, neutral_prob) * 100
        sentiment_score = positive_prob - negative_prob
        
        return SentimentResult(
            text=text[:200],
            timestamp=timestamp,
            label=label,
            confidence=min(confidence, 60),  # Lower confidence for fallback
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
            sentiment_score=sentiment_score,
            trading_signal=trading_signal,
            impact_level=self._determine_impact(text)
        )
        
    def _determine_impact(self, text: str) -> str:
        """Determine impact level of news"""
        text_lower = text.lower()
        
        for keyword in self.HIGH_IMPACT_KEYWORDS:
            if keyword in text_lower:
                return 'high'
                
        for keyword in self.MEDIUM_IMPACT_KEYWORDS:
            if keyword in text_lower:
                return 'medium'
                
        return 'low'
        
    async def _batch_processing_loop(self):
        """Process queued texts in batches"""
        while self.is_running:
            try:
                if len(self.queue) >= self.batch_size or (self.queue and len(self.queue) > 0):
                    # Take batch from queue
                    batch = self.queue[:self.batch_size]
                    self.queue = self.queue[self.batch_size:]
                    
                    texts = [item[0] for item in batch]
                    futures = [item[1] for item in batch]
                    
                    # Process
                    results = await self.analyze_batch(texts)
                    
                    # Set results
                    for i, future in enumerate(futures):
                        if not future.done():
                            future.set_result(results[i])
                            
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Batch loop error: {e}")
                await asyncio.sleep(1)
                
    async def _news_collection_loop(self):
        """Collect and analyze news periodically"""
        logger.info("Starting news sentiment collection loop...")
        
        while self.is_running:
            try:
                await self._collect_and_analyze_news()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"News collection error: {e}")
                await asyncio.sleep(60)
                
    async def _collect_and_analyze_news(self):
        """Collect news from various sources and analyze"""
        headlines = []
        
        # CoinGecko news
        try:
            url = "https://api.coingecko.com/api/v3/news"
            resp = await self.http_client.get(url)
            if resp.status_code == 200:
                news = resp.json().get('data', [])
                for item in news[:20]:
                    headlines.append(item.get('title', ''))
        except:
            pass
            
        # CryptoPanic (if API key available)
        try:
            url = "https://cryptopanic.com/api/v1/posts/?auth_token=FREE&public=true"
            resp = await self.http_client.get(url)
            if resp.status_code == 200:
                posts = resp.json().get('results', [])
                for post in posts[:20]:
                    headlines.append(post.get('title', ''))
        except:
            pass
            
        if not headlines:
            return
            
        # Analyze all headlines
        results = await self.analyze_batch(headlines)
        
        # Calculate aggregated sentiment
        positive = sum(1 for r in results if r.label == 'positive')
        negative = sum(1 for r in results if r.label == 'negative')
        neutral = sum(1 for r in results if r.label == 'neutral')
        
        avg_sentiment = sum(r.sentiment_score for r in results) / len(results) if results else 0
        
        # Get top bullish/bearish headlines
        sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)
        top_bullish = [r.text for r in sorted_results[:3] if r.sentiment_score > 0]
        top_bearish = [r.text for r in reversed(sorted_results[-3:]) if r.sentiment_score < 0]
        
        # Overall label
        if avg_sentiment > 0.2:
            overall_label = 'bullish'
        elif avg_sentiment < -0.2:
            overall_label = 'bearish'
        else:
            overall_label = 'neutral'
            
        market_sentiment = MarketSentiment(
            timestamp=datetime.utcnow().isoformat(),
            overall_sentiment=avg_sentiment,
            overall_label=overall_label,
            confidence=max(positive, negative, neutral) / len(results) * 100 if results else 0,
            news_sentiment=avg_sentiment,
            social_sentiment=0,  # Could add social media later
            bullish_count=positive,
            bearish_count=negative,
            neutral_count=neutral,
            total_analyzed=len(results),
            top_bullish=top_bullish,
            top_bearish=top_bearish
        )
        
        # Store in Redis
        await self.redis_client.set(
            'finbert:market_sentiment',
            json.dumps({
                'timestamp': market_sentiment.timestamp,
                'overall_sentiment': market_sentiment.overall_sentiment,
                'overall_label': market_sentiment.overall_label,
                'confidence': market_sentiment.confidence,
                'bullish_count': market_sentiment.bullish_count,
                'bearish_count': market_sentiment.bearish_count,
                'neutral_count': market_sentiment.neutral_count,
                'total_analyzed': market_sentiment.total_analyzed,
                'top_bullish': market_sentiment.top_bullish,
                'top_bearish': market_sentiment.top_bearish
            }),
            ex=600  # 10 minutes
        )
        
        logger.info(f"News sentiment: {overall_label} ({avg_sentiment:.2f}), "
                   f"Analyzed: {len(results)}, +{positive}/-{negative}/={neutral}")
                   
    async def get_market_sentiment(self) -> Optional[MarketSentiment]:
        """Get current market sentiment"""
        try:
            data = await self.redis_client.get('finbert:market_sentiment')
            if data:
                d = json.loads(data)
                return MarketSentiment(
                    timestamp=d['timestamp'],
                    overall_sentiment=d['overall_sentiment'],
                    overall_label=d['overall_label'],
                    confidence=d['confidence'],
                    news_sentiment=d.get('overall_sentiment', 0),
                    social_sentiment=0,
                    bullish_count=d['bullish_count'],
                    bearish_count=d['bearish_count'],
                    neutral_count=d['neutral_count'],
                    total_analyzed=d['total_analyzed'],
                    top_bullish=d.get('top_bullish', []),
                    top_bearish=d.get('top_bearish', [])
                )
        except:
            pass
        return None
        
    async def get_stats(self) -> Dict:
        """Get analyzer statistics"""
        return {
            'is_available': FINBERT_AVAILABLE and self.model is not None,
            'device': self.device,
            **self.stats
        }


# Global instance
finbert_sentiment = FinBERTSentiment()

