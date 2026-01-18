"""
SENTINEL AI - CryptoBERT Sentiment Analyzer
Superior crypto-specific sentiment analysis

WHY CRYPTOBERT > FINBERT:
- Trained on crypto Twitter, Reddit, Discord
- Understands crypto slang ("moon", "rekt", "wen lambo")
- Better at detecting pump/dump schemes
- More accurate for altcoin sentiment

Models used:
- ElKulako/cryptobert (primary)
- Fallback to FinBERT if needed
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import numpy as np
from loguru import logger
import redis.asyncio as redis
import httpx

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - using API fallback")

from config import settings


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    sentiment: str  # 'bullish', 'bearish', 'neutral'
    score: float  # -1.0 to 1.0
    confidence: float  # 0-100
    model_used: str
    crypto_specific: bool  # True if crypto terms detected


class CryptoSentimentAnalyzer:
    """
    Superior crypto sentiment analysis using CryptoBERT
    
    Features:
    - Crypto-specific model (understands moon, rekt, fud, etc.)
    - Multi-source sentiment (news, twitter, reddit)
    - Real-time market sentiment score
    - Symbol-specific sentiment tracking
    """
    
    # Crypto slang that indicates sentiment
    BULLISH_TERMS = ['moon', 'pump', 'bullish', 'ath', 'breakout', 'diamond hands', 
                     'hodl', 'buy the dip', 'accumulate', 'undervalued', 'gem']
    BEARISH_TERMS = ['dump', 'rekt', 'bearish', 'crash', 'sell', 'scam', 'rug', 
                     'fud', 'overvalued', 'dead', 'exit']
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # Models
        self.crypto_model = None
        self.crypto_tokenizer = None
        self.sentiment_pipeline = None
        self.model_loaded = False
        self.model_name = "ElKulako/cryptobert"  # Primary model
        
        # Cache
        self.sentiment_cache: Dict[str, SentimentResult] = {}
        self.symbol_sentiments: Dict[str, List[float]] = {}  # symbol -> recent scores
        
        # Stats
        self.stats = {
            'texts_analyzed': 0,
            'avg_sentiment': 0.0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'crypto_terms_detected': 0
        }
        
    async def initialize(self):
        """Initialize CryptoBERT model"""
        logger.info("Initializing CryptoBERT Sentiment Analyzer...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.is_running = True
        
        if TRANSFORMERS_AVAILABLE:
            await self._load_model()
        else:
            logger.warning("Running in API-only mode (no local model)")
            
        await self._load_cached_sentiments()
        
        logger.info(f"CryptoBERT initialized - Model: {self.model_name}, Loaded: {self.model_loaded}")
        
    async def _load_model(self):
        """Load CryptoBERT model"""
        try:
            logger.info(f"Loading CryptoBERT model: {self.model_name}")
            
            # Try CryptoBERT first
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=-1  # CPU
                )
                self.model_loaded = True
                logger.info("CryptoBERT loaded successfully!")
            except Exception as e:
                logger.warning(f"CryptoBERT failed, trying FinBERT fallback: {e}")
                
                # Fallback to FinBERT
                self.model_name = "ProsusAI/finbert"
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=-1
                )
                self.model_loaded = True
                logger.info("FinBERT fallback loaded")
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.model_loaded = False
            
    async def shutdown(self):
        """Cleanup"""
        await self._save_cached_sentiments()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def analyze_text(self, text: str, symbol: str = None) -> SentimentResult:
        """
        Analyze text sentiment with crypto-specific understanding
        """
        self.stats['texts_analyzed'] += 1
        
        # Check cache
        cache_key = hash(text[:100])
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
            
        try:
            # Detect crypto terms
            text_lower = text.lower()
            has_bullish = any(term in text_lower for term in self.BULLISH_TERMS)
            has_bearish = any(term in text_lower for term in self.BEARISH_TERMS)
            crypto_specific = has_bullish or has_bearish
            
            if crypto_specific:
                self.stats['crypto_terms_detected'] += 1
                
            # Get model sentiment
            if self.model_loaded and self.sentiment_pipeline:
                result = self.sentiment_pipeline(text[:512])[0]
                
                # Normalize based on model
                if "cryptobert" in self.model_name.lower():
                    # CryptoBERT outputs: Bullish, Bearish, Neutral
                    sentiment = result['label'].lower()
                    confidence = result['score'] * 100
                else:
                    # FinBERT outputs: positive, negative, neutral
                    label_map = {'positive': 'bullish', 'negative': 'bearish', 'neutral': 'neutral'}
                    sentiment = label_map.get(result['label'].lower(), 'neutral')
                    confidence = result['score'] * 100
                    
                # Convert to score
                if sentiment == 'bullish':
                    score = confidence / 100
                    self.stats['bullish_count'] += 1
                elif sentiment == 'bearish':
                    score = -confidence / 100
                    self.stats['bearish_count'] += 1
                else:
                    score = 0.0
                    self.stats['neutral_count'] += 1
                    
            else:
                # Fallback: keyword-based
                sentiment, score, confidence = self._keyword_sentiment(text)
                
            # Boost confidence if crypto terms align with sentiment
            if crypto_specific:
                if (has_bullish and sentiment == 'bullish') or (has_bearish and sentiment == 'bearish'):
                    confidence = min(100, confidence * 1.2)
                elif (has_bullish and sentiment == 'bearish') or (has_bearish and sentiment == 'bullish'):
                    confidence *= 0.8  # Reduce confidence for mixed signals
                    
            result = SentimentResult(
                text=text[:200],
                sentiment=sentiment,
                score=score,
                confidence=confidence,
                model_used=self.model_name,
                crypto_specific=crypto_specific
            )
            
            # Cache
            self.sentiment_cache[cache_key] = result
            
            # Track by symbol
            if symbol:
                if symbol not in self.symbol_sentiments:
                    self.symbol_sentiments[symbol] = []
                self.symbol_sentiments[symbol].append(score)
                # Keep last 50
                self.symbol_sentiments[symbol] = self.symbol_sentiments[symbol][-50:]
                
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return SentimentResult(
                text=text[:200],
                sentiment='neutral',
                score=0.0,
                confidence=0.0,
                model_used='error',
                crypto_specific=False
            )
            
    def _keyword_sentiment(self, text: str) -> Tuple[str, float, float]:
        """Fallback keyword-based sentiment"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for term in self.BULLISH_TERMS if term in text_lower)
        bearish_count = sum(1 for term in self.BEARISH_TERMS if term in text_lower)
        
        if bullish_count > bearish_count:
            return 'bullish', 0.5, 60.0
        elif bearish_count > bullish_count:
            return 'bearish', -0.5, 60.0
        else:
            return 'neutral', 0.0, 50.0
            
    async def get_symbol_sentiment(self, symbol: str) -> Dict:
        """Get aggregated sentiment for a symbol"""
        scores = self.symbol_sentiments.get(symbol, [])
        
        if not scores:
            return {
                'symbol': symbol,
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'sample_count': 0
            }
            
        avg_score = np.mean(scores)
        
        if avg_score > 0.2:
            sentiment = 'bullish'
        elif avg_score < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
            
        return {
            'symbol': symbol,
            'sentiment': sentiment,
            'score': round(avg_score, 3),
            'confidence': min(100, len(scores) * 5),  # More samples = higher confidence
            'sample_count': len(scores)
        }
        
    async def get_market_sentiment(self) -> Dict:
        """Get overall crypto market sentiment"""
        try:
            # Aggregate all symbol sentiments
            all_scores = []
            for scores in self.symbol_sentiments.values():
                all_scores.extend(scores)
                
            if not all_scores:
                # Fetch Fear & Greed as fallback
                return await self._get_fear_greed()
                
            avg_score = np.mean(all_scores)
            
            if avg_score > 0.3:
                sentiment = 'extreme_greed'
            elif avg_score > 0.1:
                sentiment = 'greed'
            elif avg_score > -0.1:
                sentiment = 'neutral'
            elif avg_score > -0.3:
                sentiment = 'fear'
            else:
                sentiment = 'extreme_fear'
                
            return {
                'overall_sentiment': sentiment,
                'overall_score': round(avg_score, 3),
                'symbols_analyzed': len(self.symbol_sentiments),
                'total_samples': len(all_scores),
                'bullish_ratio': sum(1 for s in all_scores if s > 0) / max(1, len(all_scores)),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market sentiment error: {e}")
            return {'overall_sentiment': 'neutral', 'overall_score': 0.0}
            
    async def _get_fear_greed(self) -> Dict:
        """Fetch Fear & Greed Index as fallback"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get("https://api.alternative.me/fng/")
                data = response.json()
                
                if data.get('data'):
                    fng = data['data'][0]
                    value = int(fng['value'])
                    
                    # Convert to our format
                    score = (value - 50) / 50  # 0-100 to -1 to 1
                    
                    return {
                        'overall_sentiment': fng['value_classification'].lower(),
                        'overall_score': score,
                        'fear_greed_index': value,
                        'source': 'alternative.me'
                    }
        except:
            pass
            
        return {'overall_sentiment': 'neutral', 'overall_score': 0.0}
        
    async def analyze_news_batch(self, news_items: List[Dict]) -> Dict:
        """Analyze batch of news items"""
        results = []
        
        for item in news_items[:20]:  # Limit batch size
            text = item.get('title', '') + ' ' + item.get('description', '')
            symbol = item.get('symbol')
            
            result = await self.analyze_text(text, symbol)
            results.append({
                'title': item.get('title', '')[:100],
                'sentiment': result.sentiment,
                'score': result.score,
                'confidence': result.confidence,
                'crypto_specific': result.crypto_specific
            })
            
        # Aggregate
        scores = [r['score'] for r in results]
        avg_score = np.mean(scores) if scores else 0
        
        return {
            'news_count': len(results),
            'avg_sentiment_score': round(avg_score, 3),
            'bullish_count': sum(1 for r in results if r['sentiment'] == 'bullish'),
            'bearish_count': sum(1 for r in results if r['sentiment'] == 'bearish'),
            'neutral_count': sum(1 for r in results if r['sentiment'] == 'neutral'),
            'items': results
        }
        
    async def _load_cached_sentiments(self):
        """Load cached sentiments from Redis"""
        try:
            data = await self.redis_client.get('crypto_sentiment:symbol_scores')
            if data:
                self.symbol_sentiments = json.loads(data)
        except:
            pass
            
    async def _save_cached_sentiments(self):
        """Save sentiments to Redis"""
        try:
            await self.redis_client.set(
                'crypto_sentiment:symbol_scores',
                json.dumps(self.symbol_sentiments),
                ex=3600  # 1 hour
            )
        except:
            pass
            
    async def get_stats(self) -> Dict:
        """Get analyzer statistics"""
        total = self.stats['bullish_count'] + self.stats['bearish_count'] + self.stats['neutral_count']
        
        return {
            'model_name': self.model_name,
            'model_loaded': self.model_loaded,
            'texts_analyzed': self.stats['texts_analyzed'],
            'bullish_ratio': round(self.stats['bullish_count'] / max(1, total) * 100, 1),
            'bearish_ratio': round(self.stats['bearish_count'] / max(1, total) * 100, 1),
            'neutral_ratio': round(self.stats['neutral_count'] / max(1, total) * 100, 1),
            'crypto_terms_detected': self.stats['crypto_terms_detected'],
            'symbols_tracked': len(self.symbol_sentiments)
        }


# Global instance
crypto_sentiment = CryptoSentimentAnalyzer()

