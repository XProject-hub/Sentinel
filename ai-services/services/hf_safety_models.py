"""
HUGGING FACE SAFETY MODELS - Intelligent Risk Gates

These models DON'T say BUY/SELL - they say WHEN NOT TO TRADE.

Models:
1. FinBERT - Financial sentiment (block on extreme negative)
2. Zero-Shot Topic Classifier - Detect hack/regulation/exploit news
3. Emotion Detector - Fear/Panic detection
4. Anomaly Detection - Unusual patterns

RULE: These are GATES, not signals!
"""

import logging
import asyncio
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

# Try to import transformers (optional dependency)
try:
    from transformers import pipeline
    HF_AVAILABLE = True
    logger.info("HuggingFace transformers loaded successfully")
except ImportError:
    HF_AVAILABLE = False
    logger.warning("HuggingFace transformers not available - using fallback")


class HFSafetyModels:
    """
    Hugging Face Safety Models - GATES, not signals!
    
    Purpose:
    - Block bad trades
    - Reduce risk in dangerous conditions
    - Pause trading on anomalies
    - NOT to predict price or say BUY/SELL
    """
    
    def __init__(self):
        self.initialized = False
        self.sentiment_analyzer = None
        self.topic_classifier = None
        self.emotion_analyzer = None
        
        # Cache for results (avoid repeated API calls)
        self._cache: Dict[str, Tuple[any, datetime]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Trading pause state
        self._trading_paused_until: Optional[datetime] = None
        self._pause_reason = ""
        
        # Risk modifier (1.0 = normal, 0.5 = half risk, 0 = no trading)
        self.risk_modifier = 1.0
        
        # Dangerous topics that should pause trading
        self.dangerous_topics = [
            "hack", "exploit", "security breach", "stolen funds",
            "regulation", "ban", "lawsuit", "SEC", "investigation",
            "bankruptcy", "insolvency", "fraud", "scam",
            "exchange issue", "withdrawal suspended", "delisting"
        ]
        
    async def initialize(self):
        """Initialize HuggingFace models (lazy loading)"""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace not available - safety models disabled")
            self.initialized = False
            return False
            
        try:
            logger.info("Loading HuggingFace safety models...")
            
            # 1. Financial Sentiment (FinBERT)
            # Used to detect extreme negative sentiment
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=-1  # CPU
                )
                logger.info("✓ FinBERT sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"FinBERT failed to load: {e}")
                # Fallback to lighter model
                try:
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1
                    )
                    logger.info("✓ Fallback sentiment analyzer loaded")
                except:
                    pass
            
            # 2. Zero-Shot Topic Classifier
            # Used to detect dangerous news topics
            try:
                self.topic_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1
                )
                logger.info("✓ Zero-shot topic classifier loaded")
            except Exception as e:
                logger.warning(f"Topic classifier failed to load: {e}")
                # Fallback to lighter model
                try:
                    self.topic_classifier = pipeline(
                        "zero-shot-classification",
                        model="valhalla/distilbart-mnli-12-3",
                        device=-1
                    )
                    logger.info("✓ Fallback topic classifier loaded")
                except:
                    pass
            
            # 3. Emotion Detector
            # Used to detect fear/panic in market
            try:
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=-1
                )
                logger.info("✓ Emotion analyzer loaded")
            except Exception as e:
                logger.warning(f"Emotion analyzer failed to load: {e}")
            
            self.initialized = True
            logger.info("HuggingFace safety models initialized!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize HF models: {e}")
            self.initialized = False
            return False
    
    def _get_cached(self, key: str) -> Optional[any]:
        """Get cached result if not expired"""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self._cache_ttl):
                return result
        return None
    
    def _set_cached(self, key: str, result: any):
        """Cache result"""
        self._cache[key] = (result, datetime.utcnow())
    
    async def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze financial sentiment of text
        
        Returns:
            {
                'sentiment': 'positive' | 'negative' | 'neutral',
                'score': float (0-1),
                'should_block': bool,
                'risk_modifier': float (0-1)
            }
        """
        if not self.sentiment_analyzer:
            return {'sentiment': 'neutral', 'score': 0.5, 'should_block': False, 'risk_modifier': 1.0}
        
        # Check cache
        cache_key = f"sentiment:{hash(text[:100])}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            result = self.sentiment_analyzer(text[:512])[0]  # Limit text length
            
            label = result['label'].lower()
            score = result['score']
            
            # Map to our format
            sentiment = 'neutral'
            if 'positive' in label:
                sentiment = 'positive'
            elif 'negative' in label:
                sentiment = 'negative'
            
            # Determine if we should block trading
            should_block = False
            risk_modifier = 1.0
            
            if sentiment == 'negative':
                if score > 0.9:
                    # EXTREME negative - block longs!
                    should_block = True
                    risk_modifier = 0.3
                    logger.warning(f"EXTREME NEGATIVE sentiment detected: {score:.2f}")
                elif score > 0.75:
                    # Strong negative - reduce risk
                    risk_modifier = 0.5
                    logger.info(f"Strong negative sentiment: {score:.2f}")
                elif score > 0.6:
                    # Moderate negative - slightly reduce risk
                    risk_modifier = 0.7
            
            output = {
                'sentiment': sentiment,
                'score': score,
                'should_block': should_block,
                'risk_modifier': risk_modifier,
                'raw_label': result['label']
            }
            
            self._set_cached(cache_key, output)
            return output
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'score': 0.5, 'should_block': False, 'risk_modifier': 1.0}
    
    async def classify_topic(self, text: str) -> Dict:
        """
        Classify news topic to detect dangerous events
        
        Returns:
            {
                'topic': str,
                'confidence': float,
                'is_dangerous': bool,
                'should_pause': bool,
                'pause_minutes': int
            }
        """
        if not self.topic_classifier:
            return {'topic': 'unknown', 'confidence': 0, 'is_dangerous': False, 'should_pause': False, 'pause_minutes': 0}
        
        # Check cache
        cache_key = f"topic:{hash(text[:100])}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Classify against dangerous topics + neutral
            labels = self.dangerous_topics + ["market update", "price movement", "general news"]
            
            result = self.topic_classifier(text[:512], candidate_labels=labels)
            
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            is_dangerous = top_label in self.dangerous_topics
            should_pause = is_dangerous and top_score > 0.5
            
            # Determine pause duration based on severity
            pause_minutes = 0
            if should_pause:
                if top_label in ["hack", "exploit", "security breach", "stolen funds"]:
                    pause_minutes = 60  # 1 hour for hacks
                elif top_label in ["bankruptcy", "insolvency", "fraud"]:
                    pause_minutes = 120  # 2 hours for major issues
                elif top_label in ["regulation", "ban", "SEC", "lawsuit"]:
                    pause_minutes = 30  # 30 min for regulation news
                else:
                    pause_minutes = 15  # 15 min for other issues
                
                logger.warning(f"DANGEROUS TOPIC detected: {top_label} ({top_score:.2f}) - pause {pause_minutes}min")
            
            output = {
                'topic': top_label,
                'confidence': top_score,
                'is_dangerous': is_dangerous,
                'should_pause': should_pause,
                'pause_minutes': pause_minutes,
                'all_scores': dict(zip(result['labels'][:5], result['scores'][:5]))
            }
            
            self._set_cached(cache_key, output)
            return output
            
        except Exception as e:
            logger.error(f"Topic classification failed: {e}")
            return {'topic': 'unknown', 'confidence': 0, 'is_dangerous': False, 'should_pause': False, 'pause_minutes': 0}
    
    async def detect_emotion(self, text: str) -> Dict:
        """
        Detect emotion in text (fear, anger, joy, etc.)
        
        Returns:
            {
                'emotion': str,
                'score': float,
                'is_fear_panic': bool,
                'risk_modifier': float
            }
        """
        if not self.emotion_analyzer:
            return {'emotion': 'neutral', 'score': 0.5, 'is_fear_panic': False, 'risk_modifier': 1.0}
        
        # Check cache
        cache_key = f"emotion:{hash(text[:100])}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            result = self.emotion_analyzer(text[:512])[0]
            
            emotion = result['label'].lower()
            score = result['score']
            
            # Fear/panic = reduce risk significantly
            is_fear_panic = emotion in ['fear', 'anger', 'disgust', 'sadness']
            
            risk_modifier = 1.0
            if is_fear_panic:
                if score > 0.8:
                    risk_modifier = 0.3  # Very scared market
                    logger.warning(f"FEAR/PANIC detected: {emotion} ({score:.2f})")
                elif score > 0.6:
                    risk_modifier = 0.5
                elif score > 0.4:
                    risk_modifier = 0.7
            
            output = {
                'emotion': emotion,
                'score': score,
                'is_fear_panic': is_fear_panic,
                'risk_modifier': risk_modifier
            }
            
            self._set_cached(cache_key, output)
            return output
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return {'emotion': 'neutral', 'score': 0.5, 'is_fear_panic': False, 'risk_modifier': 1.0}
    
    async def analyze_news_safety(self, news_texts: List[str]) -> Dict:
        """
        Comprehensive news safety analysis
        
        Analyzes multiple news items and returns overall safety assessment
        
        Returns:
            {
                'is_safe_to_trade': bool,
                'risk_modifier': float (0-1),
                'should_pause': bool,
                'pause_minutes': int,
                'pause_reason': str,
                'details': {...}
            }
        """
        if not self.initialized or not news_texts:
            return {
                'is_safe_to_trade': True,
                'risk_modifier': 1.0,
                'should_pause': False,
                'pause_minutes': 0,
                'pause_reason': '',
                'details': {}
            }
        
        # Analyze each news item
        sentiments = []
        topics = []
        emotions = []
        
        for text in news_texts[:5]:  # Limit to 5 most recent
            if len(text) < 10:
                continue
                
            sent = await self.analyze_sentiment(text)
            topic = await self.classify_topic(text)
            emo = await self.detect_emotion(text)
            
            sentiments.append(sent)
            topics.append(topic)
            emotions.append(emo)
        
        # Calculate overall risk
        min_sentiment_modifier = min([s['risk_modifier'] for s in sentiments], default=1.0)
        min_emotion_modifier = min([e['risk_modifier'] for e in emotions], default=1.0)
        
        # Check for dangerous topics
        dangerous_topics = [t for t in topics if t['is_dangerous']]
        should_pause = any(t['should_pause'] for t in topics)
        max_pause = max([t['pause_minutes'] for t in topics], default=0)
        
        # Check for extreme sentiment
        should_block_sentiment = any(s['should_block'] for s in sentiments)
        
        # Calculate final risk modifier
        risk_modifier = min(min_sentiment_modifier, min_emotion_modifier)
        
        # Determine if safe to trade
        is_safe = not should_pause and not should_block_sentiment and risk_modifier > 0.3
        
        # Build pause reason
        pause_reason = ""
        if should_pause and dangerous_topics:
            pause_reason = f"Dangerous news: {dangerous_topics[0]['topic']}"
        elif should_block_sentiment:
            pause_reason = "Extreme negative sentiment"
        
        # Update instance state
        if should_pause and max_pause > 0:
            self._trading_paused_until = datetime.utcnow() + timedelta(minutes=max_pause)
            self._pause_reason = pause_reason
        
        self.risk_modifier = risk_modifier
        
        return {
            'is_safe_to_trade': is_safe,
            'risk_modifier': risk_modifier,
            'should_pause': should_pause,
            'pause_minutes': max_pause,
            'pause_reason': pause_reason,
            'details': {
                'sentiment_count': len(sentiments),
                'dangerous_topics': [t['topic'] for t in dangerous_topics],
                'fear_detected': any(e['is_fear_panic'] for e in emotions),
                'extreme_negative': should_block_sentiment
            }
        }
    
    def is_trading_paused(self) -> Tuple[bool, str]:
        """Check if trading is currently paused due to safety"""
        if self._trading_paused_until and datetime.utcnow() < self._trading_paused_until:
            remaining = (self._trading_paused_until - datetime.utcnow()).total_seconds() / 60
            return True, f"{self._pause_reason} (resume in {remaining:.0f}min)"
        return False, ""
    
    def get_risk_modifier(self) -> float:
        """Get current risk modifier (0-1)"""
        return self.risk_modifier
    
    async def should_allow_trade(self, symbol: str, direction: str, news_context: List[str] = None) -> Tuple[bool, str]:
        """
        Main entry point - should we allow this trade?
        
        This is a GATE, not a signal!
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Check if trading is paused
        is_paused, pause_reason = self.is_trading_paused()
        if is_paused:
            return False, f"HF_PAUSE: {pause_reason}"
        
        # If we have news context, analyze it
        if news_context:
            safety = await self.analyze_news_safety(news_context)
            
            if not safety['is_safe_to_trade']:
                return False, f"HF_BLOCK: {safety['pause_reason']}"
            
            # If risk is very low, only allow shorts or skip
            if safety['risk_modifier'] < 0.5 and direction == 'long':
                return False, f"HF_RISKY: Risk modifier {safety['risk_modifier']:.2f} too low for longs"
        
        return True, "HF_OK"


# Singleton instance
hf_safety = HFSafetyModels()


async def get_hf_safety() -> HFSafetyModels:
    """Get or initialize HF safety models"""
    if not hf_safety.initialized:
        await hf_safety.initialize()
    return hf_safety

