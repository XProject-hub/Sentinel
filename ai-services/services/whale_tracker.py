"""
SENTINEL AI - Whale Tracker
Monitors large orders and unusual activity

Tracks:
1. Large orders in the order book (walls)
2. Large trades (tape reading)
3. Open interest changes
4. Unusual volume spikes

Whale activity often precedes significant price moves.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import redis.asyncio as redis
import json
import httpx

from config import settings


@dataclass
class WhaleAlert:
    """Represents a whale activity alert"""
    symbol: str
    alert_type: str  # 'buy_wall', 'sell_wall', 'large_buy', 'large_sell', 'oi_spike'
    size_usd: float
    price: float
    timestamp: datetime
    impact: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-100
    description: str


class WhaleTracker:
    """
    Track whale activity to improve trading decisions
    
    Signals:
    - Buy walls below price = Support = Bullish
    - Sell walls above price = Resistance = Bearish
    - Large market buys = Bullish momentum
    - Large market sells = Bearish momentum
    - Rising OI + Rising price = Strong trend
    - Rising OI + Falling price = Trend reversal likely
    """
    
    # Thresholds for "whale" activity
    MIN_ORDER_SIZE_USD = 50000  # $50k minimum for whale order
    MIN_TRADE_SIZE_USD = 25000  # $25k minimum for whale trade
    OI_CHANGE_THRESHOLD = 0.05  # 5% OI change is significant
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.http_client = httpx.AsyncClient(timeout=15.0)
        self.alerts: Dict[str, List[WhaleAlert]] = {}
        self.last_oi: Dict[str, float] = {}
        self._running = False
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await redis.from_url(settings.REDIS_URL)
            logger.info("WhaleTracker initialized")
        except Exception as e:
            logger.error(f"WhaleTracker init failed: {e}")
    
    async def start(self):
        """Start whale tracking background loop"""
        self._running = True
        asyncio.create_task(self._tracking_loop())
        logger.info("WhaleTracker started")
    
    async def stop(self):
        """Stop tracking"""
        self._running = False
        await self.http_client.aclose()
    
    async def _tracking_loop(self):
        """Main tracking loop"""
        while self._running:
            try:
                # Track top coins for whale activity
                symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
                
                for symbol in symbols:
                    await self._analyze_orderbook(symbol)
                    await self._analyze_open_interest(symbol)
                    await asyncio.sleep(1)  # Rate limiting
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Whale tracking error: {e}")
                await asyncio.sleep(10)
    
    async def _analyze_orderbook(self, symbol: str):
        """Analyze order book for whale orders (walls)"""
        try:
            url = f"https://api.bybit.com/v5/market/orderbook?category=linear&symbol={symbol}&limit=50"
            response = await self.http_client.get(url)
            
            if response.status_code != 200:
                return
            
            data = response.json()
            if data.get('retCode') != 0:
                return
            
            result = data.get('result', {})
            bids = result.get('b', [])  # [[price, size], ...]
            asks = result.get('a', [])
            
            if not bids or not asks:
                return
            
            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
            
            # Find buy walls (large bids)
            for price, size in bids:
                price = float(price)
                size = float(size)
                value_usd = price * size
                
                if value_usd >= self.MIN_ORDER_SIZE_USD:
                    # Large buy order - potential support
                    distance_pct = ((mid_price - price) / mid_price) * 100
                    
                    if distance_pct < 2:  # Within 2% of current price
                        alert = WhaleAlert(
                            symbol=symbol,
                            alert_type='buy_wall',
                            size_usd=value_usd,
                            price=price,
                            timestamp=datetime.utcnow(),
                            impact='bullish',
                            confidence=min(90, 50 + (value_usd / 10000)),
                            description=f"${value_usd:,.0f} buy wall at {price:.4f} ({distance_pct:.2f}% below)"
                        )
                        await self._store_alert(alert)
            
            # Find sell walls (large asks)
            for price, size in asks:
                price = float(price)
                size = float(size)
                value_usd = price * size
                
                if value_usd >= self.MIN_ORDER_SIZE_USD:
                    distance_pct = ((price - mid_price) / mid_price) * 100
                    
                    if distance_pct < 2:
                        alert = WhaleAlert(
                            symbol=symbol,
                            alert_type='sell_wall',
                            size_usd=value_usd,
                            price=price,
                            timestamp=datetime.utcnow(),
                            impact='bearish',
                            confidence=min(90, 50 + (value_usd / 10000)),
                            description=f"${value_usd:,.0f} sell wall at {price:.4f} ({distance_pct:.2f}% above)"
                        )
                        await self._store_alert(alert)
                        
        except Exception as e:
            logger.debug(f"Orderbook analysis failed for {symbol}: {e}")
    
    async def _analyze_open_interest(self, symbol: str):
        """Track open interest changes"""
        try:
            url = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={symbol}&intervalTime=5min&limit=2"
            response = await self.http_client.get(url)
            
            if response.status_code != 200:
                return
            
            data = response.json()
            if data.get('retCode') != 0:
                return
            
            oi_list = data.get('result', {}).get('list', [])
            if len(oi_list) < 2:
                return
            
            current_oi = float(oi_list[0].get('openInterest', 0))
            previous_oi = float(oi_list[1].get('openInterest', 0))
            
            if previous_oi == 0:
                return
            
            oi_change = (current_oi - previous_oi) / previous_oi
            
            # Store for later comparison
            self.last_oi[symbol] = current_oi
            
            # Alert on significant OI changes
            if abs(oi_change) >= self.OI_CHANGE_THRESHOLD:
                impact = 'bullish' if oi_change > 0 else 'bearish'
                
                # Get price direction to determine trend strength
                ticker_url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
                ticker_response = await self.http_client.get(ticker_url)
                price_change = 0
                
                if ticker_response.status_code == 200:
                    ticker_data = ticker_response.json()
                    if ticker_data.get('retCode') == 0:
                        ticker_list = ticker_data.get('result', {}).get('list', [])
                        if ticker_list:
                            price_change = float(ticker_list[0].get('price24hPcnt', 0)) * 100
                
                # Rising OI + Rising price = Strong bullish trend
                # Rising OI + Falling price = Accumulation, possible reversal
                # Falling OI + Rising price = Short squeeze / weak rally
                # Falling OI + Falling price = Capitulation
                
                if oi_change > 0 and price_change > 0:
                    description = f"OI +{oi_change*100:.1f}% with price +{price_change:.1f}% - Strong bullish trend"
                    impact = 'bullish'
                    confidence = 80
                elif oi_change > 0 and price_change < 0:
                    description = f"OI +{oi_change*100:.1f}% with price {price_change:.1f}% - Shorts accumulating, watch for squeeze"
                    impact = 'neutral'
                    confidence = 60
                elif oi_change < 0 and price_change > 0:
                    description = f"OI {oi_change*100:.1f}% with price +{price_change:.1f}% - Short squeeze, weak rally"
                    impact = 'bearish'
                    confidence = 65
                else:
                    description = f"OI {oi_change*100:.1f}% with price {price_change:.1f}% - Capitulation"
                    impact = 'bearish'
                    confidence = 70
                
                alert = WhaleAlert(
                    symbol=symbol,
                    alert_type='oi_spike',
                    size_usd=current_oi,
                    price=0,
                    timestamp=datetime.utcnow(),
                    impact=impact,
                    confidence=confidence,
                    description=description
                )
                await self._store_alert(alert)
                
        except Exception as e:
            logger.debug(f"OI analysis failed for {symbol}: {e}")
    
    async def _store_alert(self, alert: WhaleAlert):
        """Store alert in Redis"""
        try:
            if not self.redis_client:
                return
            
            alert_data = {
                'symbol': alert.symbol,
                'type': alert.alert_type,
                'size_usd': alert.size_usd,
                'price': alert.price,
                'timestamp': alert.timestamp.isoformat(),
                'impact': alert.impact,
                'confidence': alert.confidence,
                'description': alert.description
            }
            
            # Store in list (keep last 100)
            await self.redis_client.lpush(f"whale:alerts:{alert.symbol}", json.dumps(alert_data))
            await self.redis_client.ltrim(f"whale:alerts:{alert.symbol}", 0, 99)
            
            # Store latest for quick access
            await self.redis_client.hset(f"whale:latest:{alert.symbol}", mapping={
                k: str(v) for k, v in alert_data.items()
            })
            await self.redis_client.expire(f"whale:latest:{alert.symbol}", 300)  # 5 min TTL
            
            logger.info(f"WHALE ALERT [{alert.symbol}]: {alert.description}")
            
        except Exception as e:
            logger.error(f"Failed to store whale alert: {e}")
    
    async def get_whale_signal(self, symbol: str) -> Dict:
        """
        Get aggregated whale signal for a symbol
        
        Returns:
            {
                'signal': 'bullish' | 'bearish' | 'neutral',
                'confidence': 0-100,
                'alerts': [...recent alerts...],
                'summary': 'human readable summary'
            }
        """
        try:
            if not self.redis_client:
                return {'signal': 'neutral', 'confidence': 0, 'alerts': [], 'summary': 'Whale tracker not initialized'}
            
            # Get recent alerts
            alerts_data = await self.redis_client.lrange(f"whale:alerts:{symbol}", 0, 9)
            
            if not alerts_data:
                return {'signal': 'neutral', 'confidence': 0, 'alerts': [], 'summary': 'No whale activity detected'}
            
            alerts = [json.loads(a) for a in alerts_data]
            
            # Count bullish vs bearish signals
            bullish_score = 0
            bearish_score = 0
            
            for alert in alerts:
                # Weight by recency and confidence
                age_minutes = (datetime.utcnow() - datetime.fromisoformat(alert['timestamp'])).total_seconds() / 60
                recency_weight = max(0.1, 1 - (age_minutes / 60))  # Decay over 1 hour
                
                weight = alert['confidence'] * recency_weight
                
                if alert['impact'] == 'bullish':
                    bullish_score += weight
                elif alert['impact'] == 'bearish':
                    bearish_score += weight
            
            total_score = bullish_score + bearish_score
            
            if total_score == 0:
                return {'signal': 'neutral', 'confidence': 0, 'alerts': alerts, 'summary': 'Insufficient whale data'}
            
            bullish_ratio = bullish_score / total_score
            
            if bullish_ratio > 0.6:
                signal = 'bullish'
                confidence = min(90, bullish_ratio * 100)
                summary = f"Whale activity bullish ({len([a for a in alerts if a['impact'] == 'bullish'])} bullish signals)"
            elif bullish_ratio < 0.4:
                signal = 'bearish'
                confidence = min(90, (1 - bullish_ratio) * 100)
                summary = f"Whale activity bearish ({len([a for a in alerts if a['impact'] == 'bearish'])} bearish signals)"
            else:
                signal = 'neutral'
                confidence = 50
                summary = "Mixed whale signals"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'alerts': alerts[:5],  # Return last 5
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Failed to get whale signal for {symbol}: {e}")
            return {'signal': 'neutral', 'confidence': 0, 'alerts': [], 'summary': f'Error: {e}'}
    
    async def check_whale_support(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """
        Check if whale activity supports the trade direction
        
        Returns: (is_supported, reason)
        """
        whale_data = await self.get_whale_signal(symbol)
        
        signal = whale_data['signal']
        confidence = whale_data['confidence']
        
        # No strong whale signal - neutral (allow trade)
        if confidence < 50:
            return True, "No strong whale signal"
        
        # Whale signal matches direction
        if (direction == 'long' and signal == 'bullish') or \
           (direction == 'short' and signal == 'bearish'):
            return True, f"Whale activity supports {direction} ({whale_data['summary']})"
        
        # Whale signal opposes direction - warn but allow if not too strong
        if confidence < 70:
            return True, f"Whale activity slightly against {direction}, proceed with caution"
        
        # Strong whale signal against trade direction - block
        return False, f"Strong whale activity against {direction} ({whale_data['summary']})"


# Global instance
whale_tracker = WhaleTracker()

