"""
SENTINEL AI - Extended Data Aggregator
Real-time data from multiple sources for comprehensive market intelligence
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
from loguru import logger
import redis.asyncio as redis
import json

from config import settings


class DataAggregator:
    """
    Aggregates data from multiple sources:
    - Whale Alerts (large transactions)
    - On-chain metrics
    - Exchange announcements
    - Macro economic events (FED, CPI)
    - Liquidation data
    """
    
    def __init__(self):
        self.is_running = False
        self.redis_client = None
        self.http_client = None
        self.last_update = {}
        self._tasks = []
        
    async def initialize(self):
        """Initialize connections"""
        logger.info("Initializing Data Aggregator...")
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.is_running = True
        logger.info("Data Aggregator initialized")
        
    async def shutdown(self):
        """Cleanup"""
        self.is_running = False
        for task in self._tasks:
            task.cancel()
        if self.http_client:
            await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.close()
            
    async def start_collection(self):
        """Start all data collection tasks"""
        self._tasks = [
            asyncio.create_task(self._collect_whale_alerts()),
            asyncio.create_task(self._collect_exchange_announcements()),
            asyncio.create_task(self._collect_liquidations()),
            asyncio.create_task(self._collect_fear_greed_index()),
            asyncio.create_task(self._collect_onchain_metrics()),
        ]
        
    async def _collect_whale_alerts(self):
        """Collect whale transaction alerts (real API)"""
        while self.is_running:
            try:
                # Using Whale Alert API (free tier available)
                # Alternative: BlockCypher, Etherscan for real data
                
                # Fetch recent large BTC transactions from blockchain.info
                url = "https://blockchain.info/unconfirmed-transactions?format=json"
                response = await self.http_client.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    large_txs = []
                    
                    for tx in data.get('txs', [])[:50]:
                        total_value = sum(out.get('value', 0) for out in tx.get('out', []))
                        btc_value = total_value / 100000000  # Satoshis to BTC
                        
                        # Only track transactions > 100 BTC
                        if btc_value >= 100:
                            large_txs.append({
                                'hash': tx.get('hash', '')[:16] + '...',
                                'amount': btc_value,
                                'asset': 'BTC',
                                'type': 'transfer',
                                'timestamp': datetime.utcnow().isoformat(),
                                'usd_estimate': btc_value * 50000,  # Will be updated with real price
                            })
                            
                    if large_txs:
                        # Store in Redis
                        await self.redis_client.set(
                            'data:whale_alerts',
                            json.dumps({
                                'alerts': large_txs[:10],
                                'count': len(large_txs),
                                'timestamp': datetime.utcnow().isoformat()
                            }),
                            ex=600  # 10 min expiry
                        )
                        
                        # Calculate whale sentiment
                        if len(large_txs) > 5:
                            whale_activity = 'high'
                        elif len(large_txs) > 2:
                            whale_activity = 'moderate'
                        else:
                            whale_activity = 'low'
                            
                        await self.redis_client.hset(
                            'data:whale_sentiment',
                            mapping={
                                'activity': whale_activity,
                                'large_tx_count': str(len(large_txs)),
                                'timestamp': datetime.utcnow().isoformat()
                            }
                        )
                        
                self.last_update['whale_alerts'] = datetime.utcnow()
                await asyncio.sleep(120)  # Every 2 minutes
                
            except Exception as e:
                logger.error(f"Whale alerts collection error: {e}")
                await asyncio.sleep(60)
                
    async def _collect_exchange_announcements(self):
        """Collect exchange announcements (Binance, Bybit)"""
        while self.is_running:
            try:
                announcements = []
                
                # Binance announcements (public endpoint)
                try:
                    binance_url = "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query?catalogId=48&pageNo=1&pageSize=5"
                    response = await self.http_client.get(binance_url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get('data', {}).get('articles', []):
                            announcements.append({
                                'exchange': 'binance',
                                'title': item.get('title', ''),
                                'type': 'listing' if 'list' in item.get('title', '').lower() else 'general',
                                'url': f"https://www.binance.com/en/support/announcement/{item.get('code', '')}",
                                'timestamp': datetime.utcnow().isoformat(),
                            })
                except Exception as e:
                    logger.debug(f"Binance announcements error: {e}")
                    
                # Bybit announcements
                try:
                    bybit_url = "https://api.bybit.com/v5/announcements/index?locale=en-US&limit=5"
                    response = await self.http_client.get(bybit_url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get('result', {}).get('list', []):
                            announcements.append({
                                'exchange': 'bybit',
                                'title': item.get('title', ''),
                                'type': item.get('type', {}).get('title', 'general').lower(),
                                'url': item.get('url', ''),
                                'timestamp': item.get('publishTime', datetime.utcnow().isoformat()),
                            })
                except Exception as e:
                    logger.debug(f"Bybit announcements error: {e}")
                    
                if announcements:
                    await self.redis_client.set(
                        'data:exchange_announcements',
                        json.dumps({
                            'announcements': announcements,
                            'timestamp': datetime.utcnow().isoformat()
                        }),
                        ex=1800  # 30 min expiry
                    )
                    
                self.last_update['announcements'] = datetime.utcnow()
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Exchange announcements error: {e}")
                await asyncio.sleep(120)
                
    async def _collect_liquidations(self):
        """Collect liquidation data from exchanges"""
        while self.is_running:
            try:
                # Collect from Bybit (real endpoint)
                url = "https://api.bybit.com/v5/market/recent-trade?category=linear&symbol=BTCUSDT&limit=50"
                response = await self.http_client.get(url)
                
                liquidations = {
                    'long_liquidations': 0,
                    'short_liquidations': 0,
                    'total_volume': 0,
                }
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Get funding rate as proxy for liquidation pressure
                    funding_url = "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=1"
                    funding_response = await self.http_client.get(funding_url)
                    
                    if funding_response.status_code == 200:
                        funding_data = funding_response.json()
                        funding_list = funding_data.get('result', {}).get('list', [])
                        
                        if funding_list:
                            funding_rate = float(funding_list[0].get('fundingRate', 0))
                            
                            # Estimate liquidation pressure from funding rate
                            if funding_rate > 0.001:
                                liquidations['long_liquidations'] = abs(funding_rate) * 1000000
                                liquidations['pressure'] = 'longs_pressured'
                            elif funding_rate < -0.001:
                                liquidations['short_liquidations'] = abs(funding_rate) * 1000000
                                liquidations['pressure'] = 'shorts_pressured'
                            else:
                                liquidations['pressure'] = 'neutral'
                                
                            liquidations['funding_rate'] = funding_rate * 100  # As percentage
                            
                await self.redis_client.hset(
                    'data:liquidations',
                    mapping={k: str(v) for k, v in liquidations.items()}
                )
                
                self.last_update['liquidations'] = datetime.utcnow()
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Liquidations collection error: {e}")
                await asyncio.sleep(30)
                
    async def _collect_fear_greed_index(self):
        """Collect Fear & Greed Index (real API)"""
        while self.is_running:
            try:
                # Alternative.me Fear & Greed Index (free, real data)
                url = "https://api.alternative.me/fng/?limit=1"
                response = await self.http_client.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    fng_data = data.get('data', [{}])[0]
                    
                    await self.redis_client.hset(
                        'data:fear_greed',
                        mapping={
                            'value': fng_data.get('value', '50'),
                            'classification': fng_data.get('value_classification', 'Neutral'),
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    )
                    
                self.last_update['fear_greed'] = datetime.utcnow()
                await asyncio.sleep(3600)  # Every hour (data updates daily)
                
            except Exception as e:
                logger.error(f"Fear & Greed index error: {e}")
                await asyncio.sleep(300)
                
    async def _collect_onchain_metrics(self):
        """Collect on-chain metrics (real data from public APIs)"""
        while self.is_running:
            try:
                metrics = {}
                
                # Bitcoin network stats from blockchain.info
                try:
                    stats_url = "https://api.blockchain.info/stats"
                    response = await self.http_client.get(stats_url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        metrics['btc_hash_rate'] = data.get('hash_rate', 0) / 1e18  # EH/s
                        metrics['btc_difficulty'] = data.get('difficulty', 0)
                        metrics['btc_blocks_mined'] = data.get('n_blocks_mined', 0)
                        metrics['btc_total_btc'] = data.get('totalbc', 0) / 100000000
                        metrics['btc_market_cap'] = data.get('market_price_usd', 0) * metrics.get('btc_total_btc', 0)
                except Exception as e:
                    logger.debug(f"Blockchain.info error: {e}")
                    
                # Gas prices from Etherscan (if we have API key) or alternative
                try:
                    gas_url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
                    response = await self.http_client.get(gas_url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        result = data.get('result', {})
                        if isinstance(result, dict):
                            metrics['eth_gas_slow'] = int(result.get('SafeGasPrice', 0))
                            metrics['eth_gas_average'] = int(result.get('ProposeGasPrice', 0))
                            metrics['eth_gas_fast'] = int(result.get('FastGasPrice', 0))
                except Exception as e:
                    logger.debug(f"Etherscan error: {e}")
                    
                if metrics:
                    await self.redis_client.hset(
                        'data:onchain',
                        mapping={k: str(v) for k, v in metrics.items()}
                    )
                    
                self.last_update['onchain'] = datetime.utcnow()
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"On-chain metrics error: {e}")
                await asyncio.sleep(120)
                
    async def get_aggregated_data(self) -> Dict[str, Any]:
        """Get all aggregated data"""
        data = {}
        
        # Whale alerts
        whale_data = await self.redis_client.get('data:whale_alerts')
        if whale_data:
            data['whale_alerts'] = json.loads(whale_data)
            
        # Whale sentiment
        whale_sentiment = await self.redis_client.hgetall('data:whale_sentiment')
        if whale_sentiment:
            data['whale_sentiment'] = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in whale_sentiment.items()
            }
            
        # Exchange announcements
        announcements = await self.redis_client.get('data:exchange_announcements')
        if announcements:
            data['announcements'] = json.loads(announcements)
            
        # Liquidations
        liquidations = await self.redis_client.hgetall('data:liquidations')
        if liquidations:
            data['liquidations'] = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in liquidations.items()
            }
            
        # Fear & Greed
        fng = await self.redis_client.hgetall('data:fear_greed')
        if fng:
            data['fear_greed'] = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in fng.items()
            }
            
        # On-chain
        onchain = await self.redis_client.hgetall('data:onchain')
        if onchain:
            data['onchain'] = {
                k.decode() if isinstance(k, bytes) else k:
                float(v.decode() if isinstance(v, bytes) else v)
                for k, v in onchain.items()
            }
            
        return data
        
    async def get_market_insight(self) -> str:
        """Generate AI insight based on all data"""
        data = await self.get_aggregated_data()
        
        insights = []
        
        # Fear & Greed analysis
        fng = data.get('fear_greed', {})
        fng_value = int(fng.get('value', 50))
        fng_class = fng.get('classification', 'Neutral')
        
        if fng_value < 25:
            insights.append(f"Extreme Fear ({fng_value}) - potential buying opportunity")
        elif fng_value > 75:
            insights.append(f"Extreme Greed ({fng_value}) - consider reducing exposure")
            
        # Whale activity
        whale = data.get('whale_sentiment', {})
        whale_activity = whale.get('activity', 'low')
        if whale_activity == 'high':
            insights.append("High whale activity detected")
            
        # Liquidation pressure
        liq = data.get('liquidations', {})
        pressure = liq.get('pressure', 'neutral')
        if pressure == 'longs_pressured':
            insights.append("Long positions under pressure")
        elif pressure == 'shorts_pressured':
            insights.append("Short positions under pressure")
            
        # Combine insights
        if not insights:
            return f"Market sentiment: {fng_class}. Normal conditions, monitoring continues."
            
        return " | ".join(insights[:2])

