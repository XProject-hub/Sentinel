"""
SENTINEL AI - SMART Autonomous Trading System
24/7 Intelligent trading with real AI predictions

SMART MODE Strategy:
- Uses sentiment, news, fear & greed index for predictions
- Dynamic position sizing based on AI confidence (5-20% per trade)
- Higher confidence threshold (65%+) for quality trades
- Smart trailing stop: activate at +0.8%, trail by 0.4%
- Analyzes market regime and adapts strategy
- Learns from every trade outcome
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from loguru import logger
import redis.asyncio as redis
import json

from config import settings
from services.bybit_client import BybitV5Client
from services.learning_engine import LearningEngine, TradeOutcome


def safe_float(val, default=0.0):
    """Safely convert value to float"""
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


@dataclass
class TradeSignal:
    """Trading signal from AI analysis"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    strategy: str
    regime: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_percent: float
    reasoning: str
    sentiment_score: float = 0.0
    fear_greed: int = 50


class AutonomousTrader:
    """
    SMART AI-Driven Autonomous Trading System.
    
    Features:
    - Combines technical analysis with sentiment & news
    - Dynamic position sizing based on confidence
    - Adaptive strategy selection based on market regime
    - Continuous learning from trade outcomes
    """
    
    def __init__(self):
        self.is_running = False
        self.redis_client = None
        self.learning_engine: Optional[LearningEngine] = None
        
        # Connected exchange clients per user
        self.user_clients: Dict[str, BybitV5Client] = {}
        
        # ALL Bybit USDT Perpetual pairs - 200+ coins (Complete List)
        self.trading_pairs = [
            # === TOP TIER: Major Cryptocurrencies ===
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'TRXUSDT', 'DOTUSDT',
            'LINKUSDT', 'MATICUSDT', 'SHIBUSDT', 'LTCUSDT', 'ATOMUSDT',
            'UNIUSDT', 'XLMUSDT', 'ETCUSDT', 'BCHUSDT', 'APTUSDT',
            
            # === LAYER 1 BLOCKCHAINS ===
            'NEARUSDT', 'FILUSDT', 'ICPUSDT', 'HBARUSDT', 'VETUSDT',
            'ALGOUSDT', 'FTMUSDT', 'EGLDUSDT', 'FLOWUSDT', 'NEOUSDT',
            'XTZUSDT', 'EOSUSDT', 'IOTAUSDT', 'WAVESUSDT', 'ZILUSDT',
            'ONTUSDT', 'IOSTUSDT', 'QNTUSDT', 'KASUSDT', 'SUIUSDT',
            'SEIUSDT', 'TIAUSDT', 'TONUSDT', 'AABORUSDT',
            
            # === LAYER 2 & SCALING ===
            'ARBUSDT', 'OPUSDT', 'STRKUSDT', 'MANTAUSDT', 'BLASTUSDT',
            'ZKUSDT', 'SCROLLUSDT', 'LINEAUSDT', 'METISUSDT', 'BOBAUSDT',
            'IMXUSDT', 'LRCUSDT', 'CELRUSDT', 'CTSIUSDT', 'SKLUSDT',
            
            # === DEFI PROTOCOLS ===
            'AAVEUSDT', 'MKRUSDT', 'UNIUSDT', 'CRVUSDT', 'SNXUSDT',
            'COMPUSDT', '1INCHUSDT', 'YFIUSDT', 'SUSHIUSDT', 'BALUSDT',
            'DYDXUSDT', 'GMXUSDT', 'PENDLEUSDT', 'LDOUSDT', 'RDNTUSDT',
            'WOOUSDT', 'PERPUSDT', 'API3USDT', 'ZRXUSDT', 'RENUSDT',
            'RUNEUSDT', 'KAVAUSDT', 'CELOUSDT', 'JOEUSDT', 'QUICKUSDT',
            'UMAUSDT', 'RPLETHUSDT', 'STGUSDT', 'CAKEUSDT', 'OSMOUSDT',
            
            # === AI & DEPIN ===
            'FETUSDT', 'AGIXUSDT', 'OCEANUSDT', 'RNDRUSDT', 'TAOUSDT',
            'WLDUSDT', 'AKTUSDT', 'ARKMUSDT', 'AIUSDT', 'NFPUSDT',
            'ABORUSDT', 'PHBUSDT', 'RLCUSDT', 'GLMUSDT', 'IOTXUSDT',
            
            # === MEME COINS ===
            'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT', 'BOMEUSDT',
            'MABORUSDT', 'NEIROUSDT', 'NOTUSDT', 'MEMEUSD', 'PEOPLEUSDT',
            'SATSUSDT', 'RATSUSDT', 'MEWUSDT', 'DOGUSDT', 'COQIUSDT',
            'MYRIAUSDT', 'BIDENUSDT', 'TURBOUSUSDT', 'LADYSUSDT',
            
            # === GAMING & METAVERSE ===
            'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'GALAUSDT', 'ENJUSDT',
            'ILVUSDT', 'MAGICUSDT', 'BEAMUSDT', 'PIXELUSDT', 'APEUSDT',
            'GMTUSDT', 'LOOKSUSDT', 'BLURUSDT', 'HIGHUSDT', 'YGGUSDT',
            'PRIMEUSDT', 'PORTALUSDT', 'XABORUSDT', 'ALICEUSDT', 'TLMUSDT',
            'SLPUSDT', 'MCUSDT', 'STARUSDT', 'DARLUSDT', 'VABORUSDT',
            
            # === ORACLES & DATA ===
            'LINKUSDT', 'GRTUSDT', 'BANDUSDT', 'APIUSDT', 'PYABORUSDT',
            'PYTHUSDT', 'FLUXUSDT', 'REQUSDT', 'TELUSUSDT',
            
            # === EXCHANGE TOKENS ===
            'BNBUSDT', 'OKBUSDT', 'HTUSDT', 'MXUSDT', 'GTUSDT',
            'KCSUSDT', 'CROUSDT', 'FTUSDT', 'LEXUSDT',
            
            # === STORAGE & COMPUTING ===
            'FILUSDT', 'ARUSDT', 'STORJUSDT', 'SIACUSDT', 'BTUSDT',
            'ANKRUSDT', 'NGLUSDT', 'BLOBUSDT',
            
            # === PRIVACY COINS ===
            'XMRUSDT', 'ZECUSDT', 'DASHUSDT', 'SCRTUSDT', 'ROSEUSDT',
            'OASYUSDT', 'ZENUSDT', 'ERNUSDT',
            
            # === STAKING & LIQUID STAKING ===
            'LDOUSDT', 'RPLETHUSDT', 'CBETHUSDT', 'FRXETHUSDT', 'SWETHUSDT',
            'ANKRUSDT', 'SSVUSDT', 'OETHUSDT',
            
            # === SOCIAL & CONTENT ===
            'MASKUSDT', 'GALAUSDT', 'CHZUSDT', 'AUDIOUSDT', 'LIVEPEERUSDT',
            'THETAUSDT', 'TFUELUSDT', 'RALLYUSDT', 'BATUSDT', 'STEEMUSDT',
            
            # === CROSS-CHAIN & BRIDGES ===
            'ATOMUSDT', 'RUNEUSDT', 'RENUSD', 'SYNUSDT', 'WOOFIUSDT',
            'STGUSDT', 'MULTIUSDT', 'CELRUSDT', 'HOPUSDT', 'LIFIUSDT',
            
            # === BITCOIN ECOSYSTEM ===
            'ORDIUSDT', 'SATSUSDT', 'RATSUSDT', 'STXUSDT', 'CFXUSDT',
            'CKBUSDT', 'RSRUSDT', 'BSVUSDT', 'BTGUSDT', 'BCHUSDT',
            
            # === SOLANA ECOSYSTEM ===
            'SOLUSDT', 'RAYUSDT', 'SRMUSDT', 'ABORUSDT', 'MNDEUSDT',
            'ORCAUSDT', 'PORTOUSDT', 'SABORUSDT', 'STEPUSDT', 'SLNDUSDT',
            
            # === OTHER TRENDING ===
            'INJUSDT', 'JUPUSDT', 'JTOUSDT', 'ONDOUSDT', 'STRKUSDT',
            'TAOABRUSDT', 'WABRUSDT', 'ENAUSDT', 'ETHFIUSDT', 'ALTUSDT',
            'ACEUSDT', 'XAIUSDT', 'MANTAUSDT', 'DYMUSDT', 'PIXELUSDT',
            'PORTALUSDT', 'STRKUSDT', 'AEVOUSDT', 'MAVIAUSDT', 'VANRYUSDT',
            
            # === ADDITIONAL COINS ===
            'COTIUSDT', 'MTLUSDT', 'JASMYUSDT', 'ACHUSDT', 'HOTUSDT',
            'DENTUSDT', 'TUSDT', 'AGLDUSDT', 'ENSUSDT', 'BICOUSDT',
            'MINAUSDT', 'KSMUSDT', 'TRBUSDT', 'NMRUSDT', 'MLNUSDT',
            'LPTUSDT', 'CVXUSDT', 'FXSUSDT', 'FRAXUSDT', 'ALPHABUSDT',
            'SPELLUSDT', 'ICXUSDT', 'SCUSDT', 'KLAYUSDT', 'ONGUSDT',
            'WINUSDT', 'SUNUSDT', 'JSTUSDT', 'NKNUSDT', 'STMXUSDT',
            'DGBUSDT', 'RVNUSDT', 'KAVAUSDT', 'HARDUSDT', 'MDXUSDT',
            'PERPUSDT', 'LINAUSDT', 'XEMUSDT', 'CTKUSDT', 'TOMOUSDT',
            'PUNDIXUSDT', 'AKROUSDT', 'BAKEUSDT', 'BURGERUSDT', 'SFPUSDT',
            'LITUSDT', 'VIDTUSDT', 'EPXUSDT', 'GMTUSDT', 'ARPAUSDT',
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for pair in self.trading_pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)
        self.trading_pairs = unique_pairs
        
        # ============================================
        # ============================================
        # PROFITABLE BOT PARAMETERS
        # Key: Quality over quantity, let winners run!
        # ============================================
        
        # Analysis settings - QUALITY SIGNALS ONLY
        self.analysis_interval_seconds = 30  # Analyze every 30s (patience)
        self.min_confidence = 70.0           # Only HIGH confidence trades!
        
        # Position sizing - Conservative for safety
        self.min_position_percent = 4.0      # 4% per trade (~€5.6)
        self.max_position_percent = 8.0      # Max 8% per trade (~€11.2)
        self.max_open_positions = 8          # Max 8 positions (focus)
        
        # === PROFITABLE EXIT STRATEGY ===
        # The KEY to profitability: Let winners run, cut losers fast
        
        # STOP LOSS: Cut losers at -1.5%
        self.emergency_stop_loss = 1.5       # Exit if -1.5% from entry
        
        # TAKE PROFIT: Activate trailing ONLY after +1% profit
        self.min_profit_to_trail = 1.0       # Need +1% profit before trailing activates
        
        # TRAILING STOP: Once in profit, trail at 1.2% from peak
        self.trail_from_peak = 1.2           # Wider trailing (was 0.5%)
        
        # This gives us:
        # - Risk: -1.5% (stop loss)
        # - Reward: +1% minimum, potentially unlimited with trailing
        # - Risk/Reward ratio: At least 1:0.67, improves as profits run
        
        # Market filters - Focus on liquid pairs
        self.min_24h_volume = 5_000_000      # $5M daily volume minimum
        self.max_spread_percent = 0.15       # Max 0.15% spread (liquidity)
        
        # Track state
        self.last_trade_time: Dict[str, datetime] = {}
        self.active_positions: Dict[str, Dict] = {}
        self.market_data_cache: Dict[str, Dict] = {}
        
    async def initialize(self, learning_engine: LearningEngine):
        """Initialize autonomous trader"""
        logger.info("Initializing SMART AI Trading System...")
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.learning_engine = learning_engine
        self.is_running = True
        
        # Dynamically load available symbols after first user connects
        self.dynamic_symbols_loaded = False
        
        # Load saved settings from Redis
        await self._load_settings_from_redis()
        
        logger.info(f"SMART AI Trading System initialized with {len(self.trading_pairs)} default pairs")
        logger.info("Will fetch ALL available Bybit pairs when first user connects")
    
    async def _load_settings_from_redis(self):
        """Load user settings from Redis"""
        try:
            settings_data = await self.redis_client.hgetall('bot:settings')
            
            if settings_data:
                parsed = {
                    k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v 
                    for k, v in settings_data.items()
                }
                
                # Apply settings
                self.emergency_stop_loss = float(parsed.get('stopLossPercent', self.emergency_stop_loss))
                self.trail_from_peak = float(parsed.get('trailingStopPercent', self.trail_from_peak))
                self.min_profit_to_trail = float(parsed.get('minProfitToTrail', self.min_profit_to_trail))
                self.min_confidence = float(parsed.get('minConfidence', self.min_confidence))
                self.max_position_percent = float(parsed.get('maxPositionPercent', self.max_position_percent))
                self.max_open_positions = int(float(parsed.get('maxOpenPositions', self.max_open_positions)))
                
                # Take profit (new)
                self.take_profit_percent = float(parsed.get('takeProfitPercent', 5.0))
                
                logger.info(f"Loaded settings: SL={self.emergency_stop_loss}%, "
                           f"Trail={self.trail_from_peak}%, MinProfit={self.min_profit_to_trail}%, "
                           f"TP={self.take_profit_percent}%, Confidence={self.min_confidence}%")
            else:
                self.take_profit_percent = 5.0  # Default
                
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            self.take_profit_percent = 5.0
    
    async def _load_all_bybit_symbols(self, client: BybitV5Client):
        """Dynamically fetch ALL available USDT perpetual pairs from Bybit"""
        if self.dynamic_symbols_loaded:
            return
            
        try:
            logger.info("Fetching ALL available trading pairs from Bybit...")
            
            # Get all linear (USDT perpetual) symbols
            all_symbols = await client.get_all_symbols(category="linear")
            
            if all_symbols and len(all_symbols) > 50:
                # Use all fetched symbols
                self.trading_pairs = all_symbols
                self.dynamic_symbols_loaded = True
                logger.info(f"Loaded {len(self.trading_pairs)} USDT perpetual pairs from Bybit!")
                
                # Store in Redis for reference
                await self.redis_client.set(
                    'trading:available_symbols',
                    ','.join(self.trading_pairs)
                )
                await self.redis_client.set(
                    'trading:symbols_count',
                    str(len(self.trading_pairs))
                )
            else:
                logger.warning(f"Only found {len(all_symbols) if all_symbols else 0} symbols, keeping defaults")
                
        except Exception as e:
            logger.error(f"Failed to load dynamic symbols: {e}")
            # Keep using default symbols
        
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down SMART AI Trader...")
        self.is_running = False
        
        for user_id, client in self.user_clients.items():
            try:
                await client.close()
            except:
                pass
                
        if self.redis_client:
            await self.redis_client.close()
            
    async def connect_user(self, user_id: str, api_key: str, api_secret: str, testnet: bool = False):
        """Connect a user's exchange account"""
        try:
            client = BybitV5Client(api_key, api_secret, testnet)
            result = await client.test_connection()
            
            if result.get('success'):
                self.user_clients[user_id] = client
                logger.info(f"User {user_id} connected for SMART AI trading")
                
                # Load ALL available symbols from Bybit (only once)
                await self._load_all_bybit_symbols(client)
                
                return True
            else:
                await client.close()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect user {user_id}: {e}")
            return False
            
    async def disconnect_user(self, user_id: str):
        """Disconnect user"""
        if user_id in self.user_clients:
            await self.user_clients[user_id].close()
            del self.user_clients[user_id]
            logger.info(f"User {user_id} disconnected")
            
    async def run_trading_loop(self):
        """Main trading loop - SMART AI mode"""
        logger.info("Starting SMART AI trading loop...")
        
        cycle_count = 0
        
        while self.is_running:
            try:
                cycle_count += 1
                cycle_start = datetime.utcnow()
                
                # Get global market data first
                market_sentiment = await self._get_market_sentiment()
                
                for user_id, client in list(self.user_clients.items()):
                    try:
                        await self._process_user_trading(user_id, client, market_sentiment)
                    except Exception as e:
                        logger.error(f"Error processing user {user_id}: {e}")
                        
                # Log status every 50 cycles
                if cycle_count % 50 == 0:
                    await self._log_trading_status()
                    
                # Wait for next analysis cycle
                cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time = max(self.analysis_interval_seconds - cycle_duration, 10)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(30)
                
    async def _get_market_sentiment(self) -> Dict:
        """Get overall market sentiment from collected data"""
        try:
            # Get Fear & Greed Index
            fear_greed_data = await self.redis_client.get('data:fear_greed')
            fear_greed = 50
            fg_label = 'Neutral'
            if fear_greed_data:
                fg = json.loads(fear_greed_data)
                fear_greed = int(fg.get('value', 50))
                fg_label = fg.get('classification', 'Neutral')
                
            # Get BTC dominance and market trend
            btc_data = await self.redis_client.hgetall('regime:BTCUSDT')
            btc_trend = 'sideways'
            if btc_data:
                btc_trend = btc_data.get(b'trend', b'sideways').decode()
                
            # Get news sentiment
            news_data = await self.redis_client.get('data:news_sentiment')
            news_sentiment = 0
            if news_data:
                news = json.loads(news_data)
                news_sentiment = float(news.get('overall_sentiment', 0))
                
            return {
                'fear_greed': fear_greed,
                'fear_greed_label': fg_label,
                'btc_trend': btc_trend,
                'news_sentiment': news_sentiment,
                'is_extreme_fear': fear_greed < 25,
                'is_extreme_greed': fear_greed > 75,
                'is_bullish': fear_greed > 55 and btc_trend == 'bullish',
                'is_bearish': fear_greed < 45 and btc_trend == 'bearish',
            }
            
        except Exception as e:
            logger.warning(f"Error getting market sentiment: {e}")
            return {'fear_greed': 50, 'fear_greed_label': 'Neutral', 'btc_trend': 'sideways'}
            
    async def _process_user_trading(self, user_id: str, client: BybitV5Client, market_sentiment: Dict):
        """Process trading for a user with SMART AI"""
        
        # 1. Get current balance
        balance_result = await client.get_wallet_balance()
        if not balance_result.get('success'):
            return
            
        balance_data = balance_result.get('data', {})
        total_equity = 0
        available_usdt = 0
        
        for account in balance_data.get('list', []):
            total_equity = safe_float(account.get('totalEquity'))
            available_margin = safe_float(account.get('totalAvailableBalance'))
            
            for coin in account.get('coin', []):
                if coin.get('coin') == 'USDT':
                    wallet_bal = safe_float(coin.get('walletBalance'))
                    avail_bal = safe_float(coin.get('availableToWithdraw'))
                    available_usdt = max(wallet_bal, avail_bal, available_margin)
                    break
                    
            # Fallback
            if available_usdt < 10 and total_equity > 20:
                available_usdt = total_equity * 0.4  # 40% available
                
        if total_equity < 20:  # Minimum $20 to trade smart
            return
            
        logger.info(f"User {user_id}: Equity=${total_equity:.2f}, Available=${available_usdt:.2f}")
            
        # 2. Get current positions
        positions_result = await client.get_positions()
        current_positions = []
        
        if positions_result.get('success'):
            for pos in positions_result.get('data', {}).get('list', []):
                size = safe_float(pos.get('size'))
                if size > 0:
                    current_positions.append({
                        'symbol': pos.get('symbol'),
                        'side': pos.get('side'),
                        'size': size,
                        'entryPrice': safe_float(pos.get('avgPrice')),
                        'unrealizedPnl': safe_float(pos.get('unrealisedPnl')),
                    })
                    
        # 3. Check positions for SMART exit
        for position in current_positions:
            await self._check_smart_exit(user_id, client, position)
            
        # 4. Analyze for new opportunities (if we have room)
        if len(current_positions) < self.max_open_positions:
            await self._find_smart_trades(user_id, client, current_positions, 
                                          available_usdt, total_equity, market_sentiment)
                    
    async def _find_smart_trades(self, user_id: str, client: BybitV5Client, 
                                  current_positions: List, available_usdt: float,
                                  total_equity: float, market_sentiment: Dict):
        """Find high-quality trade opportunities using AI analysis"""
        
        signals_found = []
        
        for symbol in self.trading_pairs:
            # Skip if already in position
            if any(p['symbol'] == symbol for p in current_positions):
                continue
                
            # Skip if traded recently
            last_trade = self.last_trade_time.get(f"{user_id}:{symbol}")
            if last_trade and (datetime.utcnow() - last_trade).seconds < self.analysis_interval_seconds:
                continue
            
            # Skip if this symbol has been consistently losing
            symbol_stats = await self.redis_client.hgetall(f"symbol:stats:{symbol}")
            if symbol_stats:
                wins = int(symbol_stats.get(b'wins', 0))
                losses = int(symbol_stats.get(b'losses', 0))
                total = wins + losses
                if total >= 3 and losses > wins * 2:  # More than 2:1 loss ratio
                    logger.debug(f"SKIPPING {symbol}: AI learned it loses (W:{wins}/L:{losses})")
                    continue
                
            # Deep analysis
            signal = await self._analyze_with_ai(symbol, client, market_sentiment)
            
            if signal and signal.confidence >= self.min_confidence:
                signals_found.append(signal)
                
        # Sort by confidence and execute top signals
        signals_found.sort(key=lambda x: x.confidence, reverse=True)
        
        for signal in signals_found[:3]:  # Max 3 new trades per cycle
            if available_usdt >= 10:
                logger.info(f"SMART TRADE: {signal.symbol} {signal.action.upper()} "
                           f"confidence={signal.confidence:.0f}% reason={signal.reasoning}")
                await self._execute_smart_trade(user_id, client, signal, available_usdt, total_equity)
                available_usdt -= total_equity * (signal.position_size_percent / 100)
                
    async def _analyze_with_ai(self, symbol: str, client: BybitV5Client, 
                                market_sentiment: Dict) -> Optional[TradeSignal]:
        """Deep AI analysis combining multiple factors"""
        
        try:
            # Get real-time market data
            ticker_result = await client.get_tickers(symbol=symbol)
            if not ticker_result.get('success'):
                return None
                
            tickers = ticker_result.get('data', {}).get('list', [])
            if not tickers:
                return None
                
            ticker = tickers[0]
            last_price = safe_float(ticker.get('lastPrice'))
            price_change_24h = safe_float(ticker.get('price24hPcnt')) * 100
            volume_24h = safe_float(ticker.get('volume24h'))
            funding_rate = safe_float(ticker.get('fundingRate')) * 100
            high_24h = safe_float(ticker.get('highPrice24h'))
            low_24h = safe_float(ticker.get('lowPrice24h'))
            
            # Filter: Skip low volume
            if volume_24h < self.min_24h_volume:
                return None
                
            # Get stored regime data
            regime_data = await self.redis_client.hgetall(f"regime:{symbol}")
            
            if regime_data:
                regime = regime_data.get(b'regime', b'sideways').decode()
                volatility = safe_float(regime_data.get(b'volatility', b'1.5').decode())
                trend = regime_data.get(b'trend', b'sideways').decode()
                rsi = safe_float(regime_data.get(b'rsi', b'50').decode())
            else:
                # Calculate from price action
                if price_change_24h > 3:
                    regime = 'bull_trend'
                    trend = 'bullish'
                elif price_change_24h < -3:
                    regime = 'bear_trend'
                    trend = 'bearish'
                else:
                    regime = 'sideways'
                    trend = 'sideways'
                volatility = abs(price_change_24h) / 2
                rsi = 50 + (price_change_24h * 3)
                rsi = max(20, min(80, rsi))
                
            # Calculate price position in 24h range
            range_24h = high_24h - low_24h
            if range_24h > 0:
                price_position = (last_price - low_24h) / range_24h  # 0 = at low, 1 = at high
            else:
                price_position = 0.5
                
            # ============================================
            # SMART AI SIGNAL GENERATION
            # ============================================
            
            fear_greed = market_sentiment.get('fear_greed', 50)
            btc_trend = market_sentiment.get('btc_trend', 'sideways')
            
            # Base confidence from learning engine
            best_strategy, q_value = self.learning_engine.get_best_strategy(regime)
            base_confidence = self.learning_engine.get_strategy_confidence(regime, best_strategy)
            
            action = 'hold'
            reasoning_parts = []
            confidence_adjustments = 0
            
            # === BULLISH SIGNALS ===
            bullish_score = 0
            
            # RSI oversold
            if rsi < 35:
                bullish_score += 20
                reasoning_parts.append(f"RSI oversold ({rsi:.0f})")
                
            # Price near 24h low
            if price_position < 0.3:
                bullish_score += 15
                reasoning_parts.append("Near 24h low")
                
            # Positive momentum
            if 0.5 < price_change_24h < 5:
                bullish_score += 10
                reasoning_parts.append(f"+{price_change_24h:.1f}% momentum")
                
            # Fear in market (contrarian)
            if fear_greed < 35:
                bullish_score += 15
                reasoning_parts.append(f"Extreme fear ({fear_greed})")
                
            # Funding rate negative (shorts paying)
            if funding_rate < -0.01:
                bullish_score += 10
                reasoning_parts.append("Shorts paying funding")
                
            # === BEARISH SIGNALS ===
            bearish_score = 0
            
            # RSI overbought
            if rsi > 65:
                bearish_score += 20
                reasoning_parts.append(f"RSI overbought ({rsi:.0f})")
                
            # Price near 24h high
            if price_position > 0.7:
                bearish_score += 15
                reasoning_parts.append("Near 24h high")
                
            # Negative momentum
            if -5 < price_change_24h < -0.5:
                bearish_score += 10
                reasoning_parts.append(f"{price_change_24h:.1f}% momentum")
                
            # Greed in market (contrarian)
            if fear_greed > 65:
                bearish_score += 15
                reasoning_parts.append(f"Extreme greed ({fear_greed})")
                
            # Funding rate positive (longs paying)
            if funding_rate > 0.01:
                bearish_score += 10
                reasoning_parts.append("Longs paying funding")
                
            # === DECISION - ONLY HIGH QUALITY SIGNALS ===
            
            # CRITICAL: Check if learning engine says this regime is PROFITABLE
            # If Q-value is negative, AI has learned this regime loses money!
            if q_value < -0.5:
                logger.debug(f"{symbol}: AI learned to AVOID {regime} (Q={q_value:.2f})")
                return None
            
            # Need STRONG edge to trade (was 30, now 40)
            # Also need to beat opposite score by at least 15 (was 10)
            if bullish_score >= 40 and bullish_score > bearish_score + 15:
                action = 'buy'
                confidence_adjustments = bullish_score
            elif bearish_score >= 40 and bearish_score > bullish_score + 15:
                action = 'sell'
                confidence_adjustments = bearish_score
                
            if action == 'hold':
                return None
            
            # CRITICAL: Check if this specific strategy has worked
            # Learned strategies with negative Q-values = avoid!
            strategy_q = self.learning_engine.strategy_q_values.get(regime, {}).get(best_strategy, 0)
            if strategy_q < -1.0:
                logger.debug(f"{symbol}: AI learned {best_strategy} FAILS in {regime} (Q={strategy_q:.2f})")
                return None
                
            # Extra filter: Don't buy when market is extremely greedy
            if action == 'buy' and fear_greed > 75:
                logger.debug(f"{symbol}: Skip BUY - market too greedy ({fear_greed})")
                return None
                
            # Extra filter: Don't sell when market is extremely fearful
            if action == 'sell' and fear_greed < 25:
                logger.debug(f"{symbol}: Skip SELL - market too fearful ({fear_greed})")
                return None
                
            # Calculate final confidence (stricter formula)
            # BOOST confidence if AI has learned this strategy works well (positive Q-value)
            q_bonus = max(0, q_value * 10)  # Up to +15% for well-learned strategies
            final_confidence = min(95, base_confidence + (confidence_adjustments * 0.4) + q_bonus)
            
            # Log when AI knowledge is being applied
            if q_value > 0.5:
                logger.info(f"{symbol}: AI LEARNED {best_strategy} works in {regime} (Q={q_value:.2f}, +{q_bonus:.0f}% bonus)")
            
            # Skip if below threshold
            if final_confidence < self.min_confidence:
                return None
                
            # Calculate emergency stop loss (only for extreme cases)
            if action == 'buy':
                stop_loss = last_price * (1 - self.emergency_stop_loss / 100)
            else:
                stop_loss = last_price * (1 + self.emergency_stop_loss / 100)
                
            # Dynamic position size based on confidence
            # 65% confidence = 5% position, 95% confidence = 20% position
            confidence_factor = (final_confidence - 65) / 30  # 0 to 1
            position_size = self.min_position_percent + (
                confidence_factor * (self.max_position_percent - self.min_position_percent)
            )
            position_size = max(self.min_position_percent, min(self.max_position_percent, position_size))
            
            reasoning = " | ".join(reasoning_parts[:3])  # Top 3 reasons
            
            return TradeSignal(
                symbol=symbol,
                action=action,
                confidence=final_confidence,
                strategy=best_strategy,
                regime=regime,
                entry_price=last_price,
                stop_loss=stop_loss,
                take_profit=0,  # Using trailing stop instead
                position_size_percent=position_size,
                reasoning=reasoning,
                sentiment_score=0,
                fear_greed=fear_greed,
            )
            
        except Exception as e:
            logger.error(f"AI analysis error for {symbol}: {e}")
            return None
            
    async def _execute_smart_trade(self, user_id: str, client: BybitV5Client, 
                                    signal: TradeSignal, available_usdt: float, 
                                    total_equity: float):
        """Execute trade with smart position sizing"""
        
        try:
            # Calculate position value based on confidence-adjusted size
            trade_value = total_equity * (signal.position_size_percent / 100)
            trade_value = min(trade_value, available_usdt * 0.9)  # Keep 10% buffer
            
            # Bybit minimum is $5, we use $5.5 for safety
            if trade_value < 5.5:
                logger.warning(f"Trade value too low: ${trade_value:.2f} (min $5.5)")
                return
                
            # Calculate quantity
            quantity = trade_value / signal.entry_price
            
            # Round based on symbol requirements
            symbol = signal.symbol
            if symbol in ['BTCUSDT']:
                quantity = max(0.001, round(quantity, 3))
            elif symbol in ['ETHUSDT']:
                quantity = max(0.01, round(quantity, 2))
            elif symbol in ['BNBUSDT', 'SOLUSDT', 'LTCUSDT']:
                quantity = max(0.1, round(quantity, 1))
            elif symbol in ['XRPUSDT', 'ADAUSDT', 'DOGEUSDT']:
                quantity = max(10, int(quantity))
            else:
                if signal.entry_price > 100:
                    quantity = max(0.1, round(quantity, 2))
                elif signal.entry_price > 1:
                    quantity = max(1, round(quantity, 1))
                else:
                    quantity = max(10, int(quantity))
                    
            order_value = quantity * signal.entry_price
            if order_value < 5:
                return
                
            side = 'Buy' if signal.action == 'buy' else 'Sell'
            
            logger.info(f"SMART ORDER: {side} {symbol} qty={quantity} @ ${signal.entry_price:.2f} "
                       f"(${order_value:.2f}, {signal.position_size_percent:.1f}% of portfolio)")
            
            order_result = await client.place_order(
                symbol=symbol,
                side=side,
                order_type='Market',
                qty=str(quantity),
            )
            
            if order_result.get('success'):
                logger.info(f"SMART ORDER SUCCESS: {symbol} {side}")
                
                # Record trade
                trade_record = {
                    'user_id': user_id,
                    'symbol': symbol,
                    'side': side.lower(),
                    'quantity': quantity,
                    'entry_price': signal.entry_price,
                    'strategy': signal.strategy,
                    'regime': signal.regime,
                    'confidence': signal.confidence,
                    'position_size_percent': signal.position_size_percent,
                    'reasoning': signal.reasoning,
                    'fear_greed': signal.fear_greed,
                    'timestamp': datetime.utcnow().isoformat(),
                }
                
                await self.redis_client.lpush(f"trades:active:{user_id}", json.dumps(trade_record))
                self.last_trade_time[f"{user_id}:{symbol}"] = datetime.utcnow()
                
                # Initialize peak tracking
                peak_key = f"peak:{user_id}:{symbol}"
                await self.redis_client.set(peak_key, str(signal.entry_price))
                
            else:
                logger.error(f"SMART ORDER FAILED: {order_result.get('error')}")
                
        except Exception as e:
            logger.error(f"Smart trade execution error: {e}")
            
    async def _check_smart_exit(self, user_id: str, client: BybitV5Client, position: Dict):
        """
        TRAILING STOP from PEAK - Always active!
        
        Logic:
        - Buy at €100
        - Price goes: €100 → €105 → €107 (peak updates to €107)
        - Price drops to €106 (not 0.8% from peak, HOLD)
        - Price goes: €106 → €110 (peak updates to €110)
        - Price drops to €109 (0.8% from €110 = €109.12, so SELL!)
        
        The peak ALWAYS updates when price goes higher.
        We SELL when price drops 0.8% from the PEAK (not from entry!).
        """
        
        symbol = position['symbol']
        entry_price = position['entryPrice']
        
        # Get current price
        ticker_result = await client.get_tickers(symbol=symbol)
        if not ticker_result.get('success'):
            return
            
        tickers = ticker_result.get('data', {}).get('list', [])
        if not tickers:
            return
            
        current_price = safe_float(tickers[0].get('lastPrice'))
        
        # Calculate profit/loss from ENTRY (for emergency stop only)
        if position['side'].lower() == 'buy':
            pnl_from_entry = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_from_entry = ((entry_price - current_price) / entry_price) * 100
            
        # Get/update PEAK price - ALWAYS track highest
        peak_key = f"peak:{user_id}:{symbol}"
        peak_data = await self.redis_client.get(peak_key)
        
        if peak_data:
            peak_price = float(peak_data)
            
            # UPDATE PEAK if current price is HIGHER (for longs) or LOWER (for shorts)
            if position['side'].lower() == 'buy' and current_price > peak_price:
                peak_price = current_price
                await self.redis_client.set(peak_key, str(peak_price))
                logger.debug(f"{symbol}: New peak! ${peak_price:.2f}")
            elif position['side'].lower() == 'sell' and current_price < peak_price:
                peak_price = current_price
                await self.redis_client.set(peak_key, str(peak_price))
        else:
            # First check - initialize peak as entry price
            peak_price = entry_price
            await self.redis_client.set(peak_key, str(peak_price))
            
        # Calculate DROP FROM PEAK (not from entry!)
        if position['side'].lower() == 'buy':
            drop_from_peak = ((peak_price - current_price) / peak_price) * 100
        else:
            drop_from_peak = ((current_price - peak_price) / peak_price) * 100
            
        # Calculate profit from entry (for logging)
        if position['side'].lower() == 'buy':
            profit_from_entry = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_from_entry = ((entry_price - current_price) / entry_price) * 100
            
        should_close = False
        close_reason = ""
        
        # === PROFITABLE EXIT LOGIC ===
        # Uses settings from user configuration
        
        # Get take profit from settings (reloaded periodically)
        take_profit = getattr(self, 'take_profit_percent', 5.0)
        
        # 1. STOP LOSS: Exit if down from ENTRY (cut losers fast!)
        if pnl_from_entry <= -self.emergency_stop_loss:
            should_close = True
            close_reason = f"STOP LOSS: {pnl_from_entry:.2f}% loss (limit: -{self.emergency_stop_loss}%)"
            
        # 2. TAKE PROFIT: Exit when profit reaches target
        elif profit_from_entry >= take_profit:
            should_close = True
            close_reason = f"TAKE PROFIT: +{profit_from_entry:.2f}% (target: +{take_profit}%)"
            
        # 3. TRAILING STOP: Only activates AFTER minimum profit
        #    This lets winners run while protecting profits
        elif profit_from_entry >= self.min_profit_to_trail:
            # We're in profit! Now use trailing stop from peak
            if drop_from_peak >= self.trail_from_peak:
                should_close = True
                close_reason = f"TRAILING STOP: +{profit_from_entry:.2f}% (dropped {drop_from_peak:.2f}% from peak)"
                
        # Log position status
        logger.debug(f"{symbol}: entry=${entry_price:.2f}, current=${current_price:.2f}, "
                    f"peak=${peak_price:.2f}, drop_from_peak={drop_from_peak:.2f}%")
                    
        if should_close:
            await self.redis_client.delete(peak_key)
            await self._close_position(user_id, client, position, close_reason)
            
    async def _close_position(self, user_id: str, client: BybitV5Client, 
                               position: Dict, reason: str):
        """Close position and record for learning"""
        
        try:
            symbol = position['symbol']
            size = position['size']
            side = 'Sell' if position['side'].lower() == 'buy' else 'Buy'
            
            order_result = await client.place_order(
                symbol=symbol,
                side=side,
                order_type='Market',
                qty=str(size),
                reduce_only=True,
            )
            
            if order_result.get('success'):
                pnl = position['unrealizedPnl']
                logger.info(f"Position closed: {symbol} PnL=€{pnl:.2f} - {reason}")
                
                # Record for learning
                await self._record_trade_outcome(user_id, position, reason)
                
        except Exception as e:
            logger.error(f"Close position error: {e}")
            
    async def _record_trade_outcome(self, user_id: str, position: Dict, reason: str):
        """Record trade for AI learning"""
        
        active_trades = await self.redis_client.lrange(f"trades:active:{user_id}", 0, -1)
        
        original_trade = None
        for trade_json in active_trades:
            trade = json.loads(trade_json)
            if trade.get('symbol') == position['symbol']:
                original_trade = trade
                break
                
        if original_trade:
            pnl = position['unrealizedPnl']
            entry_price = position['entryPrice']
            
            # Create outcome for learning
            outcome = TradeOutcome(
                symbol=position['symbol'],
                strategy=original_trade.get('strategy', 'unknown'),
                entry_price=entry_price,
                exit_price=entry_price + (pnl / position['size']) if position['size'] > 0 else entry_price,
                quantity=position['size'],
                side='long' if position['side'].lower() == 'buy' else 'short',
                pnl=pnl,
                pnl_percent=(pnl / entry_price / position['size']) * 100 if position['size'] > 0 else 0,
                hold_time_seconds=0,
                market_regime=original_trade.get('regime', 'unknown'),
                volatility_at_entry=1.5,
                sentiment_at_entry=0,
                timestamp=datetime.utcnow().isoformat(),
            )
            
            # Update learning
            if self.learning_engine:
                await self.learning_engine.update_from_trade(outcome)
            
            # Track per-symbol statistics for smarter future decisions
            symbol = position['symbol']
            if pnl > 0:
                await self.redis_client.hincrby(f"symbol:stats:{symbol}", 'wins', 1)
                await self.redis_client.hincrbyfloat(f"symbol:stats:{symbol}", 'total_profit', pnl)
            else:
                await self.redis_client.hincrby(f"symbol:stats:{symbol}", 'losses', 1)
                await self.redis_client.hincrbyfloat(f"symbol:stats:{symbol}", 'total_loss', abs(pnl))
            await self.redis_client.hincrby(f"symbol:stats:{symbol}", 'total_trades', 1)
                
            # Store completed trade
            await self.redis_client.lpush(
                f"trades:completed:{user_id}",
                json.dumps({
                    **original_trade,
                    'pnl': pnl,
                    'close_reason': reason,
                    'closed_at': datetime.utcnow().isoformat(),
                })
            )
            await self.redis_client.ltrim(f"trades:completed:{user_id}", 0, 999)
            
    async def _log_trading_status(self):
        """Log trading status"""
        active_users = len(self.user_clients)
        logger.info(f"SMART AI Status: {active_users} users connected")
        
        await self.redis_client.hset(
            'trading:status',
            mapping={
                'active_users': str(active_users),
                'is_running': '1' if self.is_running else '0',
                'mode': 'SMART_AI',
                'last_update': datetime.utcnow().isoformat(),
            }
        )


# Global instance
autonomous_trader = AutonomousTrader()
