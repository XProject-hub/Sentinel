# 🛡️ SENTINEL AI - Autonomous Crypto & TradFi Trading Platform

<p align="center">
  <img src="https://img.shields.io/badge/AI-Powered-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Trading-24%2F7-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-6%2B-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-Private-red?style=for-the-badge" />
</p>

> **Professional-grade autonomous trading system** powered by 6+ AI/ML models, designed for 24/7 operation across crypto and traditional finance markets.

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [AI Models](#-ai-models)
4. [Trading Strategy](#-trading-strategy)
5. [Risk Management](#-risk-management)
6. [Features](#-features)
7. [Tech Stack](#-tech-stack)
8. [Installation](#-installation)
9. [Configuration](#-configuration)
10. [API Reference](#-api-reference)
11. [Dashboard](#-dashboard)
12. [Roadmap](#-roadmap)

---

## 🎯 Overview

Sentinel is an **autonomous AI trading bot** that:

- ✅ Trades **24/7** without human intervention
- ✅ Uses **6+ AI/ML models** for decision making
- ✅ Supports **Crypto** (500+ pairs) and **TradFi** (indices, commodities)
- ✅ Learns from every trade via **Reinforcement Learning**
- ✅ Manages risk with **Kelly Criterion** position sizing
- ✅ Provides real-time **dashboard** with trade notifications

### Key Differentiators

| Feature | Sentinel | Typical Bots |
|---------|----------|--------------|
| AI Models | 6+ ensemble | 1-2 indicators |
| Learning | Continuous RL | Static rules |
| Markets | Crypto + TradFi | Single market |
| Risk Management | Kelly + Multi-layer | Fixed % |
| Sentiment | CryptoBERT (Hugging Face) | Basic NLP |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SENTINEL PLATFORM                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Frontend  │    │  AI Services │    │   Backend   │              │
│  │  (Next.js)  │◄──►│  (FastAPI)   │◄──►│  (Laravel)  │              │
│  └─────────────┘    └──────┬──────┘    └─────────────┘              │
│         │                  │                  │                      │
│         │                  ▼                  │                      │
│         │    ┌─────────────────────────┐     │                      │
│         │    │      AI MODEL LAYER      │     │                      │
│         │    │  ┌─────┐ ┌─────┐ ┌─────┐│     │                      │
│         │    │  │XGBst│ │BERT │ │Q-Lrn││     │                      │
│         │    │  └─────┘ └─────┘ └─────┘│     │                      │
│         │    │  ┌─────┐ ┌─────┐ ┌─────┐│     │                      │
│         │    │  │Price│ │Regim│ │Edge ││     │                      │
│         │    │  │Pred │ │Detct│ │Estim││     │                      │
│         │    │  └─────┘ └─────┘ └─────┘│     │                      │
│         │    └─────────────────────────┘     │                      │
│         │                  │                  │                      │
│         ▼                  ▼                  ▼                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                    DATA LAYER                             │       │
│  │  ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────────┐   │       │
│  │  │ Redis  │  │Postgres│  │ClickHouse│  │   Kafka    │   │       │
│  │  │ Cache  │  │   DB   │  │   OLAP   │  │  Streams   │   │       │
│  │  └────────┘  └────────┘  └──────────┘  └────────────┘   │       │
│  └──────────────────────────────────────────────────────────┘       │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                    EXCHANGE LAYER                         │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │       │
│  │  │  Bybit V5   │  │   Binance   │  │  Future: More   │   │       │
│  │  │  (Primary)  │  │  (Planned)  │  │    Exchanges    │   │       │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Service Breakdown

| Service | Port | Purpose |
|---------|------|---------|
| `ai-services` | 8000 | AI/ML models, trading logic |
| `frontend` | 3000 | Next.js dashboard |
| `backend` | 9000 | Laravel API (auth, users) |
| `nginx` | 80/443 | Reverse proxy, SSL |
| `redis` | 6379 | Cache, real-time data |
| `postgres` | 5432 | User data, trade history |
| `clickhouse` | 8123 | Market data analytics |

---

## 🧠 AI Models

### 1. XGBoost Edge Classifier
```
Purpose: Fast signal/no-signal classification
Input: Market features (50+ indicators)
Output: BUY / SELL / HOLD + confidence %
Training: Every 6-12 hours on quality trades
```

### 2. CryptoBERT Sentiment (Hugging Face)
```
Model: ElKulako/cryptobert
Purpose: Crypto-specific sentiment analysis
Features:
  - Understands crypto slang (moon, rekt, hodl)
  - Trained on Twitter, Reddit, Discord
  - Better than FinBERT for crypto
Input: News text, social media
Output: Bullish / Bearish / Neutral + score
```

### 3. Q-Learning Engine (Reinforcement Learning)
```
Purpose: Strategy optimization through experience
State: Market regime + indicators + position
Actions: BUY, SELL, HOLD, SCALE_IN, SCALE_OUT
Reward: PnL - fees - slippage - drawdown penalty
Learning: Continuous with exploration decay
```

### 4. Price Predictor (Ensemble)
```
Purpose: Multi-timeframe price prediction
Components:
  - Momentum Analysis (5, 15, 60, 240 periods)
  - RSI (14-period)
  - MACD Signal
  - Bollinger Band Position
  - Trend Strength (MA crossover)
  - Volume Trend
Output: 
  - prob_up_5m: 0.62 (62% chance up in 5min)
  - prob_up_15m: 0.58
  - prob_up_1h: 0.71
  - prob_up_4h: 0.65
```

### 5. Regime Detector (HMM + XGBoost)
```
Purpose: Identify market conditions
Regimes:
  - HIGH_LIQUIDITY_TREND (best for trading)
  - RANGING (scalping opportunities)
  - HIGH_VOLATILITY (reduce size)
  - ACCUMULATION (wait for breakout)
  - DISTRIBUTION (caution)
  - NEWS_SPIKE (avoid)
Output: Current regime + recommended action
```

### 6. Edge Estimator
```
Purpose: Calculate statistical edge for trades
Metrics:
  - Win probability
  - Risk/Reward ratio
  - Kelly fraction
  - Expected value
Output: Edge score 0.0 - 1.0
```

### Model Ensemble Decision Flow

```
Market Data
    │
    ▼
┌───────────────────────────────────────────────┐
│              PARALLEL ANALYSIS                 │
├───────────────────────────────────────────────┤
│  XGBoost ──────► Signal: BUY (78%)            │
│  CryptoBERT ───► Sentiment: Bullish (0.65)    │
│  Price Pred ───► Prob Up 1h: 71%              │
│  Regime ───────► HIGH_LIQUIDITY_TREND         │
│  Edge ─────────► Score: 0.42                  │
│  Q-Learning ───► Action: BUY (Q=0.85)         │
└───────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│           VALIDATION PIPELINE                  │
├───────────────────────────────────────────────┤
│  ✅ Edge > 0.15                               │
│  ✅ Confidence > 55%                          │
│  ✅ XGBoost agrees                            │
│  ✅ CryptoBERT agrees                         │
│  ✅ Price predictor confirms                  │
│  ✅ Regime != AVOID                           │
│  ✅ Position size within limits               │
│  ✅ Risk checks passed                        │
└───────────────────────────────────────────────┘
    │
    ▼
EXECUTE TRADE ───► Bybit API
```

---

## 📈 Trading Strategy

### Core Philosophy

```
"Profit doesn't come from how much the bot TRADES,
 but from how well it knows WHEN NOT TO TRADE."
```

### Strategy Components

#### 1. Market Scanning
- Scans **500+ pairs** on Bybit
- Filters by volume, volatility, liquidity
- Ranks opportunities by edge score

#### 2. Signal Generation
- Multi-model consensus required
- Minimum edge threshold: 0.15
- Minimum confidence: 55%

#### 3. Position Sizing (Kelly Criterion)
```python
Kelly% = (p * b - q) / b

Where:
  p = win probability (from models)
  q = 1 - p
  b = win/loss ratio (from edge estimator)

# Conservative: Use 25% of Kelly
actual_size = kelly_pct * 0.25
```

#### 4. Entry Execution
- Market orders for speed
- Size adjusted for liquidity
- Slippage monitoring

#### 5. Position Management
- Trailing stop loss (tracks peak profit)
- Dynamic take profit (based on regime)
- Emergency stop loss (hard limit)

#### 6. Exit Strategy
```
IF profit > min_profit_to_trail:
    Activate trailing stop
    
IF price drops X% from peak:
    Close position
    
IF emergency_stop_loss hit:
    Immediate close
```

### Risk Presets

| Mode | Take Profit | Stop Loss | Max Position | Max Open |
|------|-------------|-----------|--------------|----------|
| **SAFE** | 1.0% | 0.5% | 5% | 5 |
| **NEUTRAL** | 3.0% | 1.5% | 10% | Unlimited |
| **AGGRESSIVE** | 8.0% | 3.0% | 20% | Unlimited |

---

## 🛡️ Risk Management

### Multi-Layer Protection

```
Layer 1: PRE-TRADE
├── Edge minimum check
├── Confidence threshold
├── Regime filter
├── Sentiment alignment
└── Position size limits

Layer 2: POSITION
├── Trailing stop loss
├── Emergency stop loss
├── Max position % of portfolio
└── Max open positions

Layer 3: PORTFOLIO
├── Max daily drawdown (1-5%)
├── Max total exposure (10-50%)
├── Correlation limits
└── Asset class limits

Layer 4: SYSTEM
├── API error handling
├── Network failure recovery
├── Auto-reconnection
└── State persistence
```

### Daily Drawdown Protection
```python
if daily_loss > max_daily_drawdown:
    STOP_ALL_TRADING
    WAIT_FOR_NEXT_DAY
```

### Quality Filter for Learning
```
ONLY learn from quality trades:
├── PnL > 0.3% (profitable)
├── Edge > 0.15 (had statistical advantage)
├── Confidence > 55%
└── Not a duplicate market context

BAD trades → Rejected from training
```

---

## ✨ Features

### Trading Features
- ✅ 24/7 autonomous trading
- ✅ Multi-exchange support (Bybit primary)
- ✅ Crypto + TradFi (indices, commodities)
- ✅ Long and Short positions
- ✅ Dynamic position sizing
- ✅ Trailing stop loss
- ✅ Emergency stop loss

### AI Features
- ✅ 6+ AI models working together
- ✅ Continuous learning from trades
- ✅ Quality-filtered training data
- ✅ Multi-timeframe analysis
- ✅ Sentiment analysis (CryptoBERT)
- ✅ Market regime detection
- ✅ Price prediction ensemble

### Dashboard Features
- ✅ Real-time portfolio view
- ✅ Live trade notifications
- ✅ PnL tracking
- ✅ Open positions monitor
- ✅ Settings configuration
- ✅ Manual position close
- ✅ Emergency SELL ALL button

### Infrastructure
- ✅ Docker containerized
- ✅ Multi-server support (load balancing)
- ✅ Auto-reconnection
- ✅ Persistent state (Redis)
- ✅ Trade history (PostgreSQL)
- ✅ Market data lake (ClickHouse)

---

## 🔧 Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | AI services |
| FastAPI | 0.109+ | REST API |
| PyTorch | 2.1+ | Deep learning |
| XGBoost | 2.0+ | Classification |
| Transformers | 4.36+ | Hugging Face models |
| Redis | 5.0+ | Caching, state |
| PostgreSQL | 15+ | Database |
| ClickHouse | Latest | Analytics |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 14 | React framework |
| TypeScript | 5+ | Type safety |
| TailwindCSS | 3+ | Styling |
| Recharts | Latest | Charts |
| Lucide | Latest | Icons |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| Docker | Containerization |
| Docker Compose | Orchestration |
| Nginx | Reverse proxy |
| Certbot | SSL certificates |
| GitHub | Version control |

---

## 📥 Installation

### Prerequisites
- Ubuntu 22.04 LTS (recommended)
- Docker & Docker Compose
- 8GB+ RAM
- 4+ CPU cores

### Quick Start

```bash
# Clone repository
git clone https://github.com/XProject-hub/Sentinel.git
cd Sentinel

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Build and start
docker compose build
docker compose up -d

# Check logs
docker logs sentinel_ai -f
```

### First-Time Setup

1. Access dashboard: `https://your-domain.com`
2. Go to Settings → Connect Exchange
3. Enter Bybit API credentials
4. Configure risk settings
5. Start the bot

---

## ⚙️ Configuration

### Environment Variables

```env
# Database
POSTGRES_HOST=postgres
POSTGRES_DB=sentinel
POSTGRES_USER=sentinel
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://redis:6379

# Exchange (encrypted in Redis)
# Set via dashboard, not env

# AI Settings
USE_V2_TRADER=true
MIN_TRADE_VALUE_USDT=5.5
```

### Bot Settings (via Dashboard)

| Setting | Description | Default |
|---------|-------------|---------|
| Risk Mode | SAFE/NEUTRAL/AGGRESSIVE | NEUTRAL |
| Take Profit % | Target profit | 3.0% |
| Stop Loss % | Maximum loss | 1.5% |
| Trailing Stop % | Trail from peak | 1.2% |
| Min Confidence | AI confidence threshold | 60% |
| Max Position % | Max % per trade | 10% |
| Max Open Positions | Position limit (0=unlimited) | 0 |

---

## 📡 API Reference

### Health Check
```bash
GET /ai/health
```

### Exchange
```bash
POST /ai/exchange/connect    # Connect API keys
GET  /ai/exchange/wallet     # Get wallet balance
GET  /ai/exchange/positions  # Get open positions
POST /ai/exchange/close-position/{symbol}  # Close position
```

### Trading
```bash
POST /ai/trader/start        # Start trading
POST /ai/trader/stop         # Stop trading
GET  /ai/trader/status       # Get trader status
POST /ai/trader/sell-all     # Emergency close all
```

### AI Models
```bash
GET /ai/crypto-sentiment/market           # Market sentiment
GET /ai/crypto-sentiment/symbol/{symbol}  # Symbol sentiment
GET /ai/price-predictor/predict/{symbol}  # Price prediction
GET /ai/price-predictor/signal/{symbol}   # Trading signal
GET /ai/capital-allocator/status          # Allocation status
GET /ai/models/summary                    # All models summary
```

### Training & Learning
```bash
GET /ai/training/stats        # Training statistics
GET /ai/training/leaderboard  # Multi-user leaderboard
GET /ai/learning/stats        # Q-Learning statistics
```

---

## 📊 Dashboard

### Main Dashboard
- Total equity display
- Today's PnL
- Open positions count
- Win rate statistics
- AI status indicators
- Live trade notifications

### Settings Page
- Risk mode selection
- Trading parameters
- AI feature toggles
- Budget configuration
- Exchange connection

### Admin Panel
- System metrics (CPU, RAM, Disk)
- AI model statistics
- Trade history
- Learning progress

---

## 🗺️ Roadmap

### ✅ Phase 1: Foundation (Complete)
- [x] Basic trading infrastructure
- [x] Bybit V5 integration
- [x] Dashboard UI
- [x] Position management

### ✅ Phase 2: AI Integration (Complete)
- [x] XGBoost classifier
- [x] Q-Learning engine
- [x] Sentiment analysis (FinBERT)
- [x] Regime detection

### ✅ Phase 3: Superior AI (Complete)
- [x] CryptoBERT (Hugging Face)
- [x] Price predictor ensemble
- [x] Capital allocator
- [x] TradFi support
- [x] Quality-filtered training
- [x] Multi-user learning

### 🔄 Phase 4: Advanced (In Progress)
- [ ] Temporal Fusion Transformer (TFT)
- [ ] PPO/SAC reinforcement learning
- [ ] Multi-exchange arbitrage
- [ ] Social media integration
- [ ] Telegram notifications

### 📋 Phase 5: Scale (Planned)
- [ ] GPU acceleration
- [ ] Distributed training
- [ ] White-label solution
- [ ] Mobile app

---

## 📈 Performance Metrics

### Target Metrics
| Metric | Target | Description |
|--------|--------|-------------|
| Daily Win Rate | >65% | Winning trades per day |
| Risk/Reward | >1.5:1 | Average win vs loss size |
| Max Drawdown | <5% | Maximum daily loss |
| Sharpe Ratio | >1.5 | Risk-adjusted returns |

### Model Accuracy Targets
| Model | Target Accuracy |
|-------|-----------------|
| XGBoost | >65% |
| Price Predictor 5m | >55% |
| Price Predictor 1h | >58% |
| Regime Detection | >70% |

---

## 🔐 Security

### API Key Protection
- Keys encrypted with AES-256
- Stored in Redis (not files)
- Never logged or exposed

### Access Control
- JWT authentication
- Rate limiting
- IP whitelisting (optional)

### Best Practices
- Use testnet first
- Start with small capital
- Monitor regularly
- Set conservative limits

---

## ⚠️ Disclaimer

**This software is for educational purposes only.**

Trading cryptocurrencies and financial instruments involves substantial risk of loss. Past performance does not guarantee future results. The developers are not responsible for any financial losses incurred while using this software.

**Always:**
- Trade only what you can afford to lose
- Test thoroughly on testnet first
- Start with small amounts
- Monitor your positions
- Understand the risks involved

---

## 📞 Support

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: This README
- **Logs**: `docker logs sentinel_ai -f`

---

## 📄 License

Private / Proprietary - All rights reserved.

---

<p align="center">
  <b>Built with 🧠 AI and ❤️ by Sentinel Team</b>
</p>
