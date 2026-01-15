# SENTINEL AI

## Autonomous Market Intelligence & Trading System

```
   _____ ______ _   _ _______ _____ _   _ ______ _      
  / ____|  ____| \ | |__   __|_   _| \ | |  ____| |     
 | (___ | |__  |  \| |  | |    | | |  \| | |__  | |     
  \___ \|  __| | . ` |  | |    | | | . ` |  __| | |     
  ____) | |____| |\  |  | |   _| |_| |\  | |____| |____ 
 |_____/|______|_| \_|  |_|  |_____|_| \_|______|______|
```

---

## Overview

SENTINEL AI is a fully autonomous trading system that:
- Monitors markets 24/7
- Analyzes news sentiment
- Detects market regimes
- Selects optimal strategies
- Executes trades automatically
- Protects your capital

**User does nothing. AI does everything.**

---

## Architecture

```
Frontend (Next.js 14)
         │
         ▼
   NGINX (Reverse Proxy + SSL)
         │
    ┌────┴────┐
    ▼         ▼
Laravel 11   FastAPI
(Auth/API)   (AI Services)
    │         │
    └────┬────┘
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
PostgreSQL  Redis   ClickHouse
```

---

## Technology Stack

### Backend
- **Laravel 11** - Auth, Billing, API Gateway
- **FastAPI** - AI/ML Services
- **PostgreSQL** - Primary database
- **Redis** - Cache, Queue, Real-time
- **ClickHouse** - Time-series analytics
- **Kafka** - Event streaming

### AI/ML
- **PyTorch** - Deep learning
- **Transformers** - NLP/Sentiment
- **CCXT** - Exchange connectivity
- **Technical Analysis** - Indicators

### Frontend
- **Next.js 14** - React framework
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **WebSockets** - Real-time updates

---

## Deployment

### Requirements
- Ubuntu 22.04 LTS
- 4GB+ RAM
- Docker & Docker Compose

### Quick Start

```bash
# 1. SSH to server
ssh root@109.104.154.183

# 2. Run setup script
bash setup-server.sh

# 3. Copy project files
scp -r ./* root@109.104.154.183:/opt/sentinel/

# 4. Start services
cd /opt/sentinel
docker compose up -d

# 5. Get SSL certificate
certbot --nginx -d sentinel.xproject.live
```

### Access

- **Dashboard**: https://sentinel.xproject.live
- **API**: https://sentinel.xproject.live/api
- **AI Services**: https://sentinel.xproject.live/ai

---

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Database
DB_PASSWORD=secure_password

# Payments
STRIPE_KEY=pk_live_xxx
STRIPE_SECRET=sk_live_xxx

# Exchange (for market data)
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx

# News APIs
CRYPTOPANIC_API_KEY=xxx
NEWSAPI_KEY=xxx
```

---

## AI Services

### Market Intelligence
Real-time collection of:
- Price ticks
- Order book depth
- Volume analysis
- Volatility metrics
- Funding rates
- Liquidation data

### Sentiment Analysis
- Crypto news monitoring
- NLP-based sentiment scoring
- Impact prediction
- Whale alert tracking

### Strategy Planning
- Market regime detection
- Automatic strategy selection
- Entry/exit optimization
- Position sizing

### Risk Management
- Max loss per trade
- Max loss per day
- Exposure limits
- Emergency stop
- Cooldown periods

---

## API Endpoints

### Authentication
```
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me
```

### Dashboard
```
GET /api/dashboard
GET /api/dashboard/summary
GET /api/dashboard/performance
GET /api/dashboard/ai-status
GET /api/dashboard/risk-status
```

### Trading
```
GET  /api/trades
GET  /api/positions
POST /api/positions/{id}/close
```

### AI Services
```
GET /ai/health
GET /ai/regime/{symbol}
GET /ai/market/data/{symbol}
GET /ai/sentiment/current
```

---

## Security

- All API keys encrypted at rest
- JWT authentication with short TTL
- 2FA support
- Rate limiting
- Audit logging
- No withdrawal permissions required

---

## Subscription Tiers

| Feature | Starter | Professional | Enterprise |
|---------|---------|--------------|------------|
| Price/mo | $49 | $149 | $499 |
| Exchanges | 1 | 3 | 10 |
| Positions | 3 | 10 | 50 |
| AI Strategies | Basic | Advanced | Full |
| Support | Email | Priority | Dedicated |

---

## Support

- **Domain**: sentinel.xproject.live
- **Server**: 109.104.154.183

---

## License

Proprietary - All rights reserved.

---

**SENTINEL AI - Your Autonomous Digital Trader**

