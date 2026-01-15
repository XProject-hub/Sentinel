#!/bin/bash
# ============================================
# SENTINEL AI - Auto-generate .env file
# ============================================

cd /opt/sentinel

echo ""
echo "============================================"
echo "   Generating secure .env configuration"
echo "============================================"
echo ""

# Generate secure passwords
DB_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
APP_KEY=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 64 | tr -dc 'a-zA-Z0-9' | head -c 64)
ENCRYPTION_KEY=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)

# Create .env file
cat > .env << EOF
# ============================================
# SENTINEL AI - Environment Configuration
# Generated: $(date)
# ============================================

# Database (AUTO-GENERATED - DO NOT CHANGE)
DB_PASSWORD=$DB_PASSWORD

# Application Keys (AUTO-GENERATED)
APP_KEY=base64:$APP_KEY
JWT_SECRET=$JWT_SECRET
ENCRYPTION_KEY=$ENCRYPTION_KEY

# Domain
DOMAIN=sentinel.xproject.live
SERVER_IP=109.104.154.183

# ============================================
# OPTIONAL - Add later when needed
# ============================================

# Stripe (Payments) - Add when you want subscriptions
STRIPE_KEY=
STRIPE_SECRET=
STRIPE_WEBHOOK_SECRET=

# Binance API (for live market data) - Add for real trading
BINANCE_API_KEY=
BINANCE_API_SECRET=

# News APIs (optional - for sentiment analysis)
CRYPTOPANIC_API_KEY=
NEWSAPI_KEY=

# Email (optional)
MAIL_HOST=
MAIL_PORT=587
MAIL_USERNAME=
MAIL_PASSWORD=
MAIL_FROM_ADDRESS=noreply@sentinel.xproject.live
MAIL_FROM_NAME="SENTINEL AI"
EOF

echo "Generated .env with secure credentials!"
echo ""
echo "Database Password: $DB_PASSWORD"
echo ""
echo "Optional APIs can be added later by editing:"
echo "  nano /opt/sentinel/.env"
echo ""
echo "============================================"

