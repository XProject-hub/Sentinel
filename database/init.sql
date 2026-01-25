-- ============================================
-- SENTINEL AI - PostgreSQL Schema
-- ============================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================
-- USERS & AUTHENTICATION
-- ============================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    subscription_tier VARCHAR(50) DEFAULT 'free',
    subscription_expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    email_verified_at TIMESTAMP,
    two_factor_enabled BOOLEAN DEFAULT false,
    two_factor_secret VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_subscription ON users(subscription_tier);

-- ============================================
-- EXCHANGE CONNECTIONS (Encrypted API Keys)
-- ============================================

CREATE TABLE exchange_connections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exchange VARCHAR(50) NOT NULL,
    name VARCHAR(100),
    api_key_encrypted BYTEA NOT NULL,
    api_secret_encrypted BYTEA NOT NULL,
    passphrase_encrypted BYTEA,
    is_testnet BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    permissions JSONB DEFAULT '{"trade": true, "withdraw": false}',
    region VARCHAR(10),  -- For Bybit regional endpoints: EU, NL, TR, KZ, GE, AE
    last_sync_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_exchange_user ON exchange_connections(user_id);

-- ============================================
-- TRADING ACCOUNTS & BALANCES
-- ============================================

CREATE TABLE account_balances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exchange VARCHAR(50) NOT NULL,
    asset VARCHAR(20) NOT NULL,
    free_balance DECIMAL(30, 10) DEFAULT 0,
    locked_balance DECIMAL(30, 10) DEFAULT 0,
    total_usd_value DECIMAL(20, 2) DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, exchange, asset)
);

CREATE INDEX idx_balance_user ON account_balances(user_id);

-- ============================================
-- AI STRATEGIES & CONFIGURATIONS
-- ============================================

CREATE TABLE ai_strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    risk_level VARCHAR(20) DEFAULT 'medium',
    min_confidence DECIMAL(5, 2) DEFAULT 0.60,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO ai_strategies (name, description, strategy_type, parameters, risk_level) VALUES
('Momentum Surge', 'Captures strong directional moves with tight risk management', 'momentum', '{"lookback": 20, "threshold": 2.5}', 'medium'),
('Grid Master', 'Range-bound trading with dynamic grid placement', 'grid', '{"levels": 10, "spacing": 0.5}', 'low'),
('Breakout Hunter', 'Identifies and trades key level breakouts', 'breakout', '{"atr_multiplier": 1.5, "confirmation_bars": 2}', 'high'),
('Mean Reversion', 'Profits from price returning to statistical means', 'mean_reversion', '{"std_dev": 2.0, "lookback": 50}', 'medium'),
('Scalp Pro', 'High-frequency small profit captures', 'scalping', '{"tick_threshold": 3, "max_hold": 60}', 'high');

-- ============================================
-- USER STRATEGY ASSIGNMENTS
-- ============================================

CREATE TABLE user_strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    strategy_id UUID REFERENCES ai_strategies(id),
    allocation_percent DECIMAL(5, 2) DEFAULT 100,
    max_position_size DECIMAL(20, 2),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- TRADES & ORDERS
-- ============================================

CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(30) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(30, 10) NOT NULL,
    price DECIMAL(30, 10),
    filled_quantity DECIMAL(30, 10) DEFAULT 0,
    average_price DECIMAL(30, 10),
    status VARCHAR(20) DEFAULT 'pending',
    strategy_id UUID REFERENCES ai_strategies(id),
    ai_confidence DECIMAL(5, 2),
    ai_reasoning TEXT,
    exchange_order_id VARCHAR(100),
    fee DECIMAL(20, 10) DEFAULT 0,
    fee_asset VARCHAR(20),
    pnl DECIMAL(20, 2),
    pnl_percent DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP,
    closed_at TIMESTAMP
);

CREATE INDEX idx_trades_user ON trades(user_id);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_created ON trades(created_at DESC);

-- ============================================
-- ACTIVE POSITIONS
-- ============================================

CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(30) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(30, 10) NOT NULL,
    current_price DECIMAL(30, 10),
    quantity DECIMAL(30, 10) NOT NULL,
    leverage DECIMAL(5, 2) DEFAULT 1,
    unrealized_pnl DECIMAL(20, 2) DEFAULT 0,
    unrealized_pnl_percent DECIMAL(10, 4) DEFAULT 0,
    stop_loss DECIMAL(30, 10),
    take_profit DECIMAL(30, 10),
    strategy_id UUID REFERENCES ai_strategies(id),
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, exchange, symbol)
);

CREATE INDEX idx_positions_user ON positions(user_id);

-- ============================================
-- RISK MANAGEMENT
-- ============================================

CREATE TABLE risk_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    max_loss_per_trade DECIMAL(5, 2) DEFAULT 2.00,
    max_loss_per_day DECIMAL(5, 2) DEFAULT 5.00,
    max_exposure_percent DECIMAL(5, 2) DEFAULT 30.00,
    max_positions INTEGER DEFAULT 5,
    cooldown_after_loss_minutes INTEGER DEFAULT 30,
    emergency_stop_enabled BOOLEAN DEFAULT true,
    emergency_stop_loss_percent DECIMAL(5, 2) DEFAULT 10.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT,
    metadata JSONB,
    resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_risk_events_user ON risk_events(user_id);
CREATE INDEX idx_risk_events_type ON risk_events(event_type);

-- ============================================
-- MARKET REGIMES
-- ============================================

CREATE TABLE market_regimes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(30) NOT NULL,
    regime VARCHAR(30) NOT NULL,
    confidence DECIMAL(5, 2) NOT NULL,
    volatility_level VARCHAR(20),
    trend_strength DECIMAL(5, 2),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_regime_symbol ON market_regimes(symbol);
CREATE INDEX idx_regime_detected ON market_regimes(detected_at DESC);

-- ============================================
-- NEWS & SENTIMENT
-- ============================================

CREATE TABLE news_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source VARCHAR(100) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    url VARCHAR(500),
    published_at TIMESTAMP NOT NULL,
    sentiment_score DECIMAL(5, 4),
    impact_level VARCHAR(20),
    related_assets TEXT[],
    processed BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_published ON news_items(published_at DESC);
CREATE INDEX idx_news_sentiment ON news_items(sentiment_score);

CREATE TABLE sentiment_aggregates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset VARCHAR(20) NOT NULL,
    timeframe VARCHAR(20) NOT NULL,
    sentiment_score DECIMAL(5, 4) NOT NULL,
    news_count INTEGER DEFAULT 0,
    social_score DECIMAL(5, 4),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset, timeframe, calculated_at)
);

-- ============================================
-- AI LEARNING & STATISTICS
-- ============================================

CREATE TABLE trading_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    period VARCHAR(20) NOT NULL,
    period_start TIMESTAMP NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),
    total_pnl DECIMAL(20, 2) DEFAULT 0,
    total_pnl_percent DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    best_trade_pnl DECIMAL(20, 2),
    worst_trade_pnl DECIMAL(20, 2),
    avg_trade_duration_minutes INTEGER,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_stats_user ON trading_statistics(user_id);
CREATE INDEX idx_stats_period ON trading_statistics(period, period_start);

CREATE TABLE ai_learning_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    learning_type VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL,
    improvement_percent DECIMAL(10, 4),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- SUBSCRIPTIONS & BILLING
-- ============================================

CREATE TABLE subscription_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) NOT NULL UNIQUE,
    price_monthly DECIMAL(10, 2) NOT NULL,
    price_yearly DECIMAL(10, 2),
    features JSONB NOT NULL,
    max_exchanges INTEGER DEFAULT 1,
    max_positions INTEGER DEFAULT 3,
    ai_strategies_enabled BOOLEAN DEFAULT true,
    priority_support BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO subscription_plans (name, price_monthly, price_yearly, features, max_exchanges, max_positions, ai_strategies_enabled, priority_support) VALUES
('Starter', 49.00, 470.00, '{"basic_ai": true, "daily_reports": true}', 1, 3, true, false),
('Professional', 149.00, 1430.00, '{"advanced_ai": true, "real_time_alerts": true, "custom_risk": true}', 3, 10, true, false),
('Enterprise', 499.00, 4790.00, '{"full_ai": true, "dedicated_support": true, "api_access": true, "white_label": true}', 10, 50, true, true);

CREATE TABLE payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    plan_id UUID REFERENCES subscription_plans(id),
    amount DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    payment_method VARCHAR(50),
    payment_provider VARCHAR(50),
    provider_transaction_id VARCHAR(255),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_payments_user ON payments(user_id);

-- ============================================
-- AUDIT LOGS
-- ============================================

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_created ON audit_logs(created_at DESC);

-- ============================================
-- NOTIFICATIONS
-- ============================================

CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT,
    severity VARCHAR(20) DEFAULT 'info',
    read BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_notifications_user ON notifications(user_id);
CREATE INDEX idx_notifications_read ON notifications(user_id, read);

