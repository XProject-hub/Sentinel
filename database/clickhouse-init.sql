-- ============================================
-- SENTINEL AI - ClickHouse Schema
-- Time-Series Analytics
-- ============================================

CREATE DATABASE IF NOT EXISTS sentinel_analytics;

-- ============================================
-- PRICE DATA (High-frequency)
-- ============================================

CREATE TABLE sentinel_analytics.price_ticks (
    timestamp DateTime64(3),
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    price Decimal(30, 10),
    volume Decimal(30, 10),
    side LowCardinality(String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
TTL timestamp + INTERVAL 90 DAY;

-- ============================================
-- ORDER BOOK SNAPSHOTS
-- ============================================

CREATE TABLE sentinel_analytics.orderbook_snapshots (
    timestamp DateTime64(3),
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    bids Array(Tuple(price Decimal(30, 10), quantity Decimal(30, 10))),
    asks Array(Tuple(price Decimal(30, 10), quantity Decimal(30, 10))),
    spread Decimal(20, 10),
    mid_price Decimal(30, 10)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
TTL timestamp + INTERVAL 30 DAY;

-- ============================================
-- OHLCV CANDLES
-- ============================================

CREATE TABLE sentinel_analytics.ohlcv (
    timestamp DateTime,
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    timeframe LowCardinality(String),
    open Decimal(30, 10),
    high Decimal(30, 10),
    low Decimal(30, 10),
    close Decimal(30, 10),
    volume Decimal(30, 10),
    trades_count UInt32
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), timeframe)
ORDER BY (symbol, timeframe, timestamp);

-- ============================================
-- FUNDING RATES
-- ============================================

CREATE TABLE sentinel_analytics.funding_rates (
    timestamp DateTime,
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    funding_rate Decimal(20, 10),
    predicted_rate Decimal(20, 10),
    next_funding_time DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp);

-- ============================================
-- LIQUIDATIONS
-- ============================================

CREATE TABLE sentinel_analytics.liquidations (
    timestamp DateTime64(3),
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    side LowCardinality(String),
    quantity Decimal(30, 10),
    price Decimal(30, 10),
    usd_value Decimal(20, 2)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp);

-- ============================================
-- OPEN INTEREST
-- ============================================

CREATE TABLE sentinel_analytics.open_interest (
    timestamp DateTime,
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    open_interest Decimal(30, 10),
    open_interest_usd Decimal(20, 2)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp);

-- ============================================
-- SENTIMENT SCORES (Time-series)
-- ============================================

CREATE TABLE sentinel_analytics.sentiment_history (
    timestamp DateTime,
    asset LowCardinality(String),
    sentiment_score Decimal(5, 4),
    fear_greed_index UInt8,
    social_volume UInt32,
    news_volume UInt32,
    source LowCardinality(String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (asset, timestamp);

-- ============================================
-- AI PREDICTIONS
-- ============================================

CREATE TABLE sentinel_analytics.ai_predictions (
    timestamp DateTime,
    symbol LowCardinality(String),
    model_name LowCardinality(String),
    prediction_type LowCardinality(String),
    predicted_value Decimal(30, 10),
    confidence Decimal(5, 4),
    timeframe LowCardinality(String),
    actual_value Decimal(30, 10),
    error Decimal(20, 10)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, model_name, timestamp);

-- ============================================
-- TRADE EXECUTION METRICS
-- ============================================

CREATE TABLE sentinel_analytics.execution_metrics (
    timestamp DateTime64(3),
    trade_id UUID,
    user_id UUID,
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    intended_price Decimal(30, 10),
    executed_price Decimal(30, 10),
    slippage Decimal(10, 6),
    latency_ms UInt32,
    fee Decimal(20, 10)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (user_id, timestamp);

-- ============================================
-- MATERIALIZED VIEWS FOR AGGREGATIONS
-- ============================================

-- Hourly OHLCV from ticks
CREATE MATERIALIZED VIEW sentinel_analytics.ohlcv_1h_mv
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
AS SELECT
    toStartOfHour(timestamp) as timestamp,
    symbol,
    exchange,
    '1h' as timeframe,
    argMin(price, timestamp) as open,
    max(price) as high,
    min(price) as low,
    argMax(price, timestamp) as close,
    sum(volume) as volume,
    count() as trades_count
FROM sentinel_analytics.price_ticks
GROUP BY symbol, exchange, toStartOfHour(timestamp);

-- Daily aggregated sentiment
CREATE MATERIALIZED VIEW sentinel_analytics.sentiment_daily_mv
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (asset, timestamp)
AS SELECT
    toStartOfDay(timestamp) as timestamp,
    asset,
    avg(sentiment_score) as avg_sentiment,
    min(sentiment_score) as min_sentiment,
    max(sentiment_score) as max_sentiment,
    sum(news_volume) as total_news,
    sum(social_volume) as total_social
FROM sentinel_analytics.sentiment_history
GROUP BY asset, toStartOfDay(timestamp);

