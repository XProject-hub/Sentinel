"""
SENTINEL AI - Configuration
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql://sentinel:password@localhost:5432/sentinel"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # ClickHouse
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 9000
    CLICKHOUSE_USER: str = "sentinel"
    CLICKHOUSE_PASSWORD: str = ""
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    
    # Exchange APIs (for market data)
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    
    # News & Data APIs
    CRYPTOPANIC_API_KEY: str = ""
    NEWSAPI_KEY: str = ""
    
    # AI Model Settings
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Trading Settings
    DEFAULT_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    MAX_CONCURRENT_TRADES: int = 10
    
    # Risk Settings
    GLOBAL_MAX_DRAWDOWN: float = 15.0  # Percent
    EMERGENCY_STOP_THRESHOLD: float = 10.0  # Percent
    
    # Rate Limits
    MAX_REQUESTS_PER_MINUTE: int = 1200
    MARKET_DATA_INTERVAL_MS: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

