"""Sentiment Analysis API Routes"""

from fastapi import APIRouter, HTTPException
from typing import Optional

router = APIRouter()


@router.get("/current")
async def get_current_sentiment():
    """Get current sentiment for all tracked assets"""
    from main import sentiment_analyzer
    
    sentiment = await sentiment_analyzer.get_current_sentiment()
    return {"success": True, "sentiment": sentiment}


@router.get("/asset/{asset}")
async def get_asset_sentiment(asset: str):
    """Get sentiment for a specific asset"""
    from main import sentiment_analyzer
    
    sentiment = await sentiment_analyzer.get_asset_sentiment(asset)
    return {"success": True, "sentiment": sentiment}


@router.get("/news")
async def get_recent_news(limit: int = 10):
    """Get recent analyzed news"""
    from main import sentiment_analyzer
    
    news = await sentiment_analyzer.get_recent_news(limit)
    return {"success": True, "news": news, "count": len(news)}

