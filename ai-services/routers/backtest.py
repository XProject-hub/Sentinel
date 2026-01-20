"""
Backtest API Routes
Run and retrieve backtest results
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from loguru import logger

from services.backtester import backtester, BacktestConfig

router = APIRouter()


class BacktestRequest(BaseModel):
    """Request model for running a backtest"""
    symbol: str = "BTCUSDT"
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None  # YYYY-MM-DD
    initial_capital: float = 1000.0
    
    # Strategy parameters
    take_profit_percent: float = 2.0
    stop_loss_percent: float = 1.0
    trailing_stop_percent: float = 0.5
    min_profit_to_trail: float = 0.5
    
    # Position sizing
    position_size_percent: float = 10.0
    max_open_positions: int = 1
    leverage: int = 1
    
    # Strategy type
    strategy: str = "trend_following"


class BacktestResponse(BaseModel):
    """Response model for backtest results"""
    success: bool
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    
    # Capital
    initial_capital: float
    final_capital: float
    
    # Performance
    total_return: float
    total_pnl: float
    
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    
    # Averages
    avg_win: float
    avg_loss: float
    avg_trade_duration: float
    
    # Recent trades (last 10)
    recent_trades: List[dict]
    
    # Equity curve (sampled for performance)
    equity_curve: List[dict]


@router.post("/run")
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Run a backtest with specified parameters
    
    Available strategies:
    - trend_following: Buy when price crosses above SMA, sell when crosses below
    - mean_reversion: Buy oversold (RSI<30, below BB), sell overbought
    - breakout: Trade when price breaks out of 20-period range
    - macd_crossover: Trade MACD line crossovers
    """
    try:
        # Initialize backtester if needed
        if backtester.redis_client is None:
            await backtester.initialize()
        
        # Create config
        config = BacktestConfig(
            symbol=request.symbol,
            start_date=request.start_date or "",
            end_date=request.end_date or "",
            initial_capital=request.initial_capital,
            take_profit_percent=request.take_profit_percent,
            stop_loss_percent=request.stop_loss_percent,
            trailing_stop_percent=request.trailing_stop_percent,
            min_profit_to_trail=request.min_profit_to_trail,
            position_size_percent=request.position_size_percent,
            max_open_positions=request.max_open_positions,
            leverage=request.leverage,
            strategy=request.strategy
        )
        
        # Run backtest
        result = await backtester.run_backtest(config)
        
        # Format trades for response
        recent_trades = []
        for trade in result.trades[-10:]:
            recent_trades.append({
                'symbol': trade.symbol,
                'direction': trade.direction,
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason
            })
        
        # Sample equity curve (every 10th point for performance)
        sampled_curve = result.equity_curve[::10] if len(result.equity_curve) > 100 else result.equity_curve
        
        return BacktestResponse(
            success=True,
            strategy=result.strategy_name,
            symbol=result.symbol,
            start_date=result.start_date.isoformat(),
            end_date=result.end_date.isoformat(),
            initial_capital=result.initial_capital,
            final_capital=result.final_capital,
            total_return=round(result.total_return, 2),
            total_pnl=round(result.total_pnl, 2),
            total_trades=result.total_trades,
            winning_trades=result.winning_trades,
            losing_trades=result.losing_trades,
            win_rate=round(result.win_rate, 1),
            max_drawdown=round(result.max_drawdown, 2),
            sharpe_ratio=round(result.sharpe_ratio, 2),
            profit_factor=round(result.profit_factor, 2) if result.profit_factor != float('inf') else 999,
            avg_win=round(result.avg_win, 2),
            avg_loss=round(result.avg_loss, 2),
            avg_trade_duration=round(result.avg_trade_duration, 1),
            recent_trades=recent_trades,
            equity_curve=sampled_curve
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.get("/strategies")
async def get_strategies():
    """Get list of available backtest strategies"""
    return {
        "strategies": [
            {
                "id": "trend_following",
                "name": "Trend Following",
                "description": "Buy when price crosses above SMA, sell when crosses below. Best in trending markets."
            },
            {
                "id": "mean_reversion",
                "name": "Mean Reversion",
                "description": "Buy oversold (RSI<30, below Bollinger Band), sell overbought. Best in ranging markets."
            },
            {
                "id": "breakout",
                "name": "Breakout",
                "description": "Trade when price breaks out of 20-period high/low range. Good for volatile markets."
            },
            {
                "id": "macd_crossover",
                "name": "MACD Crossover",
                "description": "Trade MACD line crossovers. Classic momentum strategy."
            }
        ]
    }


@router.get("/history/{symbol}")
async def get_backtest_history(symbol: str, limit: int = 10):
    """Get backtest history for a symbol"""
    try:
        if backtester.redis_client is None:
            await backtester.initialize()
        
        import json
        history = await backtester.redis_client.lrange(f"backtest:history:{symbol}", 0, limit - 1)
        
        results = []
        for item in history:
            results.append(json.loads(item))
        
        return {"symbol": symbol, "history": results}
        
    except Exception as e:
        logger.error(f"Failed to get backtest history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
async def get_available_symbols():
    """Get list of symbols available for backtesting"""
    # Return popular symbols that have good historical data
    return {
        "symbols": [
            {"symbol": "BTCUSDT", "name": "Bitcoin"},
            {"symbol": "ETHUSDT", "name": "Ethereum"},
            {"symbol": "SOLUSDT", "name": "Solana"},
            {"symbol": "XRPUSDT", "name": "XRP"},
            {"symbol": "DOGEUSDT", "name": "Dogecoin"},
            {"symbol": "ADAUSDT", "name": "Cardano"},
            {"symbol": "AVAXUSDT", "name": "Avalanche"},
            {"symbol": "DOTUSDT", "name": "Polkadot"},
            {"symbol": "LINKUSDT", "name": "Chainlink"},
            {"symbol": "MATICUSDT", "name": "Polygon"},
        ]
    }

