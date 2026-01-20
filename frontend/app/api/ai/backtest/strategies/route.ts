import { NextResponse } from 'next/server'

export async function GET() {
  try {
    const response = await fetch('http://ai-services:8000/ai/backtest/strategies', {
      cache: 'no-store',
    })

    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
  } catch (error) {
    // Fallback strategies
  }

  return NextResponse.json({
    strategies: [
      { id: 'trend_following', name: 'Trend Following', description: 'Buy when price crosses above SMA, sell when crosses below' },
      { id: 'mean_reversion', name: 'Mean Reversion', description: 'Buy oversold (RSI<30), sell overbought (RSI>70)' },
      { id: 'breakout', name: 'Breakout', description: 'Trade when price breaks 20-period range' },
      { id: 'macd_crossover', name: 'MACD Crossover', description: 'Trade MACD line crossovers' }
    ]
  })
}

