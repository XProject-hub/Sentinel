import { NextResponse } from 'next/server'

export async function GET() {
  try {
    const response = await fetch('http://ai-services:8000/ai/backtest/symbols', {
      cache: 'no-store',
    })

    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
  } catch (error) {
    // Fallback symbols
  }

  return NextResponse.json({
    symbols: [
      { symbol: 'BTCUSDT', name: 'Bitcoin' },
      { symbol: 'ETHUSDT', name: 'Ethereum' },
      { symbol: 'SOLUSDT', name: 'Solana' },
      { symbol: 'XRPUSDT', name: 'XRP' },
      { symbol: 'DOGEUSDT', name: 'Dogecoin' },
      { symbol: 'ADAUSDT', name: 'Cardano' },
      { symbol: 'AVAXUSDT', name: 'Avalanche' },
      { symbol: 'DOTUSDT', name: 'Polkadot' },
      { symbol: 'LINKUSDT', name: 'Chainlink' },
      { symbol: 'MATICUSDT', name: 'Polygon' }
    ]
  })
}

