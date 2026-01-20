import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 120000) // 2 minute timeout for backtests
    
    const response = await fetch('http://ai-services:8000/ai/backtest/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    
    clearTimeout(timeoutId)

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error: any) {
    console.error('Backtest API error:', error)
    const message = error.name === 'AbortError' 
      ? 'Backtest timed out after 2 minutes' 
      : `Failed to connect to AI services: ${error.message || 'Unknown error'}`
    return NextResponse.json(
      { success: false, detail: message },
      { status: 500 }
    )
  }
}

