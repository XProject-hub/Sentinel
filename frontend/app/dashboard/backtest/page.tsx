'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  ArrowLeft,
  Play,
  Loader2,
  TrendingUp,
  TrendingDown,
  Target,
  AlertTriangle,
  BarChart3,
  Calendar,
  DollarSign,
  Percent,
  Clock,
  Activity,
  CheckCircle,
  XCircle,
  ChevronDown,
  RefreshCw
} from 'lucide-react'
import Link from 'next/link'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts'

interface BacktestResult {
  success: boolean
  strategy: string
  symbol: string
  start_date: string
  end_date: string
  initial_capital: number
  final_capital: number
  total_return: number
  total_pnl: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
  max_drawdown: number
  sharpe_ratio: number
  profit_factor: number
  avg_win: number
  avg_loss: number
  avg_trade_duration: number
  recent_trades: any[]
  equity_curve: any[]
}

interface Strategy {
  id: string
  name: string
  description: string
}

// Default strategies and symbols - defined outside component for initial state
const initialStrategies = [
  { id: 'trend_following', name: 'Trend Following', description: 'SMA crossover strategy - best in trending markets' },
  { id: 'mean_reversion', name: 'Mean Reversion', description: 'RSI + Bollinger Bands - best in ranging markets' },
  { id: 'breakout', name: 'Breakout', description: '20-period range breakout - good for volatile markets' },
  { id: 'macd_crossover', name: 'MACD Crossover', description: 'MACD signal crossovers - classic momentum strategy' }
]

const initialSymbols = [
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

export default function BacktestPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [strategies, setStrategies] = useState<Strategy[]>(initialStrategies)
  const [symbols, setSymbols] = useState<{symbol: string, name: string}[]>(initialSymbols)
  
  // Form state
  const [config, setConfig] = useState({
    symbol: 'BTCUSDT',
    strategy: 'trend_following',
    start_date: '',
    end_date: '',
    initial_capital: 1000,
    take_profit_percent: 2.0,
    stop_loss_percent: 1.0,
    trailing_stop_percent: 0.5,
    min_profit_to_trail: 0.5,
    position_size_percent: 10,
    max_open_positions: 1,
    leverage: 1
  })

  useEffect(() => {
    loadStrategies()
    loadSymbols()
  }, [])

  const loadStrategies = async () => {
    try {
      const response = await fetch('/api/ai/backtest/strategies')
      if (response.ok) {
        const data = await response.json()
        if (data.strategies && data.strategies.length > 0) {
          setStrategies(data.strategies)
        }
      }
    } catch (e) {
      // Keep using initial strategies
    }
  }

  const loadSymbols = async () => {
    try {
      const response = await fetch('/api/ai/backtest/symbols')
      if (response.ok) {
        const data = await response.json()
        if (data.symbols && data.symbols.length > 0) {
          setSymbols(data.symbols)
        }
      }
    } catch (e) {
      // Keep using initial symbols
    }
  }

  const runBacktest = async () => {
    setIsRunning(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/ai/backtest/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })

      const data = await response.json()

      if (response.ok && data.success) {
        setResult(data)
      } else {
        // Show detailed error
        const errorMsg = data.detail || data.message || data.error || 'Backtest failed'
        setError(errorMsg)
      }
    } catch (e: any) {
      console.error('Backtest error:', e)
      setError(`Connection error: ${e.message || 'Failed to connect to server'}`)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="min-h-screen bg-sentinel-bg-primary p-6 pb-20">
      <div className="w-full max-w-[2000px] mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link 
              href="/dashboard" 
              className="p-2 glass-card rounded-lg hover:bg-sentinel-bg-tertiary transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-sentinel-text-secondary" />
            </Link>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-display font-bold text-xl text-white">SENTINEL</span>
                <span className="text-sentinel-text-muted">|</span>
                <span className="font-display font-bold text-xl">Backtester</span>
              </div>
              <p className="text-sentinel-text-muted text-sm">Test strategies on historical data</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-sentinel-accent-cyan" />
            <span className="text-sentinel-text-secondary">Historical Analysis</span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="glass-card border border-sentinel-border rounded-2xl p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-sentinel-accent-cyan" />
              Configuration
            </h2>

            <div className="space-y-4">
              {/* Symbol */}
              <div>
                <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                  Trading Pair
                </label>
                <select
                  value={config.symbol}
                  onChange={(e) => setConfig({ ...config, symbol: e.target.value })}
                  className="w-full px-4 py-3 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white focus:border-sentinel-accent-cyan focus:outline-none"
                >
                  {symbols.map((s) => (
                    <option key={s.symbol} value={s.symbol}>{s.symbol} - {s.name}</option>
                  ))}
                </select>
              </div>

              {/* Strategy */}
              <div>
                <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                  Strategy
                </label>
                <select
                  value={config.strategy}
                  onChange={(e) => setConfig({ ...config, strategy: e.target.value })}
                  className="w-full px-4 py-3 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white focus:border-sentinel-accent-cyan focus:outline-none"
                >
                  {strategies.map((s) => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
                <p className="text-xs text-sentinel-text-muted mt-1">
                  {strategies.find(s => s.id === config.strategy)?.description}
                </p>
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Start Date
                  </label>
                  <input
                    type="date"
                    value={config.start_date}
                    onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
                    className="w-full px-3 py-2 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white focus:border-sentinel-accent-cyan focus:outline-none text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    End Date
                  </label>
                  <input
                    type="date"
                    value={config.end_date}
                    onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
                    className="w-full px-3 py-2 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white focus:border-sentinel-accent-cyan focus:outline-none text-sm"
                  />
                </div>
              </div>

              {/* Capital */}
              <div>
                <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                  Initial Capital ($)
                </label>
                <input
                  type="number"
                  value={config.initial_capital}
                  onChange={(e) => setConfig({ ...config, initial_capital: parseFloat(e.target.value) || 1000 })}
                  className="w-full px-4 py-3 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white focus:border-sentinel-accent-cyan focus:outline-none"
                />
              </div>

              {/* Risk Settings */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Take Profit %
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={config.take_profit_percent}
                    onChange={(e) => setConfig({ ...config, take_profit_percent: parseFloat(e.target.value) || 2 })}
                    className="w-full px-3 py-2 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white focus:border-sentinel-accent-cyan focus:outline-none text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Stop Loss %
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={config.stop_loss_percent}
                    onChange={(e) => setConfig({ ...config, stop_loss_percent: parseFloat(e.target.value) || 1 })}
                    className="w-full px-3 py-2 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white focus:border-sentinel-accent-cyan focus:outline-none text-sm"
                  />
                </div>
              </div>

              {/* Position Size */}
              <div>
                <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                  Position Size (% of capital)
                </label>
                <input
                  type="number"
                  value={config.position_size_percent}
                  onChange={(e) => setConfig({ ...config, position_size_percent: parseFloat(e.target.value) || 10 })}
                  className="w-full px-4 py-3 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white focus:border-sentinel-accent-cyan focus:outline-none"
                />
              </div>

              {/* Leverage */}
              <div>
                <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                  Leverage
                </label>
                <select
                  value={config.leverage}
                  onChange={(e) => setConfig({ ...config, leverage: parseInt(e.target.value) })}
                  className="w-full px-4 py-3 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white focus:border-sentinel-accent-cyan focus:outline-none"
                >
                  <option value={1}>1x (No leverage)</option>
                  <option value={2}>2x</option>
                  <option value={3}>3x</option>
                  <option value={5}>5x</option>
                  <option value={10}>10x</option>
                </select>
              </div>

              {/* Run Button */}
              <button
                onClick={runBacktest}
                disabled={isRunning}
                className="w-full py-4 bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary rounded-xl font-bold hover:shadow-glow-cyan transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Running Backtest...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Run Backtest
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-6">
            {error && (
              <div className="glass-card border border-sentinel-accent-crimson/30 rounded-2xl p-6">
                <div className="flex items-center gap-3 text-sentinel-accent-crimson">
                  <AlertTriangle className="w-6 h-6" />
                  <span>{error}</span>
                </div>
              </div>
            )}

            {!result && !isRunning && !error && (
              <div className="glass-card border border-sentinel-border rounded-2xl p-12 text-center">
                <BarChart3 className="w-16 h-16 text-sentinel-text-muted mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">No Backtest Results</h3>
                <p className="text-sentinel-text-muted">
                  Configure your strategy and click "Run Backtest" to see results
                </p>
              </div>
            )}

            {isRunning && (
              <div className="glass-card border border-sentinel-border rounded-2xl p-12 text-center">
                <Loader2 className="w-16 h-16 text-sentinel-accent-cyan mx-auto mb-4 animate-spin" />
                <h3 className="text-xl font-semibold text-white mb-2">Running Backtest...</h3>
                <p className="text-sentinel-text-muted">
                  Analyzing historical data for {config.symbol}
                </p>
              </div>
            )}

            {result && (
              <>
                {/* Performance Summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="glass-card border border-sentinel-border rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <DollarSign className="w-4 h-4 text-sentinel-accent-cyan" />
                      <span className="text-sm text-sentinel-text-muted">Total Return</span>
                    </div>
                    <div className={`text-2xl font-bold ${result.total_return >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}`}>
                      {result.total_return >= 0 ? '+' : ''}{result.total_return.toFixed(2)}%
                    </div>
                  </div>

                  <div className="glass-card border border-sentinel-border rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-4 h-4 text-sentinel-accent-emerald" />
                      <span className="text-sm text-sentinel-text-muted">Win Rate</span>
                    </div>
                    <div className="text-2xl font-bold text-white">
                      {result.win_rate.toFixed(1)}%
                    </div>
                  </div>

                  <div className="glass-card border border-sentinel-border rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-sentinel-accent-amber" />
                      <span className="text-sm text-sentinel-text-muted">Max Drawdown</span>
                    </div>
                    <div className="text-2xl font-bold text-sentinel-accent-crimson">
                      -{result.max_drawdown.toFixed(2)}%
                    </div>
                  </div>

                  <div className="glass-card border border-sentinel-border rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-4 h-4 text-sentinel-accent-cyan" />
                      <span className="text-sm text-sentinel-text-muted">Total Trades</span>
                    </div>
                    <div className="text-2xl font-bold text-white">
                      {result.total_trades}
                    </div>
                  </div>
                </div>

                {/* Equity Curve */}
                <div className="glass-card border border-sentinel-border rounded-2xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Equity Curve</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={result.equity_curve}>
                        <defs>
                          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#00D9FF" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#00D9FF" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis 
                          dataKey="time" 
                          tick={{ fill: '#64748b', fontSize: 10 }}
                          tickFormatter={(value) => {
                            const date = new Date(value)
                            return `${date.getMonth()+1}/${date.getDate()}`
                          }}
                        />
                        <YAxis 
                          tick={{ fill: '#64748b', fontSize: 10 }}
                          tickFormatter={(value) => `$${value.toFixed(0)}`}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                          labelStyle={{ color: '#94a3b8' }}
                          formatter={(value: any) => [`$${value.toFixed(2)}`, 'Equity']}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="equity" 
                          stroke="#00D9FF" 
                          fill="url(#equityGradient)"
                          strokeWidth={2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Detailed Stats */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="glass-card border border-sentinel-border rounded-2xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">Performance Metrics</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Initial Capital</span>
                        <span className="text-white">${result.initial_capital.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Final Capital</span>
                        <span className={result.final_capital >= result.initial_capital ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}>
                          ${result.final_capital.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Total P&L</span>
                        <span className={result.total_pnl >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}>
                          {result.total_pnl >= 0 ? '+' : ''}${result.total_pnl.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Sharpe Ratio</span>
                        <span className="text-white">{result.sharpe_ratio.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Profit Factor</span>
                        <span className="text-white">{result.profit_factor.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="glass-card border border-sentinel-border rounded-2xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">Trade Statistics</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Winning Trades</span>
                        <span className="text-sentinel-accent-emerald">{result.winning_trades}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Losing Trades</span>
                        <span className="text-sentinel-accent-crimson">{result.losing_trades}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Average Win</span>
                        <span className="text-sentinel-accent-emerald">+{result.avg_win.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Average Loss</span>
                        <span className="text-sentinel-accent-crimson">{result.avg_loss.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sentinel-text-muted">Avg Trade Duration</span>
                        <span className="text-white">{result.avg_trade_duration.toFixed(0)} min</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recent Trades */}
                <div className="glass-card border border-sentinel-border rounded-2xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Recent Trades</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="text-sentinel-text-muted text-sm">
                          <th className="text-left py-2">Direction</th>
                          <th className="text-left py-2">Entry</th>
                          <th className="text-left py-2">Exit</th>
                          <th className="text-right py-2">P&L</th>
                          <th className="text-left py-2">Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.recent_trades.map((trade, i) => (
                          <tr key={i} className="border-t border-sentinel-border">
                            <td className="py-3">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                trade.direction === 'long' 
                                  ? 'bg-sentinel-accent-emerald/20 text-sentinel-accent-emerald' 
                                  : 'bg-sentinel-accent-crimson/20 text-sentinel-accent-crimson'
                              }`}>
                                {trade.direction.toUpperCase()}
                              </span>
                            </td>
                            <td className="py-3 text-white">${trade.entry_price?.toFixed(2)}</td>
                            <td className="py-3 text-white">${trade.exit_price?.toFixed(2)}</td>
                            <td className={`py-3 text-right ${trade.pnl >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}`}>
                              {trade.pnl >= 0 ? '+' : ''}{trade.pnl_percent?.toFixed(2)}%
                            </td>
                            <td className="py-3 text-sentinel-text-muted text-sm">{trade.exit_reason}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

