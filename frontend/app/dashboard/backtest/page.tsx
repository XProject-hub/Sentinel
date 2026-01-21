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
  Euro,
  Percent,
  Clock,
  Activity,
  CheckCircle,
  XCircle,
  ChevronDown,
  RefreshCw
} from 'lucide-react'
import Link from 'next/link'
import Logo from '@/components/Logo'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
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
  
  const getDefaultDates = () => {
    const end = new Date()
    const start = new Date()
    start.setDate(start.getDate() - 30)
    return {
      start_date: start.toISOString().split('T')[0],
      end_date: end.toISOString().split('T')[0]
    }
  }
  
  const defaultDates = getDefaultDates()
  
  const [config, setConfig] = useState({
    symbol: 'BTCUSDT',
    strategy: 'trend_following',
    start_date: defaultDates.start_date,
    end_date: defaultDates.end_date,
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
      const response = await fetch('/ai/backtest/strategies')
      if (response.ok) {
        const data = await response.json()
        if (data.strategies?.length > 0) setStrategies(data.strategies)
      }
    } catch {}
  }

  const loadSymbols = async () => {
    try {
      const response = await fetch('/ai/backtest/symbols')
      if (response.ok) {
        const data = await response.json()
        if (data.symbols?.length > 0) setSymbols(data.symbols)
      }
    } catch {}
  }

  const runBacktest = async () => {
    setIsRunning(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/ai/backtest/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })

      const data = await response.json()

      if (response.ok && data.success) {
        setResult(data)
      } else {
        setError(data.detail || data.message || data.error || 'Backtest failed')
      }
    } catch (e: any) {
      setError(`Connection error: ${e.message || 'Failed to connect'}`)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0f1a]">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-[#0a0f1a]/95 backdrop-blur-xl border-b border-white/5">
        <div className="w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/dashboard" 
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-400" />
              </Link>
              <div>
                <h1 className="text-xl font-bold text-white">Strategy Backtester</h1>
                <p className="text-sm text-gray-500">Test strategies on historical data</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              <span className="text-gray-400 text-sm">Historical Analysis</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6 pb-16">
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="bg-white/[0.02] rounded-2xl border border-white/5 p-6">
            <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-400" />
              Configuration
            </h2>

            <div className="space-y-5">
              {/* Symbol */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Trading Pair</label>
                <div className="relative">
                  <select
                    value={config.symbol}
                    onChange={(e) => setConfig({ ...config, symbol: e.target.value })}
                    className="w-full px-4 py-3 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none appearance-none cursor-pointer"
                    style={{ colorScheme: 'dark' }}
                  >
                    {symbols.map((s) => (
                      <option key={s.symbol} value={s.symbol} className="bg-[#0d1321]">
                        {s.symbol} - {s.name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 pointer-events-none" />
                </div>
              </div>

              {/* Strategy */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Strategy</label>
                <div className="relative">
                  <select
                    value={config.strategy}
                    onChange={(e) => setConfig({ ...config, strategy: e.target.value })}
                    className="w-full px-4 py-3 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none appearance-none cursor-pointer"
                    style={{ colorScheme: 'dark' }}
                  >
                    {strategies.map((s) => (
                      <option key={s.id} value={s.id} className="bg-[#0d1321]">{s.name}</option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 pointer-events-none" />
                </div>
                <p className="text-xs text-gray-500 mt-1.5">
                  {strategies.find(s => s.id === config.strategy)?.description}
                </p>
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Start Date</label>
                  <input
                    type="date"
                    value={config.start_date}
                    onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
                    className="w-full px-3 py-2.5 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none text-sm"
                    style={{ colorScheme: 'dark' }}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">End Date</label>
                  <input
                    type="date"
                    value={config.end_date}
                    onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
                    className="w-full px-3 py-2.5 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none text-sm"
                    style={{ colorScheme: 'dark' }}
                  />
                </div>
              </div>

              {/* Capital */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Initial Capital (€)</label>
                <input
                  type="number"
                  value={config.initial_capital}
                  onChange={(e) => setConfig({ ...config, initial_capital: parseFloat(e.target.value) || 1000 })}
                  className="w-full px-4 py-3 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none"
                />
              </div>

              {/* Risk Settings */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Take Profit %</label>
                  <input
                    type="number"
                    step="0.1"
                    value={config.take_profit_percent}
                    onChange={(e) => setConfig({ ...config, take_profit_percent: parseFloat(e.target.value) || 2 })}
                    className="w-full px-3 py-2.5 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Stop Loss %</label>
                  <input
                    type="number"
                    step="0.1"
                    value={config.stop_loss_percent}
                    onChange={(e) => setConfig({ ...config, stop_loss_percent: parseFloat(e.target.value) || 1 })}
                    className="w-full px-3 py-2.5 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none text-sm"
                  />
                </div>
              </div>

              {/* Position Size */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Position Size %</label>
                <input
                  type="number"
                  value={config.position_size_percent}
                  onChange={(e) => setConfig({ ...config, position_size_percent: parseFloat(e.target.value) || 10 })}
                  className="w-full px-4 py-3 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none"
                />
              </div>

              {/* Leverage */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Leverage</label>
                <div className="relative">
                  <select
                    value={config.leverage}
                    onChange={(e) => setConfig({ ...config, leverage: parseInt(e.target.value) })}
                    className="w-full px-4 py-3 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none appearance-none cursor-pointer"
                    style={{ colorScheme: 'dark' }}
                  >
                    <option value={1} className="bg-[#0d1321]">1x (No leverage)</option>
                    <option value={2} className="bg-[#0d1321]">2x</option>
                    <option value={3} className="bg-[#0d1321]">3x</option>
                    <option value={5} className="bg-[#0d1321]">5x</option>
                    <option value={10} className="bg-[#0d1321]">10x</option>
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 pointer-events-none" />
                </div>
              </div>

              {/* Run Button */}
              <button
                onClick={runBacktest}
                disabled={isRunning}
                className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-bold hover:shadow-lg hover:shadow-cyan-500/25 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
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
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6"
              >
                <div className="flex items-center gap-3 text-red-400">
                  <AlertTriangle className="w-6 h-6 flex-shrink-0" />
                  <span>{error}</span>
                </div>
              </motion.div>
            )}

            {!result && !isRunning && !error && (
              <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-12 text-center">
                <BarChart3 className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">No Backtest Results</h3>
                <p className="text-gray-500">
                  Configure your strategy and click "Run Backtest" to see results
                </p>
              </div>
            )}

            {isRunning && (
              <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-12 text-center">
                <Loader2 className="w-16 h-16 text-cyan-400 mx-auto mb-4 animate-spin" />
                <h3 className="text-xl font-semibold text-white mb-2">Running Backtest...</h3>
                <p className="text-gray-500">Analyzing historical data for {config.symbol}</p>
              </div>
            )}

            {result && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-6"
              >
                {/* Performance Summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white/[0.02] border border-white/5 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Euro className="w-4 h-4 text-cyan-400" />
                      <span className="text-sm text-gray-500">Total Return</span>
                    </div>
                    <div className={`text-2xl font-bold ${result.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {result.total_return >= 0 ? '+' : ''}{result.total_return.toFixed(2)}%
                    </div>
                  </div>

                  <div className="bg-white/[0.02] border border-white/5 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-4 h-4 text-emerald-400" />
                      <span className="text-sm text-gray-500">Win Rate</span>
                    </div>
                    <div className="text-2xl font-bold text-white">{result.win_rate.toFixed(1)}%</div>
                  </div>

                  <div className="bg-white/[0.02] border border-white/5 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      <span className="text-sm text-gray-500">Max Drawdown</span>
                    </div>
                    <div className="text-2xl font-bold text-red-400">-{result.max_drawdown.toFixed(2)}%</div>
                  </div>

                  <div className="bg-white/[0.02] border border-white/5 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-4 h-4 text-cyan-400" />
                      <span className="text-sm text-gray-500">Total Trades</span>
                    </div>
                    <div className="text-2xl font-bold text-white">{result.total_trades}</div>
                  </div>
                </div>

                {/* Equity Curve */}
                <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Equity Curve</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={result.equity_curve}>
                        <defs>
                          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
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
                          tickFormatter={(value) => `€${value.toFixed(0)}`}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                          labelStyle={{ color: '#94a3b8' }}
                          formatter={(value: any) => [`€${value.toFixed(2)}`, 'Equity']}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="equity" 
                          stroke="#06b6d4" 
                          fill="url(#equityGradient)"
                          strokeWidth={2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Detailed Stats */}
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">Performance Metrics</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Initial Capital</span>
                        <span className="text-white">€{result.initial_capital.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Final Capital</span>
                        <span className={result.final_capital >= result.initial_capital ? 'text-emerald-400' : 'text-red-400'}>
                          €{result.final_capital.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Total P&L</span>
                        <span className={result.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                          {result.total_pnl >= 0 ? '+' : ''}€{result.total_pnl.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Sharpe Ratio</span>
                        <span className="text-white">{result.sharpe_ratio.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Profit Factor</span>
                        <span className="text-white">{result.profit_factor.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">Trade Statistics</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Winning Trades</span>
                        <span className="text-emerald-400">{result.winning_trades}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Losing Trades</span>
                        <span className="text-red-400">{result.losing_trades}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Average Win</span>
                        <span className="text-emerald-400">+{result.avg_win.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Average Loss</span>
                        <span className="text-red-400">{result.avg_loss.toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Avg Duration</span>
                        <span className="text-white">{result.avg_trade_duration.toFixed(0)} min</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recent Trades */}
                {result.recent_trades?.length > 0 && (
                  <div className="bg-white/[0.02] border border-white/5 rounded-2xl overflow-hidden">
                    <div className="p-5 border-b border-white/5">
                      <h3 className="text-lg font-semibold text-white">Recent Trades</h3>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-white/5">
                            <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Direction</th>
                            <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Entry</th>
                            <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Exit</th>
                            <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">P&L</th>
                            <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Reason</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.recent_trades.map((trade, i) => (
                            <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                              <td className="px-5 py-3">
                                <span className={`px-2 py-1 rounded text-xs font-medium ${
                                  trade.direction === 'long' 
                                    ? 'bg-emerald-500/10 text-emerald-400' 
                                    : 'bg-red-500/10 text-red-400'
                                }`}>
                                  {trade.direction?.toUpperCase()}
                                </span>
                              </td>
                              <td className="px-5 py-3 text-white">€{trade.entry_price?.toFixed(2)}</td>
                              <td className="px-5 py-3 text-white">€{trade.exit_price?.toFixed(2)}</td>
                              <td className={`px-5 py-3 text-right ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {trade.pnl >= 0 ? '+' : ''}{trade.pnl_percent?.toFixed(2)}%
                              </td>
                              <td className="px-5 py-3 text-gray-500 text-sm">{trade.exit_reason}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
