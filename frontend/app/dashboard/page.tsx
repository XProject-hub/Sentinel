'use client'

import { useEffect, useState, useRef } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { 
  TrendingUp, 
  TrendingDown, 
  Shield,
  Activity,
  Wallet,
  Target,
  AlertTriangle,
  Settings,
  LogOut,
  RefreshCw,
  Loader2,
  Zap,
  BarChart3,
  Play,
  Square,
  Clock,
  DollarSign,
  Percent,
  Eye,
  ChevronRight,
  ArrowUpRight,
  ArrowDownRight,
  Waves,
  Users,
  LineChart,
  Cpu
} from 'lucide-react'
import Logo from '@/components/Logo'
import ConnectExchangePrompt from '@/components/ConnectExchangePrompt'

interface Position {
  symbol: string
  side: string
  size: string
  entryPrice: string
  markPrice: string
  unrealisedPnl: string
  leverage: string
  positionValue: string
  fundingRate?: string
}

interface WalletData {
  totalEquity: number
  totalEquityUSDT: number
  availableBalance: number
  availableBalanceUSDT: number
  totalPnL: number
  dailyPnL: number
  weeklyPnL: number
  unrealizedPnL: number
}

interface TradingStatus {
  is_autonomous_trading: boolean
  is_paused: boolean
  is_connected: boolean
  total_pairs: number
  max_positions: number
  risk_mode: string
  trailing_stop_percent: number
}

interface ConsoleLog {
  time: string
  message: string
  level: string
}

interface WhaleAlert {
  symbol: string
  type: string
  description: string
  value: number
  timestamp: string
}

interface NewsItem {
  title: string
  sentiment: 'bullish' | 'bearish' | 'neutral'
  source: string
  time: string
}

export default function DashboardPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [news, setNews] = useState<NewsItem[]>([])
  const [hasExchangeConnection, setHasExchangeConnection] = useState(false)
  const [isAdmin, setIsAdmin] = useState(false)
  const [positions, setPositions] = useState<Position[]>([])
  const [wallet, setWallet] = useState<WalletData | null>(null)
  const [tradingStatus, setTradingStatus] = useState<TradingStatus | null>(null)
  const [consoleLogs, setConsoleLogs] = useState<ConsoleLog[]>([])
  const [whaleAlerts, setWhaleAlerts] = useState<WhaleAlert[]>([])
  const [isTogglingBot, setIsTogglingBot] = useState(false)
  const [marketRegime, setMarketRegime] = useState('analyzing')
  const [aiInsight, setAiInsight] = useState<string | null>(null)
  const [fearGreed, setFearGreed] = useState<number>(50)
  const [aiConfidence, setAiConfidence] = useState<number>(0)
  const [pairsScanned, setPairsScanned] = useState<number>(0)
  
  const consoleRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    checkAuth()
  }, [])

  const checkAuth = async () => {
    const storedUser = localStorage.getItem('sentinel_user')
    const token = localStorage.getItem('token')
    
    if (!storedUser || !token) {
      window.location.href = '/login'
      return
    }

    try {
      const user = JSON.parse(storedUser)
      setIsAdmin(user.email === 'admin@sentinel.ai')
      
      // Check exchange connection
      const response = await fetch('/api/exchanges', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      
      if (response.ok) {
        const data = await response.json()
        const hasConnection = data.data?.length > 0 || user.email === 'admin@sentinel.ai'
        setHasExchangeConnection(hasConnection)
        
        if (hasConnection) {
          loadDashboardData()
        }
      }
    } catch (error) {
      console.error('Auth check failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadDashboardData = async () => {
    try {
      const [walletRes, positionsRes, statusRes, consoleRes, whaleRes] = await Promise.all([
        fetch('/ai/exchange/balance?user_id=default'),
        fetch('/ai/exchange/positions?user_id=default'),
        fetch('/ai/exchange/trading/status?user_id=default'),
        fetch('/ai/exchange/trading/console?user_id=default&limit=20'),
        fetch('/ai/market/whale-alerts?limit=5')
      ])

      if (walletRes.ok) {
        const data = await walletRes.json()
        setWallet(data.data)
      }

      if (positionsRes.ok) {
        const data = await positionsRes.json()
        setPositions(data.data?.positions || [])
      }

      if (statusRes.ok) {
        const data = await statusRes.json()
        setTradingStatus(data.data)
        setMarketRegime(data.data?.current_regime || data.data?.risk_mode || 'normal')
        setAiConfidence(data.data?.ai_confidence || 0)
        setPairsScanned(data.data?.pairs_scanned || data.data?.total_pairs || 0)
      }

      if (consoleRes.ok) {
        const data = await consoleRes.json()
        setConsoleLogs(data.data?.logs || [])
      }

      if (whaleRes.ok) {
        const data = await whaleRes.json()
        setWhaleAlerts(data.alerts || [])
      }

      // Get AI insight
      try {
        const insightRes = await fetch('/ai/exchange/trading/activity?user_id=default')
        if (insightRes.ok) {
          const data = await insightRes.json()
          if (data.data?.ai_insight) {
            setAiInsight(data.data.ai_insight)
          }
        }
      } catch {}

      // Get market news/sentiment
      try {
        const newsRes = await fetch('/ai/market/news?limit=5')
        if (newsRes.ok) {
          const data = await newsRes.json()
          setNews(data.news || [])
        }
      } catch {}

      // Get Fear & Greed index
      try {
        const fgRes = await fetch('/ai/market/fear-greed')
        if (fgRes.ok) {
          const data = await fgRes.json()
          setFearGreed(data.value || 50)
        }
      } catch {}

    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    }
  }

  // Auto refresh every 5 seconds
  useEffect(() => {
    if (hasExchangeConnection) {
      const interval = setInterval(loadDashboardData, 5000)
      return () => clearInterval(interval)
    }
  }, [hasExchangeConnection])

  const startTrading = async () => {
    setIsTogglingBot(true)
    try {
      await fetch('/ai/exchange/trading/resume?user_id=default', { method: 'POST' })
      setTradingStatus(prev => prev ? { ...prev, is_autonomous_trading: true, is_paused: false } : null)
    } catch (error) {
      console.error('Failed to start:', error)
    } finally {
      setIsTogglingBot(false)
    }
  }

  const stopTrading = async () => {
    setIsTogglingBot(true)
    try {
      await fetch('/ai/exchange/trading/disable?user_id=default', { method: 'POST' })
      setTradingStatus(prev => prev ? { ...prev, is_autonomous_trading: false, is_paused: true } : null)
    } catch (error) {
      console.error('Failed to stop:', error)
    } finally {
      setIsTogglingBot(false)
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('sentinel_user')
    localStorage.removeItem('token')
    window.location.href = '/login'
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('de-DE', { style: 'currency', currency: 'EUR' }).format(value)
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0a0f1a] flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    )
  }

  if (!hasExchangeConnection && !isAdmin) {
    return <ConnectExchangePrompt />
  }

  const totalPnl = wallet?.totalPnL || 0
  const dailyPnl = wallet?.dailyPnL || 0
  const isTrading = tradingStatus?.is_autonomous_trading && !tradingStatus?.is_paused

  return (
    <div className="min-h-screen bg-[#0a0f1a]">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-[#0a0f1a]/95 backdrop-blur-xl border-b border-white/5">
        <div className="w-full px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Logo size="md" />
              
              {/* Status Indicator */}
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
                isTrading ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-amber-500/10 border border-amber-500/30'
              }`}>
                <div className={`w-2 h-2 rounded-full ${isTrading ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                <span className={`text-xs font-medium ${isTrading ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {isTrading ? 'Trading Active' : 'Paused'}
                </span>
              </div>

              {/* Market Info Bar */}
              <div className="hidden lg:flex items-center gap-4 px-4 py-1.5 bg-white/[0.02] rounded-lg border border-white/5">
                {/* Fear & Greed */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500 uppercase">Fear/Greed</span>
                  <span className={`text-xs font-bold ${
                    fearGreed <= 25 ? 'text-red-400' :
                    fearGreed <= 45 ? 'text-orange-400' :
                    fearGreed <= 55 ? 'text-gray-400' :
                    fearGreed <= 75 ? 'text-emerald-400' :
                    'text-green-400'
                  }`}>
                    {fearGreed}
                  </span>
                </div>

                <div className="w-px h-4 bg-white/10" />

                {/* Market Regime */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500 uppercase">Regime</span>
                  <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
                    marketRegime === 'trending' ? 'bg-emerald-500/20 text-emerald-400' :
                    marketRegime === 'ranging' || marketRegime === 'range_bound' ? 'bg-amber-500/20 text-amber-400' :
                    marketRegime === 'high_volatility' ? 'bg-red-500/20 text-red-400' :
                    'bg-cyan-500/20 text-cyan-400'
                  }`}>
                    {marketRegime.replace('_', ' ').toUpperCase()}
                  </span>
                </div>

                <div className="w-px h-4 bg-white/10" />

                {/* AI Confidence */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500 uppercase">AI Conf.</span>
                  <span className={`text-xs font-bold ${
                    aiConfidence >= 70 ? 'text-emerald-400' :
                    aiConfidence >= 50 ? 'text-cyan-400' :
                    'text-amber-400'
                  }`}>
                    {aiConfidence}%
                  </span>
                </div>

                <div className="w-px h-4 bg-white/10" />

                {/* Pairs Scanned */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500 uppercase">Pairs</span>
                  <span className="text-xs font-medium text-white">{pairsScanned}</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Link 
                href="/dashboard/settings" 
                className="p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <Settings className="w-5 h-5 text-gray-400" />
              </Link>
              
              <Link 
                href="/dashboard/backtest" 
                className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-sm text-gray-400"
              >
                Backtest
              </Link>
              
              {isAdmin && (
                <Link 
                  href="/admin" 
                  className="px-4 py-2 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 transition-colors text-sm text-cyan-400"
                >
                  Admin
                </Link>
              )}
              
              <button 
                onClick={handleLogout} 
                className="p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <LogOut className="w-5 h-5 text-gray-400" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="p-6">
        {/* AI Insight Banner */}
        {aiInsight && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-gradient-to-r from-cyan-500/10 via-blue-500/5 to-violet-500/10 border border-cyan-500/20 rounded-xl flex items-center gap-3"
          >
            <Cpu className="w-5 h-5 text-cyan-400 flex-shrink-0" />
            <span className="text-sm text-gray-300">{aiInsight}</span>
          </motion.div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {/* Total Equity */}
          <div className="p-5 bg-white/[0.02] rounded-2xl border border-white/5">
            <div className="flex items-center gap-2 mb-3">
              <Wallet className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-500">Total Equity</span>
            </div>
            <div className="text-2xl font-bold text-white">
              {(wallet?.totalEquityUSDT || wallet?.totalEquity || 0).toFixed(2)} USDT
            </div>
            <div className="text-xs text-gray-500 mt-1">
              ≈ {formatCurrency(wallet?.totalEquity || 0)}
            </div>
            <div className="text-xs text-gray-600 mt-0.5">
              Available: {(wallet?.availableBalanceUSDT || wallet?.availableBalance || 0).toFixed(2)} USDT
            </div>
          </div>

          {/* Total P&L */}
          <div className="p-5 bg-white/[0.02] rounded-2xl border border-white/5">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-500">Total P&L</span>
            </div>
            <div className={`text-2xl font-bold ${totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {totalPnl >= 0 ? '+' : ''}{formatCurrency(totalPnl)}
            </div>
            <div className="flex gap-3 mt-1">
              <span className={`text-xs ${dailyPnl >= 0 ? 'text-emerald-400/70' : 'text-red-400/70'}`}>
                Today: {dailyPnl >= 0 ? '+' : ''}{formatCurrency(dailyPnl)}
              </span>
              {wallet?.weeklyPnL !== undefined && (
                <span className={`text-xs ${wallet.weeklyPnL >= 0 ? 'text-emerald-400/70' : 'text-red-400/70'}`}>
                  Week: {wallet.weeklyPnL >= 0 ? '+' : ''}{formatCurrency(wallet.weeklyPnL)}
                </span>
              )}
            </div>
          </div>

          {/* Active Positions */}
          <div className="p-5 bg-white/[0.02] rounded-2xl border border-white/5">
            <div className="flex items-center gap-2 mb-3">
              <Target className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-500">Active Positions</span>
            </div>
            <div className="text-2xl font-bold text-white">
              {positions.length}
              <span className="text-sm text-gray-500 font-normal ml-1">
                / {tradingStatus?.max_positions || 10}
              </span>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Mode: {tradingStatus?.risk_mode?.toUpperCase() || 'NORMAL'}
            </div>
            {wallet?.unrealizedPnL !== undefined && (
              <div className={`text-xs mt-0.5 ${wallet.unrealizedPnL >= 0 ? 'text-emerald-400/70' : 'text-red-400/70'}`}>
                Unrealized: {wallet.unrealizedPnL >= 0 ? '+' : ''}{formatCurrency(wallet.unrealizedPnL)}
              </div>
            )}
          </div>

          {/* Trading Control */}
          <div className="p-5 bg-white/[0.02] rounded-2xl border border-white/5">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-500">Trading Control</span>
            </div>
            <button
              onClick={isTrading ? stopTrading : startTrading}
              disabled={isTogglingBot}
              className={`w-full py-2.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
                isTrading 
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30' 
                  : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/30'
              }`}
            >
              {isTogglingBot ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : isTrading ? (
                <>
                  <Square className="w-4 h-4" />
                  Stop Trading
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start Trading
                </>
              )}
            </button>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Positions Panel */}
          <div className="lg:col-span-2 bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
            <div className="p-5 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">Open Positions</h2>
              </div>
              <button 
                onClick={loadDashboardData}
                className="p-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                <RefreshCw className="w-4 h-4 text-gray-500" />
              </button>
            </div>
            
            {positions.length === 0 ? (
              <div className="p-12 text-center">
                <Target className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-500">No open positions</p>
                <p className="text-sm text-gray-600 mt-1">AI is scanning for opportunities...</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/5">
                      <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Pair</th>
                      <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Side</th>
                      <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">Size</th>
                      <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">Entry</th>
                      <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">Mark</th>
                      <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((pos, i) => {
                      const pnl = parseFloat(pos.unrealisedPnl || '0')
                      const pnlPercent = (pnl / parseFloat(pos.positionValue || '1')) * 100
                      return (
                        <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                          <td className="px-5 py-4">
                            <span className="font-medium text-white">{pos.symbol.replace('USDT', '')}</span>
                            <span className="text-gray-500">/USDT</span>
                          </td>
                          <td className="px-5 py-4">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              pos.side === 'Buy' 
                                ? 'bg-emerald-500/10 text-emerald-400' 
                                : 'bg-red-500/10 text-red-400'
                            }`}>
                              {pos.side === 'Buy' ? 'LONG' : 'SHORT'}
                            </span>
                          </td>
                          <td className="px-5 py-4 text-right">
                            <span className="text-white">{parseFloat(pos.size).toFixed(4)}</span>
                            <span className="text-gray-500 text-xs ml-1">{pos.leverage}x</span>
                          </td>
                          <td className="px-5 py-4 text-right text-gray-400">
                            €{parseFloat(pos.entryPrice).toFixed(4)}
                          </td>
                          <td className="px-5 py-4 text-right text-gray-400">
                            €{parseFloat(pos.markPrice).toFixed(4)}
                          </td>
                          <td className="px-5 py-4 text-right">
                            <div className={`font-medium ${pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {pnl >= 0 ? '+' : ''}€{pnl.toFixed(2)}
                            </div>
                            <div className={`text-xs ${pnl >= 0 ? 'text-emerald-400/70' : 'text-red-400/70'}`}>
                              {pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Right Sidebar */}
          <div className="space-y-6">
            {/* Console */}
            <div className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
              <div className="p-4 border-b border-white/5 flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isTrading ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'}`} />
                <h2 className="font-semibold text-white text-sm">AI Console</h2>
              </div>
              
              <div ref={consoleRef} className="h-64 overflow-y-auto p-4 font-mono text-xs space-y-2">
                {consoleLogs.length === 0 ? (
                  <div className="text-gray-600">Waiting for activity...</div>
                ) : (
                  consoleLogs.map((log, i) => (
                    <div key={i} className="flex gap-2">
                      <span className="text-gray-600">{new Date(log.time).toLocaleTimeString()}</span>
                      <span className={
                        log.level === 'TRADE' ? 'text-emerald-400' :
                        log.level === 'SIGNAL' ? 'text-cyan-400' :
                        log.level === 'WARNING' ? 'text-amber-400' :
                        log.level === 'ERROR' ? 'text-red-400' :
                        'text-gray-400'
                      }>
                        {log.message}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Whale Alerts */}
            <div className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
              <div className="p-4 border-b border-white/5 flex items-center gap-2">
                <Waves className="w-4 h-4 text-cyan-400" />
                <h2 className="font-semibold text-white text-sm">Whale Activity</h2>
              </div>
              
              <div className="divide-y divide-white/5">
                {whaleAlerts.length === 0 ? (
                  <div className="p-4 text-gray-600 text-sm">No whale activity detected</div>
                ) : (
                  whaleAlerts.map((alert, i) => (
                    <div key={i} className="p-3 hover:bg-white/[0.02]">
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span className="text-white text-sm font-medium">{alert.symbol}</span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                            alert.type === 'buy_wall' ? 'bg-emerald-500/20 text-emerald-400' :
                            'bg-red-500/20 text-red-400'
                          }`}>
                            {alert.type.replace('_', ' ').toUpperCase()}
                          </span>
                        </div>
                        <span className="text-[10px] text-gray-600">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 leading-relaxed">{alert.description}</p>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Market News */}
            <div className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
              <div className="p-4 border-b border-white/5 flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <h2 className="font-semibold text-white text-sm">Market Sentiment</h2>
              </div>
              
              <div className="divide-y divide-white/5">
                {news.length === 0 ? (
                  <div className="p-4 text-gray-600 text-sm">Loading market data...</div>
                ) : (
                  news.map((item, i) => (
                    <div key={i} className="p-3 hover:bg-white/[0.02]">
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-sm text-gray-300 leading-snug flex-1">{item.title}</p>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium flex-shrink-0 ${
                          item.sentiment === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' :
                          item.sentiment === 'bearish' ? 'bg-red-500/20 text-red-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {item.sentiment.toUpperCase()}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 mt-1.5">
                        <span className="text-[10px] text-gray-600">{item.source}</span>
                        <span className="text-[10px] text-gray-700">•</span>
                        <span className="text-[10px] text-gray-600">{item.time}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
