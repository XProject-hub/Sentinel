'use client'

import { useEffect, useState, useRef } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { 
  TrendingUp, 
  TrendingDown, 
  Activity,
  Wallet,
  Target,
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
  Waves,
  LineChart,
  Cpu,
  Trophy,
  CheckCircle,
  XCircle,
  Hash,
  Euro,
  Gauge,
  Layers,
  Sparkles,
  Grid3X3
} from 'lucide-react'
import Logo from '@/components/Logo'
import ConnectExchangePrompt from '@/components/ConnectExchangePrompt'
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'

interface Position {
  symbol: string
  side: string
  size: string
  entryPrice: string
  markPrice: string
  unrealisedPnl: string
  estimatedNetPnl?: string  // NET P&L after estimated exit fees
  estimatedExitFee?: string
  leverage: string
  positionValue: string
  isBreakout?: boolean  // Flag for breakout positions (+2 extra slots)
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
}

interface ConsoleLog {
  time: string
  message: string
  level: string
}

interface WhaleAlert {
  symbol: string
  type: string
  timestamp: string
}

interface NewsItem {
  title: string
  sentiment: 'bullish' | 'bearish' | 'neutral'
  source: string
  time: string
}

interface TradeHistory {
  symbol: string
  pnl: number
  pnl_percent: number
  close_reason: string
  closed_time: string
}

interface TraderStats {
  total_trades: number
  winning_trades: number
  total_pnl: number
  max_drawdown: number
  opportunities_scanned: number
  win_rate: number
  best_trade: number
  worst_trade: number
}

interface AISignal {
  symbol: string
  direction: 'LONG' | 'SHORT'
  confidence: number
  entry_price: number
  target_price: number
  stop_loss: number
  edge: number
  reason: string
  funding_rate?: number
  long_short_ratio?: number
}

// EUR/USD rate (approximate)
const USDT_TO_EUR = 0.92

export default function DashboardPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [news, setNews] = useState<NewsItem[]>([])
  const [aiSignals, setAiSignals] = useState<AISignal[]>([])
  const [hasExchangeConnection, setHasExchangeConnection] = useState(false)
  const [isAdmin, setIsAdmin] = useState(false)
  const [positions, setPositions] = useState<Position[]>([])
  const [wallet, setWallet] = useState<WalletData | null>(null)
  const [tradingStatus, setTradingStatus] = useState<TradingStatus | null>(null)
  const [consoleLogs, setConsoleLogs] = useState<ConsoleLog[]>([])
  const [whaleAlerts, setWhaleAlerts] = useState<WhaleAlert[]>([])
  const [isTogglingBot, setIsTogglingBot] = useState(false)
  const [marketRegime, setMarketRegime] = useState('analyzing')
  const [fearGreed, setFearGreed] = useState<number>(50)
  const [aiConfidence, setAiConfidence] = useState<number>(0)
  const [breakoutAlerts, setBreakoutAlerts] = useState<{symbol: string, change: number, volume: number, time: string}[]>([])
  const [aiIntelligence, setAiIntelligence] = useState<{
    news_sentiment: string
    breakouts_detected: number
    pairs_analyzed: number
    last_action: string
    strategy_mode: string
  }>({
    news_sentiment: 'neutral',
    breakouts_detected: 0,
    pairs_analyzed: 0,
    last_action: 'Initializing...',
    strategy_mode: 'NORMAL'
  })
  const [pairsScanned, setPairsScanned] = useState<number>(0)
  const [recentTrades, setRecentTrades] = useState<TradeHistory[]>([])
  const [traderStats, setTraderStats] = useState<TraderStats | null>(null)
  const [pnlHistory, setPnlHistory] = useState<{time: string, pnl: number}[]>([])
  const [last100Pnl, setLast100Pnl] = useState<number>(0)
  const [userId, setUserId] = useState<string>('default')
  
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
      // Set the actual user ID for API calls
      const currentUserId = user.id || user.userId || user.user_id || 'default'
      setUserId(currentUserId)
      setIsAdmin(user.email === 'admin@sentinel.ai')
      
      const response = await fetch('/api/exchanges', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      
      if (response.ok) {
        const data = await response.json()
        const hasConnection = data.data?.length > 0 || user.email === 'admin@sentinel.ai'
        setHasExchangeConnection(hasConnection)
        
        if (hasConnection) {
          loadDashboardData(currentUserId)
        }
      }
    } catch (error) {
      console.error('Auth check failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadDashboardData = async (currentUserId?: string) => {
    const uid = currentUserId || userId
    try {
      const [walletRes, positionsRes, statusRes, consoleRes, whaleRes, tradesRes, statsRes] = await Promise.all([
        fetch(`/ai/exchange/balance?user_id=${uid}`),
        fetch(`/ai/exchange/positions?user_id=${uid}`),
        fetch(`/ai/exchange/trading/status?user_id=${uid}`),
        fetch(`/ai/exchange/trading/console?user_id=${uid}&limit=20`),
        fetch('/ai/market/whale-alerts?limit=10'),
        fetch(`/ai/exchange/trades/history?user_id=${uid}&limit=100`),
        fetch(`/ai/exchange/trading/stats?user_id=${uid}`)
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

      if (tradesRes.ok) {
        const data = await tradesRes.json()
        const trades = data.data?.trades || data.trades || []
        setRecentTrades(trades)
        
        // Build P&L history from last 100 trades (oldest to newest)
        if (trades.length > 0) {
          let runningPnl = 0
          const sortedTrades = [...trades].reverse()
          const history = sortedTrades.map((t: TradeHistory) => {
            runningPnl += t.pnl
            return {
              time: new Date(t.closed_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
              pnl: runningPnl
            }
          })
          setPnlHistory(history)
          // Set P&L sum from last 100 trades
          setLast100Pnl(runningPnl)
        }
      }

      if (statsRes.ok) {
        const data = await statsRes.json()
        const stats = data.data || data
        const totalTrades = stats.total_trades || 0
        const winningTrades = stats.winning_trades || 0
        const totalPnl = stats.total_pnl || 0
        
        setTraderStats({
          total_trades: totalTrades,
          winning_trades: winningTrades,
          total_pnl: totalPnl,
          max_drawdown: stats.max_drawdown || 0,
          opportunities_scanned: stats.opportunities_scanned || 0,
          win_rate: totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0,
          best_trade: stats.best_trade || 0,
          worst_trade: stats.worst_trade || 0
        })
      }

      // Get market news
      try {
        const newsRes = await fetch('/ai/market/news?limit=10')
        if (newsRes.ok) {
          const data = await newsRes.json()
          setNews(data.news || [])
        }
      } catch {}

      // Get Fear & Greed
      try {
        const fgRes = await fetch('/ai/market/fear-greed')
        if (fgRes.ok) {
          const data = await fgRes.json()
          setFearGreed(data.value || 50)
        }
      } catch {}

      // Get AI Signals (top opportunities)
      try {
        const signalsRes = await fetch('/ai/exchange/signals?limit=5')
        if (signalsRes.ok) {
          const data = await signalsRes.json()
          setAiSignals(data.signals || [])
        }
      } catch {}

      // Get AI Intelligence status (breakouts, news sentiment, etc.)
      try {
        const aiRes = await fetch('/ai/exchange/intelligence')
        if (aiRes.ok) {
          const data = await aiRes.json()
          setAiIntelligence({
            news_sentiment: data.news_sentiment || 'neutral',
            breakouts_detected: data.breakouts_detected || 0,
            pairs_analyzed: data.pairs_analyzed || 0,
            last_action: data.last_action || 'Scanning...',
            strategy_mode: data.strategy_mode || 'NORMAL'
          })
          if (data.breakout_alerts && data.breakout_alerts.length > 0) {
            setBreakoutAlerts(data.breakout_alerts)
          }
        }
      } catch {}

    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    }
  }

  useEffect(() => {
    if (hasExchangeConnection) {
      const interval = setInterval(loadDashboardData, 5000)
      return () => clearInterval(interval)
    }
  }, [hasExchangeConnection])

  const startTrading = async () => {
    setIsTogglingBot(true)
    try {
      await fetch(`/ai/exchange/trading/resume?user_id=${userId}`, { method: 'POST' })
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
      await fetch(`/ai/exchange/trading/disable?user_id=${userId}`, { method: 'POST' })
      setTradingStatus(prev => prev ? { ...prev, is_autonomous_trading: false, is_paused: true } : null)
    } catch (error) {
      console.error('Failed to stop:', error)
    } finally {
      setIsTogglingBot(false)
    }
  }

  const [closingPositions, setClosingPositions] = useState<Set<string>>(new Set())

  const closePosition = async (symbol: string) => {
    if (closingPositions.has(symbol)) return
    
    const confirmed = window.confirm(`Are you sure you want to close ${symbol} position?`)
    if (!confirmed) return

    setClosingPositions(prev => new Set(prev).add(symbol))
    try {
      const response = await fetch(`/ai/exchange/close-position/${symbol}`, {
        method: 'POST'
      })
      if (response.ok) {
        // Refresh dashboard data after closing
        setTimeout(() => loadDashboardData(), 1000)
      } else {
        const error = await response.json()
        alert(`Failed to close position: ${error.detail || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Failed to close position:', error)
      alert('Failed to close position. Check console for details.')
    } finally {
      setClosingPositions(prev => {
        const next = new Set(prev)
        next.delete(symbol)
        return next
      })
    }
  }

  const [isSellingAll, setIsSellingAll] = useState(false)

  const sellAllPositions = async () => {
    if (positions.length === 0) return
    
    const confirmed = window.confirm(`Are you sure you want to close ALL ${positions.length} positions?`)
    if (!confirmed) return

    setIsSellingAll(true)
    try {
      const response = await fetch('/ai/exchange/close-all-positions', {
        method: 'POST'
      })
      if (response.ok) {
        const result = await response.json()
        alert(`Successfully closed ${result.closed || positions.length} positions`)
        setTimeout(() => loadDashboardData(), 1000)
      } else {
        const error = await response.json()
        alert(`Failed to close all positions: ${error.detail || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Failed to close all positions:', error)
      alert('Failed to close all positions. Check console for details.')
    } finally {
      setIsSellingAll(false)
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('sentinel_user')
    localStorage.removeItem('token')
    window.location.href = '/login'
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#060a13] flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    )
  }

  if (!hasExchangeConnection && !isAdmin) {
    return <ConnectExchangePrompt />
  }

  const balanceUSDT = wallet?.totalEquityUSDT || wallet?.totalEquity || 0
  const balanceEUR = balanceUSDT * USDT_TO_EUR
  const dailyPnl = wallet?.dailyPnL || 0
  const totalPnl = traderStats?.total_pnl || 0
  const isTrading = tradingStatus?.is_autonomous_trading && !tradingStatus?.is_paused
  const winRate = traderStats?.win_rate || 0

  return (
    <div className="min-h-screen bg-[#060a13]">
      {/* Background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.02)_1px,transparent_1px)] bg-[size:44px_44px]" />
      </div>

      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-[#060a13]/90 backdrop-blur-xl border-b border-white/5">
        <div className="w-full px-3 sm:px-6 py-2 sm:py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 sm:gap-4 lg:gap-6">
              <div className="hidden sm:block">
                <Logo size="md" />
              </div>
              <div className="sm:hidden">
                <Logo size="sm" />
              </div>
              
              <div className={`flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full ${
                isTrading ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-amber-500/10 border border-amber-500/30'
              }`}>
                <div className={`w-1.5 sm:w-2 h-1.5 sm:h-2 rounded-full ${isTrading ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                <span className={`text-[10px] sm:text-xs font-medium ${isTrading ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {isTrading ? 'Active' : 'Paused'}
                </span>
              </div>

              {/* Market Info - Hidden on mobile */}
              <div className="hidden xl:flex items-center gap-4 px-4 py-1.5 bg-white/[0.02] rounded-lg border border-white/5">
                <div className="flex items-center gap-2">
                  <Gauge className={`w-3.5 h-3.5 ${fearGreed <= 40 ? 'text-red-400' : fearGreed >= 60 ? 'text-emerald-400' : 'text-gray-400'}`} />
                  <span className="text-[10px] text-gray-500">F/G</span>
                  <span className={`text-xs font-bold ${fearGreed <= 40 ? 'text-red-400' : fearGreed >= 60 ? 'text-emerald-400' : 'text-gray-400'}`}>
                    {fearGreed}
                  </span>
                </div>
                <div className="w-px h-4 bg-white/10" />
                <div className="flex items-center gap-2">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  <span className="text-[10px] text-gray-500">Regime</span>
                  <span className="text-xs text-cyan-400 uppercase">{marketRegime.replace('_', ' ')}</span>
                </div>
                <div className="w-px h-4 bg-white/10" />
                <div className="flex items-center gap-2">
                  <Sparkles className="w-3.5 h-3.5 text-violet-400" />
                  <span className="text-[10px] text-gray-500">AI Cnf</span>
                  <span className="text-xs text-violet-400 font-medium">{aiConfidence}%</span>
                </div>
                <div className="w-px h-4 bg-white/10" />
                <div className="flex items-center gap-2">
                  <Grid3X3 className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-[10px] text-gray-500">Pairs</span>
                  <span className="text-xs text-amber-400 font-medium">{pairsScanned}</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-1.5 sm:gap-3">
              <Link href="/dashboard/settings" className="p-2 sm:p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
                <Settings className="w-4 h-4 sm:w-5 sm:h-5 text-gray-400" />
              </Link>
              <Link href="/dashboard/backtest" className="hidden sm:flex px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-xs sm:text-sm text-gray-400">
                Backtest
              </Link>
              {isAdmin && (
                <Link href="/admin" className="px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 transition-colors text-xs sm:text-sm text-cyan-400">
                  Admin
                </Link>
              )}
              <button onClick={handleLogout} className="p-2 sm:p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
                <LogOut className="w-4 h-4 sm:w-5 sm:h-5 text-gray-400" />
              </button>
            </div>
          </div>
          
          {/* Mobile Market Info Bar */}
          <div className="flex xl:hidden items-center justify-between gap-2 mt-2 px-2 py-1.5 bg-white/[0.02] rounded-lg border border-white/5 overflow-x-auto">
            <div className="flex items-center gap-1 shrink-0">
              <Gauge className={`w-3 h-3 ${fearGreed <= 40 ? 'text-red-400' : fearGreed >= 60 ? 'text-emerald-400' : 'text-gray-400'}`} />
              <span className="text-[9px] text-gray-500">F/G</span>
              <span className={`text-[10px] font-bold ${fearGreed <= 40 ? 'text-red-400' : fearGreed >= 60 ? 'text-emerald-400' : 'text-gray-400'}`}>
                {fearGreed}
              </span>
            </div>
            <div className="w-px h-3 bg-white/10 shrink-0" />
            <div className="flex items-center gap-1 shrink-0">
              <Layers className="w-3 h-3 text-cyan-400" />
              <span className="text-[9px] text-gray-500">Regime</span>
              <span className="text-[10px] text-cyan-400 uppercase">{marketRegime.replace('_', ' ')}</span>
            </div>
            <div className="w-px h-3 bg-white/10 shrink-0" />
            <div className="flex items-center gap-1 shrink-0">
              <Sparkles className="w-3 h-3 text-violet-400" />
              <span className="text-[9px] text-gray-500">AI</span>
              <span className="text-[10px] text-violet-400 font-medium">{aiConfidence}%</span>
            </div>
            <div className="w-px h-3 bg-white/10 shrink-0" />
            <div className="flex items-center gap-1 shrink-0">
              <Grid3X3 className="w-3 h-3 text-amber-400" />
              <span className="text-[9px] text-gray-500">Pairs</span>
              <span className="text-[10px] text-amber-400 font-medium">{pairsScanned}</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="p-3 sm:p-6 relative">
        {/* Top Stats - Responsive grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2 sm:gap-4 mb-4 sm:mb-6">
          {/* Balance */}
          <div className="p-3 sm:p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 sm:gap-2 mb-1 sm:mb-2">
              <Wallet className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-cyan-400" />
              <span className="text-[10px] sm:text-xs text-gray-500">Balance</span>
            </div>
            <div className="text-base sm:text-xl font-bold text-white">{balanceUSDT.toFixed(2)} USDT</div>
            <div className="flex items-center gap-1 text-xs sm:text-sm text-gray-400 mt-0.5 sm:mt-1">
              <Euro className="w-2.5 h-2.5 sm:w-3 sm:h-3" />
              <span>{balanceEUR.toFixed(2)} EUR</span>
            </div>
          </div>

          {/* Daily P&L */}
          <div className="p-3 sm:p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 sm:gap-2 mb-1 sm:mb-2">
              <TrendingUp className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-400" />
              <span className="text-[10px] sm:text-xs text-gray-500">Daily P&L</span>
            </div>
            <div className={`text-base sm:text-xl font-bold ${dailyPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {dailyPnl >= 0 ? '+' : ''}{dailyPnl.toFixed(2)} €
            </div>
          </div>

          {/* Win Rate */}
          <div className="p-3 sm:p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 sm:gap-2 mb-1 sm:mb-2">
              <Trophy className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-amber-400" />
              <span className="text-[10px] sm:text-xs text-gray-500">Win Rate</span>
            </div>
            <div className={`text-base sm:text-xl font-bold ${winRate >= 50 ? 'text-emerald-400' : 'text-red-400'}`}>
              {winRate.toFixed(1)}%
            </div>
            <div className="text-[10px] sm:text-xs text-gray-500">{traderStats?.winning_trades || 0}W / {(traderStats?.total_trades || 0) - (traderStats?.winning_trades || 0)}L</div>
          </div>

          {/* Total Trades */}
          <div className="p-3 sm:p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 sm:gap-2 mb-1 sm:mb-2">
              <Hash className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-violet-400" />
              <span className="text-[10px] sm:text-xs text-gray-500">Total Trades</span>
            </div>
            <div className="text-base sm:text-xl font-bold text-white">{traderStats?.total_trades || 0}</div>
          </div>

          {/* Trading Control */}
          <div className="p-3 sm:p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 sm:gap-2 mb-1 sm:mb-2">
              <Zap className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-400" />
              <span className="text-[10px] sm:text-xs text-gray-500">Control</span>
            </div>
            <div className="flex flex-col gap-2">
              <button
                onClick={isTrading ? stopTrading : startTrading}
                disabled={isTogglingBot}
                className={`w-full py-2 sm:py-2.5 rounded-lg font-semibold text-xs sm:text-sm flex items-center justify-center gap-1.5 sm:gap-2 transition-all ${
                  isTrading 
                    ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30' 
                    : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/30'
                }`}
              >
                {isTogglingBot ? <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" /> : isTrading ? <><Square className="w-3.5 h-3.5 sm:w-4 sm:h-4" /> Stop</> : <><Play className="w-3.5 h-3.5 sm:w-4 sm:h-4" /> Start</>}
              </button>
              {positions.length > 0 && (
                <button
                  onClick={sellAllPositions}
                  disabled={isSellingAll}
                  className="w-full py-2 sm:py-2.5 rounded-lg font-semibold text-xs sm:text-sm flex items-center justify-center gap-1.5 sm:gap-2 transition-all bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 border border-amber-500/30"
                >
                  {isSellingAll ? <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" /> : <><XCircle className="w-3.5 h-3.5 sm:w-4 sm:h-4" /> Sell All</>}
                </button>
              )}
            </div>
          </div>
        </div>

        {/* OPEN POSITIONS - FIRST AND LARGE */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-3 sm:gap-4 mb-4 sm:mb-6">
          <div className="lg:col-span-3 bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="p-3 sm:p-4 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
                <h2 className="text-sm sm:text-base font-semibold text-white">Open Positions</h2>
                <span className="text-[10px] sm:text-xs text-gray-500">({positions.length})</span>
              </div>
              <button onClick={loadDashboardData} className="p-1 sm:p-1.5 rounded-lg hover:bg-white/5">
                <RefreshCw className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-500" />
              </button>
            </div>
            
            {positions.length === 0 ? (
              <div className="p-8 text-center">
                <Target className="w-10 h-10 text-gray-700 mx-auto mb-3" />
                <p className="text-gray-500 text-sm">No open positions</p>
                <p className="text-gray-600 text-xs">AI is scanning...</p>
              </div>
            ) : (
              <div className="overflow-auto" style={{ maxHeight: '350px' }}>
                <table className="w-full">
                  <thead className="sticky top-0 bg-[#060a13] z-10">
                    <tr className="border-b border-white/5">
                      <th className="text-left text-[10px] font-medium text-gray-500 px-3 py-2">PAIR</th>
                      <th className="text-left text-[10px] font-medium text-gray-500 px-3 py-2">SIDE</th>
                      <th className="text-right text-[10px] font-medium text-gray-500 px-3 py-2">VALUE</th>
                      <th className="text-right text-[10px] font-medium text-gray-500 px-3 py-2">ENTRY</th>
                      <th className="text-right text-[10px] font-medium text-gray-500 px-3 py-2">MARK</th>
                      <th className="text-right text-[10px] font-medium text-gray-500 px-3 py-2">P&L</th>
                      <th className="text-center text-[10px] font-medium text-gray-500 px-3 py-2">ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((pos, i) => {
                      const entryPrice = parseFloat(pos.entryPrice || '0')
                      const markPrice = parseFloat(pos.markPrice || '0')
                      const size = parseFloat(pos.size || '0')
                      const leverage = parseFloat(pos.leverage || '1')
                      
                      // Calculate position value: use API value or calculate from size * markPrice
                      let posValueUSDT = parseFloat(pos.positionValue || '0')
                      if (posValueUSDT === 0 && markPrice > 0 && size > 0) {
                        posValueUSDT = size * markPrice
                      }
                      const posValueEUR = posValueUSDT * USDT_TO_EUR
                      
                      // Calculate P&L percentage
                      let pnlPercent = 0
                      if (entryPrice > 0 && markPrice > 0) {
                        if (pos.side === 'Buy') {
                          pnlPercent = ((markPrice - entryPrice) / entryPrice) * 100 * leverage
                        } else {
                          pnlPercent = ((entryPrice - markPrice) / entryPrice) * 100 * leverage
                        }
                      }
                      
                      // Calculate P&L in EUR
                      const pnlEUR = posValueEUR * (pnlPercent / 100)
                      
                      // Use NET P&L (after fees) if available, otherwise gross
                      const netPnl = parseFloat(pos.estimatedNetPnl || '0')
                      const grossPnl = parseFloat(pos.unrealisedPnl || '0')
                      
                      // Prefer NET P&L (more accurate), fallback to gross
                      const apiPnl = netPnl !== 0 ? netPnl : grossPnl
                      const finalPnlEUR = apiPnl !== 0 ? apiPnl * USDT_TO_EUR : pnlEUR
                      const finalPnlPercent = apiPnl !== 0 && posValueUSDT > 0 ? (apiPnl / posValueUSDT) * 100 : pnlPercent
                      
                      return (
                        <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-1.5">
                              <span className="font-medium text-white text-sm">{pos.symbol.replace('USDT', '')}</span>
                              <span className="text-gray-600 text-[10px]">/USDT</span>
                              {pos.isBreakout && (
                                <span className="px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-400 text-[8px] font-bold animate-pulse">
                                  BREAKOUT
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="px-3 py-2">
                            <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                              pos.side === 'Buy' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                            }`}>
                              {pos.side === 'Buy' ? 'LONG' : 'SHORT'} {pos.leverage}x
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right">
                            <span className="text-white font-medium text-sm">€{posValueEUR.toFixed(2)}</span>
                          </td>
                          <td className="px-3 py-2 text-right text-gray-400 text-sm">
                            €{entryPrice.toFixed(4)}
                          </td>
                          <td className="px-3 py-2 text-right text-gray-400 text-sm">
                            €{markPrice.toFixed(4)}
                          </td>
                          <td className="px-3 py-2 text-right">
                            <div className={`font-medium text-sm ${finalPnlEUR >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {finalPnlEUR >= 0 ? '+' : ''}€{finalPnlEUR.toFixed(2)}
                            </div>
                            <div className={`text-[9px] ${finalPnlPercent >= 0 ? 'text-emerald-400/70' : 'text-red-400/70'}`}>
                              {finalPnlPercent >= 0 ? '+' : ''}{finalPnlPercent.toFixed(2)}%
                            </div>
                          </td>
                          <td className="px-3 py-2 text-center">
                            <button
                              onClick={() => closePosition(pos.symbol)}
                              disabled={closingPositions.has(pos.symbol)}
                              className={`px-2 py-1 rounded text-[10px] font-medium border transition-colors ${
                                closingPositions.has(pos.symbol)
                                  ? 'bg-gray-500/10 text-gray-400 border-gray-500/30 cursor-not-allowed'
                                  : 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border-red-500/30'
                              }`}
                            >
                              {closingPositions.has(pos.symbol) ? 'CLOSING...' : 'CLOSE'}
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* AI Signals + Market News */}
          <div className="space-y-3">
            {/* AI Signals */}
            <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
              <div className="px-3 py-2 border-b border-white/5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <h2 className="font-medium text-white text-xs">AI Signals</h2>
                </div>
                <span className="text-[9px] text-gray-500">Top Opportunities</span>
              </div>
              <div className="divide-y divide-white/5" style={{ maxHeight: '150px', overflowY: 'auto' }}>
                {aiSignals.length === 0 ? (
                  <div className="p-3 text-gray-600 text-[10px] text-center">Scanning for signals...</div>
                ) : (
                  aiSignals.map((signal, i) => (
                    <div key={i} className="px-3 py-2 hover:bg-white/[0.02]">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-white text-xs">{signal.symbol.replace('USDT', '')}</span>
                          <span className={`text-[9px] px-1.5 py-0.5 rounded ${
                            signal.direction === 'LONG' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                          }`}>
                            {signal.direction}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-[9px] text-gray-500">Edge: {signal.edge.toFixed(1)}%</span>
                          <span className={`text-[9px] font-medium ${
                            signal.confidence >= 70 ? 'text-emerald-400' : signal.confidence >= 50 ? 'text-amber-400' : 'text-gray-400'
                          }`}>
                            {signal.confidence}%
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 mt-1 text-[9px] text-gray-500">
                        <span>Entry: ${signal.entry_price.toFixed(4)}</span>
                        <span className="text-emerald-400/70">TP: ${signal.target_price.toFixed(4)}</span>
                        <span className="text-red-400/70">SL: ${signal.stop_loss.toFixed(4)}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Market News */}
            <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
              <div className="px-3 py-2 border-b border-white/5 flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <h2 className="font-medium text-white text-xs">Market News</h2>
              </div>
              <div className="overflow-y-auto divide-y divide-white/5" style={{ maxHeight: '200px' }}>
                {news.length === 0 ? (
                  <div className="p-3 text-gray-600 text-[10px] text-center">Loading news...</div>
                ) : (
                  news.map((item, i) => (
                    <div key={i} className="px-2 py-1.5 hover:bg-white/[0.02]">
                      <div className="flex items-start gap-1.5">
                        <span className={`text-[8px] px-1 py-0.5 rounded flex-shrink-0 mt-0.5 ${
                          item.sentiment === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' :
                          item.sentiment === 'bearish' ? 'bg-red-500/20 text-red-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {item.sentiment === 'bullish' ? '↑' : item.sentiment === 'bearish' ? '↓' : '•'}
                        </span>
                        <div>
                          <p className="text-[10px] text-gray-300 leading-tight">{item.title}</p>
                          <p className="text-[8px] text-gray-600">{item.source}</p>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* P&L Performance Chart + Performance Stats */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 sm:gap-4 mb-4 sm:mb-6">
          {/* P&L Chart - Last 100 Trades */}
          <div className="lg:col-span-2 bg-white/[0.02] rounded-xl border border-white/5 p-3 sm:p-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 sm:mb-4 gap-1 sm:gap-0">
              <div className="flex items-center gap-2">
                <LineChart className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
                <h2 className="text-sm sm:text-base font-semibold text-white">P&L Performance</h2>
                <span className="text-[10px] sm:text-xs text-gray-500">(Last {recentTrades.length})</span>
              </div>
              <div className={`text-base sm:text-lg font-bold ${last100Pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {last100Pnl >= 0 ? '+' : ''}€{last100Pnl.toFixed(2)}
              </div>
            </div>
            
            <div className="h-52">
              {pnlHistory.length > 1 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={pnlHistory}>
                    <defs>
                      <linearGradient id="pnlGradientPos" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="pnlGradientNeg" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: '#6b7280', fontSize: 10 }} />
                    <YAxis 
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: '#6b7280', fontSize: 10 }}
                      tickFormatter={(v) => `€${v.toFixed(0)}`}
                      domain={['dataMin', 'dataMax']}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0d1321', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                      formatter={(value: any) => [`€${value.toFixed(2)}`, 'P&L']}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="pnl" 
                      stroke={last100Pnl >= 0 ? '#10b981' : '#ef4444'} 
                      strokeWidth={2}
                      fill={last100Pnl >= 0 ? 'url(#pnlGradientPos)' : 'url(#pnlGradientNeg)'}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-600 text-sm">
                  <div className="text-center">
                    <BarChart3 className="w-10 h-10 mx-auto mb-2 text-gray-700" />
                    <p>Complete more trades to see chart</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Performance Stats (from account start) */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 p-3 sm:p-5">
            <div className="flex items-center gap-2 mb-3 sm:mb-4">
              <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
              <h2 className="text-sm sm:text-base font-semibold text-white">Performance</h2>
              <span className="text-[10px] sm:text-xs text-gray-500">(All Time)</span>
            </div>
            
            <div className="space-y-4">
              {/* Win Rate Bar */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-500">Win Rate</span>
                  <span className={winRate >= 50 ? 'text-emerald-400' : 'text-red-400'}>{winRate.toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${winRate >= 50 ? 'bg-emerald-500' : 'bg-red-500'}`} style={{ width: `${Math.min(winRate, 100)}%` }} />
                </div>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">Best Trade</div>
                  <div className="text-sm font-semibold text-emerald-400">+€{(traderStats?.best_trade || 0).toFixed(2)}</div>
                </div>
                <div className="p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">Worst Trade</div>
                  <div className="text-sm font-semibold text-red-400">€{(traderStats?.worst_trade || 0).toFixed(2)}</div>
                </div>
                <div className="p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">Max Drawdown</div>
                  <div className="text-sm font-semibold text-amber-400">{(traderStats?.max_drawdown || 0).toFixed(2)}%</div>
                </div>
                <div className="p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">Scanned</div>
                  <div className="text-sm font-semibold text-cyan-400">{((traderStats?.opportunities_scanned || 0) / 1000).toFixed(0)}K</div>
                </div>
              </div>

              {/* Total P&L */}
              <div className="pt-3 border-t border-white/5">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Total P&L</span>
                  <span className={`text-xl font-bold ${totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {totalPnl >= 0 ? '+' : ''}€{totalPnl.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Row - Recent Trades, Console, Whales */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
          {/* Recent Trades */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="p-3 sm:p-4 border-b border-white/5 flex items-center gap-2">
              <Clock className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
              <h2 className="text-sm font-semibold text-white">Recent Trades</h2>
            </div>
            
            <div className="overflow-y-auto max-h-48 sm:max-h-56">
              {recentTrades.length === 0 ? (
                <div className="p-6 text-center text-gray-600 text-sm">No recent trades</div>
              ) : (
                recentTrades.slice(0, 10).map((trade, i) => (
                  <div key={i} className="px-3 py-2 border-b border-white/5 hover:bg-white/[0.02] flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-6 h-6 rounded flex items-center justify-center ${trade.pnl >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                        {trade.pnl >= 0 ? <CheckCircle className="w-3 h-3 text-emerald-400" /> : <XCircle className="w-3 h-3 text-red-400" />}
                      </div>
                      <span className="font-medium text-white text-xs">{trade.symbol.replace('USDT', '')}</span>
                    </div>
                    <div className="text-right">
                      <div className={`font-medium text-xs ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {trade.pnl >= 0 ? '+' : ''}€{trade.pnl.toFixed(2)}
                      </div>
                      <div className={`text-[9px] ${trade.pnl >= 0 ? 'text-emerald-400/60' : 'text-red-400/60'}`}>
                        {trade.pnl_percent >= 0 ? '+' : ''}{trade.pnl_percent.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* AI Console */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="px-3 sm:px-4 py-2 sm:py-3 border-b border-white/5 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isTrading ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'}`} />
              <h2 className="text-sm font-medium text-white">AI Console</h2>
            </div>
            <div ref={consoleRef} className="h-48 sm:h-56 overflow-y-auto p-2 sm:p-3 font-mono text-[9px] sm:text-[10px] space-y-1">
              {consoleLogs.length === 0 ? (
                <div className="text-gray-600">Waiting for activity...</div>
              ) : (
                consoleLogs.slice(0, 20).map((log, i) => (
                  <div key={i} className="flex gap-2 leading-tight">
                    <span className="text-gray-600 flex-shrink-0">{new Date(log.time).toLocaleTimeString()}</span>
                    <span className={`${
                      log.level === 'TRADE' ? 'text-emerald-400' :
                      log.level === 'SIGNAL' ? 'text-cyan-400' :
                      log.level === 'WARNING' ? 'text-amber-400' :
                      log.level === 'ERROR' ? 'text-red-400' :
                      'text-gray-400'
                    }`}>{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* AI Intelligence - REAL TIME */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="px-3 sm:px-4 py-2 sm:py-3 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-violet-400" />
                <h2 className="text-sm font-medium text-white">AI Intelligence</h2>
              </div>
              <span className={`text-[9px] px-2 py-0.5 rounded ${
                aiIntelligence.news_sentiment === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' :
                aiIntelligence.news_sentiment === 'bearish' ? 'bg-red-500/20 text-red-400' :
                'bg-gray-500/20 text-gray-400'
              }`}>
                {aiIntelligence.news_sentiment.toUpperCase()}
              </span>
            </div>
            <div className="p-3 space-y-3">
              {/* Status Grid */}
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2 bg-white/[0.02] rounded-lg">
                  <div className="text-[9px] text-gray-500 uppercase">Strategy</div>
                  <div className="text-xs font-semibold text-cyan-400">{aiIntelligence.strategy_mode}</div>
                </div>
                <div className="p-2 bg-white/[0.02] rounded-lg">
                  <div className="text-[9px] text-gray-500 uppercase">Breakouts</div>
                  <div className="text-xs font-semibold text-amber-400">{aiIntelligence.breakouts_detected}</div>
                </div>
              </div>
              
              {/* Last Action */}
              <div className="p-2 bg-white/[0.02] rounded-lg">
                <div className="text-[9px] text-gray-500 uppercase mb-1">Last Action</div>
                <div className="text-[10px] text-white truncate">{aiIntelligence.last_action}</div>
              </div>
              
              {/* Breakout Alerts */}
              {breakoutAlerts.length > 0 && (
                <div>
                  <div className="text-[9px] text-gray-500 uppercase mb-2">Active Breakouts</div>
                  <div className="space-y-1 max-h-24 overflow-y-auto">
                    {breakoutAlerts.slice(0, 5).map((alert, i) => (
                      <div key={i} className="flex items-center justify-between text-[10px] p-1.5 bg-white/[0.02] rounded">
                        <span className="text-white font-medium">{alert.symbol.replace('USDT', '')}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-gray-500">${alert.volume}M</span>
                          <span className={`font-semibold ${alert.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {alert.change >= 0 ? '+' : ''}{alert.change}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {breakoutAlerts.length === 0 && (
                <div className="text-center py-2 text-gray-600 text-[10px]">
                  No breakouts detected
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
