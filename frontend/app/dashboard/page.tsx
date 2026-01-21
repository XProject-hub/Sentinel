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
  Cpu,
  Trophy,
  Flame,
  CheckCircle,
  XCircle,
  Timer,
  Hash
} from 'lucide-react'
import Logo from '@/components/Logo'
import ConnectExchangePrompt from '@/components/ConnectExchangePrompt'
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'

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

interface TradeHistory {
  symbol: string
  pnl: number
  pnl_percent: number
  close_reason: string
  closed_time: string
  side?: string
}

interface TraderStats {
  total_trades: number
  winning_trades: number
  total_pnl: number
  max_drawdown: number
  opportunities_scanned: number
  win_rate: number
  avg_profit: number
  avg_loss: number
  profit_factor: number
  streak: number
  best_trade: number
  worst_trade: number
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
  const [recentTrades, setRecentTrades] = useState<TradeHistory[]>([])
  const [traderStats, setTraderStats] = useState<TraderStats | null>(null)
  const [pnlHistory, setPnlHistory] = useState<{time: string, pnl: number}[]>([])
  
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
      const [walletRes, positionsRes, statusRes, consoleRes, whaleRes, tradesRes, statsRes] = await Promise.all([
        fetch('/ai/exchange/balance?user_id=default'),
        fetch('/ai/exchange/positions?user_id=default'),
        fetch('/ai/exchange/trading/status?user_id=default'),
        fetch('/ai/exchange/trading/console?user_id=default&limit=20'),
        fetch('/ai/market/whale-alerts?limit=5'),
        fetch('/ai/exchange/trades/history?user_id=default&limit=10'),
        fetch('/ai/exchange/trading/stats?user_id=default')
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
        
        // Build P&L history from trades
        if (trades.length > 0) {
          let runningPnl = 0
          const history = trades.slice().reverse().map((t: TradeHistory, i: number) => {
            runningPnl += t.pnl
            return {
              time: new Date(t.closed_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
              pnl: runningPnl
            }
          })
          setPnlHistory(history)
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
          avg_profit: stats.avg_profit || (winningTrades > 0 ? totalPnl / winningTrades : 0),
          avg_loss: stats.avg_loss || 0,
          profit_factor: stats.profit_factor || 0,
          streak: stats.current_streak || 0,
          best_trade: stats.best_trade || 0,
          worst_trade: stats.worst_trade || 0
        })
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
      <div className="min-h-screen bg-[#060a13] flex items-center justify-center">
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
  const winRate = traderStats?.win_rate || 0

  return (
    <div className="min-h-screen bg-[#060a13]">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.02)_1px,transparent_1px)] bg-[size:44px_44px]" />
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-cyan-500/5 rounded-full blur-[120px]" />
      </div>

      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-[#060a13]/90 backdrop-blur-xl border-b border-white/5">
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
      <main className="p-6 relative">
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

        {/* Top Stats Grid - 6 columns */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
          {/* Total Equity */}
          <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <Wallet className="w-4 h-4 text-cyan-400" />
              <span className="text-xs text-gray-500">Equity</span>
            </div>
            <div className="text-xl font-bold text-white">
              {(wallet?.totalEquityUSDT || 0).toFixed(2)}
            </div>
            <div className="text-[10px] text-gray-600 mt-0.5">USDT</div>
          </div>

          {/* Available Balance */}
          <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="w-4 h-4 text-emerald-400" />
              <span className="text-xs text-gray-500">Available</span>
            </div>
            <div className="text-xl font-bold text-white">
              {(wallet?.availableBalanceUSDT || 0).toFixed(2)}
            </div>
            <div className="text-[10px] text-gray-600 mt-0.5">USDT</div>
          </div>

          {/* Daily P&L */}
          <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-gray-400" />
              <span className="text-xs text-gray-500">Daily P&L</span>
            </div>
            <div className={`text-xl font-bold ${dailyPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {dailyPnl >= 0 ? '+' : ''}{dailyPnl.toFixed(2)}
            </div>
            <div className="text-[10px] text-gray-600 mt-0.5">EUR</div>
          </div>

          {/* Win Rate */}
          <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <Trophy className="w-4 h-4 text-amber-400" />
              <span className="text-xs text-gray-500">Win Rate</span>
            </div>
            <div className={`text-xl font-bold ${winRate >= 50 ? 'text-emerald-400' : 'text-red-400'}`}>
              {winRate.toFixed(1)}%
            </div>
            <div className="text-[10px] text-gray-600 mt-0.5">
              {traderStats?.winning_trades || 0}W / {(traderStats?.total_trades || 0) - (traderStats?.winning_trades || 0)}L
            </div>
          </div>

          {/* Total Trades */}
          <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <Hash className="w-4 h-4 text-violet-400" />
              <span className="text-xs text-gray-500">Total Trades</span>
            </div>
            <div className="text-xl font-bold text-white">
              {traderStats?.total_trades || 0}
            </div>
            <div className={`text-[10px] mt-0.5 ${
              (traderStats?.streak || 0) > 0 ? 'text-emerald-400' : 
              (traderStats?.streak || 0) < 0 ? 'text-red-400' : 'text-gray-600'
            }`}>
              {(traderStats?.streak || 0) > 0 ? `${traderStats?.streak} win streak` :
               (traderStats?.streak || 0) < 0 ? `${Math.abs(traderStats?.streak || 0)} loss streak` : 'No streak'}
            </div>
          </div>

          {/* Trading Control */}
          <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-gray-400" />
              <span className="text-xs text-gray-500">Control</span>
            </div>
            <button
              onClick={isTrading ? stopTrading : startTrading}
              disabled={isTogglingBot}
              className={`w-full py-2 rounded-lg font-semibold text-xs flex items-center justify-center gap-1.5 transition-all ${
                isTrading 
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30' 
                  : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/30'
              }`}
            >
              {isTogglingBot ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : isTrading ? (
                <>
                  <Square className="w-3 h-3" />
                  Stop
                </>
              ) : (
                <>
                  <Play className="w-3 h-3" />
                  Start
                </>
              )}
            </button>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-3 gap-4 mb-6">
          {/* P&L Chart */}
          <div className="lg:col-span-2 bg-white/[0.02] rounded-xl border border-white/5 p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <LineChart className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">P&L Performance</h2>
              </div>
              <div className={`text-lg font-bold ${totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {totalPnl >= 0 ? '+' : ''}{formatCurrency(totalPnl)}
              </div>
            </div>
            
            <div className="h-48">
              {pnlHistory.length > 1 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={pnlHistory}>
                    <defs>
                      <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={totalPnl >= 0 ? '#10b981' : '#ef4444'} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={totalPnl >= 0 ? '#10b981' : '#ef4444'} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis 
                      dataKey="time" 
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: '#6b7280', fontSize: 10 }}
                    />
                    <YAxis 
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: '#6b7280', fontSize: 10 }}
                      tickFormatter={(v) => `€${v.toFixed(0)}`}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#0d1321', 
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        fontSize: '12px'
                      }}
                      formatter={(value: any) => [`€${value.toFixed(2)}`, 'P&L']}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="pnl" 
                      stroke={totalPnl >= 0 ? '#10b981' : '#ef4444'} 
                      strokeWidth={2}
                      fill="url(#pnlGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-600 text-sm">
                  <div className="text-center">
                    <BarChart3 className="w-10 h-10 mx-auto mb-2 text-gray-700" />
                    <p>No trade history yet</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Performance Stats */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 p-5">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              <h2 className="font-semibold text-white">Performance</h2>
            </div>
            
            <div className="space-y-4">
              {/* Win Rate Visual */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-500">Win Rate</span>
                  <span className={winRate >= 50 ? 'text-emerald-400' : 'text-red-400'}>{winRate.toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all ${winRate >= 50 ? 'bg-emerald-500' : 'bg-red-500'}`}
                    style={{ width: `${Math.min(winRate, 100)}%` }}
                  />
                </div>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">Best Trade</div>
                  <div className="text-sm font-semibold text-emerald-400">
                    +{formatCurrency(traderStats?.best_trade || 0)}
                  </div>
                </div>
                <div className="p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">Worst Trade</div>
                  <div className="text-sm font-semibold text-red-400">
                    {formatCurrency(traderStats?.worst_trade || 0)}
                  </div>
                </div>
                <div className="p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">Max Drawdown</div>
                  <div className="text-sm font-semibold text-amber-400">
                    {(traderStats?.max_drawdown || 0).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">Scanned</div>
                  <div className="text-sm font-semibold text-cyan-400">
                    {((traderStats?.opportunities_scanned || 0) / 1000).toFixed(0)}K
                  </div>
                </div>
              </div>

              {/* Total P&L */}
              <div className="pt-3 border-t border-white/5">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Total P&L</span>
                  <span className={`text-lg font-bold ${(traderStats?.total_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {(traderStats?.total_pnl || 0) >= 0 ? '+' : ''}{formatCurrency(traderStats?.total_pnl || 0)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Middle Row - Open Positions & Market News */}
        <div className="grid lg:grid-cols-4 gap-4 mb-6">
          {/* Open Positions */}
          <div className="lg:col-span-3 bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="p-4 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">Open Positions</h2>
                <span className="text-xs text-gray-500 ml-1">({positions.length})</span>
              </div>
              <button 
                onClick={loadDashboardData}
                className="p-1.5 rounded-lg hover:bg-white/5 transition-colors"
              >
                <RefreshCw className="w-4 h-4 text-gray-500" />
              </button>
            </div>
            
            {positions.length === 0 ? (
              <div className="p-8 text-center">
                <Target className="w-10 h-10 text-gray-700 mx-auto mb-3" />
                <p className="text-gray-500 text-sm">No open positions</p>
                <p className="text-gray-600 text-xs mt-1">AI is scanning...</p>
              </div>
            ) : (
              <div className="overflow-x-auto max-h-72">
                <table className="w-full">
                  <thead className="sticky top-0 bg-[#060a13]">
                    <tr className="border-b border-white/5">
                      <th className="text-left text-[10px] font-medium text-gray-500 px-4 py-2">PAIR</th>
                      <th className="text-left text-[10px] font-medium text-gray-500 px-4 py-2">SIDE</th>
                      <th className="text-right text-[10px] font-medium text-gray-500 px-4 py-2">VALUE</th>
                      <th className="text-right text-[10px] font-medium text-gray-500 px-4 py-2">ENTRY</th>
                      <th className="text-right text-[10px] font-medium text-gray-500 px-4 py-2">MARK</th>
                      <th className="text-right text-[10px] font-medium text-gray-500 px-4 py-2">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((pos, i) => {
                      const entryPrice = parseFloat(pos.entryPrice || '0')
                      const markPrice = parseFloat(pos.markPrice || '0')
                      const posValue = parseFloat(pos.positionValue || '0')
                      const leverage = parseFloat(pos.leverage || '1')
                      
                      // Calculate P&L percentage based on price change and side
                      let pnlPercent = 0
                      if (entryPrice > 0 && markPrice > 0) {
                        if (pos.side === 'Buy') {
                          pnlPercent = ((markPrice - entryPrice) / entryPrice) * 100 * leverage
                        } else {
                          pnlPercent = ((entryPrice - markPrice) / entryPrice) * 100 * leverage
                        }
                      }
                      
                      // Calculate actual P&L in EUR from position value and percentage
                      const pnlEur = posValue * (pnlPercent / 100)
                      
                      // If API provides unrealisedPnl, use it instead
                      const apiPnl = parseFloat(pos.unrealisedPnl || '0')
                      const finalPnl = apiPnl !== 0 ? apiPnl : pnlEur
                      const finalPnlPercent = apiPnl !== 0 && posValue > 0 ? (apiPnl / posValue) * 100 : pnlPercent
                      
                      return (
                        <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                          <td className="px-4 py-3">
                            <span className="font-medium text-white text-sm">{pos.symbol.replace('USDT', '')}</span>
                            <span className="text-gray-600 text-xs">/USDT</span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                              pos.side === 'Buy' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                            }`}>
                              {pos.side === 'Buy' ? 'LONG' : 'SHORT'} {pos.leverage}x
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span className="text-sm text-white font-medium">€{posValue.toFixed(2)}</span>
                          </td>
                          <td className="px-4 py-3 text-right text-sm text-gray-400">
                            €{entryPrice.toFixed(4)}
                          </td>
                          <td className="px-4 py-3 text-right text-sm text-gray-400">
                            €{markPrice.toFixed(4)}
                          </td>
                          <td className="px-4 py-3 text-right">
                            <div className={`text-sm font-medium ${finalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {finalPnl >= 0 ? '+' : ''}€{finalPnl.toFixed(2)}
                            </div>
                            <div className={`text-[10px] ${finalPnlPercent >= 0 ? 'text-emerald-400/60' : 'text-red-400/60'}`}>
                              {finalPnlPercent >= 0 ? '+' : ''}{finalPnlPercent.toFixed(2)}%
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

          {/* Market News - Same row as Open Positions */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <h2 className="font-medium text-white text-sm">Market News</h2>
            </div>
            
            <div className="max-h-72 overflow-y-auto divide-y divide-white/5">
              {news.length === 0 ? (
                <div className="p-4 text-gray-600 text-xs text-center">Loading news...</div>
              ) : (
                news.slice(0, 10).map((item, i) => (
                  <div key={i} className="px-3 py-2 hover:bg-white/[0.02]">
                    <div className="flex items-start gap-2">
                      <span className={`text-[9px] px-1.5 py-0.5 rounded flex-shrink-0 mt-0.5 ${
                        item.sentiment === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' :
                        item.sentiment === 'bearish' ? 'bg-red-500/20 text-red-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {item.sentiment === 'bullish' ? '↑' : item.sentiment === 'bearish' ? '↓' : '•'}
                      </span>
                      <p className="text-[10px] text-gray-300 leading-tight line-clamp-2">{item.title}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Bottom Row - Recent Trades, Console, Whales */}
        <div className="grid lg:grid-cols-3 gap-4">
          {/* Recent Trades (Last 10) */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="p-4 border-b border-white/5 flex items-center gap-2">
              <Clock className="w-5 h-5 text-cyan-400" />
              <h2 className="font-semibold text-white text-sm">Recent Trades</h2>
              <span className="text-xs text-gray-500 ml-1">(10)</span>
            </div>
            
            {recentTrades.length === 0 ? (
              <div className="p-6 text-center">
                <Clock className="w-8 h-8 text-gray-700 mx-auto mb-2" />
                <p className="text-gray-500 text-xs">No recent trades</p>
              </div>
            ) : (
              <div className="overflow-y-auto max-h-48">
                {recentTrades.slice(0, 10).map((trade, i) => (
                  <div key={i} className="px-3 py-2 border-b border-white/5 hover:bg-white/[0.02] flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-6 h-6 rounded flex items-center justify-center ${
                        trade.pnl >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'
                      }`}>
                        {trade.pnl >= 0 ? (
                          <CheckCircle className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <XCircle className="w-3 h-3 text-red-400" />
                        )}
                      </div>
                      <div>
                        <span className="font-medium text-white text-xs">{trade.symbol.replace('USDT', '')}</span>
                      </div>
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
                ))}
              </div>
            )}
          </div>

          {/* AI Console */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isTrading ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'}`} />
              <h2 className="font-medium text-white text-sm">AI Console</h2>
            </div>
            
            <div ref={consoleRef} className="h-48 overflow-y-auto p-3 font-mono text-[10px] space-y-1">
              {consoleLogs.length === 0 ? (
                <div className="text-gray-600">Waiting for activity...</div>
              ) : (
                consoleLogs.slice(0, 15).map((log, i) => (
                  <div key={i} className="flex gap-2 leading-tight">
                    <span className="text-gray-600 flex-shrink-0">
                      {new Date(log.time).toLocaleTimeString()}
                    </span>
                    <span className={`${
                      log.level === 'TRADE' ? 'text-emerald-400' :
                      log.level === 'SIGNAL' ? 'text-cyan-400' :
                      log.level === 'WARNING' ? 'text-amber-400' :
                      log.level === 'ERROR' ? 'text-red-400' :
                      'text-gray-400'
                    }`}>
                      {log.message}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Whale Activity */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
              <Waves className="w-4 h-4 text-cyan-400" />
              <h2 className="font-medium text-white text-sm">Whale Activity</h2>
            </div>
            
            <div className="h-48 overflow-y-auto divide-y divide-white/5">
              {whaleAlerts.length === 0 ? (
                <div className="p-4 text-gray-600 text-xs text-center">No whale activity detected</div>
              ) : (
                whaleAlerts.slice(0, 8).map((alert, i) => (
                  <div key={i} className="px-3 py-2 hover:bg-white/[0.02]">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-white text-xs font-medium">{alert.symbol.replace('USDT', '')}</span>
                        <span className={`text-[9px] px-1.5 py-0.5 rounded ${
                          alert.type === 'buy_wall' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                        }`}>
                          {alert.type === 'buy_wall' ? 'BUY' : 'SELL'}
                        </span>
                      </div>
                      <span className="text-[9px] text-gray-600">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
