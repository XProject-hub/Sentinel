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
import MaintenanceNotification from '@/components/MaintenanceNotification'
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, Line, LineChart as RechartsLineChart } from 'recharts'

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

interface CoinBalance {
  coin: string
  balance: number
  usdValue: number
  free?: number
  locked?: number
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
  coins?: CoinBalance[]
  exchange?: string
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

interface SymbolTradeEvent {
  action: string
  timestamp: string
  side: string | null
  entry_price: number | null
  size: number | null
  position_value: number | null
  leverage: number
  trade_mode: string | null
  is_spot: boolean
  edge_score: number | null
  confidence: number | null
  reason: string
  pnl_percent: number | null
  pnl_value: number | null
}

interface SymbolHistoryData {
  symbol: string
  history: SymbolTradeEvent[]
  completedTrades: TradeHistory[]
  stats: {
    totalTrades: number
    totalPnl: number
    winRate: number
    wins: number
    losses: number
  }
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
  const [priceCharts, setPriceCharts] = useState<{[symbol: string]: {close: number}[]}>({})
  const [showAssetsPopup, setShowAssetsPopup] = useState(false)
  const [showHistoryModal, setShowHistoryModal] = useState(false)
  const [selectedSymbolHistory, setSelectedSymbolHistory] = useState<SymbolHistoryData | null>(null)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [showChartModal, setShowChartModal] = useState(false)
  const [chartModalSymbol, setChartModalSymbol] = useState<string>('')
  const [chartModalData, setChartModalData] = useState<{time: number, open: number, high: number, low: number, close: number, volume: number}[]>([])
  const [loadingChart, setLoadingChart] = useState(false)
  const [chartModalTicker, setChartModalTicker] = useState<{price: number, change24h: number, high24h: number, low24h: number, volume24h: number} | null>(null)
  
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
      // Admin uses 'default' as user_id in trading system
      const isAdminUser = user.email === 'admin@sentinel.ai'
      const currentUserId = isAdminUser ? 'default' : (user.id || user.userId || user.user_id || 'default')
      setUserId(currentUserId)
      setIsAdmin(isAdminUser)
      
      // Check if user has exchange connection
      // For admin, also verify with AI service that credentials exist
      let hasConnection = false
      
      if (isAdminUser) {
        // Admin: check AI service for credentials
        try {
          const aiStatusRes = await fetch(`/ai/exchange/trading/status?user_id=${currentUserId}`)
          if (aiStatusRes.ok) {
            const aiStatus = await aiStatusRes.json()
            // If we get a valid response and it doesn't say "not connected", assume connected
            // Also check balance endpoint to verify credentials work
            const balanceRes = await fetch(`/ai/exchange/balance?user_id=${currentUserId}`)
            if (balanceRes.ok) {
              const balanceData = await balanceRes.json()
              // If balance returns an error about no connection, credentials are missing
              hasConnection = balanceData.success !== false || balanceData.error !== "No exchange connected for this user"
            }
          }
        } catch (e) {
          console.error('Failed to check admin connection:', e)
          hasConnection = false
        }
      } else {
        // Regular users: check Laravel backend
        const response = await fetch('/api/exchanges', {
          headers: { 'Authorization': `Bearer ${token}` }
        })
        
        if (response.ok) {
          const data = await response.json()
          hasConnection = data.data?.length > 0
        }
      }
      
      setHasExchangeConnection(hasConnection)
      
      if (hasConnection) {
        loadDashboardData(currentUserId)
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
        const newPositions = data.data?.positions || []
        setPositions(newPositions)
        
        // Fetch sparkline data for all position symbols
        if (newPositions.length > 0) {
          const symbols = newPositions.map((p: Position) => p.symbol).join(',')
          try {
            const klinesRes = await fetch(`/ai/exchange/klines-batch?symbols=${symbols}&interval=5&limit=20&user_id=${uid}`)
            if (klinesRes.ok) {
              const klinesData = await klinesRes.json()
              if (klinesData.success) {
                setPriceCharts(klinesData.data || {})
              }
            }
          } catch (e) {
            console.warn('Failed to fetch sparkline data:', e)
          }
        }
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
        // Convert to EUR for display
        if (trades.length > 0) {
          let runningPnl = 0
          const sortedTrades = [...trades].reverse()
          const history = sortedTrades.map((t: TradeHistory) => {
            runningPnl += t.pnl * USDT_TO_EUR  // Convert to EUR
            return {
              time: new Date(t.closed_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
              pnl: runningPnl
            }
          })
          setPnlHistory(history)
          // Set P&L sum from last 100 trades (in EUR)
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
        const signalsRes = await fetch(`/ai/exchange/signals?user_id=${uid}&limit=5`)
        if (signalsRes.ok) {
          const data = await signalsRes.json()
          setAiSignals(data.signals || [])
        }
      } catch {}

      // Get AI Intelligence status (breakouts, news sentiment, etc.) - PER USER
      try {
        const aiRes = await fetch(`/ai/exchange/intelligence?user_id=${uid}`)
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
      console.log('Starting trading for user:', userId)
      const response = await fetch(`/ai/exchange/trading/resume?user_id=${userId}`, { method: 'POST' })
      const result = await response.json()
      console.log('Start trading response:', result)
      
      if (result.success) {
        setTradingStatus(prev => prev ? { ...prev, is_autonomous_trading: true, is_paused: false } : null)
      } else {
        console.error('Start trading failed:', result.error)
        alert(`Failed to start trading: ${result.error || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Failed to start:', error)
      alert('Failed to start trading - check console for details')
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
      const response = await fetch(`/ai/exchange/close-position/${symbol}?user_id=${userId}`, {
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
      const response = await fetch(`/ai/exchange/close-all-positions?user_id=${userId}`, {
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

  const openChartModal = async (newsTitle: string) => {
    // Extract symbol from news title (e.g., "BTR surges +87.9% in 24h" -> "BTRUSDT")
    const match = newsTitle.match(/^(\w+)\s+(surges|drops|pumps|dumps|rises|falls)/i)
    if (!match) return
    
    const symbol = match[1].toUpperCase() + 'USDT'
    setChartModalSymbol(symbol)
    setShowChartModal(true)
    setLoadingChart(true)
    
    try {
      // Fetch kline data for chart (1h candles, last 100)
      const klinesRes = await fetch(`/ai/exchange/klines/${symbol}?interval=60&limit=100&user_id=${userId}`)
      if (klinesRes.ok) {
        const klinesData = await klinesRes.json()
        if (klinesData.success && klinesData.data?.prices) {
          setChartModalData(klinesData.data.prices)
        }
      }
      
      // Fetch ticker data for current price info
      const tickerRes = await fetch(`/ai/exchange/ticker/${symbol}?user_id=${userId}`)
      if (tickerRes.ok) {
        const tickerData = await tickerRes.json()
        if (tickerData.success && tickerData.data) {
          setChartModalTicker({
            price: parseFloat(tickerData.data.lastPrice || '0'),
            change24h: parseFloat(tickerData.data.price24hPcnt || '0') * 100,
            high24h: parseFloat(tickerData.data.highPrice24h || '0'),
            low24h: parseFloat(tickerData.data.lowPrice24h || '0'),
            volume24h: parseFloat(tickerData.data.volume24h || '0')
          })
        }
      }
    } catch (error) {
      console.error('Failed to fetch chart data:', error)
    } finally {
      setLoadingChart(false)
    }
  }

  const fetchSymbolHistory = async (symbol: string) => {
    setLoadingHistory(true)
    setShowHistoryModal(true)
    try {
      const response = await fetch(`/ai/exchange/trade-history/${symbol}?user_id=${userId}`)
      if (response.ok) {
        const result = await response.json()
        if (result.success) {
          setSelectedSymbolHistory(result.data)
        }
      }
    } catch (error) {
      console.error('Failed to fetch symbol history:', error)
    } finally {
      setLoadingHistory(false)
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

  if (!hasExchangeConnection) {
    return <ConnectExchangePrompt />
  }

  const balanceUSDT = wallet?.totalEquityUSDT || wallet?.totalEquity || 0
  const balanceEUR = balanceUSDT * USDT_TO_EUR
  // Daily P&L comes from backend in USDT, convert to EUR for display
  const dailyPnlUSDT = wallet?.dailyPnL || 0
  const dailyPnl = dailyPnlUSDT * USDT_TO_EUR
  // Total P&L also in USDT, convert to EUR
  const totalPnlUSDT = traderStats?.total_pnl || 0
  const totalPnl = totalPnlUSDT * USDT_TO_EUR
  const isTrading = tradingStatus?.is_autonomous_trading && !tradingStatus?.is_paused
  const winRate = traderStats?.win_rate || 0
  
  // Determine primary asset to display (for Binance users with BTC, etc.)
  const coins = wallet?.coins || []
  const exchange = wallet?.exchange || 'bybit'
  const primaryCoin = coins.length > 0 
    ? coins.reduce((max, coin) => coin.usdValue > max.usdValue ? coin : max, coins[0])
    : null
  const balanceDisplay = primaryCoin 
    ? `${primaryCoin.balance.toFixed(primaryCoin.coin === 'BTC' ? 6 : primaryCoin.coin === 'ETH' ? 4 : 2)} ${primaryCoin.coin}`
    : `${balanceUSDT.toFixed(2)} USDT`

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
        {/* Maintenance Notification */}
        <MaintenanceNotification />
        
        {/* Top Stats - Responsive grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2 sm:gap-4 mb-4 sm:mb-6">
          {/* Balance */}
          <div className="p-3 sm:p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 sm:gap-2 mb-1 sm:mb-2">
              <Wallet className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-cyan-400" />
              <span className="text-[10px] sm:text-xs text-gray-500">Balance</span>
              {exchange === 'binance' && (
                <span className="text-[8px] px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 rounded">BINANCE</span>
              )}
            </div>
            <div className="text-base sm:text-xl font-bold text-white">{balanceDisplay}</div>
            <div className="flex items-center gap-1 text-xs sm:text-sm text-gray-400 mt-0.5 sm:mt-1">
              <Euro className="w-2.5 h-2.5 sm:w-3 sm:h-3" />
              <span>≈ €{balanceEUR.toFixed(2)} EUR</span>
            </div>
            {coins.length > 1 && (
              <div className="relative">
                <button 
                  onClick={() => setShowAssetsPopup(!showAssetsPopup)}
                  className="text-[9px] text-cyan-400 mt-1 hover:text-cyan-300 cursor-pointer underline"
                >
                  +{coins.length - 1} more asset{coins.length > 2 ? 's' : ''} ▼
                </button>
                {showAssetsPopup && (
                  <div className="absolute top-6 left-0 z-50 bg-gray-900 border border-white/10 rounded-lg p-3 shadow-xl min-w-[200px]">
                    <div className="text-[10px] text-gray-400 mb-2 font-semibold">All Assets:</div>
                    {coins.map((coin, idx) => (
                      <div key={idx} className="flex justify-between items-center py-1.5 border-b border-white/5 last:border-0">
                        <span className="text-white text-xs font-medium">{coin.coin}</span>
                        <div className="text-right">
                          <div className="text-white text-xs">{coin.balance.toFixed(coin.coin === 'USDT' ? 2 : 4)}</div>
                          <div className="text-gray-500 text-[9px]">≈ ${coin.usdValue.toFixed(2)}</div>
                        </div>
                      </div>
                    ))}
                    <div className="mt-2 pt-2 border-t border-white/10 flex justify-between">
                      <span className="text-gray-400 text-[10px]">Total:</span>
                      <span className="text-emerald-400 text-xs font-bold">${coins.reduce((sum, c) => sum + c.usdValue, 0).toFixed(2)}</span>
                    </div>
                    <button 
                      onClick={() => setShowAssetsPopup(false)}
                      className="mt-2 w-full text-[9px] text-gray-500 hover:text-white"
                    >
                      Close
                    </button>
                  </div>
                )}
              </div>
            )}
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
                {positions.length > 0 && (() => {
                  // Calculate total unrealized P&L across all positions
                  const totalPnl = positions.reduce((sum, pos) => {
                    const pnl = parseFloat(pos.estimatedNetPnl || pos.unrealisedPnl || '0')
                    return sum + pnl
                  }, 0)
                  const totalPnlEur = totalPnl * 0.92 // Convert to EUR
                  const isPositive = totalPnlEur >= 0
                  
                  return (
                    <span className={`ml-2 px-2 py-0.5 rounded-full text-[10px] sm:text-xs font-bold ${
                      isPositive 
                        ? 'bg-emerald-500/20 text-emerald-400' 
                        : 'bg-red-500/20 text-red-400'
                    }`}>
                      {isPositive ? '+' : ''}€{totalPnlEur.toFixed(2)}
                    </span>
                  )
                })()}
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
                      <th className="text-center text-[10px] font-medium text-gray-500 px-1 py-2" style={{width: '100px'}}>TREND</th>
                      <th className="text-left text-[10px] font-medium text-gray-500 px-3 py-2">SIDE</th>
                      <th className="text-right text-[10px] font-medium text-gray-500 px-3 py-2">SIZE / MARGIN</th>
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
                      
                      // Calculate position value: positionValue from Bybit IS the notional value (size * price)
                      // So we should NOT multiply by leverage again!
                      let notionalUSDT = parseFloat(pos.positionValue || '0')
                      if (notionalUSDT === 0 && markPrice > 0 && size > 0) {
                        notionalUSDT = size * markPrice  // This IS the notional (full exposure)
                      }
                      const notionalEUR = notionalUSDT * USDT_TO_EUR
                      const marginUSDT = notionalUSDT / leverage  // Margin = notional / leverage
                      const marginEUR = notionalEUR / leverage
                      // For P&L calculations, use notional (full position)
                      const posValueUSDT = notionalUSDT
                      const posValueEUR = notionalEUR
                      
                      // Calculate P&L percentage (on notional value, NO leverage multiplier!)
                      // Notional already includes leverage, so P&L = notional * price_change
                      let pnlPercent = 0
                      if (entryPrice > 0 && markPrice > 0) {
                        if (pos.side === 'Buy') {
                          pnlPercent = ((markPrice - entryPrice) / entryPrice) * 100
                        } else {
                          pnlPercent = ((entryPrice - markPrice) / entryPrice) * 100
                        }
                      }
                      
                      // Calculate P&L in EUR: notional * price_change
                      // This automatically includes leverage because notional = margin * leverage
                      const pnlEUR = notionalEUR * (pnlPercent / 100)
                      
                      // Use NET P&L (after fees) if available, otherwise gross
                      const netPnl = parseFloat(pos.estimatedNetPnl || '0')
                      const grossPnl = parseFloat(pos.unrealisedPnl || '0')
                      
                      // Prefer NET P&L (more accurate), fallback to gross
                      const apiPnl = netPnl !== 0 ? netPnl : grossPnl
                      const finalPnlEUR = apiPnl !== 0 ? apiPnl * USDT_TO_EUR : pnlEUR
                      // Show percentage as return on MARGIN (leveraged return), not on notional
                      const finalPnlPercent = apiPnl !== 0 && marginUSDT > 0 ? (apiPnl / marginUSDT) * 100 : pnlPercent * leverage
                      
                      return (
                        <tr 
                          key={i} 
                          className="border-b border-white/5 hover:bg-white/[0.02] cursor-pointer transition-colors"
                          onClick={() => fetchSymbolHistory(pos.symbol)}
                          title="Click to view trade history"
                        >
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
                          {/* Mini Sparkline Chart with Trend */}
                          <td className="px-1 py-1" style={{width: '100px'}}>
                            {priceCharts[pos.symbol] && priceCharts[pos.symbol].length > 1 ? (() => {
                              const rawData = priceCharts[pos.symbol]
                              const isUp = finalPnlEUR >= 0
                              const gradientId = `gradient-${pos.symbol.replace(/[^a-zA-Z]/g, '')}-${i}`
                              
                              // Normalize data to make chart more visible
                              const prices = rawData.map(d => d.close)
                              const minPrice = Math.min(...prices)
                              const maxPrice = Math.max(...prices)
                              const range = maxPrice - minPrice || 1
                              
                              // Create normalized data (0-100 scale)
                              const chartData = rawData.map(d => ({
                                value: ((d.close - minPrice) / range) * 100
                              }))
                              
                              // Calculate trend: compare first half avg to second half avg
                              const firstHalf = prices.slice(0, Math.floor(prices.length / 2))
                              const secondHalf = prices.slice(Math.floor(prices.length / 2))
                              const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length
                              const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length
                              const trendUp = secondAvg > firstAvg
                              const trendPercent = ((secondAvg - firstAvg) / firstAvg * 100).toFixed(2)
                              
                              return (
                                <div className="flex items-center gap-1">
                                  {/* Trend Arrow */}
                                  <div className={`flex flex-col items-center justify-center w-4 ${trendUp ? 'text-emerald-400' : 'text-red-400'}`}>
                                    <span className="text-[10px] font-bold">{trendUp ? '▲' : '▼'}</span>
                                  </div>
                                  {/* Chart */}
                                  <div className="relative flex-1">
                                    <ResponsiveContainer width={65} height={28}>
                                      <AreaChart data={chartData} margin={{top: 2, right: 0, bottom: 2, left: 0}}>
                                        <defs>
                                          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor={isUp ? '#10b981' : '#ef4444'} stopOpacity={0.5}/>
                                            <stop offset="100%" stopColor={isUp ? '#10b981' : '#ef4444'} stopOpacity={0.1}/>
                                          </linearGradient>
                                        </defs>
                                        <Area 
                                          type="monotone" 
                                          dataKey="value" 
                                          stroke={isUp ? '#10b981' : '#ef4444'}
                                          strokeWidth={1.5}
                                          fill={`url(#${gradientId})`}
                                          isAnimationActive={false}
                                        />
                                      </AreaChart>
                                    </ResponsiveContainer>
                                  </div>
                                </div>
                              )
                            })() : (
                              <div className="w-[90px] h-[28px] flex items-center justify-center gap-1">
                                <div className="text-gray-600 text-[10px]">—</div>
                                <div className="flex gap-0.5">
                                  <div className="w-1 h-2 bg-gray-700 rounded animate-pulse"></div>
                                  <div className="w-1 h-3 bg-gray-700 rounded animate-pulse"></div>
                                  <div className="w-1 h-2 bg-gray-700 rounded animate-pulse"></div>
                                </div>
                              </div>
                            )}
                          </td>
                          <td className="px-3 py-2">
                            <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                              pos.side === 'Buy' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                            }`}>
                              {pos.side === 'Buy' ? 'LONG' : 'SHORT'} {pos.leverage}x
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right">
                            {/* Position SIZE = notional value (actual market exposure) */}
                            <div className="text-white font-medium text-sm">€{notionalEUR.toFixed(0)}</div>
                            {/* MARGIN = collateral/deposit for this trade */}
                            <div className="text-gray-500 text-[9px]">margin: €{marginEUR.toFixed(2)}</div>
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
                    <div 
                      key={i} 
                      className="px-2 py-1.5 hover:bg-white/[0.02] cursor-pointer transition-colors"
                      onClick={() => openChartModal(item.title)}
                      title="Click to view chart"
                    >
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
                          <p className="text-[8px] text-gray-600">{item.source} • Click for chart</p>
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
                  <div className="text-sm font-semibold text-emerald-400">+€{((traderStats?.best_trade || 0) * USDT_TO_EUR).toFixed(2)}</div>
                </div>
                <div className="p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">Worst Trade</div>
                  <div className="text-sm font-semibold text-red-400">€{((traderStats?.worst_trade || 0) * USDT_TO_EUR).toFixed(2)}</div>
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
                recentTrades.slice(0, 10).map((trade, i) => {
                  const pnlEur = trade.pnl * USDT_TO_EUR
                  return (
                    <div key={i} className="px-3 py-2 border-b border-white/5 hover:bg-white/[0.02] flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className={`w-6 h-6 rounded flex items-center justify-center ${trade.pnl >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                          {trade.pnl >= 0 ? <CheckCircle className="w-3 h-3 text-emerald-400" /> : <XCircle className="w-3 h-3 text-red-400" />}
                        </div>
                        <span className="font-medium text-white text-xs">{trade.symbol.replace('USDT', '')}</span>
                      </div>
                      <div className="text-right">
                        <div className={`font-medium text-xs ${trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {pnlEur >= 0 ? '+' : ''}€{pnlEur.toFixed(2)}
                        </div>
                        <div className={`text-[9px] ${trade.pnl >= 0 ? 'text-emerald-400/60' : 'text-red-400/60'}`}>
                          {trade.pnl_percent >= 0 ? '+' : ''}{trade.pnl_percent.toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  )
                })
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

      {/* Trade History Modal */}
      {showHistoryModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-2xl max-h-[80vh] overflow-hidden"
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-slate-700 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <BarChart3 className="w-5 h-5 text-cyan-400" />
                <h3 className="text-lg font-semibold text-white">
                  {selectedSymbolHistory?.symbol.replace('USDT', '')}/USDT Trade History
                </h3>
              </div>
              <button
                onClick={() => {
                  setShowHistoryModal(false)
                  setSelectedSymbolHistory(null)
                }}
                className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
              >
                <XCircle className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-4 overflow-y-auto max-h-[calc(80vh-80px)]">
              {loadingHistory ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                </div>
              ) : selectedSymbolHistory ? (
                <>
                  {/* Stats Summary */}
                  <div className="grid grid-cols-4 gap-3 mb-6">
                    <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                      <div className="text-2xl font-bold text-white">{selectedSymbolHistory.stats.totalTrades}</div>
                      <div className="text-xs text-gray-400">Total Trades</div>
                    </div>
                    <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                      <div className={`text-2xl font-bold ${selectedSymbolHistory.stats.totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {selectedSymbolHistory.stats.totalPnl >= 0 ? '+' : ''}€{(selectedSymbolHistory.stats.totalPnl * USDT_TO_EUR).toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-400">Total P&L</div>
                    </div>
                    <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                      <div className="text-2xl font-bold text-emerald-400">{selectedSymbolHistory.stats.wins}</div>
                      <div className="text-xs text-gray-400">Wins</div>
                    </div>
                    <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                      <div className="text-2xl font-bold text-red-400">{selectedSymbolHistory.stats.losses}</div>
                      <div className="text-xs text-gray-400">Losses</div>
                    </div>
                  </div>

                  {/* Trade Events List */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-gray-400 mb-3">Recent Activity</h4>
                    {selectedSymbolHistory.history.length === 0 ? (
                      <div className="text-center py-8 text-gray-500">
                        No trade history yet for this symbol
                      </div>
                    ) : (
                      selectedSymbolHistory.history.map((event, idx) => (
                        <div 
                          key={idx}
                          className={`p-3 rounded-lg border ${
                            event.action === 'opened' 
                              ? 'bg-blue-500/10 border-blue-500/30' 
                              : event.pnl_percent && event.pnl_percent >= 0
                                ? 'bg-emerald-500/10 border-emerald-500/30'
                                : 'bg-red-500/10 border-red-500/30'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                                event.action === 'opened' ? 'bg-blue-500/20 text-blue-400' :
                                event.pnl_percent && event.pnl_percent >= 0 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                              }`}>
                                {event.action.toUpperCase()}
                              </span>
                              <span className="text-white font-medium">{event.side}</span>
                              {event.is_spot && (
                                <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 text-[10px]">SPOT</span>
                              )}
                            </div>
                            <span className="text-xs text-gray-400">
                              {new Date(event.timestamp).toLocaleString()}
                            </span>
                          </div>
                          <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                            {event.entry_price && (
                              <div>
                                <span className="text-gray-500">Entry:</span>
                                <span className="text-white ml-1">${event.entry_price.toFixed(4)}</span>
                              </div>
                            )}
                            {event.position_value && (
                              <div>
                                <span className="text-gray-500">Value:</span>
                                <span className="text-white ml-1">${event.position_value.toFixed(0)}</span>
                              </div>
                            )}
                            {event.edge_score && (
                              <div>
                                <span className="text-gray-500">Edge:</span>
                                <span className="text-cyan-400 ml-1">{event.edge_score}</span>
                              </div>
                            )}
                          </div>
                          {event.action === 'closed' && event.pnl_percent !== null && (
                            <div className="mt-2 flex items-center gap-4 text-sm">
                              <span className={`font-bold ${event.pnl_percent >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {event.pnl_percent >= 0 ? '+' : ''}{event.pnl_percent.toFixed(2)}%
                              </span>
                              {event.pnl_value !== null && (
                                <span className={event.pnl_value >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                                  {event.pnl_value >= 0 ? '+' : ''}€{(event.pnl_value * USDT_TO_EUR).toFixed(2)}
                                </span>
                              )}
                              <span className="text-gray-500 text-xs">{event.reason}</span>
                            </div>
                          )}
                          {event.action === 'opened' && (
                            <div className="mt-1 text-xs text-gray-500">{event.reason}</div>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  Failed to load history
                </div>
              )}
            </div>
          </motion.div>
        </div>
      )}

      {/* Chart Modal - Opens when clicking on Market News */}
      {showChartModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl max-h-[85vh] overflow-hidden"
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-slate-700 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <LineChart className="w-5 h-5 text-cyan-400" />
                  <h3 className="text-xl font-bold text-white">
                    {chartModalSymbol.replace('USDT', '')}<span className="text-gray-500 text-sm">/USDT</span>
                  </h3>
                </div>
                {chartModalTicker && (
                  <div className="flex items-center gap-4">
                    <span className="text-2xl font-bold text-white">
                      ${chartModalTicker.price.toFixed(chartModalTicker.price < 1 ? 6 : 2)}
                    </span>
                    <span className={`px-2 py-1 rounded text-sm font-bold ${
                      chartModalTicker.change24h >= 0 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                    }`}>
                      {chartModalTicker.change24h >= 0 ? '+' : ''}{chartModalTicker.change24h.toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
              <button
                onClick={() => {
                  setShowChartModal(false)
                  setChartModalSymbol('')
                  setChartModalData([])
                  setChartModalTicker(null)
                }}
                className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
              >
                <XCircle className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-4">
              {loadingChart ? (
                <div className="flex items-center justify-center py-24">
                  <Loader2 className="w-10 h-10 text-cyan-400 animate-spin" />
                </div>
              ) : (
                <>
                  {/* 24h Stats */}
                  {chartModalTicker && (
                    <div className="grid grid-cols-4 gap-3 mb-4">
                      <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                        <div className="text-lg font-bold text-white">${chartModalTicker.high24h.toFixed(chartModalTicker.high24h < 1 ? 6 : 2)}</div>
                        <div className="text-xs text-gray-400">24h High</div>
                      </div>
                      <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                        <div className="text-lg font-bold text-white">${chartModalTicker.low24h.toFixed(chartModalTicker.low24h < 1 ? 6 : 2)}</div>
                        <div className="text-xs text-gray-400">24h Low</div>
                      </div>
                      <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                        <div className={`text-lg font-bold ${chartModalTicker.change24h >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {chartModalTicker.change24h >= 0 ? '+' : ''}{chartModalTicker.change24h.toFixed(2)}%
                        </div>
                        <div className="text-xs text-gray-400">24h Change</div>
                      </div>
                      <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                        <div className="text-lg font-bold text-white">${(chartModalTicker.volume24h / 1000000).toFixed(2)}M</div>
                        <div className="text-xs text-gray-400">24h Volume</div>
                      </div>
                    </div>
                  )}

                  {/* Price Chart */}
                  <div className="bg-slate-800/30 rounded-xl p-4" style={{ height: '400px' }}>
                    {chartModalData.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartModalData}>
                          <defs>
                            <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={chartModalTicker && chartModalTicker.change24h >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0.3}/>
                              <stop offset="95%" stopColor={chartModalTicker && chartModalTicker.change24h >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <XAxis 
                            dataKey="time" 
                            tickFormatter={(t) => new Date(t).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                            stroke="#475569"
                            fontSize={10}
                          />
                          <YAxis 
                            domain={['auto', 'auto']}
                            tickFormatter={(v) => v < 1 ? v.toFixed(4) : v.toFixed(2)}
                            stroke="#475569"
                            fontSize={10}
                            width={60}
                          />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                            labelFormatter={(t) => new Date(t).toLocaleString()}
                            formatter={(value: number) => ['$' + value.toFixed(value < 1 ? 6 : 4), 'Price']}
                          />
                          <Area
                            type="monotone"
                            dataKey="close"
                            stroke={chartModalTicker && chartModalTicker.change24h >= 0 ? "#10b981" : "#ef4444"}
                            strokeWidth={2}
                            fill="url(#chartGradient)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500">
                        No chart data available
                      </div>
                    )}
                  </div>

                  {/* Trade Button */}
                  <div className="mt-4 flex justify-center">
                    <a
                      href={`https://www.bybit.com/trade/usdt/${chartModalSymbol}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold hover:from-cyan-400 hover:to-blue-500 transition-all flex items-center gap-2"
                    >
                      Trade on Bybit
                      <TrendingUp className="w-4 h-4" />
                    </a>
                  </div>
                </>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}
