'use client'

import { useEffect, useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { 
  TrendingUp, 
  TrendingDown, 
  Shield,
  Brain, 
  Activity,
  Wallet,
  Target,
  AlertTriangle,
  ChevronRight,
  Settings,
  Bell,
  LogOut,
  RefreshCw,
  Link as LinkIcon,
  Loader2,
  Gauge,
  ShieldCheck,
  ShieldAlert,
  Zap,
  BarChart3,
  Play,
  Square,
  Bot,
  PieChart as PieChartIcon,
  Newspaper,
  GraduationCap,
  ExternalLink,
  Clock,
  Coins,
  DollarSign
} from 'lucide-react'
import Link from 'next/link'
import Image from 'next/image'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie,
  Legend,
  Area,
  AreaChart
} from 'recharts'

interface ExchangeStatus {
  connected: boolean
  exchange: string | null
  serverIp: string
}

interface BalanceData {
  totalEquity: number
  coins: Array<{
    coin: string
    balance: number
    usdValue: number
    unrealizedPnl: number
  }>
}

interface Position {
  symbol: string
  side: string
  size: number
  entryPrice: number
  markPrice: number
  unrealizedPnl: number
  leverage: string
}

interface PnlData {
  totalPnl: number
  winningTrades: number
  losingTrades: number
  winRate: number
  trades: Array<{
    symbol: string
    side: string
    closedPnl: number
    createdTime: string
  }>
}

interface AIInsight {
  confidence: number
  confidence_label: string
  risk_status: string
  insight: string
  regime: string
  volatility: number
  trend: string
  fear_greed_index: number
  fear_greed_label: string
  recommended_action: string
}

interface TradingStatus {
  is_autonomous_trading: boolean
  trading_pairs: string[]
  max_positions: number
  recent_trades: any[]
}

interface BotActivity {
  is_running: boolean
  is_user_connected: boolean
  total_pairs_monitoring: number
  active_trades: any[]
  recent_completed: any[]
  bot_actions: any[]
}

interface NewsArticle {
  source: string
  title: string
  body: string
  url: string
  published: string
  sentiment: 'bullish' | 'bearish' | 'neutral'
  coins: string[]
}

interface NewsData {
  articles: NewsArticle[]
  sentiment: {
    bullish_percent: number
    bearish_percent: number
    neutral_percent: number
    overall: string
    total_articles: number
  }
  timestamp: string
}

interface LearningData {
  learning: {
    total_states_learned: number
    q_states: number
    patterns_learned: number
    market_states: number
    sentiment_states: number
    learning_progress: number
    learning_iterations: number
    best_strategies: Record<string, { action: string; confidence: number }>
    top_patterns: Array<{
      pattern: string
      outcome: string
      success_rate: number
      occurrences: number
    }>
  }
  history: any[]
  stats: {
    total_trades: number
    wins: number
    losses: number
    win_rate: number
    total_profit: number
    avg_profit_per_trade: number
    best_trade: number
    worst_trade: number
  }
  sources: {
    historical_data: boolean
    market_movements: boolean
    news_sentiment: boolean
    technical_patterns: boolean
    trade_outcomes: boolean
  }
}

interface TradeNotification {
  id: string
  symbol: string
  side: string
  pnl: number
  isWin: boolean
  timestamp: number
}

export default function DashboardPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [exchangeStatus, setExchangeStatus] = useState<ExchangeStatus | null>(null)
  const [balance, setBalance] = useState<BalanceData | null>(null)
  const [positions, setPositions] = useState<Position[]>([])
  const [pnlData, setPnlData] = useState<PnlData | null>(null)
  const [aiInsight, setAiInsight] = useState<AIInsight | null>(null)
  const [user, setUser] = useState<any>(null)
  const [tradingStatus, setTradingStatus] = useState<TradingStatus | null>(null)
  const [isTogglingBot, setIsTogglingBot] = useState(false)
  const [botActivity, setBotActivity] = useState<BotActivity | null>(null)
  const [newsData, setNewsData] = useState<NewsData | null>(null)
  const [learningData, setLearningData] = useState<LearningData | null>(null)
  const [notifications, setNotifications] = useState<TradeNotification[]>([])
  const previousTradesRef = useRef<string[]>([])

  useEffect(() => {
    // Check if user is logged in
    const storedUser = localStorage.getItem('sentinel_user')
    if (!storedUser) {
      // No session - redirect to login
      window.location.href = '/login'
      return
    }
    setUser(JSON.parse(storedUser))
    
    // Load real data
    loadData()
    
    // Auto-refresh every 3 seconds for real-time updates
    const interval = setInterval(() => {
      refreshDataSilent()
    }, 3000)
    
    return () => clearInterval(interval)
  }, [])

  // Watch for new completed trades and show notifications
  useEffect(() => {
    if (botActivity?.recent_completed) {
      // Load seen trades from localStorage
      const seenTradesKey = 'sentinel_seen_trades'
      const seenTrades: string[] = JSON.parse(localStorage.getItem(seenTradesKey) || '[]')
      
      const currentTradeIds = botActivity.recent_completed.map((t: any) => 
        `${t.symbol}-${t.closed_time || t.pnl}-${t.entry_price}`
      )
      
      // Find new trades that weren't seen before
      const newTrades = botActivity.recent_completed.filter((trade: any) => {
        const tradeId = `${trade.symbol}-${trade.closed_time || trade.pnl}-${trade.entry_price}`
        return !seenTrades.includes(tradeId) && !previousTradesRef.current.includes(tradeId)
      })
      
      // Show notification for each new trade (max 3 at a time)
      newTrades.slice(0, 3).forEach((trade: any, index: number) => {
        const pnl = parseFloat(trade.pnl) || 0
        const notification: TradeNotification = {
          id: `${Date.now()}-${Math.random()}`,
          symbol: trade.symbol,
          side: trade.side || 'CLOSE',
          pnl: pnl,
          isWin: pnl >= 0,
          timestamp: Date.now()
        }
        
        // Stagger notifications slightly
        setTimeout(() => {
          setNotifications(prev => [...prev, notification])
          
          // Remove notification after 5 seconds
          setTimeout(() => {
            setNotifications(prev => prev.filter(n => n.id !== notification.id))
          }, 5000)
        }, index * 300)
      })
      
      // Update localStorage with all current trade IDs (keep last 100)
      const updatedSeenTrades = [...new Set([...seenTrades, ...currentTradeIds])].slice(-100)
      localStorage.setItem(seenTradesKey, JSON.stringify(updatedSeenTrades))
      
      // Update the ref with current trade IDs
      previousTradesRef.current = currentTradeIds
    }
  }, [botActivity?.recent_completed])

  const loadBotActivity = async () => {
    try {
      const activityRes = await fetch('/ai/exchange/trading/activity')
      const activityData = await activityRes.json()
      if (activityData.success) {
        setBotActivity(activityData.data)
      }
    } catch (error) {
      console.error('Failed to load bot activity:', error)
    }
  }

  const loadAIInsight = async () => {
    try {
      const insightRes = await fetch('/ai/insight')
      const insightData = await insightRes.json()
      if (insightData.success) {
        setAiInsight(insightData.data)
      }
    } catch (error) {
      console.error('Failed to load AI insight:', error)
    }
  }

  const loadNewsData = async () => {
    try {
      const newsRes = await fetch('/ai/data/news?limit=15')
      const data = await newsRes.json()
      if (data.success) {
        setNewsData(data)
      }
    } catch (error) {
      console.error('Failed to load news:', error)
    }
  }

  const loadLearningData = async () => {
    try {
      const learningRes = await fetch('/ai/data/learning')
      const data = await learningRes.json()
      if (data.success) {
        setLearningData(data)
      }
    } catch (error) {
      console.error('Failed to load learning data:', error)
    }
  }

  const loadData = async () => {
    setIsLoading(true)
    
    try {
      // Load AI insight first (always available)
      await loadAIInsight()
      
      // Load news and learning data (always available)
      await Promise.all([loadNewsData(), loadLearningData()])
      
      // Check exchange connection status
      const statusRes = await fetch('/ai/exchange/status')
      const statusData = await statusRes.json()
      setExchangeStatus(statusData)

      if (statusData.connected) {
        // Load real balance
        const balanceRes = await fetch('/ai/exchange/balance')
        const balanceData = await balanceRes.json()
        if (balanceData.success) {
          setBalance(balanceData.data)
        }

        // Load real positions
        const posRes = await fetch('/ai/exchange/positions')
        const posData = await posRes.json()
        if (posData.success) {
          setPositions(posData.data.positions || [])
        }

        // Load real PnL
        const pnlRes = await fetch('/ai/exchange/pnl')
        const pnlDataResult = await pnlRes.json()
        if (pnlDataResult.success) {
          setPnlData(pnlDataResult.data)
        }

        // Load trading status
        const tradingRes = await fetch('/ai/exchange/trading/status')
        const tradingData = await tradingRes.json()
        if (tradingData.success) {
          setTradingStatus(tradingData.data)
        }

        // Load bot activity
        const activityRes = await fetch('/ai/exchange/trading/activity')
        const activityData = await activityRes.json()
        if (activityData.success) {
          setBotActivity(activityData.data)
        }
      }
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const refreshData = async () => {
    setIsRefreshing(true)
    await loadData()
    setIsRefreshing(false)
  }

  // Silent refresh - no loading spinner (for auto-refresh)
  const refreshDataSilent = async () => {
    try {
      // Check exchange connection status
      const statusRes = await fetch('/ai/exchange/status')
      const statusData = await statusRes.json()
      setExchangeStatus(statusData)

      // Load news and learning (always available even without exchange)
      const [newsRes, learningRes] = await Promise.all([
        fetch('/ai/data/news?limit=15'),
        fetch('/ai/data/learning')
      ])
      const [newsDataResult, learningDataResult] = await Promise.all([
        newsRes.json(),
        learningRes.json()
      ])
      if (newsDataResult.success) setNewsData(newsDataResult)
      if (learningDataResult.success) setLearningData(learningDataResult)

      if (statusData.connected) {
        // Load all data in parallel for faster refresh
        const [balanceRes, posRes, pnlRes, tradingRes, activityRes, insightRes] = await Promise.all([
          fetch('/ai/exchange/balance'),
          fetch('/ai/exchange/positions'),
          fetch('/ai/exchange/pnl'),
          fetch('/ai/exchange/trading/status'),
          fetch('/ai/exchange/trading/activity'),
          fetch('/ai/insight')
        ])

        const [balanceData, posData, pnlDataResult, tradingData, activityData, insightData] = await Promise.all([
          balanceRes.json(),
          posRes.json(),
          pnlRes.json(),
          tradingRes.json(),
          activityRes.json(),
          insightRes.json()
        ])

        if (balanceData.success) setBalance(balanceData.data)
        if (posData.success) setPositions(posData.data.positions || [])
        if (pnlDataResult.success) setPnlData(pnlDataResult.data)
        if (tradingData.success) setTradingStatus(tradingData.data)
        if (activityData.success) setBotActivity(activityData.data)
        if (insightData.success) setAiInsight(insightData.data)
      }
    } catch (error) {
      console.error('Silent refresh failed:', error)
    }
  }

  const startTrading = async () => {
    setIsTogglingBot(true)
    try {
      // Get stored credentials from localStorage or prompt user
      const storedCreds = localStorage.getItem('sentinel_api_creds')
      if (!storedCreds) {
        // Redirect to connect page if no credentials
        window.location.href = '/dashboard/connect'
        return
      }
      
      const creds = JSON.parse(storedCreds)
      
      const response = await fetch('/ai/exchange/trading/enable', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'default',
          api_key: creds.apiKey,
          api_secret: creds.apiSecret,
        })
      })
      
      const data = await response.json()
      if (data.success) {
        setTradingStatus(prev => prev ? { ...prev, is_autonomous_trading: true } : null)
      }
    } catch (error) {
      console.error('Failed to start trading:', error)
    } finally {
      setIsTogglingBot(false)
    }
  }

  const stopTrading = async () => {
    setIsTogglingBot(true)
    try {
      const response = await fetch('/ai/exchange/trading/disable?user_id=default', {
        method: 'POST',
      })
      
      const data = await response.json()
      if (data.success) {
        setTradingStatus(prev => prev ? { ...prev, is_autonomous_trading: false } : null)
      }
    } catch (error) {
      console.error('Failed to stop trading:', error)
    } finally {
      setIsTogglingBot(false)
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('sentinel_user')
    window.location.href = '/login'
  }

  const getRiskStatusColor = (status: string) => {
    switch (status) {
      case 'SAFE': return 'text-sentinel-accent-emerald'
      case 'ELEVATED': return 'text-sentinel-accent-amber'
      case 'CAUTION': return 'text-sentinel-accent-crimson'
      default: return 'text-sentinel-text-secondary'
    }
  }

  const getRiskStatusBg = (status: string) => {
    switch (status) {
      case 'SAFE': return 'bg-sentinel-accent-emerald/10'
      case 'ELEVATED': return 'bg-sentinel-accent-amber/10'
      case 'CAUTION': return 'bg-sentinel-accent-crimson/10'
      default: return 'bg-sentinel-bg-tertiary'
    }
  }

  const getRiskStatusIcon = (status: string) => {
    switch (status) {
      case 'SAFE': return ShieldCheck
      case 'ELEVATED': return ShieldAlert
      case 'CAUTION': return AlertTriangle
      default: return Shield
    }
  }

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-sentinel-bg-primary flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-sentinel-accent-cyan animate-spin mx-auto mb-4" />
          <p className="text-sentinel-text-secondary">Loading real-time data...</p>
        </div>
      </div>
    )
  }

  // Show connect exchange prompt if not connected
  if (!exchangeStatus?.connected) {
    return (
      <div className="min-h-screen bg-sentinel-bg-primary">
        {/* Navigation */}
        <nav className="sticky top-0 z-50 glass-card border-b border-sentinel-border">
          <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Image 
                src="/logo.png" 
                alt="Sentinel Logo" 
                width={40} 
                height={40} 
                className="rounded-lg"
              />
              <div>
                <span className="font-display font-bold text-lg">SENTINEL</span>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-sentinel-accent-amber" />
                  <span className="text-xs text-sentinel-text-muted">Not Connected</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {user?.isAdmin && (
                <Link href="/admin" className="px-3 py-1.5 rounded-lg bg-sentinel-accent-crimson/20 text-sentinel-accent-crimson text-sm font-medium">
                  Admin
                </Link>
              )}
              <button onClick={handleLogout} className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors">
                <LogOut className="w-5 h-5 text-sentinel-text-secondary" />
              </button>
            </div>
          </div>
        </nav>

        {/* AI Status Bar - Always visible */}
        {aiInsight && (
          <div className="bg-sentinel-bg-secondary border-b border-sentinel-border">
            <div className="max-w-[1600px] mx-auto px-6 py-3 flex items-center justify-between">
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-sentinel-accent-cyan" />
                  <span className="text-sm text-sentinel-text-secondary">AI Confidence:</span>
                  <span className="font-mono font-bold text-sentinel-accent-cyan">{aiInsight.confidence}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-sentinel-accent-amber" />
                  <span className="text-sm text-sentinel-text-secondary">Fear & Greed:</span>
                  <span className="font-mono">{aiInsight.fear_greed_index}</span>
                  <span className="text-xs text-sentinel-text-muted">({aiInsight.fear_greed_label})</span>
                </div>
              </div>
              <div className="text-sm text-sentinel-text-muted">{aiInsight.insight}</div>
            </div>
          </div>
        )}

        {/* Connect Exchange Prompt */}
        <main className="max-w-2xl mx-auto px-6 py-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <div className="w-24 h-24 mx-auto rounded-full bg-sentinel-accent-cyan/10 flex items-center justify-center mb-8">
              <LinkIcon className="w-12 h-12 text-sentinel-accent-cyan" />
            </div>
            
            <h1 className="text-3xl font-bold mb-4">Connect Your Exchange</h1>
            <p className="text-sentinel-text-secondary text-lg mb-8 max-w-md mx-auto">
              To see your real balance, positions, and P&L, you need to connect your Bybit account.
            </p>

            <Link
              href="/dashboard/connect"
              className="inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary font-bold text-lg hover:shadow-glow-cyan transition-all"
            >
              Connect Bybit Account
              <ChevronRight className="w-5 h-5" />
            </Link>

            <div className="mt-12 p-6 rounded-2xl glass-card text-left">
              <h3 className="font-semibold mb-4">Why connect?</h3>
              <ul className="space-y-3 text-sentinel-text-secondary">
                <li className="flex gap-3">
                  <Wallet className="w-5 h-5 text-sentinel-accent-cyan flex-shrink-0" />
                  <span>See your real-time wallet balance</span>
                </li>
                <li className="flex gap-3">
                  <Activity className="w-5 h-5 text-sentinel-accent-emerald flex-shrink-0" />
                  <span>View open positions and unrealized P&L</span>
                </li>
                <li className="flex gap-3">
                  <TrendingUp className="w-5 h-5 text-sentinel-accent-amber flex-shrink-0" />
                  <span>Track your trading history and profits</span>
                </li>
                <li className="flex gap-3">
                  <Brain className="w-5 h-5 text-sentinel-accent-violet flex-shrink-0" />
                  <span>Enable AI to analyze and trade for you</span>
                </li>
              </ul>
            </div>
          </motion.div>
        </main>

        {/* Footer */}
        <footer className="border-t border-sentinel-border mt-8">
          <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between text-sm text-sentinel-text-muted">
            <span>SENTINEL AI - Autonomous Digital Trader</span>
            <span>Developed by NoLimitDevelopments</span>
          </div>
        </footer>
      </div>
    )
  }

  // Show real data dashboard
  const RiskIcon = getRiskStatusIcon(aiInsight?.risk_status || 'SAFE')
  
  return (
    <div className="min-h-screen bg-sentinel-bg-primary">
      {/* Top Navigation */}
      <nav className="sticky top-0 z-50 glass-card border-b border-sentinel-border">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Image 
              src="/logo.png" 
              alt="Sentinel Logo" 
              width={40} 
              height={40} 
              className="rounded-lg"
            />
            <div>
              <span className="font-display font-bold text-lg">SENTINEL</span>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-sentinel-accent-emerald live-pulse" />
                <span className="text-xs text-sentinel-text-muted">Connected to {exchangeStatus?.exchange}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button 
              onClick={refreshData}
              className={`p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors ${isRefreshing ? 'animate-spin' : ''}`}
            >
              <RefreshCw className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
            <button className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors relative">
              <Bell className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
            {user?.isAdmin && (
              <Link href="/admin" className="px-3 py-1.5 rounded-lg bg-sentinel-accent-crimson/20 text-sentinel-accent-crimson text-sm font-medium hover:bg-sentinel-accent-crimson/30 transition-colors">
                Admin
              </Link>
            )}
            <Link href="/dashboard/settings" className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors">
              <Settings className="w-5 h-5 text-sentinel-text-secondary" />
            </Link>
            <div className="w-px h-8 bg-sentinel-border" />
            <button onClick={handleLogout} className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors">
              <LogOut className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
          </div>
        </div>
      </nav>

      {/* AI Status Bar */}
      {aiInsight && (
        <div className="bg-sentinel-bg-secondary border-b border-sentinel-border">
          <div className="max-w-[1600px] mx-auto px-6 py-3 flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-6 flex-wrap">
              {/* AI Confidence */}
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-sentinel-accent-cyan/10">
                  <Brain className="w-4 h-4 text-sentinel-accent-cyan" />
                </div>
                <div>
                  <span className="text-xs text-sentinel-text-muted block">AI Confidence</span>
                  <span className="font-mono font-bold text-sentinel-accent-cyan">{aiInsight.confidence}%</span>
                </div>
              </div>

              {/* Risk Status */}
              <div className="flex items-center gap-2">
                <div className={`p-1.5 rounded-lg ${getRiskStatusBg(aiInsight.risk_status)}`}>
                  <RiskIcon className={`w-4 h-4 ${getRiskStatusColor(aiInsight.risk_status)}`} />
                </div>
                <div>
                  <span className="text-xs text-sentinel-text-muted block">Risk Status</span>
                  <span className={`font-bold ${getRiskStatusColor(aiInsight.risk_status)}`}>{aiInsight.risk_status}</span>
                </div>
              </div>

              {/* Market Regime */}
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-sentinel-accent-violet/10">
                  <BarChart3 className="w-4 h-4 text-sentinel-accent-violet" />
                </div>
                <div>
                  <span className="text-xs text-sentinel-text-muted block">Market Regime</span>
                  <span className="font-medium capitalize">{aiInsight.regime.replace('_', ' ')}</span>
                </div>
              </div>

              {/* Fear & Greed */}
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-sentinel-accent-amber/10">
                  <Gauge className="w-4 h-4 text-sentinel-accent-amber" />
                </div>
                <div>
                  <span className="text-xs text-sentinel-text-muted block">Fear & Greed</span>
                  <span className="font-mono">{aiInsight.fear_greed_index}</span>
                  <span className="text-xs text-sentinel-text-muted ml-1">({aiInsight.fear_greed_label})</span>
                </div>
              </div>
            </div>

            {/* AI Insight */}
            <div className="flex items-center gap-2 bg-sentinel-bg-tertiary px-4 py-2 rounded-lg">
              <Zap className="w-4 h-4 text-sentinel-accent-amber" />
              <span className="text-sm">{aiInsight.insight}</span>
            </div>
          </div>
        </div>
      )}

      {/* Trading Control Panel */}
      <div className="max-w-[1600px] mx-auto px-6 pt-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-6 rounded-2xl border-2 ${
            tradingStatus?.is_autonomous_trading 
              ? 'glass-card border-sentinel-accent-emerald bg-sentinel-accent-emerald/5' 
              : 'glass-card border-sentinel-border'
          }`}
        >
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              <div className={`w-14 h-14 rounded-xl flex items-center justify-center ${
                tradingStatus?.is_autonomous_trading 
                  ? 'bg-sentinel-accent-emerald/20' 
                  : 'bg-sentinel-bg-tertiary'
              }`}>
                <Bot className={`w-8 h-8 ${
                  tradingStatus?.is_autonomous_trading 
                    ? 'text-sentinel-accent-emerald' 
                    : 'text-sentinel-text-muted'
                }`} />
              </div>
              <div>
                <h2 className="text-xl font-bold">
                  {tradingStatus?.is_autonomous_trading ? '24/7 Trading ACTIVE' : 'Autonomous Trading'}
                </h2>
                <p className="text-sentinel-text-secondary text-sm">
                  {tradingStatus?.is_autonomous_trading 
                    ? `AI is trading ${tradingStatus?.trading_pairs?.length || 0}+ crypto pairs with your real funds`
                    : 'Start the AI to trade automatically 24/7'
                  }
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {tradingStatus?.is_autonomous_trading && (
                <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-sentinel-bg-tertiary">
                  <div className="w-2 h-2 rounded-full bg-sentinel-accent-emerald live-pulse" />
                  <span className="text-sm font-mono">{tradingStatus?.trading_pairs?.length || 80}+ pairs</span>
                </div>
              )}

              {tradingStatus?.is_autonomous_trading ? (
                <button
                  onClick={stopTrading}
                  disabled={isTogglingBot}
                  className="flex items-center gap-3 px-6 py-3 rounded-xl bg-sentinel-accent-crimson text-white font-bold hover:bg-sentinel-accent-crimson/90 transition-all disabled:opacity-50"
                >
                  {isTogglingBot ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <Square className="w-5 h-5" />
                      STOP TRADING
                    </>
                  )}
                </button>
              ) : (
                <button
                  onClick={startTrading}
                  disabled={isTogglingBot}
                  className="flex items-center gap-3 px-8 py-3 rounded-xl bg-gradient-to-r from-sentinel-accent-emerald to-sentinel-accent-cyan text-sentinel-bg-primary font-bold hover:shadow-glow-cyan transition-all disabled:opacity-50"
                >
                  {isTogglingBot ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      START TRADING
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {tradingStatus?.is_autonomous_trading && (
            <div className="mt-4 pt-4 border-t border-sentinel-border">
              <div className="flex items-center gap-6 text-sm text-sentinel-text-secondary">
                <span>Max positions: <strong className="text-sentinel-text-primary">{tradingStatus?.max_positions || 10}</strong></span>
                <span>Strategy: <strong className="text-sentinel-text-primary capitalize">{aiInsight?.recommended_action || 'Auto'}</strong></span>
                <span>Regime: <strong className="text-sentinel-text-primary capitalize">{aiInsight?.regime?.replace('_', ' ') || 'Analyzing'}</strong></span>
              </div>
            </div>
          )}
        </motion.div>

        {/* Active Investments Breakdown */}
        {positions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="mt-6 p-6 rounded-2xl glass-card"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-sentinel-accent-cyan/10">
                  <Coins className="w-5 h-5 text-sentinel-accent-cyan" />
                </div>
                <h2 className="text-lg font-semibold">Active Investments by Crypto</h2>
              </div>
              <div className="text-sm text-sentinel-text-muted">
                {positions.length} open positions
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {positions.map((position, idx) => {
                const investedAmount = position.size * position.entryPrice
                const pnlPercent = investedAmount > 0 ? (position.unrealizedPnl / investedAmount) * 100 : 0
                const coinName = position.symbol.replace('USDT', '')
                
                const handleClosePosition = async (symbol: string) => {
                  if (!confirm(`Close ${symbol} position? Current P&L: €${position.unrealizedPnl.toFixed(2)}`)) {
                    return
                  }
                  
                  try {
                    const res = await fetch(`/api/ai/exchange/close-position/${symbol}`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' }
                    })
                    const data = await res.json()
                    
                    if (data.success) {
                      alert(`Position closed! P&L: €${data.data.pnl.toFixed(2)}`)
                      // Refresh will happen automatically via the 3s interval
                    } else {
                      alert(`Failed to close: ${data.error}`)
                    }
                  } catch (err) {
                    alert('Error closing position')
                  }
                }
                
                return (
                  <div 
                    key={idx} 
                    className={`p-4 rounded-xl border-2 transition-all ${
                      position.unrealizedPnl >= 0 
                        ? 'bg-sentinel-accent-emerald/5 border-sentinel-accent-emerald/30' 
                        : 'bg-sentinel-accent-crimson/5 border-sentinel-accent-crimson/30'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-bold text-lg">{coinName}</span>
                      <button
                        onClick={() => handleClosePosition(position.symbol)}
                        className={`text-xs px-2 py-1 rounded font-bold transition-all hover:scale-105 ${
                          position.side.toLowerCase() === 'buy' 
                            ? 'bg-sentinel-accent-crimson/80 hover:bg-sentinel-accent-crimson text-white' 
                            : 'bg-sentinel-accent-emerald/80 hover:bg-sentinel-accent-emerald text-white'
                        }`}
                      >
                        {position.side.toLowerCase() === 'buy' ? 'SELL' : 'BUY'}
                      </button>
                    </div>
                    
                    <div className="space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-sentinel-text-muted">Invested</span>
                        <span className="font-mono font-bold text-sentinel-accent-cyan">
                          €{investedAmount.toFixed(2)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-sentinel-text-muted">P&L</span>
                        <span className={`font-mono font-bold ${
                          position.unrealizedPnl >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'
                        }`}>
                          {position.unrealizedPnl >= 0 ? '+' : ''}€{position.unrealizedPnl.toFixed(2)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-sentinel-text-muted">Change</span>
                        <span className={`font-mono text-sm ${
                          pnlPercent >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'
                        }`}>
                          {pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%
                        </span>
                      </div>
                    </div>

                    {/* Progress bar showing P&L */}
                    <div className="mt-2 h-1.5 rounded-full bg-sentinel-bg-tertiary overflow-hidden">
                      <div 
                        className={`h-full transition-all ${
                          pnlPercent >= 0 ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-accent-crimson'
                        }`}
                        style={{ 
                          width: `${Math.min(Math.abs(pnlPercent) * 10, 100)}%`,
                          marginLeft: pnlPercent < 0 ? 'auto' : '0'
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Investment Distribution Summary */}
            <div className="mt-6 pt-4 border-t border-sentinel-border">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold font-mono text-sentinel-accent-cyan">
                    €{positions.reduce((sum, p) => sum + (p.size * p.entryPrice), 0).toFixed(2)}
                  </div>
                  <div className="text-xs text-sentinel-text-muted">Total Invested</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold font-mono">
                    {positions.length}
                  </div>
                  <div className="text-xs text-sentinel-text-muted">Open Positions</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold font-mono text-sentinel-accent-emerald">
                    {positions.filter(p => p.unrealizedPnl >= 0).length}
                  </div>
                  <div className="text-xs text-sentinel-text-muted">In Profit</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold font-mono text-sentinel-accent-crimson">
                    {positions.filter(p => p.unrealizedPnl < 0).length}
                  </div>
                  <div className="text-xs text-sentinel-text-muted">In Loss</div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-6 py-8">
        {/* Top Stats Row - REAL DATA */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Balance Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="col-span-1 lg:col-span-2 p-6 rounded-2xl glass-card"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-xl bg-sentinel-accent-cyan/10">
                  <Wallet className="w-6 h-6 text-sentinel-accent-cyan" />
                </div>
                <span className="text-sentinel-text-secondary">Total Equity</span>
              </div>
              <div className="text-xs text-sentinel-text-muted">
                Real-time from {exchangeStatus?.exchange}
              </div>
            </div>
            <div className="text-4xl font-display font-bold">
              €{balance?.totalEquity?.toLocaleString('de-DE', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0,00'}
            </div>
          </motion.div>

          {/* Realized P&L */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="p-6 rounded-2xl glass-card"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 rounded-xl bg-sentinel-accent-emerald/10">
                <Target className="w-6 h-6 text-sentinel-accent-emerald" />
              </div>
              <span className="text-sentinel-text-secondary">Realized P&L</span>
            </div>
            <div className={`text-2xl font-display font-bold ${(pnlData?.totalPnl || 0) >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}`}>
              {(pnlData?.totalPnl || 0) >= 0 ? '+' : ''}€{pnlData?.totalPnl?.toFixed(2) || '0,00'}
            </div>
            <div className="text-sm text-sentinel-text-muted mt-1">
              {pnlData?.winningTrades || 0}W / {pnlData?.losingTrades || 0}L ({pnlData?.winRate?.toFixed(1) || 0}%)
            </div>
          </motion.div>

          {/* Open Positions Count */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="p-6 rounded-2xl glass-card"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 rounded-xl bg-sentinel-accent-amber/10">
                <Activity className="w-6 h-6 text-sentinel-accent-amber" />
              </div>
              <span className="text-sentinel-text-secondary">Open Positions</span>
            </div>
            <div className="text-2xl font-display font-bold">
              {positions.length}
            </div>
            <div className="text-sm text-sentinel-text-muted mt-1">
              Active trades
            </div>
          </motion.div>
        </div>

        {/* Performance Charts Section - MOVED UP FOR VISIBILITY */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          {/* Win/Loss Chart */}
          <div className="p-6 rounded-2xl glass-card">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-sentinel-accent-violet/10">
                  <PieChartIcon className="w-5 h-5 text-sentinel-accent-violet" />
                </div>
                <h2 className="text-lg font-semibold">Win/Loss Ratio</h2>
              </div>
              <div className="text-sm text-sentinel-text-secondary">
                {((pnlData?.winningTrades || 0) + (pnlData?.losingTrades || 0))} total trades
              </div>
            </div>

            {((pnlData?.winningTrades || 0) + (pnlData?.losingTrades || 0)) === 0 ? (
              <div className="h-48 flex items-center justify-center text-sentinel-text-muted">
                <div className="text-center">
                  <BarChart3 className="w-10 h-10 mx-auto mb-2 opacity-30" />
                  <p>Waiting for completed trades...</p>
                </div>
              </div>
            ) : (
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Wins', value: pnlData?.winningTrades || 0 },
                        { name: 'Losses', value: pnlData?.losingTrades || 0 }
                      ]}
                      cx="50%"
                      cy="50%"
                      innerRadius={45}
                      outerRadius={70}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      <Cell fill="#00DC82" />
                      <Cell fill="#FF4757" />
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #2d3548', borderRadius: '8px' }}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Stats below chart */}
            <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-sentinel-border">
              <div className="text-center">
                <div className="text-xl font-bold text-sentinel-accent-emerald">{pnlData?.winningTrades || 0}</div>
                <div className="text-xs text-sentinel-text-muted">Wins</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-sentinel-accent-crimson">{pnlData?.losingTrades || 0}</div>
                <div className="text-xs text-sentinel-text-muted">Losses</div>
              </div>
              <div className="text-center">
                <div className={`text-xl font-bold ${(pnlData?.winRate || 0) >= 50 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}`}>
                  {pnlData?.winRate?.toFixed(1) || 0}%
                </div>
                <div className="text-xs text-sentinel-text-muted">Win Rate</div>
              </div>
            </div>
          </div>

          {/* P&L Performance Chart */}
          <div className="p-6 rounded-2xl glass-card">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-sentinel-accent-cyan/10">
                  <TrendingUp className="w-5 h-5 text-sentinel-accent-cyan" />
                </div>
                <h2 className="text-lg font-semibold">P&L Performance</h2>
              </div>
              <div className={`text-lg font-mono font-bold ${(pnlData?.totalPnl || 0) >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}`}>
                {(pnlData?.totalPnl || 0) >= 0 ? '+' : ''}€{pnlData?.totalPnl?.toFixed(2) || '0.00'}
              </div>
            </div>

            {!pnlData?.trades?.length ? (
              <div className="h-48 flex items-center justify-center text-sentinel-text-muted">
                <div className="text-center">
                  <TrendingUp className="w-10 h-10 mx-auto mb-2 opacity-30" />
                  <p>Chart appears after trades complete</p>
                </div>
              </div>
            ) : (
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={(() => {
                      let cumulative = 0
                      return pnlData.trades.slice().reverse().map((trade, idx) => {
                        cumulative += trade.closedPnl
                        return {
                          name: `#${idx + 1}`,
                          symbol: trade.symbol,
                          pnl: trade.closedPnl,
                          cumulative: parseFloat(cumulative.toFixed(2))
                        }
                      }).slice(-15)
                    })()}
                    margin={{ top: 5, right: 5, left: -20, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="colorPnlTop" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00DC82" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#00DC82" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" />
                    <XAxis dataKey="name" stroke="#6b7280" tick={{ fill: '#6b7280', fontSize: 10 }} />
                    <YAxis stroke="#6b7280" tick={{ fill: '#6b7280', fontSize: 10 }} tickFormatter={(v) => `€${v}`} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #2d3548', borderRadius: '8px' }}
                      formatter={(value: number) => [`€${value.toFixed(2)}`]}
                    />
                    <Area type="monotone" dataKey="cumulative" stroke="#00DC82" strokeWidth={2} fill="url(#colorPnlTop)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Summary stats */}
            <div className="grid grid-cols-4 gap-2 mt-4 pt-4 border-t border-sentinel-border">
              <div className="text-center">
                <div className="text-sm font-bold text-sentinel-accent-emerald">
                  +€{pnlData?.trades?.filter(t => t.closedPnl > 0).reduce((sum, t) => sum + t.closedPnl, 0)?.toFixed(2) || '0.00'}
                </div>
                <div className="text-xs text-sentinel-text-muted">Profit</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-sentinel-accent-crimson">
                  €{pnlData?.trades?.filter(t => t.closedPnl < 0).reduce((sum, t) => sum + t.closedPnl, 0)?.toFixed(2) || '0.00'}
                </div>
                <div className="text-xs text-sentinel-text-muted">Loss</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold">
                  €{((pnlData?.trades?.filter(t => t.closedPnl > 0).reduce((sum, t) => sum + t.closedPnl, 0) || 0) / Math.max(pnlData?.winningTrades || 1, 1)).toFixed(2)}
                </div>
                <div className="text-xs text-sentinel-text-muted">Avg Win</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold">
                  €{Math.abs((pnlData?.trades?.filter(t => t.closedPnl < 0).reduce((sum, t) => sum + t.closedPnl, 0) || 0) / Math.max(pnlData?.losingTrades || 1, 1)).toFixed(2)}
                </div>
                <div className="text-xs text-sentinel-text-muted">Avg Loss</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Main Grid */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Positions Table */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="lg:col-span-2 p-6 rounded-2xl glass-card"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold">Open Positions</h2>
              <span className="text-xs text-sentinel-text-muted">Real-time data</span>
            </div>

            {positions.length === 0 ? (
              <div className="text-center py-12 text-sentinel-text-muted">
                <Activity className="w-12 h-12 mx-auto mb-4 opacity-30" />
                <p>No open positions</p>
              </div>
            ) : (
              <>
                {/* Total Investment Summary */}
                <div className="mb-4 p-4 rounded-xl bg-sentinel-bg-tertiary">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm text-sentinel-text-muted">Total Invested in Positions</span>
                      <div className="text-2xl font-bold font-mono text-sentinel-accent-cyan">
                        €{positions.reduce((sum, p) => sum + (p.size * p.entryPrice), 0).toLocaleString('de-DE', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-sm text-sentinel-text-muted">Unrealized P&L</span>
                      <div className={`text-2xl font-bold font-mono ${
                        positions.reduce((sum, p) => sum + p.unrealizedPnl, 0) >= 0 
                          ? 'text-sentinel-accent-emerald' 
                          : 'text-sentinel-accent-crimson'
                      }`}>
                        {positions.reduce((sum, p) => sum + p.unrealizedPnl, 0) >= 0 ? '+' : ''}
                        €{positions.reduce((sum, p) => sum + p.unrealizedPnl, 0).toFixed(2)}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-sm text-sentinel-text-muted border-b border-sentinel-border">
                        <th className="pb-4 font-medium">Symbol</th>
                        <th className="pb-4 font-medium">Side</th>
                        <th className="pb-4 font-medium text-right">Size</th>
                        <th className="pb-4 font-medium text-right">Invested</th>
                        <th className="pb-4 font-medium text-right">Entry</th>
                        <th className="pb-4 font-medium text-right">Mark</th>
                        <th className="pb-4 font-medium text-right">P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map((position, idx) => {
                        const investedAmount = position.size * position.entryPrice
                        return (
                          <tr key={idx} className="border-b border-sentinel-border/50 last:border-0">
                            <td className="py-4">
                              <span className="font-mono font-medium">{position.symbol}</span>
                            </td>
                            <td className="py-4">
                              <span className={`px-2 py-1 rounded text-xs font-medium uppercase ${
                                position.side.toLowerCase() === 'buy' ? 'bg-sentinel-accent-emerald/10 text-sentinel-accent-emerald' :
                                'bg-sentinel-accent-crimson/10 text-sentinel-accent-crimson'
                              }`}>
                                {position.side}
                              </span>
                            </td>
                            <td className="py-4 text-right font-mono">
                              {position.size}
                            </td>
                            <td className="py-4 text-right font-mono font-bold text-sentinel-accent-cyan">
                              €{investedAmount.toFixed(2)}
                            </td>
                            <td className="py-4 text-right font-mono text-sentinel-text-secondary">
                              €{position.entryPrice?.toFixed(2)}
                            </td>
                            <td className="py-4 text-right font-mono">
                              €{position.markPrice?.toFixed(2)}
                            </td>
                            <td className="py-4 text-right">
                              <div className={`font-mono font-medium ${
                                position.unrealizedPnl >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'
                              }`}>
                                {position.unrealizedPnl >= 0 ? '+' : ''}€{position.unrealizedPnl?.toFixed(2)}
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </motion.div>

          {/* Recent Trades */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="p-6 rounded-2xl glass-card"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold">Recent Trades</h2>
              <span className="text-xs text-sentinel-text-muted">Real P&L</span>
            </div>

            {!pnlData?.trades?.length ? (
              <div className="text-center py-8 text-sentinel-text-muted">
                <TrendingUp className="w-10 h-10 mx-auto mb-3 opacity-30" />
                <p>No recent trades</p>
              </div>
            ) : (
              <div className="space-y-3">
                {pnlData.trades.slice(0, 10).map((trade, idx) => (
                  <div key={idx} className="flex items-center justify-between py-2 border-b border-sentinel-border/30 last:border-0">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                        trade.closedPnl >= 0 ? 'bg-sentinel-accent-emerald/10' : 'bg-sentinel-accent-crimson/10'
                      }`}>
                        {trade.closedPnl >= 0 ? (
                          <TrendingUp className="w-4 h-4 text-sentinel-accent-emerald" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-sentinel-accent-crimson" />
                        )}
                      </div>
                      <div>
                        <div className="font-mono text-sm">{trade.symbol}</div>
                        <div className="text-xs text-sentinel-text-muted capitalize">{trade.side}</div>
                      </div>
                    </div>
                    <div className={`font-mono font-medium ${
                      trade.closedPnl >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'
                    }`}>
                      {trade.closedPnl >= 0 ? '+' : ''}€{trade.closedPnl?.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>

        {/* Bot Activity Log */}
        {tradingStatus?.is_autonomous_trading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-8 p-6 rounded-2xl glass-card border-2 border-sentinel-accent-cyan/30"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-sentinel-accent-emerald live-pulse" />
                <h2 className="text-lg font-semibold">Bot Activity - LIVE</h2>
              </div>
              <span className="text-xs text-sentinel-text-muted">
                Monitoring {botActivity?.total_pairs_monitoring || 0} crypto pairs
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Active Trades */}
              <div>
                <h3 className="text-sm font-medium text-sentinel-text-secondary mb-3">Active Trades</h3>
                {botActivity?.active_trades?.length === 0 ? (
                  <div className="text-sm text-sentinel-text-muted py-4">
                    Analyzing markets...
                  </div>
                ) : (
                  <div className="space-y-2">
                    {botActivity?.active_trades?.slice(0, 5).map((trade: any, idx: number) => (
                      <div key={idx} className="p-3 rounded-lg bg-sentinel-bg-tertiary">
                        <div className="flex justify-between items-center">
                          <span className="font-mono text-sm">{trade.symbol}</span>
                          <span className={`text-xs px-2 py-0.5 rounded ${
                            trade.side === 'buy' ? 'bg-sentinel-accent-emerald/20 text-sentinel-accent-emerald' : 
                            'bg-sentinel-accent-crimson/20 text-sentinel-accent-crimson'
                          }`}>
                            {trade.side?.toUpperCase()}
                          </span>
                        </div>
                        <div className="text-xs text-sentinel-text-muted mt-1">
                          {trade.strategy} - {trade.confidence?.toFixed(0)}% confident
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Recent Actions */}
              <div>
                <h3 className="text-sm font-medium text-sentinel-text-secondary mb-3">Recent Bot Actions</h3>
                {botActivity?.bot_actions?.length === 0 ? (
                  <div className="text-sm text-sentinel-text-muted py-4">
                    Waiting for trading signals...
                  </div>
                ) : (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {botActivity?.bot_actions?.slice(0, 8).map((action: any, idx: number) => (
                      <div key={idx} className="text-xs py-1 border-b border-sentinel-border/30">
                        <div className="flex justify-between">
                          <span className="font-mono">{action.symbol}</span>
                          <span className={action.side === 'buy' ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}>
                            {action.side?.toUpperCase()}
                          </span>
                        </div>
                        <div className="text-sentinel-text-muted truncate">
                          {action.reasoning || action.strategy}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Completed Trades */}
              <div>
                <h3 className="text-sm font-medium text-sentinel-text-secondary mb-3">Bot Completed Trades</h3>
                {botActivity?.recent_completed?.length === 0 ? (
                  <div className="text-sm text-sentinel-text-muted py-4">
                    No completed trades yet. Bot is learning...
                  </div>
                ) : (
                  <div className="space-y-2">
                    {botActivity?.recent_completed?.slice(0, 5).map((trade: any, idx: number) => (
                      <div key={idx} className="p-3 rounded-lg bg-sentinel-bg-tertiary">
                        <div className="flex justify-between items-center">
                          <span className="font-mono text-sm">{trade.symbol}</span>
                          <span className={`font-mono font-medium ${
                            trade.pnl >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'
                          }`}>
                            {trade.pnl >= 0 ? '+' : ''}€{trade.pnl?.toFixed(2)}
                          </span>
                        </div>
                        <div className="text-xs text-sentinel-text-muted mt-1">
                          {trade.close_reason}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-sentinel-border text-xs text-sentinel-text-muted">
              Strategy: SMART AI | Trailing: -0.8% from PEAK | Emergency Stop: -2% | HOLD while rising | Auto-refresh 3s
            </div>
          </motion.div>
        )}

        {/* News & Learning Section */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Crypto News Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.55 }}
            className="p-6 rounded-2xl glass-card"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-sentinel-accent-amber/10">
                  <Newspaper className="w-5 h-5 text-sentinel-accent-amber" />
                </div>
                <h2 className="text-lg font-semibold">Crypto News - LIVE</h2>
              </div>
              {newsData?.sentiment && (
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-2 py-1 rounded font-medium ${
                    newsData.sentiment.overall === 'bullish' ? 'bg-sentinel-accent-emerald/20 text-sentinel-accent-emerald' :
                    newsData.sentiment.overall === 'bearish' ? 'bg-sentinel-accent-crimson/20 text-sentinel-accent-crimson' :
                    'bg-sentinel-bg-tertiary text-sentinel-text-muted'
                  }`}>
                    {newsData.sentiment.overall?.toUpperCase()} {newsData.sentiment.bullish_percent?.toFixed(0)}%
                  </span>
                </div>
              )}
            </div>

            {/* News Sentiment Bar */}
            {newsData?.sentiment && (
              <div className="mb-4">
                <div className="flex h-2 rounded-full overflow-hidden bg-sentinel-bg-tertiary">
                  <div 
                    className="bg-sentinel-accent-emerald" 
                    style={{ width: `${newsData.sentiment.bullish_percent}%` }}
                  />
                  <div 
                    className="bg-sentinel-text-muted" 
                    style={{ width: `${newsData.sentiment.neutral_percent}%` }}
                  />
                  <div 
                    className="bg-sentinel-accent-crimson" 
                    style={{ width: `${newsData.sentiment.bearish_percent}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-sentinel-text-muted mt-1">
                  <span>Bullish {newsData.sentiment.bullish_percent?.toFixed(0)}%</span>
                  <span>Neutral {newsData.sentiment.neutral_percent?.toFixed(0)}%</span>
                  <span>Bearish {newsData.sentiment.bearish_percent?.toFixed(0)}%</span>
                </div>
              </div>
            )}

            {/* News Articles */}
            {!newsData?.articles?.length ? (
              <div className="text-center py-8 text-sentinel-text-muted">
                <Newspaper className="w-10 h-10 mx-auto mb-3 opacity-30" />
                <p>Loading crypto news...</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
                {newsData.articles.slice(0, 10).map((article, idx) => (
                  <div key={idx} className="p-3 rounded-lg bg-sentinel-bg-tertiary hover:bg-sentinel-bg-secondary transition-colors">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`w-2 h-2 rounded-full ${
                            article.sentiment === 'bullish' ? 'bg-sentinel-accent-emerald' :
                            article.sentiment === 'bearish' ? 'bg-sentinel-accent-crimson' :
                            'bg-sentinel-text-muted'
                          }`} />
                          <span className="text-xs text-sentinel-text-muted truncate">{article.source}</span>
                          {article.coins?.length > 0 && (
                            <div className="flex gap-1">
                              {article.coins.slice(0, 2).map((coin, i) => (
                                <span key={i} className="text-xs px-1.5 py-0.5 rounded bg-sentinel-accent-cyan/10 text-sentinel-accent-cyan">
                                  {coin}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                        <a 
                          href={article.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-sm font-medium hover:text-sentinel-accent-cyan transition-colors line-clamp-2"
                        >
                          {article.title}
                        </a>
                        <p className="text-xs text-sentinel-text-muted mt-1 line-clamp-1">
                          {article.body}
                        </p>
                      </div>
                      <a 
                        href={article.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="p-1.5 rounded hover:bg-sentinel-bg-tertiary"
                      >
                        <ExternalLink className="w-4 h-4 text-sentinel-text-muted" />
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-4 pt-4 border-t border-sentinel-border text-xs text-sentinel-text-muted flex items-center gap-2">
              <Clock className="w-3 h-3" />
              Updated: {newsData?.timestamp ? new Date(newsData.timestamp).toLocaleTimeString() : 'Loading...'} | 
              Sources: CryptoCompare, CoinGecko, CoinPaprika, Reddit
            </div>
          </motion.div>

          {/* AI Learning Panel - Enhanced Multi-Source */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="p-6 rounded-2xl glass-card"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-sentinel-accent-violet/10">
                  <GraduationCap className="w-5 h-5 text-sentinel-accent-violet" />
                </div>
                <h2 className="text-lg font-semibold">AI Learning Engine</h2>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-sentinel-accent-emerald live-pulse" />
                <span className="text-sm font-mono text-sentinel-accent-violet">
                  {learningData?.learning?.learning_progress?.toFixed(0) || 0}%
                </span>
              </div>
            </div>

            {/* Learning Sources - 5 indicators */}
            <div className="grid grid-cols-5 gap-2 mb-4">
              <div className="text-center p-2 rounded-lg bg-sentinel-bg-tertiary">
                <div className="text-lg font-bold text-sentinel-accent-cyan">
                  {learningData?.learning?.q_states || 0}
                </div>
                <div className="text-[10px] text-sentinel-text-muted">Strategies</div>
              </div>
              <div className="text-center p-2 rounded-lg bg-sentinel-bg-tertiary">
                <div className="text-lg font-bold text-sentinel-accent-amber">
                  {learningData?.learning?.patterns_learned || 0}
                </div>
                <div className="text-[10px] text-sentinel-text-muted">Patterns</div>
              </div>
              <div className="text-center p-2 rounded-lg bg-sentinel-bg-tertiary">
                <div className="text-lg font-bold text-sentinel-accent-emerald">
                  {learningData?.learning?.market_states || 0}
                </div>
                <div className="text-[10px] text-sentinel-text-muted">Markets</div>
              </div>
              <div className="text-center p-2 rounded-lg bg-sentinel-bg-tertiary">
                <div className="text-lg font-bold text-sentinel-accent-crimson">
                  {learningData?.learning?.sentiment_states || 0}
                </div>
                <div className="text-[10px] text-sentinel-text-muted">Sentiment</div>
              </div>
              <div className="text-center p-2 rounded-lg bg-sentinel-bg-tertiary">
                <div className="text-lg font-bold text-white">
                  {learningData?.learning?.total_states_learned || 0}
                </div>
                <div className="text-[10px] text-sentinel-text-muted">Total</div>
              </div>
            </div>

            {/* Learning Progress Bar */}
            <div className="mb-4">
              <div className="h-2 rounded-full bg-sentinel-bg-tertiary overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-sentinel-accent-violet via-sentinel-accent-cyan to-sentinel-accent-emerald transition-all duration-500"
                  style={{ width: `${Math.min(learningData?.learning?.learning_progress || 0, 100)}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-sentinel-text-muted mt-1">
                <span>Learning from: Historical, Market, News, Patterns, Trades</span>
                <span>Iteration #{learningData?.learning?.learning_iterations || 0}</span>
              </div>
            </div>

            {/* Trade Statistics */}
            <div className="grid grid-cols-4 gap-2 mb-4">
              <div className="p-2 rounded-lg bg-sentinel-bg-tertiary text-center">
                <div className="text-lg font-bold text-sentinel-accent-cyan">
                  {learningData?.stats?.total_trades || 0}
                </div>
                <div className="text-[10px] text-sentinel-text-muted">Trades</div>
              </div>
              <div className="p-2 rounded-lg bg-sentinel-bg-tertiary text-center">
                <div className={`text-lg font-bold ${(learningData?.stats?.win_rate || 0) >= 50 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}`}>
                  {learningData?.stats?.win_rate?.toFixed(0) || 0}%
                </div>
                <div className="text-[10px] text-sentinel-text-muted">Win Rate</div>
              </div>
              <div className="p-2 rounded-lg bg-sentinel-bg-tertiary text-center">
                <div className="text-lg font-bold text-sentinel-accent-emerald">
                  €{(learningData?.stats?.best_trade || 0).toFixed(2)}
                </div>
                <div className="text-[10px] text-sentinel-text-muted">Best</div>
              </div>
              <div className="p-2 rounded-lg bg-sentinel-bg-tertiary text-center">
                <div className="text-lg font-bold text-sentinel-accent-crimson">
                  €{Math.abs(learningData?.stats?.worst_trade || 0).toFixed(2)}
                </div>
                <div className="text-[10px] text-sentinel-text-muted">Worst</div>
              </div>
            </div>

            {/* Best Learned Strategies */}
            <div className="mb-3">
              <h3 className="text-xs font-medium text-sentinel-text-secondary mb-2">Learned Strategies by Market Regime</h3>
              {!learningData?.learning?.best_strategies || Object.keys(learningData.learning.best_strategies).length === 0 ? (
                <div className="text-sm text-sentinel-text-muted py-3 text-center">
                  <Brain className="w-6 h-6 mx-auto mb-1 opacity-30 animate-pulse" />
                  <span className="text-xs">AI is learning from market data...</span>
                </div>
              ) : (
                <div className="space-y-1.5 max-h-28 overflow-y-auto">
                  {Object.entries(learningData.learning.best_strategies).slice(0, 6).map(([regime, strategy]: [string, any], idx) => (
                    <div key={idx} className="flex items-center justify-between p-1.5 rounded bg-sentinel-bg-tertiary">
                      <span className="text-xs font-mono text-sentinel-text-muted capitalize">{regime.replace('_', ' ')}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs px-1.5 py-0.5 rounded bg-sentinel-accent-cyan/20 text-sentinel-accent-cyan capitalize">
                          {strategy.action}
                        </span>
                        <span className="text-xs text-sentinel-text-muted">{strategy.confidence?.toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Top Learned Patterns */}
            {learningData?.learning?.top_patterns && learningData.learning.top_patterns.length > 0 && (
              <div className="mb-3">
                <h3 className="text-xs font-medium text-sentinel-text-secondary mb-2">Top Technical Patterns</h3>
                <div className="space-y-1 max-h-20 overflow-y-auto">
                  {learningData.learning.top_patterns.slice(0, 3).map((pattern: any, idx: number) => (
                    <div key={idx} className="flex items-center justify-between p-1.5 rounded bg-sentinel-bg-tertiary">
                      <span className="text-xs font-mono truncate max-w-[120px]">{pattern.pattern}</span>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs ${pattern.outcome === 'up' ? 'text-sentinel-accent-emerald' : pattern.outcome === 'down' ? 'text-sentinel-accent-crimson' : 'text-sentinel-text-muted'}`}>
                          {pattern.outcome?.toUpperCase()}
                        </span>
                        <span className="text-xs text-sentinel-accent-amber">{pattern.success_rate?.toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recent Learning Events */}
            <div>
              <h3 className="text-xs font-medium text-sentinel-text-secondary mb-2">Recent Learning</h3>
              {!learningData?.history?.length ? (
                <div className="text-xs text-sentinel-text-muted py-2">
                  Continuous learning active...
                </div>
              ) : (
                <div className="space-y-1 max-h-16 overflow-y-auto">
                  {learningData.history.slice(0, 4).map((event: any, idx: number) => (
                    <div key={idx} className="text-xs flex items-center justify-between py-0.5">
                      <span className="text-sentinel-text-muted truncate max-w-[150px]">{event.type || 'learn'}: {event.state?.substring(0, 15)}</span>
                      <span className={event.reward > 0 ? 'text-sentinel-accent-emerald' : event.reward < 0 ? 'text-sentinel-accent-crimson' : 'text-sentinel-text-muted'}>
                        {event.reward > 0 ? '+' : ''}{event.reward?.toFixed(2) || '0'}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="mt-3 pt-3 border-t border-sentinel-border text-[10px] text-sentinel-text-muted">
              Multi-Source AI: Historical Data + Market Movements + News Sentiment + Technical Patterns + Trade Outcomes
            </div>
          </motion.div>
        </div>

        {/* Asset Balances */}
        {balance?.coins && balance.coins.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-6 p-6 rounded-2xl glass-card"
          >
            <h2 className="text-lg font-semibold mb-6">Asset Balances</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {balance.coins.map((coin, idx) => (
                <div key={idx} className="p-4 rounded-xl bg-sentinel-bg-tertiary">
                  <div className="font-mono font-semibold text-sentinel-accent-cyan">{coin.coin}</div>
                  <div className="text-lg font-bold mt-1">{coin.balance?.toFixed(4)}</div>
                  <div className="text-sm text-sentinel-text-muted">€{coin.usdValue?.toFixed(2)}</div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-sentinel-border mt-8">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between text-sm text-sentinel-text-muted">
          <span>SENTINEL AI - Autonomous Digital Trader</span>
          <span>Developed by NoLimitDevelopments</span>
        </div>
      </footer>

      {/* Trade Notifications - Bottom Right */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3">
        {notifications.map((notification) => (
          <motion.div
            key={notification.id}
            initial={{ opacity: 0, x: 100, scale: 0.8 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 100, scale: 0.8 }}
            className={`p-4 rounded-xl shadow-2xl backdrop-blur-sm border min-w-[280px] ${
              notification.isWin 
                ? 'bg-sentinel-accent-emerald/20 border-sentinel-accent-emerald/50' 
                : 'bg-sentinel-accent-crimson/20 border-sentinel-accent-crimson/50'
            }`}
          >
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${
                notification.isWin 
                  ? 'bg-sentinel-accent-emerald/30' 
                  : 'bg-sentinel-accent-crimson/30'
              }`}>
                {notification.isWin 
                  ? <TrendingUp className="w-5 h-5 text-sentinel-accent-emerald" />
                  : <TrendingDown className="w-5 h-5 text-sentinel-accent-crimson" />
                }
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-sentinel-text-primary">
                    {notification.symbol}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    notification.isWin 
                      ? 'bg-sentinel-accent-emerald/30 text-sentinel-accent-emerald' 
                      : 'bg-sentinel-accent-crimson/30 text-sentinel-accent-crimson'
                  }`}>
                    {notification.isWin ? 'WIN' : 'LOSS'}
                  </span>
                </div>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-xs text-sentinel-text-secondary">
                    Position Closed
                  </span>
                  <span className={`font-bold ${
                    notification.isWin 
                      ? 'text-sentinel-accent-emerald' 
                      : 'text-sentinel-accent-crimson'
                  }`}>
                    {notification.isWin ? '+' : ''}€{notification.pnl.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
            {/* Progress bar for timeout */}
            <div className="mt-2 h-1 bg-sentinel-bg-tertiary rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: '100%' }}
                animate={{ width: '0%' }}
                transition={{ duration: 5, ease: 'linear' }}
                className={`h-full ${
                  notification.isWin 
                    ? 'bg-sentinel-accent-emerald' 
                    : 'bg-sentinel-accent-crimson'
                }`}
              />
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
