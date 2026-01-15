'use client'

import { useEffect, useState } from 'react'
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
  Bot
} from 'lucide-react'
import Link from 'next/link'

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

  useEffect(() => {
    // Check if user is logged in
    const storedUser = localStorage.getItem('sentinel_user')
    if (storedUser) {
      setUser(JSON.parse(storedUser))
    }
    
    // Load real data
    loadData()
    
    // Refresh AI insight every 30 seconds
    const interval = setInterval(() => {
      loadAIInsight()
    }, 30000)
    
    return () => clearInterval(interval)
  }, [])

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

  const loadData = async () => {
    setIsLoading(true)
    
    try {
      // Load AI insight first (always available)
      await loadAIInsight()
      
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
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-sentinel-accent-cyan to-sentinel-accent-emerald flex items-center justify-center">
                <Shield className="w-6 h-6 text-sentinel-bg-primary" strokeWidth={2.5} />
              </div>
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
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-sentinel-accent-cyan to-sentinel-accent-emerald flex items-center justify-center">
              <Shield className="w-6 h-6 text-sentinel-bg-primary" strokeWidth={2.5} />
            </div>
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
            <button className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors">
              <Settings className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
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

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-sm text-sentinel-text-muted border-b border-sentinel-border">
                      <th className="pb-4 font-medium">Symbol</th>
                      <th className="pb-4 font-medium">Side</th>
                      <th className="pb-4 font-medium text-right">Size</th>
                      <th className="pb-4 font-medium text-right">Entry</th>
                      <th className="pb-4 font-medium text-right">Mark</th>
                      <th className="pb-4 font-medium text-right">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((position, idx) => (
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
                    ))}
                  </tbody>
                </table>
              </div>
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
    </div>
  )
}
