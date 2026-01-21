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
  Euro
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
  leverage: string
  positionValue: string
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

// EUR/USD rate (approximate)
const USDT_TO_EUR = 0.92

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
  const [fearGreed, setFearGreed] = useState<number>(50)
  const [aiConfidence, setAiConfidence] = useState<number>(0)
  const [pairsScanned, setPairsScanned] = useState<number>(0)
  const [recentTrades, setRecentTrades] = useState<TradeHistory[]>([])
  const [traderStats, setTraderStats] = useState<TraderStats | null>(null)
  const [pnlHistory, setPnlHistory] = useState<{time: string, pnl: number}[]>([])
  const [last100Pnl, setLast100Pnl] = useState<number>(0)
  
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
        fetch('/ai/market/whale-alerts?limit=10'),
        fetch('/ai/exchange/trades/history?user_id=default&limit=100'),
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
        <div className="w-full px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Logo size="md" />
              
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
                isTrading ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-amber-500/10 border border-amber-500/30'
              }`}>
                <div className={`w-2 h-2 rounded-full ${isTrading ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                <span className={`text-xs font-medium ${isTrading ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {isTrading ? 'Trading Active' : 'Paused'}
                </span>
              </div>

              {/* Market Info */}
              <div className="hidden lg:flex items-center gap-4 px-4 py-1.5 bg-white/[0.02] rounded-lg border border-white/5">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500">F/G</span>
                  <span className={`text-xs font-bold ${fearGreed <= 40 ? 'text-red-400' : fearGreed >= 60 ? 'text-emerald-400' : 'text-gray-400'}`}>
                    {fearGreed}
                  </span>
                </div>
                <div className="w-px h-4 bg-white/10" />
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500">Regime</span>
                  <span className="text-xs text-cyan-400 uppercase">{marketRegime.replace('_', ' ')}</span>
                </div>
                <div className="w-px h-4 bg-white/10" />
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500">AI</span>
                  <span className="text-xs text-white">{aiConfidence}%</span>
                </div>
                <div className="w-px h-4 bg-white/10" />
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500">Pairs</span>
                  <span className="text-xs text-white">{pairsScanned}</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Link href="/dashboard/settings" className="p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
                <Settings className="w-5 h-5 text-gray-400" />
              </Link>
              <Link href="/dashboard/backtest" className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-sm text-gray-400">
                Backtest
              </Link>
              {isAdmin && (
                <Link href="/admin" className="px-4 py-2 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 transition-colors text-sm text-cyan-400">
                  Admin
                </Link>
              )}
              <button onClick={handleLogout} className="p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
                <LogOut className="w-5 h-5 text-gray-400" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="p-6 relative">
        {/* Top Stats - 5 columns */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
          {/* Balance */}
          <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <Wallet className="w-4 h-4 text-cyan-400" />
              <span className="text-xs text-gray-500">Balance</span>
            </div>
            <div className="text-xl font-bold text-white">{balanceUSDT.toFixed(2)} USDT</div>
            <div className="flex items-center gap-1 text-sm text-gray-400 mt-1">
              <Euro className="w-3 h-3" />
              <span>{balanceEUR.toFixed(2)} EUR</span>
            </div>
          </div>

          {/* Daily P&L */}
          <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-gray-400" />
              <span className="text-xs text-gray-500">Daily P&L</span>
            </div>
            <div className={`text-xl font-bold ${dailyPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {dailyPnl >= 0 ? '+' : ''}{dailyPnl.toFixed(2)} €
            </div>
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
            <div className="text-xs text-gray-500">{traderStats?.winning_trades || 0}W / {(traderStats?.total_trades || 0) - (traderStats?.winning_trades || 0)}L</div>
          </div>

          {/* Total Trades */}
          <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <Hash className="w-4 h-4 text-violet-400" />
              <span className="text-xs text-gray-500">Total Trades</span>
            </div>
            <div className="text-xl font-bold text-white">{traderStats?.total_trades || 0}</div>
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
              className={`w-full py-2.5 rounded-lg font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
                isTrading 
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30' 
                  : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/30'
              }`}
            >
              {isTogglingBot ? <Loader2 className="w-4 h-4 animate-spin" /> : isTrading ? <><Square className="w-4 h-4" /> Stop</> : <><Play className="w-4 h-4" /> Start</>}
            </button>
          </div>
        </div>

        {/* OPEN POSITIONS - FIRST AND LARGE */}
        <div className="grid lg:grid-cols-4 gap-4 mb-6">
          <div className="lg:col-span-3 bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="p-4 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">Open Positions</h2>
                <span className="text-xs text-gray-500">({positions.length})</span>
              </div>
              <button onClick={loadDashboardData} className="p-1.5 rounded-lg hover:bg-white/5">
                <RefreshCw className="w-4 h-4 text-gray-500" />
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
                      
                      // Use API unrealisedPnl if available
                      const apiPnl = parseFloat(pos.unrealisedPnl || '0')
                      const finalPnlEUR = apiPnl !== 0 ? apiPnl * USDT_TO_EUR : pnlEUR
                      const finalPnlPercent = apiPnl !== 0 && posValueUSDT > 0 ? (apiPnl / posValueUSDT) * 100 : pnlPercent
                      
                      return (
                        <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                          <td className="px-3 py-2">
                            <span className="font-medium text-white text-sm">{pos.symbol.replace('USDT', '')}</span>
                            <span className="text-gray-600 text-[10px]">/USDT</span>
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
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Market News */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <h2 className="font-medium text-white text-sm">Market News</h2>
            </div>
            <div className="overflow-y-auto divide-y divide-white/5" style={{ maxHeight: '360px' }}>
              {news.length === 0 ? (
                <div className="p-4 text-gray-600 text-xs text-center">Loading news...</div>
              ) : (
                news.map((item, i) => (
                  <div key={i} className="px-3 py-2 hover:bg-white/[0.02]">
                    <div className="flex items-start gap-2">
                      <span className={`text-[9px] px-1.5 py-0.5 rounded flex-shrink-0 mt-0.5 ${
                        item.sentiment === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' :
                        item.sentiment === 'bearish' ? 'bg-red-500/20 text-red-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {item.sentiment === 'bullish' ? '↑' : item.sentiment === 'bearish' ? '↓' : '•'}
                      </span>
                      <div>
                        <p className="text-[11px] text-gray-300 leading-tight">{item.title}</p>
                        <p className="text-[9px] text-gray-600 mt-1">{item.source}</p>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* P&L Performance Chart + Performance Stats */}
        <div className="grid lg:grid-cols-3 gap-4 mb-6">
          {/* P&L Chart - Last 100 Trades */}
          <div className="lg:col-span-2 bg-white/[0.02] rounded-xl border border-white/5 p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <LineChart className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">P&L Performance</h2>
                <span className="text-xs text-gray-500">(Last {recentTrades.length} Trades)</span>
              </div>
              <div className={`text-lg font-bold ${last100Pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
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
          <div className="bg-white/[0.02] rounded-xl border border-white/5 p-5">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              <h2 className="font-semibold text-white">Performance</h2>
              <span className="text-xs text-gray-500">(All Time)</span>
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
        <div className="grid lg:grid-cols-3 gap-4">
          {/* Recent Trades */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="p-4 border-b border-white/5 flex items-center gap-2">
              <Clock className="w-5 h-5 text-cyan-400" />
              <h2 className="font-semibold text-white text-sm">Recent Trades</h2>
            </div>
            
            <div className="overflow-y-auto max-h-56">
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
            <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isTrading ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'}`} />
              <h2 className="font-medium text-white text-sm">AI Console</h2>
            </div>
            <div ref={consoleRef} className="h-56 overflow-y-auto p-3 font-mono text-[10px] space-y-1">
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

          {/* Whale Activity */}
          <div className="bg-white/[0.02] rounded-xl border border-white/5 overflow-hidden">
            <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
              <Waves className="w-4 h-4 text-cyan-400" />
              <h2 className="font-medium text-white text-sm">Whale Activity</h2>
            </div>
            <div className="h-56 overflow-y-auto divide-y divide-white/5">
              {whaleAlerts.length === 0 ? (
                <div className="p-4 text-gray-600 text-xs text-center">No whale activity</div>
              ) : (
                whaleAlerts.map((alert, i) => (
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
                      <span className="text-[9px] text-gray-600">{new Date(alert.timestamp).toLocaleTimeString()}</span>
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
