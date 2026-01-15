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
  RefreshCw
} from 'lucide-react'
import Link from 'next/link'

// Mock data - in production this comes from API
const mockDashboardData = {
  balance: {
    total: 54892.45,
    change24h: 2.34,
    changeAmount: 1284.32
  },
  performance: {
    today: 487.23,
    todayPercent: 0.89,
    week: 2341.87,
    weekPercent: 4.45,
    month: 8934.12,
    monthPercent: 19.42
  },
  aiStatus: {
    active: true,
    confidence: 0.78,
    regime: 'sideways',
    strategy: 'Grid Master',
    insight: 'Market consolidating near key support. Maintaining grid positions with reduced exposure.',
    lastAnalysis: '2 min ago'
  },
  riskStatus: {
    status: 'SAFE',
    todayLoss: 0.42,
    maxLoss: 5.0,
    exposure: 23.5,
    maxExposure: 30.0,
    positionsCount: 3
  },
  positions: [
    { symbol: 'BTCUSDT', side: 'long', entry: 42150.00, current: 42890.00, pnl: 324.50, pnlPercent: 1.76 },
    { symbol: 'ETHUSDT', side: 'long', entry: 2245.00, current: 2312.00, pnl: 187.25, pnlPercent: 2.98 },
    { symbol: 'SOLUSDT', side: 'short', entry: 98.50, current: 96.20, pnl: 115.00, pnlPercent: 2.34 },
  ],
  recentTrades: [
    { symbol: 'BTCUSDT', side: 'buy', pnl: 156.32, time: '14:32' },
    { symbol: 'ETHUSDT', side: 'sell', pnl: -42.18, time: '13:15' },
    { symbol: 'BNBUSDT', side: 'buy', pnl: 89.45, time: '11:47' },
  ]
}

export default function DashboardPage() {
  const [data, setData] = useState(mockDashboardData)
  const [isLoading, setIsLoading] = useState(false)

  const refreshData = () => {
    setIsLoading(true)
    // Simulate API call
    setTimeout(() => setIsLoading(false), 1000)
  }

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
                <span className="text-xs text-sentinel-text-muted">AI Active</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button 
              onClick={refreshData}
              className={`p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors ${isLoading ? 'animate-spin' : ''}`}
            >
              <RefreshCw className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
            <button className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors relative">
              <Bell className="w-5 h-5 text-sentinel-text-secondary" />
              <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-sentinel-accent-crimson" />
            </button>
            <Link href="/admin" className="px-3 py-1.5 rounded-lg bg-sentinel-accent-crimson/20 text-sentinel-accent-crimson text-sm font-medium hover:bg-sentinel-accent-crimson/30 transition-colors">
              Admin
            </Link>
            <button className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors">
              <Settings className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
            <div className="w-px h-8 bg-sentinel-border" />
            <button className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors">
              <LogOut className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-6 py-8">
        {/* Top Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Balance Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0 }}
            className="col-span-1 lg:col-span-2 p-6 rounded-2xl glass-card"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-xl bg-sentinel-accent-cyan/10">
                  <Wallet className="w-6 h-6 text-sentinel-accent-cyan" />
                </div>
                <span className="text-sentinel-text-secondary">Total Balance</span>
              </div>
              <div className={`flex items-center gap-1 px-3 py-1 rounded-full ${
                data.balance.change24h >= 0 ? 'status-safe' : 'status-danger'
              }`}>
                {data.balance.change24h >= 0 ? (
                  <TrendingUp className="w-4 h-4" />
                ) : (
                  <TrendingDown className="w-4 h-4" />
                )}
                <span className="text-sm font-medium">
                  {data.balance.change24h >= 0 ? '+' : ''}{data.balance.change24h}%
                </span>
              </div>
            </div>
            <div className="flex items-end gap-4">
              <span className="text-4xl font-display font-bold">
                ${data.balance.total.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </span>
              <span className={`text-lg mb-1 ${data.balance.change24h >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}`}>
                {data.balance.change24h >= 0 ? '+' : ''}${data.balance.changeAmount.toLocaleString()}
              </span>
            </div>
          </motion.div>

          {/* Today P&L */}
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
              <span className="text-sentinel-text-secondary">Today</span>
            </div>
            <div className="text-2xl font-display font-bold text-sentinel-accent-emerald">
              +${data.performance.today.toLocaleString()}
            </div>
            <div className="text-sm text-sentinel-text-muted mt-1">
              +{data.performance.todayPercent}% profit
            </div>
          </motion.div>

          {/* Week P&L */}
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
              <span className="text-sentinel-text-secondary">This Week</span>
            </div>
            <div className="text-2xl font-display font-bold text-sentinel-accent-emerald">
              +${data.performance.week.toLocaleString()}
            </div>
            <div className="text-sm text-sentinel-text-muted mt-1">
              +{data.performance.weekPercent}% profit
            </div>
          </motion.div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - AI Status & Risk */}
          <div className="space-y-6">
            {/* AI Status Card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="p-6 rounded-2xl glass-card"
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-sentinel-accent-violet/10">
                    <Brain className="w-6 h-6 text-sentinel-accent-violet" />
                  </div>
                  <div>
                    <div className="font-semibold">AI Status</div>
                    <div className="text-xs text-sentinel-text-muted">{data.aiStatus.lastAnalysis}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-sentinel-accent-emerald live-pulse" />
                  <span className="text-sm text-sentinel-accent-emerald font-medium">Active</span>
                </div>
              </div>

              {/* Confidence Meter */}
              <div className="mb-6">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-sentinel-text-secondary">AI Confidence</span>
                  <span className="text-sentinel-text-primary font-mono">{(data.aiStatus.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 rounded-full bg-sentinel-bg-tertiary overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${data.aiStatus.confidence * 100}%` }}
                    transition={{ duration: 1, delay: 0.5 }}
                    className="h-full rounded-full bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald"
                  />
                </div>
              </div>

              {/* Strategy Info */}
              <div className="space-y-3 mb-6">
                <div className="flex justify-between">
                  <span className="text-sentinel-text-secondary">Market Regime</span>
                  <span className="text-sentinel-text-primary font-medium capitalize">{data.aiStatus.regime}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sentinel-text-secondary">Active Strategy</span>
                  <span className="text-sentinel-accent-cyan font-medium">{data.aiStatus.strategy}</span>
                </div>
              </div>

              {/* AI Insight */}
              <div className="p-4 rounded-xl bg-sentinel-bg-tertiary border border-sentinel-border">
                <div className="text-xs text-sentinel-text-muted mb-2">AI INSIGHT</div>
                <p className="text-sm text-sentinel-text-primary leading-relaxed">
                  {data.aiStatus.insight}
                </p>
              </div>
            </motion.div>

            {/* Risk Status Card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="p-6 rounded-2xl glass-card"
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-sentinel-accent-emerald/10">
                    <Shield className="w-6 h-6 text-sentinel-accent-emerald" />
                  </div>
                  <div className="font-semibold">Risk Status</div>
                </div>
                <div className={`px-3 py-1.5 rounded-lg font-mono text-sm font-bold ${
                  data.riskStatus.status === 'SAFE' ? 'status-safe' :
                  data.riskStatus.status === 'CAUTION' ? 'status-caution' : 'status-danger'
                }`}>
                  {data.riskStatus.status}
                </div>
              </div>

              <div className="space-y-4">
                {/* Daily Loss */}
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-sentinel-text-secondary">Daily Loss</span>
                    <span className="text-sentinel-text-primary">{data.riskStatus.todayLoss}% / {data.riskStatus.maxLoss}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-sentinel-bg-tertiary overflow-hidden">
                    <div 
                      className="h-full rounded-full bg-sentinel-accent-emerald"
                      style={{ width: `${(data.riskStatus.todayLoss / data.riskStatus.maxLoss) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Exposure */}
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-sentinel-text-secondary">Exposure</span>
                    <span className="text-sentinel-text-primary">{data.riskStatus.exposure}% / {data.riskStatus.maxExposure}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-sentinel-bg-tertiary overflow-hidden">
                    <div 
                      className="h-full rounded-full bg-sentinel-accent-amber"
                      style={{ width: `${(data.riskStatus.exposure / data.riskStatus.maxExposure) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Positions */}
                <div className="flex justify-between text-sm pt-2">
                  <span className="text-sentinel-text-secondary">Active Positions</span>
                  <span className="text-sentinel-text-primary font-mono">{data.riskStatus.positionsCount}</span>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Center Column - Positions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="lg:col-span-2 p-6 rounded-2xl glass-card"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold">Active Positions</h2>
              <button className="flex items-center gap-1 text-sm text-sentinel-accent-cyan hover:underline">
                View All <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-sm text-sentinel-text-muted border-b border-sentinel-border">
                    <th className="pb-4 font-medium">Symbol</th>
                    <th className="pb-4 font-medium">Side</th>
                    <th className="pb-4 font-medium text-right">Entry</th>
                    <th className="pb-4 font-medium text-right">Current</th>
                    <th className="pb-4 font-medium text-right">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {data.positions.map((position, idx) => (
                    <motion.tr
                      key={position.symbol}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + idx * 0.1 }}
                      className="border-b border-sentinel-border/50 last:border-0"
                    >
                      <td className="py-4">
                        <span className="font-mono font-medium">{position.symbol}</span>
                      </td>
                      <td className="py-4">
                        <span className={`px-2 py-1 rounded text-xs font-medium uppercase ${
                          position.side === 'long' ? 'bg-sentinel-accent-emerald/10 text-sentinel-accent-emerald' :
                          'bg-sentinel-accent-crimson/10 text-sentinel-accent-crimson'
                        }`}>
                          {position.side}
                        </span>
                      </td>
                      <td className="py-4 text-right font-mono text-sentinel-text-secondary">
                        ${position.entry.toLocaleString()}
                      </td>
                      <td className="py-4 text-right font-mono">
                        ${position.current.toLocaleString()}
                      </td>
                      <td className="py-4 text-right">
                        <div className={`font-mono font-medium ${
                          position.pnl >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'
                        }`}>
                          {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)}
                        </div>
                        <div className={`text-xs ${
                          position.pnl >= 0 ? 'text-sentinel-accent-emerald/70' : 'text-sentinel-accent-crimson/70'
                        }`}>
                          {position.pnl >= 0 ? '+' : ''}{position.pnlPercent}%
                        </div>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Recent Trades */}
            <div className="mt-8 pt-6 border-t border-sentinel-border">
              <h3 className="text-sm font-medium text-sentinel-text-secondary mb-4">Recent Trades</h3>
              <div className="space-y-3">
                {data.recentTrades.map((trade, idx) => (
                  <div key={idx} className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                        trade.pnl >= 0 ? 'bg-sentinel-accent-emerald/10' : 'bg-sentinel-accent-crimson/10'
                      }`}>
                        {trade.pnl >= 0 ? (
                          <TrendingUp className="w-4 h-4 text-sentinel-accent-emerald" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-sentinel-accent-crimson" />
                        )}
                      </div>
                      <div>
                        <div className="font-mono text-sm">{trade.symbol}</div>
                        <div className="text-xs text-sentinel-text-muted">{trade.time}</div>
                      </div>
                    </div>
                    <div className={`font-mono font-medium ${
                      trade.pnl >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'
                    }`}>
                      {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
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

