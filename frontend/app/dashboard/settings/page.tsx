'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Settings, 
  Shield, 
  Zap, 
  TrendingUp,
  AlertTriangle,
  Save,
  RotateCcw,
  ArrowLeft,
  DollarSign,
  Percent,
  Target,
  StopCircle,
  Wallet,
  PieChart,
  Loader2,
  CheckCircle,
  XCircle,
  Activity,
  BarChart3,
  Coins,
  Building2
} from 'lucide-react'
import Link from 'next/link'
import Image from 'next/image'

interface BotSettings {
  // Risk Mode
  riskMode: 'safe' | 'neutral' | 'aggressive'
  
  // Trading Parameters
  takeProfitPercent: number
  stopLossPercent: number
  trailingStopPercent: number
  minProfitToTrail: number
  minConfidence: number
  
  // Position Sizing
  maxPositionPercent: number
  maxOpenPositions: number
  
  // Budget Allocation
  totalBudget: number
  cryptoBudget: number
  tradFiBudget: number
  
  // Markets
  enableCrypto: boolean
  enableTradFi: boolean
  
  // AI Settings
  useAiSignals: boolean
  learnFromTrades: boolean
}

const defaultSettings: BotSettings = {
  riskMode: 'neutral',
  takeProfitPercent: 2.0,
  stopLossPercent: 1.5,
  trailingStopPercent: 1.2,
  minProfitToTrail: 1.0,
  minConfidence: 70,
  maxPositionPercent: 8,
  maxOpenPositions: 8,
  totalBudget: 150,
  cryptoBudget: 100,
  tradFiBudget: 50,
  enableCrypto: true,
  enableTradFi: false,
  useAiSignals: true,
  learnFromTrades: true,
}

const riskPresets = {
  safe: {
    takeProfitPercent: 1.0,
    stopLossPercent: 0.5,
    trailingStopPercent: 0.8,
    minProfitToTrail: 0.5,
    minConfidence: 80,
    maxPositionPercent: 5,
    maxOpenPositions: 5,
  },
  neutral: {
    takeProfitPercent: 2.0,
    stopLossPercent: 1.5,
    trailingStopPercent: 1.2,
    minProfitToTrail: 1.0,
    minConfidence: 70,
    maxPositionPercent: 8,
    maxOpenPositions: 8,
  },
  aggressive: {
    takeProfitPercent: 5.0,
    stopLossPercent: 3.0,
    trailingStopPercent: 2.0,
    minProfitToTrail: 2.0,
    minConfidence: 55,
    maxPositionPercent: 15,
    maxOpenPositions: 15,
  }
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<BotSettings>(defaultSettings)
  const [isSaving, setIsSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [isSellingAll, setIsSellingAll] = useState(false)
  const [sellAllStatus, setSellAllStatus] = useState<string | null>(null)
  const [equity, setEquity] = useState<number>(0)

  useEffect(() => {
    loadSettings()
    loadEquity()
  }, [])

  const loadSettings = async () => {
    try {
      const res = await fetch('/ai/exchange/settings')
      const data = await res.json()
      if (data.success && data.data) {
        setSettings(prev => ({ ...prev, ...data.data }))
      }
    } catch (error) {
      console.error('Failed to load settings:', error)
    }
  }

  const loadEquity = async () => {
    try {
      const res = await fetch('/ai/exchange/balance')
      const data = await res.json()
      if (data.success) {
        setEquity(data.data.totalEquity || 0)
        setSettings(prev => ({
          ...prev,
          totalBudget: Math.round(data.data.totalEquity || 0)
        }))
      }
    } catch (error) {
      console.error('Failed to load equity:', error)
    }
  }

  const saveSettings = async () => {
    setIsSaving(true)
    setSaveStatus('idle')
    
    try {
      const res = await fetch('/ai/exchange/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      })
      
      const data = await res.json()
      
      if (data.success) {
        setSaveStatus('success')
        setTimeout(() => setSaveStatus('idle'), 3000)
      } else {
        setSaveStatus('error')
      }
    } catch (error) {
      setSaveStatus('error')
    } finally {
      setIsSaving(false)
    }
  }

  const sellAll = async () => {
    if (!confirm('Are you sure you want to SELL ALL positions immediately? This cannot be undone!')) {
      return
    }
    
    setIsSellingAll(true)
    setSellAllStatus('Closing all positions...')
    
    try {
      const res = await fetch('/ai/exchange/sell-all', {
        method: 'POST'
      })
      
      const data = await res.json()
      
      if (data.success) {
        setSellAllStatus(`Closed ${data.data.closedCount} positions. Total P&L: €${data.data.totalPnl?.toFixed(2)}`)
      } else {
        setSellAllStatus(`Error: ${data.error}`)
      }
    } catch (error) {
      setSellAllStatus('Failed to close positions')
    } finally {
      setIsSellingAll(false)
    }
  }

  const applyRiskPreset = (mode: 'safe' | 'neutral' | 'aggressive') => {
    setSettings(prev => ({
      ...prev,
      riskMode: mode,
      ...riskPresets[mode]
    }))
  }

  const updateBudgetAllocation = (crypto: number) => {
    const tradFi = settings.totalBudget - crypto
    setSettings(prev => ({
      ...prev,
      cryptoBudget: crypto,
      tradFiBudget: Math.max(0, tradFi)
    }))
  }

  return (
    <div className="min-h-screen bg-sentinel-bg-primary">
      {/* Header */}
      <nav className="sticky top-0 z-50 glass-card border-b border-sentinel-border">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <Image 
              src="/logo.png" 
              alt="Sentinel Logo" 
              width={36} 
              height={36} 
              className="rounded-lg"
            />
            <div>
              <h1 className="font-display font-bold text-xl">Bot Settings</h1>
              <p className="text-xs text-sentinel-text-muted">Configure your trading parameters</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={sellAll}
              disabled={isSellingAll}
              className="px-4 py-2 rounded-xl bg-sentinel-accent-crimson/20 text-sentinel-accent-crimson 
                         font-semibold hover:bg-sentinel-accent-crimson/30 transition-all flex items-center gap-2
                         disabled:opacity-50"
            >
              {isSellingAll ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <StopCircle className="w-4 h-4" />
              )}
              SELL ALL
            </button>
            
            <button
              onClick={saveSettings}
              disabled={isSaving}
              className="px-6 py-2 rounded-xl bg-sentinel-accent-emerald font-semibold 
                         hover:bg-sentinel-accent-emerald/80 transition-all flex items-center gap-2
                         disabled:opacity-50"
            >
              {isSaving ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : saveStatus === 'success' ? (
                <CheckCircle className="w-4 h-4" />
              ) : saveStatus === 'error' ? (
                <XCircle className="w-4 h-4" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              {saveStatus === 'success' ? 'Saved!' : saveStatus === 'error' ? 'Error' : 'Save Settings'}
            </button>
          </div>
        </div>
      </nav>

      {/* Sell All Status */}
      {sellAllStatus && (
        <div className="max-w-6xl mx-auto px-6 pt-4">
          <div className={`p-4 rounded-xl ${
            sellAllStatus.includes('Error') || sellAllStatus.includes('Failed') 
              ? 'bg-sentinel-accent-crimson/10 border border-sentinel-accent-crimson/30' 
              : 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30'
          }`}>
            {sellAllStatus}
          </div>
        </div>
      )}

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* Risk Mode Selection */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 rounded-2xl glass-card"
        >
          <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <Shield className="w-5 h-5 text-sentinel-accent-cyan" />
            Risk Mode
          </h2>
          
          <div className="grid grid-cols-3 gap-4">
            {/* SAFE Mode */}
            <button
              onClick={() => applyRiskPreset('safe')}
              className={`p-6 rounded-xl border-2 transition-all ${
                settings.riskMode === 'safe'
                  ? 'border-sentinel-accent-emerald bg-sentinel-accent-emerald/10'
                  : 'border-sentinel-border hover:border-sentinel-accent-emerald/50'
              }`}
            >
              <div className="flex flex-col items-center gap-3">
                <div className={`p-4 rounded-full ${
                  settings.riskMode === 'safe' ? 'bg-sentinel-accent-emerald/20' : 'bg-sentinel-bg-tertiary'
                }`}>
                  <Shield className={`w-8 h-8 ${
                    settings.riskMode === 'safe' ? 'text-sentinel-accent-emerald' : 'text-sentinel-text-muted'
                  }`} />
                </div>
                <div className="text-center">
                  <h3 className="font-bold text-lg">SAFE</h3>
                  <p className="text-sm text-sentinel-text-muted mt-1">
                    Small profits, minimal risk
                  </p>
                  <div className="mt-3 text-xs space-y-1 text-sentinel-text-secondary">
                    <p>Take Profit: +1%</p>
                    <p>Stop Loss: -0.5%</p>
                    <p>Max 5 positions</p>
                  </div>
                </div>
              </div>
            </button>

            {/* NEUTRAL Mode */}
            <button
              onClick={() => applyRiskPreset('neutral')}
              className={`p-6 rounded-xl border-2 transition-all ${
                settings.riskMode === 'neutral'
                  ? 'border-sentinel-accent-amber bg-sentinel-accent-amber/10'
                  : 'border-sentinel-border hover:border-sentinel-accent-amber/50'
              }`}
            >
              <div className="flex flex-col items-center gap-3">
                <div className={`p-4 rounded-full ${
                  settings.riskMode === 'neutral' ? 'bg-sentinel-accent-amber/20' : 'bg-sentinel-bg-tertiary'
                }`}>
                  <Activity className={`w-8 h-8 ${
                    settings.riskMode === 'neutral' ? 'text-sentinel-accent-amber' : 'text-sentinel-text-muted'
                  }`} />
                </div>
                <div className="text-center">
                  <h3 className="font-bold text-lg">NEUTRAL</h3>
                  <p className="text-sm text-sentinel-text-muted mt-1">
                    Balanced risk/reward
                  </p>
                  <div className="mt-3 text-xs space-y-1 text-sentinel-text-secondary">
                    <p>Take Profit: +2%</p>
                    <p>Stop Loss: -1.5%</p>
                    <p>Max 8 positions</p>
                  </div>
                </div>
              </div>
            </button>

            {/* AGGRESSIVE Mode */}
            <button
              onClick={() => applyRiskPreset('aggressive')}
              className={`p-6 rounded-xl border-2 transition-all ${
                settings.riskMode === 'aggressive'
                  ? 'border-sentinel-accent-crimson bg-sentinel-accent-crimson/10'
                  : 'border-sentinel-border hover:border-sentinel-accent-crimson/50'
              }`}
            >
              <div className="flex flex-col items-center gap-3">
                <div className={`p-4 rounded-full ${
                  settings.riskMode === 'aggressive' ? 'bg-sentinel-accent-crimson/20' : 'bg-sentinel-bg-tertiary'
                }`}>
                  <Zap className={`w-8 h-8 ${
                    settings.riskMode === 'aggressive' ? 'text-sentinel-accent-crimson' : 'text-sentinel-text-muted'
                  }`} />
                </div>
                <div className="text-center">
                  <h3 className="font-bold text-lg">AGGRESSIVE</h3>
                  <p className="text-sm text-sentinel-text-muted mt-1">
                    High risk, high reward
                  </p>
                  <div className="mt-3 text-xs space-y-1 text-sentinel-text-secondary">
                    <p>Take Profit: +5%</p>
                    <p>Stop Loss: -3%</p>
                    <p>Max 15 positions</p>
                  </div>
                </div>
              </div>
            </button>
          </div>
        </motion.section>

        {/* Trading Parameters */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="p-6 rounded-2xl glass-card"
        >
          <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <Target className="w-5 h-5 text-sentinel-accent-cyan" />
            Trading Parameters
          </h2>
          
          <div className="grid grid-cols-2 gap-6">
            {/* Take Profit */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Take Profit %
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={settings.takeProfitPercent}
                  onChange={(e) => setSettings(prev => ({ 
                    ...prev, 
                    takeProfitPercent: parseFloat(e.target.value),
                    riskMode: 'neutral' // Custom settings
                  }))}
                  className="flex-1 accent-sentinel-accent-emerald"
                />
                <span className="w-16 text-right font-mono text-sentinel-accent-emerald">
                  +{settings.takeProfitPercent}%
                </span>
              </div>
              <p className="text-xs text-sentinel-text-muted mt-1">
                Sell when profit reaches this level
              </p>
            </div>

            {/* Stop Loss */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Stop Loss %
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0.3"
                  max="5"
                  step="0.1"
                  value={settings.stopLossPercent}
                  onChange={(e) => setSettings(prev => ({ 
                    ...prev, 
                    stopLossPercent: parseFloat(e.target.value),
                    riskMode: 'neutral'
                  }))}
                  className="flex-1 accent-sentinel-accent-crimson"
                />
                <span className="w-16 text-right font-mono text-sentinel-accent-crimson">
                  -{settings.stopLossPercent}%
                </span>
              </div>
              <p className="text-xs text-sentinel-text-muted mt-1">
                Sell when loss reaches this level
              </p>
            </div>

            {/* Trailing Stop */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Trailing Stop %
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0.3"
                  max="3"
                  step="0.1"
                  value={settings.trailingStopPercent}
                  onChange={(e) => setSettings(prev => ({ 
                    ...prev, 
                    trailingStopPercent: parseFloat(e.target.value),
                    riskMode: 'neutral'
                  }))}
                  className="flex-1 accent-sentinel-accent-amber"
                />
                <span className="w-16 text-right font-mono text-sentinel-accent-amber">
                  {settings.trailingStopPercent}%
                </span>
              </div>
              <p className="text-xs text-sentinel-text-muted mt-1">
                Trail price by this % from peak
              </p>
            </div>

            {/* Min Profit to Trail */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Min Profit to Activate Trail
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0.3"
                  max="3"
                  step="0.1"
                  value={settings.minProfitToTrail}
                  onChange={(e) => setSettings(prev => ({ 
                    ...prev, 
                    minProfitToTrail: parseFloat(e.target.value),
                    riskMode: 'neutral'
                  }))}
                  className="flex-1 accent-sentinel-accent-cyan"
                />
                <span className="w-16 text-right font-mono text-sentinel-accent-cyan">
                  +{settings.minProfitToTrail}%
                </span>
              </div>
              <p className="text-xs text-sentinel-text-muted mt-1">
                Trailing only activates after this profit
              </p>
            </div>

            {/* Min Confidence */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Minimum AI Confidence
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="50"
                  max="90"
                  step="5"
                  value={settings.minConfidence}
                  onChange={(e) => setSettings(prev => ({ 
                    ...prev, 
                    minConfidence: parseInt(e.target.value),
                    riskMode: 'neutral'
                  }))}
                  className="flex-1 accent-sentinel-accent-violet"
                />
                <span className="w-16 text-right font-mono text-sentinel-accent-violet">
                  {settings.minConfidence}%
                </span>
              </div>
              <p className="text-xs text-sentinel-text-muted mt-1">
                Only trade when AI is this confident
              </p>
            </div>

            {/* Max Position Size */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Max Position Size (% of portfolio)
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="2"
                  max="25"
                  step="1"
                  value={settings.maxPositionPercent}
                  onChange={(e) => setSettings(prev => ({ 
                    ...prev, 
                    maxPositionPercent: parseInt(e.target.value),
                    riskMode: 'neutral'
                  }))}
                  className="flex-1 accent-sentinel-accent-cyan"
                />
                <span className="w-16 text-right font-mono">
                  {settings.maxPositionPercent}%
                </span>
              </div>
              <p className="text-xs text-sentinel-text-muted mt-1">
                Max €{(equity * settings.maxPositionPercent / 100).toFixed(2)} per trade
              </p>
            </div>
          </div>
        </motion.section>

        {/* Budget Allocation */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="p-6 rounded-2xl glass-card"
        >
          <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <PieChart className="w-5 h-5 text-sentinel-accent-cyan" />
            Budget Allocation
          </h2>
          
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sentinel-text-secondary">Total Available</span>
              <span className="font-mono text-xl font-bold">€{equity.toFixed(2)}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6 mb-6">
            {/* Crypto Budget */}
            <div className={`p-4 rounded-xl border-2 ${
              settings.enableCrypto ? 'border-sentinel-accent-amber bg-sentinel-accent-amber/5' : 'border-sentinel-border'
            }`}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Coins className="w-5 h-5 text-sentinel-accent-amber" />
                  <span className="font-medium">Crypto Trading</span>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.enableCrypto}
                    onChange={(e) => setSettings(prev => ({ ...prev, enableCrypto: e.target.checked }))}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-sentinel-bg-tertiary rounded-full peer 
                                  peer-checked:bg-sentinel-accent-amber transition-colors"></div>
                  <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                  peer-checked:translate-x-5"></div>
                </label>
              </div>
              
              <div className="text-3xl font-bold font-mono text-sentinel-accent-amber mb-2">
                €{settings.cryptoBudget.toFixed(0)}
              </div>
              
              <input
                type="range"
                min="0"
                max={equity}
                step="10"
                value={settings.cryptoBudget}
                onChange={(e) => updateBudgetAllocation(parseFloat(e.target.value))}
                disabled={!settings.enableCrypto}
                className="w-full accent-sentinel-accent-amber disabled:opacity-50"
              />
              
              <p className="text-xs text-sentinel-text-muted mt-2">
                BTC, ETH, SOL, and 300+ altcoins
              </p>
            </div>

            {/* TradFi Budget */}
            <div className={`p-4 rounded-xl border-2 ${
              settings.enableTradFi ? 'border-sentinel-accent-cyan bg-sentinel-accent-cyan/5' : 'border-sentinel-border'
            }`}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Building2 className="w-5 h-5 text-sentinel-accent-cyan" />
                  <span className="font-medium">TradFi Markets</span>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.enableTradFi}
                    onChange={(e) => setSettings(prev => ({ ...prev, enableTradFi: e.target.checked }))}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-sentinel-bg-tertiary rounded-full peer 
                                  peer-checked:bg-sentinel-accent-cyan transition-colors"></div>
                  <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                  peer-checked:translate-x-5"></div>
                </label>
              </div>
              
              <div className="text-3xl font-bold font-mono text-sentinel-accent-cyan mb-2">
                €{settings.tradFiBudget.toFixed(0)}
              </div>
              
              <input
                type="range"
                min="0"
                max={equity}
                step="10"
                value={settings.tradFiBudget}
                onChange={(e) => updateBudgetAllocation(equity - parseFloat(e.target.value))}
                disabled={!settings.enableTradFi}
                className="w-full accent-sentinel-accent-cyan disabled:opacity-50"
              />
              
              <p className="text-xs text-sentinel-text-muted mt-2">
                Gold, Oil, Forex, Commodities
              </p>
            </div>
          </div>

          {/* Allocation Bar */}
          <div className="h-4 rounded-full bg-sentinel-bg-tertiary overflow-hidden flex">
            <div 
              className="h-full bg-sentinel-accent-amber transition-all"
              style={{ width: `${(settings.cryptoBudget / equity) * 100}%` }}
            />
            <div 
              className="h-full bg-sentinel-accent-cyan transition-all"
              style={{ width: `${(settings.tradFiBudget / equity) * 100}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-sentinel-text-muted mt-2">
            <span>Crypto: {((settings.cryptoBudget / equity) * 100).toFixed(0)}%</span>
            <span>TradFi: {((settings.tradFiBudget / equity) * 100).toFixed(0)}%</span>
          </div>
        </motion.section>

        {/* AI Settings */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="p-6 rounded-2xl glass-card"
        >
          <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-sentinel-accent-cyan" />
            AI Settings
          </h2>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
              <div>
                <h3 className="font-medium">Use AI Signals</h3>
                <p className="text-sm text-sentinel-text-muted">
                  Let AI analyze market and generate trade signals
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.useAiSignals}
                  onChange={(e) => setSettings(prev => ({ ...prev, useAiSignals: e.target.checked }))}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-sentinel-border rounded-full peer 
                                peer-checked:bg-sentinel-accent-emerald transition-colors"></div>
                <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                peer-checked:translate-x-5"></div>
              </label>
            </div>

            <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
              <div>
                <h3 className="font-medium">Learn from Trades</h3>
                <p className="text-sm text-sentinel-text-muted">
                  AI learns and improves from every trade outcome
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.learnFromTrades}
                  onChange={(e) => setSettings(prev => ({ ...prev, learnFromTrades: e.target.checked }))}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-sentinel-border rounded-full peer 
                                peer-checked:bg-sentinel-accent-emerald transition-colors"></div>
                <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                peer-checked:translate-x-5"></div>
              </label>
            </div>

            <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
              <div>
                <h3 className="font-medium">Max Open Positions</h3>
                <p className="text-sm text-sentinel-text-muted">
                  Limit diversification to manage risk
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setSettings(prev => ({ 
                    ...prev, 
                    maxOpenPositions: Math.max(1, prev.maxOpenPositions - 1) 
                  }))}
                  className="w-8 h-8 rounded-lg bg-sentinel-bg-secondary hover:bg-sentinel-border transition-colors"
                >
                  -
                </button>
                <span className="w-12 text-center font-mono text-lg">{settings.maxOpenPositions}</span>
                <button
                  onClick={() => setSettings(prev => ({ 
                    ...prev, 
                    maxOpenPositions: Math.min(20, prev.maxOpenPositions + 1) 
                  }))}
                  className="w-8 h-8 rounded-lg bg-sentinel-bg-secondary hover:bg-sentinel-border transition-colors"
                >
                  +
                </button>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Quick Actions */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="grid grid-cols-2 gap-4"
        >
          <button
            onClick={() => setSettings(defaultSettings)}
            className="p-4 rounded-xl glass-card hover:bg-sentinel-bg-tertiary transition-colors
                       flex items-center justify-center gap-2 text-sentinel-text-secondary"
          >
            <RotateCcw className="w-5 h-5" />
            Reset to Defaults
          </button>
          
          <button
            onClick={saveSettings}
            disabled={isSaving}
            className="p-4 rounded-xl bg-sentinel-accent-emerald hover:bg-sentinel-accent-emerald/80 
                       transition-colors flex items-center justify-center gap-2 font-semibold
                       disabled:opacity-50"
          >
            {isSaving ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
            Save All Settings
          </button>
        </motion.section>
      </main>

      {/* Footer */}
      <footer className="border-t border-sentinel-border mt-12 py-6">
        <div className="max-w-6xl mx-auto px-6 text-center text-sm text-sentinel-text-muted">
          Developed by NoLimitDevelopments
        </div>
      </footer>
    </div>
  )
}

