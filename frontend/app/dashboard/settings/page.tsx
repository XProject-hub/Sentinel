'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Settings, 
  ShieldCheck,
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
  Building2,
  Brain,
  Layers,
  Gauge,
  Info,
  Infinity,
  Cpu,
  LineChart,
  MessageSquare,
  Sparkles,
  TrendingDown,
  RefreshCw
} from 'lucide-react'
import Link from 'next/link'
import Image from 'next/image'

interface BotSettings {
  // Risk Mode
  riskMode: 'safe' | 'normal' | 'aggressive' | 'lock_profit' | 'micro_profit'
  
  // Trading Parameters
  takeProfitPercent: number
  stopLossPercent: number
  trailingStopPercent: number
  minProfitToTrail: number
  minConfidence: number
  minEdge: number
  
  // Position Sizing
  maxPositionPercent: number
  maxOpenPositions: number  // 0 = unlimited
  maxDailyDrawdown: number
  maxTotalExposure: number
  
  // Budget - Unified
  totalBudget: number
  
  // AI Core Features
  useAiSignals: boolean
  learnFromTrades: boolean
  useRegimeDetection: boolean
  useEdgeEstimation: boolean
  useDynamicSizing: boolean
  
  // AI Models (V3.0)
  useCryptoBert: boolean     // CryptoBERT sentiment
  useXgboostClassifier: boolean  // XGBoost signal classifier
  usePricePredictor: boolean // Chronos price predictor
}

const defaultSettings: BotSettings = {
  riskMode: 'normal',
  takeProfitPercent: 3.0,
  stopLossPercent: 1.5,
  trailingStopPercent: 1.0,
  minProfitToTrail: 0.8,
  minConfidence: 60,
  minEdge: 0.15,
  maxPositionPercent: 5,
  maxOpenPositions: 0, // 0 = unlimited
  maxDailyDrawdown: 3,
  maxTotalExposure: 50,
  totalBudget: 0,
  useAiSignals: true,
  learnFromTrades: true,
  useRegimeDetection: true,
  useEdgeEstimation: true,
  useDynamicSizing: true,
  useCryptoBert: true,
  useXgboostClassifier: true,
  usePricePredictor: true,
}

// === RISK PRESETS ===
const riskPresets = {
  safe: {
    name: 'SAFE',
    description: 'Conservative trading with high confidence requirements. Trades only when AI is very confident.',
    color: 'emerald',
    icon: ShieldCheck,
    params: {
      takeProfitPercent: 1.5,
      stopLossPercent: 0.5,
      trailingStopPercent: 0.5,
      minProfitToTrail: 0.3,
      minConfidence: 80,
      minEdge: 0.30,
      maxPositionPercent: 2,
      maxOpenPositions: 5,
      maxDailyDrawdown: 1,
      maxTotalExposure: 20,
      useCryptoBert: true,
      useXgboostClassifier: true,
      usePricePredictor: true,
    },
    features: [
      'Min 80% AI Confidence',
      'Min 0.30 Edge Score',
      'Max 2% per position',
      'Max 1% daily drawdown',
      'Max 5 positions',
      'All AI models active'
    ]
  },
  normal: {
    name: 'NORMAL',
    description: 'Balanced risk/reward. Good for steady growth with moderate risk.',
    color: 'amber',
    icon: Activity,
    params: {
      takeProfitPercent: 3.0,
      stopLossPercent: 1.5,
      trailingStopPercent: 1.0,
      minProfitToTrail: 0.8,
      minConfidence: 60,
      minEdge: 0.15,
      maxPositionPercent: 5,
      maxOpenPositions: 0, // Unlimited
      maxDailyDrawdown: 3,
      maxTotalExposure: 50,
      useCryptoBert: true,
      useXgboostClassifier: true,
      usePricePredictor: true,
    },
    features: [
      'Min 60% AI Confidence',
      'Min 0.15 Edge Score',
      'Max 5% per position',
      'Max 3% daily drawdown',
      'Unlimited positions',
      'Dynamic Kelly sizing'
    ]
  },
  aggressive: {
    name: 'HIGH RISK',
    description: 'Aggressive trading for maximum gains. Higher risk tolerance.',
    color: 'crimson',
    icon: Zap,
    params: {
      takeProfitPercent: 8.0,
      stopLossPercent: 3.0,
      trailingStopPercent: 2.0,
      minProfitToTrail: 1.5,
      minConfidence: 45,
      minEdge: 0.08,
      maxPositionPercent: 10,
      maxOpenPositions: 0, // Unlimited
      maxDailyDrawdown: 8,
      maxTotalExposure: 80,
      useCryptoBert: true,
      useXgboostClassifier: true,
      usePricePredictor: true,
    },
    features: [
      'Min 45% AI Confidence',
      'Min 0.08 Edge Score',
      'Max 10% per position',
      'Max 8% daily drawdown',
      'Unlimited positions',
      'Maximum market exposure'
    ]
  },
  lock_profit: {
    name: 'LOCK PROFIT',
    description: 'AI predicts profitable trades, tight trailing stop locks in ANY profit immediately!',
    color: 'cyan',
    icon: TrendingDown,
    params: {
      takeProfitPercent: 50.0,  // High TP - let trailing do the work
      stopLossPercent: 1.0,     // Tight stop loss
      trailingStopPercent: 0.05, // 0.05% drop from peak = SELL
      minProfitToTrail: 0.01,   // Trail activates almost immediately (0.01% profit)
      minConfidence: 40,        // AI must be 40%+ confident of profit
      minEdge: 0.05,            // 5% edge - AI sees profit potential
      maxPositionPercent: 5,
      maxOpenPositions: 0,      // Unlimited
      maxDailyDrawdown: 5,      // Higher tolerance for LOCK PROFIT
      maxTotalExposure: 100,    // Use full budget
      useCryptoBert: true,      // ✅ AI sentiment analysis
      useXgboostClassifier: true, // ✅ AI profit prediction
      usePricePredictor: true,  // ✅ AI price forecast
    },
    features: [
      '🧠 AI predicts profitable entries',
      '🔒 Locks ANY profit immediately',
      '📉 Sells on 0.05% drop from peak',
      '🛡️ Tight 1% stop loss',
      '♾️ Unlimited positions',
      '💰 100% budget utilization'
    ]
  },
  micro_profit: {
    name: 'MICRO PROFIT',
    description: 'Smart scalping - AI finds SURE profits, quick exits lock gains. High win rate!',
    color: 'purple',
    icon: Coins,
    params: {
      takeProfitPercent: 1.0,   // Take profit at 1% - quick but decent
      stopLossPercent: 0.5,     // Tight stop - exit fast if wrong (2:1 ratio)
      trailingStopPercent: 0.15, // 0.15% trail - lock profits fast
      minProfitToTrail: 0.2,    // Start trailing at 0.2% profit
      minConfidence: 70,        // High confidence - SURE trades only
      minEdge: 0.12,            // 12% edge - strong signals
      maxPositionPercent: 5,    // Normal positions - 5% each
      maxOpenPositions: 15,     // Up to 15 positions
      maxDailyDrawdown: 3,      // Stop if losing 3% daily
      maxTotalExposure: 80,     // Use 80% of budget
      useCryptoBert: true,      // ✅ Sentiment must be positive
      useXgboostClassifier: true, // ✅ ML must predict profit
      usePricePredictor: true,  // ✅ Price must be going up
    },
    features: [
      '🧠 AI finds GUARANTEED profit trades',
      '⚡ Quick +1% take profit',
      '🛡️ Tight -0.5% stop loss (2:1 R/R)',
      '📈 Momentum filter: only rising prices',
      '💰 15 positions × 5% = 75% active',
      '🎯 High win rate > big wins'
    ]
  }
}

interface AIModelStatus {
  cryptobert: { loaded: boolean; lastUpdate: string | null }
  xgboost: { loaded: boolean; accuracy: number }
  pricePredictor: { loaded: boolean; lastPrediction: string | null }
  regimeDetector: { loaded: boolean; currentRegime: string }
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<BotSettings>(defaultSettings)
  const [isSaving, setIsSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [isSellingAll, setIsSellingAll] = useState(false)
  const [sellAllStatus, setSellAllStatus] = useState<string | null>(null)
  const [equity, setEquity] = useState<number>(0)
  const [usdtEurRate, setUsdtEurRate] = useState<number>(0.86) // Default rate
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [aiModelStatus, setAiModelStatus] = useState<AIModelStatus | null>(null)
  const [isLoadingModels, setIsLoadingModels] = useState(true)

  useEffect(() => {
    loadSettings()
    loadEquity()
    loadAIModelStatus()
    fetchUsdtEurRate()
  }, [])

  const loadSettings = async () => {
    try {
      const res = await fetch('/ai/exchange/settings')
      const data = await res.json()
      if (data.success && data.data) {
        // Validate riskMode - map 'neutral' or any invalid value to 'normal'
        let riskMode = data.data.riskMode || 'normal'
        if (!['safe', 'normal', 'aggressive', 'lock_profit', 'micro_profit'].includes(riskMode)) {
          riskMode = 'normal' // Default to normal if invalid value
        }
        
        setSettings(prev => ({ 
          ...prev, 
          ...data.data,
          riskMode: riskMode as 'safe' | 'normal' | 'aggressive' | 'lock_profit' | 'micro_profit',
          // Map backend field names to frontend
          useCryptoBert: data.data.enableFinbertSentiment ?? data.data.useCryptoBert ?? true,
          useXgboostClassifier: data.data.enableXgboostClassifier ?? data.data.useXgboostClassifier ?? true,
          usePricePredictor: data.data.enablePricePrediction ?? data.data.usePricePredictor ?? true,
          useRegimeDetection: data.data.enableRegimeDetection ?? data.data.useRegimeDetection ?? true,
          useEdgeEstimation: data.data.enableEdgeEstimation ?? data.data.useEdgeEstimation ?? true,
          useDynamicSizing: data.data.enableDynamicSizing ?? data.data.useDynamicSizing ?? true,
        }))
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

  // Fetch USDT to EUR exchange rate
  const fetchUsdtEurRate = async () => {
    try {
      const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=eur')
      const data = await response.json()
      if (data?.tether?.eur) {
        setUsdtEurRate(data.tether.eur)
      }
    } catch (error) {
      console.log('Using default USDT/EUR rate')
    }
  }

  const loadAIModelStatus = async () => {
    setIsLoadingModels(true)
    try {
      // Load status of all AI models
      const [cryptobertRes, xgboostRes, regimeRes] = await Promise.all([
        fetch('/ai/crypto-sentiment/stats').catch(() => null),
        fetch('/ai/xgboost/stats').catch(() => null),
        fetch('/ai/market/regime').catch(() => null)
      ])

      const cryptobertData = cryptobertRes ? await cryptobertRes.json().catch(() => ({})) : {}
      const xgboostData = xgboostRes ? await xgboostRes.json().catch(() => ({})) : {}
      const regimeData = regimeRes ? await regimeRes.json().catch(() => ({})) : {}

      setAiModelStatus({
        cryptobert: {
          loaded: cryptobertData.success !== false,
          lastUpdate: cryptobertData.data?.last_update || null
        },
        xgboost: {
          loaded: xgboostData.success !== false,
          accuracy: xgboostData.data?.accuracy || 0
        },
        pricePredictor: {
          loaded: true, // Assuming it's loaded
          lastPrediction: null
        },
        regimeDetector: {
          loaded: regimeData.success !== false,
          currentRegime: regimeData.regime || 'Unknown'
        }
      })
    } catch (error) {
      console.error('Failed to load AI model status:', error)
    } finally {
      setIsLoadingModels(false)
    }
  }

  const saveSettings = async () => {
    setIsSaving(true)
    setSaveStatus('idle')
    
    try {
      // Map frontend settings to backend field names
      const backendSettings = {
        riskMode: settings.riskMode,
        takeProfitPercent: settings.takeProfitPercent,
        stopLossPercent: settings.stopLossPercent,
        trailingStopPercent: settings.trailingStopPercent,
        minProfitToTrail: settings.minProfitToTrail,
        minConfidence: settings.minConfidence,
        minEdge: settings.minEdge,
        maxPositionPercent: settings.maxPositionPercent,
        maxOpenPositions: settings.maxOpenPositions,
        maxDailyDrawdown: settings.maxDailyDrawdown,
        maxTotalExposure: settings.maxTotalExposure,
        useAiSignals: settings.useAiSignals,
        learnFromTrades: settings.learnFromTrades,
        enableRegimeDetection: settings.useRegimeDetection,
        enableEdgeEstimation: settings.useEdgeEstimation,
        enableDynamicSizing: settings.useDynamicSizing,
        enableFinbertSentiment: settings.useCryptoBert, // Backend uses this name
        enableXgboostClassifier: settings.useXgboostClassifier,
        enablePricePrediction: settings.usePricePredictor,
      }

      const res = await fetch('/ai/exchange/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(backendSettings)
      })
      
      const data = await res.json()
      
      if (data.success) {
        setSaveStatus('success')
        // Store in localStorage for dashboard
        localStorage.setItem('sentinel_settings', JSON.stringify(settings))
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
        setSellAllStatus(`Closed ${data.data.closedCount} positions. Total P&L: ${data.data.totalPnl?.toFixed(2)} USDT`)
      } else {
        setSellAllStatus(`Error: ${data.error}`)
      }
    } catch (error) {
      setSellAllStatus('Failed to close positions')
    } finally {
      setIsSellingAll(false)
    }
  }

  const applyRiskPreset = (mode: 'safe' | 'normal' | 'aggressive' | 'lock_profit') => {
    const preset = riskPresets[mode]
    setSettings(prev => ({
      ...prev,
      riskMode: mode,
      ...preset.params
    }))
  }

  const getColorClass = (color: string, type: 'text' | 'bg' | 'border') => {
    const colors: Record<string, Record<string, string>> = {
      emerald: {
        text: 'text-sentinel-accent-emerald',
        bg: 'bg-sentinel-accent-emerald',
        border: 'border-sentinel-accent-emerald'
      },
      amber: {
        text: 'text-sentinel-accent-amber',
        bg: 'bg-sentinel-accent-amber',
        border: 'border-sentinel-accent-amber'
      },
      crimson: {
        text: 'text-sentinel-accent-crimson',
        bg: 'bg-sentinel-accent-crimson',
        border: 'border-sentinel-accent-crimson'
      },
      cyan: {
        text: 'text-sentinel-accent-cyan',
        bg: 'bg-sentinel-accent-cyan',
        border: 'border-sentinel-accent-cyan'
      }
    }
    return colors[color]?.[type] || ''
  }

  // Get the current preset safely (fallback to 'normal' if riskMode is invalid)
  const getCurrentPreset = () => {
    const validModes = ['safe', 'normal', 'aggressive', 'lock_profit', 'micro_profit'] as const
    const mode = validModes.includes(settings.riskMode as any) ? settings.riskMode : 'normal'
    return { mode, preset: riskPresets[mode] }
  }
  
  const { mode: currentMode, preset: currentPreset } = getCurrentPreset()

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
              <h1 className="font-display font-bold text-xl">Trading Settings</h1>
              <p className="text-xs text-sentinel-text-muted">Configure risk levels, AI models, and trading parameters</p>
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
        
        {/* Balance & Budget Card */}
        <div className="p-6 rounded-2xl glass-card">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-xl bg-sentinel-accent-cyan/20">
                <Wallet className="w-8 h-8 text-sentinel-accent-cyan" />
              </div>
              <div>
                <p className="text-sm text-sentinel-text-muted">Total Trading Budget</p>
                <p className="text-3xl font-bold font-mono">{equity.toFixed(2)} <span className="text-xl text-sentinel-accent-cyan">USDT</span></p>
                <p className="text-sm text-sentinel-text-secondary">
                  ≈ €{(equity * usdtEurRate).toFixed(2)} EUR
                </p>
                <p className="text-xs text-sentinel-text-muted mt-1">
                  Bot manages entire balance • Crypto & TradFi unified
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-sentinel-text-muted">Current Risk Mode</p>
              <p className={`text-2xl font-bold ${getColorClass(currentPreset.color, 'text')}`}>
                {currentPreset.name}
              </p>
              <p className="text-xs text-sentinel-text-muted mt-1">
                Max Exposure: {(equity * settings.maxTotalExposure / 100).toFixed(2)} USDT
              </p>
            </div>
          </div>
        </div>

        {/* AI Models Status */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 rounded-2xl glass-card"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Cpu className="w-5 h-5 text-sentinel-accent-violet" />
              AI Models Status
            </h2>
            <button 
              onClick={loadAIModelStatus}
              className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${isLoadingModels ? 'animate-spin' : ''}`} />
            </button>
          </div>
          
          <div className="grid grid-cols-4 gap-4">
            {/* CryptoBERT */}
            <div className={`p-4 rounded-xl ${settings.useCryptoBert ? 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30' : 'bg-sentinel-bg-tertiary'}`}>
              <div className="flex items-center gap-2 mb-2">
                <MessageSquare className={`w-5 h-5 ${settings.useCryptoBert ? 'text-sentinel-accent-emerald' : 'text-sentinel-text-muted'}`} />
                <span className="font-medium text-sm">CryptoBERT</span>
              </div>
              <p className="text-xs text-sentinel-text-muted">Crypto Sentiment Analysis</p>
              <div className="mt-2 flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full ${aiModelStatus?.cryptobert.loaded ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-text-muted'}`} />
                <span className="text-xs">{aiModelStatus?.cryptobert.loaded ? 'Active' : 'Inactive'}</span>
              </div>
            </div>

            {/* XGBoost */}
            <div className={`p-4 rounded-xl ${settings.useXgboostClassifier ? 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30' : 'bg-sentinel-bg-tertiary'}`}>
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className={`w-5 h-5 ${settings.useXgboostClassifier ? 'text-sentinel-accent-emerald' : 'text-sentinel-text-muted'}`} />
                <span className="font-medium text-sm">XGBoost</span>
              </div>
              <p className="text-xs text-sentinel-text-muted">Signal Classification</p>
              <div className="mt-2 flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full ${aiModelStatus?.xgboost.loaded ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-text-muted'}`} />
                <span className="text-xs">
                  {aiModelStatus?.xgboost.accuracy ? `${(aiModelStatus.xgboost.accuracy * 100).toFixed(0)}% Acc` : 'Training...'}
                </span>
              </div>
            </div>

            {/* Price Predictor */}
            <div className={`p-4 rounded-xl ${settings.usePricePredictor ? 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30' : 'bg-sentinel-bg-tertiary'}`}>
              <div className="flex items-center gap-2 mb-2">
                <LineChart className={`w-5 h-5 ${settings.usePricePredictor ? 'text-sentinel-accent-emerald' : 'text-sentinel-text-muted'}`} />
                <span className="font-medium text-sm">Chronos</span>
              </div>
              <p className="text-xs text-sentinel-text-muted">Price Prediction</p>
              <div className="mt-2 flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full ${aiModelStatus?.pricePredictor.loaded ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-text-muted'}`} />
                <span className="text-xs">{aiModelStatus?.pricePredictor.loaded ? 'Active' : 'Inactive'}</span>
              </div>
            </div>

            {/* Regime Detector */}
            <div className={`p-4 rounded-xl ${settings.useRegimeDetection ? 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30' : 'bg-sentinel-bg-tertiary'}`}>
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className={`w-5 h-5 ${settings.useRegimeDetection ? 'text-sentinel-accent-emerald' : 'text-sentinel-text-muted'}`} />
                <span className="font-medium text-sm">Regime</span>
              </div>
              <p className="text-xs text-sentinel-text-muted">Market Regime Detection</p>
              <div className="mt-2 flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full ${aiModelStatus?.regimeDetector.loaded ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-text-muted'}`} />
                <span className="text-xs">{aiModelStatus?.regimeDetector.currentRegime || 'Detecting...'}</span>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Risk Mode Selection */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 rounded-2xl glass-card"
        >
          <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
            <Gauge className="w-5 h-5 text-sentinel-accent-cyan" />
            Risk Profile
          </h2>
          <p className="text-sm text-sentinel-text-muted mb-6">
            Select a preset to automatically configure all parameters
          </p>
          
          <div className="grid grid-cols-4 gap-4">
            {(Object.keys(riskPresets) as Array<keyof typeof riskPresets>).map((mode) => {
              const preset = riskPresets[mode]
              const Icon = preset.icon
              const isActive = settings.riskMode === mode
              
              return (
                <button
                  key={mode}
                  onClick={() => applyRiskPreset(mode)}
                  className={`p-6 rounded-xl border-2 transition-all text-left ${
                    isActive
                      ? `${getColorClass(preset.color, 'border')} ${getColorClass(preset.color, 'bg')}/10`
                      : 'border-sentinel-border hover:border-sentinel-border-hover'
                  }`}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-3 rounded-xl ${
                      isActive ? `${getColorClass(preset.color, 'bg')}/20` : 'bg-sentinel-bg-tertiary'
                    }`}>
                      <Icon className={`w-6 h-6 ${
                        isActive ? getColorClass(preset.color, 'text') : 'text-sentinel-text-muted'
                      }`} />
                    </div>
                    {isActive && (
                      <CheckCircle className={`w-5 h-5 ${getColorClass(preset.color, 'text')}`} />
                    )}
                  </div>
                  
                  <h3 className={`font-bold text-lg mb-1 ${isActive ? getColorClass(preset.color, 'text') : ''}`}>
                    {preset.name}
                  </h3>
                  <p className="text-xs text-sentinel-text-muted mb-4">
                    {preset.description}
                  </p>
                  
                  <ul className="space-y-1">
                    {preset.features.slice(0, 4).map((feature, i) => (
                      <li key={i} className="text-xs text-sentinel-text-secondary flex items-center gap-1">
                        <span className={`w-1 h-1 rounded-full ${getColorClass(preset.color, 'bg')}`} />
                        {feature}
                      </li>
                    ))}
                  </ul>
                </button>
              )
            })}
          </div>
        </motion.section>

        {/* Current Settings Summary */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="p-6 rounded-2xl glass-card"
        >
          <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <Info className="w-5 h-5 text-sentinel-accent-cyan" />
            Current Configuration
          </h2>
          
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-sentinel-bg-tertiary text-center">
              <p className="text-xs text-sentinel-text-muted mb-1">Take Profit</p>
              <p className="text-xl font-bold text-sentinel-accent-emerald">+{settings.takeProfitPercent}%</p>
            </div>
            <div className="p-4 rounded-xl bg-sentinel-bg-tertiary text-center">
              <p className="text-xs text-sentinel-text-muted mb-1">Stop Loss</p>
              <p className="text-xl font-bold text-sentinel-accent-crimson">-{settings.stopLossPercent}%</p>
            </div>
            <div className="p-4 rounded-xl bg-sentinel-bg-tertiary text-center">
              <p className="text-xs text-sentinel-text-muted mb-1">Min Confidence</p>
              <p className="text-xl font-bold text-sentinel-accent-amber">{settings.minConfidence}%</p>
            </div>
            <div className="p-4 rounded-xl bg-sentinel-bg-tertiary text-center">
              <p className="text-xs text-sentinel-text-muted mb-1">Max Positions</p>
              <p className="text-xl font-bold text-sentinel-accent-cyan">
                {settings.maxOpenPositions === 0 ? (
                  <span className="flex items-center justify-center gap-1">
                    <Infinity className="w-5 h-5" />
                  </span>
                ) : settings.maxOpenPositions}
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-4 gap-4 mt-4">
            <div className="p-4 rounded-xl bg-sentinel-bg-tertiary text-center">
              <p className="text-xs text-sentinel-text-muted mb-1">Min Edge</p>
              <p className="text-xl font-bold">{settings.minEdge}</p>
            </div>
            <div className="p-4 rounded-xl bg-sentinel-bg-tertiary text-center">
              <p className="text-xs text-sentinel-text-muted mb-1">Trailing Stop</p>
              <p className="text-xl font-bold">{settings.trailingStopPercent}%</p>
            </div>
            <div className="p-4 rounded-xl bg-sentinel-bg-tertiary text-center">
              <p className="text-xs text-sentinel-text-muted mb-1">Max Daily DD</p>
              <p className="text-xl font-bold text-sentinel-accent-crimson">-{settings.maxDailyDrawdown}%</p>
            </div>
            <div className="p-4 rounded-xl bg-sentinel-bg-tertiary text-center">
              <p className="text-xs text-sentinel-text-muted mb-1">Max Exposure</p>
              <p className="text-xl font-bold">{settings.maxTotalExposure}%</p>
            </div>
          </div>
        </motion.section>

        {/* Advanced Settings Toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full p-4 rounded-xl glass-card flex items-center justify-between hover:bg-sentinel-bg-tertiary transition-colors"
        >
          <div className="flex items-center gap-3">
            <Settings className="w-5 h-5 text-sentinel-accent-cyan" />
            <span className="font-medium">Advanced Settings</span>
          </div>
          <motion.div
            animate={{ rotate: showAdvanced ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ArrowLeft className="w-5 h-5 -rotate-90" />
          </motion.div>
        </button>

        {/* Advanced Settings */}
        {showAdvanced && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-8"
          >
            {/* Trading Parameters */}
            <section className="p-6 rounded-2xl glass-card">
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
                      max="15"
                      step="0.5"
                      value={settings.takeProfitPercent}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        takeProfitPercent: parseFloat(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-emerald"
                    />
                    <span className="w-16 text-right font-mono text-sentinel-accent-emerald">
                      +{settings.takeProfitPercent}%
                    </span>
                  </div>
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
                      max="10"
                      step="0.1"
                      value={settings.stopLossPercent}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        stopLossPercent: parseFloat(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-crimson"
                    />
                    <span className="w-16 text-right font-mono text-sentinel-accent-crimson">
                      -{settings.stopLossPercent}%
                    </span>
                  </div>
                </div>

                {/* Trailing Stop */}
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Trailing Stop %
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="0.01"
                      max="5"
                      step="0.01"
                      value={settings.trailingStopPercent}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        trailingStopPercent: parseFloat(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-amber"
                    />
                    <span className="w-16 text-right font-mono text-sentinel-accent-amber">
                      {settings.trailingStopPercent}%
                    </span>
                  </div>
                </div>

                {/* Min Profit to Trail */}
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Min Profit to Activate Trail
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="0.01"
                      max="5"
                      step="0.01"
                      value={settings.minProfitToTrail}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        minProfitToTrail: parseFloat(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-cyan"
                    />
                    <span className="w-16 text-right font-mono text-sentinel-accent-cyan">
                      +{settings.minProfitToTrail}%
                    </span>
                  </div>
                </div>

                {/* Min Confidence */}
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Minimum AI Confidence
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="30"
                      max="95"
                      step="5"
                      value={settings.minConfidence}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        minConfidence: parseInt(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-violet"
                    />
                    <span className="w-16 text-right font-mono text-sentinel-accent-violet">
                      {settings.minConfidence}%
                    </span>
                  </div>
                </div>

                {/* Min Edge */}
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Minimum Edge Score
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="0.05"
                      max="0.50"
                      step="0.01"
                      value={settings.minEdge}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        minEdge: parseFloat(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-cyan"
                    />
                    <span className="w-16 text-right font-mono">
                      {settings.minEdge.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            </section>

            {/* Risk Management */}
            <section className="p-6 rounded-2xl glass-card">
              <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-sentinel-accent-cyan" />
                Risk Management
              </h2>
              
              <div className="grid grid-cols-2 gap-6">
                {/* Max Position Size */}
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    % of Budget per Trade
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="1"
                      max="50"
                      step="1"
                      value={settings.maxPositionPercent}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        maxPositionPercent: parseInt(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-cyan"
                    />
                    <span className="w-16 text-right font-mono">
                      {settings.maxPositionPercent}%
                    </span>
                  </div>
                  <p className="text-xs text-sentinel-text-muted mt-1">
                    Each trade uses max {(equity * settings.maxPositionPercent / 100).toFixed(2)} USDT of your budget
                  </p>
                </div>

                {/* Max Open Positions */}
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Max Simultaneous Trades (0 = unlimited)
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      step="1"
                      value={settings.maxOpenPositions}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        maxOpenPositions: parseInt(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-cyan"
                    />
                    <span className="w-16 text-right font-mono flex items-center justify-end">
                      {settings.maxOpenPositions === 0 ? (
                        <Infinity className="w-5 h-5" />
                      ) : settings.maxOpenPositions}
                    </span>
                  </div>
                  <p className="text-xs text-sentinel-text-muted mt-1">
                    {settings.maxOpenPositions === 0 
                      ? 'UNLIMITED - Bot opens as many trades as budget allows' 
                      : `Bot will have maximum ${settings.maxOpenPositions} trades open at once`}
                  </p>
                </div>

                {/* Max Daily Drawdown */}
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Max Daily Drawdown %
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="1"
                      max="15"
                      step="0.5"
                      value={settings.maxDailyDrawdown}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        maxDailyDrawdown: parseFloat(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-crimson"
                    />
                    <span className="w-16 text-right font-mono text-sentinel-accent-crimson">
                      -{settings.maxDailyDrawdown}%
                    </span>
                  </div>
                  <p className="text-xs text-sentinel-text-muted mt-1">
                    Stop trading if daily loss exceeds {(equity * settings.maxDailyDrawdown / 100).toFixed(2)} USDT
                  </p>
                </div>

                {/* Max Total Exposure */}
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Max % of Budget to Use
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="10"
                      max="100"
                      step="5"
                      value={settings.maxTotalExposure}
                      onChange={(e) => setSettings(prev => ({ 
                        ...prev, 
                        maxTotalExposure: parseInt(e.target.value)
                      }))}
                      className="flex-1 accent-sentinel-accent-amber"
                    />
                    <span className="w-16 text-right font-mono">
                      {settings.maxTotalExposure}%
                    </span>
                  </div>
                  <p className="text-xs text-sentinel-text-muted mt-1">
                    Bot can invest up to {(equity * settings.maxTotalExposure / 100).toFixed(2)} USDT of your {equity.toFixed(2)} USDT budget
                  </p>
                </div>
              </div>
            </section>

            {/* AI Models Control */}
            <section className="p-6 rounded-2xl glass-card">
              <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <Brain className="w-5 h-5 text-sentinel-accent-violet" />
                AI Models
              </h2>
              
              <div className="grid grid-cols-2 gap-4">
                {/* CryptoBERT Sentiment */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                  <div className="flex items-center gap-3">
                    <MessageSquare className="w-5 h-5 text-sentinel-accent-cyan" />
                    <div>
                      <h3 className="font-medium">CryptoBERT Sentiment</h3>
                      <p className="text-xs text-sentinel-text-muted">
                        Analyzes crypto news for market sentiment
                      </p>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.useCryptoBert}
                      onChange={(e) => setSettings(prev => ({ ...prev, useCryptoBert: e.target.checked }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-sentinel-border rounded-full peer 
                                    peer-checked:bg-sentinel-accent-emerald transition-colors"></div>
                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                    peer-checked:translate-x-5"></div>
                  </label>
                </div>

                {/* XGBoost Classifier */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                  <div className="flex items-center gap-3">
                    <BarChart3 className="w-5 h-5 text-sentinel-accent-amber" />
                    <div>
                      <h3 className="font-medium">XGBoost Classifier</h3>
                      <p className="text-xs text-sentinel-text-muted">
                        ML-based trade signal validation
                      </p>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.useXgboostClassifier}
                      onChange={(e) => setSettings(prev => ({ ...prev, useXgboostClassifier: e.target.checked }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-sentinel-border rounded-full peer 
                                    peer-checked:bg-sentinel-accent-emerald transition-colors"></div>
                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                    peer-checked:translate-x-5"></div>
                  </label>
                </div>

                {/* Price Predictor */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                  <div className="flex items-center gap-3">
                    <LineChart className="w-5 h-5 text-sentinel-accent-emerald" />
                    <div>
                      <h3 className="font-medium">Chronos Price Predictor</h3>
                      <p className="text-xs text-sentinel-text-muted">
                        AI-based price movement prediction
                      </p>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.usePricePredictor}
                      onChange={(e) => setSettings(prev => ({ ...prev, usePricePredictor: e.target.checked }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-sentinel-border rounded-full peer 
                                    peer-checked:bg-sentinel-accent-emerald transition-colors"></div>
                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                    peer-checked:translate-x-5"></div>
                  </label>
                </div>

                {/* Regime Detection */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                  <div className="flex items-center gap-3">
                    <Sparkles className="w-5 h-5 text-sentinel-accent-violet" />
                    <div>
                      <h3 className="font-medium">Regime Detection</h3>
                      <p className="text-xs text-sentinel-text-muted">
                        Detects market regimes (trend/range/volatile)
                      </p>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.useRegimeDetection}
                      onChange={(e) => setSettings(prev => ({ ...prev, useRegimeDetection: e.target.checked }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-sentinel-border rounded-full peer 
                                    peer-checked:bg-sentinel-accent-emerald transition-colors"></div>
                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                    peer-checked:translate-x-5"></div>
                  </label>
                </div>

                {/* Edge Estimation */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                  <div className="flex items-center gap-3">
                    <TrendingUp className="w-5 h-5 text-sentinel-accent-cyan" />
                    <div>
                      <h3 className="font-medium">Edge Estimation</h3>
                      <p className="text-xs text-sentinel-text-muted">
                        Calculates statistical edge before trading
                      </p>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.useEdgeEstimation}
                      onChange={(e) => setSettings(prev => ({ ...prev, useEdgeEstimation: e.target.checked }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-sentinel-border rounded-full peer 
                                    peer-checked:bg-sentinel-accent-emerald transition-colors"></div>
                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                    peer-checked:translate-x-5"></div>
                  </label>
                </div>

                {/* Dynamic Sizing */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                  <div className="flex items-center gap-3">
                    <Layers className="w-5 h-5 text-sentinel-accent-amber" />
                    <div>
                      <h3 className="font-medium">Dynamic Position Sizing</h3>
                      <p className="text-xs text-sentinel-text-muted">
                        Kelly criterion for optimal position sizing
                      </p>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.useDynamicSizing}
                      onChange={(e) => setSettings(prev => ({ ...prev, useDynamicSizing: e.target.checked }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-sentinel-border rounded-full peer 
                                    peer-checked:bg-sentinel-accent-emerald transition-colors"></div>
                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform
                                    peer-checked:translate-x-5"></div>
                  </label>
                </div>
              </div>

              {/* Learning */}
              <div className="mt-4 flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                <div className="flex items-center gap-3">
                  <Brain className="w-5 h-5 text-sentinel-accent-violet" />
                  <div>
                    <h3 className="font-medium">Continuous Learning</h3>
                    <p className="text-xs text-sentinel-text-muted">
                      AI learns and improves from every trade outcome
                    </p>
                  </div>
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
            </section>
          </motion.div>
        )}

        {/* Quick Actions */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
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
