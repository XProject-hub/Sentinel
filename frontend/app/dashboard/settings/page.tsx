'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { 
  ArrowLeft,
  Save,
  Loader2,
  Shield,
  Target,
  Percent,
  AlertTriangle,
  Zap,
  Settings,
  ChevronDown,
  CheckCircle,
  RefreshCw,
  Brain,
  TrendingUp,
  Activity,
  BarChart3,
  Cpu,
  Gauge,
  Lock,
  Sliders,
  Layers,
  Eye,
  Waves,
  LineChart,
  CircuitBoard,
  Sparkles
} from 'lucide-react'
import Logo from '@/components/Logo'

interface BotSettings {
  riskMode: 'NORMAL' | 'LOCK_PROFIT' | 'MICRO_PROFIT' | 'SCALPER' | 'SWING'
  takeProfitPercent: number
  stopLossPercent: number
  trailingStopPercent: number
  minProfitToTrail: number
  maxOpenPositions: number
  maxPositionPercent: number
  maxTotalExposure: number
  kellyMultiplier: number
  minEdge: number
  minConfidence: number
  leverageMode: string
  useDynamicSizing: boolean
  useWhaleDetection: boolean
  useFundingRate: boolean
  useRegimeFilter: boolean
  usePatternRecognition: boolean
  useSentimentAnalysis: boolean
  useXGBoost: boolean
  useEdgeEstimation: boolean
  useQLearning: boolean
  momentumThreshold: number
}

const defaultSettings: BotSettings = {
  riskMode: 'NORMAL',
  takeProfitPercent: 0,
  stopLossPercent: 1.5,
  trailingStopPercent: 0.13,
  minProfitToTrail: 0.35,
  maxOpenPositions: 5,
  maxPositionPercent: 10,
  maxTotalExposure: 80,
  kellyMultiplier: 0.3,
  minEdge: 0.10,
  minConfidence: 60,
  leverageMode: 'AUTO',
  useDynamicSizing: true,
  useWhaleDetection: true,
  useFundingRate: true,
  useRegimeFilter: true,
  usePatternRecognition: true,
  useSentimentAnalysis: true,
  useXGBoost: true,
  useEdgeEstimation: true,
  useQLearning: true,
  momentumThreshold: 0
}

const riskPresets = {
  NORMAL: {
    takeProfitPercent: 0,
    stopLossPercent: 1.5,
    trailingStopPercent: 0.13,
    minProfitToTrail: 0.35,
    description: 'Balanced risk/reward with trailing stops',
    icon: Activity,
    color: 'cyan'
  },
  LOCK_PROFIT: {
    takeProfitPercent: 2.0,
    stopLossPercent: 1.0,
    trailingStopPercent: 0.25,
    minProfitToTrail: 0.5,
    description: 'Lock profits quickly with tight stops',
    icon: Lock,
    color: 'emerald'
  },
  MICRO_PROFIT: {
    takeProfitPercent: 0.5,
    stopLossPercent: 0.3,
    trailingStopPercent: 0.10,
    minProfitToTrail: 0.2,
    description: 'Many small profits, minimal risk',
    icon: Sparkles,
    color: 'violet'
  },
  SCALPER: {
    takeProfitPercent: 0.3,
    stopLossPercent: 0.2,
    trailingStopPercent: 0.08,
    minProfitToTrail: 0.15,
    description: 'Ultra-fast high frequency trades',
    icon: Zap,
    color: 'amber'
  },
  SWING: {
    takeProfitPercent: 5.0,
    stopLossPercent: 2.5,
    trailingStopPercent: 0.5,
    minProfitToTrail: 1.0,
    description: 'Longer holds for bigger moves',
    icon: TrendingUp,
    color: 'blue'
  }
}

const aiModels = [
  { key: 'useWhaleDetection', name: 'Whale Detection', desc: 'Detect large buy/sell walls from order books', icon: Waves },
  { key: 'useFundingRate', name: 'Funding Rate', desc: 'Consider funding rates in trade decisions', icon: Percent },
  { key: 'useRegimeFilter', name: 'Market Regime', desc: 'Adjust strategy based on market conditions', icon: BarChart3 },
  { key: 'useDynamicSizing', name: 'Dynamic Sizing', desc: 'Kelly Criterion position sizing', icon: Sliders },
  { key: 'usePatternRecognition', name: 'Pattern Recognition', desc: 'ML-based price pattern detection', icon: Eye },
  { key: 'useSentimentAnalysis', name: 'Sentiment Analysis', desc: 'CryptoBERT news & social sentiment', icon: Brain },
  { key: 'useXGBoost', name: 'XGBoost ML', desc: 'ML model for price prediction', icon: Cpu },
  { key: 'useEdgeEstimation', name: 'Edge Estimation', desc: 'Statistical edge per symbol', icon: LineChart },
  { key: 'useQLearning', name: 'Q-Learning', desc: 'Reinforcement learning strategy', icon: CircuitBoard },
]

export default function SettingsPage() {
  const [settings, setSettings] = useState<BotSettings>(defaultSettings)
  const [isSaving, setIsSaving] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [hasChanges, setHasChanges] = useState(false)
  const [activeSection, setActiveSection] = useState('strategy')

  useEffect(() => {
    loadSettings()
  }, [])

  const loadSettings = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/ai/exchange/settings?user_id=default')
      if (response.ok) {
        const data = await response.json()
        if (data.data) {
          const backendData = data.data
          // Normalize riskMode to uppercase
          const riskMode = (backendData.riskMode || 'NORMAL').toUpperCase() as keyof typeof riskPresets
          const validRiskMode = riskPresets[riskMode] ? riskMode : 'NORMAL'
          
          const mapped = {
            ...backendData,
            riskMode: validRiskMode,
            useRegimeFilter: backendData.useRegimeDetection ?? backendData.useRegimeFilter ?? true,
            useEdgeEstimation: backendData.useEdgeEstimation ?? true,
            useSentimentAnalysis: backendData.useCryptoBert ?? backendData.useSentimentAnalysis ?? true,
            useXGBoost: backendData.useXgboostClassifier ?? backendData.useXGBoost ?? true,
            usePatternRecognition: backendData.usePatternRecognition ?? backendData.usePricePredictor ?? true,
            useWhaleDetection: backendData.useWhaleDetection ?? true,
            useFundingRate: backendData.useFundingRate ?? true,
            useQLearning: backendData.useQLearning ?? true,
            useDynamicSizing: backendData.useDynamicSizing ?? true,
          }
          setSettings(prev => ({ ...prev, ...mapped }))
        }
      }
    } catch (error) {
      console.error('Failed to load settings:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const saveSettings = async () => {
    setIsSaving(true)
    setSaveStatus('idle')
    try {
      const settingsToSave = {
        ...settings,
        useRegimeDetection: settings.useRegimeFilter,
        useCryptoBert: settings.useSentimentAnalysis,
        useXgboostClassifier: settings.useXGBoost,
        usePricePredictor: settings.usePatternRecognition,
      }
      
      const response = await fetch('/ai/exchange/settings?user_id=default', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settingsToSave)
      })

      if (response.ok) {
        setSaveStatus('success')
        setHasChanges(false)
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

  const updateSetting = <K extends keyof BotSettings>(key: K, value: BotSettings[K]) => {
    setSettings(prev => ({ ...prev, [key]: value }))
    setHasChanges(true)
  }

  const applyPreset = (mode: keyof typeof riskPresets) => {
    const preset = riskPresets[mode]
    setSettings(prev => ({
      ...prev,
      riskMode: mode,
      takeProfitPercent: preset.takeProfitPercent,
      stopLossPercent: preset.stopLossPercent,
      trailingStopPercent: preset.trailingStopPercent,
      minProfitToTrail: preset.minProfitToTrail
    }))
    setHasChanges(true)
  }

  const toggleAiModel = (key: string) => {
    updateSetting(key as keyof BotSettings, !settings[key as keyof BotSettings])
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#060a13] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-10 h-10 text-cyan-500 animate-spin" />
          <p className="text-gray-400">Loading settings...</p>
        </div>
      </div>
    )
  }

  const navItems = [
    { id: 'strategy', label: 'Strategy', icon: Shield },
    { id: 'exit', label: 'Exit Rules', icon: Target },
    { id: 'position', label: 'Position Size', icon: Layers },
    { id: 'filters', label: 'AI Filters', icon: Gauge },
    { id: 'models', label: 'AI Models', icon: Cpu },
  ]

  return (
    <div className="min-h-screen bg-[#060a13]">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.03)_1px,transparent_1px)] bg-[size:44px_44px]" />
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-cyan-500/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-blue-500/5 rounded-full blur-[120px]" />
      </div>

      {/* Header */}
      <header className="sticky top-0 z-50 bg-[#060a13]/90 backdrop-blur-xl border-b border-white/5">
        <div className="w-full px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link 
                href="/dashboard" 
                className="flex items-center gap-3 p-2 -ml-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-400" />
                <Logo size="sm" />
              </Link>
              <div className="h-6 w-px bg-white/10" />
              <div>
                <h1 className="text-lg font-semibold text-white">Trading Configuration</h1>
                <p className="text-xs text-gray-500">Configure AI trading parameters</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={loadSettings}
                className="p-2.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-all"
                title="Refresh"
              >
                <RefreshCw className="w-4 h-4 text-gray-400" />
              </button>
              
              <button
                onClick={saveSettings}
                disabled={isSaving || !hasChanges}
                className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all ${
                  hasChanges
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30'
                    : 'bg-white/5 text-gray-500 border border-white/10 cursor-not-allowed'
                }`}
              >
                {isSaving ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : saveStatus === 'success' ? (
                  <CheckCircle className="w-4 h-4" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                {isSaving ? 'Saving...' : saveStatus === 'success' ? 'Saved!' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex relative">
        {/* Sidebar Navigation */}
        <aside className="w-64 min-h-[calc(100vh-64px)] bg-[#0a0f1a]/50 border-r border-white/5 p-4 sticky top-16">
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveSection(item.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                    activeSection === item.id
                      ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {item.label}
                </button>
              )
            })}
          </nav>

          {/* Quick Stats */}
          <div className="mt-8 p-4 rounded-xl bg-white/[0.02] border border-white/5">
            <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-3">Current Mode</h3>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-sm font-medium text-white">{(settings.riskMode || 'NORMAL').replace('_', ' ')}</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {riskPresets[settings.riskMode as keyof typeof riskPresets]?.description || 'Balanced risk/reward strategy'}
            </p>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6 lg:p-8">
          <div className="max-w-5xl space-y-6">
            
            {/* Strategy Section */}
            {activeSection === 'strategy' && (
              <section className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                    <Shield className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-white">Trading Strategy</h2>
                    <p className="text-sm text-gray-500">Select a preset or customize your approach</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {(Object.keys(riskPresets) as Array<keyof typeof riskPresets>).map((mode) => {
                    const preset = riskPresets[mode]
                    const Icon = preset.icon
                    const isActive = settings.riskMode === mode
                    
                    return (
                      <button
                        key={mode}
                        onClick={() => applyPreset(mode)}
                        className={`relative p-5 rounded-2xl border text-left transition-all group ${
                          isActive
                            ? 'bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border-cyan-500/30 shadow-lg shadow-cyan-500/10'
                            : 'bg-white/[0.02] border-white/10 hover:border-white/20 hover:bg-white/[0.04]'
                        }`}
                      >
                        {isActive && (
                          <div className="absolute top-3 right-3">
                            <CheckCircle className="w-5 h-5 text-cyan-400" />
                          </div>
                        )}
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4 ${
                          isActive ? 'bg-cyan-500/20' : 'bg-white/5'
                        }`}>
                          <Icon className={`w-5 h-5 ${isActive ? 'text-cyan-400' : 'text-gray-400'}`} />
                        </div>
                        <h3 className={`font-semibold mb-1 ${isActive ? 'text-cyan-400' : 'text-white'}`}>
                          {mode.replace('_', ' ')}
                        </h3>
                        <p className="text-xs text-gray-500 leading-relaxed">{preset.description}</p>
                        
                        <div className="mt-4 pt-4 border-t border-white/5 grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-gray-500">TP:</span>
                            <span className="ml-1 text-emerald-400">{preset.takeProfitPercent || 'Trail'}%</span>
                          </div>
                          <div>
                            <span className="text-gray-500">SL:</span>
                            <span className="ml-1 text-red-400">{preset.stopLossPercent}%</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Trail:</span>
                            <span className="ml-1 text-amber-400">{preset.trailingStopPercent}%</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Min:</span>
                            <span className="ml-1 text-cyan-400">{preset.minProfitToTrail}%</span>
                          </div>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </section>
            )}

            {/* Exit Rules Section */}
            {activeSection === 'exit' && (
              <section className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                    <Target className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-white">Exit Rules</h2>
                    <p className="text-sm text-gray-500">Configure take profit and stop loss parameters</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Take Profit */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                          <TrendingUp className="w-4 h-4 text-emerald-400" />
                        </div>
                        <span className="font-medium text-white">Take Profit</span>
                      </div>
                      <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-sm font-mono">
                        {settings.takeProfitPercent === 0 ? 'OFF' : `${settings.takeProfitPercent}%`}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="10"
                      step="0.5"
                      value={settings.takeProfitPercent}
                      onChange={(e) => updateSetting('takeProfitPercent', parseFloat(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-emerald-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">Set to 0 to use trailing stop only</p>
                  </div>

                  {/* Stop Loss */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-red-500/10 flex items-center justify-center">
                          <AlertTriangle className="w-4 h-4 text-red-400" />
                        </div>
                        <span className="font-medium text-white">Stop Loss</span>
                      </div>
                      <span className="px-3 py-1 rounded-full bg-red-500/10 text-red-400 text-sm font-mono">
                        {settings.stopLossPercent}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="5"
                      step="0.1"
                      value={settings.stopLossPercent}
                      onChange={(e) => updateSetting('stopLossPercent', parseFloat(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-red-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">Maximum loss before forced exit</p>
                  </div>

                  {/* Trailing Stop */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center">
                          <Activity className="w-4 h-4 text-amber-400" />
                        </div>
                        <span className="font-medium text-white">Trailing Stop</span>
                      </div>
                      <span className="px-3 py-1 rounded-full bg-amber-500/10 text-amber-400 text-sm font-mono">
                        {settings.trailingStopPercent}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.05"
                      max="2"
                      step="0.01"
                      value={settings.trailingStopPercent}
                      onChange={(e) => updateSetting('trailingStopPercent', parseFloat(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-amber-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">Sell when price drops this % from peak</p>
                  </div>

                  {/* Min Profit to Trail */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-cyan-500/10 flex items-center justify-center">
                          <Sparkles className="w-4 h-4 text-cyan-400" />
                        </div>
                        <span className="font-medium text-white">Min Profit to Trail</span>
                      </div>
                      <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-mono">
                        {settings.minProfitToTrail}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="3"
                      step="0.05"
                      value={settings.minProfitToTrail}
                      onChange={(e) => updateSetting('minProfitToTrail', parseFloat(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">Trailing activates after this profit</p>
                  </div>
                </div>
              </section>
            )}

            {/* Position Sizing Section */}
            {activeSection === 'position' && (
              <section className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                    <Layers className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-white">Position Sizing</h2>
                    <p className="text-sm text-gray-500">Control trade sizes and risk limits</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Max Open Positions */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium text-white">Max Open Positions</span>
                      <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-mono">
                        {settings.maxOpenPositions}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      step="1"
                      value={settings.maxOpenPositions}
                      onChange={(e) => updateSetting('maxOpenPositions', parseInt(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-cyan-500"
                    />
                    <div className="flex justify-between text-xs text-gray-600 mt-2">
                      <span>1</span>
                      <span>10</span>
                      <span>20</span>
                    </div>
                  </div>

                  {/* Max Position Percent */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium text-white">Max Position Size</span>
                      <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-mono">
                        {settings.maxPositionPercent}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      step="1"
                      value={settings.maxPositionPercent}
                      onChange={(e) => updateSetting('maxPositionPercent', parseInt(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">Max % of equity per trade</p>
                  </div>

                  {/* Max Total Exposure */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium text-white">Max Total Exposure</span>
                      <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-mono">
                        {settings.maxTotalExposure}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="20"
                      max="100"
                      step="5"
                      value={settings.maxTotalExposure}
                      onChange={(e) => updateSetting('maxTotalExposure', parseInt(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">Total portfolio allocation limit</p>
                  </div>

                  {/* Kelly Multiplier */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium text-white">Kelly Multiplier</span>
                      <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-mono">
                        {settings.kellyMultiplier}x
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={settings.kellyMultiplier}
                      onChange={(e) => updateSetting('kellyMultiplier', parseFloat(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">Higher = larger positions, more risk</p>
                  </div>
                </div>
              </section>
            )}

            {/* AI Filters Section */}
            {activeSection === 'filters' && (
              <section className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                    <Gauge className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-white">AI Filters</h2>
                    <p className="text-sm text-gray-500">Control trade quality requirements</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Min Edge */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium text-white">Min Edge Score</span>
                      <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-mono">
                        {settings.minEdge}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={settings.minEdge}
                      onChange={(e) => updateSetting('minEdge', parseFloat(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">Higher = fewer but better trades</p>
                  </div>

                  {/* Min Confidence */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium text-white">Min Confidence</span>
                      <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-mono">
                        {settings.minConfidence}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="40"
                      max="90"
                      step="5"
                      value={settings.minConfidence}
                      onChange={(e) => updateSetting('minConfidence', parseInt(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">AI confidence threshold for trades</p>
                  </div>

                  {/* Leverage Mode */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium text-white">Leverage Mode</span>
                    </div>
                    <div className="relative">
                      <select
                        value={settings.leverageMode}
                        onChange={(e) => updateSetting('leverageMode', e.target.value)}
                        className="w-full px-4 py-3 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none appearance-none cursor-pointer"
                      >
                        <option value="AUTO">AUTO (AI decides)</option>
                        <option value="1">1x (No leverage)</option>
                        <option value="2">2x</option>
                        <option value="3">3x</option>
                        <option value="5">5x</option>
                        <option value="10">10x</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 pointer-events-none" />
                    </div>
                  </div>

                  {/* Momentum Threshold */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium text-white">Momentum Threshold</span>
                      <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-sm font-mono">
                        {settings.momentumThreshold === 0 ? 'OFF' : `${settings.momentumThreshold}%`}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="0.1"
                      step="0.01"
                      value={settings.momentumThreshold}
                      onChange={(e) => updateSetting('momentumThreshold', parseFloat(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-cyan-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">Min 24h momentum to trade</p>
                  </div>
                </div>
              </section>
            )}

            {/* AI Models Section */}
            {activeSection === 'models' && (
              <section className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                    <Cpu className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-white">AI Models</h2>
                    <p className="text-sm text-gray-500">Enable or disable AI analysis features</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {aiModels.map((model) => {
                    const Icon = model.icon
                    const isEnabled = settings[model.key as keyof BotSettings] as boolean
                    
                    return (
                      <div
                        key={model.key}
                        className={`p-4 rounded-xl border transition-all ${
                          isEnabled 
                            ? 'bg-cyan-500/5 border-cyan-500/20' 
                            : 'bg-white/[0.02] border-white/5'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-3">
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                              isEnabled ? 'bg-cyan-500/10' : 'bg-white/5'
                            }`}>
                              <Icon className={`w-5 h-5 ${isEnabled ? 'text-cyan-400' : 'text-gray-500'}`} />
                            </div>
                            <div>
                              <h3 className={`font-medium text-sm ${isEnabled ? 'text-white' : 'text-gray-400'}`}>
                                {model.name}
                              </h3>
                              <p className="text-xs text-gray-500 mt-0.5">{model.desc}</p>
                            </div>
                          </div>
                          <button
                            onClick={() => toggleAiModel(model.key)}
                            className={`relative w-11 h-6 rounded-full transition-colors ${
                              isEnabled ? 'bg-cyan-500' : 'bg-white/10'
                            }`}
                          >
                            <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow-lg transition-transform ${
                              isEnabled ? 'left-[22px]' : 'left-0.5'
                            }`} />
                          </button>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Models Summary */}
                <div className="p-4 rounded-xl bg-gradient-to-r from-cyan-500/5 to-blue-500/5 border border-cyan-500/10">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm font-medium text-white">
                      {aiModels.filter(m => settings[m.key as keyof BotSettings]).length} of {aiModels.length} models active
                    </span>
                  </div>
                  <p className="text-xs text-gray-500">
                    More active models provide deeper analysis but may reduce trade frequency
                  </p>
                </div>
              </section>
            )}

            {/* Risk Warning */}
            <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/20 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-medium text-amber-400">Risk Warning</span>
                <p className="text-sm text-gray-400 mt-1">
                  Trading involves significant risk. Past performance doesn&apos;t guarantee future results. 
                  Only trade with capital you can afford to lose.
                </p>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
