'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { 
  Brain,
  ArrowLeft,
  Save,
  Loader2,
  Shield,
  Target,
  Percent,
  AlertTriangle,
  Zap,
  TrendingUp,
  Settings,
  ChevronDown,
  CheckCircle,
  Info,
  RefreshCw
} from 'lucide-react'

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
    description: 'Balanced risk/reward. Uses trailing stops for exits.'
  },
  LOCK_PROFIT: {
    takeProfitPercent: 2.0,
    stopLossPercent: 1.0,
    trailingStopPercent: 0.25,
    minProfitToTrail: 0.5,
    description: 'Locks in profits quickly with tighter stops.'
  },
  MICRO_PROFIT: {
    takeProfitPercent: 0.5,
    stopLossPercent: 0.3,
    trailingStopPercent: 0.10,
    minProfitToTrail: 0.2,
    description: 'Many small profits with minimal risk per trade.'
  },
  SCALPER: {
    takeProfitPercent: 0.3,
    stopLossPercent: 0.2,
    trailingStopPercent: 0.08,
    minProfitToTrail: 0.15,
    description: 'Ultra-fast trades. High frequency, small profits.'
  },
  SWING: {
    takeProfitPercent: 5.0,
    stopLossPercent: 2.5,
    trailingStopPercent: 0.5,
    minProfitToTrail: 1.0,
    description: 'Longer holds for bigger moves. Patient approach.'
  }
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<BotSettings>(defaultSettings)
  const [isSaving, setIsSaving] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [hasChanges, setHasChanges] = useState(false)

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
          // Map backend names to frontend names
          const backendData = data.data
          const mapped = {
            ...backendData,
            // Map AI model settings
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
      // Map frontend names to backend names
      const settingsToSave = {
        ...settings,
        // Map AI model settings for backend
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
      ...preset
    }))
    setHasChanges(true)
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0a0f1a] flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#0a0f1a]">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-[#0a0f1a]/95 backdrop-blur-xl border-b border-white/5">
        <div className="w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/dashboard" 
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-400" />
              </Link>
              <div>
                <h1 className="text-xl font-bold text-white">Trading Settings</h1>
                <p className="text-sm text-gray-500">Configure AI trading parameters</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={loadSettings}
                className="p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <RefreshCw className="w-5 h-5 text-gray-400" />
              </button>
              
              <button
                onClick={saveSettings}
                disabled={isSaving || !hasChanges}
                className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-all ${
                  hasChanges
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:shadow-lg hover:shadow-cyan-500/25'
                    : 'bg-white/5 text-gray-500 cursor-not-allowed'
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

      {/* Main Content */}
      <main className="p-6">
        <div className="max-w-5xl mx-auto space-y-6">
          
          {/* Risk Mode Section */}
          <section className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
            <div className="p-5 border-b border-white/5">
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">Risk Mode</h2>
              </div>
              <p className="text-sm text-gray-500 mt-1">Select a trading strategy preset</p>
            </div>
            
            <div className="p-5 grid md:grid-cols-3 gap-4">
              {(Object.keys(riskPresets) as Array<keyof typeof riskPresets>).map((mode) => (
                <button
                  key={mode}
                  onClick={() => applyPreset(mode)}
                  className={`p-4 rounded-xl border text-left transition-all ${
                    settings.riskMode === mode
                      ? 'bg-cyan-500/10 border-cyan-500/30'
                      : 'bg-white/[0.02] border-white/10 hover:border-white/20'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className={`font-semibold ${settings.riskMode === mode ? 'text-cyan-400' : 'text-white'}`}>
                      {mode.replace('_', ' ')}
                    </span>
                    {settings.riskMode === mode && (
                      <CheckCircle className="w-4 h-4 text-cyan-400" />
                    )}
                  </div>
                  <p className="text-xs text-gray-500">{riskPresets[mode].description}</p>
                </button>
              ))}
            </div>
          </section>

          {/* Exit Strategy */}
          <section className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
            <div className="p-5 border-b border-white/5">
              <div className="flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">Exit Strategy</h2>
              </div>
              <p className="text-sm text-gray-500 mt-1">Configure take profit and stop loss settings</p>
            </div>
            
            <div className="p-5 grid md:grid-cols-2 gap-6">
              {/* Take Profit */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Take Profit %</span>
                  <span className="text-sm text-cyan-400 font-mono">
                    {settings.takeProfitPercent === 0 ? 'OFF' : `${settings.takeProfitPercent}%`}
                  </span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.5"
                  value={settings.takeProfitPercent}
                  onChange={(e) => updateSetting('takeProfitPercent', parseFloat(e.target.value))}
                  className="w-full accent-cyan-500"
                />
                <p className="text-xs text-gray-600 mt-1">Set to 0 to use trailing stop only</p>
              </div>

              {/* Stop Loss */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Stop Loss %</span>
                  <span className="text-sm text-red-400 font-mono">{settings.stopLossPercent}%</span>
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="5"
                  step="0.1"
                  value={settings.stopLossPercent}
                  onChange={(e) => updateSetting('stopLossPercent', parseFloat(e.target.value))}
                  className="w-full accent-red-500"
                />
              </div>

              {/* Trailing Stop */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Trailing Stop %</span>
                  <span className="text-sm text-amber-400 font-mono">{settings.trailingStopPercent}%</span>
                </label>
                <input
                  type="range"
                  min="0.05"
                  max="2"
                  step="0.01"
                  value={settings.trailingStopPercent}
                  onChange={(e) => updateSetting('trailingStopPercent', parseFloat(e.target.value))}
                  className="w-full accent-amber-500"
                />
                <p className="text-xs text-gray-600 mt-1">Sell when price drops this % from peak</p>
              </div>

              {/* Min Profit to Trail */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Min Profit to Trail</span>
                  <span className="text-sm text-emerald-400 font-mono">{settings.minProfitToTrail}%</span>
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="3"
                  step="0.05"
                  value={settings.minProfitToTrail}
                  onChange={(e) => updateSetting('minProfitToTrail', parseFloat(e.target.value))}
                  className="w-full accent-emerald-500"
                />
                <p className="text-xs text-gray-600 mt-1">Trailing activates after this profit</p>
              </div>
            </div>
          </section>

          {/* Position Sizing */}
          <section className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
            <div className="p-5 border-b border-white/5">
              <div className="flex items-center gap-2">
                <Percent className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">Position Sizing</h2>
              </div>
              <p className="text-sm text-gray-500 mt-1">Control trade sizes and limits</p>
            </div>
            
            <div className="p-5 grid md:grid-cols-2 gap-6">
              {/* Max Open Positions */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Max Open Positions</span>
                  <span className="text-sm text-cyan-400 font-mono">{settings.maxOpenPositions}</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="1"
                  value={settings.maxOpenPositions}
                  onChange={(e) => updateSetting('maxOpenPositions', parseInt(e.target.value))}
                  className="w-full accent-cyan-500"
                />
              </div>

              {/* Max Position Percent */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Max Position Size %</span>
                  <span className="text-sm text-cyan-400 font-mono">{settings.maxPositionPercent}%</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="30"
                  step="1"
                  value={settings.maxPositionPercent}
                  onChange={(e) => updateSetting('maxPositionPercent', parseInt(e.target.value))}
                  className="w-full accent-cyan-500"
                />
                <p className="text-xs text-gray-600 mt-1">Max % of equity per trade</p>
              </div>

              {/* Max Total Exposure */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Max Total Exposure %</span>
                  <span className="text-sm text-cyan-400 font-mono">{settings.maxTotalExposure}%</span>
                </label>
                <input
                  type="range"
                  min="20"
                  max="100"
                  step="5"
                  value={settings.maxTotalExposure}
                  onChange={(e) => updateSetting('maxTotalExposure', parseInt(e.target.value))}
                  className="w-full accent-cyan-500"
                />
                <p className="text-xs text-gray-600 mt-1">Total portfolio allocation limit</p>
              </div>

              {/* Kelly Multiplier */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Kelly Multiplier</span>
                  <span className="text-sm text-cyan-400 font-mono">{settings.kellyMultiplier}</span>
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={settings.kellyMultiplier}
                  onChange={(e) => updateSetting('kellyMultiplier', parseFloat(e.target.value))}
                  className="w-full accent-cyan-500"
                />
                <p className="text-xs text-gray-600 mt-1">Higher = larger positions, more risk</p>
              </div>

              {/* Dynamic Sizing Toggle */}
              <div className="md:col-span-2 flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">Dynamic Position Sizing</span>
                  <p className="text-xs text-gray-500 mt-0.5">AI adjusts size based on confidence (Kelly Criterion)</p>
                </div>
                <button
                  onClick={() => updateSetting('useDynamicSizing', !settings.useDynamicSizing)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.useDynamicSizing ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.useDynamicSizing ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>
          </section>

          {/* AI Filters */}
          <section className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
            <div className="p-5 border-b border-white/5">
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">AI Filters</h2>
              </div>
              <p className="text-sm text-gray-500 mt-1">Control trade quality requirements</p>
            </div>
            
            <div className="p-5 grid md:grid-cols-2 gap-6">
              {/* Min Edge */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Min Edge Score</span>
                  <span className="text-sm text-cyan-400 font-mono">{settings.minEdge}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="0.5"
                  step="0.01"
                  value={settings.minEdge}
                  onChange={(e) => updateSetting('minEdge', parseFloat(e.target.value))}
                  className="w-full accent-cyan-500"
                />
                <p className="text-xs text-gray-600 mt-1">Higher = fewer but better trades</p>
              </div>

              {/* Min Confidence */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Min Confidence %</span>
                  <span className="text-sm text-cyan-400 font-mono">{settings.minConfidence}%</span>
                </label>
                <input
                  type="range"
                  min="40"
                  max="90"
                  step="5"
                  value={settings.minConfidence}
                  onChange={(e) => updateSetting('minConfidence', parseInt(e.target.value))}
                  className="w-full accent-cyan-500"
                />
              </div>

              {/* Leverage Mode */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Leverage Mode
                </label>
                <div className="relative">
                  <select
                    value={settings.leverageMode}
                    onChange={(e) => updateSetting('leverageMode', e.target.value)}
                    className="w-full px-4 py-3 bg-[#0d1321] border border-white/10 rounded-xl text-white focus:border-cyan-500/50 focus:outline-none appearance-none cursor-pointer"
                    style={{ colorScheme: 'dark' }}
                  >
                    <option value="AUTO" className="bg-[#0d1321] text-white">AUTO (AI decides)</option>
                    <option value="1" className="bg-[#0d1321] text-white">1x (No leverage)</option>
                    <option value="2" className="bg-[#0d1321] text-white">2x</option>
                    <option value="3" className="bg-[#0d1321] text-white">3x</option>
                    <option value="5" className="bg-[#0d1321] text-white">5x</option>
                    <option value="10" className="bg-[#0d1321] text-white">10x</option>
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 pointer-events-none" />
                </div>
              </div>

              {/* Momentum Threshold */}
              <div>
                <label className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Momentum Threshold</span>
                  <span className="text-sm text-cyan-400 font-mono">
                    {settings.momentumThreshold === 0 ? 'OFF' : `${settings.momentumThreshold}%`}
                  </span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="0.1"
                  step="0.01"
                  value={settings.momentumThreshold}
                  onChange={(e) => updateSetting('momentumThreshold', parseFloat(e.target.value))}
                  className="w-full accent-cyan-500"
                />
                <p className="text-xs text-gray-600 mt-1">Min 24h momentum to trade</p>
              </div>
            </div>
          </section>

          {/* AI Models */}
          <section className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
            <div className="p-5 border-b border-white/5">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">AI Models</h2>
              </div>
              <p className="text-sm text-gray-500 mt-1">Enable or disable AI analysis features</p>
            </div>
            
            <div className="p-5 space-y-4">
              {/* Whale Detection */}
              <div className="flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">Whale Detection</span>
                  <p className="text-xs text-gray-500 mt-0.5">Detect large buy/sell walls from order books</p>
                </div>
                <button
                  onClick={() => updateSetting('useWhaleDetection', !settings.useWhaleDetection)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.useWhaleDetection ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.useWhaleDetection ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              {/* Funding Rate Analysis */}
              <div className="flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">Funding Rate Analysis</span>
                  <p className="text-xs text-gray-500 mt-0.5">Consider funding rates in trade decisions</p>
                </div>
                <button
                  onClick={() => updateSetting('useFundingRate', !settings.useFundingRate)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.useFundingRate ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.useFundingRate ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              {/* Market Regime Filter */}
              <div className="flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">Market Regime Filter</span>
                  <p className="text-xs text-gray-500 mt-0.5">Adjust strategy based on market conditions</p>
                </div>
                <button
                  onClick={() => updateSetting('useRegimeFilter', !settings.useRegimeFilter)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.useRegimeFilter ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.useRegimeFilter ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              {/* Dynamic Position Sizing */}
              <div className="flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">Dynamic Position Sizing</span>
                  <p className="text-xs text-gray-500 mt-0.5">Kelly Criterion based position sizing</p>
                </div>
                <button
                  onClick={() => updateSetting('useDynamicSizing', !settings.useDynamicSizing)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.useDynamicSizing ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.useDynamicSizing ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              {/* Pattern Recognition */}
              <div className="flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">Pattern Recognition</span>
                  <p className="text-xs text-gray-500 mt-0.5">ML-based price pattern detection</p>
                </div>
                <button
                  onClick={() => updateSetting('usePatternRecognition', !settings.usePatternRecognition)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.usePatternRecognition ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.usePatternRecognition ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              {/* Sentiment Analysis */}
              <div className="flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">Sentiment Analysis</span>
                  <p className="text-xs text-gray-500 mt-0.5">CryptoBERT sentiment from news & social</p>
                </div>
                <button
                  onClick={() => updateSetting('useSentimentAnalysis', !settings.useSentimentAnalysis)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.useSentimentAnalysis ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.useSentimentAnalysis ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              {/* XGBoost Predictions */}
              <div className="flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">XGBoost Predictions</span>
                  <p className="text-xs text-gray-500 mt-0.5">ML model for price movement prediction</p>
                </div>
                <button
                  onClick={() => updateSetting('useXGBoost', !settings.useXGBoost)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.useXGBoost ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.useXGBoost ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              {/* Edge Estimation */}
              <div className="flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">Edge Estimation</span>
                  <p className="text-xs text-gray-500 mt-0.5">Statistical edge calculation per symbol</p>
                </div>
                <button
                  onClick={() => updateSetting('useEdgeEstimation', !settings.useEdgeEstimation)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.useEdgeEstimation ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.useEdgeEstimation ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              {/* Q-Learning Strategy */}
              <div className="flex items-center justify-between p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <div>
                  <span className="font-medium text-white">Q-Learning Strategy</span>
                  <p className="text-xs text-gray-500 mt-0.5">Reinforcement learning for trade decisions</p>
                </div>
                <button
                  onClick={() => updateSetting('useQLearning', !settings.useQLearning)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.useQLearning ? 'bg-cyan-500' : 'bg-white/10'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white shadow transition-transform ${
                    settings.useQLearning ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>
          </section>

          {/* Warning */}
          <div className="p-4 bg-amber-500/5 border border-amber-500/20 rounded-xl flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-medium text-amber-400">Risk Warning</span>
              <p className="text-sm text-gray-400 mt-1">
                Trading involves significant risk. Past performance doesn't guarantee future results. 
                Only trade with capital you can afford to lose.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
