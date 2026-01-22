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
  Sparkles,
  HelpCircle,
  BookOpen,
  Info,
  DollarSign,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react'
import Logo from '@/components/Logo'

interface BotSettings {
  // AI Full Auto Mode
  aiFullAuto: boolean  // When ON: AI manages everything automatically
  useMaxTradeTime: boolean  // Use preset's max trade time limit
  
  strategyPreset: 'scalp' | 'micro' | 'swing' | 'conservative' | 'balanced' | 'aggressive'
  riskMode: 'SCALP' | 'MICRO' | 'SWING' | 'CONSERVATIVE' | 'BALANCED' | 'AGGRESSIVE'
  takeProfitPercent: number
  stopLossPercent: number
  trailingStopPercent: number
  minProfitToTrail: number
  maxOpenPositions: number
  maxPositionPercent: number
  maxTotalExposure: number
  maxDailyDrawdown: number
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
  breakoutExtraSlots: boolean  // Allow +2 extra positions for breakouts
}

const defaultSettings: BotSettings = {
  // AI Full Auto Mode
  aiFullAuto: false,  // OFF by default - user controls strategy
  useMaxTradeTime: true,  // Use preset's time limit by default
  
  strategyPreset: 'micro',  // MICRO is the best default (70-80% winrate)
  riskMode: 'MICRO',
  takeProfitPercent: 0.9,  // MICRO default
  stopLossPercent: 0.5,    // MICRO default
  trailingStopPercent: 0.8,
  minProfitToTrail: 0.5,
  maxOpenPositions: 10,
  maxPositionPercent: 10,
  maxTotalExposure: 80,
  maxDailyDrawdown: 0,  // 0 = OFF (no daily limit)
  kellyMultiplier: 0.3,
  minEdge: 0.10,
  minConfidence: 60,
  leverageMode: 'auto',
  useDynamicSizing: true,
  useWhaleDetection: true,
  useFundingRate: true,
  useRegimeFilter: true,
  usePatternRecognition: true,
  useSentimentAnalysis: true,
  useXGBoost: true,
  useEdgeEstimation: true,
  useQLearning: true,
  momentumThreshold: 0,
  breakoutExtraSlots: false  // OFF by default - user must enable
}

// Strategy presets - PROFESSIONAL PRESETS (from ChatGPT quant analysis)
// Each preset has optimized entry thresholds, AI requirements, and exit rules
const riskPresets = {
  // ⚡ SCALP - Fastest, most dangerous
  SCALP: {
    takeProfitPercent: 0.55,  // +0.4% to +0.7% (average)
    stopLossPercent: 0.35,
    trailingStopPercent: 0.12,
    minProfitToTrail: 0.35,
    winrate: '65-75%',
    regime: 'RANGE',
    description: '⚡ Quick bounce, small profit - HIGH RISK, HIGH FREQ',
    icon: Zap,
    color: 'red'
  },
  // 💎 MICRO - HEALTHIEST (RECOMMENDED)
  MICRO: {
    takeProfitPercent: 0.9,   // +0.6% to +1.2% (average)
    stopLossPercent: 0.5,
    trailingStopPercent: 0.14,
    minProfitToTrail: 0.45,
    winrate: '70-80%',
    regime: 'RANGE/CHOPPY',
    description: '💎 DEFAULT MONEY PRINTER - Best risk/reward ⭐',
    icon: Target,
    color: 'emerald'
  },
  // 🧠 SWING - Slow but stable
  SWING: {
    takeProfitPercent: 4.0,   // +2% to +6% (average)
    stopLossPercent: 1.2,
    trailingStopPercent: 0.35,
    minProfitToTrail: 1.2,
    winrate: '55-65%',
    regime: 'TREND',
    description: '🧠 Bigger moves, less trades - R:R > 2.5',
    icon: TrendingUp,
    color: 'purple'
  },
  // Simple presets for beginners
  CONSERVATIVE: {
    takeProfitPercent: 1.0,
    stopLossPercent: 0.6,
    trailingStopPercent: 0.2,
    minProfitToTrail: 0.15,
    winrate: '75-85%',
    regime: 'ANY',
    description: '🛡️ Capital preservation - tight everything',
    icon: Shield,
    color: 'blue'
  },
  BALANCED: {
    takeProfitPercent: 3.0,
    stopLossPercent: 1.5,
    trailingStopPercent: 0.8,
    minProfitToTrail: 0.5,
    winrate: '60-70%',
    regime: 'ANY',
    description: '⚖️ Middle ground - for beginners',
    icon: Activity,
    color: 'cyan'
  },
  AGGRESSIVE: {
    takeProfitPercent: 5.0,
    stopLossPercent: 2.5,
    trailingStopPercent: 1.2,
    minProfitToTrail: 1.0,
    winrate: '50-60%',
    regime: 'TREND',
    description: '🔥 Big wins, big losses - for risk takers',
    icon: Sparkles,
    color: 'orange'
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
  const [userId, setUserId] = useState<string>('default')

  useEffect(() => {
    // Get user ID from localStorage (each user has their own settings!)
    // MUST match dashboard logic: admin@sentinel.ai -> 'default', others -> their UUID
    const storedUser = localStorage.getItem('sentinel_user')
    
    let currentUserId = 'default'
    if (storedUser) {
      try {
        const user = JSON.parse(storedUser)
        // Admin user is identified by EMAIL, not numeric ID
        const isAdmin = user.email === 'admin@sentinel.ai'
        currentUserId = isAdmin ? 'default' : (user.id || user.userId || user.user_id || 'default')
      } catch (e) {
        console.error('Failed to parse user from localStorage')
      }
    }
    
    setUserId(currentUserId)
    console.log('Settings page - User ID:', currentUserId)
    
    // DEBUG: Show alert with user info so we can verify on phone
    if (typeof window !== 'undefined') {
      const debugInfo = `User ID: ${currentUserId}\nEmail: ${storedUser ? JSON.parse(storedUser).email : 'N/A'}`
      console.log('DEBUG Settings:', debugInfo)
      // Uncomment to show alert on page load:
      // alert(debugInfo)
    }
  }, [])

  useEffect(() => {
    if (userId) {
      loadSettings()
    }
  }, [userId])

  const loadSettings = async () => {
    setIsLoading(true)
    try {
      console.log('Loading settings for user:', userId)
      const response = await fetch(`/ai/exchange/settings?user_id=${userId}`)
      if (response.ok) {
        const data = await response.json()
        if (data.data) {
          const backendData = data.data
          // Normalize riskMode to uppercase
          const riskMode = (backendData.riskMode || 'MICRO').toUpperCase() as keyof typeof riskPresets
          const validRiskMode = riskPresets[riskMode] ? riskMode : 'MICRO'
          
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
      
      console.log('Saving settings for user:', userId)
      const response = await fetch(`/ai/exchange/settings?user_id=${userId}`, {
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
    // Map frontend mode names to backend strategyPreset names
    const presetMap: { [key: string]: string } = {
      'SCALP': 'scalp',
      'MICRO': 'micro',
      'SWING': 'swing',
      'CONSERVATIVE': 'conservative',
      'BALANCED': 'balanced',
      'AGGRESSIVE': 'aggressive'
    }
    setSettings(prev => ({
      ...prev,
      strategyPreset: (presetMap[mode] || 'micro') as BotSettings['strategyPreset'],
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
    { id: 'guide', label: 'User Guide', icon: HelpCircle },
  ]

  // Get stored user email for debug display
  const storedUserForDebug = typeof window !== 'undefined' ? localStorage.getItem('sentinel_user') : null
  const debugEmail = storedUserForDebug ? JSON.parse(storedUserForDebug).email : 'N/A'

  return (
    <div className="min-h-screen bg-[#060a13]">
      {/* DEBUG BANNER - Commented out after confirming multi-user isolation works
      <div className="bg-yellow-500/20 border border-yellow-500/50 text-yellow-300 text-xs px-4 py-2 text-center">
        🔍 DEBUG: user_id={userId} | email={debugEmail}
      </div>
      */}

      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.03)_1px,transparent_1px)] bg-[size:44px_44px]" />
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-cyan-500/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-blue-500/5 rounded-full blur-[120px]" />
      </div>

      {/* Header */}
      <header className="sticky top-0 z-50 bg-[#060a13]/90 backdrop-blur-xl border-b border-white/5">
        <div className="w-full px-3 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14 sm:h-16">
            <div className="flex items-center gap-2 sm:gap-4">
              <Link 
                href="/dashboard" 
                className="flex items-center gap-2 sm:gap-3 p-1.5 sm:p-2 -ml-1 sm:-ml-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                <ArrowLeft className="w-4 h-4 sm:w-5 sm:h-5 text-gray-400" />
                <div className="hidden sm:block"><Logo size="sm" /></div>
              </Link>
              <div className="hidden sm:block h-6 w-px bg-white/10" />
              <div>
                <h1 className="text-sm sm:text-lg font-semibold text-white">Settings</h1>
                <p className="hidden sm:block text-xs text-gray-500">Configure AI trading parameters</p>
              </div>
            </div>
            
            <div className="flex items-center gap-2 sm:gap-3">
              <button
                onClick={loadSettings}
                className="p-2 sm:p-2.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-all"
                title="Refresh"
              >
                <RefreshCw className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-400" />
              </button>
              
              <button
                onClick={saveSettings}
                disabled={isSaving || !hasChanges}
                className={`flex items-center gap-1.5 sm:gap-2 px-3 sm:px-5 py-2 sm:py-2.5 rounded-xl font-medium text-xs sm:text-sm transition-all ${
                  hasChanges
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30'
                    : 'bg-white/5 text-gray-500 border border-white/10 cursor-not-allowed'
                }`}
              >
                {isSaving ? (
                  <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" />
                ) : saveStatus === 'success' ? (
                  <CheckCircle className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                ) : (
                  <Save className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                )}
                <span className="hidden xs:inline">{isSaving ? 'Saving...' : saveStatus === 'success' ? 'Saved!' : 'Save'}</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-col lg:flex-row relative">
        {/* Mobile Navigation - Horizontal scroll */}
        <div className="lg:hidden overflow-x-auto border-b border-white/5 bg-[#0a0f1a]/50 sticky top-16 z-40">
          <nav className="flex gap-1 p-2 min-w-max">
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveSection(item.id)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                    activeSection === item.id
                      ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </button>
              )
            })}
          </nav>
        </div>

        {/* Desktop Sidebar Navigation */}
        <aside className="hidden lg:block w-64 min-h-[calc(100vh-64px)] bg-[#0a0f1a]/50 border-r border-white/5 p-4 sticky top-16">
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
              <span className="text-sm font-medium text-white">{(settings.riskMode || 'MICRO').replace('_', ' ')}</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {riskPresets[settings.riskMode as keyof typeof riskPresets]?.description || 'Balanced risk/reward strategy'}
            </p>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8">
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
                    <p className="text-sm text-gray-500">Select a preset or let AI manage everything</p>
                  </div>
                </div>

                {/* AI FULL AUTO MODE */}
                <div className="mb-6 p-5 rounded-2xl bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-purple-500/20">
                        <Brain className="w-5 h-5 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">AI Full Auto Mode</h3>
                        <p className="text-xs text-gray-400">AI manages strategy, positions & risk automatically</p>
                      </div>
                    </div>
                    <button
                      onClick={() => updateSetting('aiFullAuto', !settings.aiFullAuto)}
                      className={`relative w-14 h-7 rounded-full transition-colors ${
                        settings.aiFullAuto ? 'bg-purple-500' : 'bg-gray-700'
                      }`}
                    >
                      <div className={`absolute top-1 w-5 h-5 rounded-full bg-white transition-transform ${
                        settings.aiFullAuto ? 'translate-x-8' : 'translate-x-1'
                      }`} />
                    </button>
                  </div>
                  
                  {settings.aiFullAuto && (
                    <div className="mt-3 space-y-3">
                      <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                        <p className="text-xs text-purple-300">
                          🤖 <span className="font-semibold">AI is in full control!</span> The bot will automatically:
                        </p>
                        <ul className="mt-2 text-xs text-gray-400 space-y-1">
                          <li>• Switch between SCALP, MICRO, SWING based on market</li>
                          <li>• Decide number of positions (3-15 based on conditions)</li>
                          <li>• Set TP%, SL%, trailing for each strategy</li>
                          <li>• Adjust confidence/edge thresholds based on win rate</li>
                          <li>• Apply ALL safety filters (RSI, BTC correlation, spread, etc.)</li>
                        </ul>
                      </div>
                      
                      {/* MAX DAILY DRAWDOWN - Only user control in AI Full Auto */}
                      <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30">
                        <div className="flex items-center gap-2 mb-3">
                          <AlertTriangle className="w-4 h-4 text-red-400" />
                          <p className="text-sm font-semibold text-red-400">Your Only Protection</p>
                        </div>
                        <p className="text-xs text-gray-400 mb-3">
                          Max Daily Drawdown is the ONLY setting you control. AI respects this limit to protect your capital.
                        </p>
                        <div className="flex items-center gap-4">
                          <div className="flex-1">
                            <label className="text-xs text-gray-500">Max Daily Loss %</label>
                            <input
                              type="number"
                              value={settings.maxDailyDrawdown}
                              onChange={(e) => updateSetting('maxDailyDrawdown', parseFloat(e.target.value) || 0)}
                              className="w-full mt-1 px-3 py-2 bg-black/30 border border-red-500/30 rounded-lg text-white text-sm focus:border-red-500 focus:outline-none"
                              min={0}
                              max={100}
                              step={0.5}
                            />
                          </div>
                          <div className="text-right">
                            <p className="text-xs text-gray-500">0 = No limit</p>
                            <p className="text-lg font-bold text-red-400">
                              {settings.maxDailyDrawdown > 0 ? `-${settings.maxDailyDrawdown}%` : 'OFF'}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {!settings.aiFullAuto && (
                    <div className="flex items-center justify-between mt-3 pt-3 border-t border-white/10">
                      <div>
                        <p className="text-sm text-gray-300">Use Max Trade Time</p>
                        <p className="text-xs text-gray-500">Close positions after preset's time limit</p>
                      </div>
                      <button
                        onClick={() => updateSetting('useMaxTradeTime', !settings.useMaxTradeTime)}
                        className={`relative w-12 h-6 rounded-full transition-colors ${
                          settings.useMaxTradeTime ? 'bg-cyan-500' : 'bg-gray-700'
                        }`}
                      >
                        <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                          settings.useMaxTradeTime ? 'translate-x-7' : 'translate-x-1'
                        }`} />
                      </button>
                    </div>
                  )}
                </div>

                {/* Strategy Presets - only show if not in AI Full Auto */}
                {!settings.aiFullAuto && (
                  <p className="text-sm text-gray-400 mb-4">Select your trading strategy:</p>
                )}
                
                <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 ${settings.aiFullAuto ? 'opacity-50 pointer-events-none' : ''}`}>
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

                  {/* Max Daily Drawdown */}
                  <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium text-white">Max Daily Drawdown</span>
                      <span className={`px-3 py-1 rounded-full text-sm font-mono ${
                        settings.maxDailyDrawdown === 0 
                          ? 'bg-gray-500/10 text-gray-400' 
                          : 'bg-red-500/10 text-red-400'
                      }`}>
                        {settings.maxDailyDrawdown === 0 ? 'OFF' : `${settings.maxDailyDrawdown}%`}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="20"
                      step="1"
                      value={settings.maxDailyDrawdown}
                      onChange={(e) => updateSetting('maxDailyDrawdown', parseInt(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer bg-white/10 accent-red-500"
                    />
                    <p className="text-xs text-gray-500 mt-3">0 = OFF (no daily loss limit). Stops trading after X% daily loss.</p>
                  </div>

                  {/* Breakout Extra Slots */}
                  <div className={`p-5 rounded-2xl border transition-all ${
                    settings.breakoutExtraSlots 
                      ? 'bg-emerald-500/5 border-emerald-500/20' 
                      : 'bg-white/[0.02] border-white/5'
                  }`}>
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <Zap className={`w-4 h-4 ${settings.breakoutExtraSlots ? 'text-emerald-400' : 'text-gray-500'}`} />
                          <span className="font-medium text-white">Breakout Extra Slots</span>
                        </div>
                        <p className="text-xs text-gray-500">
                          Allow +2 extra positions for high-conviction breakouts (coins moving +10% or more). 
                          When enabled: if you have 7 max positions, breakouts can open up to 9.
                        </p>
                      </div>
                      <button
                        onClick={() => updateSetting('breakoutExtraSlots', !settings.breakoutExtraSlots)}
                        className={`relative w-12 h-6 rounded-full transition-colors ml-4 ${
                          settings.breakoutExtraSlots ? 'bg-emerald-500' : 'bg-white/10'
                        }`}
                      >
                        <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow-lg transition-transform ${
                          settings.breakoutExtraSlots ? 'left-[26px]' : 'left-0.5'
                        }`} />
                      </button>
                    </div>
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
                        <option value="auto">AUTO (AI decides)</option>
                        <option value="1x">1x (No leverage)</option>
                        <option value="2x">2x</option>
                        <option value="3x">3x</option>
                        <option value="5x">5x</option>
                        <option value="10x">10x</option>
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

            {/* User Guide Section */}
            {activeSection === 'guide' && (
              <section className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                    <BookOpen className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-white">User Guide</h2>
                    <p className="text-sm text-gray-500">Complete documentation for Sentinel AI Trading Bot</p>
                  </div>
                </div>

                {/* Dashboard Header Guide */}
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5 space-y-4">
                  <h3 className="text-lg font-semibold text-cyan-400 flex items-center gap-2">
                    <Info className="w-5 h-5" />
                    Dashboard Header Stats
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">F/G (Fear & Greed Index)</div>
                      <p className="text-xs text-gray-400">
                        Market sentiment indicator from 0-100. Below 25 = Extreme Fear (potential buying opportunity), 
                        above 75 = Extreme Greed (potential selling opportunity). Helps gauge overall market mood.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">Regime</div>
                      <p className="text-xs text-gray-400">
                        Current market condition detected by AI:<br/>
                        • <span className="text-green-400">trending</span> - Strong trend, good for trades<br/>
                        • <span className="text-yellow-400">range_bound</span> - Sideways movement, be cautious<br/>
                        • <span className="text-red-400">high_volatility</span> - Risky conditions<br/>
                        • <span className="text-orange-400">low_liquidity</span> - Slippage risk
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">AI Confidence</div>
                      <p className="text-xs text-gray-400">
                        The AI&apos;s current confidence level in market conditions. Higher percentage means 
                        the AI is more certain about its analysis. Below 50% indicates uncertain market conditions.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">Pairs Scanned</div>
                      <p className="text-xs text-gray-400">
                        Total number of trading pairs the AI is actively monitoring on Bybit. 
                        More pairs = more opportunities but requires more processing.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Balance Section Guide */}
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5 space-y-4">
                  <h3 className="text-lg font-semibold text-emerald-400 flex items-center gap-2">
                    <DollarSign className="w-5 h-5" />
                    Balance & Performance
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">Balance (USDT / EUR)</div>
                      <p className="text-xs text-gray-400">
                        Your total account equity in USDT and converted to EUR. This includes both 
                        available balance and value locked in open positions.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">Daily P&L</div>
                      <p className="text-xs text-gray-400">
                        Profit or Loss for the current day in EUR. Resets at midnight UTC. 
                        <span className="text-green-400"> Green = profit</span>, 
                        <span className="text-red-400"> Red = loss</span>.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">Win Rate</div>
                      <p className="text-xs text-gray-400">
                        Percentage of profitable trades. Shows total Wins (W) and Losses (L). 
                        A win rate above 50% with proper risk management is generally profitable.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">Total Trades</div>
                      <p className="text-xs text-gray-400">
                        Total number of completed trades since account creation. 
                        Used to track overall trading activity and calculate statistics.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Open Positions Guide */}
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5 space-y-4">
                  <h3 className="text-lg font-semibold text-blue-400 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    Open Positions Table
                  </h3>
                  
                  <div className="space-y-3">
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">PAIR</div>
                      <p className="text-xs text-gray-400">
                        The cryptocurrency trading pair (e.g., BTCUSDT = Bitcoin/USDT). 
                        Shows which asset you have a position in.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">SIDE</div>
                      <p className="text-xs text-gray-400">
                        <span className="text-green-400">LONG</span> - You profit when price goes UP<br/>
                        <span className="text-red-400">SHORT</span> - You profit when price goes DOWN<br/>
                        The leverage multiplier (e.g., 2x, 5x) amplifies both gains and losses.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">VALUE</div>
                      <p className="text-xs text-gray-400">
                        The current value of your position in EUR. This is your initial investment 
                        multiplied by leverage and adjusted for current P&L.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">ENTRY / MARK</div>
                      <p className="text-xs text-gray-400">
                        <strong>Entry</strong> - The price at which you entered the trade<br/>
                        <strong>Mark</strong> - The current market price<br/>
                        Compare these to see how the price has moved since entry.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">P&L (Profit & Loss)</div>
                      <p className="text-xs text-gray-400">
                        Shows your unrealized profit or loss in EUR and percentage.<br/>
                        <span className="text-green-400">Green/Positive</span> = Making profit<br/>
                        <span className="text-red-400">Red/Negative</span> = Currently at a loss<br/>
                        This is &quot;unrealized&quot; until you close the position.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">ACTION (Close Button)</div>
                      <p className="text-xs text-gray-400">
                        Manually close a position immediately at market price. Use this when you want 
                        to exit a trade without waiting for AI to close it automatically.
                      </p>
                    </div>
                  </div>
                </div>

                {/* AI Panels Guide */}
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5 space-y-4">
                  <h3 className="text-lg font-semibold text-violet-400 flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    AI Intelligence Panels
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">AI Signals</div>
                      <p className="text-xs text-gray-400">
                        Top trading opportunities identified by the AI. Shows potential entry points with 
                        target price (TP) and stop loss (SL) levels. Higher confidence signals are more reliable.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">AI Console</div>
                      <p className="text-xs text-gray-400">
                        Real-time log of AI activity. Shows trades opened/closed, breakout detections, 
                        and system status. Useful for monitoring what the bot is doing.
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">AI Intelligence Panel</div>
                      <p className="text-xs text-gray-400">
                        • <strong>Strategy</strong> - Current trading mode (NORMAL, LOCK_PROFIT, etc.)<br/>
                        • <strong>Breakouts</strong> - Number of coins with significant price moves<br/>
                        • <strong>Last Action</strong> - Most recent AI activity<br/>
                        • <strong>Active Breakouts</strong> - Coins currently breaking out
                      </p>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <div className="text-sm font-medium text-white mb-1">P&L Performance Chart</div>
                      <p className="text-xs text-gray-400">
                        Visual chart of your last 50-100 trades. Shows cumulative profit/loss over time. 
                        Green = positive performance, Red = negative. Helps identify trends in your trading.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Settings Guide */}
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5 space-y-4">
                  <h3 className="text-lg font-semibold text-amber-400 flex items-center gap-2">
                    <Settings className="w-5 h-5" />
                    Settings Explained
                  </h3>
                  
                  {/* Trading Strategy */}
                  <div className="p-4 rounded-xl bg-gradient-to-r from-white/[0.02] to-transparent border-l-2 border-cyan-500">
                    <h4 className="text-sm font-semibold text-cyan-400 mb-3">Trading Strategy</h4>
                    <div className="space-y-2 text-xs text-gray-400">
                      <p><span className="text-white font-medium">NORMAL</span> - Balanced approach with trailing stop protection. Best for most market conditions.</p>
                      <p><span className="text-white font-medium">LOCK PROFIT</span> - Aggressive profit protection. Sells quickly when profit starts dropping.</p>
                      <p><span className="text-white font-medium">MICRO PROFIT</span> - Takes many small profits. Higher win rate but smaller gains per trade.</p>
                      <p><span className="text-white font-medium">SCALPER</span> - Very fast trades with tight stops. Requires stable market conditions.</p>
                      <p><span className="text-white font-medium">SWING</span> - Holds positions longer for bigger moves. More patient approach.</p>
                    </div>
                  </div>

                  {/* Exit Rules */}
                  <div className="p-4 rounded-xl bg-gradient-to-r from-white/[0.02] to-transparent border-l-2 border-emerald-500">
                    <h4 className="text-sm font-semibold text-emerald-400 mb-3">Exit Rules</h4>
                    <div className="space-y-2 text-xs text-gray-400">
                      <p><span className="text-white font-medium">Take Profit %</span> - Automatically sells when profit reaches this percentage. Set to 0 to disable (use trailing stop only).</p>
                      <p><span className="text-white font-medium">Stop Loss %</span> - Maximum loss before automatic exit. Protects from large losses.</p>
                      <p><span className="text-white font-medium">Trailing Stop %</span> - Once in profit, sells if price drops by this % from peak. Locks in gains.</p>
                      <p><span className="text-white font-medium">Min Profit to Trail</span> - Minimum profit required before trailing stop activates.</p>
                    </div>
                  </div>

                  {/* Position Sizing */}
                  <div className="p-4 rounded-xl bg-gradient-to-r from-white/[0.02] to-transparent border-l-2 border-blue-500">
                    <h4 className="text-sm font-semibold text-blue-400 mb-3">Position Sizing</h4>
                    <div className="space-y-2 text-xs text-gray-400">
                      <p><span className="text-white font-medium">Max Open Positions</span> - Maximum number of simultaneous trades. More positions = more diversification but more risk.</p>
                      <p><span className="text-white font-medium">Max Position Size %</span> - Maximum percentage of your equity per trade.</p>
                      <p><span className="text-white font-medium">Max Total Exposure %</span> - Maximum total percentage of equity in all positions combined.</p>
                      <p><span className="text-white font-medium">Max Daily Drawdown</span> - Stops trading if daily loss exceeds this %. Set to 0 to disable.</p>
                      <p><span className="text-white font-medium">Kelly Multiplier</span> - Adjusts position sizing aggressiveness. Higher = bigger positions, more risk.</p>
                    </div>
                  </div>

                  {/* AI Filters */}
                  <div className="p-4 rounded-xl bg-gradient-to-r from-white/[0.02] to-transparent border-l-2 border-violet-500">
                    <h4 className="text-sm font-semibold text-violet-400 mb-3">AI Filters</h4>
                    <div className="space-y-2 text-xs text-gray-400">
                      <p><span className="text-white font-medium">Min Edge</span> - Minimum expected advantage required to enter a trade. Higher = fewer but better trades.</p>
                      <p><span className="text-white font-medium">Min Confidence</span> - Minimum AI confidence required. Higher = more selective, fewer trades.</p>
                      <p><span className="text-white font-medium">Momentum Threshold</span> - Minimum price momentum required. Filters out weak moves.</p>
                      <p><span className="text-white font-medium">Leverage Mode</span> - AUTO lets AI decide leverage, or set fixed 1x-10x.</p>
                    </div>
                  </div>

                  {/* AI Models */}
                  <div className="p-4 rounded-xl bg-gradient-to-r from-white/[0.02] to-transparent border-l-2 border-pink-500">
                    <h4 className="text-sm font-semibold text-pink-400 mb-3">AI Models</h4>
                    <div className="space-y-2 text-xs text-gray-400">
                      <p><span className="text-white font-medium">Whale Detection</span> - Tracks large wallet movements to follow smart money.</p>
                      <p><span className="text-white font-medium">Funding Rate Analysis</span> - Uses funding rates to detect over-leveraged markets.</p>
                      <p><span className="text-white font-medium">Market Regime Filter</span> - Adjusts strategy based on current market conditions.</p>
                      <p><span className="text-white font-medium">Dynamic Position Sizing</span> - AI adjusts position sizes based on confidence.</p>
                      <p><span className="text-white font-medium">Pattern Recognition</span> - Identifies chart patterns for entry/exit signals.</p>
                      <p><span className="text-white font-medium">Sentiment Analysis</span> - Analyzes market sentiment from news and social data.</p>
                      <p><span className="text-white font-medium">Q-Learning Strategy</span> - Self-learning AI that improves from past trades.</p>
                    </div>
                  </div>
                </div>

                {/* Backtest Guide */}
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5 space-y-4">
                  <h3 className="text-lg font-semibold text-orange-400 flex items-center gap-2">
                    <LineChart className="w-5 h-5" />
                    Backtest System
                  </h3>
                  
                  <div className="space-y-3 text-xs text-gray-400">
                    <p>
                      The <strong className="text-white">Backtest</strong> feature allows you to test trading strategies on historical data 
                      before risking real money.
                    </p>
                    <div className="p-3 rounded-lg bg-white/[0.02]">
                      <p className="mb-2"><span className="text-white font-medium">How to use:</span></p>
                      <ol className="list-decimal list-inside space-y-1">
                        <li>Select a trading pair (e.g., BTCUSDT)</li>
                        <li>Choose date range (recommended: at least 30 days)</li>
                        <li>Set initial capital amount</li>
                        <li>Configure strategy settings</li>
                        <li>Run backtest and analyze results</li>
                      </ol>
                    </div>
                    <p className="text-amber-400">
                      Note: Past performance in backtests does not guarantee future results. 
                      Market conditions change and historical patterns may not repeat.
                    </p>
                  </div>
                </div>

                {/* Quick Tips */}
                <div className="p-5 rounded-2xl bg-gradient-to-r from-cyan-500/5 to-blue-500/5 border border-cyan-500/10 space-y-4">
                  <h3 className="text-lg font-semibold text-cyan-400 flex items-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    Quick Tips for Success
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300">Start with smaller position sizes while learning</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300">Always use stop losses to protect capital</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300">Monitor the AI Console for trade activity</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300">Use NORMAL strategy until you understand the system</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300">Check market regime before expecting high activity</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300">Backtest strategies before using with real money</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300">Don&apos;t overtrade - quality over quantity</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300">Keep some balance available for new opportunities</span>
                    </div>
                  </div>
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
