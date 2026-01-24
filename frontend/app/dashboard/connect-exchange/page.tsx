'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Shield, 
  Key, 
  Eye, 
  EyeOff, 
  CheckCircle, 
  AlertCircle, 
  Loader2,
  ArrowRight,
  Info,
  ExternalLink,
  Wallet,
  Lock
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'

interface ExchangeConnection {
  id: string
  exchange: string
  name: string
  is_testnet: boolean
  is_active: boolean
  api_key_masked: string
  last_sync_at: string | null
}

export default function ConnectExchangePage() {
  const router = useRouter()
  const [step, setStep] = useState(1)
  const [selectedExchange, setSelectedExchange] = useState<string>('bybit')
  const [formData, setFormData] = useState({
    name: 'My Bybit Account',
    api_key: '',
    api_secret: '',
    is_testnet: false,
  })

  // Update form name when exchange is selected
  useEffect(() => {
    if (selectedExchange === 'binance') {
      setFormData(prev => ({ ...prev, name: 'My Binance Account' }))
    } else {
      setFormData(prev => ({ ...prev, name: 'My Bybit Account' }))
    }
  }, [selectedExchange])
  const [showSecret, setShowSecret] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [existingConnections, setExistingConnections] = useState<ExchangeConnection[]>([])
  const [loadingConnections, setLoadingConnections] = useState(true)

  // Check for existing connections
  useEffect(() => {
    const fetchConnections = async () => {
      try {
        const token = localStorage.getItem('token')
        if (!token) {
          router.push('/login')
          return
        }

        const response = await fetch('/api/exchanges', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        })

        if (response.ok) {
          const data = await response.json()
          setExistingConnections(data.data || [])
        }
      } catch (error) {
        console.error('Failed to fetch connections:', error)
      } finally {
        setLoadingConnections(false)
      }
    }

    fetchConnections()
  }, [router])

  const exchanges = [
    {
      id: 'bybit',
      name: 'Bybit',
      description: 'Leading crypto derivatives exchange',
      supported: true,
      apiGuideUrl: 'https://www.bybit.com/future-activity/en/developer',
    },
    {
      id: 'binance',
      name: 'Binance',
      description: "World's largest crypto exchange",
      supported: true,
      apiGuideUrl: 'https://www.binance.com/en/my/settings/api-management',
    },
    {
      id: 'okx',
      name: 'OKX',
      description: 'Global crypto trading platform',
      supported: false,
      comingSoon: true,
    },
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setIsLoading(true)

    try {
      const token = localStorage.getItem('token')
      if (!token) {
        router.push('/login')
        return
      }

      const response = await fetch('/api/exchanges', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          exchange: selectedExchange,
          ...formData,
        }),
      })

      const data = await response.json()

      if (response.ok && data.success) {
        setSuccess(true)
        setTimeout(() => {
          router.push('/dashboard')
        }, 2000)
      } else {
        setError(data.message || 'Failed to connect exchange')
      }
    } catch (err) {
      setError('Network error. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  if (loadingConnections) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
      </div>
    )
  }

  // If user already has any exchange connection, show management view
  const existingConnection = existingConnections.find(c => c.exchange === 'bybit' || c.exchange === 'binance')
  if (existingConnection) {
    const exchangeName = existingConnection.exchange === 'binance' ? 'Binance' : 'Bybit'
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6">
        <div className="max-w-2xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className={`w-16 h-16 rounded-xl flex items-center justify-center ${
                existingConnection.exchange === 'binance' 
                  ? 'bg-gradient-to-br from-yellow-400 to-yellow-600' 
                  : 'bg-gradient-to-br from-yellow-500 to-orange-500'
              }`}>
                <Wallet className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Exchange Connected</h1>
                <p className="text-slate-400">Your {exchangeName} account is already connected</p>
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-xl p-4 mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-400">Exchange</span>
                <span className="text-white font-medium">{exchangeName}</span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-400">Account Name</span>
                <span className="text-white font-medium">{existingConnection.name}</span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-400">API Key</span>
                <span className="text-white font-mono text-sm">{existingConnection.api_key_masked}</span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-400">Mode</span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  existingConnection.is_testnet 
                    ? 'bg-yellow-500/20 text-yellow-400' 
                    : 'bg-green-500/20 text-green-400'
                }`}>
                  {existingConnection.is_testnet ? 'Testnet' : 'Mainnet'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Status</span>
                <span className={`flex items-center gap-1.5 ${
                  existingConnection.is_active ? 'text-green-400' : 'text-red-400'
                }`}>
                  {existingConnection.is_active ? (
                    <>
                      <CheckCircle className="w-4 h-4" />
                      Active
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-4 h-4" />
                      Inactive
                    </>
                  )}
                </span>
              </div>
            </div>

            <div className="flex gap-3">
              <Link
                href="/dashboard"
                className="flex-1 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold text-center hover:from-cyan-400 hover:to-blue-500 transition-all"
              >
                Go to Dashboard
              </Link>
              <Link
                href="/dashboard/settings"
                className="px-6 py-3 bg-slate-800 text-white rounded-xl font-semibold hover:bg-slate-700 transition-all"
              >
                Settings
              </Link>
            </div>
          </motion.div>
        </div>
      </div>
    )
  }

  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center p-6">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="bg-slate-900/50 border border-green-500/30 rounded-2xl p-8 text-center max-w-md"
        >
          <div className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
            <CheckCircle className="w-10 h-10 text-green-400" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Exchange Connected!</h2>
          <p className="text-slate-400 mb-4">Your {selectedExchange === 'binance' ? 'Binance' : 'Bybit'} account has been successfully connected.</p>
          <p className="text-sm text-slate-500">Redirecting to dashboard...</p>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Connect Your Exchange</h1>
          <p className="text-slate-400">Link your exchange account to start AI-powered trading</p>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center justify-center gap-4 mb-8">
          {[1, 2, 3].map((s) => (
            <div key={s} className="flex items-center">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-all ${
                step >= s 
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white' 
                  : 'bg-slate-800 text-slate-500'
              }`}>
                {step > s ? <CheckCircle className="w-5 h-5" /> : s}
              </div>
              {s < 3 && (
                <div className={`w-16 h-1 mx-2 rounded ${
                  step > s ? 'bg-cyan-500' : 'bg-slate-800'
                }`} />
              )}
            </div>
          ))}
        </div>
        <div className="flex justify-center gap-16 mb-8 text-sm">
          <span className={step >= 1 ? 'text-cyan-400' : 'text-slate-500'}>Select Exchange</span>
          <span className={step >= 2 ? 'text-cyan-400' : 'text-slate-500'}>API Credentials</span>
          <span className={step >= 3 ? 'text-cyan-400' : 'text-slate-500'}>Verify</span>
        </div>

        {/* Step 1: Select Exchange */}
        {step === 1 && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4"
          >
            {exchanges.map((exchange) => (
              <button
                key={exchange.id}
                onClick={() => {
                  if (exchange.supported) {
                    setSelectedExchange(exchange.id)
                    setStep(2)
                  }
                }}
                disabled={!exchange.supported}
                className={`p-6 rounded-2xl border text-left transition-all ${
                  exchange.supported
                    ? 'bg-slate-900/50 border-slate-700 hover:border-cyan-500 hover:bg-slate-800/50 cursor-pointer'
                    : 'bg-slate-900/30 border-slate-800 cursor-not-allowed opacity-60'
                } ${selectedExchange === exchange.id ? 'border-cyan-500 bg-slate-800/50' : ''}`}
              >
                <div className="w-12 h-12 bg-slate-800 rounded-xl flex items-center justify-center mb-4">
                  <Wallet className="w-6 h-6 text-cyan-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-1">{exchange.name}</h3>
                <p className="text-sm text-slate-400 mb-3">{exchange.description}</p>
                {exchange.comingSoon ? (
                  <span className="text-xs text-yellow-400 bg-yellow-400/10 px-2 py-1 rounded">Coming Soon</span>
                ) : (
                  <span className="text-xs text-green-400 bg-green-400/10 px-2 py-1 rounded">Available</span>
                )}
              </button>
            ))}
          </motion.div>
        )}

        {/* Step 2: API Credentials */}
        {step === 2 && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="max-w-xl mx-auto"
          >
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 mb-6">
              {/* Info Box - Dynamic based on exchange */}
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4 mb-6">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                  <div className="text-sm">
                    <p className="text-blue-300 font-medium mb-1">How to get your API Keys:</p>
                    {selectedExchange === 'binance' ? (
                      <>
                        <ol className="text-blue-200/80 space-y-1 list-decimal list-inside">
                          <li>Log in to your Binance account</li>
                          <li>Go to API Management in settings</li>
                          <li>Create new API key</li>
                          <li>Enable "Enable Futures" permission</li>
                          <li>Optionally restrict to your IP address</li>
                        </ol>
                        <a 
                          href="https://www.binance.com/en/my/settings/api-management" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-300 mt-2"
                        >
                          Open Binance API Settings <ExternalLink className="w-3 h-3" />
                        </a>
                      </>
                    ) : (
                      <>
                        <ol className="text-blue-200/80 space-y-1 list-decimal list-inside">
                          <li>Log in to your Bybit account</li>
                          <li>Go to API Management in settings</li>
                          <li>Create new API key with "Contract" permissions</li>
                          <li>Enable "Trade" permission only (read is automatic)</li>
                        </ol>
                        <a 
                          href="https://www.bybit.com/app/user/api-management" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-300 mt-2"
                        >
                          Open Bybit API Settings <ExternalLink className="w-3 h-3" />
                        </a>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <form onSubmit={(e) => { e.preventDefault(); setStep(3); }}>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Account Name
                    </label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                      placeholder={selectedExchange === 'binance' ? 'My Binance Account' : 'My Bybit Account'}
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      <Key className="w-4 h-4 inline mr-1" />
                      API Key
                    </label>
                    <input
                      type="text"
                      value={formData.api_key}
                      onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                      className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-white font-mono text-sm placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                      placeholder="Enter your API Key"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      <Lock className="w-4 h-4 inline mr-1" />
                      API Secret
                    </label>
                    <div className="relative">
                      <input
                        type={showSecret ? 'text' : 'password'}
                        value={formData.api_secret}
                        onChange={(e) => setFormData({ ...formData, api_secret: e.target.value })}
                        className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-white font-mono text-sm placeholder-slate-500 focus:border-cyan-500 focus:outline-none pr-12"
                        placeholder="Enter your API Secret"
                        required
                      />
                      <button
                        type="button"
                        onClick={() => setShowSecret(!showSecret)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                      >
                        {showSecret ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                      </button>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 p-4 bg-slate-800/50 rounded-xl">
                    <input
                      type="checkbox"
                      id="testnet"
                      checked={formData.is_testnet}
                      onChange={(e) => setFormData({ ...formData, is_testnet: e.target.checked })}
                      className="w-5 h-5 rounded border-slate-600 bg-slate-700 text-cyan-500 focus:ring-cyan-500"
                    />
                    <label htmlFor="testnet" className="text-sm">
                      <span className="text-white font-medium">Use Testnet</span>
                      <span className="text-slate-400 ml-1">(for testing with fake money)</span>
                    </label>
                  </div>
                </div>

                <div className="flex gap-3 mt-6">
                  <button
                    type="button"
                    onClick={() => setStep(1)}
                    className="px-6 py-3 bg-slate-800 text-white rounded-xl font-semibold hover:bg-slate-700 transition-all"
                  >
                    Back
                  </button>
                  <button
                    type="submit"
                    disabled={!formData.api_key || !formData.api_secret}
                    className="flex-1 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold hover:from-cyan-400 hover:to-blue-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    Continue
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </form>
            </div>

            {/* Security Notice */}
            <div className="flex items-start gap-3 p-4 bg-green-500/10 border border-green-500/30 rounded-xl">
              <Shield className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
              <div className="text-sm">
                <p className="text-green-300 font-medium">Your keys are encrypted</p>
                <p className="text-green-200/70">API credentials are encrypted using AES-256 before storage. We never have access to your funds - only trading permissions.</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Step 3: Verify & Connect */}
        {step === 3 && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="max-w-xl mx-auto"
          >
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Confirm Connection</h2>
              
              <div className="space-y-3 mb-6">
                <div className="flex justify-between p-3 bg-slate-800/50 rounded-xl">
                  <span className="text-slate-400">Exchange</span>
                  <span className="text-white font-medium capitalize">{selectedExchange}</span>
                </div>
                <div className="flex justify-between p-3 bg-slate-800/50 rounded-xl">
                  <span className="text-slate-400">Account Name</span>
                  <span className="text-white font-medium">{formData.name}</span>
                </div>
                <div className="flex justify-between p-3 bg-slate-800/50 rounded-xl">
                  <span className="text-slate-400">API Key</span>
                  <span className="text-white font-mono text-sm">
                    {formData.api_key.slice(0, 6)}...{formData.api_key.slice(-4)}
                  </span>
                </div>
                <div className="flex justify-between p-3 bg-slate-800/50 rounded-xl">
                  <span className="text-slate-400">Mode</span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    formData.is_testnet 
                      ? 'bg-yellow-500/20 text-yellow-400' 
                      : 'bg-green-500/20 text-green-400'
                  }`}>
                    {formData.is_testnet ? 'Testnet' : 'Live Trading'}
                  </span>
                </div>
              </div>

              {error && (
                <div className="mb-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                  <span className="text-red-300 text-sm">{error}</span>
                </div>
              )}

              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => setStep(2)}
                  disabled={isLoading}
                  className="px-6 py-3 bg-slate-800 text-white rounded-xl font-semibold hover:bg-slate-700 transition-all disabled:opacity-50"
                >
                  Back
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={isLoading}
                  className="flex-1 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold hover:from-cyan-400 hover:to-blue-500 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Connecting...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="w-5 h-5" />
                      Connect Exchange
                    </>
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

