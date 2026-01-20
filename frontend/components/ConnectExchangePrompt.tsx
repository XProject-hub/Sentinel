'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Wallet, 
  ArrowRight, 
  Shield, 
  Zap, 
  Brain, 
  ChevronDown,
  Key,
  Eye,
  EyeOff,
  Lock,
  CheckCircle,
  AlertCircle,
  Loader2,
  Info,
  ExternalLink
} from 'lucide-react'
import { useRouter } from 'next/navigation'

interface Exchange {
  id: string
  name: string
  available: boolean
  description: string
}

// Server IP for API whitelist - configured in environment
const SERVER_IP = process.env.NEXT_PUBLIC_SERVER_IP || '167.235.69.240'

export default function ConnectExchangePrompt() {
  const router = useRouter()
  const [step, setStep] = useState<'select' | 'credentials' | 'verify'>('select')
  const [selectedExchange, setSelectedExchange] = useState<string>('bybit')
  const [showDropdown, setShowDropdown] = useState(false)
  const [showSecret, setShowSecret] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [ipCopied, setIpCopied] = useState(false)
  
  const [formData, setFormData] = useState({
    name: 'My Bybit Account',
    api_key: '',
    api_secret: '',
    is_testnet: false,
  })

  const exchanges: Exchange[] = [
    { id: 'bybit', name: 'Bybit', available: true, description: 'Leading crypto derivatives exchange' },
    { id: 'binance', name: 'Binance', available: false, description: 'Coming soon' },
    { id: 'okx', name: 'OKX', available: false, description: 'Coming soon' },
    { id: 'kucoin', name: 'KuCoin', available: false, description: 'Coming soon' },
  ]

  const selectedExchangeData = exchanges.find(e => e.id === selectedExchange)

  const handleSubmit = async () => {
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
          window.location.reload()
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

  if (success) {
    return (
      <div className="min-h-screen bg-sentinel-bg-primary flex items-center justify-center p-6">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="glass-card border border-sentinel-accent-emerald/30 rounded-2xl p-8 text-center max-w-md"
        >
          <div className="w-20 h-20 bg-sentinel-accent-emerald/20 rounded-full flex items-center justify-center mx-auto mb-6">
            <CheckCircle className="w-10 h-10 text-sentinel-accent-emerald" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Exchange Connected!</h2>
          <p className="text-sentinel-text-secondary mb-4">Your account has been successfully connected.</p>
          <p className="text-sm text-sentinel-text-muted">Redirecting to dashboard...</p>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-sentinel-bg-primary p-6">
      <div className="max-w-2xl mx-auto pt-10">
        
        {/* Step: Select Exchange */}
        {step === 'select' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            {/* Icon */}
            <div className="w-20 h-20 bg-gradient-to-br from-sentinel-accent-cyan to-sentinel-accent-emerald rounded-2xl flex items-center justify-center mx-auto mb-6">
              <Wallet className="w-10 h-10 text-white" />
            </div>
            
            {/* Title */}
            <h1 className="text-3xl font-bold text-white mb-3">Connect Your Exchange</h1>
            <p className="text-sentinel-text-secondary mb-8 max-w-md mx-auto">
              To start AI-powered trading, connect your exchange account. 
              Sentinel will analyze markets and trade on your behalf.
            </p>

            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              <div className="glass-card rounded-xl p-4">
                <Brain className="w-8 h-8 text-sentinel-accent-cyan mx-auto mb-2" />
                <h3 className="text-white font-semibold mb-1">AI Trading</h3>
                <p className="text-sentinel-text-muted text-sm">Automated trades using advanced AI</p>
              </div>
              <div className="glass-card rounded-xl p-4">
                <Shield className="w-8 h-8 text-sentinel-accent-emerald mx-auto mb-2" />
                <h3 className="text-white font-semibold mb-1">Secure</h3>
                <p className="text-sentinel-text-muted text-sm">Encrypted API keys, no withdrawal access</p>
              </div>
              <div className="glass-card rounded-xl p-4">
                <Zap className="w-8 h-8 text-sentinel-accent-amber mx-auto mb-2" />
                <h3 className="text-white font-semibold mb-1">24/7 Active</h3>
                <p className="text-sentinel-text-muted text-sm">Never miss a trading opportunity</p>
              </div>
            </div>

            {/* Exchange Dropdown */}
            <div className="max-w-md mx-auto mb-6">
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2 text-left">
                Select Exchange
              </label>
              <div className="relative">
                <button
                  onClick={() => setShowDropdown(!showDropdown)}
                  className="w-full px-4 py-3 glass-card border border-sentinel-border rounded-xl text-white text-left flex items-center justify-between hover:border-sentinel-accent-cyan transition-colors"
                >
                  <span>{selectedExchangeData?.name || 'Select exchange'}</span>
                  <ChevronDown className={`w-5 h-5 text-sentinel-text-muted transition-transform ${showDropdown ? 'rotate-180' : ''}`} />
                </button>
                
                {showDropdown && (
                  <div className="absolute top-full left-0 right-0 mt-2 glass-card border border-sentinel-border rounded-xl overflow-hidden z-10">
                    {exchanges.map((exchange) => (
                      <button
                        key={exchange.id}
                        onClick={() => {
                          if (exchange.available) {
                            setSelectedExchange(exchange.id)
                            setShowDropdown(false)
                          }
                        }}
                        disabled={!exchange.available}
                        className={`w-full px-4 py-3 text-left flex items-center justify-between ${
                          exchange.available 
                            ? 'hover:bg-sentinel-bg-tertiary cursor-pointer' 
                            : 'opacity-50 cursor-not-allowed'
                        } ${selectedExchange === exchange.id ? 'bg-sentinel-bg-tertiary' : ''}`}
                      >
                        <div>
                          <span className="text-white font-medium">{exchange.name}</span>
                          <span className="text-sentinel-text-muted text-sm ml-2">{exchange.description}</span>
                        </div>
                        {!exchange.available && (
                          <span className="text-xs text-sentinel-accent-amber bg-sentinel-accent-amber/10 px-2 py-1 rounded">Soon</span>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* CTA Button */}
            <button
              onClick={() => setStep('credentials')}
              disabled={!selectedExchangeData?.available}
              className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary rounded-xl font-bold text-lg hover:shadow-glow-cyan transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Connect {selectedExchangeData?.name} Account
              <ArrowRight className="w-5 h-5" />
            </button>

            {/* Note */}
            <p className="text-sentinel-text-muted text-sm mt-6">
              Don't have a Bybit account?{' '}
              <a 
                href="https://www.bybit.com/invite?ref=RG8G6XN" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-sentinel-accent-cyan hover:underline"
              >
                Create one here
              </a>
            </p>
          </motion.div>
        )}

        {/* Step: Enter Credentials */}
        {step === 'credentials' && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <button
              onClick={() => setStep('select')}
              className="text-sentinel-text-secondary hover:text-white mb-6 flex items-center gap-2"
            >
              <ArrowRight className="w-4 h-4 rotate-180" />
              Back to exchange selection
            </button>

            <div className="glass-card border border-sentinel-border rounded-2xl p-6">
              <h2 className="text-xl font-bold text-white mb-4">
                Enter {selectedExchangeData?.name} API Credentials
              </h2>

              {/* Server IP for Whitelist */}
              <div className="bg-sentinel-accent-amber/10 border border-sentinel-accent-amber/30 rounded-xl p-4 mb-4">
                <div className="flex items-start gap-3">
                  <Shield className="w-5 h-5 text-sentinel-accent-amber flex-shrink-0 mt-0.5" />
                  <div className="text-sm w-full">
                    <p className="text-sentinel-accent-amber font-medium mb-2">IP Whitelist Required</p>
                    <p className="text-sentinel-text-secondary mb-2">
                      Add this IP address to your Bybit API whitelist:
                    </p>
                    <div className="flex items-center gap-2 bg-sentinel-bg-primary rounded-lg p-3">
                      <code className="text-white font-mono text-lg flex-1">{SERVER_IP}</code>
                      <button
                        type="button"
                        onClick={() => {
                          navigator.clipboard.writeText(SERVER_IP)
                          setIpCopied(true)
                          setTimeout(() => setIpCopied(false), 2000)
                        }}
                        className="px-3 py-1.5 bg-sentinel-accent-amber/20 text-sentinel-accent-amber rounded-lg text-xs font-medium hover:bg-sentinel-accent-amber/30 transition-colors flex items-center gap-1"
                      >
                        {ipCopied ? (
                          <>
                            <CheckCircle className="w-3 h-3" />
                            Copied
                          </>
                        ) : (
                          'Copy'
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Info Box */}
              <div className="bg-sentinel-accent-cyan/10 border border-sentinel-accent-cyan/30 rounded-xl p-4 mb-6">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-sentinel-accent-cyan flex-shrink-0 mt-0.5" />
                  <div className="text-sm">
                    <p className="text-sentinel-accent-cyan font-medium mb-1">How to get your API Keys:</p>
                    <ol className="text-sentinel-text-secondary space-y-1 list-decimal list-inside">
                      <li>Log in to your Bybit account</li>
                      <li>Go to API Management in settings</li>
                      <li>Create new API key with "Contract" permissions</li>
                      <li>Enable "Trade" permission only (read is automatic)</li>
                      <li className="text-sentinel-accent-amber">Add the IP address above to whitelist</li>
                    </ol>
                    <a 
                      href="https://www.bybit.com/app/user/api-management" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-sentinel-accent-cyan hover:underline mt-2"
                    >
                      Open Bybit API Settings <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    Account Name
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full px-4 py-3 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white placeholder-sentinel-text-muted focus:border-sentinel-accent-cyan focus:outline-none"
                    placeholder="My Bybit Account"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    <Key className="w-4 h-4 inline mr-1" />
                    API Key
                  </label>
                  <input
                    type="text"
                    value={formData.api_key}
                    onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                    className="w-full px-4 py-3 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white font-mono text-sm placeholder-sentinel-text-muted focus:border-sentinel-accent-cyan focus:outline-none"
                    placeholder="Enter your API Key"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                    <Lock className="w-4 h-4 inline mr-1" />
                    API Secret
                  </label>
                  <div className="relative">
                    <input
                      type={showSecret ? 'text' : 'password'}
                      value={formData.api_secret}
                      onChange={(e) => setFormData({ ...formData, api_secret: e.target.value })}
                      className="w-full px-4 py-3 bg-sentinel-bg-tertiary border border-sentinel-border rounded-xl text-white font-mono text-sm placeholder-sentinel-text-muted focus:border-sentinel-accent-cyan focus:outline-none pr-12"
                      placeholder="Enter your API Secret"
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowSecret(!showSecret)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-sentinel-text-muted hover:text-white"
                    >
                      {showSecret ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-4 bg-sentinel-bg-tertiary rounded-xl">
                  <input
                    type="checkbox"
                    id="testnet"
                    checked={formData.is_testnet}
                    onChange={(e) => setFormData({ ...formData, is_testnet: e.target.checked })}
                    className="w-5 h-5 rounded border-sentinel-border bg-sentinel-bg-secondary text-sentinel-accent-cyan focus:ring-sentinel-accent-cyan"
                  />
                  <label htmlFor="testnet" className="text-sm">
                    <span className="text-white font-medium">Use Testnet</span>
                    <span className="text-sentinel-text-muted ml-1">(for testing with fake money)</span>
                  </label>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setStep('select')}
                  className="px-6 py-3 glass-card border border-sentinel-border text-white rounded-xl font-semibold hover:bg-sentinel-bg-tertiary transition-all"
                >
                  Back
                </button>
                <button
                  onClick={() => setStep('verify')}
                  disabled={!formData.api_key || !formData.api_secret}
                  className="flex-1 py-3 bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary rounded-xl font-bold hover:shadow-glow-cyan transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  Continue
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Security Notice */}
            <div className="mt-6 flex items-start gap-3 p-4 bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30 rounded-xl">
              <Shield className="w-5 h-5 text-sentinel-accent-emerald flex-shrink-0 mt-0.5" />
              <div className="text-sm">
                <p className="text-sentinel-accent-emerald font-medium">Your keys are encrypted</p>
                <p className="text-sentinel-text-secondary">API credentials are encrypted using AES-256 before storage. We never have access to your funds - only trading permissions.</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Step: Verify & Connect */}
        {step === 'verify' && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <button
              onClick={() => setStep('credentials')}
              className="text-sentinel-text-secondary hover:text-white mb-6 flex items-center gap-2"
            >
              <ArrowRight className="w-4 h-4 rotate-180" />
              Back to credentials
            </button>

            <div className="glass-card border border-sentinel-border rounded-2xl p-6">
              <h2 className="text-xl font-bold text-white mb-4">Confirm Connection</h2>
              
              <div className="space-y-3 mb-6">
                <div className="flex justify-between p-3 bg-sentinel-bg-tertiary rounded-xl">
                  <span className="text-sentinel-text-muted">Exchange</span>
                  <span className="text-white font-medium">{selectedExchangeData?.name}</span>
                </div>
                <div className="flex justify-between p-3 bg-sentinel-bg-tertiary rounded-xl">
                  <span className="text-sentinel-text-muted">Account Name</span>
                  <span className="text-white font-medium">{formData.name}</span>
                </div>
                <div className="flex justify-between p-3 bg-sentinel-bg-tertiary rounded-xl">
                  <span className="text-sentinel-text-muted">API Key</span>
                  <span className="text-white font-mono text-sm">
                    {formData.api_key.slice(0, 6)}...{formData.api_key.slice(-4)}
                  </span>
                </div>
                <div className="flex justify-between p-3 bg-sentinel-bg-tertiary rounded-xl">
                  <span className="text-sentinel-text-muted">Mode</span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    formData.is_testnet 
                      ? 'bg-sentinel-accent-amber/20 text-sentinel-accent-amber' 
                      : 'bg-sentinel-accent-emerald/20 text-sentinel-accent-emerald'
                  }`}>
                    {formData.is_testnet ? 'Testnet' : 'Live Trading'}
                  </span>
                </div>
              </div>

              {error && (
                <div className="mb-4 p-4 bg-sentinel-accent-crimson/10 border border-sentinel-accent-crimson/30 rounded-xl flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-sentinel-accent-crimson flex-shrink-0" />
                  <span className="text-sentinel-accent-crimson text-sm">{error}</span>
                </div>
              )}

              <div className="flex gap-3">
                <button
                  onClick={() => setStep('credentials')}
                  disabled={isLoading}
                  className="px-6 py-3 glass-card border border-sentinel-border text-white rounded-xl font-semibold hover:bg-sentinel-bg-tertiary transition-all disabled:opacity-50"
                >
                  Back
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={isLoading}
                  className="flex-1 py-3 bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary rounded-xl font-bold hover:shadow-glow-cyan transition-all disabled:opacity-50 flex items-center justify-center gap-2"
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
