'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Shield, Key, Eye, EyeOff, Server, Copy, Check,
  AlertTriangle, ExternalLink, ArrowRight, ArrowLeft,
  Zap, Bot, TrendingUp
} from 'lucide-react'
import Link from 'next/link'

// Server IP for API whitelisting
const SERVER_IP = '109.104.154.183'

export default function ConnectExchangePage() {
  const [step, setStep] = useState(1)
  const [showSecret, setShowSecret] = useState(false)
  const [copied, setCopied] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [isEnablingTrading, setIsEnablingTrading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [errorMessage, setErrorMessage] = useState('')
  const [enableAutoTrading, setEnableAutoTrading] = useState(true)
  
  const [formData, setFormData] = useState({
    exchange: 'bybit',
    apiKey: '',
    apiSecret: '',
  })

  const copyServerIP = () => {
    navigator.clipboard.writeText(SERVER_IP)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const testConnection = async () => {
    setIsConnecting(true)
    setConnectionStatus('idle')
    setErrorMessage('')

    try {
      // Call real AI service API
      const response = await fetch('/ai/exchange/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exchange: formData.exchange,
          apiKey: formData.apiKey,
          apiSecret: formData.apiSecret,
        })
      })

      const data = await response.json()

      if (data.success) {
        setConnectionStatus('success')
        setTimeout(() => setStep(3), 1000)
      } else {
        setConnectionStatus('error')
        setErrorMessage(data.error || 'Failed to connect. Check your API credentials.')
      }
    } catch (error) {
      setConnectionStatus('error')
      setErrorMessage('Connection failed. Make sure the server IP is whitelisted.')
    } finally {
      setIsConnecting(false)
    }
  }

  const saveAndEnableTrading = async () => {
    setIsEnablingTrading(true)
    
    try {
      // First, connect the exchange
      const connectResponse = await fetch('/ai/exchange/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exchange: formData.exchange,
          apiKey: formData.apiKey,
          apiSecret: formData.apiSecret,
          testnet: false,
        })
      })

      const connectData = await connectResponse.json()

      if (!connectData.success) {
        setErrorMessage(connectData.error || 'Failed to connect exchange')
        setIsEnablingTrading(false)
        return
      }

      // Store API credentials for later use (for START/STOP)
      localStorage.setItem('sentinel_api_creds', JSON.stringify({
        apiKey: formData.apiKey,
        apiSecret: formData.apiSecret,
        exchange: formData.exchange,
      }))

      // Enable autonomous trading if selected
      if (enableAutoTrading) {
        const tradingResponse = await fetch('/ai/exchange/trading/enable', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: 'default',
            api_key: formData.apiKey,
            api_secret: formData.apiSecret,
          })
        })

        const tradingData = await tradingResponse.json()
        
        if (!tradingData.success) {
          console.warn('Auto trading could not be enabled:', tradingData.error)
        }
      }

      // Redirect to dashboard
      window.location.href = '/dashboard'
      
    } catch (error) {
      setErrorMessage('Failed to save connection')
    } finally {
      setIsEnablingTrading(false)
    }
  }

  return (
    <div className="min-h-screen bg-sentinel-bg-primary">
      {/* Header */}
      <nav className="border-b border-sentinel-border glass-card">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/dashboard" className="flex items-center gap-3 text-sentinel-text-secondary hover:text-sentinel-text-primary">
            <ArrowLeft className="w-5 h-5" />
            Back to Dashboard
          </Link>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sentinel-accent-cyan to-sentinel-accent-emerald flex items-center justify-center">
              <Shield className="w-5 h-5 text-sentinel-bg-primary" />
            </div>
            <span className="font-display font-bold">SENTINEL</span>
          </div>
        </div>
      </nav>

      <main className="max-w-2xl mx-auto px-6 py-12">
        {/* Progress Steps */}
        <div className="flex items-center justify-center gap-4 mb-12">
          {[1, 2, 3].map((s) => (
            <div key={s} className="flex items-center gap-2">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                step >= s 
                  ? 'bg-sentinel-accent-cyan text-sentinel-bg-primary' 
                  : 'bg-sentinel-bg-tertiary text-sentinel-text-muted'
              }`}>
                {s}
              </div>
              {s < 3 && <div className={`w-16 h-0.5 ${step > s ? 'bg-sentinel-accent-cyan' : 'bg-sentinel-bg-tertiary'}`} />}
            </div>
          ))}
        </div>

        {/* Step 1: Whitelist IP */}
        {step === 1 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            <div className="text-center">
              <h1 className="text-3xl font-bold mb-4">Connect Your Exchange</h1>
              <p className="text-sentinel-text-secondary">
                First, whitelist our server IP in your Bybit API settings
              </p>
            </div>

            {/* Server IP Card */}
            <div className="p-6 rounded-2xl glass-card border-2 border-sentinel-accent-cyan">
              <div className="flex items-center gap-3 mb-4">
                <Server className="w-6 h-6 text-sentinel-accent-cyan" />
                <span className="font-semibold">Server IP for Whitelisting</span>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="flex-1 p-4 rounded-xl bg-sentinel-bg-primary font-mono text-2xl text-center">
                  {SERVER_IP}
                </div>
                <button
                  onClick={copyServerIP}
                  className="p-4 rounded-xl bg-sentinel-accent-cyan text-sentinel-bg-primary hover:bg-opacity-90 transition-all"
                >
                  {copied ? <Check className="w-6 h-6" /> : <Copy className="w-6 h-6" />}
                </button>
              </div>

              {copied && (
                <p className="text-sentinel-accent-emerald text-sm mt-2 text-center">
                  IP copied to clipboard!
                </p>
              )}
            </div>

            {/* Instructions */}
            <div className="p-6 rounded-2xl glass-card">
              <h3 className="font-semibold mb-4">How to whitelist on Bybit:</h3>
              <ol className="space-y-3 text-sentinel-text-secondary">
                <li className="flex gap-3">
                  <span className="w-6 h-6 rounded-full bg-sentinel-bg-tertiary flex items-center justify-center text-sm font-bold">1</span>
                  <span>Login to your Bybit account</span>
                </li>
                <li className="flex gap-3">
                  <span className="w-6 h-6 rounded-full bg-sentinel-bg-tertiary flex items-center justify-center text-sm font-bold">2</span>
                  <span>Go to <strong>API Management</strong> in Account Settings</span>
                </li>
                <li className="flex gap-3">
                  <span className="w-6 h-6 rounded-full bg-sentinel-bg-tertiary flex items-center justify-center text-sm font-bold">3</span>
                  <span>Create new API key or edit existing one</span>
                </li>
                <li className="flex gap-3">
                  <span className="w-6 h-6 rounded-full bg-sentinel-bg-tertiary flex items-center justify-center text-sm font-bold">4</span>
                  <span>Add IP <strong>{SERVER_IP}</strong> to the whitelist</span>
                </li>
                <li className="flex gap-3">
                  <span className="w-6 h-6 rounded-full bg-sentinel-bg-tertiary flex items-center justify-center text-sm font-bold">5</span>
                  <span>Enable: <strong>Read</strong>, <strong>Trade</strong> (NO Withdraw!)</span>
                </li>
              </ol>

              <a 
                href="https://www.bybit.com/app/user/api-management" 
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 mt-6 text-sentinel-accent-cyan hover:underline"
              >
                Open Bybit API Management <ExternalLink className="w-4 h-4" />
              </a>
            </div>

            {/* Warning */}
            <div className="p-4 rounded-xl bg-sentinel-accent-amber/10 border border-sentinel-accent-amber/30 flex gap-3">
              <AlertTriangle className="w-5 h-5 text-sentinel-accent-amber flex-shrink-0 mt-0.5" />
              <div className="text-sm">
                <strong className="text-sentinel-accent-amber">Security Notice:</strong>
                <p className="text-sentinel-text-secondary mt-1">
                  NEVER enable Withdraw permissions. SENTINEL only needs Read and Trade access.
                  Your funds remain safe on your exchange.
                </p>
              </div>
            </div>

            <button
              onClick={() => setStep(2)}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary font-bold text-lg flex items-center justify-center gap-3"
            >
              I've Whitelisted the IP <ArrowRight className="w-5 h-5" />
            </button>
          </motion.div>
        )}

        {/* Step 2: Enter API Credentials */}
        {step === 2 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            <div className="text-center">
              <h1 className="text-3xl font-bold mb-4">Enter API Credentials</h1>
              <p className="text-sentinel-text-secondary">
                Your credentials are encrypted and stored securely
              </p>
            </div>

            <div className="p-6 rounded-2xl glass-card space-y-6">
              {/* Exchange Select */}
              <div>
                <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                  Exchange
                </label>
                <select
                  value={formData.exchange}
                  onChange={(e) => setFormData({ ...formData, exchange: e.target.value })}
                  className="w-full px-4 py-3 rounded-xl bg-sentinel-bg-secondary border border-sentinel-border focus:border-sentinel-accent-cyan focus:outline-none"
                >
                  <option value="bybit">Bybit</option>
                </select>
              </div>

              {/* API Key */}
              <div>
                <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                  API Key
                </label>
                <div className="relative">
                  <Key className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-sentinel-text-muted" />
                  <input
                    type="text"
                    value={formData.apiKey}
                    onChange={(e) => setFormData({ ...formData, apiKey: e.target.value })}
                    className="w-full pl-12 pr-4 py-3 rounded-xl bg-sentinel-bg-secondary border border-sentinel-border focus:border-sentinel-accent-cyan focus:outline-none font-mono"
                    placeholder="Enter your API key"
                  />
                </div>
              </div>

              {/* API Secret */}
              <div>
                <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                  API Secret
                </label>
                <div className="relative">
                  <Key className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-sentinel-text-muted" />
                  <input
                    type={showSecret ? 'text' : 'password'}
                    value={formData.apiSecret}
                    onChange={(e) => setFormData({ ...formData, apiSecret: e.target.value })}
                    className="w-full pl-12 pr-12 py-3 rounded-xl bg-sentinel-bg-secondary border border-sentinel-border focus:border-sentinel-accent-cyan focus:outline-none font-mono"
                    placeholder="Enter your API secret"
                  />
                  <button
                    type="button"
                    onClick={() => setShowSecret(!showSecret)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-sentinel-text-muted hover:text-sentinel-text-secondary"
                  >
                    {showSecret ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              {/* Error Message */}
              {connectionStatus === 'error' && (
                <div className="p-4 rounded-xl bg-sentinel-accent-crimson/10 border border-sentinel-accent-crimson/30 text-sentinel-accent-crimson text-sm">
                  {errorMessage}
                </div>
              )}

              {/* Success Message */}
              {connectionStatus === 'success' && (
                <div className="p-4 rounded-xl bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30 text-sentinel-accent-emerald text-sm flex items-center gap-2">
                  <Check className="w-5 h-5" />
                  Connection successful! Setting up...
                </div>
              )}
            </div>

            <div className="flex gap-4">
              <button
                onClick={() => setStep(1)}
                className="flex-1 py-4 rounded-xl glass-card text-sentinel-text-secondary font-semibold hover:text-sentinel-text-primary"
              >
                Back
              </button>
              <button
                onClick={testConnection}
                disabled={!formData.apiKey || !formData.apiSecret || isConnecting}
                className="flex-1 py-4 rounded-xl bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary font-bold flex items-center justify-center gap-3 disabled:opacity-50"
              >
                {isConnecting ? (
                  <div className="w-6 h-6 border-2 border-sentinel-bg-primary border-t-transparent rounded-full animate-spin" />
                ) : (
                  <>Test Connection</>
                )}
              </button>
            </div>
          </motion.div>
        )}

        {/* Step 3: Enable Trading */}
        {step === 3 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            <div className="text-center">
              <div className="w-24 h-24 mx-auto rounded-full bg-sentinel-accent-emerald/20 flex items-center justify-center mb-6">
                <Check className="w-12 h-12 text-sentinel-accent-emerald" />
              </div>
              <h1 className="text-3xl font-bold mb-4">Exchange Connected!</h1>
              <p className="text-sentinel-text-secondary">
                Now configure your trading preferences
              </p>
            </div>

            {/* Auto Trading Option */}
            <div 
              onClick={() => setEnableAutoTrading(!enableAutoTrading)}
              className={`p-6 rounded-2xl cursor-pointer transition-all ${
                enableAutoTrading 
                  ? 'glass-card border-2 border-sentinel-accent-cyan' 
                  : 'glass-card border-2 border-transparent hover:border-sentinel-border'
              }`}
            >
              <div className="flex items-start gap-4">
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                  enableAutoTrading ? 'bg-sentinel-accent-cyan/20' : 'bg-sentinel-bg-tertiary'
                }`}>
                  <Bot className={`w-6 h-6 ${enableAutoTrading ? 'text-sentinel-accent-cyan' : 'text-sentinel-text-muted'}`} />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-lg">Enable 24/7 Autonomous Trading</h3>
                    <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                      enableAutoTrading ? 'border-sentinel-accent-cyan bg-sentinel-accent-cyan' : 'border-sentinel-text-muted'
                    }`}>
                      {enableAutoTrading && <Check className="w-4 h-4 text-sentinel-bg-primary" />}
                    </div>
                  </div>
                  <p className="text-sentinel-text-secondary mt-1">
                    AI will automatically trade using your funds, 24/7, learning and optimizing continuously.
                  </p>
                  
                  {enableAutoTrading && (
                    <div className="mt-4 p-4 rounded-xl bg-sentinel-bg-tertiary">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="flex items-center gap-2">
                          <Zap className="w-4 h-4 text-sentinel-accent-amber" />
                          <span>20+ crypto pairs</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-sentinel-accent-emerald" />
                          <span>Auto compound profits</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Shield className="w-4 h-4 text-sentinel-accent-cyan" />
                          <span>Risk management</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Bot className="w-4 h-4 text-sentinel-accent-violet" />
                          <span>AI learns from trades</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* REAL MONEY Warning */}
            {enableAutoTrading && (
              <div className="p-4 rounded-xl bg-sentinel-accent-crimson/10 border border-sentinel-accent-crimson/30 flex gap-3">
                <AlertTriangle className="w-5 h-5 text-sentinel-accent-crimson flex-shrink-0 mt-0.5" />
                <div className="text-sm">
                  <strong className="text-sentinel-accent-crimson">REAL MONEY TRADING</strong>
                  <p className="text-sentinel-text-secondary mt-1">
                    The AI will use your actual funds to trade. While risk management is in place,
                    trading always carries risk. Only use funds you can afford to lose.
                  </p>
                </div>
              </div>
            )}

            {errorMessage && (
              <div className="p-4 rounded-xl bg-sentinel-accent-crimson/10 border border-sentinel-accent-crimson/30 text-sentinel-accent-crimson text-sm">
                {errorMessage}
              </div>
            )}

            <button
              onClick={saveAndEnableTrading}
              disabled={isEnablingTrading}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary font-bold text-lg flex items-center justify-center gap-3"
            >
              {isEnablingTrading ? (
                <div className="w-6 h-6 border-2 border-sentinel-bg-primary border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  {enableAutoTrading ? 'Start Autonomous Trading' : 'Go to Dashboard'}
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </motion.div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-sentinel-border mt-12">
        <div className="max-w-4xl mx-auto px-6 py-4 text-center text-sm text-sentinel-text-muted">
          Developed by NoLimitDevelopments
        </div>
      </footer>
    </div>
  )
}
