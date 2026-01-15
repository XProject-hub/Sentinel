'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Shield, Key, Eye, EyeOff, Server, Copy, Check,
  AlertTriangle, ExternalLink, ArrowRight, ArrowLeft
} from 'lucide-react'
import Link from 'next/link'

// Server IP for API whitelisting
const SERVER_IP = '109.104.154.183'

export default function ConnectExchangePage() {
  const [step, setStep] = useState(1)
  const [showSecret, setShowSecret] = useState(false)
  const [copied, setCopied] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [errorMessage, setErrorMessage] = useState('')
  
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
      const response = await fetch('/api/exchanges/test', {
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
        setTimeout(() => setStep(3), 1500)
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

  const saveConnection = async () => {
    setIsConnecting(true)
    
    try {
      const response = await fetch('/api/exchanges', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })

      if (response.ok) {
        window.location.href = '/dashboard'
      }
    } catch (error) {
      setErrorMessage('Failed to save connection')
    } finally {
      setIsConnecting(false)
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
                  <option value="binance">Binance (Coming Soon)</option>
                  <option value="okx">OKX (Coming Soon)</option>
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
                  Connection successful! Redirecting...
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

        {/* Step 3: Success */}
        {step === 3 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8 text-center"
          >
            <div className="w-24 h-24 mx-auto rounded-full bg-sentinel-accent-emerald/20 flex items-center justify-center">
              <Check className="w-12 h-12 text-sentinel-accent-emerald" />
            </div>

            <div>
              <h1 className="text-3xl font-bold mb-4">Exchange Connected!</h1>
              <p className="text-sentinel-text-secondary">
                Your Bybit account is now connected to SENTINEL AI.
                The AI will start analyzing markets and executing trades.
              </p>
            </div>

            <div className="p-6 rounded-2xl glass-card text-left">
              <h3 className="font-semibold mb-4">What happens next:</h3>
              <ul className="space-y-3 text-sentinel-text-secondary">
                <li className="flex gap-3">
                  <Check className="w-5 h-5 text-sentinel-accent-emerald flex-shrink-0" />
                  <span>AI starts monitoring your connected exchange</span>
                </li>
                <li className="flex gap-3">
                  <Check className="w-5 h-5 text-sentinel-accent-emerald flex-shrink-0" />
                  <span>Market analysis and sentiment tracking begins</span>
                </li>
                <li className="flex gap-3">
                  <Check className="w-5 h-5 text-sentinel-accent-emerald flex-shrink-0" />
                  <span>Risk management rules are applied automatically</span>
                </li>
                <li className="flex gap-3">
                  <Check className="w-5 h-5 text-sentinel-accent-emerald flex-shrink-0" />
                  <span>All trades and profits shown in real-time on dashboard</span>
                </li>
              </ul>
            </div>

            <button
              onClick={saveConnection}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary font-bold text-lg"
            >
              Go to Dashboard
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

