'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  ArrowRight, 
  Shield, 
  Zap, 
  Key,
  Eye,
  EyeOff,
  Lock,
  CheckCircle,
  AlertCircle,
  Loader2,
  Info,
  ExternalLink,
  Copy,
  LogOut
} from 'lucide-react'
import { useRouter } from 'next/navigation'

const SERVER_IP = process.env.NEXT_PUBLIC_SERVER_IP || '167.235.69.240'

export default function ConnectExchangePrompt() {
  const router = useRouter()
  const [step, setStep] = useState<'info' | 'credentials'>('info')
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

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('sentinel_user')
    window.location.href = '/login'
  }

  const copyIP = async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(SERVER_IP)
      } else {
        const textArea = document.createElement('textarea')
        textArea.value = SERVER_IP
        textArea.style.position = 'fixed'
        textArea.style.left = '-999999px'
        document.body.appendChild(textArea)
        textArea.select()
        document.execCommand('copy')
        document.body.removeChild(textArea)
      }
      setIpCopied(true)
      setTimeout(() => setIpCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const handleConnect = async () => {
    if (!formData.api_key || !formData.api_secret) {
      setError('Please enter both API Key and Secret')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const token = localStorage.getItem('token')
      
      const response = await fetch('/api/exchanges', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          ...formData,
          exchange: 'bybit'
        })
      })

      const data = await response.json()

      if (response.ok) {
        setSuccess(true)
        localStorage.setItem('sentinel_api_creds', JSON.stringify({
          apiKey: formData.api_key,
          apiSecret: formData.api_secret
        }))
        setTimeout(() => router.push('/dashboard'), 2000)
      } else {
        setError(data.message || data.error || 'Failed to connect exchange')
      }
    } catch (err) {
      setError('Connection failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  if (success) {
    return (
      <div className="min-h-screen bg-[#0a0f1a] flex items-center justify-center p-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center mx-auto mb-8">
            <CheckCircle className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-4">Connected!</h1>
          <p className="text-gray-400">Redirecting to dashboard...</p>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#0a0f1a] flex items-center justify-center p-6">
      <button
        onClick={handleLogout}
        className="fixed top-6 right-6 p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
        title="Logout"
      >
        <LogOut className="w-5 h-5 text-gray-400" />
      </button>
      
      <div className="w-full max-w-xl">
        {/* Info Step */}
        {step === 'info' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center mx-auto mb-8">
              <Key className="w-10 h-10 text-white" />
            </div>
            
            <h1 className="text-3xl font-bold text-white mb-4">Connect Your Exchange</h1>
            <p className="text-gray-400 mb-8">
              Link your Bybit account to start autonomous AI trading
            </p>

            {/* Security Features */}
            <div className="grid grid-cols-3 gap-4 mb-8">
              {[
                { icon: Shield, label: 'Encrypted' },
                { icon: Lock, label: 'Read-Only Safe' },
                { icon: Zap, label: 'Instant Setup' }
              ].map((item, i) => (
                <div key={i} className="p-4 bg-white/5 rounded-xl border border-white/10">
                  <item.icon className="w-6 h-6 text-cyan-400 mx-auto mb-2" />
                  <span className="text-xs text-gray-400">{item.label}</span>
                </div>
              ))}
            </div>

            {/* IP Whitelist Notice */}
            <div className="p-4 bg-amber-500/5 border border-amber-500/20 rounded-xl mb-8 text-left">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-gray-300 mb-2">
                    Add this IP to your Bybit API whitelist:
                  </p>
                  <div className="flex items-center gap-2">
                    <code className="px-3 py-1.5 bg-black/30 rounded-lg text-amber-400 font-mono text-sm">
                      {SERVER_IP}
                    </code>
                    <button
                      onClick={copyIP}
                      className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                    >
                      {ipCopied ? (
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-400" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={() => setStep('credentials')}
              className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold hover:shadow-lg hover:shadow-cyan-500/25 transition-all flex items-center justify-center gap-2"
            >
              Continue
              <ArrowRight className="w-5 h-5" />
            </button>

            <a
              href="https://www.bybit.com/invite?ref=RG8G6XN"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 mt-6 text-sm text-gray-500 hover:text-gray-400 transition-colors"
            >
              Don't have a Bybit account?
              <ExternalLink className="w-3 h-3" />
            </a>
          </motion.div>
        )}

        {/* Credentials Step */}
        {step === 'credentials' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <button
              onClick={() => setStep('info')}
              className="mb-8 text-gray-400 hover:text-white text-sm flex items-center gap-1"
            >
              ← Back
            </button>

            <h1 className="text-2xl font-bold text-white mb-2">Enter API Credentials</h1>
            <p className="text-gray-400 mb-8">
              Your keys are encrypted and only used for trading
            </p>

            {error && (
              <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <span className="text-red-400 text-sm">{error}</span>
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  API Key
                </label>
                <input
                  type="text"
                  value={formData.api_key}
                  onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                  className="w-full px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 focus:outline-none font-mono text-sm"
                  placeholder="Enter your API key"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  API Secret
                </label>
                <div className="relative">
                  <input
                    type={showSecret ? 'text' : 'password'}
                    value={formData.api_secret}
                    onChange={(e) => setFormData({ ...formData, api_secret: e.target.value })}
                    className="w-full px-4 py-3.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 focus:outline-none font-mono text-sm pr-12"
                    placeholder="Enter your API secret"
                  />
                  <button
                    type="button"
                    onClick={() => setShowSecret(!showSecret)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                  >
                    {showSecret ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-3 p-4 bg-white/[0.02] rounded-xl border border-white/10">
                <input
                  type="checkbox"
                  id="testnet"
                  checked={formData.is_testnet}
                  onChange={(e) => setFormData({ ...formData, is_testnet: e.target.checked })}
                  className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-cyan-500 focus:ring-cyan-500"
                />
                <label htmlFor="testnet" className="text-sm text-gray-400">
                  Use Testnet (for testing with fake money)
                </label>
              </div>

              <button
                onClick={handleConnect}
                disabled={isLoading}
                className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold hover:shadow-lg hover:shadow-cyan-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <Shield className="w-5 h-5" />
                    Connect Exchange
                  </>
                )}
              </button>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
