'use client'

import { motion } from 'framer-motion'
import { Wallet, ArrowRight, Shield, Zap, Brain } from 'lucide-react'
import Link from 'next/link'

export default function ConnectExchangePrompt() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-2xl w-full"
      >
        {/* Main Card */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 text-center">
          {/* Icon */}
          <div className="w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <Wallet className="w-10 h-10 text-white" />
          </div>
          
          {/* Title */}
          <h1 className="text-3xl font-bold text-white mb-3">
            Connect Your Exchange
          </h1>
          <p className="text-slate-400 mb-8 max-w-md mx-auto">
            To start AI-powered trading, connect your Bybit account. 
            Sentinel will analyze markets and trade on your behalf.
          </p>

          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-slate-800/50 rounded-xl p-4">
              <Brain className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
              <h3 className="text-white font-semibold mb-1">AI Trading</h3>
              <p className="text-slate-400 text-sm">Automated trades using advanced AI</p>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4">
              <Shield className="w-8 h-8 text-green-400 mx-auto mb-2" />
              <h3 className="text-white font-semibold mb-1">Secure</h3>
              <p className="text-slate-400 text-sm">Encrypted API keys, no withdrawal access</p>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4">
              <Zap className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
              <h3 className="text-white font-semibold mb-1">24/7 Active</h3>
              <p className="text-slate-400 text-sm">Never miss a trading opportunity</p>
            </div>
          </div>

          {/* CTA Button */}
          <Link
            href="/dashboard/connect-exchange"
            className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold hover:from-cyan-400 hover:to-blue-500 transition-all text-lg"
          >
            Connect Bybit Account
            <ArrowRight className="w-5 h-5" />
          </Link>

          {/* Note */}
          <p className="text-slate-500 text-sm mt-6">
            Don't have a Bybit account?{' '}
            <a 
              href="https://www.bybit.com/register" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-cyan-400 hover:underline"
            >
              Create one here
            </a>
          </p>
        </div>

        {/* Info Box */}
        <div className="mt-6 bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
          <p className="text-blue-300 text-sm">
            <strong>Your API keys are safe:</strong> We only request trading permissions. 
            Withdrawal is never enabled, and all credentials are encrypted.
          </p>
        </div>
      </motion.div>
    </div>
  )
}

