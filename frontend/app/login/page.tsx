'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Shield, Eye, EyeOff, ArrowRight, AlertCircle } from 'lucide-react'
import Link from 'next/link'

export default function LoginPage() {
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // In production, call actual API
      // const response = await fetch('/api/auth/login', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(formData)
      // })
      
      window.location.href = '/dashboard'
    } catch (err) {
      setError('Invalid email or password')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex">
      {/* Left Panel - Form */}
      <div className="flex-1 flex flex-col justify-center px-8 lg:px-16 xl:px-24">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-md w-full mx-auto"
        >
          {/* Logo */}
          <Link href="/" className="inline-flex items-center gap-3 mb-12">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-sentinel-accent-cyan to-sentinel-accent-emerald flex items-center justify-center">
              <Shield className="w-7 h-7 text-sentinel-bg-primary" strokeWidth={2.5} />
            </div>
            <span className="font-display font-bold text-2xl">SENTINEL</span>
          </Link>

          {/* Header */}
          <h1 className="text-3xl font-bold mb-2">Welcome back</h1>
          <p className="text-sentinel-text-secondary mb-8">
            Sign in to access your autonomous trading dashboard.
          </p>

          {/* Error Message */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-3 p-4 rounded-xl bg-sentinel-accent-crimson/10 border border-sentinel-accent-crimson/30 mb-6"
            >
              <AlertCircle className="w-5 h-5 text-sentinel-accent-crimson" />
              <span className="text-sentinel-accent-crimson">{error}</span>
            </motion.div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Email Address
              </label>
              <input
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className="w-full px-4 py-3 rounded-xl bg-sentinel-bg-secondary border border-sentinel-border focus:border-sentinel-accent-cyan focus:outline-none focus:ring-1 focus:ring-sentinel-accent-cyan transition-all text-sentinel-text-primary"
                placeholder="you@example.com"
                required
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="w-full px-4 py-3 pr-12 rounded-xl bg-sentinel-bg-secondary border border-sentinel-border focus:border-sentinel-accent-cyan focus:outline-none focus:ring-1 focus:ring-sentinel-accent-cyan transition-all text-sentinel-text-primary"
                  placeholder="Enter your password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-sentinel-text-muted hover:text-sentinel-text-secondary"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Forgot Password */}
            <div className="flex justify-end">
              <Link href="/forgot-password" className="text-sm text-sentinel-accent-cyan hover:underline">
                Forgot password?
              </Link>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary font-bold text-lg flex items-center justify-center gap-3 hover:shadow-glow-cyan transition-all disabled:opacity-50"
            >
              {isLoading ? (
                <div className="w-6 h-6 border-2 border-sentinel-bg-primary border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  Sign In
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </form>

          {/* Register Link */}
          <p className="text-center text-sentinel-text-secondary mt-8">
            Don't have an account?{' '}
            <Link href="/register" className="text-sentinel-accent-cyan hover:underline font-medium">
              Create account
            </Link>
          </p>
        </motion.div>
      </div>

      {/* Right Panel - Visual */}
      <div className="hidden lg:flex flex-1 bg-sentinel-bg-secondary relative overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-20" />
        <div className="absolute top-1/4 right-1/4 w-[500px] h-[500px] bg-glow-cyan opacity-40" />
        <div className="absolute bottom-1/4 left-1/4 w-[400px] h-[400px] bg-glow-emerald opacity-30" />
        
        {/* Content */}
        <div className="relative z-10 flex flex-col justify-center px-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            <h2 className="text-4xl font-display font-bold mb-6 leading-tight">
              AI-Driven
              <br />
              <span className="text-gradient-cyan">Capital Protection</span>
            </h2>
            <p className="text-xl text-sentinel-text-secondary leading-relaxed max-w-md">
              While you rest, SENTINEL AI monitors markets, analyzes trends, 
              and protects your investments 24/7.
            </p>

            {/* Stats */}
            <div className="flex gap-8 mt-12">
              <div>
                <div className="text-3xl font-display font-bold text-sentinel-accent-cyan">94.2%</div>
                <div className="text-sm text-sentinel-text-muted">AI Accuracy</div>
              </div>
              <div>
                <div className="text-3xl font-display font-bold text-sentinel-accent-emerald">24/7</div>
                <div className="text-sm text-sentinel-text-muted">Active Trading</div>
              </div>
              <div>
                <div className="text-3xl font-display font-bold text-sentinel-accent-amber">0</div>
                <div className="text-sm text-sentinel-text-muted">Emotions</div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

