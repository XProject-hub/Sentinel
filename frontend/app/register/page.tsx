'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Eye, EyeOff, ArrowRight, Check, AlertCircle } from 'lucide-react'
import Link from 'next/link'
import Image from 'next/image'

export default function RegisterPage() {
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    // Validate passwords match
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      setIsLoading(false)
      return
    }

    // Validate password strength
    if (passwordStrength() < 2) {
      setError('Password is too weak. Use at least 8 characters with uppercase and numbers.')
      setIsLoading(false)
      return
    }

    try {
      // Call real backend API
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          password: formData.password,
          password_confirmation: formData.confirmPassword,
        })
      })

      const data = await response.json()

      if (data.success) {
        // Store user session
        localStorage.setItem('sentinel_user', JSON.stringify({
          ...data.data.user,
          token: data.data.token,
          isAdmin: false
        }))
        window.location.href = '/dashboard'
      } else {
        // Handle validation errors
        if (data.errors) {
          const firstError = Object.values(data.errors)[0]
          setError(Array.isArray(firstError) ? firstError[0] : String(firstError))
        } else {
          setError(data.message || 'Registration failed. Please try again.')
        }
      }
    } catch (err) {
      setError('Unable to connect to server. Please try again later.')
    } finally {
      setIsLoading(false)
    }
  }

  const passwordStrength = () => {
    const password = formData.password
    let strength = 0
    if (password.length >= 8) strength++
    if (/[A-Z]/.test(password)) strength++
    if (/[0-9]/.test(password)) strength++
    if (/[^A-Za-z0-9]/.test(password)) strength++
    return strength
  }

  const strengthColors = ['bg-sentinel-accent-crimson', 'bg-sentinel-accent-amber', 'bg-sentinel-accent-amber', 'bg-sentinel-accent-emerald']
  const strengthLabels = ['Weak', 'Fair', 'Good', 'Strong']

  return (
    <div className="min-h-screen flex">
      {/* Left Panel - Visual */}
      <div className="hidden lg:flex flex-1 bg-sentinel-bg-secondary relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-20" />
        <div className="absolute top-1/3 left-1/4 w-[500px] h-[500px] bg-glow-cyan opacity-40" />
        <div className="absolute bottom-1/3 right-1/4 w-[400px] h-[400px] bg-glow-emerald opacity-30" />
        
        <div className="relative z-10 flex flex-col justify-center px-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            <h2 className="text-4xl font-display font-bold mb-6 leading-tight">
              Your Autonomous
              <br />
              <span className="text-gradient-emerald">Digital Trader</span>
            </h2>
            <p className="text-xl text-sentinel-text-secondary leading-relaxed max-w-md mb-12">
              Join thousands of traders who trust SENTINEL AI to protect and grow their capital.
            </p>

            {/* Features */}
            <div className="space-y-4">
              {[
                'Real-time market intelligence',
                'AI-powered strategy selection',
                'Automated risk management',
                'Capital protection system',
              ].map((feature, idx) => (
                <motion.div
                  key={feature}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + idx * 0.1 }}
                  className="flex items-center gap-3"
                >
                  <div className="w-6 h-6 rounded-full bg-sentinel-accent-emerald/20 flex items-center justify-center">
                    <Check className="w-4 h-4 text-sentinel-accent-emerald" />
                  </div>
                  <span className="text-sentinel-text-primary">{feature}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>

      {/* Right Panel - Form */}
      <div className="flex-1 flex flex-col justify-center px-8 lg:px-16 xl:px-24">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-md w-full mx-auto"
        >
          {/* Logo */}
          <Link href="/" className="inline-flex items-center gap-3 mb-12">
            <Image 
              src="/logo.png" 
              alt="Sentinel Logo" 
              width={48} 
              height={48} 
              className="rounded-xl"
            />
            <span className="font-display font-bold text-2xl">SENTINEL</span>
          </Link>

          {/* Header */}
          <h1 className="text-3xl font-bold mb-2">Create Account</h1>
          <p className="text-sentinel-text-secondary mb-8">
            Start your journey with autonomous AI trading.
          </p>

          {/* Error Message */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-3 p-4 rounded-xl bg-sentinel-accent-crimson/10 border border-sentinel-accent-crimson/30 mb-6"
            >
              <AlertCircle className="w-5 h-5 text-sentinel-accent-crimson flex-shrink-0" />
              <span className="text-sentinel-accent-crimson text-sm">{error}</span>
            </motion.div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Name */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Full Name
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-4 py-3 rounded-xl bg-sentinel-bg-secondary border border-sentinel-border focus:border-sentinel-accent-cyan focus:outline-none focus:ring-1 focus:ring-sentinel-accent-cyan transition-all text-sentinel-text-primary"
                placeholder="John Doe"
                required
              />
            </div>

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
                  placeholder="Create a strong password"
                  required
                  minLength={8}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-sentinel-text-muted hover:text-sentinel-text-secondary"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              
              {/* Password Strength */}
              {formData.password && (
                <div className="mt-3">
                  <div className="flex gap-1 mb-2">
                    {[0, 1, 2, 3].map((idx) => (
                      <div
                        key={idx}
                        className={`h-1 flex-1 rounded-full transition-all ${
                          idx < passwordStrength() ? strengthColors[passwordStrength() - 1] : 'bg-sentinel-bg-tertiary'
                        }`}
                      />
                    ))}
                  </div>
                  <span className={`text-xs ${
                    passwordStrength() >= 3 ? 'text-sentinel-accent-emerald' : 
                    passwordStrength() >= 2 ? 'text-sentinel-accent-amber' : 'text-sentinel-accent-crimson'
                  }`}>
                    {strengthLabels[passwordStrength() - 1] || 'Too weak'}
                  </span>
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div>
              <label className="block text-sm font-medium text-sentinel-text-secondary mb-2">
                Confirm Password
              </label>
              <input
                type="password"
                value={formData.confirmPassword}
                onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                className={`w-full px-4 py-3 rounded-xl bg-sentinel-bg-secondary border focus:outline-none focus:ring-1 transition-all text-sentinel-text-primary ${
                  formData.confirmPassword && formData.confirmPassword !== formData.password
                    ? 'border-sentinel-accent-crimson focus:border-sentinel-accent-crimson focus:ring-sentinel-accent-crimson'
                    : 'border-sentinel-border focus:border-sentinel-accent-cyan focus:ring-sentinel-accent-cyan'
                }`}
                placeholder="Confirm your password"
                required
              />
              {formData.confirmPassword && formData.confirmPassword !== formData.password && (
                <p className="text-xs text-sentinel-accent-crimson mt-2">Passwords do not match</p>
              )}
            </div>

            {/* Terms */}
            <div className="flex items-start gap-3 pt-2">
              <input
                type="checkbox"
                id="terms"
                required
                className="mt-1 w-4 h-4 rounded border-sentinel-border bg-sentinel-bg-secondary text-sentinel-accent-cyan focus:ring-sentinel-accent-cyan"
              />
              <label htmlFor="terms" className="text-sm text-sentinel-text-secondary">
                I agree to the{' '}
                <Link href="/terms" className="text-sentinel-accent-cyan hover:underline">Terms of Service</Link>
                {' '}and{' '}
                <Link href="/privacy" className="text-sentinel-accent-cyan hover:underline">Privacy Policy</Link>
              </label>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading || formData.password !== formData.confirmPassword}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary font-bold text-lg flex items-center justify-center gap-3 hover:shadow-glow-cyan transition-all disabled:opacity-50"
            >
              {isLoading ? (
                <div className="w-6 h-6 border-2 border-sentinel-bg-primary border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  Create Account
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </form>

          {/* Login Link */}
          <p className="text-center text-sentinel-text-secondary mt-8">
            Already have an account?{' '}
            <Link href="/login" className="text-sentinel-accent-cyan hover:underline font-medium">
              Sign in
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  )
}
