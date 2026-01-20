'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Zap, Brain, ArrowRight, Shield, User, LogOut } from 'lucide-react'
import Link from 'next/link'

interface LiveStats {
  totalVolume: number
  activeUsers: number
  aiAccuracy: number
  uptime: number
  totalTrades: number
  winRate: number
}

interface UserInfo {
  name: string
  email: string
}

export default function LandingPage() {
  const [mounted, setMounted] = useState(false)
  const [stats, setStats] = useState<LiveStats>({
    totalVolume: 0,
    activeUsers: 0,
    aiAccuracy: 0,
    uptime: 99.99,
    totalTrades: 0,
    winRate: 0
  })
  const [user, setUser] = useState<UserInfo | null>(null)
  const [showUserMenu, setShowUserMenu] = useState(false)

  useEffect(() => {
    setMounted(true)
    
    // Check if user is logged in
    const checkAuth = async () => {
      const token = localStorage.getItem('auth_token')
      if (token) {
        try {
          const res = await fetch('/api/auth/me', {
            headers: { 'Authorization': `Bearer ${token}` }
          })
          if (res.ok) {
            const data = await res.json()
            setUser(data.data || data.user || data)
          }
        } catch (e) {
          // Not logged in or token expired
          localStorage.removeItem('auth_token')
        }
      }
    }
    checkAuth()
    
    // Fetch live stats
    const fetchStats = async () => {
      try {
        const res = await fetch('/ai/admin/stats')
        if (res.ok) {
          const data = await res.json()
          setStats({
            totalVolume: data.total_volume || data.totalVolume || 0,
            activeUsers: data.active_users || data.activeUsers || 0,
            aiAccuracy: data.ai_accuracy || data.aiAccuracy || data.win_rate || 0,
            uptime: data.uptime || 99.99,
            totalTrades: data.total_trades || data.totalTrades || 0,
            winRate: data.win_rate || data.winRate || 0
          })
        }
      } catch (e) {
        // Use fallback stats if API fails
        console.log('Using fallback stats')
      }
    }
    
    fetchStats()
    // Refresh stats every 30 seconds
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [])

  const handleLogout = () => {
    localStorage.removeItem('auth_token')
    setUser(null)
    setShowUserMenu(false)
  }

  const formatVolume = (value: number) => {
    if (value >= 1000000000) return `€${(value / 1000000000).toFixed(1)}B`
    if (value >= 1000000) return `€${(value / 1000000).toFixed(1)}M`
    if (value >= 1000) return `€${(value / 1000).toFixed(1)}K`
    return `€${value.toFixed(0)}`
  }

  const formatNumber = (value: number) => {
    return value.toLocaleString('en-US')
  }

  if (!mounted) return null

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass-card border-b border-sentinel-border">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="font-display font-bold text-2xl tracking-tight text-white">SENTINEL</span>
          </div>
          
          <div className="flex items-center gap-6">
            {user ? (
              // Logged in - show user menu
              <div className="relative">
                <button 
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg glass-card hover:border-sentinel-accent-cyan transition-all"
                >
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-sentinel-accent-cyan to-sentinel-accent-emerald flex items-center justify-center">
                    <User className="w-4 h-4 text-sentinel-bg-primary" />
                  </div>
                  <span className="text-sentinel-text-primary font-medium">{user.name}</span>
                </button>
                
                {showUserMenu && (
                  <div className="absolute right-0 mt-2 w-48 glass-card rounded-xl border border-sentinel-border shadow-xl overflow-hidden">
                    <Link 
                      href="/dashboard" 
                      className="block px-4 py-3 text-sentinel-text-primary hover:bg-sentinel-bg-secondary transition-colors"
                    >
                      Dashboard
                    </Link>
                    <Link 
                      href="/dashboard/settings" 
                      className="block px-4 py-3 text-sentinel-text-primary hover:bg-sentinel-bg-secondary transition-colors"
                    >
                      Settings
                    </Link>
                    <button 
                      onClick={handleLogout}
                      className="w-full text-left px-4 py-3 text-sentinel-status-error hover:bg-sentinel-bg-secondary transition-colors flex items-center gap-2"
                    >
                      <LogOut className="w-4 h-4" />
                      Sign Out
                    </button>
                  </div>
                )}
              </div>
            ) : (
              // Not logged in - show sign in / get started
              <>
                <Link href="/login" className="text-sentinel-text-secondary hover:text-sentinel-text-primary transition-colors">
                  Sign In
                </Link>
                <Link 
                  href="/register" 
                  className="px-5 py-2.5 rounded-lg bg-sentinel-accent-cyan text-sentinel-bg-primary font-semibold hover:bg-opacity-90 transition-all"
                >
                  Get Started
                </Link>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center px-6 pt-32 pb-20">
        <div className="max-w-5xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-8">
              <span className="w-2 h-2 rounded-full bg-sentinel-accent-emerald live-pulse" />
              <span className="text-sm text-sentinel-text-secondary">Autonomous AI Trading Active</span>
            </div>

            {/* Headline */}
            <h1 className="text-5xl md:text-7xl font-display font-bold leading-tight mb-6">
              <span className="text-sentinel-text-primary">Your Capital.</span>
              <br />
              <span className="text-gradient-cyan">AI Protected.</span>
            </h1>

            {/* Subheadline */}
            <p className="text-xl md:text-2xl text-sentinel-text-secondary max-w-3xl mx-auto mb-12 leading-relaxed">
              SENTINEL AI monitors markets, analyzes sentiment, plans strategies, and executes trades.
              <br className="hidden md:block" />
              <span className="text-sentinel-text-primary font-medium">You just watch the profits.</span>
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
              <Link 
                href={user ? "/dashboard" : "/register"}
                className="group px-8 py-4 rounded-xl bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary font-bold text-lg flex items-center gap-3 hover:shadow-glow-cyan transition-all"
              >
                {user ? "Go to Dashboard" : "Start Trading with AI"}
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link 
                href="/demo"
                className="px-8 py-4 rounded-xl glass-card text-sentinel-text-primary font-semibold text-lg hover:border-sentinel-accent-cyan transition-all"
              >
                View Live Demo
              </Link>
            </div>
          </motion.div>

          {/* Feature Cards */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="grid md:grid-cols-3 gap-6"
          >
            <FeatureCard
              icon={<Brain className="w-8 h-8" />}
              title="Market Intelligence"
              description="Real-time analysis of price action, order flow, news sentiment, and on-chain data."
              color="cyan"
              delay={0}
            />
            <FeatureCard
              icon={<Zap className="w-8 h-8" />}
              title="Autonomous Trading"
              description="AI selects strategies, optimizes entries, and executes trades without human intervention."
              color="emerald"
              delay={0.1}
            />
            <FeatureCard
              icon={<Shield className="w-8 h-8" />}
              title="Capital Protection"
              description="Strict risk management with automatic stop-loss, position limits, and emergency controls."
              color="violet"
              delay={0.2}
            />
          </motion.div>
        </div>
      </section>

      {/* Live Stats Bar */}
      <section className="border-t border-sentinel-border glass-card">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <LiveStatItem 
              label="Total Volume" 
              value={formatVolume(stats.totalVolume)} 
              isLive={stats.totalVolume > 0}
            />
            <LiveStatItem 
              label="Active Users" 
              value={formatNumber(stats.activeUsers)} 
              isLive={stats.activeUsers > 0}
            />
            <LiveStatItem 
              label="AI Win Rate" 
              value={`${stats.aiAccuracy.toFixed(1)}%`} 
              isLive={stats.aiAccuracy > 0}
            />
            <LiveStatItem 
              label="Uptime" 
              value={`${stats.uptime.toFixed(2)}%`} 
              isLive={true}
            />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-sentinel-border bg-sentinel-bg-secondary/50">
        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <span className="text-sentinel-text-muted text-sm">
            SENTINEL AI - Autonomous Digital Trader
          </span>
          <span className="text-sentinel-text-muted text-sm">
            Developed by NoLimitDevelopments
          </span>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({ 
  icon, 
  title, 
  description, 
  color,
  delay 
}: { 
  icon: React.ReactNode
  title: string
  description: string
  color: 'cyan' | 'emerald' | 'violet'
  delay: number
}) {
  const colorMap = {
    cyan: 'text-sentinel-accent-cyan border-sentinel-accent-cyan/20 hover:border-sentinel-accent-cyan/50',
    emerald: 'text-sentinel-accent-emerald border-sentinel-accent-emerald/20 hover:border-sentinel-accent-emerald/50',
    violet: 'text-sentinel-accent-violet border-sentinel-accent-violet/20 hover:border-sentinel-accent-violet/50',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 + delay }}
      className={`p-6 rounded-2xl glass-card ${colorMap[color]} transition-all group cursor-default`}
    >
      <div className={`mb-4 ${colorMap[color].split(' ')[0]}`}>
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-sentinel-text-primary mb-2">{title}</h3>
      <p className="text-sentinel-text-secondary text-sm leading-relaxed">{description}</p>
    </motion.div>
  )
}

function LiveStatItem({ label, value, isLive }: { label: string; value: string; isLive: boolean }) {
  return (
    <div className="text-center">
      <div className="flex items-center justify-center gap-2 mb-1">
        {isLive && (
          <span className="w-2 h-2 rounded-full bg-sentinel-accent-emerald live-pulse" />
        )}
        <div className="text-2xl md:text-3xl font-display font-bold text-sentinel-text-primary">
          {value}
        </div>
      </div>
      <div className="text-sm text-sentinel-text-muted">{label}</div>
    </div>
  )
}
