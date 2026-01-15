'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Shield, Zap, Brain, ArrowRight } from 'lucide-react'
import Link from 'next/link'

export default function LandingPage() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass-card border-b border-sentinel-border">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-sentinel-accent-cyan to-sentinel-accent-emerald flex items-center justify-center">
              <Shield className="w-6 h-6 text-sentinel-bg-primary" strokeWidth={2.5} />
            </div>
            <span className="font-display font-bold text-xl tracking-tight">SENTINEL</span>
          </div>
          
          <div className="flex items-center gap-6">
            <Link href="/login" className="text-sentinel-text-secondary hover:text-sentinel-text-primary transition-colors">
              Sign In
            </Link>
            <Link 
              href="/register" 
              className="px-5 py-2.5 rounded-lg bg-sentinel-accent-cyan text-sentinel-bg-primary font-semibold hover:bg-opacity-90 transition-all"
            >
              Get Started
            </Link>
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
                href="/register"
                className="group px-8 py-4 rounded-xl bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald text-sentinel-bg-primary font-bold text-lg flex items-center gap-3 hover:shadow-glow-cyan transition-all"
              >
                Start Trading with AI
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

      {/* Stats Bar */}
      <section className="border-t border-sentinel-border glass-card">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <StatItem label="Total Volume" value="$847M" />
            <StatItem label="Active Users" value="12,459" />
            <StatItem label="AI Accuracy" value="94.2%" />
            <StatItem label="Uptime" value="99.99%" />
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

function StatItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <div className="text-2xl md:text-3xl font-display font-bold text-sentinel-text-primary mb-1">
        {value}
      </div>
      <div className="text-sm text-sentinel-text-muted">{label}</div>
    </div>
  )
}

