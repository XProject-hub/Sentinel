'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { 
  Brain, 
  Shield, 
  Zap, 
  TrendingUp, 
  BarChart3, 
  Lock,
  ArrowRight,
  CheckCircle,
  Activity,
  Target,
  Cpu,
  LineChart,
  Globe,
  Users,
  Clock,
  ChevronRight
} from 'lucide-react'
import Logo from '@/components/Logo'

export default function LandingPage() {
  const [stats, setStats] = useState({
    totalTrades: 2847,
    winRate: 51.9,
    activeUsers: 1,
    pairsMonitored: 650
  })

  useEffect(() => {
    // Fetch real stats
    const fetchStats = async () => {
      try {
        const response = await fetch('/ai/admin/stats')
        if (response.ok) {
          const data = await response.json()
          if (data.success) {
            setStats(prev => ({
              ...prev,
              totalTrades: data.data?.total_trades || prev.totalTrades,
              winRate: data.data?.win_rate || prev.winRate,
              activeUsers: data.data?.active_users || prev.activeUsers
            }))
          }
        }
      } catch (e) {
        // Keep defaults
      }
    }
    fetchStats()
  }, [])

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning models analyze market patterns 24/7, identifying high-probability trading opportunities.',
      iconClass: 'text-cyan-400',
      bgClass: 'bg-cyan-500/10'
    },
    {
      icon: Shield,
      title: 'Risk Management',
      description: 'Intelligent position sizing with Kelly Criterion, trailing stops, and automatic risk controls protect your capital.',
      iconClass: 'text-emerald-400',
      bgClass: 'bg-emerald-500/10'
    },
    {
      icon: Zap,
      title: 'Lightning Fast Execution',
      description: 'Sub-second order execution ensures you never miss a trade. React to market changes instantly.',
      iconClass: 'text-amber-400',
      bgClass: 'bg-amber-500/10'
    },
    {
      icon: Target,
      title: 'Precision Trading',
      description: 'Edge estimation and confidence scoring ensure only the highest quality trades are executed.',
      iconClass: 'text-violet-400',
      bgClass: 'bg-violet-500/10'
    },
    {
      icon: Activity,
      title: 'Live Market Monitoring',
      description: 'Real-time scanning of 650+ trading pairs across multiple timeframes and market conditions.',
      iconClass: 'text-rose-400',
      bgClass: 'bg-rose-500/10'
    },
    {
      icon: LineChart,
      title: 'Continuous Learning',
      description: 'The AI learns from every trade, continuously improving its strategies and adapting to market changes.',
      iconClass: 'text-blue-400',
      bgClass: 'bg-blue-500/10'
    }
  ]

  const howItWorks = [
    {
      step: '01',
      title: 'Connect Your Exchange',
      description: 'Securely link your Bybit account using API keys. Your funds stay in your control.'
    },
    {
      step: '02',
      title: 'Configure Your Settings',
      description: 'Set your risk tolerance, position sizes, and trading preferences. Start conservative.'
    },
    {
      step: '03',
      title: 'AI Takes Over',
      description: 'Sentinel analyzes markets 24/7, executing trades based on AI signals and your settings.'
    },
    {
      step: '04',
      title: 'Monitor & Grow',
      description: 'Track performance in real-time. The AI learns and improves with every trade.'
    }
  ]

  return (
    <div className="min-h-screen bg-[#0a0f1a]">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0a0f1a]/80 backdrop-blur-xl border-b border-white/5">
        <div className="w-full px-6 lg:px-12 py-4">
          <div className="flex items-center justify-between">
            <Logo size="md" />
            
            <div className="hidden md:flex items-center gap-8">
              <a href="#features" className="text-sm text-gray-400 hover:text-white transition-colors">Features</a>
              <a href="#how-it-works" className="text-sm text-gray-400 hover:text-white transition-colors">How It Works</a>
              <a href="#stats" className="text-sm text-gray-400 hover:text-white transition-colors">Statistics</a>
            </div>
            
            <div className="flex items-center gap-3">
              <Link 
                href="/login" 
                className="px-4 py-2 text-sm font-medium text-gray-300 hover:text-white transition-colors"
              >
                Sign In
              </Link>
              <Link 
                href="/register" 
                className="px-5 py-2.5 text-sm font-medium bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg hover:shadow-lg hover:shadow-cyan-500/25 transition-all"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6 lg:px-12 overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-radial from-cyan-500/5 to-transparent rounded-full" />
        </div>
        
        {/* Grid Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px]" />
        
        <div className="relative max-w-7xl mx-auto">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-8">
              <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-sm text-cyan-400 font-medium">AI Trading System Active</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              Autonomous
              <span className="block bg-gradient-to-r from-cyan-400 via-blue-400 to-violet-400 bg-clip-text text-transparent">
                AI Trading
              </span>
            </h1>
            
            <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
              Advanced artificial intelligence that analyzes markets, manages risk, and executes trades 24/7. 
              Your capital, our intelligence.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link 
                href="/register" 
                className="group flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold hover:shadow-xl hover:shadow-cyan-500/25 transition-all"
              >
                Start Trading Now
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link 
                href="/login" 
                className="flex items-center gap-2 px-8 py-4 bg-white/5 text-white rounded-xl font-semibold border border-white/10 hover:bg-white/10 transition-all"
              >
                <Lock className="w-5 h-5" />
                Sign In
              </Link>
            </div>
          </motion.div>
          
          {/* Stats Bar */}
          <motion.div 
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            id="stats"
            className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-4"
          >
            {[
              { label: 'Total Trades', value: stats.totalTrades.toLocaleString(), icon: BarChart3 },
              { label: 'Win Rate', value: `${stats.winRate}%`, icon: TrendingUp },
              { label: 'Active Users', value: stats.activeUsers.toString(), icon: Users },
              { label: 'Pairs Monitored', value: '650+', icon: Globe }
            ].map((stat, i) => (
              <div key={i} className="relative group">
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="relative p-6 bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 hover:border-cyan-500/30 transition-colors">
                  <stat.icon className="w-6 h-6 text-cyan-400 mb-3" />
                  <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
                  <div className="text-sm text-gray-500">{stat.label}</div>
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-6 lg:px-12 bg-gradient-to-b from-[#0a0f1a] to-[#0d1321]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">Intelligent Trading Features</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Powered by advanced AI models that learn and adapt to market conditions
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                viewport={{ once: true }}
                className="group p-6 bg-white/[0.02] rounded-2xl border border-white/5 hover:border-cyan-500/30 hover:bg-white/[0.04] transition-all"
              >
                <div className={`w-12 h-12 rounded-xl ${feature.bgClass} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <feature.icon className={`w-6 h-6 ${feature.iconClass}`} />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-gray-400 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-20 px-6 lg:px-12">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">How It Works</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Get started in minutes with our simple setup process
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {howItWorks.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                viewport={{ once: true }}
                className="relative"
              >
                {i < howItWorks.length - 1 && (
                  <div className="hidden lg:block absolute top-8 left-full w-full h-px bg-gradient-to-r from-cyan-500/50 to-transparent" />
                )}
                <div className="text-5xl font-bold text-cyan-500/20 mb-4">{item.step}</div>
                <h3 className="text-xl font-semibold text-white mb-2">{item.title}</h3>
                <p className="text-gray-400">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6 lg:px-12">
        <div className="max-w-4xl mx-auto">
          <div className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-cyan-500/10 via-blue-500/10 to-violet-500/10 border border-white/10 p-12 text-center">
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:32px_32px]" />
            
            <div className="relative">
              <h2 className="text-4xl font-bold text-white mb-4">Ready to Start?</h2>
              <p className="text-gray-400 mb-8 max-w-xl mx-auto">
                Join Sentinel and let AI work for you. Connect your exchange and start trading in minutes.
              </p>
              <Link 
                href="/register" 
                className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl font-semibold hover:shadow-xl hover:shadow-cyan-500/25 transition-all"
              >
                Create Free Account
                <ChevronRight className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </section>

    </div>
  )
}
