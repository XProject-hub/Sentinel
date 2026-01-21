'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { 
  Brain,
  ArrowLeft,
  Users,
  Activity,
  Server,
  RefreshCw,
  Loader2,
  CheckCircle,
  AlertTriangle,
  TrendingUp,
  Target,
  Database,
  Cpu,
  HardDrive,
  Zap,
  BarChart3,
  Clock,
  Shield,
  LogOut
} from 'lucide-react'

interface User {
  id: string
  email: string
  name: string
  exchange: string | null
  exchangeConnected: boolean
  isActive: boolean
  isPaused: boolean
  isAdmin: boolean
  createdAt: string
  totalTrades: number
  totalPnl: number
  winRate: number
}

interface AIModel {
  name: string
  progress: number
  dataPoints: number
  status: 'learning' | 'ready' | 'expert'
  lastUpdate: string
}

interface SystemStats {
  cpu_percent: number
  memory_percent: number
  disk_percent: number
  data_disk_percent: number
  uptime: string
  services: { name: string; status: string; healthy: boolean }[]
}

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<'users' | 'ai' | 'system'>('users')
  const [isLoading, setIsLoading] = useState(true)
  const [users, setUsers] = useState<User[]>([])
  const [aiModels, setAiModels] = useState<AIModel[]>([])
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 10000)
    return () => clearInterval(interval)
  }, [activeTab])

  const loadData = async () => {
    try {
      if (activeTab === 'users') {
        const response = await fetch('/ai/admin/users')
        if (response.ok) {
          const data = await response.json()
          setUsers(data.data?.users || [])
        }
      } else if (activeTab === 'ai') {
        const response = await fetch('/ai/admin/ai-stats')
        if (response.ok) {
          const data = await response.json()
          if (data.models) {
            setAiModels(data.models)
          }
        }
      } else if (activeTab === 'system') {
        const response = await fetch('/ai/admin/system')
        if (response.ok) {
          const data = await response.json()
          setSystemStats(data)
        }
      }
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('sentinel_user')
    localStorage.removeItem('token')
    window.location.href = '/login'
  }

  const tabs = [
    { id: 'users', label: 'Users', icon: Users },
    { id: 'ai', label: 'AI Learning', icon: Brain },
    { id: 'system', label: 'System', icon: Server }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'learning': return 'text-amber-400 bg-amber-500/10'
      case 'ready': return 'text-cyan-400 bg-cyan-500/10'
      case 'expert': return 'text-emerald-400 bg-emerald-500/10'
      default: return 'text-gray-400 bg-gray-500/10'
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0f1a]">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-[#0a0f1a]/95 backdrop-blur-xl border-b border-white/5">
        <div className="w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/dashboard" 
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-400" />
              </Link>
              <div>
                <h1 className="text-xl font-bold text-white">Admin Panel</h1>
                <p className="text-sm text-gray-500">System management & monitoring</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={loadData}
                className="p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <RefreshCw className={`w-5 h-5 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
              </button>
              
              <Link
                href="/dashboard"
                className="px-4 py-2 rounded-lg bg-cyan-500/10 text-cyan-400 text-sm font-medium hover:bg-cyan-500/20 transition-colors"
              >
                Dashboard
              </Link>
              
              <button 
                onClick={handleLogout}
                className="p-2.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
              >
                <LogOut className="w-5 h-5 text-gray-400" />
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 mt-4">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => { setActiveTab(tab.id as any); setIsLoading(true) }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === tab.id
                    ? 'bg-white/10 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        {/* Users Tab */}
        {activeTab === 'users' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden"
          >
            <div className="p-5 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Users className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">Registered Users</h2>
              </div>
              <span className="text-sm text-gray-500">Total: {users.length}</span>
            </div>

            {isLoading ? (
              <div className="p-12 flex justify-center">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              </div>
            ) : users.length === 0 ? (
              <div className="p-12 text-center">
                <Users className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-500">No users found</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/5">
                      <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">User</th>
                      <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Exchange</th>
                      <th className="text-left text-xs font-medium text-gray-500 px-5 py-3">Status</th>
                      <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">Trades</th>
                      <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">Win Rate</th>
                      <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">P&L</th>
                      <th className="text-right text-xs font-medium text-gray-500 px-5 py-3">Joined</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map((user) => (
                      <tr key={user.id} className="border-b border-white/5 hover:bg-white/[0.02]">
                        <td className="px-5 py-4">
                          <div className="flex items-center gap-3">
                            <div className={`w-2 h-2 rounded-full ${user.isActive ? 'bg-emerald-400' : 'bg-gray-500'}`} />
                            <div>
                              <span className="font-medium text-white">{user.name}</span>
                              {user.isAdmin && (
                                <span className="ml-2 px-1.5 py-0.5 text-[10px] bg-cyan-500/20 text-cyan-400 rounded">
                                  ADMIN
                                </span>
                              )}
                              <div className="text-xs text-gray-500">{user.email}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-5 py-4">
                          {user.exchangeConnected ? (
                            <span className="text-emerald-400 text-sm">Bybit</span>
                          ) : (
                            <span className="text-gray-500 text-sm">Not connected</span>
                          )}
                        </td>
                        <td className="px-5 py-4">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 w-fit ${
                            user.isActive ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' :
                            user.isPaused ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30' :
                            'bg-gray-500/10 text-gray-400 border border-gray-500/30'
                          }`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${
                              user.isActive ? 'bg-emerald-400' :
                              user.isPaused ? 'bg-amber-400' :
                              'bg-gray-400'
                            }`} />
                            {user.isActive ? 'Trading' : user.isPaused ? 'Paused' : 'Inactive'}
                          </span>
                        </td>
                        <td className="px-5 py-4 text-right">
                          <span className="font-mono text-white">{user.totalTrades.toLocaleString()}</span>
                        </td>
                        <td className="px-5 py-4 text-right">
                          <span className={`font-mono ${user.winRate >= 50 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {user.winRate.toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-5 py-4 text-right">
                          <span className={`font-mono ${user.totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {user.totalPnl >= 0 ? '+' : ''}€{user.totalPnl.toFixed(2)}
                          </span>
                        </td>
                        <td className="px-5 py-4 text-right text-gray-400 text-sm">
                          {new Date(user.createdAt).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </motion.div>
        )}

        {/* AI Learning Tab */}
        {activeTab === 'ai' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* AI Overview */}
            <div className="bg-white/[0.02] rounded-2xl border border-white/5 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">AI Learning Overview</h2>
              </div>
              
              <div className="p-4 bg-cyan-500/5 border border-cyan-500/20 rounded-xl mb-6">
                <p className="text-sm text-gray-300 leading-relaxed">
                  <span className="text-cyan-400 font-medium">How AI Learning Works:</span> Sentinel's AI 
                  continuously learns from market data and trade outcomes. Progress shows data accumulation - 
                  more data means more accurate predictions. "Expert" status indicates the model has enough 
                  historical data for reliable decision-making, but learning never stops.
                </p>
              </div>

              {isLoading ? (
                <div className="py-12 flex justify-center">
                  <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                </div>
              ) : (
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {aiModels.map((model, i) => (
                    <div key={i} className="p-4 bg-white/[0.02] rounded-xl border border-white/10">
                      <div className="flex items-center justify-between mb-3">
                        <span className="font-medium text-white">{model.name}</span>
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(model.status)}`}>
                          {model.status.toUpperCase()}
                        </span>
                      </div>
                      
                      <div className="mb-3">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-gray-500">Progress</span>
                          <span className="text-cyan-400">{model.progress.toFixed(0)}%</span>
                        </div>
                        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full transition-all ${
                              model.status === 'expert' ? 'bg-emerald-500' :
                              model.status === 'ready' ? 'bg-cyan-500' :
                              'bg-amber-500'
                            }`}
                            style={{ width: `${Math.min(model.progress, 100)}%` }}
                          />
                        </div>
                      </div>
                      
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-500">{model.dataPoints.toLocaleString()} data points</span>
                        <span className="text-gray-500">{model.lastUpdate}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Learning Explanation */}
            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-4 bg-amber-500/5 border border-amber-500/20 rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-amber-400" />
                  <span className="font-medium text-amber-400">Learning</span>
                </div>
                <p className="text-sm text-gray-400">
                  Collecting initial data. Predictions may be less accurate.
                </p>
              </div>
              
              <div className="p-4 bg-cyan-500/5 border border-cyan-500/20 rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-4 h-4 text-cyan-400" />
                  <span className="font-medium text-cyan-400">Ready</span>
                </div>
                <p className="text-sm text-gray-400">
                  Has enough data for good predictions. Still improving.
                </p>
              </div>
              
              <div className="p-4 bg-emerald-500/5 border border-emerald-500/20 rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span className="font-medium text-emerald-400">Expert</span>
                </div>
                <p className="text-sm text-gray-400">
                  Fully calibrated with extensive data. Highest accuracy.
                </p>
              </div>
            </div>
          </motion.div>
        )}

        {/* System Tab */}
        {activeTab === 'system' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* System Stats */}
            <div className="grid md:grid-cols-4 gap-4">
              {[
                { label: 'CPU Usage', value: systemStats?.cpu_percent || 0, icon: Cpu, color: 'cyan' },
                { label: 'Memory', value: systemStats?.memory_percent || 0, icon: Activity, color: 'violet' },
                { label: 'System Disk', value: systemStats?.disk_percent || 0, icon: HardDrive, color: 'amber' },
                { label: 'Data Disk', value: systemStats?.data_disk_percent || 0, icon: Database, color: 'emerald' }
              ].map((stat, i) => (
                <div key={i} className="p-5 bg-white/[0.02] rounded-2xl border border-white/5">
                  <div className="flex items-center gap-2 mb-3">
                    <stat.icon className={`w-4 h-4 text-${stat.color}-400`} />
                    <span className="text-sm text-gray-500">{stat.label}</span>
                  </div>
                  <div className="text-2xl font-bold text-white mb-2">{stat.value.toFixed(1)}%</div>
                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className={`h-full bg-${stat.color}-500 rounded-full`}
                      style={{ width: `${stat.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Services */}
            <div className="bg-white/[0.02] rounded-2xl border border-white/5 overflow-hidden">
              <div className="p-5 border-b border-white/5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Server className="w-5 h-5 text-cyan-400" />
                  <h2 className="font-semibold text-white">Services</h2>
                </div>
                <span className="text-sm text-gray-500">
                  Uptime: {systemStats?.uptime || 'N/A'}
                </span>
              </div>
              
              {isLoading ? (
                <div className="p-12 flex justify-center">
                  <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                </div>
              ) : (
                <div className="p-5 grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {(systemStats?.services || []).map((service, i) => (
                    <div 
                      key={i} 
                      className={`p-4 rounded-xl border ${
                        service.healthy 
                          ? 'bg-emerald-500/5 border-emerald-500/20' 
                          : 'bg-red-500/5 border-red-500/20'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-white">{service.name}</span>
                        <div className="flex items-center gap-1.5">
                          <div className={`w-2 h-2 rounded-full ${
                            service.healthy ? 'bg-emerald-400' : 'bg-red-400'
                          }`} />
                          <span className={`text-xs ${
                            service.healthy ? 'text-emerald-400' : 'text-red-400'
                          }`}>
                            {service.status}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </main>
    </div>
  )
}
