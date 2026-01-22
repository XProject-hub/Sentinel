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
  LogOut,
  Bell,
  Send,
  Trash2,
  Info,
  XCircle
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
  status: 'learning' | 'ready' | 'junior' | 'amateur' | 'professional' | 'expert'
  lastUpdate: string
  description?: string
}

interface SystemStats {
  cpu_percent: number
  memory_percent: number
  disk_percent: number
  data_disk_percent: number
  uptime: string
  services: { name: string; status: string; healthy: boolean }[]
}

interface MaintenanceNotification {
  active: boolean
  message: string
  type: 'info' | 'warning' | 'danger'
  scheduled_at: string
  created_at: string
}

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<'users' | 'ai' | 'system' | 'maintenance'>('users')
  const [isLoading, setIsLoading] = useState(true)
  const [users, setUsers] = useState<User[]>([])
  const [aiModels, setAiModels] = useState<AIModel[]>([])
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)
  const [maintenance, setMaintenance] = useState<MaintenanceNotification | null>(null)
  const [newMessage, setNewMessage] = useState('')
  const [newType, setNewType] = useState<'info' | 'warning' | 'danger'>('info')
  const [newScheduledAt, setNewScheduledAt] = useState('')
  const [isSending, setIsSending] = useState(false)

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
      } else if (activeTab === 'maintenance') {
        const response = await fetch('/ai/admin/maintenance')
        if (response.ok) {
          const data = await response.json()
          setMaintenance(data)
          if (data.message) {
            setNewMessage(data.message)
            setNewType(data.type || 'info')
            setNewScheduledAt(data.scheduled_at || '')
          }
        }
      }
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const sendNotification = async () => {
    if (!newMessage.trim()) return
    
    setIsSending(true)
    try {
      const params = new URLSearchParams({
        message: newMessage,
        notification_type: newType,
        scheduled_at: newScheduledAt,
        active: 'true'
      })
      
      const response = await fetch(`/ai/admin/maintenance?${params}`, {
        method: 'POST'
      })
      
      if (response.ok) {
        alert('Notification sent successfully!')
        loadData()
      } else {
        alert('Failed to send notification')
      }
    } catch (error) {
      console.error('Failed to send notification:', error)
      alert('Failed to send notification')
    } finally {
      setIsSending(false)
    }
  }

  const clearNotification = async () => {
    if (!confirm('Are you sure you want to clear the notification?')) return
    
    setIsSending(true)
    try {
      const response = await fetch('/ai/admin/maintenance', {
        method: 'DELETE'
      })
      
      if (response.ok) {
        setMaintenance(null)
        setNewMessage('')
        setNewScheduledAt('')
        alert('Notification cleared!')
      }
    } catch (error) {
      console.error('Failed to clear notification:', error)
    } finally {
      setIsSending(false)
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
    { id: 'system', label: 'System', icon: Server },
    { id: 'maintenance', label: 'Notifications', icon: Bell }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'learning': return 'text-gray-400 bg-gray-500/10'      // Gray - just starting
      case 'ready': return 'text-amber-400 bg-amber-500/10'       // Amber - basic functionality
      case 'junior': return 'text-yellow-400 bg-yellow-500/10'    // Yellow - learning
      case 'amateur': return 'text-cyan-400 bg-cyan-500/10'       // Cyan - getting better
      case 'professional': return 'text-blue-400 bg-blue-500/10'  // Blue - good
      case 'expert': return 'text-emerald-400 bg-emerald-500/10'  // Green - true expert
      default: return 'text-gray-400 bg-gray-500/10'
    }
  }
  
  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'learning': return 'LEARNING'
      case 'ready': return 'READY'
      case 'junior': return 'JUNIOR'
      case 'amateur': return 'AMATEUR'
      case 'professional': return 'PRO'
      case 'expert': return 'EXPERT'
      default: return status.toUpperCase()
    }
  }
  
  const getProgressBarColor = (status: string) => {
    switch (status) {
      case 'learning': return 'bg-gray-500'
      case 'ready': return 'bg-amber-500'
      case 'junior': return 'bg-yellow-500'
      case 'amateur': return 'bg-cyan-500'
      case 'professional': return 'bg-blue-500'
      case 'expert': return 'bg-emerald-500'
      default: return 'bg-gray-500'
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
                  more data means more accurate predictions. <span className="text-emerald-400 font-medium">Expert</span> status 
                  requires MASSIVE amounts of data (tens of thousands of data points) for truly reliable decision-making.
                </p>
              </div>
              
              {/* Level Legend */}
              <div className="flex flex-wrap gap-3 mb-6 p-3 bg-white/[0.02] rounded-xl border border-white/5">
                <span className="text-xs text-gray-400">Levels:</span>
                <span className="px-2 py-0.5 rounded text-xs font-medium text-gray-400 bg-gray-500/10">LEARNING</span>
                <span className="px-2 py-0.5 rounded text-xs font-medium text-amber-400 bg-amber-500/10">READY</span>
                <span className="px-2 py-0.5 rounded text-xs font-medium text-yellow-400 bg-yellow-500/10">JUNIOR</span>
                <span className="px-2 py-0.5 rounded text-xs font-medium text-cyan-400 bg-cyan-500/10">AMATEUR</span>
                <span className="px-2 py-0.5 rounded text-xs font-medium text-blue-400 bg-blue-500/10">PRO</span>
                <span className="px-2 py-0.5 rounded text-xs font-medium text-emerald-400 bg-emerald-500/10">EXPERT</span>
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
                          {getStatusLabel(model.status)}
                        </span>
                      </div>
                      
                      <div className="mb-3">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-gray-500">Progress to Expert</span>
                          <span className={getStatusColor(model.status).split(' ')[0]}>{model.progress.toFixed(1)}%</span>
                        </div>
                        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full transition-all ${getProgressBarColor(model.status)}`}
                            style={{ width: `${Math.min(model.progress, 100)}%` }}
                          />
                        </div>
                      </div>
                      
                      <div className="flex justify-between text-xs mb-2">
                        <span className="text-gray-500">{model.dataPoints.toLocaleString()} data points</span>
                        <span className="text-gray-500">{model.lastUpdate}</span>
                      </div>
                      
                      {model.description && (
                        <p className="text-xs text-gray-500 truncate" title={model.description}>
                          {model.description}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Learning Explanation */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="p-4 bg-gray-500/5 border border-gray-500/20 rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-gray-400" />
                  <span className="font-medium text-gray-400">Learning (0-5%)</span>
                </div>
                <p className="text-sm text-gray-500">
                  Just starting. Collecting initial data. NOT ready for reliable trading.
                </p>
              </div>
              
              <div className="p-4 bg-amber-500/5 border border-amber-500/20 rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-amber-400" />
                  <span className="font-medium text-amber-400">Ready (5-20%)</span>
                </div>
                <p className="text-sm text-gray-400">
                  Basic functionality. Can make simple decisions but still learning.
                </p>
              </div>
              
              <div className="p-4 bg-yellow-500/5 border border-yellow-500/20 rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  <span className="font-medium text-yellow-400">Junior (20-40%)</span>
                </div>
                <p className="text-sm text-gray-400">
                  Learning patterns. Starting to recognize market conditions.
                </p>
              </div>
              
              <div className="p-4 bg-cyan-500/5 border border-cyan-500/20 rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-cyan-400" />
                  <span className="font-medium text-cyan-400">Amateur (40-60%)</span>
                </div>
                <p className="text-sm text-gray-400">
                  Getting better. Can identify good opportunities. Still improving.
                </p>
              </div>
              
              <div className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-blue-400" />
                  <span className="font-medium text-blue-400">Professional (60-80%)</span>
                </div>
                <p className="text-sm text-gray-400">
                  Strong performance. Good at predicting market moves. Reliable.
                </p>
              </div>
              
              <div className="p-4 bg-emerald-500/5 border border-emerald-500/20 rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span className="font-medium text-emerald-400">Expert (80-100%)</span>
                </div>
                <p className="text-sm text-gray-400">
                  Massive data. Highest accuracy. Truly reliable predictions.
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
                { label: 'CPU Usage', value: systemStats?.cpu_percent || 0, icon: Cpu, iconClass: 'text-cyan-400', barClass: 'bg-cyan-500' },
                { label: 'Memory', value: systemStats?.memory_percent || 0, icon: Activity, iconClass: 'text-violet-400', barClass: 'bg-violet-500' },
                { label: 'System Disk', value: systemStats?.disk_percent || 0, icon: HardDrive, iconClass: 'text-amber-400', barClass: 'bg-amber-500' },
                { label: 'Data Disk', value: systemStats?.data_disk_percent || 0, icon: Database, iconClass: 'text-emerald-400', barClass: 'bg-emerald-500' }
              ].map((stat, i) => (
                <div key={i} className="p-5 bg-white/[0.02] rounded-2xl border border-white/5">
                  <div className="flex items-center gap-2 mb-3">
                    <stat.icon className={`w-4 h-4 ${stat.iconClass}`} />
                    <span className="text-sm text-gray-500">{stat.label}</span>
                  </div>
                  <div className="text-2xl font-bold text-white mb-2">{stat.value.toFixed(1)}%</div>
                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${stat.barClass} rounded-full`}
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

        {/* Maintenance Tab */}
        {activeTab === 'maintenance' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Current Notification */}
            <div className="bg-white/[0.02] rounded-2xl border border-white/5 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Bell className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">Current Notification</h2>
              </div>
              
              {maintenance?.active ? (
                <div className={`p-4 rounded-xl border ${
                  maintenance.type === 'danger' ? 'bg-red-500/10 border-red-500/30' :
                  maintenance.type === 'warning' ? 'bg-amber-500/10 border-amber-500/30' :
                  'bg-blue-500/10 border-blue-500/30'
                }`}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      {maintenance.type === 'danger' ? (
                        <XCircle className="w-5 h-5 text-red-400 mt-0.5" />
                      ) : maintenance.type === 'warning' ? (
                        <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5" />
                      ) : (
                        <Info className="w-5 h-5 text-blue-400 mt-0.5" />
                      )}
                      <div>
                        <p className={`font-medium ${
                          maintenance.type === 'danger' ? 'text-red-400' :
                          maintenance.type === 'warning' ? 'text-amber-400' :
                          'text-blue-400'
                        }`}>
                          {maintenance.type === 'danger' ? 'Important Notice' :
                           maintenance.type === 'warning' ? 'Maintenance Notice' :
                           'System Notice'}
                        </p>
                        <p className="text-gray-300 mt-1">{maintenance.message}</p>
                        {maintenance.scheduled_at && (
                          <p className="text-xs text-gray-500 mt-2">
                            Scheduled: {maintenance.scheduled_at}
                          </p>
                        )}
                        <p className="text-xs text-gray-600 mt-1">
                          Created: {new Date(maintenance.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={clearNotification}
                      disabled={isSending}
                      className="p-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="p-4 bg-white/[0.02] rounded-xl border border-white/5 text-center">
                  <Bell className="w-8 h-8 text-gray-600 mx-auto mb-2" />
                  <p className="text-gray-500">No active notification</p>
                </div>
              )}
            </div>

            {/* Send New Notification */}
            <div className="bg-white/[0.02] rounded-2xl border border-white/5 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Send className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">Send Notification to All Users</h2>
              </div>

              <div className="space-y-4">
                {/* Message */}
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Message</label>
                  <textarea
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    placeholder="Enter notification message..."
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 resize-none"
                    rows={3}
                  />
                </div>

                {/* Type */}
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Type</label>
                  <div className="flex gap-3">
                    {[
                      { id: 'info', label: 'Info', color: 'blue', icon: Info },
                      { id: 'warning', label: 'Warning', color: 'amber', icon: AlertTriangle },
                      { id: 'danger', label: 'Critical', color: 'red', icon: XCircle }
                    ].map((type) => (
                      <button
                        key={type.id}
                        onClick={() => setNewType(type.id as any)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all ${
                          newType === type.id
                            ? `bg-${type.color}-500/20 border-${type.color}-500/50 text-${type.color}-400`
                            : 'bg-white/5 border-white/10 text-gray-400 hover:border-white/20'
                        }`}
                      >
                        <type.icon className="w-4 h-4" />
                        {type.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Scheduled At */}
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Scheduled Time (optional)</label>
                  <input
                    type="text"
                    value={newScheduledAt}
                    onChange={(e) => setNewScheduledAt(e.target.value)}
                    placeholder="e.g., Jan 25, 2026 at 10:00 UTC"
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50"
                  />
                </div>

                {/* Preview */}
                {newMessage && (
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Preview</label>
                    <div className={`p-4 rounded-xl border ${
                      newType === 'danger' ? 'bg-red-500/10 border-red-500/30' :
                      newType === 'warning' ? 'bg-amber-500/10 border-amber-500/30' :
                      'bg-blue-500/10 border-blue-500/30'
                    }`}>
                      <div className="flex items-start gap-3">
                        {newType === 'danger' ? (
                          <XCircle className="w-5 h-5 text-red-400 mt-0.5" />
                        ) : newType === 'warning' ? (
                          <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5" />
                        ) : (
                          <Info className="w-5 h-5 text-blue-400 mt-0.5" />
                        )}
                        <div>
                          <p className={`font-medium ${
                            newType === 'danger' ? 'text-red-400' :
                            newType === 'warning' ? 'text-amber-400' :
                            'text-blue-400'
                          }`}>
                            {newType === 'danger' ? '⚠️ Important Notice' :
                             newType === 'warning' ? '📢 Maintenance Notice' :
                             'ℹ️ System Notice'}
                          </p>
                          <p className="text-gray-300 mt-1">{newMessage}</p>
                          {newScheduledAt && (
                            <p className="text-xs text-gray-500 mt-2">
                              Scheduled: {newScheduledAt}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Send Button */}
                <button
                  onClick={sendNotification}
                  disabled={!newMessage.trim() || isSending}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
                >
                  {isSending ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <Send className="w-5 h-5" />
                      Send Notification
                    </>
                  )}
                </button>

                {/* Clear Button */}
                {maintenance?.active && (
                  <button
                    onClick={clearNotification}
                    disabled={isSending}
                    className="w-full py-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 font-medium hover:bg-red-500/20 transition-colors flex items-center justify-center gap-2"
                  >
                    <Trash2 className="w-5 h-5" />
                    Clear Current Notification
                  </button>
                )}
              </div>
            </div>

            {/* Help */}
            <div className="bg-cyan-500/5 border border-cyan-500/20 rounded-xl p-4">
              <h3 className="font-medium text-cyan-400 mb-2">📝 How it works</h3>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• <strong className="text-white">Info</strong> - General announcements (blue banner)</li>
                <li>• <strong className="text-white">Warning</strong> - Scheduled maintenance (amber banner)</li>
                <li>• <strong className="text-white">Critical</strong> - Urgent issues (red banner)</li>
                <li>• Users can dismiss the notification, but it will reappear if you send a new one</li>
                <li>• Notifications appear at the top of the dashboard</li>
              </ul>
            </div>
          </motion.div>
        )}
      </main>
    </div>
  )
}
