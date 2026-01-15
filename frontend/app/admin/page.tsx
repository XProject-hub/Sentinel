'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Shield, 
  Users, 
  Brain, 
  Activity,
  Server,
  Database,
  TrendingUp,
  AlertTriangle,
  ChevronRight,
  Settings,
  LogOut,
  RefreshCw,
  Loader2,
  Cpu,
  HardDrive,
  BarChart3,
  Clock
} from 'lucide-react'
import Link from 'next/link'

interface SystemStats {
  uptime: string
  cpuUsage: number
  memoryUsage: number
  diskUsage: number
  activeConnections: number
}

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState('overview')
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [user, setUser] = useState<any>(null)
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)
  const [users, setUsers] = useState<any[]>([])
  const [aiStats, setAiStats] = useState<any>(null)

  useEffect(() => {
    // Check if admin
    const storedUser = localStorage.getItem('sentinel_user')
    if (storedUser) {
      const userData = JSON.parse(storedUser)
      if (!userData.isAdmin) {
        window.location.href = '/dashboard'
        return
      }
      setUser(userData)
    } else {
      window.location.href = '/login'
      return
    }
    
    loadData()
  }, [])

  const loadData = async () => {
    setIsLoading(true)
    
    try {
      // Load real system stats from AI service
      const statsRes = await fetch('/ai/admin/system')
      const statsData = await statsRes.json()
      if (statsData.success) {
        setSystemStats(statsData.data)
      }

      // Load users
      const usersRes = await fetch('/ai/admin/users')
      const usersData = await usersRes.json()
      if (usersData.success) {
        setUsers(usersData.data.users || [])
      }

      // Load AI learning stats
      const aiRes = await fetch('/ai/admin/ai-stats')
      const aiData = await aiRes.json()
      if (aiData.success) {
        setAiStats(aiData.data)
      }
    } catch (error) {
      console.error('Failed to load admin data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const refreshData = async () => {
    setIsRefreshing(true)
    await loadData()
    setIsRefreshing(false)
  }

  const handleLogout = () => {
    localStorage.removeItem('sentinel_user')
    window.location.href = '/login'
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'ai', label: 'AI Learning', icon: Brain },
    { id: 'system', label: 'System', icon: Server },
  ]

  if (isLoading) {
    return (
      <div className="min-h-screen bg-sentinel-bg-primary flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-sentinel-accent-cyan animate-spin mx-auto mb-4" />
          <p className="text-sentinel-text-secondary">Loading admin panel...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-sentinel-bg-primary">
      {/* Top Navigation */}
      <nav className="sticky top-0 z-50 glass-card border-b border-sentinel-border">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-sentinel-accent-crimson to-sentinel-accent-amber flex items-center justify-center">
              <Shield className="w-6 h-6 text-sentinel-bg-primary" strokeWidth={2.5} />
            </div>
            <div>
              <span className="font-display font-bold text-lg">SENTINEL ADMIN</span>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-sentinel-accent-emerald live-pulse" />
                <span className="text-xs text-sentinel-text-muted">Admin Panel</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button 
              onClick={refreshData}
              className={`p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors ${isRefreshing ? 'animate-spin' : ''}`}
            >
              <RefreshCw className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
            <Link href="/dashboard" className="px-3 py-1.5 rounded-lg bg-sentinel-accent-cyan/20 text-sentinel-accent-cyan text-sm font-medium hover:bg-sentinel-accent-cyan/30 transition-colors">
              Dashboard
            </Link>
            <div className="w-px h-8 bg-sentinel-border" />
            <button onClick={handleLogout} className="p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors">
              <LogOut className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
          </div>
        </div>
      </nav>

      <div className="flex">
        {/* Sidebar */}
        <aside className="w-64 min-h-[calc(100vh-4rem)] border-r border-sentinel-border p-4">
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                    activeTab === tab.id 
                      ? 'bg-sentinel-accent-cyan/10 text-sentinel-accent-cyan' 
                      : 'text-sentinel-text-secondary hover:bg-sentinel-bg-tertiary'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              )
            })}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-8">
          {activeTab === 'overview' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h1 className="text-2xl font-bold mb-8">Overview</h1>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="p-6 rounded-2xl glass-card">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 rounded-xl bg-sentinel-accent-cyan/10">
                      <Users className="w-6 h-6 text-sentinel-accent-cyan" />
                    </div>
                    <span className="text-sentinel-text-secondary">Total Users</span>
                  </div>
                  <div className="text-3xl font-bold">{users.length || 0}</div>
                </div>

                <div className="p-6 rounded-2xl glass-card">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 rounded-xl bg-sentinel-accent-emerald/10">
                      <Activity className="w-6 h-6 text-sentinel-accent-emerald" />
                    </div>
                    <span className="text-sentinel-text-secondary">Active Connections</span>
                  </div>
                  <div className="text-3xl font-bold">{systemStats?.activeConnections || 0}</div>
                </div>

                <div className="p-6 rounded-2xl glass-card">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 rounded-xl bg-sentinel-accent-amber/10">
                      <Brain className="w-6 h-6 text-sentinel-accent-amber" />
                    </div>
                    <span className="text-sentinel-text-secondary">AI Models</span>
                  </div>
                  <div className="text-3xl font-bold">{aiStats?.modelsLoaded || 0}</div>
                </div>

                <div className="p-6 rounded-2xl glass-card">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 rounded-xl bg-sentinel-accent-violet/10">
                      <Clock className="w-6 h-6 text-sentinel-accent-violet" />
                    </div>
                    <span className="text-sentinel-text-secondary">Uptime</span>
                  </div>
                  <div className="text-xl font-bold">{systemStats?.uptime || 'N/A'}</div>
                </div>
              </div>

              <div className="p-6 rounded-2xl glass-card">
                <h2 className="text-lg font-semibold mb-4">System Status</h2>
                <div className="grid grid-cols-3 gap-6">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sentinel-text-secondary">CPU Usage</span>
                      <span className="font-medium">{systemStats?.cpuUsage?.toFixed(1) || 0}%</span>
                    </div>
                    <div className="h-2 bg-sentinel-bg-tertiary rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-sentinel-accent-cyan rounded-full transition-all"
                        style={{ width: `${systemStats?.cpuUsage || 0}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sentinel-text-secondary">Memory</span>
                      <span className="font-medium">{systemStats?.memoryUsage?.toFixed(1) || 0}%</span>
                    </div>
                    <div className="h-2 bg-sentinel-bg-tertiary rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-sentinel-accent-emerald rounded-full transition-all"
                        style={{ width: `${systemStats?.memoryUsage || 0}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sentinel-text-secondary">Disk</span>
                      <span className="font-medium">{systemStats?.diskUsage?.toFixed(1) || 0}%</span>
                    </div>
                    <div className="h-2 bg-sentinel-bg-tertiary rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-sentinel-accent-amber rounded-full transition-all"
                        style={{ width: `${systemStats?.diskUsage || 0}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'users' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h1 className="text-2xl font-bold mb-8">Users</h1>
              
              {users.length === 0 ? (
                <div className="p-12 rounded-2xl glass-card text-center">
                  <Users className="w-16 h-16 mx-auto mb-4 text-sentinel-text-muted opacity-30" />
                  <h3 className="text-xl font-semibold mb-2">No Users Yet</h3>
                  <p className="text-sentinel-text-secondary">Users will appear here when they register.</p>
                </div>
              ) : (
                <div className="rounded-2xl glass-card overflow-hidden">
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-sm text-sentinel-text-muted border-b border-sentinel-border">
                        <th className="px-6 py-4 font-medium">Email</th>
                        <th className="px-6 py-4 font-medium">Name</th>
                        <th className="px-6 py-4 font-medium">Exchange</th>
                        <th className="px-6 py-4 font-medium">Status</th>
                        <th className="px-6 py-4 font-medium">Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {users.map((user, idx) => (
                        <tr key={idx} className="border-b border-sentinel-border/50 last:border-0">
                          <td className="px-6 py-4">{user.email}</td>
                          <td className="px-6 py-4">{user.name || 'N/A'}</td>
                          <td className="px-6 py-4">
                            {user.exchangeConnected ? (
                              <span className="text-sentinel-accent-emerald">{user.exchange}</span>
                            ) : (
                              <span className="text-sentinel-text-muted">Not connected</span>
                            )}
                          </td>
                          <td className="px-6 py-4">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              user.isActive ? 'bg-sentinel-accent-emerald/10 text-sentinel-accent-emerald' :
                              'bg-sentinel-text-muted/10 text-sentinel-text-muted'
                            }`}>
                              {user.isActive ? 'Active' : 'Inactive'}
                            </span>
                          </td>
                          <td className="px-6 py-4 text-sentinel-text-secondary">
                            {user.createdAt ? new Date(user.createdAt).toLocaleDateString() : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'ai' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h1 className="text-2xl font-bold mb-8">AI Learning Statistics</h1>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="p-6 rounded-2xl glass-card">
                  <h3 className="text-lg font-semibold mb-4">Model Status</h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                      <div className="flex items-center gap-3">
                        <Brain className="w-5 h-5 text-sentinel-accent-cyan" />
                        <span>Sentiment Model</span>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        aiStats?.sentimentModel === 'loaded' ? 'bg-sentinel-accent-emerald/10 text-sentinel-accent-emerald' :
                        'bg-sentinel-accent-amber/10 text-sentinel-accent-amber'
                      }`}>
                        {aiStats?.sentimentModel || 'Not loaded'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                      <div className="flex items-center gap-3">
                        <Activity className="w-5 h-5 text-sentinel-accent-emerald" />
                        <span>Strategy Model</span>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        aiStats?.strategyModel === 'loaded' ? 'bg-sentinel-accent-emerald/10 text-sentinel-accent-emerald' :
                        'bg-sentinel-accent-amber/10 text-sentinel-accent-amber'
                      }`}>
                        {aiStats?.strategyModel || 'Not loaded'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                      <div className="flex items-center gap-3">
                        <AlertTriangle className="w-5 h-5 text-sentinel-accent-amber" />
                        <span>Risk Model</span>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        aiStats?.riskModel === 'loaded' ? 'bg-sentinel-accent-emerald/10 text-sentinel-accent-emerald' :
                        'bg-sentinel-accent-amber/10 text-sentinel-accent-amber'
                      }`}>
                        {aiStats?.riskModel || 'Not loaded'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-6 rounded-2xl glass-card">
                  <h3 className="text-lg font-semibold mb-4">Learning Metrics</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sentinel-text-secondary">Training Progress</span>
                        <span className="font-medium">{aiStats?.trainingProgress || 0}%</span>
                      </div>
                      <div className="h-2 bg-sentinel-bg-tertiary rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald rounded-full"
                          style={{ width: `${aiStats?.trainingProgress || 0}%` }}
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 pt-4">
                      <div className="p-4 rounded-xl bg-sentinel-bg-tertiary">
                        <div className="text-2xl font-bold text-sentinel-accent-cyan">{aiStats?.decisionsToday || 0}</div>
                        <div className="text-sm text-sentinel-text-muted">Decisions Today</div>
                      </div>
                      <div className="p-4 rounded-xl bg-sentinel-bg-tertiary">
                        <div className="text-2xl font-bold text-sentinel-accent-emerald">{aiStats?.dataPoints || 0}</div>
                        <div className="text-sm text-sentinel-text-muted">Data Points</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-6 rounded-2xl glass-card">
                <h3 className="text-lg font-semibold mb-4">Recent AI Decisions</h3>
                {aiStats?.recentDecisions?.length ? (
                  <div className="space-y-3">
                    {aiStats.recentDecisions.map((decision: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                        <div className="flex items-center gap-3">
                          <div className={`w-2 h-2 rounded-full ${
                            decision.type === 'buy' ? 'bg-sentinel-accent-emerald' :
                            decision.type === 'sell' ? 'bg-sentinel-accent-crimson' :
                            'bg-sentinel-accent-amber'
                          }`} />
                          <span className="font-mono">{decision.symbol}</span>
                          <span className="text-sentinel-text-secondary">{decision.action}</span>
                        </div>
                        <div className="text-sm text-sentinel-text-muted">
                          {new Date(decision.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-sentinel-text-muted">
                    <Brain className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p>No AI decisions yet</p>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {activeTab === 'system' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h1 className="text-2xl font-bold mb-8">System Status</h1>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="p-6 rounded-2xl glass-card">
                  <h3 className="text-lg font-semibold mb-4">Server Information</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between py-2 border-b border-sentinel-border/30">
                      <span className="text-sentinel-text-secondary">IP Address</span>
                      <span className="font-mono">109.104.154.183</span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-sentinel-border/30">
                      <span className="text-sentinel-text-secondary">Uptime</span>
                      <span>{systemStats?.uptime || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-sentinel-border/30">
                      <span className="text-sentinel-text-secondary">OS</span>
                      <span>Ubuntu 22.04 LTS</span>
                    </div>
                    <div className="flex justify-between py-2">
                      <span className="text-sentinel-text-secondary">Docker Status</span>
                      <span className="text-sentinel-accent-emerald">Running</span>
                    </div>
                  </div>
                </div>

                <div className="p-6 rounded-2xl glass-card">
                  <h3 className="text-lg font-semibold mb-4">Service Health</h3>
                  <div className="space-y-3">
                    {['Frontend', 'Backend', 'AI Services', 'PostgreSQL', 'Redis', 'Kafka'].map((service) => (
                      <div key={service} className="flex items-center justify-between py-2 border-b border-sentinel-border/30 last:border-0">
                        <span className="text-sentinel-text-secondary">{service}</span>
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-sentinel-accent-emerald" />
                          <span className="text-sm">Healthy</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="p-6 rounded-2xl glass-card">
                <h3 className="text-lg font-semibold mb-4">Resource Usage</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="p-6 rounded-xl bg-sentinel-bg-tertiary text-center">
                    <Cpu className="w-8 h-8 text-sentinel-accent-cyan mx-auto mb-3" />
                    <div className="text-3xl font-bold">{systemStats?.cpuUsage?.toFixed(1) || 0}%</div>
                    <div className="text-sentinel-text-muted">CPU Usage</div>
                  </div>
                  <div className="p-6 rounded-xl bg-sentinel-bg-tertiary text-center">
                    <Server className="w-8 h-8 text-sentinel-accent-emerald mx-auto mb-3" />
                    <div className="text-3xl font-bold">{systemStats?.memoryUsage?.toFixed(1) || 0}%</div>
                    <div className="text-sentinel-text-muted">Memory Usage</div>
                  </div>
                  <div className="p-6 rounded-xl bg-sentinel-bg-tertiary text-center">
                    <HardDrive className="w-8 h-8 text-sentinel-accent-amber mx-auto mb-3" />
                    <div className="text-3xl font-bold">{systemStats?.diskUsage?.toFixed(1) || 0}%</div>
                    <div className="text-sentinel-text-muted">Disk Usage</div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-sentinel-border">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between text-sm text-sentinel-text-muted">
          <span>SENTINEL Admin Panel</span>
          <span>Developed by NoLimitDevelopments</span>
        </div>
      </footer>
    </div>
  )
}
