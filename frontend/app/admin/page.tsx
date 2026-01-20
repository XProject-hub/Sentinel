'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { 
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
import Image from 'next/image'

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

      // Load AI stats from admin endpoint (for modelsLoaded count)
      const aiAdminRes = await fetch('/ai/admin/ai-stats')
      const aiAdminData = await aiAdminRes.json()
      
      // Load AI learning stats from real learning engine
      const aiRes = await fetch('/ai/learning/stats')
      const aiData = await aiRes.json()
      
      // Load recent learning events
      const eventsRes = await fetch('/ai/learning/events?limit=10')
      const eventsData = await eventsRes.json()
      
      // Load detailed learning status from new endpoint
      const learningStatusRes = await fetch('/ai/exchange/learning/status')
      const learningStatusData = await learningStatusRes.json()
      
      // Combine all AI data
      setAiStats({
        ...(aiData.success ? aiData.data : {}),
        ...(aiAdminData.success ? aiAdminData.data : {}),
        recentDecisions: eventsData.success ? eventsData.data : [],
        learningStatus: learningStatusData.success ? learningStatusData.data : null
      })
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
            <Image 
              src="/logo.png" 
              alt="Sentinel Logo" 
              width={40} 
              height={40} 
              className="rounded-lg"
            />
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
              
              {/* AI Models Status */}
              <div className="p-6 rounded-2xl glass-card mb-8">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">AI Learning Models</h3>
                  <div className="text-2xl font-bold text-sentinel-accent-cyan">
                    {aiStats?.modelsLoaded || 0} / {aiStats?.totalModels || 5} Active
                  </div>
                </div>
                <div className="grid grid-cols-5 gap-4">
                  <div className={`p-4 rounded-xl text-center ${aiStats?.strategyModel === 'active' ? 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30' : 'bg-sentinel-bg-tertiary'}`}>
                    <div className={`w-3 h-3 rounded-full mx-auto mb-2 ${aiStats?.strategyModel === 'active' ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-text-muted'}`} />
                    <div className="text-sm font-medium">Strategy</div>
                    <div className="text-xs text-sentinel-text-muted capitalize">{aiStats?.strategyModel || 'loading'}</div>
                  </div>
                  <div className={`p-4 rounded-xl text-center ${aiStats?.patternModel === 'active' ? 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30' : 'bg-sentinel-bg-tertiary'}`}>
                    <div className={`w-3 h-3 rounded-full mx-auto mb-2 ${aiStats?.patternModel === 'active' ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-text-muted'}`} />
                    <div className="text-sm font-medium">Patterns</div>
                    <div className="text-xs text-sentinel-text-muted capitalize">{aiStats?.patternModel || 'loading'}</div>
                  </div>
                  <div className={`p-4 rounded-xl text-center ${aiStats?.marketModel === 'active' ? 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30' : 'bg-sentinel-bg-tertiary'}`}>
                    <div className={`w-3 h-3 rounded-full mx-auto mb-2 ${aiStats?.marketModel === 'active' ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-text-muted'}`} />
                    <div className="text-sm font-medium">Market</div>
                    <div className="text-xs text-sentinel-text-muted capitalize">{aiStats?.marketModel || 'loading'}</div>
                  </div>
                  <div className={`p-4 rounded-xl text-center ${aiStats?.sentimentModel === 'active' ? 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30' : 'bg-sentinel-bg-tertiary'}`}>
                    <div className={`w-3 h-3 rounded-full mx-auto mb-2 ${aiStats?.sentimentModel === 'active' ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-text-muted'}`} />
                    <div className="text-sm font-medium">Sentiment</div>
                    <div className="text-xs text-sentinel-text-muted capitalize">{aiStats?.sentimentModel || 'loading'}</div>
                  </div>
                  <div className={`p-4 rounded-xl text-center ${aiStats?.technicalModel === 'active' ? 'bg-sentinel-accent-emerald/10 border border-sentinel-accent-emerald/30' : 'bg-sentinel-bg-tertiary'}`}>
                    <div className={`w-3 h-3 rounded-full mx-auto mb-2 ${aiStats?.technicalModel === 'active' ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-text-muted'}`} />
                    <div className="text-sm font-medium">Technical</div>
                    <div className="text-xs text-sentinel-text-muted capitalize">{aiStats?.technicalModel || 'loading'}</div>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-sentinel-border">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-sentinel-text-muted">Training Progress</span>
                    <span className="font-mono">{aiStats?.trainingProgress?.toFixed(1) || 0}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-sentinel-bg-tertiary overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-sentinel-accent-violet via-sentinel-accent-cyan to-sentinel-accent-emerald transition-all"
                      style={{ width: `${aiStats?.trainingProgress || 0}%` }}
                    />
                  </div>
                  <div className="text-xs text-sentinel-text-muted mt-2">
                    Learning Iteration #{aiStats?.learningIterations || 0} | {aiStats?.totalStatesLearned || 0} total states learned
                  </div>
                </div>
              </div>
              
              {/* DETAILED LEARNING PROGRESS */}
              {aiStats?.learningStatus && (
                <div className="p-6 rounded-2xl glass-card mb-8 border-2 border-sentinel-accent-violet/30">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <div className="p-3 rounded-xl bg-sentinel-accent-violet/10">
                        <Brain className="w-6 h-6 text-sentinel-accent-violet" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold">AI Learning Progress</h3>
                        <p className="text-sm text-sentinel-text-muted">{aiStats.learningStatus.expert_level}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold text-sentinel-accent-violet">
                        {aiStats.learningStatus.overall_progress?.toFixed(0) || 0}%
                      </div>
                      <div className="text-xs text-sentinel-text-muted">Overall Progress</div>
                    </div>
                  </div>

                  {/* Overall Progress Bar */}
                  <div className="mb-6">
                    <div className="h-4 rounded-full bg-sentinel-bg-tertiary overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-sentinel-accent-crimson via-sentinel-accent-amber via-sentinel-accent-cyan to-sentinel-accent-emerald transition-all duration-500"
                        style={{ width: `${aiStats.learningStatus.overall_progress || 0}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-sentinel-text-muted mt-1">
                      <span>Beginner</span>
                      <span>Learning</span>
                      <span>Expert</span>
                    </div>
                  </div>

                  {/* Individual Model Progress */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {aiStats.learningStatus.models && Object.entries(aiStats.learningStatus.models).map(([key, model]: [string, any]) => (
                      <div key={key} className="p-4 rounded-xl bg-sentinel-bg-tertiary">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">{model.name}</span>
                          <span className={`text-xs px-2 py-0.5 rounded ${
                            model.status === 'expert' ? 'bg-sentinel-accent-emerald/20 text-sentinel-accent-emerald' :
                            model.status === 'learning' || model.status === 'active' ? 'bg-sentinel-accent-amber/20 text-sentinel-accent-amber' :
                            'bg-sentinel-accent-crimson/20 text-sentinel-accent-crimson'
                          }`}>
                            {model.status?.toUpperCase()}
                          </span>
                        </div>
                        <div className="h-2 rounded-full bg-sentinel-bg-secondary overflow-hidden mb-2">
                          <div 
                            className={`h-full rounded-full transition-all ${
                              model.status === 'expert' ? 'bg-sentinel-accent-emerald' :
                              model.status === 'learning' || model.status === 'active' ? 'bg-sentinel-accent-amber' :
                              'bg-sentinel-accent-crimson'
                            }`}
                            style={{ width: `${model.progress || 0}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-xs text-sentinel-text-muted">
                          <span>{model.description || '-'}</span>
                          <span className="font-mono">{model.progress?.toFixed(0) || 0}%</span>
                        </div>
                        {model.needed_for_expert > 0 && (
                          <div className="text-xs text-sentinel-accent-amber mt-1">
                            Need {model.needed_for_expert.toLocaleString()} more for expert
                          </div>
                        )}
                        {/* Specific stats */}
                        {model.total_trades !== undefined && (
                          <div className="text-xs text-sentinel-text-muted mt-1">
                            Trades: {model.total_trades.toLocaleString()} | Win Rate: {model.win_rate}%
                          </div>
                        )}
                        {model.trades_collected !== undefined && (
                          <div className="text-xs text-sentinel-text-muted mt-1">
                            Collected: {model.trades_collected.toLocaleString()} trades
                          </div>
                        )}
                        {model.episodes_completed !== undefined && (
                          <div className="text-xs text-sentinel-text-muted mt-1">
                            Episodes: {model.episodes_completed.toLocaleString()} | Strategies: {model.strategies_learned || 0}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Recommendations */}
                  {aiStats.learningStatus.recommendations?.length > 0 && (
                    <div className="mt-4 p-4 rounded-xl bg-sentinel-accent-amber/10 border border-sentinel-accent-amber/30">
                      <h4 className="font-medium text-sentinel-accent-amber mb-2">📋 Recommendations</h4>
                      <ul className="text-sm text-sentinel-text-secondary space-y-1">
                        {aiStats.learningStatus.recommendations.map((rec: string, idx: number) => (
                          <li key={idx}>• {rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Learning Stats */}
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
                <div className="p-4 rounded-2xl glass-card text-center">
                  <div className="text-2xl font-bold text-sentinel-accent-cyan">{aiStats?.qStates || 0}</div>
                  <div className="text-xs text-sentinel-text-muted">Q-States</div>
                </div>
                <div className="p-4 rounded-2xl glass-card text-center">
                  <div className="text-2xl font-bold text-sentinel-accent-amber">{aiStats?.patternsLearned || 0}</div>
                  <div className="text-xs text-sentinel-text-muted">Patterns</div>
                </div>
                <div className="p-4 rounded-2xl glass-card text-center">
                  <div className="text-2xl font-bold text-sentinel-accent-emerald">{aiStats?.marketStates || 0}</div>
                  <div className="text-xs text-sentinel-text-muted">Market States</div>
                </div>
                <div className="p-4 rounded-2xl glass-card text-center">
                  <div className="text-2xl font-bold text-sentinel-accent-crimson">{aiStats?.sentimentStates || 0}</div>
                  <div className="text-xs text-sentinel-text-muted">Sentiment</div>
                </div>
                <div className="p-4 rounded-2xl glass-card text-center">
                  <div className="text-2xl font-bold">{aiStats?.totalTrades || 0}</div>
                  <div className="text-xs text-sentinel-text-muted">Trades</div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div className="p-6 rounded-2xl glass-card">
                  <div className="text-3xl font-bold text-sentinel-accent-cyan">{aiStats?.totalTrades || aiStats?.total_trades || 0}</div>
                  <div className="text-sm text-sentinel-text-muted">Total Trades</div>
                </div>
                <div className="p-6 rounded-2xl glass-card">
                  <div className="text-3xl font-bold text-sentinel-accent-emerald">{(aiStats?.winRate || aiStats?.win_rate || 0).toFixed(1)}%</div>
                  <div className="text-sm text-sentinel-text-muted">Win Rate</div>
                </div>
                <div className="p-6 rounded-2xl glass-card">
                  <div className={`text-3xl font-bold ${(aiStats?.total_pnl || 0) >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}`}>
                    {aiStats?.totalPnl || `€${(aiStats?.total_pnl || 0).toFixed(2)}`}
                  </div>
                  <div className="text-sm text-sentinel-text-muted">Total P&L</div>
                </div>
                <div className="p-6 rounded-2xl glass-card">
                  <div className="text-3xl font-bold text-sentinel-accent-amber">{aiStats?.totalStatesLearned || aiStats?.strategies_learned || 0}</div>
                  <div className="text-sm text-sentinel-text-muted">States Learned</div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="p-6 rounded-2xl glass-card">
                  <h3 className="text-lg font-semibold mb-4">Exploration Rate</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sentinel-text-secondary">AI Exploration</span>
                        <span className="font-medium">{aiStats?.exploration_rate?.toFixed(1) || 10}%</span>
                      </div>
                      <div className="h-3 bg-sentinel-bg-tertiary rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald rounded-full"
                          style={{ width: `${aiStats?.exploration_rate || 10}%` }}
                        />
                      </div>
                      <p className="text-xs text-sentinel-text-muted mt-2">
                        Higher = more experimentation, Lower = more exploitation of learned strategies
                      </p>
                    </div>
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sentinel-text-secondary">Max Drawdown</span>
                        <span className="font-medium text-sentinel-accent-crimson">{aiStats?.max_drawdown?.toFixed(2) || 0}%</span>
                      </div>
                      <div className="h-3 bg-sentinel-bg-tertiary rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-sentinel-accent-crimson rounded-full"
                          style={{ width: `${Math.min(100, (aiStats?.max_drawdown || 0) * 10)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-6 rounded-2xl glass-card">
                  <h3 className="text-lg font-semibold mb-4">Best Performing Strategies</h3>
                  {aiStats?.best_performing?.length ? (
                    <div className="space-y-3">
                      {aiStats.best_performing.map((item: any, idx: number) => (
                        <div key={idx} className="flex items-center justify-between p-3 rounded-xl bg-sentinel-bg-tertiary">
                          <div>
                            <span className="font-medium capitalize">{item.strategy}</span>
                            <span className="text-sentinel-text-muted ml-2 text-sm">in {item.regime.replace('_', ' ')}</span>
                          </div>
                          <div className="text-right">
                            <div className="text-sentinel-accent-emerald font-mono">{item.confidence.toFixed(0)}%</div>
                            <div className="text-xs text-sentinel-text-muted">confidence</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-sentinel-text-muted">
                      <Brain className="w-10 h-10 mx-auto mb-3 opacity-30" />
                      <p>Learning in progress...</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="p-6 rounded-2xl glass-card">
                <h3 className="text-lg font-semibold mb-4">Recent Learning Events</h3>
                {aiStats?.recentDecisions?.length ? (
                  <div className="space-y-3">
                    {aiStats.recentDecisions.map((event: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between p-4 rounded-xl bg-sentinel-bg-tertiary">
                        <div className="flex items-center gap-3">
                          <div className={`w-2 h-2 rounded-full ${
                            event.reward > 0 ? 'bg-sentinel-accent-emerald' :
                            event.reward < 0 ? 'bg-sentinel-accent-crimson' :
                            'bg-sentinel-accent-amber'
                          }`} />
                          <span className="font-mono capitalize">{event.strategy}</span>
                          <span className="text-sentinel-text-secondary text-sm">in {event.regime?.replace('_', ' ')}</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className={`font-mono ${event.reward >= 0 ? 'text-sentinel-accent-emerald' : 'text-sentinel-accent-crimson'}`}>
                            {event.reward >= 0 ? '+' : ''}{event.reward?.toFixed(2)} reward
                          </div>
                          <div className="text-sm text-sentinel-text-muted">
                            Q={event.new_q?.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-sentinel-text-muted">
                    <Brain className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p>No learning events yet - AI will learn from trades</p>
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
