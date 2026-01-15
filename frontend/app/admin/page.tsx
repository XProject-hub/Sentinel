'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Shield, Users, Activity, Brain, Server, 
  TrendingUp, BarChart3, Database, Cpu,
  Clock, Globe, Zap, AlertTriangle,
  ChevronRight, RefreshCw, Settings,
  UserPlus, UserCheck, UserX, Bot,
  Layers, Network, Terminal
} from 'lucide-react'
import Link from 'next/link'

// Mock admin data
const mockAdminData = {
  overview: {
    totalUsers: 12459,
    activeUsers: 3847,
    newUsersToday: 156,
    totalRevenue: 847523.45,
    revenueToday: 12847.32,
    activeSubscriptions: 8934,
  },
  traffic: {
    requestsToday: 2847563,
    requestsPerSecond: 847,
    avgResponseTime: 23,
    errorRate: 0.02,
    bandwidthUsed: '847 GB',
    uniqueVisitors: 15847,
  },
  ai: {
    totalPredictions: 15847234,
    accuracy: 94.7,
    modelsActive: 5,
    learningCycles: 847234,
    dataPointsProcessed: 847523847,
    lastTraining: '2 hours ago',
    sentimentAccuracy: 91.3,
    regimeDetectionAccuracy: 96.2,
    strategyWinRate: 78.4,
  },
  system: {
    cpuUsage: 34,
    memoryUsage: 67,
    diskUsage: 45,
    uptime: '99.99%',
    containers: 10,
    healthyContainers: 9,
  },
  recentUsers: [
    { id: 1, name: 'John Doe', email: 'john@example.com', plan: 'Professional', status: 'active', joined: '2 hours ago' },
    { id: 2, name: 'Jane Smith', email: 'jane@example.com', plan: 'Starter', status: 'active', joined: '5 hours ago' },
    { id: 3, name: 'Bob Wilson', email: 'bob@example.com', plan: 'Enterprise', status: 'active', joined: '1 day ago' },
    { id: 4, name: 'Alice Brown', email: 'alice@example.com', plan: 'Professional', status: 'pending', joined: '1 day ago' },
    { id: 5, name: 'Charlie Davis', email: 'charlie@example.com', plan: 'Starter', status: 'active', joined: '2 days ago' },
  ],
  aiModels: [
    { name: 'Market Regime Detector', version: 'v2.4.1', accuracy: 96.2, status: 'active', lastUpdate: '1 hour ago', predictions: 234567 },
    { name: 'Sentiment Analyzer', version: 'v3.1.0', accuracy: 91.3, status: 'active', lastUpdate: '30 min ago', predictions: 567890 },
    { name: 'Strategy Selector', version: 'v1.8.2', accuracy: 78.4, status: 'active', lastUpdate: '2 hours ago', predictions: 123456 },
    { name: 'Risk Predictor', version: 'v2.0.0', accuracy: 89.7, status: 'training', lastUpdate: '5 min ago', predictions: 345678 },
    { name: 'Entry Optimizer', version: 'v1.5.3', accuracy: 82.1, status: 'active', lastUpdate: '4 hours ago', predictions: 456789 },
  ],
  learningProgress: [
    { time: '00:00', dataPoints: 12000, accuracy: 92.1 },
    { time: '04:00', dataPoints: 18000, accuracy: 92.8 },
    { time: '08:00', dataPoints: 25000, accuracy: 93.4 },
    { time: '12:00', dataPoints: 34000, accuracy: 94.1 },
    { time: '16:00', dataPoints: 42000, accuracy: 94.5 },
    { time: '20:00', dataPoints: 51000, accuracy: 94.7 },
  ]
}

export default function AdminDashboard() {
  const [data, setData] = useState(mockAdminData)
  const [activeTab, setActiveTab] = useState('overview')
  const [isRefreshing, setIsRefreshing] = useState(false)

  const refreshData = () => {
    setIsRefreshing(true)
    setTimeout(() => setIsRefreshing(false), 1000)
  }

  return (
    <div className="min-h-screen bg-sentinel-bg-primary">
      {/* Admin Navigation */}
      <nav className="sticky top-0 z-50 glass-card border-b border-sentinel-border">
        <div className="max-w-[1800px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-sentinel-accent-crimson to-sentinel-accent-amber flex items-center justify-center">
                <Shield className="w-6 h-6 text-sentinel-bg-primary" strokeWidth={2.5} />
              </div>
              <div>
                <span className="font-display font-bold text-lg">SENTINEL</span>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-sentinel-accent-crimson/20 text-sentinel-accent-crimson">ADMIN</span>
                </div>
              </div>
            </Link>
          </div>

          <div className="flex items-center gap-2">
            {['overview', 'users', 'ai', 'traffic', 'system'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all capitalize ${
                  activeTab === tab 
                    ? 'bg-sentinel-accent-cyan text-sentinel-bg-primary' 
                    : 'text-sentinel-text-secondary hover:text-sentinel-text-primary hover:bg-sentinel-bg-tertiary'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-4">
            <button 
              onClick={refreshData}
              className={`p-2 rounded-lg hover:bg-sentinel-bg-tertiary transition-colors ${isRefreshing ? 'animate-spin' : ''}`}
            >
              <RefreshCw className="w-5 h-5 text-sentinel-text-secondary" />
            </button>
            <Link href="/dashboard" className="px-4 py-2 rounded-lg bg-sentinel-bg-tertiary text-sentinel-text-secondary hover:text-sentinel-text-primary text-sm">
              Exit Admin
            </Link>
          </div>
        </div>
      </nav>

      <main className="max-w-[1800px] mx-auto px-6 py-8">
        {activeTab === 'overview' && <OverviewTab data={data} />}
        {activeTab === 'users' && <UsersTab data={data} />}
        {activeTab === 'ai' && <AITab data={data} />}
        {activeTab === 'traffic' && <TrafficTab data={data} />}
        {activeTab === 'system' && <SystemTab data={data} />}
      </main>

      {/* Footer */}
      <footer className="border-t border-sentinel-border mt-8">
        <div className="max-w-[1800px] mx-auto px-6 py-4 flex items-center justify-between text-sm text-sentinel-text-muted">
          <span>SENTINEL AI - Admin Panel</span>
          <span>Developed by NoLimitDevelopments</span>
        </div>
      </footer>
    </div>
  )
}

// Overview Tab
function OverviewTab({ data }: { data: typeof mockAdminData }) {
  return (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard icon={<Users />} label="Total Users" value={data.overview.totalUsers.toLocaleString()} color="cyan" />
        <StatCard icon={<UserCheck />} label="Active Now" value={data.overview.activeUsers.toLocaleString()} color="emerald" />
        <StatCard icon={<UserPlus />} label="New Today" value={`+${data.overview.newUsersToday}`} color="amber" />
        <StatCard icon={<TrendingUp />} label="Revenue Today" value={`$${data.overview.revenueToday.toLocaleString()}`} color="emerald" />
        <StatCard icon={<BarChart3 />} label="Total Revenue" value={`$${(data.overview.totalRevenue / 1000).toFixed(0)}K`} color="violet" />
        <StatCard icon={<Zap />} label="Subscriptions" value={data.overview.activeSubscriptions.toLocaleString()} color="cyan" />
      </div>

      {/* Main Content */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Recent Users */}
        <div className="lg:col-span-2 p-6 rounded-2xl glass-card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold">Recent Users</h2>
            <Link href="/admin/users" className="text-sm text-sentinel-accent-cyan hover:underline flex items-center gap-1">
              View All <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="space-y-3">
            {data.recentUsers.map((user) => (
              <div key={user.id} className="flex items-center justify-between p-3 rounded-xl bg-sentinel-bg-tertiary/50">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-sentinel-accent-cyan/20 flex items-center justify-center">
                    <span className="text-sentinel-accent-cyan font-semibold">{user.name[0]}</span>
                  </div>
                  <div>
                    <div className="font-medium">{user.name}</div>
                    <div className="text-sm text-sentinel-text-muted">{user.email}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`px-2 py-1 rounded text-xs font-medium ${
                    user.plan === 'Enterprise' ? 'bg-sentinel-accent-violet/20 text-sentinel-accent-violet' :
                    user.plan === 'Professional' ? 'bg-sentinel-accent-cyan/20 text-sentinel-accent-cyan' :
                    'bg-sentinel-bg-elevated text-sentinel-text-secondary'
                  }`}>{user.plan}</div>
                  <div className="text-xs text-sentinel-text-muted mt-1">{user.joined}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* AI Status */}
        <div className="p-6 rounded-2xl glass-card">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 rounded-xl bg-sentinel-accent-violet/10">
              <Brain className="w-6 h-6 text-sentinel-accent-violet" />
            </div>
            <h2 className="text-lg font-semibold">AI Status</h2>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-sentinel-text-secondary">Total Predictions</span>
              <span className="font-mono">{(data.ai.totalPredictions / 1000000).toFixed(1)}M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sentinel-text-secondary">Overall Accuracy</span>
              <span className="font-mono text-sentinel-accent-emerald">{data.ai.accuracy}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sentinel-text-secondary">Models Active</span>
              <span className="font-mono">{data.ai.modelsActive}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sentinel-text-secondary">Learning Cycles</span>
              <span className="font-mono">{(data.ai.learningCycles / 1000).toFixed(0)}K</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sentinel-text-secondary">Data Processed</span>
              <span className="font-mono">{(data.ai.dataPointsProcessed / 1000000).toFixed(0)}M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sentinel-text-secondary">Last Training</span>
              <span className="font-mono text-sentinel-accent-cyan">{data.ai.lastTraining}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Users Tab
function UsersTab({ data }: { data: typeof mockAdminData }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={<Users />} label="Total Users" value={data.overview.totalUsers.toLocaleString()} color="cyan" />
        <StatCard icon={<UserCheck />} label="Active" value={data.overview.activeUsers.toLocaleString()} color="emerald" />
        <StatCard icon={<UserPlus />} label="New Today" value={`+${data.overview.newUsersToday}`} color="amber" />
        <StatCard icon={<UserX />} label="Inactive" value={(data.overview.totalUsers - data.overview.activeUsers).toLocaleString()} color="crimson" />
      </div>

      <div className="p-6 rounded-2xl glass-card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold">All Users</h2>
          <div className="flex items-center gap-3">
            <input 
              type="text" 
              placeholder="Search users..." 
              className="px-4 py-2 rounded-lg bg-sentinel-bg-tertiary border border-sentinel-border focus:border-sentinel-accent-cyan focus:outline-none text-sm"
            />
            <button className="px-4 py-2 rounded-lg bg-sentinel-accent-cyan text-sentinel-bg-primary font-medium text-sm">
              Add User
            </button>
          </div>
        </div>

        <table className="w-full">
          <thead>
            <tr className="text-left text-sm text-sentinel-text-muted border-b border-sentinel-border">
              <th className="pb-4 font-medium">User</th>
              <th className="pb-4 font-medium">Email</th>
              <th className="pb-4 font-medium">Plan</th>
              <th className="pb-4 font-medium">Status</th>
              <th className="pb-4 font-medium">Joined</th>
              <th className="pb-4 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {data.recentUsers.map((user) => (
              <tr key={user.id} className="border-b border-sentinel-border/50">
                <td className="py-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-sentinel-accent-cyan/20 flex items-center justify-center">
                      <span className="text-sentinel-accent-cyan text-sm font-semibold">{user.name[0]}</span>
                    </div>
                    <span className="font-medium">{user.name}</span>
                  </div>
                </td>
                <td className="py-4 text-sentinel-text-secondary">{user.email}</td>
                <td className="py-4">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    user.plan === 'Enterprise' ? 'bg-sentinel-accent-violet/20 text-sentinel-accent-violet' :
                    user.plan === 'Professional' ? 'bg-sentinel-accent-cyan/20 text-sentinel-accent-cyan' :
                    'bg-sentinel-bg-elevated text-sentinel-text-secondary'
                  }`}>{user.plan}</span>
                </td>
                <td className="py-4">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    user.status === 'active' ? 'bg-sentinel-accent-emerald/20 text-sentinel-accent-emerald' :
                    'bg-sentinel-accent-amber/20 text-sentinel-accent-amber'
                  }`}>{user.status}</span>
                </td>
                <td className="py-4 text-sentinel-text-muted">{user.joined}</td>
                <td className="py-4">
                  <button className="text-sentinel-accent-cyan hover:underline text-sm">Edit</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// AI Tab
function AITab({ data }: { data: typeof mockAdminData }) {
  return (
    <div className="space-y-6">
      {/* AI Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={<Brain />} label="Predictions" value={`${(data.ai.totalPredictions / 1000000).toFixed(1)}M`} color="violet" />
        <StatCard icon={<Zap />} label="Accuracy" value={`${data.ai.accuracy}%`} color="emerald" />
        <StatCard icon={<Bot />} label="Models Active" value={data.ai.modelsActive.toString()} color="cyan" />
        <StatCard icon={<Database />} label="Data Points" value={`${(data.ai.dataPointsProcessed / 1000000).toFixed(0)}M`} color="amber" />
      </div>

      {/* AI Models */}
      <div className="p-6 rounded-2xl glass-card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold">AI Models</h2>
          <button className="px-4 py-2 rounded-lg bg-sentinel-accent-violet text-white font-medium text-sm flex items-center gap-2">
            <Bot className="w-4 h-4" /> Train All Models
          </button>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.aiModels.map((model, idx) => (
            <motion.div
              key={model.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="p-4 rounded-xl bg-sentinel-bg-tertiary border border-sentinel-border"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-sentinel-accent-violet" />
                  <span className="font-medium text-sm">{model.name}</span>
                </div>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  model.status === 'active' ? 'bg-sentinel-accent-emerald/20 text-sentinel-accent-emerald' :
                  'bg-sentinel-accent-amber/20 text-sentinel-accent-amber'
                }`}>{model.status}</span>
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-sentinel-text-muted">Version</span>
                  <span className="font-mono">{model.version}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sentinel-text-muted">Accuracy</span>
                  <span className="font-mono text-sentinel-accent-emerald">{model.accuracy}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sentinel-text-muted">Predictions</span>
                  <span className="font-mono">{(model.predictions / 1000).toFixed(0)}K</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sentinel-text-muted">Last Update</span>
                  <span className="font-mono text-sentinel-accent-cyan">{model.lastUpdate}</span>
                </div>
              </div>

              {/* Accuracy Bar */}
              <div className="mt-3">
                <div className="h-1.5 rounded-full bg-sentinel-bg-primary overflow-hidden">
                  <div 
                    className="h-full rounded-full bg-gradient-to-r from-sentinel-accent-cyan to-sentinel-accent-emerald"
                    style={{ width: `${model.accuracy}%` }}
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Learning Progress */}
      <div className="p-6 rounded-2xl glass-card">
        <h2 className="text-lg font-semibold mb-6">Learning Progress (24h)</h2>
        <div className="grid grid-cols-6 gap-4">
          {data.learningProgress.map((point, idx) => (
            <div key={idx} className="text-center">
              <div className="h-32 flex items-end justify-center mb-2">
                <div 
                  className="w-8 rounded-t bg-gradient-to-t from-sentinel-accent-violet to-sentinel-accent-cyan"
                  style={{ height: `${(point.accuracy - 90) * 20}%` }}
                />
              </div>
              <div className="text-xs text-sentinel-text-muted">{point.time}</div>
              <div className="text-sm font-mono text-sentinel-accent-emerald">{point.accuracy}%</div>
              <div className="text-xs text-sentinel-text-muted">{(point.dataPoints / 1000).toFixed(0)}K pts</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Traffic Tab
function TrafficTab({ data }: { data: typeof mockAdminData }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard icon={<Globe />} label="Requests Today" value={`${(data.traffic.requestsToday / 1000000).toFixed(1)}M`} color="cyan" />
        <StatCard icon={<Zap />} label="Req/Second" value={data.traffic.requestsPerSecond.toString()} color="emerald" />
        <StatCard icon={<Clock />} label="Avg Response" value={`${data.traffic.avgResponseTime}ms`} color="amber" />
        <StatCard icon={<AlertTriangle />} label="Error Rate" value={`${data.traffic.errorRate}%`} color="crimson" />
        <StatCard icon={<Database />} label="Bandwidth" value={data.traffic.bandwidthUsed} color="violet" />
        <StatCard icon={<Users />} label="Unique Visitors" value={data.traffic.uniqueVisitors.toLocaleString()} color="cyan" />
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="p-6 rounded-2xl glass-card">
          <h2 className="text-lg font-semibold mb-6">Request Distribution</h2>
          <div className="space-y-4">
            {[
              { endpoint: '/api/dashboard', requests: 847234, percentage: 35 },
              { endpoint: '/api/trades', requests: 523456, percentage: 22 },
              { endpoint: '/ai/market', requests: 423123, percentage: 18 },
              { endpoint: '/ai/sentiment', requests: 312456, percentage: 13 },
              { endpoint: '/ws', requests: 287456, percentage: 12 },
            ].map((item) => (
              <div key={item.endpoint}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-mono text-sentinel-text-secondary">{item.endpoint}</span>
                  <span className="text-sentinel-text-muted">{(item.requests / 1000).toFixed(0)}K</span>
                </div>
                <div className="h-2 rounded-full bg-sentinel-bg-tertiary overflow-hidden">
                  <div 
                    className="h-full rounded-full bg-sentinel-accent-cyan"
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-6 rounded-2xl glass-card">
          <h2 className="text-lg font-semibold mb-6">Geographic Distribution</h2>
          <div className="space-y-3">
            {[
              { country: 'United States', visitors: 4523, flag: 'US' },
              { country: 'United Kingdom', visitors: 2847, flag: 'GB' },
              { country: 'Germany', visitors: 2156, flag: 'DE' },
              { country: 'Japan', visitors: 1847, flag: 'JP' },
              { country: 'Australia', visitors: 1234, flag: 'AU' },
              { country: 'Other', visitors: 3240, flag: 'XX' },
            ].map((item) => (
              <div key={item.country} className="flex items-center justify-between p-2 rounded-lg bg-sentinel-bg-tertiary/50">
                <span>{item.country}</span>
                <span className="font-mono text-sentinel-text-secondary">{item.visitors.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// System Tab
function SystemTab({ data }: { data: typeof mockAdminData }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard icon={<Cpu />} label="CPU Usage" value={`${data.system.cpuUsage}%`} color="cyan" />
        <StatCard icon={<Layers />} label="Memory" value={`${data.system.memoryUsage}%`} color="amber" />
        <StatCard icon={<Database />} label="Disk" value={`${data.system.diskUsage}%`} color="emerald" />
        <StatCard icon={<Clock />} label="Uptime" value={data.system.uptime} color="violet" />
        <StatCard icon={<Server />} label="Containers" value={data.system.containers.toString()} color="cyan" />
        <StatCard icon={<Activity />} label="Healthy" value={`${data.system.healthyContainers}/${data.system.containers}`} color="emerald" />
      </div>

      <div className="p-6 rounded-2xl glass-card">
        <h2 className="text-lg font-semibold mb-6">Container Status</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { name: 'sentinel_nginx', status: 'running', cpu: 2, memory: 64 },
            { name: 'sentinel_frontend', status: 'running', cpu: 8, memory: 256 },
            { name: 'sentinel_backend', status: 'running', cpu: 5, memory: 128 },
            { name: 'sentinel_ai', status: 'running', cpu: 45, memory: 2048 },
            { name: 'sentinel_postgres', status: 'running', cpu: 3, memory: 512 },
            { name: 'sentinel_redis', status: 'running', cpu: 1, memory: 64 },
            { name: 'sentinel_clickhouse', status: 'running', cpu: 8, memory: 1024 },
            { name: 'sentinel_kafka', status: 'restarting', cpu: 0, memory: 0 },
            { name: 'sentinel_zookeeper', status: 'running', cpu: 2, memory: 256 },
          ].map((container) => (
            <div key={container.name} className="p-4 rounded-xl bg-sentinel-bg-tertiary border border-sentinel-border">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-sentinel-accent-cyan" />
                  <span className="font-mono text-sm">{container.name}</span>
                </div>
                <span className={`w-2 h-2 rounded-full ${
                  container.status === 'running' ? 'bg-sentinel-accent-emerald' : 'bg-sentinel-accent-amber animate-pulse'
                }`} />
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <div className="text-sentinel-text-muted text-xs">CPU</div>
                  <div className="font-mono">{container.cpu}%</div>
                </div>
                <div>
                  <div className="text-sentinel-text-muted text-xs">Memory</div>
                  <div className="font-mono">{container.memory}MB</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Stat Card Component
function StatCard({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string; color: 'cyan' | 'emerald' | 'amber' | 'crimson' | 'violet' }) {
  const colorMap = {
    cyan: 'bg-sentinel-accent-cyan/10 text-sentinel-accent-cyan',
    emerald: 'bg-sentinel-accent-emerald/10 text-sentinel-accent-emerald',
    amber: 'bg-sentinel-accent-amber/10 text-sentinel-accent-amber',
    crimson: 'bg-sentinel-accent-crimson/10 text-sentinel-accent-crimson',
    violet: 'bg-sentinel-accent-violet/10 text-sentinel-accent-violet',
  }

  return (
    <div className="p-4 rounded-xl glass-card">
      <div className={`w-10 h-10 rounded-lg ${colorMap[color]} flex items-center justify-center mb-3`}>
        {icon}
      </div>
      <div className="text-2xl font-display font-bold">{value}</div>
      <div className="text-sm text-sentinel-text-muted">{label}</div>
    </div>
  )
}

