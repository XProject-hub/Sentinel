'use client'

import { useState, useEffect } from 'react'
import { AlertTriangle, Info, XCircle, Clock, X } from 'lucide-react'

interface MaintenanceData {
  active: boolean
  message: string
  type: 'info' | 'warning' | 'danger'
  scheduled_at?: string
  created_at?: string
}

export default function MaintenanceNotification() {
  const [maintenance, setMaintenance] = useState<MaintenanceData | null>(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    const fetchMaintenance = async () => {
      try {
        const response = await fetch('/ai/admin/maintenance')
        if (response.ok) {
          const data = await response.json()
          if (data.active) {
            setMaintenance(data)
            // Check if user dismissed this notification
            const dismissedAt = localStorage.getItem('maintenance_dismissed')
            if (dismissedAt && data.created_at) {
              // If dismissed after this notification was created, keep dismissed
              if (new Date(dismissedAt) > new Date(data.created_at)) {
                setDismissed(true)
              } else {
                setDismissed(false)
              }
            }
          } else {
            setMaintenance(null)
          }
        }
      } catch (error) {
        console.error('Failed to fetch maintenance notification:', error)
      }
    }

    fetchMaintenance()
    // Refresh every 5 minutes
    const interval = setInterval(fetchMaintenance, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  const handleDismiss = () => {
    setDismissed(true)
    localStorage.setItem('maintenance_dismissed', new Date().toISOString())
  }

  if (!maintenance || !maintenance.active || dismissed) {
    return null
  }

  const typeStyles = {
    info: {
      bg: 'bg-blue-500/10',
      border: 'border-blue-500/30',
      text: 'text-blue-400',
      icon: Info
    },
    warning: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      text: 'text-amber-400',
      icon: AlertTriangle
    },
    danger: {
      bg: 'bg-red-500/10',
      border: 'border-red-500/30',
      text: 'text-red-400',
      icon: XCircle
    }
  }

  const style = typeStyles[maintenance.type] || typeStyles.info
  const Icon = style.icon

  return (
    <div className={`${style.bg} ${style.border} border rounded-xl p-4 mb-4`}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <Icon className={`w-5 h-5 ${style.text} mt-0.5 flex-shrink-0`} />
          <div>
            <p className={`font-semibold ${style.text}`}>
              {maintenance.type === 'danger' ? '⚠️ Important Notice' : 
               maintenance.type === 'warning' ? '📢 Maintenance Notice' : 
               'ℹ️ System Notice'}
            </p>
            <p className="text-sm text-gray-300 mt-1">{maintenance.message}</p>
            {maintenance.scheduled_at && (
              <div className="flex items-center gap-2 mt-2 text-xs text-gray-400">
                <Clock className="w-3 h-3" />
                <span>Scheduled: {maintenance.scheduled_at}</span>
              </div>
            )}
          </div>
        </div>
        <button
          onClick={handleDismiss}
          className="text-gray-500 hover:text-gray-300 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

