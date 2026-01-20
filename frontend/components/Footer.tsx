'use client'

import { useEffect, useState } from 'react'

interface VersionInfo {
  version: string
  build_date: string
  git_commit: string
}

export default function Footer() {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    version: 'v3.0',
    build_date: new Date().toLocaleDateString('en-GB').replace(/\//g, ''),
    git_commit: '...'
  })

  useEffect(() => {
    // Fetch version info from API
    const fetchVersion = async () => {
      try {
        const response = await fetch('/api/version')
        if (response.ok) {
          const data = await response.json()
          setVersionInfo({
            version: data.version || 'v3.0',
            build_date: data.build_date || new Date().toLocaleDateString('en-GB').replace(/\//g, ''),
            git_commit: data.git_commit || '...'
          })
        }
      } catch (error) {
        // Use fallback - current date
        const today = new Date()
        const dateStr = `${today.getDate().toString().padStart(2, '0')}${(today.getMonth() + 1).toString().padStart(2, '0')}${today.getFullYear()}`
        setVersionInfo({
          version: 'v3.0',
          build_date: dateStr,
          git_commit: '...'
        })
      }
    }

    fetchVersion()
  }, [])

  return (
    <footer className="fixed bottom-0 left-0 right-0 bg-sentinel-bg-secondary/80 backdrop-blur-sm border-t border-sentinel-border z-50">
      <div className="max-w-7xl mx-auto px-4 py-2 flex items-center justify-between text-xs text-sentinel-text-muted">
        <div className="flex items-center gap-4">
          <span className="font-semibold text-sentinel-text-secondary">SENTINEL AI</span>
          <span className="hidden sm:inline">Autonomous Trading Intelligence</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-sentinel-text-secondary">{versionInfo.version}</span>
          <span className="text-sentinel-border">|</span>
          <span className="font-mono text-sentinel-accent-cyan/60">{versionInfo.build_date}-{versionInfo.git_commit}</span>
        </div>
      </div>
    </footer>
  )
}
