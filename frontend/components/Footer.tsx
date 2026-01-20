'use client'

import { useEffect, useState } from 'react'

interface VersionInfo {
  build_date: string
  git_commit: string
}

export default function Footer() {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    build_date: '',
    git_commit: ''
  })

  useEffect(() => {
    // Get current date
    const today = new Date()
    const dateStr = `${today.getDate().toString().padStart(2, '0')}.${(today.getMonth() + 1).toString().padStart(2, '0')}.${today.getFullYear()}`
    
    // Fetch git commit from API
    const fetchVersion = async () => {
      try {
        const response = await fetch('/api/version')
        if (response.ok) {
          const data = await response.json()
          setVersionInfo({
            build_date: dateStr,
            git_commit: data.git_commit?.substring(0, 7) || ''
          })
        } else {
          setVersionInfo({
            build_date: dateStr,
            git_commit: ''
          })
        }
      } catch (error) {
        setVersionInfo({
          build_date: dateStr,
          git_commit: ''
        })
      }
    }

    fetchVersion()
  }, [])

  return (
    <footer className="fixed bottom-0 left-0 right-0 bg-sentinel-bg-secondary/90 backdrop-blur-sm border-t border-sentinel-border z-50">
      <div className="w-full px-6 py-2 flex items-center justify-between text-xs text-sentinel-text-muted">
        <div className="flex items-center gap-4">
          <span className="font-semibold text-sentinel-text-secondary">SENTINEL AI</span>
          <span className="hidden sm:inline text-sentinel-text-muted">Autonomous Trading Intelligence</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sentinel-text-secondary">Version:</span>
          <span className="font-mono text-sentinel-accent-cyan">
            {versionInfo.build_date}{versionInfo.git_commit ? ` - ${versionInfo.git_commit}` : ''}
          </span>
        </div>
      </div>
    </footer>
  )
}
