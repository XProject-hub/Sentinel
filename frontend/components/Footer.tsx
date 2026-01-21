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
    // Get current date in DDMMYYYY format
    const today = new Date()
    const dateStr = `${today.getDate().toString().padStart(2, '0')}${(today.getMonth() + 1).toString().padStart(2, '0')}${today.getFullYear()}`
    
    const fetchVersion = async () => {
      const envCommit = process.env.NEXT_PUBLIC_GIT_COMMIT
      
      try {
        const response = await fetch('/ai/admin/version')
        if (response.ok) {
          const data = await response.json()
          const commit = data.git_commit?.substring(0, 7) || envCommit?.substring(0, 7) || ''
          setVersionInfo({
            build_date: data.build_date || dateStr,
            git_commit: commit !== 'unknown' ? commit : ''
          })
        } else {
          setVersionInfo({
            build_date: dateStr,
            git_commit: envCommit?.substring(0, 7) || ''
          })
        }
      } catch (error) {
        setVersionInfo({
          build_date: dateStr,
          git_commit: envCommit?.substring(0, 7) || ''
        })
      }
    }

    fetchVersion()
  }, [])

  return (
    <footer className="fixed bottom-0 left-0 right-0 bg-[#0a0f1a]/95 backdrop-blur-xl border-t border-white/5 z-50">
      <div className="w-full px-6 py-2.5 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Mini logo */}
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 flex items-center justify-center">
              <svg viewBox="0 0 100 100" className="w-full h-full">
                <defs>
                  <linearGradient id="footerGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                  </linearGradient>
                </defs>
                <path
                  d="M50 5 L90 27.5 L90 72.5 L50 95 L10 72.5 L10 27.5 Z"
                  fill="url(#footerGradient)"
                />
                <text x="50" y="68" textAnchor="middle" fill="white" fontSize="52" fontWeight="700" fontFamily="system-ui">S</text>
              </svg>
            </div>
            <span className="text-sm font-semibold text-white">SENTINEL</span>
          </div>
          <span className="hidden md:inline text-xs text-gray-500">|</span>
          <span className="hidden md:inline text-xs text-gray-500">Autonomous Trading Intelligence</span>
        </div>
        
        <div className="flex items-center gap-4">
          <span className="hidden sm:inline text-xs text-gray-600">NoLimitDevelopments</span>
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-gray-500">v</span>
            <span className="text-xs font-mono text-cyan-400">
              {versionInfo.build_date}{versionInfo.git_commit ? `-${versionInfo.git_commit}` : ''}
            </span>
          </div>
        </div>
      </div>
    </footer>
  )
}
