'use client'

// Version info - updated on each build
const BUILD_DATE = '20012026'
const GIT_COMMIT = '6a8651f'
const VERSION = 'v3.0'

export default function Footer() {
  return (
    <footer className="fixed bottom-0 left-0 right-0 bg-sentinel-bg-secondary/80 backdrop-blur-sm border-t border-sentinel-border z-50">
      <div className="max-w-7xl mx-auto px-4 py-2 flex items-center justify-between text-xs text-sentinel-text-muted">
        <div className="flex items-center gap-4">
          <span className="font-semibold text-sentinel-text-secondary">SENTINEL AI</span>
          <span className="hidden sm:inline">Autonomous Trading Intelligence</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="font-mono">{VERSION}</span>
          <span className="hidden sm:inline font-mono text-sentinel-accent-cyan/60">{BUILD_DATE}-{GIT_COMMIT}</span>
        </div>
      </div>
    </footer>
  )
}

