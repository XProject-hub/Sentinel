import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'SENTINEL AI | Autonomous Trading Intelligence',
  description: 'AI-driven capital protection & profit optimization. Your autonomous digital trader.',
  icons: {
    icon: '/favicon.ico',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-sentinel-bg-primary">
        {/* Background Effects */}
        <div className="fixed inset-0 pointer-events-none">
          {/* Grid Pattern */}
          <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-30" />
          
          {/* Ambient Glows */}
          <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-glow-cyan opacity-30" />
          <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-glow-emerald opacity-20" />
        </div>
        
        {/* Content */}
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  )
}

