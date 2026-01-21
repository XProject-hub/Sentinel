import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'SENTINEL AI | Autonomous Trading Intelligence',
  description: 'AI-driven autonomous trading with advanced machine learning. Your capital, our intelligence.',
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
      <body className="min-h-screen bg-[#0a0f1a] antialiased">
        {children}
      </body>
    </html>
  )
}
