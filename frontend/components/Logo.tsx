'use client'

interface LogoProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  showText?: boolean
  className?: string
}

export default function Logo({ size = 'md', showText = true, className = '' }: LogoProps) {
  const sizes = {
    sm: { icon: 32, text: 'text-lg' },
    md: { icon: 40, text: 'text-xl' },
    lg: { icon: 48, text: 'text-2xl' },
    xl: { icon: 64, text: 'text-3xl' }
  }

  const { icon, text } = sizes[size]

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* Custom Logo - Stylized S with circuit pattern */}
      <div 
        className="relative flex items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 via-blue-500 to-violet-600 shadow-lg shadow-cyan-500/25"
        style={{ width: icon, height: icon }}
      >
        {/* Inner glow */}
        <div className="absolute inset-1 rounded-lg bg-gradient-to-br from-cyan-400/20 to-transparent" />
        
        {/* S Letter with tech style */}
        <svg 
          viewBox="0 0 24 24" 
          fill="none" 
          className="relative z-10"
          style={{ width: icon * 0.6, height: icon * 0.6 }}
        >
          {/* Circuit dots */}
          <circle cx="4" cy="6" r="1" fill="rgba(255,255,255,0.4)" />
          <circle cx="20" cy="18" r="1" fill="rgba(255,255,255,0.4)" />
          <circle cx="4" cy="12" r="0.5" fill="rgba(255,255,255,0.3)" />
          <circle cx="20" cy="12" r="0.5" fill="rgba(255,255,255,0.3)" />
          
          {/* Stylized S */}
          <path
            d="M17 6.5C17 6.5 15.5 4 12 4C8.5 4 6 6 6 8.5C6 11 8 12 12 12.5C16 13 18 14 18 16.5C18 19 15.5 20 12 20C8.5 20 7 17.5 7 17.5"
            stroke="white"
            strokeWidth="2.5"
            strokeLinecap="round"
            fill="none"
          />
          
          {/* Tech accent lines */}
          <path
            d="M4 6 L6 8.5"
            stroke="rgba(255,255,255,0.4)"
            strokeWidth="1"
            strokeLinecap="round"
          />
          <path
            d="M20 18 L18 16.5"
            stroke="rgba(255,255,255,0.4)"
            strokeWidth="1"
            strokeLinecap="round"
          />
        </svg>
        
        {/* Pulse effect */}
        <div className="absolute inset-0 rounded-xl bg-cyan-400/20 animate-pulse" style={{ animationDuration: '3s' }} />
      </div>
      
      {showText && (
        <span className={`font-bold text-white tracking-tight ${text}`}>
          SENTINEL
        </span>
      )}
    </div>
  )
}

