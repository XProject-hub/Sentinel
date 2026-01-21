'use client'

interface LogoProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  showText?: boolean
  className?: string
}

export default function Logo({ size = 'md', showText = true, className = '' }: LogoProps) {
  const sizes = {
    sm: { icon: 28, text: 'text-base', gap: 'gap-2' },
    md: { icon: 36, text: 'text-xl', gap: 'gap-2.5' },
    lg: { icon: 44, text: 'text-2xl', gap: 'gap-3' },
    xl: { icon: 56, text: 'text-3xl', gap: 'gap-3' }
  }

  const { icon, text, gap } = sizes[size]

  return (
    <div className={`flex items-center ${gap} ${className}`}>
      {/* Professional hexagon logo with S */}
      <div 
        className="relative flex items-center justify-center"
        style={{ width: icon, height: icon }}
      >
        <svg 
          viewBox="0 0 100 100" 
          fill="none" 
          className="w-full h-full"
        >
          {/* Hexagon background */}
          <defs>
            <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#06b6d4" />
              <stop offset="50%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
            <linearGradient id="innerGlow" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0" />
            </linearGradient>
          </defs>
          
          {/* Main hexagon */}
          <path
            d="M50 5 L90 27.5 L90 72.5 L50 95 L10 72.5 L10 27.5 Z"
            fill="url(#logoGradient)"
          />
          
          {/* Inner glow */}
          <path
            d="M50 12 L82 31 L82 69 L50 88 L18 69 L18 31 Z"
            fill="url(#innerGlow)"
          />
          
          {/* Bold S letter */}
          <text
            x="50"
            y="68"
            textAnchor="middle"
            fill="white"
            fontSize="52"
            fontWeight="700"
            fontFamily="system-ui, -apple-system, sans-serif"
          >
            S
          </text>
        </svg>
      </div>
      
      {showText && (
        <div className="flex flex-col leading-none">
          <span className={`font-bold text-white tracking-wide ${text}`}>
            SENTINEL
          </span>
          {size !== 'sm' && (
            <span className="text-[10px] text-cyan-400/70 tracking-[0.2em] uppercase mt-0.5">
              AI Trading
            </span>
          )}
        </div>
      )}
    </div>
  )
}
