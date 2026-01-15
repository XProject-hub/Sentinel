import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Sentinel Dark Theme - Professional & Bold
        sentinel: {
          bg: {
            primary: '#0a0b0d',
            secondary: '#12141a',
            tertiary: '#1a1d26',
            elevated: '#21242f',
          },
          accent: {
            cyan: '#00d4ff',
            emerald: '#00ff88',
            amber: '#ffb800',
            crimson: '#ff3366',
            violet: '#8b5cf6',
          },
          text: {
            primary: '#f0f2f5',
            secondary: '#9ca3b0',
            muted: '#5d6470',
          },
          border: {
            DEFAULT: '#2a2f3a',
            light: '#363d4d',
          }
        }
      },
      fontFamily: {
        display: ['JetBrains Mono', 'monospace'],
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'grid-pattern': 'linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)',
        'glow-cyan': 'radial-gradient(ellipse at center, rgba(0,212,255,0.15) 0%, transparent 70%)',
        'glow-emerald': 'radial-gradient(ellipse at center, rgba(0,255,136,0.1) 0%, transparent 70%)',
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
      boxShadow: {
        'glow-cyan': '0 0 40px rgba(0,212,255,0.3)',
        'glow-emerald': '0 0 40px rgba(0,255,136,0.3)',
        'glow-crimson': '0 0 40px rgba(255,51,102,0.3)',
        'card': '0 4px 24px rgba(0,0,0,0.4)',
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        glow: {
          '0%': { opacity: '0.5' },
          '100%': { opacity: '1' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}
export default config

