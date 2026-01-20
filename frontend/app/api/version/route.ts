import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Try to fetch from AI services
    const response = await fetch('http://ai-services:8000/ai/admin/version', {
      cache: 'no-store',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
  } catch (error) {
    // AI services not available, return fallback
  }

  // Fallback version info
  const now = new Date()
  const buildDate = `${now.getDate().toString().padStart(2, '0')}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getFullYear()}`
  
  return NextResponse.json({
    version: 'v3.0',
    build_date: buildDate,
    git_commit: process.env.GIT_COMMIT || 'local',
    full_version: `v3.0-${buildDate}`
  })
}

