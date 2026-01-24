#!/bin/bash
# Quick AI services update script
# Usage: ./update-ai.sh

cd /opt/sentinel

echo "📥 Pulling latest code..."
git pull

echo "🏗️ Building AI services..."
export GIT_COMMIT=$(git rev-parse --short HEAD)
docker compose build ai-services

echo "🚀 Restarting AI services..."
docker compose up -d ai-services

echo "📋 Showing logs (Ctrl+C to exit)..."
docker compose logs -f sentinel_ai
