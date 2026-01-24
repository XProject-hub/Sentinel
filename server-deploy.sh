#!/bin/bash
# ============================================
# SENTINEL AI - Auto Deploy Script
# Run this on server after git pull
# ============================================

set -e

cd /opt/sentinel

echo ""
echo "============================================"
echo "   SENTINEL AI - Deploying Updates"
echo "============================================"
echo ""

# Pull latest code
echo "[1/5] Pulling latest code..."
git pull origin main

# Export git commit for version tracking
export GIT_COMMIT=$(git rev-parse --short HEAD)
echo "Git commit: $GIT_COMMIT"

# Check if .env exists
if [ ! -f .env ]; then
    echo "[!] Creating .env from template..."
    cp env.template .env
    echo "[!] IMPORTANT: Edit .env with your credentials!"
    echo "    nano .env"
    exit 1
fi

# Rebuild containers if Dockerfiles changed
echo "[2/5] Building containers..."
docker compose build --parallel

# Restart services
echo "[3/5] Restarting services..."
docker compose down
docker compose up -d

# Wait for services to be healthy
echo "[4/5] Waiting for services..."
sleep 10

# Check status
echo "[5/5] Checking service status..."
docker compose ps

echo ""
echo "============================================"
echo "   Deployment Complete!"
echo "============================================"
echo ""
echo "Services running at:"
echo "  - Frontend: https://sentinel.xproject.live"
echo "  - API:      https://sentinel.xproject.live/api"
echo "  - AI:       https://sentinel.xproject.live/ai"
echo ""

