# SENTINEL AI - Deployment Guide

## First Time Setup (Server)

```bash
# 1. SSH to server
ssh root@109.104.154.183

# 2. Install prerequisites
apt update && apt install -y git docker.io docker-compose-plugin

# 3. Clone repository
cd /opt
git clone https://github.com/XProject-hub/Sentinel.git sentinel
cd sentinel

# 4. Setup environment
cp env.template .env
nano .env   # Fill in your credentials

# 5. Start services
docker compose up -d

# 6. Get SSL certificate
certbot --nginx -d sentinel.xproject.live
```

## Updating (After Changes)

**On Server - Just run:**
```bash
cd /opt/sentinel
./server-deploy.sh
```

Or manually:
```bash
cd /opt/sentinel
git pull origin main
docker compose build
docker compose up -d
```

## Quick Commands

```bash
# View logs
docker compose logs -f

# Restart specific service
docker compose restart frontend
docker compose restart ai-services
docker compose restart backend

# Stop everything
docker compose down

# Full rebuild
docker compose build --no-cache
docker compose up -d
```

## Troubleshooting

```bash
# Check container status
docker compose ps

# View specific logs
docker compose logs frontend
docker compose logs ai-services

# Enter container
docker compose exec backend bash
docker compose exec ai-services bash

# Check disk space
df -h

# Check memory
free -h
```

