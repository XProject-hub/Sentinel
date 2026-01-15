#!/bin/bash
# ============================================
# SENTINEL AI - Complete Fresh Server Setup
# Ubuntu 22.04 LTS
# ============================================

set -e

DOMAIN="sentinel.xproject.live"

echo ""
echo "============================================"
echo "   SENTINEL AI - Fresh Server Setup"
echo "   Ubuntu 22.04 LTS"
echo "============================================"
echo ""

# Update system
echo "[1/8] Updating system..."
apt update && apt upgrade -y

# Install base packages
echo "[2/8] Installing base packages..."
apt install -y \
    curl \
    wget \
    git \
    unzip \
    htop \
    nano \
    ufw \
    fail2ban \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common

# Install Docker
echo "[3/8] Installing Docker..."
# Remove old versions
apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Add Docker's official GPG key
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start Docker
systemctl start docker
systemctl enable docker

# Test Docker
echo "[4/8] Testing Docker..."
docker --version
docker compose version

# Configure firewall
echo "[5/8] Configuring firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Configure fail2ban
echo "[6/8] Configuring fail2ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
EOF
systemctl restart fail2ban

# Install Certbot for SSL
echo "[7/8] Installing Certbot..."
apt install -y certbot python3-certbot-nginx nginx

# System optimizations
echo "[8/8] Applying system optimizations..."
cat >> /etc/sysctl.conf << 'EOF'

# SENTINEL AI optimizations
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
vm.swappiness = 10
fs.file-max = 2097152
EOF
sysctl -p 2>/dev/null || true

# Increase limits
cat >> /etc/security/limits.conf << 'EOF'
* soft nofile 65535
* hard nofile 65535
EOF

# Generate .env with secure passwords
echo "[9/9] Generating .env configuration..."
cd /opt/sentinel
chmod +x setup-env.sh
./setup-env.sh

echo ""
echo "============================================"
echo "   Installation Complete!"
echo "============================================"
echo ""
echo "Docker: $(docker --version)"
echo "Compose: $(docker compose version)"
echo ""
echo "Starting SENTINEL AI..."
echo ""

# Start services
cd /opt/sentinel
docker compose up -d

echo ""
echo "============================================"
echo "   SENTINEL AI is starting!"
echo "============================================"
echo ""
echo "Wait 2-3 minutes for containers to build."
echo ""
echo "Check status:  docker compose ps"
echo "View logs:     docker compose logs -f"
echo ""
echo "Get SSL certificate:"
echo "  certbot --nginx -d $DOMAIN"
echo ""
echo "Dashboard: https://$DOMAIN"
echo ""
echo "============================================"

