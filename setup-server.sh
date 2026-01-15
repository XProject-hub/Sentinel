#!/bin/bash
# ============================================
# SENTINEL AI - Complete Server Setup
# Ubuntu 22.04 LTS - 109.104.154.183
# Domain: sentinel.xproject.live
# ============================================

set -e

DOMAIN="sentinel.xproject.live"
PROJECT_DIR="/opt/sentinel"

echo ""
echo "============================================"
echo "   SENTINEL AI - Server Setup Starting"
echo "============================================"
echo ""

# Update system
echo "[1/10] Updating system packages..."
apt update && apt upgrade -y

# Install base dependencies
echo "[2/10] Installing dependencies..."
apt install -y \
    curl wget git unzip htop \
    software-properties-common \
    apt-transport-https \
    ca-certificates gnupg lsb-release \
    fail2ban ufw

# Install Docker
echo "[3/10] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt update
    apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi
systemctl start docker
systemctl enable docker

# Configure firewall
echo "[4/10] Configuring firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Create project directory
echo "[5/10] Setting up project directory..."
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Generate secure passwords
echo "[6/10] Generating secure credentials..."
DB_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
APP_KEY=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Create .env file
cat > .env << EOF
# Generated at $(date)
DB_PASSWORD=$DB_PASSWORD
APP_KEY=base64:$APP_KEY
JWT_SECRET=$JWT_SECRET
ENCRYPTION_KEY=$ENCRYPTION_KEY
DOMAIN=$DOMAIN
EOF

echo ""
echo "============================================"
echo "   Credentials saved to $PROJECT_DIR/.env"
echo "============================================"
echo ""

# Install Certbot
echo "[7/10] Installing Certbot for SSL..."
apt install -y certbot python3-certbot-nginx

# Configure fail2ban
echo "[8/10] Configuring fail2ban..."
cat > /etc/fail2ban/jail.local << EOF
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

# System optimizations
echo "[9/10] Applying system optimizations..."
cat >> /etc/sysctl.conf << EOF

# SENTINEL AI optimizations
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
vm.swappiness = 10
EOF
sysctl -p

# Create deployment script
echo "[10/10] Creating deployment script..."
cat > $PROJECT_DIR/deploy.sh << 'DEPLOY'
#!/bin/bash
cd /opt/sentinel

echo "Pulling latest changes..."
git pull origin main

echo "Building containers..."
docker compose build

echo "Starting services..."
docker compose up -d

echo "Checking status..."
docker compose ps

echo ""
echo "Deployment complete!"
DEPLOY
chmod +x $PROJECT_DIR/deploy.sh

echo ""
echo "============================================"
echo "   SENTINEL AI - Server Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy project files to server:"
echo "   scp -r ./* root@109.104.154.183:/opt/sentinel/"
echo ""
echo "2. SSH to server and run:"
echo "   cd /opt/sentinel"
echo "   docker compose up -d"
echo ""
echo "3. Get SSL certificate:"
echo "   certbot --nginx -d sentinel.xproject.live"
echo ""
echo "4. Access your dashboard at:"
echo "   https://sentinel.xproject.live"
echo ""
echo "============================================"

