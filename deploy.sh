#!/bin/bash
# ============================================
# SENTINEL AI - Server Deployment Script
# Ubuntu 22.04 LTS
# ============================================

set -e

DOMAIN="sentinel.xproject.live"
SERVER_IP="109.104.154.183"
PROJECT_DIR="/opt/sentinel"

echo "============================================"
echo "SENTINEL AI - Deployment Starting"
echo "============================================"

# Update system
apt update && apt upgrade -y

# Install dependencies
apt install -y \
    curl \
    wget \
    git \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
systemctl start docker
systemctl enable docker

# Install Node.js 20 LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Install PHP 8.3 and Composer
add-apt-repository ppa:ondrej/php -y
apt update
apt install -y php8.3-cli php8.3-fpm php8.3-mbstring php8.3-xml php8.3-curl php8.3-pgsql php8.3-redis php8.3-zip php8.3-bcmath

# Install Composer
curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

# Install Python 3.11
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Create project directory
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Generate secure password
DB_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)

# Create .env file
cat > .env << EOF
DB_PASSWORD=$DB_PASSWORD
APP_KEY=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)
EOF

# Install Certbot for SSL
apt install -y certbot python3-certbot-nginx

# Configure firewall
ufw allow 22
ufw allow 80
ufw allow 443
ufw --force enable

echo "============================================"
echo "SENTINEL AI - Base Setup Complete"
echo "============================================"
echo "Domain: $DOMAIN"
echo "Server: $SERVER_IP"
echo "Project: $PROJECT_DIR"
echo "============================================"
echo "Next: Copy project files and run docker-compose up -d"
echo "============================================"

