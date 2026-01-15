# ============================================
# SENTINEL AI - Upload to Server (Windows)
# ============================================

$SERVER = "109.104.154.183"
$USER = "root"
$DEST = "/opt/sentinel"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   SENTINEL AI - Upload to Server" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if scp is available
if (!(Get-Command scp -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: scp command not found. Install OpenSSH or use WSL." -ForegroundColor Red
    exit 1
}

Write-Host "Uploading files to $USER@$SERVER`:$DEST" -ForegroundColor Yellow
Write-Host ""

# Create destination directory on server
Write-Host "[1/4] Creating directory on server..." -ForegroundColor Green
ssh $USER@$SERVER "mkdir -p $DEST"

# Upload files
Write-Host "[2/4] Uploading docker-compose.yml and configs..." -ForegroundColor Green
scp docker-compose.yml $USER@${SERVER}:$DEST/
scp .env.example $USER@${SERVER}:$DEST/
scp setup-server.sh $USER@${SERVER}:$DEST/
scp README.md $USER@${SERVER}:$DEST/

Write-Host "[3/4] Uploading backend..." -ForegroundColor Green
scp -r backend $USER@${SERVER}:$DEST/

Write-Host "[4/4] Uploading frontend and AI services..." -ForegroundColor Green
scp -r frontend $USER@${SERVER}:$DEST/
scp -r ai-services $USER@${SERVER}:$DEST/
scp -r nginx $USER@${SERVER}:$DEST/
scp -r database $USER@${SERVER}:$DEST/

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "   Upload Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. SSH to server:" -ForegroundColor White
Write-Host "   ssh root@109.104.154.183" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Run setup:" -ForegroundColor White
Write-Host "   cd /opt/sentinel" -ForegroundColor Cyan
Write-Host "   bash setup-server.sh" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Start services:" -ForegroundColor White
Write-Host "   docker compose up -d" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Get SSL:" -ForegroundColor White
Write-Host "   certbot --nginx -d sentinel.xproject.live" -ForegroundColor Cyan
Write-Host ""

