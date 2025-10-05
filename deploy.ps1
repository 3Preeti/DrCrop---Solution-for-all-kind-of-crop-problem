# DrCrop - AI Crop Disease Detector
# Production deployment script for Windows/PowerShell

param(
    [Parameter(Mandatory = $false)]
    [string]$Environment = "production",
    
    [Parameter(Mandatory = $false)]
    [switch]$SkipBuild = $false,
    
    [Parameter(Mandatory = $false)]
    [switch]$SkipTests = $false,
    
    [Parameter(Mandatory = $false)]
    [string]$ModelPath = "",
    
    [Parameter(Mandatory = $false)]
    [switch]$Help
)

if ($Help) {
    Write-Host @"
DrCrop Deployment Script

Usage: .\deploy.ps1 [OPTIONS]

Options:
  -Environment     Target environment (development, staging, production) [default: production]
  -SkipBuild      Skip Docker build process
  -SkipTests      Skip running tests
  -ModelPath      Path to trained model file
  -Help           Show this help message

Examples:
  .\deploy.ps1                                    # Full production deployment
  .\deploy.ps1 -Environment development           # Development deployment
  .\deploy.ps1 -SkipBuild -SkipTests             # Quick deployment
  .\deploy.ps1 -ModelPath ".\my_model.h5"       # Deploy with specific model

Requirements:
  - Docker Desktop for Windows
  - PowerShell 5.1 or higher
  - Git (for version tagging)
  - Internet connection (for downloading dependencies)
"@
    exit 0
}

# Color output functions
function Write-Success { param($Message) Write-Host $Message -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host $Message -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host $Message -ForegroundColor Red }
function Write-Info { param($Message) Write-Host $Message -ForegroundColor Cyan }

# Configuration
$PROJECT_NAME = "DrCrop"
$VERSION = "1.0.0"
$DOCKER_IMAGE = "drcrop-ai-detector"
$CONTAINER_NAME = "drcrop-container"

Write-Info "=========================================="
Write-Info "DrCrop AI Crop Disease Detector Deployment"
Write-Info "Environment: $Environment"
Write-Info "Version: $VERSION"
Write-Info "=========================================="

# Verify prerequisites
Write-Info "Checking prerequisites..."

# Check Docker
try {
    $dockerVersion = docker --version
    Write-Success "✓ Docker found: $dockerVersion"
}
catch {
    Write-Error "✗ Docker not found. Please install Docker Desktop."
    exit 1
}

# Check PowerShell version
$psVersion = $PSVersionTable.PSVersion.Major
if ($psVersion -lt 5) {
    Write-Error "✗ PowerShell 5.1 or higher required. Current version: $psVersion"
    exit 1
}
Write-Success "✓ PowerShell version: $psVersion"

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Success "✓ Docker daemon is running"
}
catch {
    Write-Error "✗ Docker daemon not running. Please start Docker Desktop."
    exit 1
}

# Create necessary directories
Write-Info "Creating deployment directories..."
$directories = @("logs", "uploads", "models", "database", "backups", "monitoring")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Success "✓ Created directory: $dir"
    }
}

# Copy model file if provided
if ($ModelPath -and (Test-Path $ModelPath)) {
    Write-Info "Copying model file..."
    Copy-Item $ModelPath ".\models\trained_model.h5" -Force
    Write-Success "✓ Model copied to deployment directory"
}
elseif ($ModelPath) {
    Write-Warning "⚠ Model file not found: $ModelPath"
}

# Environment-specific configuration
switch ($Environment) {
    "development" {
        $dockerComposeFile = "docker-compose.dev.yml"
        $envFile = ".env.development"
    }
    "staging" {
        $dockerComposeFile = "docker-compose.staging.yml"
        $envFile = ".env.staging"
    }
    default {
        $dockerComposeFile = "docker-compose.yml"
        $envFile = ".env"
    }
}

# Create environment file if it doesn't exist
if (-not (Test-Path $envFile)) {
    Write-Info "Creating environment configuration..."
    Copy-Item ".env.example" $envFile
    Write-Warning "⚠ Please configure $envFile with your settings"
}

# Run tests if not skipped
if (-not $SkipTests) {
    Write-Info "Running tests..."
    try {
        # Create simple test script
        $testScript = @"
import sys
import os
sys.path.append('.')

# Test model import
try:
    from models.crop_disease_model import CropDiseaseDetector
    print("✓ Model import test passed")
except Exception as e:
    print(f"✗ Model import test failed: {e}")
    sys.exit(1)

# Test database import
try:
    from database.disease_database import DiseaseDatabase
    print("✓ Database import test passed")
except Exception as e:
    print(f"✗ Database import test failed: {e}")
    sys.exit(1)

print("All tests passed!")
"@
        
        $testScript | Out-File -FilePath "test_deployment.py" -Encoding UTF8
        python test_deployment.py
        Remove-Item "test_deployment.py" -Force
        Write-Success "✓ All tests passed"
    }
    catch {
        Write-Error "✗ Tests failed. Use -SkipTests to bypass."
        exit 1
    }
}

# Build Docker image if not skipped
if (-not $SkipBuild) {
    Write-Info "Building Docker image..."
    try {
        docker build -t ${DOCKER_IMAGE}:${VERSION} -t ${DOCKER_IMAGE}:latest .
        Write-Success "✓ Docker image built successfully"
    }
    catch {
        Write-Error "✗ Docker build failed"
        exit 1
    }
}
else {
    Write-Warning "⚠ Skipping Docker build"
}

# Stop existing containers
Write-Info "Stopping existing containers..."
try {
    docker-compose -f $dockerComposeFile down
    Write-Success "✓ Existing containers stopped"
}
catch {
    Write-Warning "⚠ No existing containers to stop"
}

# Deploy with Docker Compose
Write-Info "Deploying with Docker Compose..."
try {
    docker-compose -f $dockerComposeFile up -d
    Write-Success "✓ Deployment completed successfully"
}
catch {
    Write-Error "✗ Deployment failed"
    exit 1
}

# Wait for services to start
Write-Info "Waiting for services to start..."
Start-Sleep -Seconds 10

# Health check
Write-Info "Performing health check..."
$maxAttempts = 30
$attempt = 0
$healthy = $false

while ($attempt -lt $maxAttempts -and -not $healthy) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            $healthy = $true
            Write-Success "✓ API is healthy and responding"
        }
    }
    catch {
        $attempt++
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
}

if (-not $healthy) {
    Write-Error "✗ Health check failed after $maxAttempts attempts"
    Write-Info "Checking container logs..."
    docker-compose -f $dockerComposeFile logs drcrop-app
    exit 1
}

# Create nginx configuration for frontend
$nginxConfig = @"
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    upstream api {
        server drcrop-app:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # Frontend static files
        location / {
            root /usr/share/nginx/html;
            index index.html;
            try_files `$uri `$uri/ /index.html;
        }
        
        # API proxy
        location /api/ {
            proxy_pass http://api/api/;
            proxy_set_header Host `$host;
            proxy_set_header X-Real-IP `$remote_addr;
            proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto `$scheme;
            client_max_body_size 50M;
        }
        
        # Uploads proxy
        location /uploads/ {
            proxy_pass http://api/uploads/;
            proxy_set_header Host `$host;
        }
    }
}
"@

# Create nginx directory and config
$nginxDir = "nginx"
if (-not (Test-Path $nginxDir)) {
    New-Item -ItemType Directory -Path $nginxDir -Force | Out-Null
}
$nginxConfig | Out-File -FilePath "$nginxDir\nginx.conf" -Encoding UTF8

# Display deployment information
Write-Success "=========================================="
Write-Success "Deployment completed successfully!"
Write-Success "=========================================="
Write-Info "Services:"
Write-Info "  • API Server: http://localhost:8000"
Write-Info "  • Web Interface: http://localhost"
Write-Info "  • API Documentation: http://localhost:8000/api/docs"
Write-Info "  • Health Check: http://localhost:8000/api/health"

if ($Environment -eq "development") {
    Write-Info "  • Grafana Dashboard: http://localhost:3000 (admin/admin123)"
    Write-Info "  • Prometheus: http://localhost:9090"
}

Write-Info ""
Write-Info "Useful commands:"
Write-Info "  • View logs: docker-compose -f $dockerComposeFile logs -f"
Write-Info "  • Stop services: docker-compose -f $dockerComposeFile down"
Write-Info "  • Restart services: docker-compose -f $dockerComposeFile restart"
Write-Info "  • Update services: docker-compose -f $dockerComposeFile pull && docker-compose -f $dockerComposeFile up -d"

Write-Success "DrCrop is now running and ready to detect crop diseases!"

# Optional: Open browser
$openBrowser = Read-Host "Open web interface in browser? (y/N)"
if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
    Start-Process "http://localhost"
}