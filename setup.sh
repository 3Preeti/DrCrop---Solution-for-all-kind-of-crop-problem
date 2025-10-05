#!/bin/bash
# DrCrop Quick Start Script for Unix/Linux systems

echo "🌱 DrCrop - AI Crop Disease Detector Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.10+ and try again."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is required but not installed."
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "✅ Using $PIP_CMD for package installation"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
$PIP_CMD install -r requirements.txt

# Copy environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your configuration"
fi

# Initialize database
echo "🗄️ Initializing disease database..."
python3 database/disease_database.py

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads logs models/saved data/{train,validation,test}

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "🚀 To start DrCrop:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Start API server: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
echo "   3. Open frontend/index.html in your browser"
echo ""
echo "🔗 API will be available at: http://localhost:8000"
echo "📖 API Documentation: http://localhost:8000/api/docs"
echo ""
echo "📋 Optional: To train your own model:"
echo "   python3 scripts/train_model.py --create-sample"
echo "   # Add your images to data/train/, data/validation/, data/test/"
echo "   python3 scripts/train_model.py --epochs 100"
echo ""
echo "Happy farming! 🚜🌾"