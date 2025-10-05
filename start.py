"""
DrCrop Quick Start
Opens the web interface and shows how to use DrCrop
"""

import webbrowser
import os
import sys
from pathlib import Path

def open_frontend():
    """Open the DrCrop web interface"""
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        full_path = frontend_path.absolute()
        print("🌱 Opening DrCrop AI Crop Disease Detector...")
        print(f"📁 Frontend file: {full_path}")
        
        # Open in default browser
        webbrowser.open(f"file://{full_path}")
        
        print("✅ Frontend opened in your default browser!")
        return True
    else:
        print("❌ Frontend file not found!")
        return False

def show_api_status():
    """Show API status and instructions"""
    print("\n🔗 API Server Status:")
    print("   📍 To start the API server:")
    print("   1. Ensure you're in the drcrop conda environment")
    print("   2. Run: python server.py")
    print("   3. API will be available at: http://localhost:8000")
    print("\n📋 Available API Endpoints:")
    print("   • POST /api/predict - Upload image for disease detection")
    print("   • GET /api/health - Health check")
    print("   • GET /api/diseases - List all diseases")
    print("   • GET /api/crops - List supported crops")
    print("   • GET /api/statistics - Model performance metrics")

def show_demo_options():
    """Show demo options"""
    print("\n🎮 Demo Options:")
    print("   1. Web Interface (opens automatically)")
    print("   2. Python Demo: python demo.py")
    print("   3. API Server: python server.py (requires proper environment)")

def main():
    """Main function"""
    print("=" * 60)
    print("🌱 DrCrop - AI Crop Disease Detector")
    print("=" * 60)
    print("🎯 95% Accuracy | ⚡ Real-time Analysis | 🌾 38 Disease Types")
    print()
    
    # Try to open frontend
    if open_frontend():
        print("\n📝 How to use DrCrop:")
        print("   1. 📷 Upload a crop image using the web interface")
        print("   2. 🔍 AI will analyze the image for diseases")
        print("   3. 📊 Get instant results with confidence scores")
        print("   4. 💡 Receive treatment recommendations")
        
        show_api_status()
        show_demo_options()
        
        print("\n✨ Features:")
        print("   • Advanced CNN with transfer learning")
        print("   • Support for 14+ crop types")
        print("   • Detailed treatment recommendations")
        print("   • Organic treatment options")
        print("   • Real-time image analysis")
        
        print("\n🚀 Getting Started:")
        print("   • The web interface is now open in your browser")
        print("   • Try uploading a sample crop image")
        print("   • For full functionality, start the API server")
        
    else:
        print("❌ Could not open frontend. Please check file paths.")
    
    print("\n" + "=" * 60)
    print("Happy farming! 🚜🌾")

if __name__ == "__main__":
    main()