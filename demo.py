"""
DrCrop Quick Demo - Minimal working example
This demonstrates the AI crop disease detection without dependencies
"""

import os
import sys
import json
from pathlib import Path

# Simple disease database for demo
DEMO_DISEASES = {
    "tomato_healthy": {
        "disease": "Healthy Tomato",
        "confidence": 0.98,
        "description": "Your tomato plant appears healthy!",
        "recommendations": [
            "Continue current care routine",
            "Monitor for any changes",
            "Maintain consistent watering"
        ]
    },
    "tomato_early_blight": {
        "disease": "Tomato Early Blight",
        "confidence": 0.92,
        "description": "Early blight detected - fungal disease causing dark spots",
        "recommendations": [
            "Remove affected leaves immediately",
            "Apply fungicide spray",
            "Improve air circulation",
            "Avoid overhead watering"
        ]
    },
    "potato_late_blight": {
        "disease": "Potato Late Blight",
        "confidence": 0.89,
        "description": "Late blight - serious disease requiring immediate action",
        "recommendations": [
            "Remove infected plants immediately",
            "Apply preventive fungicide",
            "Improve drainage",
            "Monitor weather conditions"
        ]
    }
}

class DrCropDemo:
    """Simple demo version of DrCrop AI"""
    
    def __init__(self):
        self.version = "1.0.0-demo"
        print("🌱 DrCrop AI Crop Disease Detector - Demo Version")
        print("=" * 50)
    
    def analyze_image(self, image_path=None):
        """Simulate disease analysis"""
        if not image_path:
            # Demo mode - cycle through diseases
            import random
            disease_key = random.choice(list(DEMO_DISEASES.keys()))
        else:
            # Simple heuristic based on filename
            image_name = os.path.basename(image_path).lower()
            if "healthy" in image_name:
                disease_key = "tomato_healthy"
            elif "early" in image_name or "blight" in image_name:
                disease_key = "tomato_early_blight"
            elif "late" in image_name or "potato" in image_name:
                disease_key = "potato_late_blight"
            else:
                disease_key = "tomato_healthy"
        
        result = DEMO_DISEASES[disease_key].copy()
        result["analysis_time"] = "0.8 seconds"
        result["model_version"] = self.version
        
        return result
    
    def print_analysis(self, result):
        """Print formatted analysis results"""
        print(f"\n🔍 Analysis Results:")
        print(f"Disease: {result['disease']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Analysis Time: {result['analysis_time']}")
        print(f"\n📝 Description:")
        print(f"  {result['description']}")
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
        print()
    
    def run_demo(self):
        """Run interactive demo"""
        print("🚀 Running DrCrop Demo...")
        print("\nChoose an option:")
        print("1. Analyze sample 'healthy tomato'")
        print("2. Analyze sample 'tomato with early blight'")
        print("3. Analyze sample 'potato with late blight'")
        print("4. Random analysis")
        print("5. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "1":
                    result = self.analyze_image("healthy_tomato.jpg")
                elif choice == "2":
                    result = self.analyze_image("early_blight_tomato.jpg")
                elif choice == "3":
                    result = self.analyze_image("potato_late_blight.jpg")
                elif choice == "4":
                    result = self.analyze_image()
                elif choice == "5":
                    print("👋 Thanks for trying DrCrop!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    continue
                
                self.print_analysis(result)
                
                continue_demo = input("Try another analysis? (y/n): ").strip().lower()
                if continue_demo != 'y':
                    print("👋 Thanks for trying DrCrop!")
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Thanks for trying DrCrop!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("DrCrop AI Crop Disease Detector")
    print("Demo Version - No dependencies required!")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("models"):
        print("ℹ️  Note: Run this from the DrCrop directory for full functionality")
    
    # Run demo
    demo = DrCropDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()