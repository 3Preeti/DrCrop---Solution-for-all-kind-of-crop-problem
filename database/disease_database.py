"""
Comprehensive Disease Information Database for DrCrop
Contains detailed information about crop diseases, symptoms, treatments, and prevention
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class DiseaseDatabase:
    """Database containing comprehensive crop disease information"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), 'disease_info.json')
        self.disease_data = self._load_disease_data()
    
    def _load_disease_data(self):
        """Load disease data from JSON file or create default database"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default database
            default_data = self._create_default_database()
            self._save_database(default_data)
            return default_data
    
    def _save_database(self, data):
        """Save disease data to JSON file"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _create_default_database(self):
        """Create comprehensive default disease database"""
        return {
            "Apple___Apple_scab": {
                "crop_type": "Apple",
                "disease_name": "Apple Scab",
                "severity": "moderate",
                "description": "Apple scab is a fungal disease caused by Venturia inaequalis that affects apple trees. It causes dark, scaly lesions on leaves, fruit, and twigs.",
                "symptoms": [
                    "Dark, scaly spots on leaves",
                    "Circular lesions on fruit surface",
                    "Premature leaf drop",
                    "Cracked and distorted fruit",
                    "Olive-green to black spots on young leaves"
                ],
                "treatment": [
                    "Apply fungicide sprays during wet weather",
                    "Use copper-based fungicides in early season",
                    "Remove infected leaves and fruit debris",
                    "Improve air circulation through pruning",
                    "Apply preventive fungicide treatments"
                ],
                "prevention": [
                    "Plant disease-resistant apple varieties",
                    "Ensure proper spacing for air circulation",
                    "Remove fallen leaves and debris",
                    "Avoid overhead watering",
                    "Regular monitoring during humid conditions"
                ],
                "organic_treatment": [
                    "Apply neem oil spray",
                    "Use baking soda solution (1 tsp per quart water)",
                    "Compost tea applications",
                    "Beneficial bacteria sprays"
                ]
            },
            
            "Apple___Black_rot": {
                "crop_type": "Apple",
                "disease_name": "Black Rot",
                "severity": "high",
                "description": "Black rot is a serious fungal disease caused by Botryosphaeria obtusa that affects apple trees, causing fruit rot and cankers on branches.",
                "symptoms": [
                    "Circular brown spots on fruit",
                    "Black, mummified fruit",
                    "Cankers on branches and trunk",
                    "Yellowing and wilting of leaves",
                    "Concentric rings in fruit lesions"
                ],
                "treatment": [
                    "Remove and destroy infected fruit and branches",
                    "Apply fungicide sprays during growing season",
                    "Prune out cankers during dormant season",
                    "Improve tree nutrition and vigor",
                    "Use systemic fungicides for severe infections"
                ],
                "prevention": [
                    "Plant in well-draining soil",
                    "Avoid mechanical injuries to trees",
                    "Regular sanitation of orchard floor",
                    "Proper pruning for air circulation",
                    "Stress reduction through proper irrigation"
                ],
                "organic_treatment": [
                    "Copper sulfate applications",
                    "Bordeaux mixture spray",
                    "Remove infected plant material",
                    "Beneficial microorganism applications"
                ]
            },
            
            "Tomato___Early_blight": {
                "crop_type": "Tomato",
                "disease_name": "Early Blight",
                "severity": "moderate",
                "description": "Early blight is a common fungal disease caused by Alternaria solani that affects tomato plants, causing leaf spots and fruit rot.",
                "symptoms": [
                    "Dark brown spots with concentric rings on leaves",
                    "Yellowing of lower leaves",
                    "Stem lesions with dark centers",
                    "Fruit rot near the stem end",
                    "Premature defoliation"
                ],
                "treatment": [
                    "Apply fungicide containing chlorothalonil or copper",
                    "Remove affected plant parts immediately",
                    "Improve air circulation around plants",
                    "Avoid overhead watering",
                    "Apply mulch to prevent soil splash"
                ],
                "prevention": [
                    "Crop rotation with non-solanaceous plants",
                    "Plant resistant varieties when available",
                    "Proper spacing between plants",
                    "Drip irrigation instead of overhead watering",
                    "Regular field sanitation"
                ],
                "organic_treatment": [
                    "Neem oil applications",
                    "Baking soda spray (1 tbsp per gallon)",
                    "Compost tea foliar feeding",
                    "Copper-based organic fungicides"
                ]
            },
            
            "Tomato___Late_blight": {
                "crop_type": "Tomato",
                "disease_name": "Late Blight",
                "severity": "high",
                "description": "Late blight is a devastating disease caused by Phytophthora infestans that can destroy entire tomato crops within days under favorable conditions.",
                "symptoms": [
                    "Water-soaked lesions on leaves",
                    "White fuzzy growth on leaf undersides",
                    "Brown to black lesions on stems",
                    "Dark, firm lesions on fruit",
                    "Rapid plant collapse in humid conditions"
                ],
                "treatment": [
                    "Apply preventive fungicides immediately",
                    "Remove and destroy infected plants",
                    "Improve drainage and air circulation",
                    "Use systemic fungicides for protection",
                    "Monitor weather conditions closely"
                ],
                "prevention": [
                    "Plant certified disease-free seedlings",
                    "Avoid overhead irrigation",
                    "Ensure proper plant spacing",
                    "Remove volunteer potato plants",
                    "Apply preventive fungicide programs"
                ],
                "organic_treatment": [
                    "Copper-based fungicides",
                    "Potassium bicarbonate sprays",
                    "Milk spray solution (1:10 ratio)",
                    "Immediate removal of infected tissue"
                ]
            },
            
            "Potato___Early_blight": {
                "crop_type": "Potato",
                "disease_name": "Early Blight",
                "severity": "moderate",
                "description": "Early blight in potatoes is caused by Alternaria solani and affects leaves, stems, and tubers, reducing yield and quality.",
                "symptoms": [
                    "Circular dark spots with concentric rings on leaves",
                    "Yellowing and death of lower leaves",
                    "Dark lesions on stems",
                    "Sunken, dark lesions on tubers",
                    "Reduced plant vigor and yield"
                ],
                "treatment": [
                    "Apply fungicides containing mancozeb or chlorothalonil",
                    "Remove infected plant debris",
                    "Ensure proper nutrition, especially potassium",
                    "Avoid mechanical injury to plants",
                    "Harvest tubers before severe infection"
                ],
                "prevention": [
                    "Use certified disease-free seed potatoes",
                    "Implement crop rotation",
                    "Avoid overhead irrigation",
                    "Hill soil properly around plants",
                    "Remove volunteer plants and weeds"
                ],
                "organic_treatment": [
                    "Copper-based organic fungicides",
                    "Neem oil applications",
                    "Proper composting of plant debris",
                    "Beneficial microorganism applications"
                ]
            },
            
            "Potato___Late_blight": {
                "crop_type": "Potato",
                "disease_name": "Late Blight",
                "severity": "high",
                "description": "Late blight is the most destructive potato disease, caused by Phytophthora infestans, historically responsible for the Irish Potato Famine.",
                "symptoms": [
                    "Water-soaked lesions on leaves",
                    "White, fuzzy growth on leaf undersides",
                    "Blackened stems and petioles",
                    "Brown, firm rot on tubers",
                    "Foul odor from infected tubers"
                ],
                "treatment": [
                    "Apply systemic fungicides immediately",
                    "Destroy infected plants completely",
                    "Avoid harvesting wet tubers",
                    "Cure tubers properly before storage",
                    "Monitor storage conditions carefully"
                ],
                "prevention": [
                    "Plant certified disease-free seed",
                    "Implement strict crop rotation",
                    "Monitor weather for disease-favorable conditions",
                    "Use resistant varieties when available",
                    "Eliminate volunteer potatoes and tomatoes"
                ],
                "organic_treatment": [
                    "Copper sulfate applications",
                    "Bordeaux mixture spray",
                    "Immediate destruction of infected plants",
                    "Soil solarization between crops"
                ]
            },
            
            "Corn_(maize)___Northern_Leaf_Blight": {
                "crop_type": "Corn",
                "disease_name": "Northern Leaf Blight",
                "severity": "moderate",
                "description": "Northern leaf blight is a fungal disease caused by Exserohilum turcicum that affects corn leaves, reducing photosynthesis and yield.",
                "symptoms": [
                    "Long, elliptical gray-green lesions on leaves",
                    "Lesions with distinct margins",
                    "Dark spores on lesion surface",
                    "Premature senescence of leaves",
                    "Reduced grain fill and yield"
                ],
                "treatment": [
                    "Apply foliar fungicides during early infection",
                    "Use strobilurin or triazole fungicides",
                    "Time applications based on disease pressure",
                    "Consider economic thresholds for treatment",
                    "Rotate fungicide modes of action"
                ],
                "prevention": [
                    "Plant resistant corn hybrids",
                    "Implement crop rotation with non-hosts",
                    "Bury crop residue through tillage",
                    "Avoid dense plant populations",
                    "Monitor fields regularly during growing season"
                ],
                "organic_treatment": [
                    "Copper-based fungicides",
                    "Biological control agents",
                    "Crop rotation with legumes",
                    "Compost application for soil health"
                ]
            },
            
            "Grape___Black_rot": {
                "crop_type": "Grape",
                "disease_name": "Black Rot",
                "severity": "high",
                "description": "Black rot is a serious fungal disease of grapes caused by Guignardia bidwellii, affecting leaves, shoots, and fruit clusters.",
                "symptoms": [
                    "Circular brown spots on leaves with dark borders",
                    "Small black lesions on young shoots",
                    "Brown, circular spots on fruit",
                    "Shriveled, mummified berries",
                    "Reduced fruit quality and yield"
                ],
                "treatment": [
                    "Apply preventive fungicide sprays",
                    "Remove mummified berries and infected canes",
                    "Use copper or sulfur-based fungicides",
                    "Improve air circulation through pruning",
                    "Apply treatments during wet weather"
                ],
                "prevention": [
                    "Select disease-resistant grape varieties",
                    "Prune for good air circulation",
                    "Remove fallen leaves and fruit debris",
                    "Avoid overhead irrigation",
                    "Monitor humidity levels in vineyard"
                ],
                "organic_treatment": [
                    "Sulfur dust applications",
                    "Copper hydroxide sprays",
                    "Bordeaux mixture treatments",
                    "Biological control agents"
                ]
            },
            
            # Healthy plant entries
            "Apple___healthy": {
                "crop_type": "Apple",
                "disease_name": "Healthy",
                "severity": "none",
                "description": "Healthy apple plants show vigorous growth, green foliage, and normal fruit development without any disease symptoms.",
                "symptoms": [
                    "Bright green, uniform foliage",
                    "Strong, upright growth",
                    "Normal fruit development",
                    "No lesions or discoloration",
                    "Good overall plant vigor"
                ],
                "treatment": [
                    "Continue current management practices",
                    "Maintain regular monitoring",
                    "Ensure proper nutrition",
                    "Maintain adequate soil moisture",
                    "Continue preventive care program"
                ],
                "prevention": [
                    "Regular inspection for early disease detection",
                    "Maintain proper plant nutrition",
                    "Ensure adequate water management",
                    "Practice good orchard sanitation",
                    "Follow integrated pest management"
                ],
                "organic_treatment": [
                    "Continue organic management practices",
                    "Regular compost applications",
                    "Beneficial microorganism supplements",
                    "Natural pest control methods"
                ]
            },
            
            "Tomato___healthy": {
                "crop_type": "Tomato",
                "disease_name": "Healthy",
                "severity": "none",
                "description": "Healthy tomato plants display vigorous growth, dark green foliage, and normal fruit production without disease symptoms.",
                "symptoms": [
                    "Dark green, turgid leaves",
                    "Strong stem growth",
                    "Normal flowering and fruit set",
                    "No wilting or discoloration",
                    "Healthy root development"
                ],
                "treatment": [
                    "Maintain current care routine",
                    "Continue regular watering schedule",
                    "Monitor for any changes",
                    "Ensure proper fertilization",
                    "Maintain optimal growing conditions"
                ],
                "prevention": [
                    "Regular monitoring for disease symptoms",
                    "Proper spacing for air circulation",
                    "Consistent watering practices",
                    "Balanced fertilization program",
                    "Good garden hygiene practices"
                ],
                "organic_treatment": [
                    "Continue organic growing methods",
                    "Compost tea applications",
                    "Companion planting benefits",
                    "Natural soil amendments"
                ]
            },
            
            "Potato___healthy": {
                "crop_type": "Potato",
                "disease_name": "Healthy",
                "severity": "none", 
                "description": "Healthy potato plants show robust growth, green foliage, and proper tuber development without any disease signs.",
                "symptoms": [
                    "Vigorous green foliage",
                    "Strong root and tuber development",
                    "Normal plant height and structure",
                    "No leaf spotting or wilting",
                    "Healthy flowering (if applicable)"
                ],
                "treatment": [
                    "Continue successful management practices",
                    "Maintain consistent care routine",
                    "Monitor for any changes",
                    "Ensure proper hilling",
                    "Continue regular inspection"
                ],
                "prevention": [
                    "Regular field monitoring",
                    "Proper crop rotation",
                    "Adequate nutrition program",
                    "Good water management",
                    "Disease prevention practices"
                ],
                "organic_treatment": [
                    "Maintain organic certification practices",
                    "Natural soil enhancement",
                    "Beneficial insect habitat",
                    "Organic matter incorporation"
                ]
            },
            
            "Corn_(maize)___healthy": {
                "crop_type": "Corn",
                "disease_name": "Healthy",
                "severity": "none",
                "description": "Healthy corn plants exhibit strong growth, green leaves, and normal ear development without disease symptoms.",
                "symptoms": [
                    "Tall, sturdy stalks",
                    "Green, turgid leaves",
                    "Normal ear development",
                    "Good root establishment",
                    "Uniform plant growth"
                ],
                "treatment": [
                    "Continue current agronomic practices",
                    "Maintain fertilization program",
                    "Monitor growth stages",
                    "Ensure adequate water supply",
                    "Continue pest monitoring"
                ],
                "prevention": [
                    "Regular field scouting",
                    "Proper nutrient management",
                    "Weed control practices",
                    "Integrated pest management",
                    "Good field hygiene"
                ],
                "organic_treatment": [
                    "Organic fertilization methods",
                    "Natural pest control",
                    "Crop rotation benefits",
                    "Soil health maintenance"
                ]
            },
            
            "Grape___healthy": {
                "crop_type": "Grape",
                "disease_name": "Healthy",
                "severity": "none",
                "description": "Healthy grape vines show vigorous growth, green foliage, and normal fruit cluster development.",
                "symptoms": [
                    "Bright green, full foliage",
                    "Strong vine growth",
                    "Normal cluster development",
                    "No leaf spotting or wilting",
                    "Good overall vine vigor"
                ],
                "treatment": [
                    "Continue vineyard management practices",
                    "Maintain pruning schedule",
                    "Monitor for changes",
                    "Ensure proper irrigation",
                    "Continue nutrition program"
                ],
                "prevention": [
                    "Regular vineyard inspection",
                    "Proper canopy management",
                    "Good air circulation",
                    "Disease monitoring program",
                    "Integrated vineyard management"
                ],
                "organic_treatment": [
                    "Organic vineyard practices",
                    "Natural soil amendments",
                    "Beneficial organism habitat",
                    "Organic pest control methods"
                ]
            }
        }
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """Get complete information about a specific disease"""
        return self.disease_data.get(disease_name)
    
    def get_all_diseases(self) -> List[Dict]:
        """Get list of all diseases with basic information"""
        diseases = []
        for disease_name, info in self.disease_data.items():
            diseases.append({
                "disease_name": disease_name,
                "display_name": info.get("disease_name", disease_name),
                "crop_type": info.get("crop_type", "Unknown"),
                "severity": info.get("severity", "unknown"),
                "description": info.get("description", "No description available")
            })
        return diseases
    
    def get_diseases_by_crop(self, crop_type: str) -> List[Dict]:
        """Get all diseases for a specific crop type"""
        diseases = []
        for disease_name, info in self.disease_data.items():
            if info.get("crop_type", "").lower() == crop_type.lower():
                diseases.append({
                    "disease_name": disease_name,
                    "display_name": info.get("disease_name", disease_name),
                    "severity": info.get("severity", "unknown"),
                    "description": info.get("description", "No description available")
                })
        return diseases
    
    def get_supported_crops(self) -> List[str]:
        """Get list of all supported crop types"""
        crops = set()
        for info in self.disease_data.values():
            crop_type = info.get("crop_type")
            if crop_type:
                crops.add(crop_type)
        return sorted(list(crops))
    
    def search_diseases(self, query: str) -> List[Dict]:
        """Search diseases by name, symptoms, or description"""
        query = query.lower()
        results = []
        
        for disease_name, info in self.disease_data.items():
            # Search in disease name
            if query in disease_name.lower() or query in info.get("disease_name", "").lower():
                results.append({
                    "disease_name": disease_name,
                    "display_name": info.get("disease_name", disease_name),
                    "crop_type": info.get("crop_type", "Unknown"),
                    "match_type": "name"
                })
                continue
            
            # Search in description
            if query in info.get("description", "").lower():
                results.append({
                    "disease_name": disease_name,
                    "display_name": info.get("disease_name", disease_name),
                    "crop_type": info.get("crop_type", "Unknown"),
                    "match_type": "description"
                })
                continue
            
            # Search in symptoms
            symptoms = info.get("symptoms", [])
            for symptom in symptoms:
                if query in symptom.lower():
                    results.append({
                        "disease_name": disease_name,
                        "display_name": info.get("disease_name", disease_name),
                        "crop_type": info.get("crop_type", "Unknown"),
                        "match_type": "symptom"
                    })
                    break
        
        return results
    
    def add_disease(self, disease_name: str, disease_info: Dict):
        """Add a new disease to the database"""
        self.disease_data[disease_name] = disease_info
        self._save_database(self.disease_data)
    
    def update_disease(self, disease_name: str, updates: Dict):
        """Update existing disease information"""
        if disease_name in self.disease_data:
            self.disease_data[disease_name].update(updates)
            self._save_database(self.disease_data)
            return True
        return False
    
    def get_treatment_recommendations(self, disease_name: str, organic_only: bool = False) -> Dict:
        """Get treatment recommendations for a disease"""
        disease_info = self.get_disease_info(disease_name)
        if not disease_info:
            return {"error": "Disease not found"}
        
        recommendations = {
            "disease": disease_info.get("disease_name", disease_name),
            "severity": disease_info.get("severity", "unknown"),
            "immediate_actions": [],
            "treatments": disease_info.get("organic_treatment" if organic_only else "treatment", []),
            "prevention": disease_info.get("prevention", [])
        }
        
        # Add immediate actions based on severity
        severity = disease_info.get("severity", "unknown")
        if severity == "high":
            recommendations["immediate_actions"] = [
                "Isolate affected plants immediately",
                "Remove and destroy infected plant material",
                "Apply treatment as soon as possible",
                "Monitor surrounding plants closely"
            ]
        elif severity == "moderate":
            recommendations["immediate_actions"] = [
                "Remove affected plant parts",
                "Apply preventive treatments to healthy plants",
                "Improve growing conditions",
                "Monitor for spread"
            ]
        else:
            recommendations["immediate_actions"] = [
                "Continue monitoring",
                "Maintain good growing practices",
                "Apply preventive measures"
            ]
        
        return recommendations

def create_database():
    """Create and initialize the disease database"""
    db = DiseaseDatabase()
    print(f"Disease database created with {len(db.disease_data)} diseases")
    print(f"Supported crops: {', '.join(db.get_supported_crops())}")
    return db

if __name__ == "__main__":
    # Create and test the database
    db = create_database()
    
    # Test some functionality
    print("\nTesting database functionality:")
    
    # Get disease info
    tomato_blight = db.get_disease_info("Tomato___Early_blight")
    if tomato_blight:
        print(f"\nTomato Early Blight: {tomato_blight['description']}")
    
    # Search diseases
    search_results = db.search_diseases("blight")
    print(f"\nSearch results for 'blight': {len(search_results)} diseases found")
    
    # Get treatment recommendations
    treatment = db.get_treatment_recommendations("Tomato___Early_blight")
    print(f"\nTreatment for Early Blight: {len(treatment['treatments'])} recommendations")
