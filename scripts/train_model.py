"""
Training utilities for DrCrop AI Crop Disease Detection
Handles data preparation, model training, evaluation, and visualization
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json
import argparse
from datetime import datetime
import logging

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from crop_disease_model import CropDiseaseDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages dataset preparation and organization"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "validation"
        self.test_dir = self.data_dir / "test"
    
    def organize_dataset(self, source_dir, train_split=0.7, val_split=0.2, test_split=0.1):
        """
        Organize dataset into train/validation/test splits
        
        Args:
            source_dir: Directory containing disease class folders
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
        """
        import random
        import shutil
        
        source_path = Path(source_dir)
        
        # Create directories
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each disease class
        for class_dir in source_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # Create class directories in each split
                (self.train_dir / class_name).mkdir(exist_ok=True)
                (self.val_dir / class_name).mkdir(exist_ok=True)
                (self.test_dir / class_name).mkdir(exist_ok=True)
                
                # Get all images
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
                random.shuffle(images)
                
                # Calculate split indices
                n_images = len(images)
                train_end = int(n_images * train_split)
                val_end = train_end + int(n_images * val_split)
                
                # Split images
                train_images = images[:train_end]
                val_images = images[train_end:val_end]
                test_images = images[val_end:]
                
                # Copy images to respective directories
                for img in train_images:
                    shutil.copy2(img, self.train_dir / class_name / img.name)
                
                for img in val_images:
                    shutil.copy2(img, self.val_dir / class_name / img.name)
                
                for img in test_images:
                    shutil.copy2(img, self.test_dir / class_name / img.name)
                
                logger.info(f"Class {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        info = {}
        
        for split_name, split_dir in [("train", self.train_dir), ("validation", self.val_dir), ("test", self.test_dir)]:
            split_info = {}
            total_images = 0
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    class_images = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpeg")))
                    split_info[class_dir.name] = class_images
                    total_images += class_images
            
            info[split_name] = {
                "classes": split_info,
                "total_images": total_images,
                "num_classes": len(split_info)
            }
        
        return info
    
    def create_sample_dataset(self):
        """Create a sample dataset structure for demonstration"""
        sample_diseases = [
            "Tomato___healthy",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Potato___healthy",
            "Potato___Early_blight",
            "Corn_(maize)___healthy",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Apple___healthy",
            "Apple___Apple_scab",
            "Grape___healthy"
        ]
        
        # Create sample directory structure
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            for disease in sample_diseases:
                (split_dir / disease).mkdir(parents=True, exist_ok=True)
                
                # Create placeholder text files (in real scenario, these would be images)
                num_samples = 100 if split_dir == self.train_dir else 20
                for i in range(num_samples):
                    placeholder_file = split_dir / disease / f"sample_{i:03d}.txt"
                    placeholder_file.write_text(f"Placeholder for {disease} image {i}")
        
        logger.info("Sample dataset structure created")

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model_save_dir="./models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.detector = None
        self.training_history = None
    
    def train_model(self, data_dir, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train the crop disease detection model
        
        Args:
            data_dir: Directory containing train/validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        data_path = Path(data_dir)
        train_dir = data_path / "train"
        val_dir = data_path / "validation"
        
        # Validate directories exist
        if not train_dir.exists() or not val_dir.exists():
            raise ValueError("Training or validation directory not found")
        
        # Count classes
        num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
        logger.info(f"Training with {num_classes} classes")
        
        # Initialize model
        self.detector = CropDiseaseDetector(num_classes=num_classes)
        self.detector.build_model()
        
        # Create data generators
        train_generator, val_generator = self.detector.create_data_generators(
            str(train_dir), str(val_dir), batch_size=batch_size
        )
        
        # Train model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_save_dir / f"crop_disease_model_{timestamp}.h5"
        
        logger.info("Starting model training...")
        self.training_history = self.detector.train(
            train_generator, val_generator, epochs=epochs, model_save_path=str(model_path)
        )
        
        # Save training history
        history_path = self.model_save_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Training history saved to {history_path}")
        
        return str(model_path), str(history_path)
    
    def evaluate_model(self, model_path, test_data_dir):
        """
        Evaluate trained model on test data
        
        Args:
            model_path: Path to saved model
            test_data_dir: Directory containing test data
        """
        # Load model
        self.detector = CropDiseaseDetector()
        self.detector.load_model(model_path)
        
        # Create test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate
        results = self.detector.evaluate_model(test_generator)
        
        # Generate classification report
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(
            results['true_labels'], 
            results['predictions'], 
            target_names=class_names,
            output_dict=True
        )
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_path = self.model_save_dir / f"evaluation_results_{timestamp}.json"
        
        eval_results = {
            "test_accuracy": float(results['test_accuracy']),
            "test_loss": float(results['test_loss']),
            "classification_report": report,
            "model_path": model_path,
            "evaluation_timestamp": timestamp
        }
        
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_path}")
        logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
        
        return eval_results
    
    def plot_training_history(self, history_path, save_path=None):
        """Plot training history"""
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(history['loss'], label='Training Loss')
        axes[0, 1].plot(history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy difference
        acc_diff = np.array(history['accuracy']) - np.array(history['val_accuracy'])
        axes[1, 0].plot(acc_diff)
        axes[1, 0].set_title('Training vs Validation Accuracy Difference')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy Difference')
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[1, 1].plot(history['lr'])
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training plots saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, eval_results_path, save_path=None):
        """Plot confusion matrix from evaluation results"""
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
        
        report = eval_results['classification_report']
        class_names = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        # Create confusion matrix from classification report
        # Note: This is a simplified visualization. For actual confusion matrix,
        # you'd need the raw predictions and true labels
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a matrix of F1-scores for visualization
        f1_scores = [report[class_name]['f1-score'] for class_name in class_names]
        matrix = np.diag(f1_scores)
        
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Model Performance by Class (F1-Scores on Diagonal)')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train DrCrop Disease Detection Model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory containing dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--organize-data', action='store_true', help='Organize dataset before training')
    parser.add_argument('--source-dir', type=str, help='Source directory for dataset organization')
    parser.add_argument('--evaluate-only', type=str, help='Only evaluate existing model (provide model path)')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset structure')
    
    args = parser.parse_args()
    
    # Initialize components
    dataset_manager = DatasetManager(args.data_dir)
    trainer = ModelTrainer()
    
    try:
        # Create sample dataset if requested
        if args.create_sample:
            logger.info("Creating sample dataset structure...")
            dataset_manager.create_sample_dataset()
            logger.info("Sample dataset created. Add your images to the appropriate directories.")
            return
        
        # Organize dataset if requested
        if args.organize_data:
            if not args.source_dir:
                raise ValueError("--source-dir required when --organize-data is specified")
            
            logger.info("Organizing dataset...")
            dataset_manager.organize_dataset(args.source_dir)
        
        # Print dataset info
        dataset_info = dataset_manager.get_dataset_info()
        logger.info("Dataset Information:")
        for split, info in dataset_info.items():
            logger.info(f"  {split}: {info['total_images']} images, {info['num_classes']} classes")
        
        # Evaluation only mode
        if args.evaluate_only:
            logger.info(f"Evaluating model: {args.evaluate_only}")
            test_dir = Path(args.data_dir) / "test"
            if not test_dir.exists():
                raise ValueError("Test directory not found")
            
            eval_results = trainer.evaluate_model(args.evaluate_only, str(test_dir))
            logger.info("Evaluation completed successfully")
            return
        
        # Train model
        logger.info("Starting model training...")
        model_path, history_path = trainer.train_model(
            args.data_dir, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Plot training history
        plot_path = Path(trainer.model_save_dir) / f"training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        trainer.plot_training_history(history_path, str(plot_path))
        
        # Evaluate on test set if available
        test_dir = Path(args.data_dir) / "test"
        if test_dir.exists():
            logger.info("Evaluating on test set...")
            eval_results = trainer.evaluate_model(model_path, str(test_dir))
            
            # Plot confusion matrix
            eval_path = trainer.model_save_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            confusion_plot_path = trainer.model_save_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            trainer.plot_confusion_matrix(str(eval_path), str(confusion_plot_path))
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
