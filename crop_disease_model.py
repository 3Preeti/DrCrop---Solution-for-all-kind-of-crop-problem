"""
DrCrop - Advanced Crop Disease Detection Model
Uses CNN with transfer learning for high-accuracy disease detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
from pathlib import Path
import json

class CropDiseaseDetector:
    def __init__(self, input_shape=(224, 224, 3), num_classes=38):
        """
        Initialize the Crop Disease Detection Model
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of disease classes to predict
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        # Disease classes mapping
        self.disease_classes = {
            0: 'Apple___Apple_scab',
            1: 'Apple___Black_rot',
            2: 'Apple___Cedar_apple_rust',
            3: 'Apple___healthy',
            4: 'Blueberry___healthy',
            5: 'Cherry_(including_sour)___Powdery_mildew',
            6: 'Cherry_(including_sour)___healthy',
            7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            8: 'Corn_(maize)___Common_rust_',
            9: 'Corn_(maize)___Northern_Leaf_Blight',
            10: 'Corn_(maize)___healthy',
            11: 'Grape___Black_rot',
            12: 'Grape___Esca_(Black_Measles)',
            13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            14: 'Grape___healthy',
            15: 'Orange___Haunglongbing_(Citrus_greening)',
            16: 'Peach___Bacterial_spot',
            17: 'Peach___healthy',
            18: 'Pepper,_bell___Bacterial_spot',
            19: 'Pepper,_bell___healthy',
            20: 'Potato___Early_blight',
            21: 'Potato___Late_blight',
            22: 'Potato___healthy',
            23: 'Raspberry___healthy',
            24: 'Soybean___healthy',
            25: 'Squash___Powdery_mildew',
            26: 'Strawberry___Leaf_scorch',
            27: 'Strawberry___healthy',
            28: 'Tomato___Bacterial_spot',
            29: 'Tomato___Early_blight',
            30: 'Tomato___Late_blight',
            31: 'Tomato___Leaf_Mold',
            32: 'Tomato___Septoria_leaf_spot',
            33: 'Tomato___Spider_mites Two-spotted_spider_mite',
            34: 'Tomato___Target_Spot',
            35: 'Tomato___Yellow_Leaf_Curl_Virus',
            36: 'Tomato___mosaic_virus',
            37: 'Tomato___healthy'
        }
    
    def build_model(self):
        """Build the CNN model with transfer learning"""
        
        # Create input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Load pre-trained EfficientNetB3 as base model
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Build the complete model
        model = models.Sequential([
            # Data preprocessing
            layers.Rescaling(1./255, input_shape=self.input_shape),
            
            # Base model
            base_model,
            
            # Custom classifier
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        self.model = model
        return model
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create data generators with augmentation"""
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=100, model_save_path='crop_disease_model.h5'):
        """Train the model"""
        
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Initial training
        print("Starting initial training...")
        history1 = self.model.fit(
            train_generator,
            epochs=epochs//2,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning: Unfreeze some layers
        print("Starting fine-tuning...")
        self.model.layers[1].trainable = True  # Unfreeze base model
        
        # Use lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Continue training
        history2 = self.model.fit(
            train_generator,
            epochs=epochs//2,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        return self.history
    
    def predict_disease(self, image_path, confidence_threshold=0.7):
        """
        Predict disease from image
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a trained model first.")
        
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.input_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'disease': self.disease_classes[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        # Determine if prediction is reliable
        is_healthy = 'healthy' in self.disease_classes[predicted_class_idx].lower()
        is_confident = confidence >= confidence_threshold
        
        result = {
            'predicted_disease': self.disease_classes[predicted_class_idx],
            'confidence': confidence,
            'is_healthy': is_healthy,
            'is_confident': is_confident,
            'top_3_predictions': top_3_predictions,
            'crop_type': self._extract_crop_type(self.disease_classes[predicted_class_idx])
        }
        
        return result
    
    def _extract_crop_type(self, disease_name):
        """Extract crop type from disease name"""
        crop_name = disease_name.split('___')[0]
        return crop_name.replace('_', ' ').title()
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path):
        """Save the current model"""
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save")
    
    def evaluate_model(self, test_generator):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Evaluate
        results = self.model.evaluate(test_generator, verbose=1)
        
        # Get predictions for confusion matrix
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        return {
            'test_loss': results[0],
            'test_accuracy': results[1],
            'test_top_3_accuracy': results[2] if len(results) > 2 else None,
            'predictions': predicted_classes,
            'true_labels': true_classes
        }

def create_sample_model():
    """Create and return a sample model instance"""
    detector = CropDiseaseDetector()
    model = detector.build_model()
    
    print("Model Architecture:")
    model.summary()
    
    return detector

if __name__ == "__main__":
    # Create sample model
    detector = create_sample_model()
    
    # Save model architecture diagram
    tf.keras.utils.plot_model(
        detector.model, 
        to_file='model_architecture.png', 
        show_shapes=True, 
        show_layer_names=True
    )
    
    print("Sample model created successfully!")
    print(f"Total parameters: {detector.model.count_params():,}")