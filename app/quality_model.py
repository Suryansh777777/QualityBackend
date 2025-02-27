# app/quality_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import io
from PIL import Image

@register_keras_serializable()
class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([inputs.shape[i] if self.noise_shape[i] is None else self.noise_shape[i]
                     for i in range(len(self.noise_shape))])

@register_keras_serializable()
def swish(x):
    return x * tf.sigmoid(x)

class QualityModel:
    def __init__(self, model_path):
        # Register all the custom objects needed
        custom_objects = {
            'swish': swish,
            'FixedDropout': FixedDropout
        }
        
        try:
            # Try loading with custom objects
            self.model = load_model(model_path, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Alternative: If you can't load the model, you might need to rebuild it
            # This would be a placeholder for a rebuild function
            raise
        
        # Define class labels and shelf life mapping (same as before)
        self.class_labels = [
            'fresh_apple_ripe', 'fresh_apple_underripe', 'fresh_apple_overripe',
            'fresh_banana_ripe', 'fresh_banana_underripe', 'fresh_banana_overripe',
            'fresh_orange_ripe', 'fresh_orange_underripe', 'fresh_orange_overripe',
            'fresh_capsicum_ripe', 'fresh_capsicum_underripe', 'fresh_capsicum_overripe',
            'fresh_bitterground_ripe', 'fresh_bitterground_underripe', 'fresh_bitterground_overripe',
            'fresh_tomato_ripe', 'fresh_tomato_underripe', 'fresh_tomato_overripe',
            'rotten_apple', 'rotten_banana', 'rotten_orange',
            'rotten_capsicum', 'rotten_bitterground', 'rotten_tomato'
        ]
        
        self.shelf_life = {
            'fresh_apple_ripe': 7, 'fresh_apple_underripe': 10, 'fresh_apple_overripe': 3,
            'fresh_banana_ripe': 5, 'fresh_banana_underripe': 7, 'fresh_banana_overripe': 2,
            'fresh_orange_ripe': 10, 'fresh_orange_underripe': 14, 'fresh_orange_overripe': 5,
            'fresh_capsicum_ripe': 7, 'fresh_capsicum_underripe': 10, 'fresh_capsicum_overripe': 3,
            'fresh_bitterground_ripe': 6, 'fresh_bitterground_underripe': 8, 'fresh_bitterground_overripe': 2,
            'fresh_tomato_ripe': 8, 'fresh_tomato_underripe': 12, 'fresh_tomato_overripe': 4,
            'rotten_apple': 0, 'rotten_banana': 0, 'rotten_orange': 0,
            'rotten_capsicum': 0, 'rotten_bitterground': 0, 'rotten_tomato': 0
        }
    
    
    def preprocess_image(self, image_data):
        """Process image data from bytes or file path"""
        if isinstance(image_data, bytes):
            # If image is provided as bytes (from API upload)
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((224, 224))
        else:
            # If image is provided as file path
            image = load_img(image_data, target_size=(224, 224))
        
        # Convert to array and normalize
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    
    def predict(self, image_data):
        """Predict the quality of a fruit/vegetable image"""
        # Preprocess the image
        processed_image = self.preprocess_image(image_data)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = self.class_labels[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Parse fruit type and state
        parts = predicted_class.split('_')
        freshness = parts[0]  # 'fresh' or 'rotten'
        fruit_type = parts[1]  # 'apple', 'banana', etc.
        
        # Determine ripeness (if fresh)
        ripeness = None
        if freshness == "fresh" and len(parts) > 2:
            ripeness = parts[2]  # 'ripe', 'underripe', or 'overripe'
        
        # Get shelf life
        shelf_life_days = self.shelf_life.get(predicted_class, 0)
        
        # Prepare result
        result = {
            "class": predicted_class,
            "fruit_type": fruit_type,
            "freshness": freshness,
            "ripeness": ripeness,
            "shelf_life_days": shelf_life_days,
            "confidence": round(confidence * 100, 2)
        }
        
        return result