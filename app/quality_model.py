
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import pickle

class FruitQualityModel:
    def __init__(self):
        # Update these paths to your model files
        MODEL_PATH = 'models/best_model.keras'
        ENCODER_PATH = 'models/label_encoder.pkl'
        
        self.model = load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def predict(self, image_path):
        try:
            # Load and preprocess image
            image = load_img(image_path, target_size=(224, 224))
            image_array = img_to_array(image)
            image_array = preprocess_input(image_array)
            image_array = np.expand_dims(image_array, axis=0)

            # Make prediction
            predictions = self.model.predict(image_array)
            predicted_class = self.label_encoder.inverse_transform([np.argmax(predictions)])[0]
            confidence = float(np.max(predictions))

            # Calculate quality metrics (you'll need to adjust these based on your model)
            quality_score = self._calculate_quality_score(predictions)
            nutritional_data = self._estimate_nutritional_data(predictions)
            physical_properties = self._estimate_physical_properties(predictions)

            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'quality_score': quality_score,
                'nutritional_data': nutritional_data,
                'physical_properties': physical_properties
            }

        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

    def _calculate_quality_score(self, predictions):
        # Implement your quality score calculation logic
        return float(np.mean(predictions) * 100)

    def _estimate_nutritional_data(self, predictions):
        # Implement your nutritional estimation logic
        return {
            'sugars': 30,
            'fiber': 20,
            'vitamins': 15,
            'minerals': 25,
            'proteins': 10
        }

    def _estimate_physical_properties(self, predictions):
        # Implement your physical properties estimation logic
        return {
            'weight': 150,
            'size': {'length': 7.5, 'width': 7.2},
            'firmness': 75
        }