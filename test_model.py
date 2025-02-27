import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
from app.quality_model import QualityModel
from PIL import Image
import tensorflow as tf
import numpy as np

def test_model():
    try:
        # Initialize the model
        print("Initializing model...")
        model_path = os.getenv("MODEL_PATH", "models/deep_model.h5")
        model_service = QualityModel(model_path)
        print("✅ Model loaded successfully!")

        # Test with a sample image
        test_image_path = "/home/surya/codes/Quality-Analysis-of-Fruits-and-Vegetables/backend/images/test_image2.png"  # Adjust path as needed
        if not os.path.exists(test_image_path):
            print("❌ Test image not found!")
            return

        # Load and test the image
        print("\nLoading and testing image...")
        with open(test_image_path, 'rb') as image_file:
            image_bytes = image_file.read()
            result = model_service.predict(image_bytes)
        
        print("\nPrediction Results:")
        print("==================")
        for key, value in result.items():
            if key == "confidence":
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value}")
        
        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\nDebug information:")
        print("=================")
        print(f"Python path: {os.environ.get('PYTHONPATH')}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Model directory contents: {os.listdir('models') if os.path.exists('models') else 'models directory not found'}")
        raise

# def test_multiple_images():
    """Test the model with multiple images from a directory"""
    try:
        print("Initializing model...")
        model_path = os.getenv("MODEL_PATH", "models/deep_model.h5")
        model_service = QualityModel(model_path)
        print("✅ Model loaded successfully!")

        # Test directory containing multiple images
        test_dir = "images/test_samples"
        if not os.path.exists(test_dir):
            print(f"❌ Test directory {test_dir} not found!")
            return

        print("\nTesting multiple images...")
        for image_file in os.listdir(test_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_dir, image_file)
                print(f"\nTesting: {image_file}")
                print("------------------------")
                
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                    result = model_service.predict(image_bytes)
                
                print(f"Results for {image_file}:")
                for key, value in result.items():
                    if key == "confidence":
                        print(f"{key}: {value:.2f}%")
                    else:
                        print(f"{key}: {value}")

        print("\n✅ Batch testing completed successfully!")

    except Exception as e:
        print(f"\n❌ Batch testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting single image test...")
    print("============================")
    test_model()
    
    print("\n\nStarting multiple images test...")
    print("================================")
    # test_multiple_images()