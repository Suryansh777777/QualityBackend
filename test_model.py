from app.quality_model import FruitQualityModel
import os
def test_model():
    try:
        # Initialize the model
        print("Initializing model...")
        model_service = FruitQualityModel()
        print("✅ Model loaded successfully!")

        # Test with a sample image
        test_image_path = "image3.png"  # Update this path
        if not os.path.exists(test_image_path):
            print("❌ Test image not found!")
            return

        print("\nTesting prediction...")
        result = model_service.predict(test_image_path)
        
        print("\nPrediction Results:")
        print("==================")
        print(f"Predicted Class: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Quality Score: {result['quality_score']:.2f}")
        
        print("\nNutritional Data:")
        print("================")
        for key, value in result['nutritional_data'].items():
            print(f"{key.capitalize()}: {value}")
        
        print("\nPhysical Properties:")
        print("==================")
        print(f"Weight: {result['physical_properties']['weight']}g")
        print(f"Size: {result['physical_properties']['size']['length']}cm x {result['physical_properties']['size']['width']}cm")
        print(f"Firmness: {result['physical_properties']['firmness']}%")
        
        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\nDebug information:")
        print("=================")
        print(f"Python path: {os.environ.get('PYTHONPATH')}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Model directory contents: {os.listdir('models') if os.path.exists('models') else 'models directory not found'}")

if __name__ == "__main__":
    test_model()