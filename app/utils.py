# app/utils.py
import os
import logging
from PIL import Image
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quality-analysis")

def ensure_directory_exists(directory_path):
    """Ensure that a directory exists, creating it if necessary"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def validate_image(image_data):
    """Validate that image data is a valid image"""
    try:
        img = Image.open(io.BytesIO(image_data))
        img.verify()  # Verify it's a valid image
        return True
    except Exception as e:
        logger.error(f"Invalid image: {str(e)}")
        return False