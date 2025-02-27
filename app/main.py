# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.quality_model import QualityModel
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
MODEL_PATH = os.getenv("MODEL_PATH", "models/deep_model.h5")
quality_model = QualityModel(MODEL_PATH)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to predict fruit quality from uploaded image
    """
    try:
        contents = await file.read()
        prediction = quality_model.predict(contents)
        print(prediction)
        return prediction
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

