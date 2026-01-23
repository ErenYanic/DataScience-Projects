import sys
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add project root to system path to enable imports from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import HousePricePredictor

# ==============================================================================
# 1. API CONFIGURATION & DATA MODELS
# ==============================================================================

# Initialise FastAPI application
# Includes metadata for the auto-generated documentation
app = FastAPI(
    title="House Price Prediction API",
    description="Production-ready API for estimating house prices in HousePrices - Advanced Regression Techniques.",
    version="1.0.0"
)

# Global variable to hold the predictor instance
# Loaded on startup to avoid reloading heavy artefacts for every request
predictor = None

class HouseFeatures(BaseModel):
    """
    Input schema definition using Pydantic.
    Validates that the input payload is a dictionary of features.
    
    Example:
    {
        "OverallQual": 7,
        "GrLivArea": 1500,
        "Neighborhood": "CollgCr",
        ...
    }
    """
    features: Dict[str, Any]

# ==============================================================================
# 2. LIFECYCLE EVENTS
# ==============================================================================

@app.on_event("startup")
def load_model():
    """
    Executes on application startup.
    Loads the trained model and preprocessor into memory.
    """
    global predictor
    try:
        # Initialise the inference engine
        # Ensure 'final_model.joblib' exists in the 'models' directory
        predictor = HousePricePredictor(model_filename="final_model.joblib")
        print("API Startup: Model artefacts loaded successfully.")
    except Exception as e:
        print(f"API Startup Failed: {e}")
        # In a real scenario, we might want to stop the server here
        pass

# ==============================================================================
# 3. ENDPOINTS
# ==============================================================================

@app.get("/")
def health_check():
    """
    Simple health check endpoint.
    Used to verify if the API server is running and responsive.
    """
    return {"status": "active", "message": "House Price Prediction API is online."}

@app.post("/predict")
def predict_price(payload: HouseFeatures):
    """
    Main inference endpoint.
    
    Parameters
    ----------
    payload : HouseFeatures
        JSON object containing house characteristics.
        
    Returns
    -------
    dict
        Predicted price in USD.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        # Extract raw dictionary from Pydantic model
        raw_data = payload.features
        
        # Generate prediction using our logic in src/predict.py
        price = predictor.predict(raw_data)
        
        return {
            "prediction_usd": round(price, 2),
            "status": "success"
        }
        
    except Exception as e:
        # Handle unexpected errors during inference
        raise HTTPException(status_code=400, detail=str(e))

# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Run the server using Uvicorn
    # '0.0.0.0' makes it accessible externally, port 8000 is standard
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)