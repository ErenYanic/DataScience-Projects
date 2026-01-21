import joblib
import pandas as pd
import numpy as np
import os
import sys
import json
import warnings

# Add parent directory to system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom classes to ensure correct unpickling
from src.model_utils import HousePricePreprocessor, SklearnWrapper

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class HousePricePredictor:
    """
    Inference engine for House Price Prediction.
    Loads trained artefacts (model and preprocessor) to generate predictions.
    """
    
    def __init__(self, model_dir="models", model_filename="final_model.joblib"):
        # Define absolute paths for model artefacts
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(self.base_dir, model_dir)
        self.model_path = os.path.join(self.model_dir, model_filename)
        self.prep_path = os.path.join(self.model_dir, "preprocessor.joblib")
        
        self._load_artefacts()

    def _load_artefacts(self):
        """Loads serialised model and preprocessor from disk."""
        print(f"Loading artefacts from: {self.model_dir}...")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.prep_path):
            raise FileNotFoundError(
                f"Critical Error: Artefacts missing.\n"
                f"Expected: {self.model_path}\n"
                f"Expected: {self.prep_path}\n"
                f"Ensure the training pipeline ('src/train.py') has been executed."
            )

        try:
            # Load Preprocessor
            self.preprocessor = joblib.load(self.prep_path)
            
            # Load Model (Handle metadata dictionary if present)
            loaded_obj = joblib.load(self.model_path)
            
            if isinstance(loaded_obj, dict) and "model" in loaded_obj:
                self.model = loaded_obj["model"]
                meta = loaded_obj.get("metrics", {})
                print(f"Model loaded successfully.")
                if 'val_rmsle' in meta:
                    print(f"  - Model Performance (Val RMSLE): {meta['val_rmsle']:.4f}")
            else:
                self.model = loaded_obj
                print("Raw model loaded.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load artefacts: {e}")

    def predict(self, raw_data_dict):
        """
        Generates a price prediction for a single house.

        Parameters
        ----------
        raw_data_dict : dict
            Dictionary containing raw feature values.

        Returns
        -------
        float
            Predicted price in USD.
        """
        # 1. Convert dictionary to single-row DataFrame
        input_df = pd.DataFrame([raw_data_dict])
        
        # 2. Transform data (Imputation, encoding, scaling)
        try:
            processed_data = self.preprocessor.transform(input_df)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")
        
        # 3. Predict (Log Scale)
        log_prediction = self.model.predict(processed_data)[0]
        
        # 4. Inverse Transform (Log -> Real Price)
        # Apply expm1 to revert log1p transformation used during training
        real_prediction = np.expm1(log_prediction)
        
        return real_prediction

def main():
    # --- Example Usage Scenario ---
    
    # Configuration: Ensure this matches the output filename from training
    MODEL_FILENAME = "final_model.joblib" 
    
    try:
        predictor = HousePricePredictor(model_filename=MODEL_FILENAME)
    except FileNotFoundError as e:
        print(e)
        return

    # Sample input data simulating an API payload
    sample_house = {
        "MSSubClass": 120,
        "MSZoning": "RP",
        "LotFrontage": 85.0,
        "LotArea": 8765,
        "Neighborhood": "SWISU",
        "OverallQual": 4,
        "OverallCond": 5,
        "YearBuilt": 1995,
        "YearRemodAdd": 2003,
        "BsmtFinSF1": 798,
        "TotalBsmtSF": 856,
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "GrLivArea": 2345,
        "FullBath": 3,
        "BedroomAbvGr": 4,
        "TotRmsAbvGrd": 9,
        "Fireplaces": 2,
        "GarageCars": 2,
        "GarageArea": 650,
        # Missing features are handled automatically by the preprocessor
    }

    print("\n--- Prediction Test ---")
    print(f"Input Features: {json.dumps(sample_house, indent=2)}")
    
    # Execute prediction
    try:
        price = predictor.predict(sample_house)
        print(f"\nPredicted Sale Price: ${price:,.2f}")
    except Exception as e:
        print(f"\nPrediction Error: {e}")

if __name__ == "__main__":
    main()