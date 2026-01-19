import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from sklearn.model_selection import train_test_split

# Add parent directory to path to allow imports from 'src'
# This ensures the script runs correctly regardless of the execution directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_utils import HousePricePreprocessor, save_production_model
from src.training_utils import train_and_optimize_models, train_dynamic_stacking

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """
    Main execution pipeline for the House Price Prediction project.
    
    Workflow:
    1. Load and Split Data (Raw).
    2. Preprocessing (Fit & Transform).
    3. Hyperparameter Optimisation (Optuna).
    4. Stacking Model Training.
    5. Serialisation (Save Artefacts).
    """
    
    # --------------------------------------------------------------------------
    # 1. SETUP & DATA LOADING
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STARTING AUTOMATED TRAINING PIPELINE")
    print("="*60)

    # Define dynamic paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[1/5] Loading dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Critical Error: Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    # Separate Target and Features
    X = df.drop("SalePrice", axis=1)
    y = np.log1p(df["SalePrice"]) # Apply Log transformation to target
    
    # Split Data (Must match the logic used in R&D to prevent leakage)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   - Training samples:   {X_train.shape[0]}")
    print(f"   - Validation samples: {X_val.shape[0]}")

    # --------------------------------------------------------------------------
    # 2. PREPROCESSING
    # --------------------------------------------------------------------------
    print("\n[2/5] Executing Preprocessing Pipeline...")
    
    # Initialise the custom preprocessor
    preprocessor = HousePricePreprocessor()
    
    # Fit ONLY on training data to avoid data leakage
    preprocessor.fit(X_train, y_train)
    
    # Transform both sets
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    # Save the preprocessor artefact immediately
    prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, prep_path)
    print(f"   ✓ Preprocessor saved to: {prep_path}")

    # --------------------------------------------------------------------------
    # 3. HYPERPARAMETER OPTIMISATION
    # --------------------------------------------------------------------------
    print("\n[3/5] Running Hyperparameter Optimisation (Optuna)...")
    print("   - Note: Training 5 robust models. This might take a few minutes.")
    
    # Using a moderate number of trials for production stability vs. time trade-off
    optuna_results, summary_df = train_and_optimize_models(
        X_train=X_train_proc,
        y_train=y_train,
        X_val=X_val_proc,
        y_val=y_val,
        models=['catboost', 'xgboost', 'elasticnet', 'lightgbm', 'random_forest'],
        n_trials=200,  # Adjustable: Increase for better accuracy, decrease for speed
        cv=5,
        verbose=1
    )
    
    print("\n   --- Optimisation Leaderboard ---")
    print(summary_df[['Model', 'CV_RMSLE', 'Val_RMSLE']].to_string(index=False))

    # --------------------------------------------------------------------------
    # 4. STACKING MODEL TRAINING
    # --------------------------------------------------------------------------
    print("\n[4/5] Training Final Stacking Ensemble...")
    
    final_stack_model, final_metrics = train_dynamic_stacking(
        results=optuna_results, 
        X_train=X_train_proc, 
        y_train=y_train,
        X_val=X_val_proc, 
        y_val=y_val, 
        included_models=['catboost', 'xgboost', 'elasticnet', 'lightgbm', 'random_forest'],
        verbose=1
    )

    # --------------------------------------------------------------------------
    # 5. FINAL SERIALISATION (With Versioning)
    # --------------------------------------------------------------------------
    print("\n[5/5] Saving Production Artefacts...")
    
    # 1. Generate a timestamp for versioning (e.g., 20260119_153021)
    import datetime
    import shutil
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. Define filenames
    versioned_filename = f"model_{timestamp}.joblib"
    latest_filename = "final_model.joblib"
    
    # 3. Save the versioned copy (The Archive)
    # This ensures we never lose a trained model history.
    versioned_path = save_production_model(
        final_stack_model, 
        final_metrics, 
        filename=versioned_filename, 
        output_dir=MODELS_DIR
    )
    print(f"   ✓ Archived version saved: {versioned_filename}")
    
    # 4. Update the 'Latest' copy (The Active Production Model)
    # We copy the newly created versioned file to 'final_model.joblib'.
    # Predict.py will always look for 'final_model.joblib'.
    latest_path = os.path.join(MODELS_DIR, latest_filename)
    shutil.copy(versioned_path, latest_path)
    print(f"   ✓ Active production model updated: {latest_filename}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Active Model: {latest_path}")
    print(f"Validation RMSLE: {final_metrics['val_rmsle']:.5f}")
    print("="*60)

if __name__ == "__main__":
    main()