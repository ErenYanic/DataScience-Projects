import joblib
import os
import datetime
from sklearn.base import BaseEstimator, RegressorMixin, clone

# Compatibility Wrapper for Sklearn 1.6+
class SklearnWrapper(BaseEstimator, RegressorMixin):
    """
    A compatibility wrapper for third-party estimators (e.g., CatBoost, XGBoost)
    to ensure full compliance with Scikit-learn 1.6+ strict tag checks.

    This wrapper implements the new '__sklearn_tags__' method required by 
    sklearn 1.6+ to validate the estimator type.
    """
    
    # Legacy support (Scikit-learn < 1.6)
    _estimator_type = "regressor"

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)
    
    def get_params(self, deep=True):
        return {"estimator": self.estimator}

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params["estimator"]
        return self

    # --- CRITICAL FIX FOR SKLEARN 1.6+ ---
    def __sklearn_tags__(self):
        # Call the parent's tags generation (BaseEstimator)
        tags = super().__sklearn_tags__()
        # Explicitly set the estimator type in the new tags structure
        tags.estimator_type = "regressor"
        return tags
    
def save_production_model(model, metrics, filename="stacking_model_v1.joblib", output_dir="models"):
    """
    Serialises and saves the trained model along with its metadata to the disk.

    This function creates a dictionary artefact containing:
    - The trained model object.
    - Performance metrics (RMSLE, R2, etc.).
    - Timestamp of training.
    - Framework versions (useful for debugging compatibility issues).

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The trained StackingRegressor (or any other sklearn estimator).
    metrics : dict
        A dictionary containing evaluation scores (e.g., {'val_rmsle': 0.13}).
    filename : str, default='stacking_model_v1.joblib'
        The name of the output file.
    output_dir : str, default='models'
        The directory where the model will be stored.

    Returns
    -------
    filepath : str
        The full path to the saved model artefact.
    """
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Prepare metadata
    artefact = {
        "model": model,
        "metrics": metrics,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "meta_info": "Stacking Regressor with CatBoost, XGBoost, LGBM, RF, ElasticNet"
    }

    # Define full path
    filepath = os.path.join(output_dir, filename)

    # Save using joblib (more efficient than pickle for numpy arrays)
    try:
        joblib.dump(artefact, filepath)
        print(f"{'='*60}")
        print(f"Model successfully saved to: {filepath}")
        print(f"Metrics stored: {metrics}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"Error whilst saving the model: {e}")
        raise

    return filepath

def load_production_model(filepath):
    """
    Loads the model artefact from the disk for inference.

    CRITICAL NOTE: The 'SklearnWrapper' class definition MUST be present 
    in the scope where this function is called, otherwise joblib will fail.

    Parameters
    ----------
    filepath : str
        Path to the .joblib file.

    Returns
    -------
    model : sklearn.base.BaseEstimator
        The trained model ready for prediction.
    metadata : dict
        The metadata dictionary (metrics, date, etc.).
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    print(f"Loading model from: {filepath}...")
    
    try:
        artefact = joblib.load(filepath)
        
        # Check if it's our custom dictionary format or just a raw model
        if isinstance(artefact, dict) and "model" in artefact:
            model = artefact["model"]
            metadata = {k: v for k, v in artefact.items() if k != "model"}
            print(f"✓ Model loaded successfully (Trained on: {metadata.get('training_date', 'Unknown')})")
        else:
            # Fallback for raw models
            model = artefact
            metadata = {}
            print("✓ Raw model loaded successfully (No metadata found).")
            
        return model, metadata

    except AttributeError as e:
        if "SklearnWrapper" in str(e):
            raise ImportError(
                "CRITICAL ERROR: 'SklearnWrapper' class is missing! "
                "You must define or import the SklearnWrapper class before loading this model."
            ) from e
        raise e