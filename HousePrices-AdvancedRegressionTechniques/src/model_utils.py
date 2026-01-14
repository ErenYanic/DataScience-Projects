import joblib
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler

# Compatibility Wrapper for Sklearn 1.6+ 
class SklearnWrapper(BaseEstimator, RegressorMixin):
    """
    A compatibility wrapper for third-party estimators (e.g., CatBoost, XGBoost)
    to ensure full compliance with Scikit-learn 1.6+ strict tag checks.

    This wrapper implements the new '__sklearn_tags__' method required by 
    sklearn 1.6+ to validate the estimator type.
    """
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

# Preprocessing Pipeline
class HousePricePreprocessor(BaseEstimator, TransformerMixin):
    """
    Centralised preprocessing pipeline that handles cleaning, feature engineering,
    encoding, and scaling. Ensures the data structure matches the final 47 features
    used during model training.
    """
    def __init__(self):
        # The definitive list of 47 features expected by the final model
        self.final_columns = [
            'LivingArea_and_OverallQuality', 'TotalSF', 'FullBath', 'GarageCars', 'BsmtFinSF1', 
            'Neighborhood', 'BedroomAbvGr', '2ndFlrSF', 'Fireplaces', 'TotalQual', 'TotalBsmtSF', 
            'GrLivArea', 'TotRmsAbvGrd', 'RemodAge', 'OverallQual_and_OverallCond', 'LuxuryScore', 
            'GarageArea', 'HouseAge', 'LotFrontage', 'TotalOutdoorSF', 'LotArea', 'BsmtQual_num', 
            'HalfBath', 'BsmtUnfSF', 'TotalCond', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 
            'BsmtExposure_No', 'BsmtExposure_None', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 
            'MSZoning_RM', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'BsmtFinType1_BLQ', 
            'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_None', 'BsmtFinType1_Rec', 
            'BsmtFinType1_Unf', 'FireplaceQu_Fa', 'FireplaceQu_Gd', 'FireplaceQu_None', 
            'FireplaceQu_Po', 'FireplaceQu_TA'
        ]
        
        # Mappings for ordinal features
        self.qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None': 0}
        
        # Columns designated for One-Hot Encoding
        self.ohe_columns = [
            'BsmtExposure', 'MSZoning', 'KitchenQual', 'BsmtFinType1', 'FireplaceQu'
        ]
        
        # State variables learnt during fitting
        self.neighborhood_map = {}
        self.global_target_mean = 0
        self.medians = {}
        self.scaler = StandardScaler()

    def _impute_missing(self, X):
        """Internal method to handle missing values using learnt medians."""
        X = X.copy()
        
        # Numerical imputation
        num_cols = ['LotFrontage', 'GarageArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'GarageCars']
        for col in num_cols:
            if col in X.columns:
                median_val = self.medians.get(col, 0)
                X[col] = X[col].fillna(median_val)
        
        # Categorical imputation
        cat_cols = ['BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'KitchenQual', 'MSZoning', 'BsmtQual']
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna('None')
                
        return X

    def _feature_engineering(self, df):
        """Applies domain-specific feature creation logic."""
        df = df.copy()
        
        # Map quality features to numeric values safely
        qual_cols = ['ExterQual', 'KitchenQual', 'BsmtQual', 'HeatingQC', 'GarageQual', 'ExterCond', 'BsmtCond', 'GarageCond']
        for col in qual_cols:
            if col in df.columns:
                df[f"{col}_num"] = df[col].map(self.qual_map).fillna(0)
            else:
                df[f"{col}_num"] = 0

        # Feature creation logic matching the training phase
        # Safely using .get() with default 0 for arithmetic operations
        df["LivingArea_and_OverallQuality"] = df.get("GrLivArea", 0) * df.get("OverallQual", 0)
        df["TotalSF"] = df.get("GrLivArea", 0) + df.get("TotalBsmtSF", 0)
        
        df["TotalQual"] = (df.get("OverallQual", 0) + df.get("ExterQual_num", 0) +
                           df.get("KitchenQual_num", 0) + df.get("BsmtQual_num", 0) +
                           df.get("HeatingQC_num", 0) + df.get("GarageQual_num", 0))

        df["RemodAge"] = df.get("YrSold", 0) - df.get("YearRemodAdd", 0)
        df["OverallQual_and_OverallCond"] = df.get("OverallQual", 0) * df.get("OverallCond", 0)

        # Luxury Score Calculation - FIXED FOR ROBUSTNESS
        # We check if the column exists to use .astype(), otherwise use scalar 0.
        
        # 1. Pool
        if "PoolArea" in df.columns:
            pool = (df["PoolArea"].fillna(0) > 0).astype(int)
        else:
            pool = 0
            
        # 2. Fireplaces
        if "Fireplaces" in df.columns:
            fire = (df["Fireplaces"].fillna(0) > 1).astype(int)
        else:
            fire = 0
            
        # 3. Garage
        if "GarageCars" in df.columns:
            garage = (df["GarageCars"].fillna(0) > 2).astype(int)
        else:
            garage = 0
            
        # 4. Overall Quality
        if "OverallQual" in df.columns:
            qual = (df["OverallQual"].fillna(0) >= 8).astype(int)
        else:
            qual = 0
            
        df["LuxuryScore"] = (pool * 3 + fire * 2 + garage * 2 + qual * 3)

        df["HouseAge"] = df.get("YrSold", 0) - df.get("YearBuilt", 0)

        # Outdoor Space
        total_porch = (df.get("OpenPorchSF", 0) + df.get("EnclosedPorch", 0) + 
                       df.get("3SsnPorch", 0) + df.get("ScreenPorch", 0))
        df["TotalOutdoorSF"] = total_porch + df.get("WoodDeckSF", 0)

        df["TotalCond"] = (df.get("OverallCond", 0) + df.get("ExterCond_num", 0) +
                           df.get("BsmtCond_num", 0) + df.get("GarageCond_num", 0))
                           
        return df

    def fit(self, X, y=None):
        """Learns statistical parameters (medians, target encoding, scaling) from training data."""
        X = X.copy()
        
        # 1. Learn medians for imputation
        for col in ['LotFrontage', 'GarageArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'GarageCars']:
            if col in X.columns:
                self.medians[col] = X[col].median()
        
        # 2. Apply engineering to prepare for encoding learning
        X = self._impute_missing(X)
        X = self._feature_engineering(X)
        
        # 3. Learn Target Encoding (Neighborhood)
        if y is not None and 'Neighborhood' in X.columns:
            self.neighborhood_map = y.groupby(X['Neighborhood']).mean().to_dict()
            self.global_target_mean = y.mean()
        
        # 4. Simulate transform to fit the Scaler correctly
        X_transformed = self.transform(X, fit_mode=True)
        self.scaler.fit(X_transformed)
        
        return self

    def transform(self, X, fit_mode=False):
        """Transforms raw data into the final processed format."""
        X = X.copy()
        
        # 1. Impute & Engineer
        X = self._impute_missing(X)
        X = self._feature_engineering(X)
        
        # 2. Apply Target Encoding
        if 'Neighborhood' in X.columns:
            # Fill unseen neighbourhoods with the global mean
            X['Neighborhood'] = X['Neighborhood'].map(self.neighborhood_map).fillna(self.global_target_mean)
        
        # 3. Apply One-Hot Encoding (using pandas for simplicity in this context)
        for col in self.ohe_columns:
            if col in X.columns:
                dummies = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X, dummies], axis=1)
                
        # 4. Ensure Column Consistency (Critical for Production)
        # Create missing columns with 0, drop extra columns
        for col in self.final_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Select exactly the 47 features in the correct order
        X_final = X[self.final_columns]
        
        # 5. Scaling
        if fit_mode:
            return X_final # Return unscaled data for scaler fitting
            
        X_scaled = self.scaler.transform(X_final)
        
        # Return as DataFrame to preserve column names
        return pd.DataFrame(X_scaled, columns=self.final_columns, index=X.index)


# Serialization Utilities
def save_production_model(model, metrics, filename="stacking_model_v1.joblib", output_dir="models"):
    """
    Serialises and saves the trained model along with its metadata.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    artefact = {
        "model": model,
        "metrics": metrics,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "meta_info": "Stacking Regressor Pipeline"
    }

    filepath = os.path.join(output_dir, filename)
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
    Loads the model artefact from the disk.
    Requires 'SklearnWrapper' and 'HousePricePreprocessor' to be importable.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    print(f"Loading model from: {filepath}...")
    
    try:
        artefact = joblib.load(filepath)
        
        if isinstance(artefact, dict) and "model" in artefact:
            model = artefact["model"]
            metadata = {k: v for k, v in artefact.items() if k != "model"}
            print(f"✓ Model loaded successfully (Trained on: {metadata.get('training_date', 'Unknown')})")
        else:
            model = artefact
            metadata = {}
            print("✓ Raw model loaded successfully (No metadata found).")
            
        return model, metadata

    except AttributeError as e:
        if "SklearnWrapper" in str(e) or "HousePricePreprocessor" in str(e):
            raise ImportError(
                "CRITICAL ERROR: Helper classes are missing! "
                "Ensure src.model_utils is imported correctly."
            ) from e
        raise e