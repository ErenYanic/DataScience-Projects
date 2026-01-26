import joblib
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.preprocessing import MaxAbsScaler

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
    encoding, and scaling. Ensures the data structure matches the final features
    used during model training after data leakage fixes.
    """
    def __init__(self):
        # The definitive list of features expected by the final model (derived from X_train_proc columns)
        self.final_columns = [
            'LivingArea_and_OverallQuality', 'TotalSF', 'BsmtFinSF1', 'FullBath', 'GarageCars', 
            '2ndFlrSF', 'TotalBsmtSF', 'Neighborhood', '1stFlrSF', 'GrLivArea', 'TotalQual', 
            'BedroomAbvGr', 'OverallQual_and_OverallCond', 'TotalBath', 'RemodAge', 'LotFrontage', 
            'OverallQual', 'HalfBath', 'TotalCond', 'GarageArea', 'KitchenQual_num', 'Kitchen_to_TotRms', 
            'BsmtQual_num', 'QualityLevel_Poor', 'QualityLevel_Average', 'QualityLevel_Good', 
            'QualityLevel_Excellent', 'FireplaceQu_Fa', 'FireplaceQu_Gd', 'FireplaceQu_None', 
            'FireplaceQu_Po', 'FireplaceQu_TA', 'Neighborhood_Price_Level_Lower_Middle', 
            'Neighborhood_Price_Level_Luxury', 'Neighborhood_Price_Level_Middle', 
            'Neighborhood_Price_Level_Upper_Middle', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 
            'MSZoning_RM', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 
            'BsmtFinType1_None', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'KitchenQual_Fa', 
            'KitchenQual_Gd', 'KitchenQual_TA', 'MSSubClass_1.5storey_unf', 'MSSubClass_1storey_1945-', 
            'MSSubClass_1storey_1946+', 'MSSubClass_1storey_PUD_1946+', 'MSSubClass_1storey_unf_attic', 
            'MSSubClass_2.5storey_all_ages', 'MSSubClass_2family_conversion', 'MSSubClass_2storey_1945-', 
            'MSSubClass_2storey_1946+', 'MSSubClass_2storey_PUD_1946+', 'MSSubClass_PUD_multilevel', 
            'MSSubClass_duplex_all_style_age', 'MSSubClass_split_foyer', 'MSSubClass_split_multilevel'
        ]
        
        # Mappings for ordinal features
        self.qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, 'None': 0}
        
        # Columns designated for One-Hot Encoding
        # Note: Added QualityLevel, Neighborhood_Price_Level, and MSSubClass as they appear in final columns
        self.ohe_columns = [
            'BsmtExposure', 'MSZoning', 'KitchenQual', 'BsmtFinType1', 'FireplaceQu',
            'QualityLevel', 'Neighborhood_Price_Level', 'MSSubClass'
        ]
        
        # Neighborhood mapping for grouping
        self.neighborhood_tiers = {
            "Luxury": ["NoRidge", "NridgHt", "StoneBr"],
            "Upper_Middle": ["Timber", "Veenker", "Somerst", "ClearCr", "Crawfor"],
            "Middle": ["CollgCr", "Blmngtn", "Gilbert", "NWAmes", "SawyerW"],
            "Lower_Middle": ["Mitchel", "NAmes", "NPkVill", "SWISU", "Blueste", 
                            "Sawyer", "OldTown", "Edwards", "BrkSide"],
            "Budget": ["BrDale", "IDOTRR", "MeadowV"]
        }
        
        # State variables learnt during fitting
        self.neighborhood_map = {}
        self.global_target_mean = 0
        self.medians = {}
        self.lot_frontage_by_neighborhood = {}
        self.lot_frontage_global_median = 0
        self.electrical_mode = None
        self.scaler = MaxAbsScaler()

    def _impute_missing(self, X):
        """
        Internal method to handle missing values using learnt parameters.
        Mirrors the logic from create_missing_value_handler in the notebook.
        """
        X = X.copy()
        
        # 1. Categorical: Fill with "None" (feature doesn't exist)
        cat_none_cols = [
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "Alley", "PoolQC", "MiscFeature", "FireplaceQu", "MasVnrType", "Fence",
            "KitchenQual", "MSZoning"
        ]
        for col in cat_none_cols:
            if col in X.columns:
                X[col] = X[col].fillna("None")

        # 2. Numerical: Fill with 0 (feature doesn't exist)
        num_zero_cols = [
            "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", 
            "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", 
            "GarageCars", "GarageArea"
        ]
        for col in num_zero_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # 3. LotFrontage: Use learnt neighbourhood-specific medians
        if "LotFrontage" in X.columns and "Neighborhood" in X.columns:
            X["LotFrontage"] = X.apply(
                lambda row: self.lot_frontage_by_neighborhood.get(
                    row["Neighborhood"], 
                    self.lot_frontage_global_median
                ) if pd.isna(row["LotFrontage"]) else row["LotFrontage"],
                axis=1
            )

        # 4. GarageYrBlt: Sentinel value for missing
        if "GarageYrBlt" in X.columns:
            X["GarageYrBlt"] = X["GarageYrBlt"].fillna(-1)

        # 5. Electrical: Use learnt mode
        if "Electrical" in X.columns and self.electrical_mode is not None:
            X["Electrical"] = X["Electrical"].fillna(self.electrical_mode)

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
        df["LivingArea_and_OverallQuality"] = df.get("GrLivArea", 0) * df.get("OverallQual", 0)
        df["TotalSF"] = df.get("GrLivArea", 0) + df.get("TotalBsmtSF", 0)
        
        df["TotalQual"] = (df.get("OverallQual", 0) + df.get("ExterQual_num", 0) +
                           df.get("KitchenQual_num", 0) + df.get("BsmtQual_num", 0) +
                           df.get("HeatingQC_num", 0) + df.get("GarageQual_num", 0))

        df["RemodAge"] = df.get("YrSold", 0) - df.get("YearRemodAdd", 0)
        df["OverallQual_and_OverallCond"] = df.get("OverallQual", 0) * df.get("OverallCond", 0)

        # 1. Quality Level Grouping (Binning)
        if "OverallQual" in df.columns:
            df["QualityLevel"] = pd.cut(
                df["OverallQual"],
                bins=[0, 2, 4, 5, 7, 10],
                labels=["Very Poor", "Poor", "Average", "Good", "Excellent"],
                include_lowest=True
            ).astype(str) # Convert to string for OHE
        else:
            df["QualityLevel"] = "Average" # Default fallback

        # 2. Neighborhood Price Level
        neighborhood_price_map = {neighborhood: tier 
                                for tier, neighborhoods in self.neighborhood_tiers.items() 
                                for neighborhood in neighborhoods}
        
        if "Neighborhood" in df.columns:
            df["Neighborhood_Price_Level"] = df["Neighborhood"].map(neighborhood_price_map).fillna("Other")
        else:
            df["Neighborhood_Price_Level"] = "Other"

        # 3. Total Bathrooms
        # Using .get() with default 0 to be safe against missing columns
        df["TotalBath"] = (df.get("FullBath", 0) +
                           0.5 * df.get("HalfBath", 0) +
                           df.get("BsmtFullBath", 0) +
                           0.5 * df.get("BsmtHalfBath", 0))

        # 4. Kitchen Ratio
        # Avoid division by zero
        tot_rms = df.get("TotRmsAbvGrd", 0)
        kitchens = df.get("KitchenAbvGr", 0)
        df["Kitchen_to_TotRms"] = np.where(tot_rms > 0, kitchens / tot_rms, 0)
        
        # 5. MSSubClass Mapping (if needed for OHE consistency)
        # Note: MSSubClass is numeric but treated as categorical. 
        # In the provided columns, it appears as 'MSSubClass_2storey...' which implies conversion to string.
        if "MSSubClass" in df.columns:
            # Re-apply the mapping logic if it's still numeric
            # Assuming 'mssubclass_map' was applied earlier or needs to be applied here.
            # Based on column names, let's ensure it's a string for OHE.
            # Using a simplified mapping logic based on your previous file or treating as str
            # If explicit mapping is required, it should be defined here. 
            # For now, converting to string enables OHE to generate 'MSSubClass_XX'.
            # However, looking at your column list 'MSSubClass_1storey_1946+', 
            # it implies a specific text mapping was used. Let's include that mapping.
            
            mssubclass_map = {
                20: '1storey_1946+', 30: '1storey_1945-', 40: '1storey_unf_attic',
                45: '1.5storey_unf', 50: '1.5storey_fin', 60: '2storey_1946+',
                70: '2storey_1945-', 75: '2.5storey_all_ages', 80: 'split_multilevel',
                85: 'split_foyer', 90: 'duplex_all_style_age', 120: '1storey_PUD_1946+',
                150: '1.5storey_PUD_all', 160: '2storey_PUD_1946+', 180: 'PUD_multilevel',
                190: '2family_conversion'
            }
            # Only map if it's numeric, otherwise assume it's already mapped
            if pd.api.types.is_numeric_dtype(df["MSSubClass"]):
                 df["MSSubClass"] = df["MSSubClass"].map(mssubclass_map).fillna("Other")

        df["TotalCond"] = (df.get("OverallCond", 0) + df.get("ExterCond_num", 0) +
                           df.get("BsmtCond_num", 0) + df.get("GarageCond_num", 0))
                           
        return df

    def fit(self, X, y=None):
        """Learns statistical parameters (medians, target encoding, scaling) from training data."""
        X = X.copy()
        
        # 1. Learn medians for numerical imputation
        num_median_cols = ['GarageArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'GarageCars']
        for col in num_median_cols:
            if col in X.columns:
                self.medians[col] = X[col].median()
        
        # 2. Learn LotFrontage neighbourhood-specific medians (to avoid data leakage)
        if "LotFrontage" in X.columns and "Neighborhood" in X.columns:
            self.lot_frontage_by_neighborhood = X.groupby("Neighborhood")["LotFrontage"].median().to_dict()
            self.lot_frontage_global_median = X["LotFrontage"].median()
        
        # 3. Learn Electrical mode
        if "Electrical" in X.columns:
            modes = X["Electrical"].mode()
            self.electrical_mode = modes[0] if not modes.empty else None
        
        # 4. Apply engineering to prepare for encoding learning
        X = self._impute_missing(X)
        X = self._feature_engineering(X)
        
        # 5. Learn Target Encoding (Neighborhood)
        if y is not None and 'Neighborhood' in X.columns:
            self.neighborhood_map = y.groupby(X['Neighborhood']).mean().to_dict()
            self.global_target_mean = y.mean()
        
        # 6. Simulate transform to fit the Scaler correctly
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
        
        # 3. Apply One-Hot Encoding
        for col in self.ohe_columns:
            if col in X.columns:
                dummies = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X, dummies], axis=1)
                
        # 4. Ensure Column Consistency (Critical for Production)
        # Create missing columns with 0, drop extra columns
        for col in self.final_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Select exactly the final features in the correct order
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