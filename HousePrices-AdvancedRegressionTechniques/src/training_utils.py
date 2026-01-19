import time
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Import the wrapper from our local utility module to ensure compatibility
from src.model_utils import SklearnWrapper

def train_and_optimize_models(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    models='all',
    cv=5,
    n_trials=100,
    random_state=42,
    n_jobs=-1,
    verbose=1
):
    """
    Trains and optimizes multiple regression models using Optuna.
    Pruning: Stops unpromising trials early (saves RAM/time) for SVR, RF, XGB, LGBM, and CatBoost

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features (already encoded and scaled)
    y_train : pd.Series or np.ndarray
        Training target variable
    X_val : pd.DataFrame or np.ndarray, optional
        Validation features (for final evaluation)
    y_val : pd.Series or np.ndarray, optional
        Validation target variable
    models : str or list, default='all'
        Which models to train:
        - 'all': Train all 7 models
        - list: ['elasticnet', 'xgboost', 'lightgbm', ...]
    cv : int, default=5
        Number of cross-validation folds
    n_trials : int, default=100
        Number of Optuna trials per model
    random_state : int, default=42
        Random state for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = use all cores)
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed)
    
    Returns
    -------
    results : dict
        Detailed results for each model with best_model, best_params, scores
    results_df : pd.DataFrame
        Summary DataFrame sorted by CV score
    
    Examples
    --------
    >>> # Train all models
    >>> results, df = train_and_optimize_models(X_train_proc, y_train, X_val_proc, y_val)
    >>> 
    >>> # Train only specific models (faster)
    >>> results, df = train_and_optimize_models(
    ...     X_train_proc, y_train, 
    ...     models=['elasticnet', 'xgboost', 'catboost'],
    ...     n_trials=50
    ... )
    """
    
    # Setup cross-validation
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Suppress Optuna logs if verbose=0
    optuna.logging.set_verbosity(optuna.logging.WARNING if verbose == 0 else optuna.logging.INFO)
    
    # Objective Functions
    # Fast Models (Use cross_val_score, pruning is less critical here)
    def objective_elasticnet(trial):
        params = {
            'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
            'max_iter': trial.suggest_categorical('max_iter', [1000, 2000, 5000, 10000]),
            'random_state': random_state
        }
        model = ElasticNet(**params)
        scores = cross_val_score(model, X_train, y_train, cv=kfold, 
                                 scoring='neg_mean_squared_log_error', n_jobs=n_jobs)
        return np.sqrt(-scores.mean())
    
    def objective_knn(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
            'n_jobs': n_jobs
        }
        model = KNeighborsRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=kfold,
                                 scoring='neg_mean_squared_log_error', n_jobs=n_jobs)
        return np.sqrt(-scores.mean())
    
    # Heavy Models (Manual CV loop for Pruning support)
    def objective_svr(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.2)
        }
        
        fold_scores = []
        # Manual CV loop to enable pruning
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = SVR(**params)
            model.fit(X_tr, y_tr)
            
            # Predict and Clip (safety for RMSLE)
            y_pred = np.clip(model.predict(X_val_fold), 0, None)
            rmsle = np.sqrt(mean_squared_log_error(y_val_fold, y_pred))
            fold_scores.append(rmsle)
            
            # Report intermediate result to Optuna
            trial.report(rmsle, fold_idx)
            
            # Prune if the trial is unpromising
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(fold_scores)
    
    def objective_random_forest(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 30, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': random_state,
            'n_jobs': n_jobs 
        }
        
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)
            
            y_pred = np.clip(model.predict(X_val_fold), 0, None)
            rmsle = np.sqrt(mean_squared_log_error(y_val_fold, y_pred))
            fold_scores.append(rmsle)
            
            trial.report(rmsle, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(fold_scores)
    
    def objective_xgboost(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'random_state': random_state,
            'n_jobs': n_jobs,
            'verbosity': 0
        }
        
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            y_pred = np.clip(model.predict(X_val_fold), 0, None)
            rmsle = np.sqrt(mean_squared_log_error(y_val_fold, y_pred))
            fold_scores.append(rmsle)
            
            trial.report(rmsle, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(fold_scores)
    
    def objective_lightgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'random_state': random_state,
            'n_jobs': n_jobs,
            'verbose': -1
        }
        
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            y_pred = np.clip(model.predict(X_val_fold), 0, None)
            rmsle = np.sqrt(mean_squared_log_error(y_val_fold, y_pred))
            fold_scores.append(rmsle)
            
            trial.report(rmsle, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(fold_scores)
    
    def objective_catboost(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300, step=50),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 7),
            'random_state': random_state,
            'verbose': 0
        }

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr)
            y_pred = np.clip(model.predict(X_val_fold), 0, None)
            rmsle = np.sqrt(mean_squared_log_error(y_val_fold, y_pred))
            fold_scores.append(rmsle)
            
            trial.report(rmsle, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(fold_scores)
    
    # Model registry
    MODEL_REGISTRY = {
        'elasticnet': objective_elasticnet,
        'knn': objective_knn,
        'svr': objective_svr,
        'random_forest': objective_random_forest,
        'xgboost': objective_xgboost,
        'lightgbm': objective_lightgbm,
        'catboost': objective_catboost
    }
    
    # Determine which models to train
    if models == 'all':
        selected_models = list(MODEL_REGISTRY.keys())
    else:
        selected_models = models
    
    # Validate model names
    invalid_models = set(selected_models) - set(MODEL_REGISTRY.keys())
    if invalid_models:
        raise ValueError(f"Invalid model names: {invalid_models}. Valid: {list(MODEL_REGISTRY.keys())}")
    

    # Pruner configuration
    # n_warmup_steps=2: Pruning starts after the 2nd fold (0 and 1 are safe).
    # This is a conservative approach to avoid killing potentially good models too early.
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    
    # Results storage
    results = {}
    summary_data = []
    
    if verbose >= 1:
        print(f"\n{'='*70}")
        print(f"Training {len(selected_models)} models with Optuna (RMSLE metric)")
        print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Trials per model: {n_trials} | CV folds: {cv}")
        if X_val is not None:
            print(f"Validation set: {X_val.shape[0]} samples")
        print(f"{'='*70}\n")
    
    # Train each model
    for i, model_name in enumerate(selected_models, 1):
        if verbose >= 1:
            print(f"[{i}/{len(selected_models)}] Optimizing {model_name.upper()} ({n_trials} trials)...")
        
        start_time = time.time()

        try:
            # Create study for each model
            study = optuna.create_study(
                direction='minimize',
                pruner=pruner,
                sampler=optuna.samplers.TPESampler(seed=random_state)
            )
            
            objective_func = MODEL_REGISTRY[model_name]
            
            # Run optimization
            # n_jobs=1: Using sequential optimization to save RAM and allow models 
            # to use internal parallelization (n_jobs=-1).
            study.optimize(
                objective_func,
                n_trials=n_trials,
                show_progress_bar=(verbose >= 1),
                n_jobs=1,
                catch=(Exception,) # Catches model-specific errors without crashing the loop
            )
            
            # Best params and score
            best_params = study.best_params
            cv_score = study.best_value
            
            # Re-instantiate the best model
            if model_name == 'elasticnet':
                best_model = ElasticNet(**best_params)
            elif model_name == 'knn':
                best_model = KNeighborsRegressor(**best_params, n_jobs=n_jobs)
            elif model_name == 'svr':
                best_model = SVR(**best_params)
            elif model_name == 'random_forest':
                best_model = RandomForestRegressor(**best_params, n_jobs=n_jobs)
            elif model_name == 'xgboost':
                best_model = XGBRegressor(**best_params)
            elif model_name == 'lightgbm':
                best_model = LGBMRegressor(**best_params)
            elif model_name == 'catboost':
                best_model = CatBoostRegressor(**best_params)
            
            # Fit on full training set
            best_model.fit(X_train, y_train)
            
            # Metrics on Train
            y_train_pred = np.clip(best_model.predict(X_train), 0, None)
            train_rmsle = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            
            # Metrics on Validation (if provided)
            val_rmsle = val_mae = val_r2 = None
            if X_val is not None and y_val is not None:
                y_val_pred = np.clip(best_model.predict(X_val), 0, None)
                val_rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))
                val_mae = mean_absolute_error(y_val, y_val_pred)
                val_r2 = r2_score(y_val, y_val_pred)
            
            elapsed = time.time() - start_time
            
            # Store results
            results[model_name] = {
                'best_model': best_model,
                'best_params': best_params,
                'cv_rmsle': cv_score,
                'train_rmsle': train_rmsle,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'val_rmsle': val_rmsle,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'fit_time': elapsed,
                'study': study
            }
            
            summary_data.append({
                'Model': model_name,
                'CV_RMSLE': cv_score,
                'Train_RMSLE': train_rmsle,
                'Train_R2': train_r2,
                'Train_MAE': train_mae,
                'Val_RMSLE': val_rmsle,
                'Val_R2': val_r2,
                'Val_MAE': val_mae,
                'Time(s)': elapsed
            })
        
            
            if verbose >= 1:
                print(f"--- {model_name.upper()} RESULTS ---")
                print(f"CV RMSLE: {cv_score:.4f}")
                print(f"Train RMSLE: {train_rmsle:.4f} | MAE: {train_mae:.4f} | R2: {train_r2:.4f}")
                if val_rmsle: print(f"Val RMSLE: {val_rmsle:.4f} | MAE: {val_mae:.4f} | R2: {val_r2:.4f}")
                print(f"Time: {elapsed:.2f}s | Trials: {len(study.trials)}")
                print(f"Best params: {best_params}")
                print()
                
        except Exception as e:
            print(f"!! ERROR optimizing {model_name}: {str(e)}")
            continue

    # Final summary
    results_df = pd.DataFrame(summary_data).sort_values('CV_RMSLE').reset_index(drop=True)
    
    if verbose >= 1:
        print(f"{'='*70}")
        print("RESULTS SUMMARY (sorted by CV_RMSLE):")
        print(results_df.to_string(index=False))
        print(f"{'='*70}\n")
    
    return results, results_df

def train_dynamic_stacking(
    results,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    included_models='all',
    meta_learner=None,
    cv=5,
    random_state=42,
    verbose=1
):
    """
    Trains and optimises a Stacking Regressor using pre-tuned base models.
    
    This function dynamically reconstructs model instances using the 'best_params' 
    found in the previous optimisation step. It avoids data leakage by retraining 
    base models from scratch within the stacking cross-validation scheme.
    
    It also automatically wraps boosting models (CatBoost, XGBoost, LightGBM)
    to prevent compatibility issues with newer Scikit-learn versions.

    Parameters
    ----------
    results : dict
        Dictionary containing results from 'train_and_optimize_models', specifically 
        requiring the 'best_params' key for each model.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target variable.
    X_val : pd.DataFrame or np.ndarray, optional
        Validation features for final evaluation.
    y_val : pd.Series or np.ndarray, optional
        Validation target variable.
    included_models : list or str, default='all'
        Specifies which models to include in the stack:
        - 'all': Includes all models present in the 'results' dictionary.
        - list: A specific list of model names, e.g., ['catboost', 'elasticnet'].
    meta_learner : sklearn estimator, optional
        The final estimator used to combine base model predictions.
        If None, defaults to RidgeCV (L2 regularised linear regression).
    cv : int, default=5
        Number of cross-validation folds for the stacking procedure.
    random_state : int, default=42
        Random state for reproducibility.
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress/summary).

    Returns
    -------
    stacking_model : sklearn.ensemble.StackingRegressor
        The fitted Stacking Regressor ready for prediction.
    metrics : dict
        Performance metrics (RMSLE, MAE, R2) on training and validation sets.

    Examples
    --------
    >>> # Stack specific models from previous results
    >>> stack_model, metrics = train_dynamic_stacking(
    ...     results=results,
    ...     X_train=X_train, y_train=y_train,
    ...     included_models=['catboost', 'xgboost', 'elasticnet']
    ... )
    """
    
    # Model Factory: Maps string identifiers to class objects
    MODEL_FACTORY = {
        'xgboost': XGBRegressor,
        'lightgbm': LGBMRegressor,
        'catboost': CatBoostRegressor,
        'random_forest': RandomForestRegressor,
        'elasticnet': ElasticNet,
        'knn': KNeighborsRegressor,
        'svr': SVR
    }
    
    # Identify models to include in the stack
    available_models = list(results.keys())
    if included_models == 'all':
        target_models = available_models
    else:
        # Filter models: must exist in both 'results' and user request
        target_models = [m for m in included_models if m in available_models]
    
    if verbose >= 1:
        print(f"{'='*70}")
        print(f"Building Stacking Regressor with {len(target_models)} models")
        print(f"Included: {', '.join([m.upper() for m in target_models])}")
        print(f"{'='*70}")

    # Instantiate base estimators with optimised parameters
    estimators = []
    for model_name in target_models:
        if model_name not in MODEL_FACTORY:
            if verbose >= 1:
                print(f"Warning: No class definition found for '{model_name}'. Skipped.")
            continue
            
        best_params = results[model_name]['best_params']
        model_class = MODEL_FACTORY[model_name]
        
        # Handle random_state safety for applicable models
        if 'random_state' in best_params:
            model_instance = model_class(**best_params)
        elif hasattr(model_class(), 'random_state'):
            model_instance = model_class(**best_params, random_state=random_state)
        else:
            model_instance = model_class(**best_params)

        # Apply wrapper to specific boosting models to fix sklearn 1.6+ compatibility
        if model_name in ['catboost', 'xgboost', 'lightgbm']:
            model_instance = SklearnWrapper(model_instance)

        estimators.append((model_name, model_instance))
        
    # Configure Meta-Learner (Default: RidgeCV)
    if meta_learner is None:
        meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0])

    # Initialise and fit Stacking Regressor
    # n_jobs=-1 ensures parallel execution for base model training
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=cv,
        n_jobs=-1,
        passthrough=False
    )
    
    if verbose >= 1:
        print("\nTraining Stacking Regressor (this may take time)...")
    
    stacking_regressor.fit(X_train, y_train)
    
    # Calculate performance metrics
    metrics = {}
    
    # Training metrics
    y_train_pred = np.clip(stacking_regressor.predict(X_train), 0, None)
    metrics['train_rmsle'] = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
    metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
    metrics['train_r2'] = r2_score(y_train, y_train_pred)
    
    # Validation metrics
    if X_val is not None and y_val is not None:
        y_val_pred = np.clip(stacking_regressor.predict(X_val), 0, None)
        metrics['val_rmsle'] = np.sqrt(mean_squared_log_error(y_val, y_val_pred))
        metrics['val_mae'] = mean_absolute_error(y_val, y_val_pred)
        metrics['val_r2'] = r2_score(y_val, y_val_pred)
        
        if verbose >= 1:
            print(f"\nStacking Results:")
            print(f"Train RMSLE: {metrics['train_rmsle']:.4f}")
            print(f"Val RMSLE: {metrics['val_rmsle']:.4f}")
            print(f"Val R2   : {metrics['val_r2']:.4f}")
            print(f"{'='*70}\n")
    
    return stacking_regressor, metrics