import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from cattle_weight.config import RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RANDOM_STATE, N_FOLDS

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def build_and_validate(X, y, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    """
    Trains MLR and RFR models on the extracted morphometric features (X) and weight targets (y).
    Performs n-fold Cross-Validation to evaluate model performance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    lr_models = []
    rf_models = []
    
    lr_mae_list, lr_rmse_list, lr_r2_list, lr_mape_list = [], [], [], []
    rf_mae_list, rf_rmse_list, rf_r2_list, rf_mape_list = [], [], [], []
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Linear Regression (MLR)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_models.append(lr)
        lr_pred = lr.predict(X_test)
        
        lr_mae_list.append(mean_absolute_error(y_test, lr_pred))
        lr_rmse_list.append(root_mean_squared_error(y_test, lr_pred))
        lr_r2_list.append(r2_score(y_test, lr_pred))
        lr_mape_list.append(calculate_mape(y_test, lr_pred))
        
        # Random Forest Regressor (RFR)
        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=random_state
        )
        rf.fit(X_train, y_train)
        rf_models.append(rf)
        rf_pred = rf.predict(X_test)
        
        rf_mae_list.append(mean_absolute_error(y_test, rf_pred))
        rf_rmse_list.append(root_mean_squared_error(y_test, rf_pred))
        rf_r2_list.append(r2_score(y_test, rf_pred))
        rf_mape_list.append(calculate_mape(y_test, rf_pred))
        
    # Fit final models on full dataset
    final_lr = LinearRegression()
    final_lr.fit(X_scaled, y)
    
    final_rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=random_state
    )
    final_rf.fit(X_scaled, y)
    
    cv_metrics = {
        'lr': {
            'mae': (np.mean(lr_mae_list), np.std(lr_mae_list)),
            'rmse': (np.mean(lr_rmse_list), np.std(lr_rmse_list)),
            'r2': (np.mean(lr_r2_list), np.std(lr_r2_list)),
            'mape': (np.mean(lr_mape_list), np.std(lr_mape_list))
        },
        'rf': {
            'mae': (np.mean(rf_mae_list), np.std(rf_mae_list)),
            'rmse': (np.mean(rf_rmse_list), np.std(rf_rmse_list)),
            'r2': (np.mean(rf_r2_list), np.std(rf_r2_list)),
            'mape': (np.mean(rf_mape_list), np.std(rf_mape_list))
        }
    }
    
    return final_lr, final_rf, scaler, cv_metrics
