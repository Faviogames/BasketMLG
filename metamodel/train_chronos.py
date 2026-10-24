import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import os

# =============================================================================
# CHRONOS TRAINER: Time Travel Model Training
# =============================================================================

DATASET_FILE = r'c:\Users\machi\Documents\BasketMLG v2\data\training_dataset_chronos.csv'
MODEL_OUTPUT_FILE = r'c:\Users\machi\Documents\BasketMLG v2\metamodel\models\xgboost_chronos_v1.json'
METADATA_OUTPUT_FILE = r'c:\Users\machi\Documents\BasketMLG v2\metamodel\models\metadata_chronos_v1.json'

def train_chronos_model():
    print(f"Loading dataset from {DATASET_FILE}...")
    if not os.path.exists(DATASET_FILE):
        print("Error: Dataset not found. Run chronos_miner.py first.")
        return

    df = pd.read_csv(DATASET_FILE)
    
    # Features to use (Pulse Engine)
    features = [
        'minutes_elapsed',
        'current_total',
        'score_diff',
        'possessions_avg',
        'ortg_home',
        'ortg_away',
        'true_pace',
        'efficiency_gap',
        'pace_projection'
    ]
    
    target = 'final_total'
    
    print(f"Training with {len(df)} samples...")
    print(f"Features: {features}")
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost Parameters (Optimized for regression)
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mae'
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    
    # Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.2f} points")
    print(f"R2 Score: {r2:.4f}")
    
    # Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': features, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feat_imp)
    
    # Save Model
    model.save_model(MODEL_OUTPUT_FILE)
    print(f"\nModel saved to {MODEL_OUTPUT_FILE}")
    
    # Save Metadata
    metadata = {
        'features': features,
        'mae': mae,
        'r2': r2,
        'training_samples': len(df)
    }
    
    with open(METADATA_OUTPUT_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to {METADATA_OUTPUT_FILE}")

if __name__ == "__main__":
    train_chronos_model()
