import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import os

def generate_base_predictions():
    print("="*70)
    print("GENERATING BASE PREDICTIONS (STACKING LAYER)")
    print("="*70)
    
    # 1. Load v3 dataset
    input_path = 'data/training_dataset_live_v3.csv'
    output_path = 'data/training_dataset_live_v4_stacked.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} matches.")
    
    # 2. Prepare features for Base Model (XGBoost)
    # The base model should only see "raw" features, not necessarily the advanced context ones
    # But for simplicity and strength, we'll give it the numeric features available.
    # IMPORTANT: We must NOT include the target 'final_total' in features.
    
    exclude_cols = ['match_id', 'final_total', 'base_model_prediction', 'volatility_class', 'rwt_trend', 'pace_trend', 'qmd_trend']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter numeric only
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['final_total']
    
    print(f"Training Base XGBoost on {X.shape[1]} features...")
    
    # 3. K-Fold Cross Validation to generate Out-of-Fold predictions
    # This ensures the Metamodel doesn't see predictions from a model that already saw the answer.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    base_predictions = np.zeros(len(df))
    
    fold = 1
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Train Base Model
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predict on Validation Fold
        preds = model.predict(X_val)
        base_predictions[val_index] = preds
        
        # Evaluate Fold
        mae = np.mean(np.abs(preds - y_val))
        print(f"   Fold {fold}/5 - MAE: {mae:.2f}")
        fold += 1
        
    # 4. Update Dataset
    df['base_model_prediction'] = base_predictions
    
    # Calculate error of base model for reference
    total_mae = np.mean(np.abs(df['final_total'] - df['base_model_prediction']))
    print(f"\nOverall Base Model MAE: {total_mae:.2f}")
    
    # 5. Save Stacked Dataset
    df.to_csv(output_path, index=False)
    print(f"Saved stacked dataset to {output_path}")
    print("Ready for Metamodel training.")

if __name__ == "__main__":
    generate_base_predictions()
