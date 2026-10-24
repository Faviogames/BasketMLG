import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_model():
    model_path = 'models/NBA.joblib'
    data_path = 'data/training_dataset_live_v4_stacked.csv'
    
    print(f"Loading model metadata from {model_path}...")
    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict) and 'mae' in model_data:
            print(f"Stored MAE in metadata (Training/Test): {model_data['mae']}")
    except Exception as e:
        print(f"Error loading model: {e}")

    print(f"\nLoading data from {data_path}...")
    if not os.path.exists(data_path):
        print("Dataset not found.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    if 'final_total' not in df.columns:
        print("Target 'final_total' not found.")
        return
        
    if 'base_model_prediction' not in df.columns:
        print("Feature 'base_model_prediction' not found.")
        return
    
    y = df['final_total']
    preds = df['base_model_prediction']
    
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    print("\n=== VERIFICATION RESULTS (Base Model on Live Dataset) ===")
    print(f"Dataset: {data_path}")
    print(f"Rows evaluated: {len(y)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    print(f"\nUser Claim: MAE ~6 (likely Pre-game/Training)")
    print(f"Actual Live/Halftime Performance: MAE {mae:.4f}")
    
    if abs(mae - 12.66) < 0.5:
        print("\nCONCLUSION: The 'MAE 12.66' reported earlier corresponds to the Base Model's performance on the LIVE/HALFTIME dataset.")
        print("The 'MAE 6' is likely the model's performance on pre-game data.")

if __name__ == "__main__":
    verify_model()
