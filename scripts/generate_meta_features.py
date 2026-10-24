import pandas as pd
import numpy as np
import sys
import os
import json
from lightgbm import Booster

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.live_features import calculate_pace_poisson

def generate_meta_features():
    print("Loading dataset...")
    df = pd.read_csv('data/training_dataset_live_v4_stacked.csv')
    
    # 1. Generate Pace Poisson Features
    print("Generating Pace Poisson features...")
    pace_projections = []
    probs_over = []
    probs_under = []
    
    for _, row in df.iterrows():
        # Simulate halftime state
        live_pace_metrics = {'live_pace_estimate': row['pace_halftime']}
        # We treat halftime total as the score at 24 minutes
        # calculate_pace_poisson infers minutes from q_scores if not provided, 
        # but we can pass time_info explicitly.
        # We'll pass a dummy q_scores that sums to halftime_total
        q_scores = {'q1': row['halftime_total']} 
        time_info = {'minutes_remaining': 24.0, 'minutes_played': 24.0}
        market_total = row['bookie_line']
        
        # We need to hack calculate_pace_poisson slightly or understand how it calculates minutes
        # It calculates minutes based on q_scores length if time_info is missing.
        # But we provide time_info.
        # However, it calculates current_total_points from q_scores.sum().
        
        try:
            if pd.isna(row['pace_halftime']) or pd.isna(row['halftime_total']):
                 raise ValueError("Missing halftime data")

            result = calculate_pace_poisson(
                live_pace_metrics=live_pace_metrics,
                q_scores=q_scores,
                market_total=market_total,
                time_info=time_info
            )
            val = result.get('pace_poisson_projection', np.nan)
            pace_projections.append(val)
            probs_over.append(result.get('prob_over', 0.5))
            probs_under.append(result.get('prob_under', 0.5))
        except Exception as e:
            # print(f"Error row {_}: {e}") # Reduce noise
            pace_projections.append(np.nan)
            probs_over.append(0.5)
            probs_under.append(0.5)
            
    df['pace_poisson_projection'] = pace_projections
    df['poisson_prob_over'] = probs_over
    df['poisson_prob_under'] = probs_under
    
    nan_count = df['pace_poisson_projection'].isna().sum()
    print(f"Pace Poisson generated. NaNs: {nan_count}/{len(df)}")
    
    # Fill NaNs with base_model_prediction as fallback
    df['pace_poisson_projection'] = df['pace_poisson_projection'].fillna(df['base_model_prediction'])
    print(f"Filled NaNs. Mean: {df['pace_poisson_projection'].mean():.2f}")

    # 2. Generate LightGBM Adjuster Predictions
    print("Generating LightGBM Adjuster predictions...")
    model_path = 'metamodel/models/lightgbm_adjuster_v1.txt'
    features_path = 'metamodel/models/features_adjuster_v1.txt'
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        bst = Booster(model_file=model_path)
        with open(features_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
            
        # Prepare X
        # Ensure base_model_prediction is present (it is in the csv)
        # We need to make sure we have all features the model expects
        # The model was trained on the same dataset, so columns should match
        
        # Check for missing columns
        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns for Adjuster: {missing_cols}")
            # Fill with 0 or mean?
            for c in missing_cols:
                df[c] = 0
        
        X = df[feature_names]
        
        # Predict Residual
        residual_preds = bst.predict(X)
        df['adjuster_residual_pred'] = residual_preds
        df['adjuster_total_pred'] = df['base_model_prediction'] + residual_preds
        
        print(f"Adjuster predictions generated. Mean Residual: {np.mean(residual_preds):.2f}")
    else:
        print("Error: LightGBM Adjuster model not found. Skipping.")
        df['adjuster_residual_pred'] = 0
        df['adjuster_total_pred'] = df['base_model_prediction']

    # Save
    output_path = 'data/training_dataset_meta_v1.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved meta-features dataset to {output_path}")

if __name__ == "__main__":
    generate_meta_features()
