import lightgbm as lgb
import json
import pandas as pd
import os

def show_full_importance():
    model_path = 'metamodel/models/lightgbm_live_v2_final.txt'
    metadata_path = 'metamodel/models/metadata_v2_final_lgb.json'
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print("Error: Model or metadata not found.")
        return

    # Load Model
    bst = lgb.Booster(model_file=model_path)
    
    # Load Feature Names
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        feature_names = metadata.get('feature_names', [])
        
    # Get Importance
    importance = bst.feature_importance(importance_type='gain')
    
    # Create DataFrame
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Calculate % Contribution
    total_gain = df_imp['Importance'].sum()
    df_imp['Contribution (%)'] = (df_imp['Importance'] / total_gain) * 100
    
    print(f"\n{'='*60}")
    print(f"FULL FEATURE IMPORTANCE REPORT (LightGBM v2)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} | {'Feature Name':<30} | {'Gain':<10} | {'Contrib %':<10}")
    print(f"{'-'*65}")
    
    for i, row in df_imp.iterrows():
        rank = i + 1 # This index is from the original df, we need a counter
    
    # Reset index for clean ranking
    df_imp = df_imp.reset_index(drop=True)
    
    for i, row in df_imp.iterrows():
        # Highlight new features
        is_new = any(x in row['Feature'] for x in ['Pace', 'FTR', 'eFG', 'TOV', 'Foul'])
        marker = "[NEW]" if is_new else ""
        
        print(f"{i+1:<5} | {row['Feature']:<30} | {row['Importance']:<10.1f} | {row['Contribution (%)']:<9.2f}% {marker}")

    print(f"{'='*60}")

if __name__ == "__main__":
    show_full_importance()
