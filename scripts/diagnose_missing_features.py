"""
Script to analyze missing features in live predictions
Modifies live_mode.py temporarily to log all missing features to a file
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def add_missing_features_logging():
    """
    Adds code to log missing features to a file during live predictions
    """
    
    live_mode_path = "c:/Users/machi/Documents/BasketMLG v2/analysis/live_mode.py"
    
    # Code to insert after line 1994 (after the imputation percentage print)
    logging_code = '''
        # DIAGNOSTIC: Log missing features to file
        if missing_count > 0:
            import json
            from datetime import datetime
            
            diagnostic_file = "missing_features_log.json"
            
            diagnostic_data = {
                'timestamp': datetime.now().isoformat(),
                'total_features': len(X_pred.columns),
                'missing_count': missing_count,
                'imputation_percentage': imputation_percentage,
                'missing_features': list(missing_features),
                'critical_missing': [f for f in missing_features if f in CRITICAL_FEATURES],
                'all_expected_features': list(X_pred.columns)
            }
            
            # Append to file
            if os.path.exists(diagnostic_file):
                with open(diagnostic_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(diagnostic_data)
            
            with open(diagnostic_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            print(f"[DIAGNOSTIC] Missing features logged to {diagnostic_file}")
'''
    
    print(f"Code to add after line 1994 in {live_mode_path}:")
    print(logging_code)
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("1. Open analysis/live_mode.py")
    print("2. Find line 1994 (the imputation percentage print)")
    print("3. Add the logging code above")
    print("4. Run the live system again")
    print("5. Check 'missing_features_log.json' for detailed analysis")
    print("="*60)

if __name__ == "__main__":
    add_missing_features_logging()
