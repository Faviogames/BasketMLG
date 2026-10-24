import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.metamodel_integration import aplicar_metamodelo_ai

def test_metamodel_integration():
    print("Testing Metamodel Integration...")

    # Dummy Data
    base_prediction = 220.5
    
    q_scores = {
        'q1_home': 28, 'q1_away': 26,
        'q2_home': 30, 'q2_away': 28,
        'q3_home': 0, 'q3_away': 0,
        'q4_home': 0, 'q4_away': 0
    }
    
    balance_features = {
        'is_potential_blowout': 0,
        'blowout_severity': 0.0,
        'garbage_time_risk': 0.1,
        'intensity_drop_factor': 0.95
    }
    
    live_pace_metrics = {
        'pace_halftime': 224.0,
        'enhanced_pace_estimate': 222.0
    }
    
    time_info = {
        'total_minutes_played': 24.0,
        'quarter_stage': 'halftime'
    }
    
    # New Advanced Signals
    live_signals = {
        'current_pace_vs_avg': 2.5,
        'pace_acceleration': 0.5,
        'avg_pace': 100.0,
        'home_efg_live': 0.55,
        'away_efg_live': 0.52,
        'home_to_rate': 0.12,
        'away_to_rate': 0.14,
        'home_ft_rate': 0.22,
        'away_ft_rate': 0.18,
        'momentum_score': 0.6,
        'q2_surge': 1.0,
        'qmd_momentum': 0.3,
        'home_fti': 0.4,
        'away_fti': 0.6,
        'home_in_bonus': 0,
        'away_in_bonus': 1,
        'quarter_variance': 120.0,
        'svi_home': 1.2,
        'svi_away': 1.1,
        'svi_total': 2.3,
        'rwt': 0.8,
        # New Quarter Metrics
        'Q1_Pace': 102.0,
        'Q1_FTR': 0.25,
        'Q1_eFG': 0.54,
        'Q1_TOV_Rate': 0.13,
        'Q1_Foul_Density': 0.20,
        'Q2_Pace': 104.0,
        'Q2_FTR': 0.28,
        'Q2_eFG': 0.56,
        'Q2_TOV_Rate': 0.11,
        'Q2_Foul_Density': 0.22
    }
    
    team_trends = {'home_trend': 0.6, 'away_trend': 0.4}
    bookie_line = 221.5

    try:
        prediction, metadata = aplicar_metamodelo_ai(
            base_prediction=base_prediction,
            q_scores=q_scores,
            balance_features=balance_features,
            live_pace_metrics=live_pace_metrics,
            time_info=time_info,
            live_signals=live_signals,
            team_trends=team_trends,
            bookie_line=bookie_line
        )
        
        print(f"\n[SUCCESS] Prediction Success!")
        print(f"Base Prediction: {base_prediction}")
        print(f"Final Prediction: {prediction}")
        print(f"Adjustment: {metadata['adjustment']}")
        print(f"Model Version: {metadata.get('model_version')}")
        print(f"Features Used: {metadata.get('features_used')}")
        print(f"Missing Features: {metadata.get('missing_features_count')}")
        
        if metadata.get('missing_features_count') > 0:
            print(f"Missing List: {metadata.get('missing_features_list')}")
            
    except Exception as e:
        print(f"\n[ERROR] Prediction Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_metamodel_integration()
