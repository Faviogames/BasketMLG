import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.context_features import calculate_advanced_metrics
from analysis.metamodel_integration import aplicar_metamodelo_ai

def reproduce_issue():
    print("=== REPRODUCING LIVE PREDICTION BUG ===")
    
    # 1. Mock Data from User Log
    # Match: Philadelphia 76ers 67 - Orlando Magic 100 (Total 167)
    # Time: Q3, 29.2min played
    
    home_team = "Philadelphia 76ers"
    away_team = "Orlando Magic"
    
    # Reconstruct quarter scores roughly from total 167 and "Q1=70, Q2=76, Q3=21" (Totals)
    # This implies:
    # Q1 Total = 70
    # Q2 Total = 76
    # Q3 Total (so far) = 21
    # Total = 167. Correct.
    
    q_scores = {
        'q1_home': 30, 'q1_away': 40, # Approx for 70 total
        'q2_home': 30, 'q2_away': 46, # Approx for 76 total
        'q3_home': 7,  'q3_away': 14, # Approx for 21 total
        'q4_home': 0,  'q4_away': 0
    }
    
    minutes_played = 29.2
    
    # Mock Stats from Log
    # Stat Field goals: Home 22/49, Away 38/68
    # Stat Free throws: Home 18/20, Away 16/20
    # Stat Rebounds: Home 23, Away 32 (Off: 3, 11)
    # Stat Turnovers: Home 12, Away 5
    
    current_stats_data = {
        'home_field_goals_made': 22, 'home_field_goals_attempted': 49,
        'away_field_goals_made': 38, 'away_field_goals_attempted': 68,
        'home_free_throws_made': 18, 'home_free_throws_attempted': 20,
        'away_free_throws_made': 16, 'away_free_throws_attempted': 20,
        'home_offensive_rebounds': 3, 'away_offensive_rebounds': 11,
        'home_turnovers': 12, 'away_turnovers': 5,
        'home_score': 67, 'away_score': 100
    }
    
    # Mock api_stats_to_add (The combined stats that were being passed incorrectly)
    api_stats_to_add = {
        'field_goals_made': 60,
        'field_goals_attempted': 117,
        'free_throws_made': 34,
        'free_throws_attempted': 40,
        'total_rebounds': 55,
        'turnovers': 17
        # Note: No home/away splits here!
    }
    
    time_info = {
        'total_remaining_minutes': 48.0 - minutes_played,
        'current_pace_per_minute': 167 / minutes_played, # ~5.7
        'quarter_stage': 'q3_progress',
        'total_minutes_played': minutes_played
    }
    
    print(f"\n[SCENARIO] {home_team} vs {away_team}")
    print(f"Time: {minutes_played} min | Score: {167} pts")
    
    # 2. Reproduce the BUG: Passing api_stats_to_add instead of current_stats_data
    print("\n--- TEST 1: CURRENT BROKEN LOGIC (Passing combined stats) ---")
    try:
        # This mimics the bug in live_mode.py line 2632
        stats_source_bug = api_stats_to_add 
        
        adv_metrics_bug = calculate_advanced_metrics(stats_source_bug, time_info)
        
        print(f"TS% Home: {adv_metrics_bug.get('ts_home')}")
        print(f"TS% Away: {adv_metrics_bug.get('ts_away')}")
        print(f"Possessions: {adv_metrics_bug.get('possessions_avg')}")
        
        # Run Metamodel with these bad metrics
        live_signals_bug = {'current_stats': current_stats_data}
        live_signals_bug.update(adv_metrics_bug)
        
        pred_bug, meta_bug = aplicar_metamodelo_ai(
            base_prediction=244.9, # From log
            q_scores=q_scores,
            balance_features={},
            live_pace_metrics={},
            time_info=time_info,
            live_signals=live_signals_bug,
            team_trends={},
            bookie_line=259.5
        )
        
        print(f"Chronos Prediction (Bugged): {pred_bug:.1f}")
        print(f"Adjustment: {meta_bug.get('adjustment'):.1f}")
        
    except Exception as e:
        print(f"Error in Test 1: {e}")

    # 3. Reproduce the FIX: Passing current_stats_data
    print("\n--- TEST 2: FIXED LOGIC (Passing split stats) ---")
    try:
        stats_source_fix = current_stats_data
        
        adv_metrics_fix = calculate_advanced_metrics(stats_source_fix, time_info)
        
        print(f"TS% Home: {adv_metrics_fix.get('ts_home'):.3f}")
        print(f"TS% Away: {adv_metrics_fix.get('ts_away'):.3f}")
        print(f"Possessions: {adv_metrics_fix.get('possessions_avg'):.1f}")
        
        # Run Metamodel with good metrics
        live_signals_fix = {'current_stats': current_stats_data}
        live_signals_fix.update(adv_metrics_fix)
        
        pred_fix, meta_fix = aplicar_metamodelo_ai(
            base_prediction=244.9,
            q_scores=q_scores,
            balance_features={},
            live_pace_metrics={},
            time_info=time_info,
            live_signals=live_signals_fix,
            team_trends={},
            bookie_line=259.5
        )
        
        print(f"Chronos Prediction (Fixed): {pred_fix:.1f}")
        print(f"Adjustment: {meta_fix.get('adjustment'):.1f}")
        
    except Exception as e:
        print(f"Error in Test 2: {e}")

    # 4. Safe Projection Calculation
    print("\n--- TEST 3: SAFE PROJECTION ---")
    current_pace = 167 / minutes_played
    remaining_minutes = 48.0 - minutes_played
    safe_proj = 167 + (remaining_minutes * current_pace)
    print(f"Current Pace: {current_pace:.2f} pts/min")
    print(f"Safe Projection: {safe_proj:.1f} pts")

if __name__ == "__main__":
    reproduce_issue()
