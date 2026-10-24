import json
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List

# =============================================================================
# CHRONOS MINER: Historical Simulation & Feature Extraction
# =============================================================================
# Purpose: Reconstruct "live" game states from historical JSON data to train
#          the Metamodel with advanced metrics (Pulse Engine) immediately.
# =============================================================================

INPUT_FILE = r'c:\Users\machi\Documents\BasketMLG v2\NBA.json'
OUTPUT_FILE = r'c:\Users\machi\Documents\BasketMLG v2\data\training_dataset_chronos.csv'

def calculate_advanced_metrics(stats: Dict[str, Any], minutes_played: float) -> Dict[str, float]:
    """
    Calculates synthetic advanced metrics (Pulse Engine).
    Same logic as the Visionary Roadmap Phase 1.
    """
    metrics = {}
    try:
        # Extract basic components (with fallback to 0)
        home_fga = stats.get('2point_field_goals_attempted', 0) + stats.get('3point_field_goals_attempted', 0)
        away_fga = stats.get('away_2point_field_goals_attempted', 0) + stats.get('away_3point_field_goals_attempted', 0) # Note: structure might differ in JSON
        
        # Adjust for JSON structure where home/away are separate dictionaries
        # This function assumes 'stats' is a merged dict or we handle it outside.
        # Let's handle it outside to be cleaner.
        pass 
    except Exception:
        pass
    return metrics

def process_game(game: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a single game and generate training rows for Q1, Halftime, and Q3.
    """
    rows = []
    
    match_id = game.get('match_id')
    home_team = game.get('home_team')
    away_team = game.get('away_team')
    final_home = int(game.get('home_score', 0))
    final_away = int(game.get('away_score', 0))
    final_total = final_home + final_away
    
    quarter_stats = game.get('quarter_stats', {})
    quarter_scores = game.get('quarter_scores', {})
    
    if not quarter_stats or not quarter_scores:
        return []

    # Helper to safely get int
    def get_int(d, k):
        try:
            return int(d.get(k, 0))
        except:
            return 0

    # Initialize accumulators
    acc_stats = {
        'home': {'fga': 0, 'fta': 0, 'orb': 0, 'to': 0, 'score': 0},
        'away': {'fga': 0, 'fta': 0, 'orb': 0, 'to': 0, 'score': 0}
    }
    
    stages = [('Q1', 12), ('Q2', 24), ('Q3', 36)]
    
    for q_name, minutes_elapsed in stages:
        if q_name not in quarter_stats:
            break
            
        q_data = quarter_stats[q_name]
        
        # Check structure: is it by team name?
        # The JSON structure shows: "Q1": { "Team A": {...}, "Team B": {...} }
        # We need to map home/away teams to the keys in q_data
        
        home_q_stats = q_data.get(home_team, {})
        away_q_stats = q_data.get(away_team, {})
        
        if not home_q_stats or not away_q_stats:
            # Try fuzzy match or skip
            # For now, strict match
            continue

        # Update accumulators
        # Home
        h_fga = get_int(home_q_stats, '2point_field_goals_attempted') + get_int(home_q_stats, '3point_field_goals_attempted')
        h_fta = get_int(home_q_stats, 'free_throws_attempted')
        h_orb = get_int(home_q_stats, 'offensive_rebounds')
        h_to  = get_int(home_q_stats, 'turnovers')
        h_pts = get_int(home_q_stats, 'points_scored')
        
        acc_stats['home']['fga'] += h_fga
        acc_stats['home']['fta'] += h_fta
        acc_stats['home']['orb'] += h_orb
        acc_stats['home']['to']  += h_to
        acc_stats['home']['score'] += h_pts
        
        # Away
        a_fga = get_int(away_q_stats, '2point_field_goals_attempted') + get_int(away_q_stats, '3point_field_goals_attempted')
        a_fta = get_int(away_q_stats, 'free_throws_attempted')
        a_orb = get_int(away_q_stats, 'offensive_rebounds')
        a_to  = get_int(away_q_stats, 'turnovers')
        a_pts = get_int(away_q_stats, 'points_scored')
        
        acc_stats['away']['fga'] += a_fga
        acc_stats['away']['fta'] += a_fta
        acc_stats['away']['orb'] += a_orb
        acc_stats['away']['to']  += a_to
        acc_stats['away']['score'] += a_pts
        
        # --- CALCULATE PULSE ENGINE METRICS ---
        # Possessions = 0.96 * (FGA + TO + 0.44 * FTA - ORB)
        home_poss = 0.96 * (acc_stats['home']['fga'] + acc_stats['home']['to'] + 0.44 * acc_stats['home']['fta'] - acc_stats['home']['orb'])
        away_poss = 0.96 * (acc_stats['away']['fga'] + acc_stats['away']['to'] + 0.44 * acc_stats['away']['fta'] - acc_stats['away']['orb'])
        avg_poss = (home_poss + away_poss) / 2.0
        
        # ORTG (Pts / 100 Poss)
        home_ortg = (acc_stats['home']['score'] / home_poss * 100) if home_poss > 0 else 0
        away_ortg = (acc_stats['away']['score'] / away_poss * 100) if away_poss > 0 else 0
        
        # True Pace (Poss / 48m)
        # Current pace = avg_poss / minutes_elapsed * 48
        true_pace = (avg_poss / minutes_elapsed) * 48.0 if minutes_elapsed > 0 else 0
        
        # Efficiency Gap
        eff_gap = home_ortg - away_ortg
        
        # --- CREATE TRAINING ROW ---
        row = {
            'match_id': match_id,
            'stage': q_name,  # Q1, Q2 (Halftime), Q3
            'minutes_elapsed': minutes_elapsed,
            
            # Target
            'final_total': final_total,
            
            # State
            'current_total': acc_stats['home']['score'] + acc_stats['away']['score'],
            'score_diff': acc_stats['home']['score'] - acc_stats['away']['score'],
            
            # Pulse Engine Features
            'possessions_home': home_poss,
            'possessions_away': away_poss,
            'possessions_avg': avg_poss,
            'ortg_home': home_ortg,
            'ortg_away': away_ortg,
            'true_pace': true_pace,
            'efficiency_gap': eff_gap,
            
            # Derived Features (similar to live_mode)
            'pace_projection': (acc_stats['home']['score'] + acc_stats['away']['score']) / minutes_elapsed * 48.0
        }
        
        rows.append(row)
        
    return rows

def main():
    print(f"Starting Chronos Miner...")
    print(f"Reading {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} games. Processing...")
    
    all_rows = []
    processed_count = 0
    
    for game in data:
        game_rows = process_game(game)
        if game_rows:
            all_rows.extend(game_rows)
            processed_count += 1
            
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count} games...")
            
    print(f"Finished processing. Total training rows generated: {len(all_rows)}")
    
    df = pd.DataFrame(all_rows)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}")
    print("Sample data:")
    print(df.head())

if __name__ == "__main__":
    main()
