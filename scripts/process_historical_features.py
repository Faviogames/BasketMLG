import json
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.live_features import calculate_advanced_live_features
from analysis.context_features import calculate_context_features

def process_historical_data():
    print("="*70)
    print("PROCESSING HISTORICAL DATA - NBA.json")
    print("="*70)

    input_file = 'leagues/NBA.json'
    output_file = 'data/training_dataset_live_v3.csv'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)

    print(f"Found {len(matches)} matches.")

    processed_data = []
    
    for i, match in enumerate(matches):
        if i % 100 == 0:
            print(f"Processing match {i}/{len(matches)}...", end='\r')

        try:
            # Basic info
            home_team = match.get('home_team')
            away_team = match.get('away_team')
            final_home = int(match.get('home_score', 0))
            final_away = int(match.get('away_score', 0))
            final_total = final_home + final_away
            
            # Extract Bookie Line (Pre-match)
            line_odds = match.get('line_odds', {})
            bookie_line = float(line_odds.get('total_line', 0.0))

            # Quarter Scores (SOLO Q1 y Q2 para simular Halftime)
            q_scores_raw = match.get('quarter_scores', {})
            q_scores_normalized = {}
            
            # Solo extraemos Q1 y Q2 para el cálculo de features
            for q in ['Q1', 'Q2']:
                if q in q_scores_raw:
                    q_scores_normalized[f'{q.lower()}_home'] = int(q_scores_raw[q].get('home_score', 0))
                    q_scores_normalized[f'{q.lower()}_away'] = int(q_scores_raw[q].get('away_score', 0))
            
            # Calcular totales al descanso
            q1_home = q_scores_normalized.get('q1_home', 0)
            q1_away = q_scores_normalized.get('q1_away', 0)
            q2_home = q_scores_normalized.get('q2_home', 0)
            q2_away = q_scores_normalized.get('q2_away', 0)
            
            halftime_home = q1_home + q2_home
            halftime_away = q1_away + q2_away
            halftime_total = halftime_home + halftime_away

            # Quarter Stats (SOLO Q1 y Q2)
            q_stats_raw = match.get('quarter_stats', {})
            q_stats_normalized = {}

            for q in ['Q1', 'Q2']:
                if q in q_stats_raw:
                    q_stats_normalized[q.lower()] = {}
                    # Map team names to home/away
                    if home_team in q_stats_raw[q]:
                        q_stats_normalized[q.lower()]['home'] = q_stats_raw[q][home_team]
                    if away_team in q_stats_raw[q]:
                        q_stats_normalized[q.lower()]['away'] = q_stats_raw[q][away_team]

            # Prepare game_data for feature extraction
            game_data = {
                'quarter_scores': q_scores_normalized,
                'quarter_stats': q_stats_normalized
            }
            
            # 1. Calculate Advanced Features (SVI, RWT, Pace, etc.)
            advanced_features = calculate_advanced_live_features(game_data)
            
            # DEBUG: Print features for the first match to verify new metrics
            if i == 0:
                print("\n[DEBUG] New Advanced Features Keys:")
                for k in sorted(advanced_features.keys()):
                    if any(x in k for x in ['Pace', 'FTR', 'eFG', 'TOV', 'Foul']):
                        print(f"   - {k}: {advanced_features[k]}")
            
            
            # 2. Calculate Context Features (Standard 41 features)
            # Necesitamos preparar los inputs para calculate_context_features
            
            # Balance features dummy/estimados
            balance_features = {
                'current_total': halftime_total,
                'is_close_game': 1 if abs(halftime_home - halftime_away) <= 10 else 0,
                'garbage_time_risk': 0.0,
                'intensity_drop_factor': 1.0,
                'is_potential_blowout': 1 if abs(halftime_home - halftime_away) > 15 else 0
            }
            
            # Time info (Halftime)
            time_info = {
                'total_remaining_minutes': 24.0,
                'current_pace_per_minute': halftime_total / 24.0 if halftime_total > 0 else 0,
                'quarter_stage': 'halftime'
            }
            
            # Live signals (Dummy por ahora, o extraer de stats si es posible)
            live_signals = {
                'home_fti': 0.5,
                'away_fti': 0.5
            }
            
            team_trends = {
                'home': {'trend_factor': 1.0},
                'away': {'trend_factor': 1.0}
            }
            
            # Calcular stats derivadas (eFG, TO%, FT%)
            # Esto es crucial para que coincida con el modelo anterior
            home_stats = {'efg': 0.5, 'to_rate': 0.13, 'ft_rate': 0.20}
            away_stats = {'efg': 0.5, 'to_rate': 0.13, 'ft_rate': 0.20}
            
            # Intentar calcular stats reales desde quarter_stats si existen
            try:
                h_fga = 0; h_fgm = 0; h_3pa = 0; h_3pm = 0; h_to = 0; h_fta = 0; h_ftm = 0
                a_fga = 0; a_fgm = 0; a_3pa = 0; a_3pm = 0; a_to = 0; a_fta = 0; a_ftm = 0
                
                for q in ['q1', 'q2']:
                    if q in q_stats_normalized:
                        h_data = q_stats_normalized[q].get('home', {})
                        a_data = q_stats_normalized[q].get('away', {})
                        
                        h_fga += h_data.get('2point_field_goals_attempted', 0) + h_data.get('3point_field_goals_attempted', 0)
                        h_fgm += h_data.get('2point_field_goals_made', 0) + h_data.get('3point_field_goals_made', 0)
                        h_3pm += h_data.get('3point_field_goals_made', 0)
                        h_to += h_data.get('turnovers', 0)
                        h_fta += h_data.get('free_throws_attempted', 0)
                        h_ftm += h_data.get('free_throws_made', 0)
                        
                        a_fga += a_data.get('2point_field_goals_attempted', 0) + a_data.get('3point_field_goals_attempted', 0)
                        a_fgm += a_data.get('2point_field_goals_made', 0) + a_data.get('3point_field_goals_made', 0)
                        a_3pm += a_data.get('3point_field_goals_made', 0)
                        a_to += a_data.get('turnovers', 0)
                        a_fta += a_data.get('free_throws_attempted', 0)
                        a_ftm += a_data.get('free_throws_made', 0)
                
                # eFG% = (FGM + 0.5 * 3PM) / FGA
                if h_fga > 0: home_stats['efg'] = (h_fgm + 0.5 * h_3pm) / h_fga
                if a_fga > 0: away_stats['efg'] = (a_fgm + 0.5 * a_3pm) / a_fga
                
                # TO Rate approx = TO / (FGA + 0.44*FTA + TO)
                h_poss = h_fga + 0.44*h_fta + h_to
                if h_poss > 0: home_stats['to_rate'] = h_to / h_poss
                
                a_poss = a_fga + 0.44*a_fta + a_to
                if a_poss > 0: away_stats['to_rate'] = a_to / a_poss
                
                # FT Rate = FTM / FGA
                if h_fga > 0: home_stats['ft_rate'] = h_ftm / h_fga
                if a_fga > 0: away_stats['ft_rate'] = a_ftm / a_fga
                
            except Exception:
                pass

            context_features = calculate_context_features(
                q_scores=q_scores_normalized,
                balance_features=balance_features,
                live_pace_metrics={'live_pace_estimate': advanced_features.get('avg_pace', 95.0)},
                time_info=time_info,
                live_signals=live_signals,
                team_trends=team_trends,
                bookie_line=bookie_line
            )

            # Combine everything
            row = {
                'match_id': match.get('match_id'),
                'final_total': final_total,
                # Base features
                'halftime_total': halftime_total,
                'pace_halftime': advanced_features.get('avg_pace', 95.0),
                'efg_home': home_stats['efg'],
                'efg_away': away_stats['efg'],
                'to_rate_home': home_stats['to_rate'],
                'to_rate_away': away_stats['to_rate'],
                'ft_rate_home': home_stats['ft_rate'],
                'ft_rate_away': away_stats['ft_rate'],
                'home_lead': halftime_home - halftime_away,
                'is_close_game': 1 if abs(halftime_home - halftime_away) <= 10 else 0,
                'momentum_score': (q2_home + q2_away) - (q1_home + q1_away),
                'base_model_prediction': halftime_total * 2.0, # Dummy base prediction
                
                # Context features
                **context_features,
                
                # Advanced Live Features (SVI, RWT, etc.)
                **advanced_features
            }
            
            processed_data.append(row)

        except Exception as e:
            if i < 5:
                print(f"Error processing match {match.get('match_id')}: {e}")
                import traceback
                traceback.print_exc()
            continue

    print(f"\nSuccessfully processed {len(processed_data)} matches.")

    # Create DataFrame
    df = pd.DataFrame(processed_data)
    
    # Drop non-numeric columns except match_id
    # numeric_df = df.select_dtypes(include=[np.number])
    
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    process_historical_data()
