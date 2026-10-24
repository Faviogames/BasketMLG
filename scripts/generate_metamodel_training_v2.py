"""
Enhanced training data generator for Metamodel v2
Generates dataset with base_model_prediction + context features
"""

import json
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.context_features import calculate_context_features

def generate_metamodel_training_data_v2():
    """
    Genera training data MEJORADO para el Metamodel con:
    - 11 features live originales (halftime_total, pace, efg, etc.)
    - base_model_prediction (predicción XGBoost)
    - ~30 context features (blowout, pace, momentum, fouls, trends, etc.)
    
    Total: ~40-45 features
    """
    
    print("="*60)
    print("Generando Training Data Mejorado para Metamodel v2")
    print("="*60)
    
    # 1. Cargar datos históricos
    print("\n[1/5] Cargando datos históricos...")
    try:
        with open('NBA.json', 'r', encoding='utf-8') as f:
            games = json.load(f)
        print(f"   OK Cargados {len(games)} juegos")
    except FileNotFoundError:
        print("   ERROR: NBA.json not found")
        return
    
    # 3. Procesar cada juego
    print("\n[3/5] Procesando juegos y generando features...")
    data = []
    skipped = 0
    
    for idx, game in enumerate(games):
        if idx % 100 == 0:
            print(f"   Progreso: {idx}/{len(games)} juegos procesados...")
        
        try:
            # Extraer quarter scores
            q_scores = game.get('quarter_scores', {})
            if not q_scores or 'Q1' not in q_scores or 'Q2' not in q_scores:
                skipped += 1
                continue
                
            q1_home = int(q_scores.get('Q1', {}).get('home_score', 0))
            q1_away = int(q_scores.get('Q1', {}).get('away_score', 0))
            q2_home = int(q_scores.get('Q2', {}).get('home_score', 0))
            q2_away = int(q_scores.get('Q2', {}).get('away_score', 0))
            
            halftime_total = q1_home + q1_away + q2_home + q2_away
            if halftime_total == 0:
                skipped += 1
                continue
            
            # Final total (target)
            final_total = float(game.get('total_score', 0))
            if final_total == 0:
                final_total = float(game.get('home_score', 0)) + float(game.get('away_score', 0))
            
            if final_total == 0:
                skipped += 1
                continue
            
            # ==========================================
            # CALCULAR FEATURES LIVE (11 originales)
            # ==========================================
            quarter_stats = game.get('quarter_stats', {})
            home_team = game.get('home_team', 'Unknown')
            away_team = game.get('away_team', 'Unknown')
            
            # Acumuladores para halftime (Q1 + Q2)
            home_stats = {'fga': 0, 'fgm': 0, 'fta': 0, 'to': 0, 'orb': 0, '3pm': 0, '3pa': 0}
            away_stats = {'fga': 0, 'fgm': 0, 'fta': 0, 'to': 0, 'orb': 0, '3pm': 0, '3pa': 0}
            
            for q in ['Q1', 'Q2']:
                if q in quarter_stats:
                    q_data = quarter_stats[q]
                    
                    # Función helper para extraer valores
                    def get_val(d, keys):
                        for k in keys:
                            if k in d: 
                                return float(str(d[k]).replace('%',''))
                        return 0.0
                    
                    # Procesar Home
                    if home_team in q_data:
                        h_data = q_data[home_team]
                        home_stats['fga'] += get_val(h_data, ['field_goals_attempted', 'Field Goals Attempted'])
                        home_stats['fgm'] += get_val(h_data, ['field_goals_made', 'Field Goals Made'])
                        home_stats['fta'] += get_val(h_data, ['free_throws_attempted', 'Free Throws Attempted'])
                        home_stats['to'] += get_val(h_data, ['turnovers', 'Turnovers'])
                        home_stats['orb'] += get_val(h_data, ['offensive_rebounds', 'Offensive Rebounds'])
                        home_stats['3pm'] += get_val(h_data, ['3_point_field_goals_made', '3point_field_goals_made'])
                        home_stats['3pa'] += get_val(h_data, ['3_point_field_g_attempted', '3point_field_goals_attempted'])
                        
                        if home_stats['fga'] == 0:
                            p2a = get_val(h_data, ['2_point_field_g_attempted', '2point_field_goals_attempted'])
                            p3a = get_val(h_data, ['3_point_field_g_attempted', '3point_field_goals_attempted'])
                            home_stats['fga'] += (p2a + p3a)
                    
                    # Procesar Away
                    if away_team in q_data:
                        a_data = q_data[away_team]
                        away_stats['fga'] += get_val(a_data, ['field_goals_attempted', 'Field Goals Attempted'])
                        away_stats['fgm'] += get_val(a_data, ['field_goals_made', 'Field Goals Made'])
                        away_stats['fta'] += get_val(a_data, ['free_throws_attempted', 'Free Throws Attempted'])
                        away_stats['to'] += get_val(a_data, ['turnovers', 'Turnovers'])
                        away_stats['orb'] += get_val(a_data, ['offensive_rebounds', 'Offensive Rebounds'])
                        away_stats['3pm'] += get_val(a_data, ['3_point_field_goals_made', '3point_field_goals_made'])
                        away_stats['3pa'] += get_val(a_data, ['3_point_field_g_attempted', '3point_field_goals_attempted'])
                        
                        if away_stats['fga'] == 0:
                            p2a = get_val(a_data, ['2_point_field_g_attempted', '2point_field_goals_attempted'])
                            p3a = get_val(a_data, ['3_point_field_g_attempted', '3point_field_goals_attempted'])
                            away_stats['fga'] += (p2a + p3a)
            
            # Calcular pace
            poss_home = home_stats['fga'] - home_stats['orb'] + home_stats['to'] + (0.44 * home_stats['fta'])
            poss_away = away_stats['fga'] - away_stats['orb'] + away_stats['to'] + (0.44 * away_stats['fta'])
            pace_halftime = (poss_home + poss_away) / 2.0
            if pace_halftime == 0: 
                pace_halftime = 95.0  # Fallback
            
            # eFG%
            efg_home = (home_stats['fgm'] + 0.5 * home_stats['3pm']) / max(1, home_stats['fga'])
            efg_away = (away_stats['fgm'] + 0.5 * away_stats['3pm']) / max(1, away_stats['fga'])
            
            # TO Rate
            to_rate_home = home_stats['to'] / max(1, poss_home)
            to_rate_away = away_stats['to'] / max(1, poss_away)
            
            # FT Rate
            ft_rate_home = home_stats['fta'] / max(1, home_stats['fga'])
            ft_rate_away = away_stats['fta'] / max(1, away_stats['fga'])
            
            # Lead, close game, momentum
            home_lead = (q1_home + q2_home) - (q1_away + q2_away)
            is_close_game = 1 if abs(home_lead) <= 10 else 0
            momentum_score = (q2_home + q2_away) - (q1_home + q1_away)
            
            # ==========================================
            # NUEVO: CALCULAR BASE MODEL PREDICTION
            # ==========================================
            # Aquí iría la lógica para calcular features históricas
            # y generar predicción XGBoost, pero para simplicidad
            # usamos una aproximación basada en halftime total
            # En producción, esto debería usar el modelo real con features completas
            
            # Aproximación simple: proyección lineal
            base_model_prediction = halftime_total * 2.0  # Simplificación temporal
            
            # ==========================================
            # NUEVO: CALCULAR CONTEXT FEATURES
            # ==========================================
            # Simular balance_features, live_pace_metrics, etc.
            q_scores_dict = {
                'q1_home': q1_home, 'q1_away': q1_away,
                'q2_home': q2_home, 'q2_away': q2_away,
                'q3_home': 0, 'q3_away': 0  # Solo halftime
            }
            
            balance_features = {
                'current_total': halftime_total,
                'is_close_game': is_close_game,
                'garbage_time_risk': 0.0,  # No aplica en halftime
                'intensity_drop_factor': 1.0,
                'is_potential_blowout': 1 if abs(home_lead) > 15 else 0
            }
            
            live_pace_metrics = {
                'live_pace_estimate': pace_halftime
            }
            
            time_info = {
                'total_remaining_minutes': 24.0,  # 2do tiempo
                'current_pace_per_minute': halftime_total / 24.0,  # Aprox
                'quarter_stage': 'halftime'
            }
            
            live_signals = {
                'home_fti': 0.5,  # Foul trouble index (mock)
                'away_fti': 0.5
            }
            
            team_trends = {
                'home': {'trend_factor': 1.0},
                'away': {'trend_factor': 1.0}
            }
            
            bookie_line = game.get('closing_line', None)
            
            # CALCULAR CONTEXT FEATURES
            context_features_dict = calculate_context_features(
                q_scores=q_scores_dict,
                balance_features=balance_features,
                live_pace_metrics=live_pace_metrics,
                time_info=time_info,
                live_signals=live_signals,
                team_trends=team_trends,
                bookie_line=bookie_line
            )
            
            # ==========================================
            # CONSTRUIR ROW COMPLETO
            # ==========================================
            row = {
                # 11 features originales
                'halftime_total': halftime_total,
                'pace_halftime': pace_halftime,
                'efg_home': efg_home,
                'efg_away': efg_away,
                'to_rate_home': to_rate_home,
                'to_rate_away': to_rate_away,
                'ft_rate_home': ft_rate_home,
                'ft_rate_away': ft_rate_away,
                'home_lead': home_lead,
                'is_close_game': is_close_game,
                'momentum_score': momentum_score,
                
                # NUEVO: Base model prediction
                'base_model_prediction': base_model_prediction,
                
                # NUEVO: Context features (~30)
                **context_features_dict,
                
                # Target
                'final_total': final_total
            }
            
            data.append(row)
            
        except (IndexError, KeyError, ValueError, ZeroDivisionError) as e:
            skipped += 1
            continue
    
    # 4. Crear DataFrame
    print(f"\n[4/5] Creando dataset...")
    print(f"   Juegos procesados: {len(data)}")
    print(f"   Juegos skipped: {skipped}")
    
    df = pd.DataFrame(data)
    
    # 5. Guardar
    print(f"\n[5/5] Guardando dataset...")
    os.makedirs('data', exist_ok=True)
    output_path = 'data/training_dataset_live_v2.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"[OK] Dataset guardado: {output_path}")
    print(f"[OK] Total registros: {len(df)}")
    print(f"[OK] Total features: {len(df.columns) - 1}")  # -1 por target
    print(f"\nFeatures incluidas:")
    for col in df.columns:
        if col != 'final_total':
            print(f"  - {col}")
    print(f"{'='*60}")
    
    return df

if __name__ == "__main__":
    generate_metamodel_training_data_v2()
