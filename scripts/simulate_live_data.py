import json
import pandas as pd
import numpy as np
import os

def simulate_live_data():
    print("Simulating live data from NBA.json...")
    try:
        with open('NBA.json', 'r', encoding='utf-8') as f:
            games = json.load(f)
    except FileNotFoundError:
        print("NBA.json not found.")
        return

    data = []
    for game in games:
        # Extract halftime state
            # Extract halftime state
            try:
                q_scores = game.get('quarter_scores', {})
                if not q_scores:
                    continue
                    
                q1_home = int(q_scores.get('Q1', {}).get('home_score', 0))
                q1_away = int(q_scores.get('Q1', {}).get('away_score', 0))
                q2_home = int(q_scores.get('Q2', {}).get('home_score', 0))
                q2_away = int(q_scores.get('Q2', {}).get('away_score', 0))
                
                halftime_total = q1_home + q1_away + q2_home + q2_away
                final_total = float(game['total_score']) if 'total_score' in game else (float(game['home_score']) + float(game['away_score']))
                
                # ✅ FIX: Calcular features reales desde quarter_stats
                quarter_stats = game.get('quarter_stats', {})
                
                # Acumuladores para halftime (Q1 + Q2)
                home_stats = {'fga': 0, 'fgm': 0, 'fta': 0, 'to': 0, 'orb': 0, '3pm': 0, '3pa': 0}
                away_stats = {'fga': 0, 'fgm': 0, 'fta': 0, 'to': 0, 'orb': 0, '3pm': 0, '3pa': 0}
                
                home_team = game.get('home_team')
                away_team = game.get('away_team')
                
                for q in ['Q1', 'Q2']:
                    if q in quarter_stats:
                        q_data = quarter_stats[q]
                        
                        # Procesar Home
                        if home_team in q_data:
                            h_data = q_data[home_team]
                            # Normalizar claves básicas si es necesario (simple mapping local)
                            def get_val(d, keys):
                                for k in keys:
                                    if k in d: return float(str(d[k]).replace('%',''))
                                return 0.0
                            
                            home_stats['fga'] += get_val(h_data, ['field_goals_attempted', 'Field Goals Attempted'])
                            home_stats['fgm'] += get_val(h_data, ['field_goals_made', 'Field Goals Made'])
                            home_stats['fta'] += get_val(h_data, ['free_throws_attempted', 'Free Throws Attempted'])
                            home_stats['to'] += get_val(h_data, ['turnovers', 'Turnovers'])
                            home_stats['orb'] += get_val(h_data, ['offensive_rebounds', 'Offensive Rebounds'])
                            home_stats['3pm'] += get_val(h_data, ['3_point_field_goals_made', '3point_field_goals_made'])
                            home_stats['3pa'] += get_val(h_data, ['3_point_field_g_attempted', '3point_field_goals_attempted'])

                            # Si no hay FGA directo, sumar 2P + 3P
                            if home_stats['fga'] == 0:
                                p2a = get_val(h_data, ['2_point_field_g_attempted', '2point_field_goals_attempted'])
                                p3a = get_val(h_data, ['3_point_field_g_attempted', '3point_field_goals_attempted'])
                                home_stats['fga'] += (p2a + p3a)
                                
                        # Procesar Away
                        if away_team in q_data:
                            a_data = q_data[away_team]
                            def get_val(d, keys):
                                for k in keys:
                                    if k in d: return float(str(d[k]).replace('%',''))
                                return 0.0
                                
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

                # Calcular Features
                # Pace = (Poss_Home + Poss_Away) / 2
                # Poss = FGA - ORB + TO + 0.44 * FTA
                poss_home = home_stats['fga'] - home_stats['orb'] + home_stats['to'] + (0.44 * home_stats['fta'])
                poss_away = away_stats['fga'] - away_stats['orb'] + away_stats['to'] + (0.44 * away_stats['fta'])
                
                pace_halftime = (poss_home + poss_away) / 2.0
                if pace_halftime == 0: pace_halftime = 95.0 # Fallback
                
                # eFG% = (FGM + 0.5 * 3PM) / FGA
                efg_home = (home_stats['fgm'] + 0.5 * home_stats['3pm']) / max(1, home_stats['fga'])
                efg_away = (away_stats['fgm'] + 0.5 * away_stats['3pm']) / max(1, away_stats['fga'])
                
                # TO Rate = TO / Poss
                to_rate_home = home_stats['to'] / max(1, poss_home)
                to_rate_away = away_stats['to'] / max(1, poss_away)
                
                # FT Rate = FTA / FGA
                ft_rate_home = home_stats['fta'] / max(1, home_stats['fga'])
                ft_rate_away = away_stats['fta'] / max(1, away_stats['fga'])

                row = {
                    'halftime_total': halftime_total,
                    'pace_halftime': pace_halftime,
                    'efg_home': efg_home,
                    'efg_away': efg_away,
                    'to_rate_home': to_rate_home,
                    'to_rate_away': to_rate_away,
                    'ft_rate_home': ft_rate_home,
                    'ft_rate_away': ft_rate_away,
                    'home_lead': (q1_home + q2_home) - (q1_away + q2_away),
                    'is_close_game': 1 if abs((q1_home + q2_home) - (q1_away + q2_away)) <= 10 else 0,
                    'momentum_score': (q2_home + q2_away) - (q1_home + q1_away),
                    'final_total': final_total
                }
                data.append(row)
            except (IndexError, KeyError, ValueError) as e:
                # print(f"Error processing game: {e}")
                continue
            
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/training_dataset_live.csv', index=False)
    print(f"Saved {len(df)} records to data/training_dataset_live.csv")

if __name__ == "__main__":
    simulate_live_data()
