# ===========================================
# Archivo: core/data_processing.py (v3.2 - FUSIÓN FINAL)
# ✅ BASE: Mantenida la lógica de imputación original (MAE 5.2).
# ✅ AÑADIDO: Extracción quirúrgica de stats de 2/3 puntos.
# ✅ FUSIÓN: Lógica integrada para obtener lo mejor de ambas versiones.
# ===========================================
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

from config import KEY_MAP, DEFAULT_STATS

# 🎯 ESTADÍSTICAS ÚTILES PARA OVER/UNDER (OPTIMIZADAS V2.6)
USEFUL_MATCH_STATS = [
    'field_goals_attempted', 'field_goals_made',
    '3_point_field_g_attempted', '3_point_field_goals_made', 
    'free_throws_attempted', 'free_throws_made',
    'turnovers', 'offensive_rebounds', 'defensive_rebounds',
    '2_point_field_g_attempted', '2_point_field_goals_made',
    'assists'
]

# ❌ ESTADÍSTICAS DESCARTADAS (NUEVA LISTA V2.6)
DISCARDED_STATS = [
    '2_point_field_goals_', '3_point_field_goals_', 'free_throws_',
    'field_goals_',
    'total_rebounds',
    'blocks', 'steals', 'personal_fouls'
]

def safe_int(value) -> int:
    """Convierte un valor a entero de forma segura."""
    try:
        if isinstance(value, str):
            return int(value.strip())
        return int(value)
    except (ValueError, TypeError):
        return 0

def safe_float(value) -> float:
    """Convierte un valor a float de forma segura."""
    try:
        if isinstance(value, str):
            if '%' in value:
                return float(value.replace('%', '').strip()) / 100
            return float(value.strip())
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def normalize_stats_keys(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Traduce claves de estadísticas a formato estándar MEJORADO."""
    normalized = {}
    for key, value in stats_dict.items():
        clean_key = key.lower().replace(" ", "_").replace("-", "").replace("__", "_")
        
        # Mapeo manual para las claves inconsistentes del JSON
        if clean_key == "2_point_field_g_attempted":
            clean_key = "2point_field_goals_attempted"
        elif clean_key == "2_point_field_goals_made":
            clean_key = "2point_field_goals_made" 
        elif clean_key == "2_point_field_goals_":
            clean_key = "2point_field_goals_percentage"
        elif clean_key == "3_point_field_g_attempted":
            clean_key = "3point_field_goals_attempted"
        elif clean_key == "3_point_field_goals_made":
            clean_key = "3point_field_goals_made"
        elif clean_key == "3_point_field_goals_":
            clean_key = "3point_field_goals_percentage"
        elif clean_key == "field_goals_":
            clean_key = "field_goals_percentage"
        elif clean_key == "free_throws_":
            clean_key = "free_throws_percentage"
        
        standard_key = KEY_MAP.get(clean_key, clean_key)
        normalized[standard_key] = value
    
    return normalized

def filter_useful_match_stats(raw_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Filtra solo estadísticas útiles para Over/Under."""
    filtered_stats = {}
    for stat in USEFUL_MATCH_STATS:
        if stat in raw_stats:
            filtered_stats[stat] = raw_stats[stat]
    return filtered_stats

def validate_match_data(match: Dict[str, Any]) -> bool:
    """Valida que un partido tenga los datos mínimos requeridos."""
    required_fields = ['home_team', 'away_team', 'date', 'home_score', 'away_score']
    if any(field not in match or not match[field] for field in required_fields):
        return False
    
    if 'quarter_scores' not in match or not isinstance(match['quarter_scores'], dict):
        return False
        
    required_quarters = ['Q1', 'Q2']
    for quarter in required_quarters:
        if quarter not in match['quarter_scores']:
            return False
        quarter_data = match['quarter_scores'][quarter]
        if not isinstance(quarter_data, dict) or 'home_score' not in quarter_data or 'away_score' not in quarter_data:
            return False
    
    return True

def clean_team_names(team_name: str) -> str:
    """Limpia y estandariza nombres de equipos."""
    if not team_name: return ""
    cleaned = team_name.strip()
    name_mappings = {'USA': 'United States', 'UK': 'United Kingdom'}
    return name_mappings.get(cleaned, cleaned)

def parse_match_date(date_str: str) -> datetime:
    """Parsea la fecha de un partido a objeto datetime."""
    for fmt in ('%d.%m.%Y %H:%M', '%d.%m.%Y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    try:
        if len(date_str.split('.')) == 2:
            date_str += f".{datetime.now().year}"
        return datetime.strptime(date_str, '%d.%m.%Y')
    except ValueError:
        return datetime.now()

def extract_final_scores(match: Dict[str, Any]) -> Tuple[int, int]:
    """Extrae las puntuaciones finales de un partido."""
    home_score = safe_int(match.get('home_score', 0))
    away_score = safe_int(match.get('away_score', 0))
    return home_score, away_score

def extract_quarter_data_for_alerts(match: Dict[str, Any], home_team: str, away_team: str) -> Dict[str, Any]:
    """
    Función original mantenida, pero la extracción de stats se complementará
    en el bucle principal para asegurar que todos los datos necesarios estén presentes.
    """
    quarter_scores = match.get('quarter_scores', {})
    enriched_data = {}
    
    # Crear datos por cuarto (código existente mantenido)
    for quarter_num in range(1, 5):
        quarter_key = f'Q{quarter_num}'
        if quarter_key in quarter_scores:
            quarter_data = quarter_scores[quarter_key]
            home_points = safe_int(quarter_data.get('home_score', 0))
            away_points = safe_int(quarter_data.get('away_score', 0))
            
            enriched_data[f'home_q{quarter_num}_points'] = home_points
            enriched_data[f'away_q{quarter_num}_points'] = away_points
        else:
            enriched_data[f'home_q{quarter_num}_points'] = 0
            enriched_data[f'away_q{quarter_num}_points'] = 0

    # Procesar algunas stats básicas si están disponibles
    if 'match_stats' in match:
        match_stats = match['match_stats']
        if home_team in match_stats:
            home_stats = match_stats[home_team]
            enriched_data['home_turnovers'] = safe_int(home_stats.get('turnovers', 0))
            enriched_data['home_assists'] = safe_int(home_stats.get('assists', 0))
        if away_team in match_stats:
            away_stats = match_stats[away_team]
            enriched_data['away_turnovers'] = safe_int(away_stats.get('turnovers', 0))
            enriched_data['away_assists'] = safe_int(away_stats.get('assists', 0))

    # Cálculo de mitades
    q1_h, q1_a = enriched_data.get('home_q1_points', 0), enriched_data.get('away_q1_points', 0)
    q2_h, q2_a = enriched_data.get('home_q2_points', 0), enriched_data.get('away_q2_points', 0)
    q3_h, q3_a = enriched_data.get('home_q3_points', 0), enriched_data.get('away_q3_points', 0)
    q4_h, q4_a = enriched_data.get('home_q4_points', 0), enriched_data.get('away_q4_points', 0)
    
    enriched_data['home_first_half_points'] = q1_h + q2_h
    enriched_data['away_first_half_points'] = q1_a + q2_a
    enriched_data['home_second_half_points'] = q3_h + q4_h
    enriched_data['away_second_half_points'] = q3_a + q4_a
    enriched_data['home_second_half_surge'] = enriched_data['home_second_half_points'] - enriched_data['home_first_half_points']
    enriched_data['away_second_half_surge'] = enriched_data['away_second_half_points'] - enriched_data['away_first_half_points']
    
    return enriched_data

def process_match_stats_new_structure(match: Dict[str, Any]) -> Dict[str, Any]:
    """Procesa match_stats de la nueva estructura."""
    if 'match_stats' not in match: return {}
    processed_stats = {}
    for team_name, team_stats in match['match_stats'].items():
        filtered_stats = filter_useful_match_stats(team_stats)
        normalized_stats = normalize_stats_keys(filtered_stats)
        processed_team_stats = {}
        for stat_key, stat_value in normalized_stats.items():
            if 'percentage' in stat_key or stat_key.endswith('_'):
                processed_team_stats[stat_key] = safe_float(stat_value)
            else:
                processed_team_stats[stat_key] = safe_int(stat_value)
        processed_stats[team_name] = processed_team_stats
    return processed_stats

def impute_missing_stats(team_stats: Dict[str, Any], team_name: str, 
                        all_matches: List[Dict], league_averages: Dict[str, float]) -> Dict[str, Any]:
    """Sistema de imputación en cascada: equipo → liga → valores por defecto."""
    team_averages = calculate_team_league_averages(all_matches, team_name)
    essential_stats = list(DEFAULT_STATS.keys())
    imputed_stats = team_stats.copy()
    
    for stat in essential_stats:
        if stat not in imputed_stats or imputed_stats[stat] == 0 or pd.isna(imputed_stats[stat]):
            imputed_stats[stat] = team_averages.get(stat, league_averages.get(stat, DEFAULT_STATS.get(stat, 0)))
    
    return imputed_stats

def calculate_team_league_averages(all_matches: List[Dict], team_name: str) -> Dict[str, float]:
    """Calcula promedios históricos de un equipo."""
    team_matches_stats = []
    for match in all_matches:
        if match.get('home_team') == team_name or match.get('away_team') == team_name:
            processed_stats = process_match_stats_new_structure(match)
            if team_name in processed_stats:
                team_stats = processed_stats[team_name].copy()
                is_home = match.get('home_team') == team_name
                team_stats['points_scored'] = safe_int(match.get('home_score' if is_home else 'away_score', 0))
                team_stats['points_allowed'] = safe_int(match.get('away_score' if is_home else 'home_score', 0))
                team_matches_stats.append(team_stats)
    
    if not team_matches_stats: return {}
    df_team = pd.DataFrame(team_matches_stats)
    return {col: df_team[col].mean() for col in df_team.select_dtypes(include=np.number).columns if not df_team[col].isna().all()}

def calculate_league_averages(all_matches: List[Dict]) -> Dict[str, float]:
    """Calcula promedios de toda la liga."""
    all_team_stats = [team_stats for match in all_matches for team_name, team_stats in process_match_stats_new_structure(match).items()]
    if not all_team_stats: return {}
    df_league = pd.DataFrame(all_team_stats)
    return {col: df_league[col].mean() for col in df_league.select_dtypes(include=np.number).columns if not df_league[col].isna().all()}

def process_raw_matches(league_data_raw: List[Dict]) -> Tuple[List[Dict], Dict[str, float]]:
    """
    🔄 FUNCIÓN PRINCIPAL V3.1 - FUSIÓN ÓPTIMA
    ✅ Mantiene la lógica de imputación de la versión original (MAE 5.2).
    ✅ Añade la extracción completa de datos para las nuevas features.
    """
    processed_matches = []
    print("🔍 Procesando datos RAW con pipeline V3.1 (Fusión Óptima)...")
    
    if len(league_data_raw) < 500:
        league_averages = calculate_league_averages(league_data_raw)
    else:
        league_averages = {}
    
    valid_matches, invalid_matches = 0, 0
    
    for match in league_data_raw:
        try:
            if not validate_match_data(match):
                invalid_matches += 1
                continue
            
            home_team, away_team = clean_team_names(match['home_team']), clean_team_names(match['away_team'])
            home_score, away_score = extract_final_scores(match)
            
            processed_match = {
                'match_id': match.get('match_id', ''),
                # ✅ SOLUCIÓN: Usar la función de parseo de fecha original, que era más robusta.
                'date': parse_match_date(match['date']),
                'home_team': home_team, 'away_team': away_team,
                'home_score': home_score, 'away_score': away_score,
                'total_score': home_score + away_score,
                'raw_match': match
            }
            
            # 1. Procesar datos de cuarto (lógica original)
            quarter_alert_data = extract_quarter_data_for_alerts(match, home_team, away_team)
            processed_match.update(quarter_alert_data)

            # 1.b Aplanar quarter_stats (boxscore por cuarto) a columnas por equipo/cuarto
            #    Claves generadas (por ejemplo para Q1 home):
            #    home_q1_field_goals_attempted, home_q1_free_throws_attempted, home_q1_turnovers, home_q1_offensive_rebounds, etc.
            try:
                if 'quarter_stats' in match and isinstance(match['quarter_stats'], dict):
                    for qn in ['Q1', 'Q2', 'Q3', 'Q4']:
                        if qn not in match['quarter_stats']:
                            continue
                        qdata = match['quarter_stats'][qn]
                        if not isinstance(qdata, dict):
                            continue

                        for team, tstats in qdata.items():
                            prefix = 'home' if team == home_team else ('away' if team == away_team else None)
                            if not prefix or not isinstance(tstats, dict):
                                continue

                            # Getter seguro
                            def _gi(k):
                                return safe_int(tstats.get(k, 0))

                            # Extraer métricas por cuarto
                            two_att  = _gi('2point_field_goals_attempted')
                            two_made = _gi('2point_field_goals_made')
                            three_att  = _gi('3point_field_goals_attempted')
                            three_made = _gi('3point_field_goals_made')
                            fta = _gi('free_throws_attempted')
                            ftm = _gi('free_throws_made')
                            oreb = _gi('offensive_rebounds')
                            dreb = _gi('defensive_rebounds')
                            tov  = _gi('turnovers')
                            ast  = _gi('assists')
                            stl  = _gi('steals')
                            blk  = _gi('blocks')
                            pf   = _gi('personal_fouls')
                            pts  = _gi('points_scored')

                            fga = two_att + three_att
                            fgm = two_made + three_made

                            qnum = qn[1]  # '1','2','3','4'
                            base = f'{prefix}_q{qnum}'

                            processed_match[f'{base}_2point_field_goals_attempted'] = two_att
                            processed_match[f'{base}_2point_field_goals_made']      = two_made
                            processed_match[f'{base}_3point_field_goals_attempted'] = three_att
                            processed_match[f'{base}_3point_field_goals_made']      = three_made
                            processed_match[f'{base}_field_goals_attempted']       = fga
                            processed_match[f'{base}_field_goals_made']            = fgm
                            processed_match[f'{base}_free_throws_attempted']       = fta
                            processed_match[f'{base}_free_throws_made']            = ftm
                            processed_match[f'{base}_offensive_rebounds']          = oreb
                            processed_match[f'{base}_defensive_rebounds']          = dreb
                            processed_match[f'{base}_turnovers']                   = tov
                            processed_match[f'{base}_assists']                     = ast
                            processed_match[f'{base}_steals']                      = stl
                            processed_match[f'{base}_blocks']                      = blk
                            processed_match[f'{base}_personal_fouls']              = pf
                            processed_match[f'{base}_points']                      = pts
            except Exception as _qerr:
                # Evitar romper el pipeline si alguna fila viene incompleta
                pass

            # 1.c Totales por partido derivados de quarter_stats si no hay match_stats
            #     Creamos un bloque match['match_stats'] con las claves que espera el pipeline legacy
            try:
                if (('match_stats' not in match) or (not match.get('match_stats'))) and 'quarter_stats' in match:
                    totals = {}
                    for team in [home_team, away_team]:
                        prefix = 'home' if team == home_team else 'away'

                        def s(key_base):
                            return safe_int(processed_match.get(f'{prefix}_q1_{key_base}', 0)) + \
                                   safe_int(processed_match.get(f'{prefix}_q2_{key_base}', 0)) + \
                                   safe_int(processed_match.get(f'{prefix}_q3_{key_base}', 0)) + \
                                   safe_int(processed_match.get(f'{prefix}_q4_{key_base}', 0))

                        two_att   = s('2point_field_goals_attempted')
                        two_made  = s('2point_field_goals_made')
                        three_att = s('3point_field_goals_attempted')
                        three_made= s('3point_field_goals_made')

                        totals[team] = {
                            # Claves "raw" esperadas por filter_useful_match_stats/normalize_stats_keys
                            'field_goals_attempted': two_att + three_att,
                            'field_goals_made': two_made + three_made,

                            '2_point_field_g_attempted': two_att,
                            '2_point_field_goals_made': two_made,

                            '3_point_field_g_attempted': three_att,
                            '3_point_field_goals_made': three_made,

                            'free_throws_attempted': s('free_throws_attempted'),
                            'free_throws_made':      s('free_throws_made'),

                            'offensive_rebounds':    s('offensive_rebounds'),
                            'defensive_rebounds':    s('defensive_rebounds'),
                            'turnovers':             s('turnovers'),
                            'assists':               s('assists'),
                            'steals':                s('steals'),
                            'blocks':                s('blocks'),
                            'personal_fouls':        s('personal_fouls'),
                        }
                    match['match_stats'] = totals
            except Exception:
                # Si algo falla, se omite y continúa con el pipeline estándar
                pass
            
            # 2. ✅ SOLUCIÓN: Lógica de imputación y aplanamiento de la versión original (MAE 5.2)
            if 'match_stats' in match:
                processed_stats = process_match_stats_new_structure(match)
                for team_name, team_stats in processed_stats.items():
                    # Mantenemos la imputación inteligente
                    imputed_stats = impute_missing_stats(team_stats, team_name, league_data_raw, league_averages)
                    prefix = 'home' if team_name == home_team else 'away'
                    for stat_key, stat_value in imputed_stats.items():
                        processed_match[f'{prefix}_{stat_key}'] = stat_value

            # 3. ✅ SOLUCIÓN: Añadir las stats de 2/3 puntos que faltaban para las nuevas features
            # Este bloque se asegura de que los datos estén ahí, incluso si la imputación no los cubrió.
            if 'match_stats' in match and isinstance(match['match_stats'], dict):
                home_stats_raw = match['match_stats'].get(home_team, {})
                away_stats_raw = match['match_stats'].get(away_team, {})

                # Stats de 2 puntos
                processed_match['home_2_point_field_g_attempted'] = safe_int(home_stats_raw.get('2_point_field_g_attempted', 0))
                processed_match['home_2_point_field_goals_made'] = safe_int(home_stats_raw.get('2_point_field_goals_made', 0))
                processed_match['away_2_point_field_g_attempted'] = safe_int(away_stats_raw.get('2_point_field_g_attempted', 0))
                processed_match['away_2_point_field_goals_made'] = safe_int(away_stats_raw.get('2_point_field_goals_made', 0))
                
                # Stats de 3 puntos
                processed_match['home_3_point_field_g_attempted'] = safe_int(home_stats_raw.get('3_point_field_g_attempted', 0))
                processed_match['home_3_point_field_goals_made'] = safe_int(home_stats_raw.get('3_point_field_goals_made', 0))
                processed_match['away_3_point_field_g_attempted'] = safe_int(away_stats_raw.get('3_point_field_g_attempted', 0))
                processed_match['away_3_point_field_goals_made'] = safe_int(away_stats_raw.get('3_point_field_goals_made', 0))

                # ➕ Derivar posesiones por equipo y pace del partido (aprovecha match_stats; si vienen de quarter_stats, heredan esa precisión)
                try:
                    for side in ['home', 'away']:
                        fga = safe_int(processed_match.get(f'{side}_field_goals_attempted', 0))
                        oreb = safe_int(processed_match.get(f'{side}_offensive_rebounds', 0))
                        tov = safe_int(processed_match.get(f'{side}_turnovers', 0))
                        fta = safe_int(processed_match.get(f'{side}_free_throws_attempted', 0))
                        poss = fga - oreb + tov + 0.44 * fta
                        processed_match[f'{side}_possessions'] = max(poss, 0)
                    hp = processed_match.get('home_possessions', 0.0)
                    ap = processed_match.get('away_possessions', 0.0)
                    processed_match['pace_game'] = (hp + ap) / 2.0 if (hp and ap) else 0.0
                except Exception:
                    pass

            processed_matches.append(processed_match)
            valid_matches += 1
            
        except Exception as e:
            print(f"⚠️ Error procesando partido {match.get('match_id', 'unknown')}: {e}")
            invalid_matches += 1
            continue
    
    print(f"✅ Procesamiento completado:")
    print(f"   📊 Partidos válidos: {valid_matches}")
    print(f"   ❌ Partidos descartados: {invalid_matches}")
    
    return processed_matches, league_averages

def _canonicalize_league_name(name: str) -> str:
    """
    Normaliza el nombre de la liga para que sea estable entre entrenamiento y modo live.
    Reglas:
    - Quita sufijos de etapa después de ' - ' (e.g., 'Eurobasket - SEMI-FINALS' -> 'Eurobasket')
    - Unifica variantes conocidas (wnba -> WNBA, euro* -> EUROBASKET, nbl -> NBL)
    - Mantiene el resto tal cual si no coincide con reglas conocidas
    """
    n = (name or '').strip()
    if ' - ' in n:
        n = n.split(' - ', 1)[0].strip()
    low = n.lower()
    if 'wnba' in low:
        return 'WNBA'
    if 'nbl' in low or 'national_basketball' in low or 'aus' in low:
        return 'NBL'
    if 'euro' in low:
        # Unificación genérica para tus datos EURO* (Eurobasket/Europe/EuroLeague buckets)
        # Si más adelante necesitas distinguir Euroleague, ajusta aquí.
        return 'EUROBASKET'
    return n

def load_league_data(data_folder: str) -> Dict[str, List[Dict]]:
    """Carga datos con nueva estructura (liga canónica derivada del nombre de archivo, con fallback al stage)."""
    all_data = {}
    if not os.path.exists(data_folder):
        print(f"❌ ERROR: No se encontró la carpeta {data_folder}")
        return all_data
    
    for filename in os.listdir(data_folder):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(data_folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    continue

                # 1) Preferir nombre canónico derivado del archivo
                file_stem = Path(filename).stem.replace('_', ' ').strip()
                league_from_file = _canonicalize_league_name(file_stem) if file_stem else None

                # 2) Fallback: usar 'stage' del JSON, normalizado
                stage_value = data[0].get('stage', 'Liga Desconocida') if isinstance(data, list) and data else 'Liga Desconocida'
                league_from_stage = _canonicalize_league_name(stage_value) if isinstance(stage_value, str) else 'Liga Desconocida'

                league_name = league_from_file or league_from_stage or 'Liga Desconocida'

                if league_name not in all_data:
                    all_data[league_name] = []
                
                all_data[league_name].extend(data)
                print(f"✅ Cargado: {filename} → {league_name} ({len(data)} partidos)")
                
        except Exception as e:
            print(f"❌ Error al cargar {filename}: {e}")
    
    return all_data

