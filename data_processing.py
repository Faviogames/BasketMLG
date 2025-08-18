# ===========================================
# Archivo: core/data_processing.py (v2.7 - ESTABLE + MEJORAS V2.6)
# ✅ INTEGRADAS: Todas las mejoras críticas de v2.6
# ✅ OPTIMIZADO: Lista de stats para Over/Under  
# ✅ MEJORADO: Procesamiento de match_stats completo
# ===========================================
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple

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
    🔧 VERSIÓN V2.7 - MEJORADA CON PROCESAMIENTO COMPLETO DE MATCH_STATS
    ✅ INTEGRADO: Procesamiento de turnovers y 6 nuevas estadísticas de v2.6
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

    # 🆕 MEJORA CRÍTICA V2.6: Procesar match_stats para turnovers y otras estadísticas
    if 'match_stats' in match:
        match_stats = match['match_stats']
        
        # Procesar estadísticas del equipo local
        if home_team in match_stats:
            home_stats = match_stats[home_team]
            enriched_data['home_turnovers'] = safe_int(home_stats.get('turnovers', 0))
            enriched_data['home_assists'] = safe_int(home_stats.get('assists', 0))
            enriched_data['home_field_goals_made'] = safe_int(home_stats.get('field_goals_made', 0))
            enriched_data['home_field_goals_attempted'] = safe_int(home_stats.get('field_goals_attempted', 0))
            enriched_data['home_steals'] = safe_int(home_stats.get('steals', 0))
            enriched_data['home_personal_fouls'] = safe_int(home_stats.get('personal_fouls', 0))
        
        # Procesar estadísticas del equipo visitante
        if away_team in match_stats:
            away_stats = match_stats[away_team]
            enriched_data['away_turnovers'] = safe_int(away_stats.get('turnovers', 0))
            enriched_data['away_assists'] = safe_int(away_stats.get('assists', 0))
            enriched_data['away_field_goals_made'] = safe_int(away_stats.get('field_goals_made', 0))
            enriched_data['away_field_goals_attempted'] = safe_int(away_stats.get('field_goals_attempted', 0))
            enriched_data['away_steals'] = safe_int(away_stats.get('steals', 0))
            enriched_data['away_personal_fouls'] = safe_int(away_stats.get('personal_fouls', 0))

    # Cálculo de mitades (mantenido del código original)
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
    🔄 FUNCIÓN PRINCIPAL V2.7 - INTEGRADAS TODAS LAS MEJORAS V2.6
    ✅ OPTIMIZACIÓN: Cálculo inteligente de promedios según tamaño de liga
    ✅ MEJORADO: Imputación efectiva con stats aplicadas al processed_match
    """
    processed_matches = []
    print("🔍 Procesando datos RAW con NUEVA ESTRUCTURA OPTIMIZADA...")
    
    # 🎯 OPTIMIZACIÓN V2.6: Solo calcular promedios de liga si la liga es pequeña
    if len(league_data_raw) < 500:
        print("📊 Liga pequeña: Calculando promedios de liga para imputación...")
        league_averages = calculate_league_averages(league_data_raw)
    else:
        print("📊 Liga grande: Saltando cálculo de promedios de liga (optimización)")
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
                'date': parse_match_date(match['date']),
                'home_team': home_team, 'away_team': away_team,
                'home_score': home_score, 'away_score': away_score,
                'total_score': home_score + away_score,
                'raw_match': match
            }
            
            # Procesar datos de cuarto para alertas
            quarter_alert_data = extract_quarter_data_for_alerts(match, home_team, away_team)
            processed_match.update(quarter_alert_data)
            
            # 🔄 MEJORA V2.6: Imputación efectiva con stats aplicadas
            if 'match_stats' in match:
                processed_stats = process_match_stats_new_structure(match)
                for team_name, team_stats in processed_stats.items():
                    imputed_stats = impute_missing_stats(team_stats, team_name, league_data_raw, league_averages)
                    # Añadir stats imputadas al processed_match con prefijos
                    prefix = 'home' if team_name == home_team else 'away'
                    for stat_key, stat_value in imputed_stats.items():
                        processed_match[f'{prefix}_{stat_key}'] = stat_value
            
            processed_matches.append(processed_match)
            valid_matches += 1
            
        except Exception as e:
            print(f"⚠️ Error procesando partido {match.get('match_id', 'unknown')}: {e}")
            invalid_matches += 1
            continue
    
    print(f"✅ Procesamiento completado:")
    print(f"   📊 Partidos válidos: {valid_matches}")
    print(f"   ❌ Partidos descartados: {invalid_matches}")
    print(f"   🎯 Stats optimizadas: {len(USEFUL_MATCH_STATS)} incluidas, {len(DISCARDED_STATS)} descartadas")
    if league_averages:
        print(f"   🔧 Imputación aplicada usando {len(league_averages)} promedios de liga")
    else:
        print("   🔧 Imputación optimizada para liga grande (sin promedios de liga)")
    
    # 🛡️ BUG FIX: Retornar los promedios de la liga junto con los partidos
    return processed_matches, league_averages

def load_league_data(data_folder: str) -> Dict[str, List[Dict]]:
    """Carga datos con nueva estructura."""
    all_data = {}
    if not os.path.exists(data_folder):
        print(f"❌ ERROR: No se encontró la carpeta {data_folder}")
        return all_data
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not data: continue
                    
                    league_name = data[0].get('stage', 'Liga Desconocida')
                    if league_name not in all_data:
                        all_data[league_name] = []
                    
                    all_data[league_name].extend(data)
                    print(f"✅ Cargado: {filename} → {league_name} ({len(data)} partidos)")
                    
            except Exception as e:
                print(f"❌ Error al cargar {filename}: {e}")
    
    return all_data