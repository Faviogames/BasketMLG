# ===========================================
# Archivo: core/data_processing.py (v2.1)
# Procesamiento adaptado a NUEVA ESTRUCTURA JSON
# Con filtrado inteligente de stats útiles
# ===========================================
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple

from config import KEY_MAP, DEFAULT_STATS

# 🎯 ESTADÍSTICAS ÚTILES PARA OVER/UNDER (descartando bajo valor)
USEFUL_MATCH_STATS = [
    # 🔥 ALTO VALOR:
    'field_goals_attempted', 'field_goals_made', 'field_goals_',
    '3_point_field_g_attempted', '3_point_field_goals_made',
    'free_throws_attempted', 'free_throws_made',
    'turnovers', 'offensive_rebounds',
    
    # 🚀 MEDIO VALOR (seleccionadas):
    '2_point_field_g_attempted', '2_point_field_goals_made',
    'defensive_rebounds', 'assists', 'personal_fouls'
]

# ❌ ESTADÍSTICAS DESCARTADAS (bajo valor/redundantes):
DISCARDED_STATS = [
    '2_point_field_goals_',     # Redundante - calculable
    '3_point_field_goals_',     # Redundante - calculable
    'free_throws_',             # Redundante - calculable  
    'total_rebounds',           # Redundante - suma simple
    'blocks',                   # Poco impacto en totales
    'steals'                    # Poco impacto en totales
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
        # Limpiar la clave
        clean_key = key.lower().replace(" ", "_").replace("-", "").replace("__", "_")
        
        # 🆕 MAPEO ESPECÍFICO PARA NUEVA ESTRUCTURA
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
        
        # Aplicar mapeo tradicional si existe
        standard_key = KEY_MAP.get(clean_key, clean_key)
        normalized[standard_key] = value
    
    return normalized

def filter_useful_match_stats(raw_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    🎯 NUEVA FUNCIÓN: Filtra solo estadísticas útiles para Over/Under
    """
    filtered_stats = {}
    
    for stat in USEFUL_MATCH_STATS:
        if stat in raw_stats:
            filtered_stats[stat] = raw_stats[stat]
    
    return filtered_stats

def validate_match_data(match: Dict[str, Any]) -> bool:
    """Valida que un partido tenga los datos mínimos requeridos."""
    required_fields = ['home_team', 'away_team', 'date', 'home_score', 'away_score']
    
    # Verificar campos obligatorios
    for field in required_fields:
        if field not in match or not match[field]:
            return False
    
    # 🆕 VALIDAR NUEVA ESTRUCTURA: quarter_scores debe existir
    if 'quarter_scores' not in match or not isinstance(match['quarter_scores'], dict):
        return False
        
    # Validar que tenga al menos Q1 y Q2
    required_quarters = ['Q1', 'Q2']
    for quarter in required_quarters:
        if quarter not in match['quarter_scores']:
            return False
        quarter_data = match['quarter_scores'][quarter]
        if not isinstance(quarter_data, dict):
            return False
        if 'home_score' not in quarter_data or 'away_score' not in quarter_data:
            return False
    
    return True

def clean_team_names(team_name: str) -> str:
    """Limpia y estandariza nombres de equipos."""
    if not team_name:
        return ""
    
    # Eliminar espacios extra y estandarizar
    cleaned = team_name.strip()
    
    # Mapeo de nombres comunes
    name_mappings = {
        'USA': 'United States',
        'UK': 'United Kingdom',
        # Puedes expandir esto según necesidad
    }
    
    return name_mappings.get(cleaned, cleaned)

def parse_match_date(date_str: str) -> datetime:
    """Parsea la fecha de un partido a objeto datetime."""
    try:
        return datetime.strptime(date_str, '%d.%m.%Y %H:%M')
    except ValueError:
        try:
            if len(date_str.split('.')) == 2:
                date_str += f".{datetime.now().year}"
            return datetime.strptime(date_str, '%d.%m.%Y')
        except ValueError:
            try:
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
    🎯 ADAPTADO: Extrae datos por cuarto de la NUEVA ESTRUCTURA
    """
    quarter_scores = match.get('quarter_scores', {})
    
    enriched_data = {}
    
    # 🆕 PROCESAR NUEVA ESTRUCTURA quarter_scores
    for quarter_num in range(1, 5):  # Q1, Q2, Q3, Q4
        quarter_key = f'Q{quarter_num}'
        
        if quarter_key in quarter_scores:
            quarter_data = quarter_scores[quarter_key]
            home_points = safe_int(quarter_data.get('home_score', 0))
            away_points = safe_int(quarter_data.get('away_score', 0))
            
            # Datos específicos por cuarto para cada equipo
            enriched_data[f'home_q{quarter_num}_points'] = home_points
            enriched_data[f'away_q{quarter_num}_points'] = away_points
    
    # Calcular mitades y surges si tenemos suficientes datos
    q1_h = enriched_data.get('home_q1_points', 0)
    q1_a = enriched_data.get('away_q1_points', 0) 
    q2_h = enriched_data.get('home_q2_points', 0)
    q2_a = enriched_data.get('away_q2_points', 0)
    q3_h = enriched_data.get('home_q3_points', 0)
    q3_a = enriched_data.get('away_q3_points', 0)
    q4_h = enriched_data.get('home_q4_points', 0)
    q4_a = enriched_data.get('away_q4_points', 0)
    
    # Mitades
    enriched_data['home_first_half_points'] = q1_h + q2_h
    enriched_data['away_first_half_points'] = q1_a + q2_a
    enriched_data['home_second_half_points'] = q3_h + q4_h  
    enriched_data['away_second_half_points'] = q3_a + q4_a
    
    # Surges
    enriched_data['home_second_half_surge'] = enriched_data['home_second_half_points'] - enriched_data['home_first_half_points']
    enriched_data['away_second_half_surge'] = enriched_data['away_second_half_points'] - enriched_data['away_first_half_points']
    
    return enriched_data

def process_match_stats_new_structure(match: Dict[str, Any]) -> Dict[str, Any]:
    """
    🆕 NUEVA FUNCIÓN: Procesa match_stats de la nueva estructura
    """
    if 'match_stats' not in match:
        return {}
    
    processed_stats = {}
    
    for team_name, team_stats in match['match_stats'].items():
        # 🎯 FILTRAR SOLO STATS ÚTILES
        filtered_stats = filter_useful_match_stats(team_stats)
        
        # 🔄 NORMALIZAR CLAVES
        normalized_stats = normalize_stats_keys(filtered_stats)
        
        # 🔢 CONVERTIR A TIPOS CORRECTOS
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
            if stat in team_averages and not pd.isna(team_averages[stat]):
                imputed_stats[stat] = team_averages[stat]
            elif stat in league_averages and not pd.isna(league_averages[stat]):
                imputed_stats[stat] = league_averages[stat]
            else:
                imputed_stats[stat] = DEFAULT_STATS.get(stat, 0)
    
    return imputed_stats

def calculate_team_league_averages(all_matches: List[Dict], team_name: str) -> Dict[str, float]:
    """Calcula promedios históricos de un equipo - ADAPTADO A NUEVA ESTRUCTURA."""
    team_matches = []
    
    for match in all_matches:
        if match.get('home_team') == team_name or match.get('away_team') == team_name:
            # 🆕 PROCESAR match_stats DE NUEVA ESTRUCTURA
            processed_stats = process_match_stats_new_structure(match)
            
            if team_name in processed_stats:
                team_stats = processed_stats[team_name].copy()
                
                # Añadir puntos anotados/permitidos
                if match.get('home_team') == team_name:
                    team_stats['points_scored'] = safe_int(match.get('home_score', 0))
                    team_stats['points_allowed'] = safe_int(match.get('away_score', 0))
                else:
                    team_stats['points_scored'] = safe_int(match.get('away_score', 0))
                    team_stats['points_allowed'] = safe_int(match.get('home_score', 0))
                
                team_matches.append(team_stats)
    
    if not team_matches:
        return {}
    
    df_team = pd.DataFrame(team_matches)
    averages = {}
    
    for col in df_team.select_dtypes(include=[np.number]).columns:
        averages[col] = df_team[col].mean() if not df_team[col].isna().all() else 0.0
    
    return averages

def calculate_league_averages(all_matches: List[Dict]) -> Dict[str, float]:
    """Calcula promedios de toda la liga - ADAPTADO A NUEVA ESTRUCTURA."""
    all_team_stats = []
    
    for match in all_matches:
        processed_stats = process_match_stats_new_structure(match)
        
        for team_name, team_stats in processed_stats.items():
            all_team_stats.append(team_stats)
    
    if not all_team_stats:
        return {}
    
    df_league = pd.DataFrame(all_team_stats)
    averages = {}
    
    for col in df_league.select_dtypes(include=[np.number]).columns:
        averages[col] = df_league[col].mean() if not df_league[col].isna().all() else 0.0
    
    return averages

def process_raw_matches(league_data_raw: List[Dict]) -> List[Dict]:
    """
    🔄 FUNCIÓN PRINCIPAL ACTUALIZADA: Procesa nueva estructura JSON
    """
    processed_matches = []
    
    print("🔍 Procesando datos RAW con NUEVA ESTRUCTURA...")
    
    # Calcular promedios de liga para imputación
    print("📊 Calculando promedios de liga...")
    league_averages = calculate_league_averages(league_data_raw)
    
    valid_matches = 0
    invalid_matches = 0
    
    for match in league_data_raw:
        try:
            # 🆕 VALIDAR NUEVA ESTRUCTURA
            if not validate_match_data(match):
                invalid_matches += 1
                continue
            
            # 🆕 PROCESAR NUEVA ESTRUCTURA
            home_team = clean_team_names(match['home_team'])
            away_team = clean_team_names(match['away_team'])
            match_date = parse_match_date(match['date'])
            home_score, away_score = extract_final_scores(match)
            
            # Crear estructura de partido procesado
            processed_match = {
                'match_id': match.get('match_id', ''),
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'total_score': home_score + away_score,
                'raw_match': match  # Conservar datos originales
            }
            
            # 🆕 EXTRAER DATOS POR CUARTO (nueva estructura)
            quarter_alert_data = extract_quarter_data_for_alerts(match, home_team, away_team)
            processed_match.update(quarter_alert_data)
            
            processed_matches.append(processed_match)
            valid_matches += 1
            
        except Exception as e:
            print(f"⚠️ Error procesando partido {match.get('match_id', 'unknown')}: {e}")
            invalid_matches += 1
            continue
    
    print(f"✅ Procesamiento completado:")
    print(f"   📊 Partidos válidos: {valid_matches}")
    print(f"   ❌ Partidos descartados: {invalid_matches}")
    print(f"   🎯 Stats filtradas: {len(USEFUL_MATCH_STATS)} incluidas, {len(DISCARDED_STATS)} descartadas")
    
    return processed_matches

def load_league_data(data_folder: str) -> Dict[str, List[Dict]]:
    """
    📂 FUNCIÓN ACTUALIZADA: Carga datos con nueva estructura
    """
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
                    if not data:
                        continue
                    
                    # 🆕 DETECTAR LEAGUE NAME DE NUEVA ESTRUCTURA
                    # Usar el primer partido para detectar la liga
                    league_name = data[0].get('stage', 'Liga Desconocida')
                    
                    if league_name not in all_data:
                        all_data[league_name] = []
                    
                    all_data[league_name].extend(data)
                    print(f"✅ Cargado: {filename} → {league_name} ({len(data)} partidos)")
                    
            except Exception as e:
                print(f"❌ Error al cargar {filename}: {e}")
    
    return all_data