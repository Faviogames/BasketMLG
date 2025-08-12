# ===========================================
# Archivo: config.py (v1.1)
# Configuraciones y constantes centralizadas con alertas de balance
# ===========================================

import os
from pathlib import Path

# ===========================================
# FASE 1: SISTEMA DE FILTRADO DE FEATURES MUERTAS
# ===========================================

# Lista de features muertas (basada en analisis JSON)
UNIVERSALLY_USELESS_FEATURES = [
    # Four Factors (21 features)
    'home_avg_possessions_last_5', 'away_avg_possessions_last_5', 'diff_avg_possessions_last_5',
    'home_avg_ortg_last_5', 'away_avg_ortg_last_5', 'diff_avg_ortg_last_5',
    'home_avg_drtg_last_5', 'away_avg_drtg_last_5', 'diff_avg_drtg_last_5',
    'home_avg_efg_percentage_last_5', 'away_avg_efg_percentage_last_5', 'diff_avg_efg_percentage_last_5',
    'home_avg_tov_percentage_last_5', 'away_avg_tov_percentage_last_5', 'diff_avg_tov_percentage_last_5',
    'home_avg_oreb_percentage_last_5', 'away_avg_oreb_percentage_last_5', 'diff_avg_oreb_percentage_last_5',
    'home_avg_ft_rate_last_5', 'away_avg_ft_rate_last_5', 'diff_avg_ft_rate_last_5',
    
    # Advanced Stats Problematicas (9 features)
    'home_avg_pace_last_5', 'away_avg_pace_last_5', 'diff_avg_pace_last_5',
    'home_avg_offensive_efficiency_last_5', 'away_avg_offensive_efficiency_last_5', 'diff_avg_offensive_efficiency_last_5',
    'home_avg_defensive_efficiency_last_5', 'away_avg_defensive_efficiency_last_5', 'diff_avg_defensive_efficiency_last_5'
]

DEAD_FEATURES_AVAILABLE = True
ENABLE_FEATURE_FILTERING = False
print(f"Lista de features muertas integrada: {len(UNIVERSALLY_USELESS_FEATURES)} features")

# 📁 CONFIGURACIONES DE CARPETAS
DATA_FOLDER = './leagues'
MODELS_FOLDER = './models'
PROCESSED_FILES_PATH = './processed_files.json'

# 📊 METRICAS DE MOMENTUM CRITICAS (con EMA)
MOMENTUM_STATS_COLS = [
    'win', 'plus_minus', 'win_rate', 'avg_plus_minus', 
    'scoring_efficiency', 'defensive_stops', 'clutch_performance'
]

# 📈 ESTADISTICAS AVANZADAS (con promedio simple)
ADVANCED_STATS_COLS = [
    'possessions', 'ortg', 'drtg',
    'efg_percentage', 'tov_percentage', 'oreb_percentage', 'ft_rate',
    'pace', 'offensive_efficiency', 'defensive_efficiency'
]

# 🎯 METRICAS DE PERFORMANCE CONTEXTUAL
PERFORMANCE_CONTEXT_COLS = [
    'home_advantage_factor', 'comeback_ability', 'consistency_index'
]

# 🆕 METRICAS POR CUARTO ESPECIFICO (para alertas)
QUARTER_SPECIFIC_COLS = [
    'q1_points', 'q2_points', 'q3_points', 'q4_points',
    'first_half_points', 'second_half_points', 'second_half_surge'
]

# ⚡ RANGOS DE EMA PARA DIFERENTES ASPECTOS DEL MOMENTUM
EMA_RANGES = {
    'short_term': 3,   # Ultimos 3 partidos - momentum inmediato
    'medium_term': 7,  # Ultimos 7 partidos - forma reciente  
    'long_term': 15    # Ultimos 15 partidos - tendencia estacional
}

# 🔧 CARACTERISTICAS PRE-PARTIDO BASICAS
PRE_GAME_FEATURES_BASIC = [
    'home_avg_pts_scored_last_5', 'home_avg_pts_allowed_last_5', 'home_avg_total_pts_last_5',
    'away_avg_pts_scored_last_5', 'away_avg_pts_allowed_last_5', 'away_avg_total_pts_last_5',
    'diff_avg_pts_scored_last_5', 'diff_avg_pts_allowed_last_5',
]

# 🎪 CARACTERISTICAS DE MOMENTUM (multiples rangos EMA)
PRE_GAME_FEATURES_MOMENTUM = []
for stat in MOMENTUM_STATS_COLS:
    for ema_name, ema_period in EMA_RANGES.items():
        PRE_GAME_FEATURES_MOMENTUM.extend([
            f'home_ema_{stat}_{ema_name}_{ema_period}',
            f'away_ema_{stat}_{ema_name}_{ema_period}',
            f'diff_ema_{stat}_{ema_name}_{ema_period}'
        ])

# 📊 CARACTERISTICAS AVANZADAS
PRE_GAME_FEATURES_ADVANCED = []
for stat in ADVANCED_STATS_COLS:
    PRE_GAME_FEATURES_ADVANCED.extend([
        f'home_avg_{stat}_last_5',
        f'away_avg_{stat}_last_5',
        f'diff_avg_{stat}_last_5'
    ])

# 🏠 CARACTERISTICAS DE CONTEXTO DE PERFORMANCE
PRE_GAME_FEATURES_CONTEXT = []
for stat in PERFORMANCE_CONTEXT_COLS:
    PRE_GAME_FEATURES_CONTEXT.extend([
        f'home_avg_{stat}_last_10',
        f'away_avg_{stat}_last_10',
        f'diff_avg_{stat}_last_10'
    ])

# 🔴 CARACTERISTICAS EN VIVO
LIVE_GAME_FEATURES = [
    'q1_total', 'q2_total', 'q3_total', 'halftime_total', 'q3_end_total',
    'q1_diff', 'q2_diff', 'q3_diff', 'q2_trend', 'q3_trend', 'quarter_variance',
    'live_pace_estimate', 'live_efficiency_home', 'live_efficiency_away',
    'live_momentum_shift', 'quarter_consistency', 'comeback_indicator'
]

# ⚖️ CARACTERISTICAS DE BALANCE (sistema de deteccion de palizas)
BALANCE_FEATURES = [
    'game_balance_score', 'is_game_unbalanced', 'intensity_drop_factor',
    'blowout_momentum', 'expected_q4_drop', 'lead_stability'
]

# 🎯 TODAS LAS CARACTERISTICAS COMBINADAS (INCLUYENDO BALANCE)
PRE_GAME_FEATURES = (PRE_GAME_FEATURES_BASIC + PRE_GAME_FEATURES_MOMENTUM + 
                    PRE_GAME_FEATURES_ADVANCED + PRE_GAME_FEATURES_CONTEXT)

FEATURES_TO_USE = PRE_GAME_FEATURES + LIVE_GAME_FEATURES + BALANCE_FEATURES

# ===========================================
# Funcion de filtrado y variable filtrada
# ===========================================

def get_filtered_features_to_use():
    """Retorna FEATURES_TO_USE filtradas sin features muertas"""
    if ENABLE_FEATURE_FILTERING and DEAD_FEATURES_AVAILABLE:
        filtered = [f for f in FEATURES_TO_USE if f not in UNIVERSALLY_USELESS_FEATURES]
        print(f"Features filtradas: {len(FEATURES_TO_USE)} -> {len(filtered)} (eliminadas: {len(FEATURES_TO_USE) - len(filtered)})")
        return filtered
    else:
        return FEATURES_TO_USE

# Nueva variable para usar en training
FILTERED_FEATURES_TO_USE = get_filtered_features_to_use()

# 🔄 MAPEO DE NORMALIZACION DE CLAVES
KEY_MAP = {
    'field_goals_': 'field_goals_percentage',
    '2-point_field_g_attempted': '2point_field_goals_attempted',
    '2point_field_goals_': '2point_field_goals_percentage',
    '3-point_field_g_attempted': '3point_field_goals_attempted',
    '3-point_field_goals_': '3point_field_goals_percentage',
    'free_throws_': 'free_throws_percentage'
}

# 🚨 CONFIGURACION DEL SISTEMA DE ALERTAS
ALERT_TYPES = {
    'UNDER_PERFORMANCE': "⚠️ {team} anotando {diff:.1f} puntos menos que su promedio en {quarter} ({current:.1f} vs {avg:.1f})",
    'OVER_PERFORMANCE': "🔥 {team} anotando {diff:.1f} puntos mas que su promedio en {quarter} ({current:.1f} vs {avg:.1f})",
    'SURGE_EXPECTED': "🚀 {team} historicamente tiene repunte en {period} (promedio: +{surge:.1f} pts)",
    'COLD_STREAK': "🧊 {team} en racha fria: {consecutive} cuartos consecutivos por debajo del promedio",
    'HOT_STREAK': "🔥 {team} en racha caliente: {consecutive} cuartos consecutivos por encima del promedio",
    'SLOW_START_RECOVERY': "📈 {team} suele recuperarse tras inicios lentos (probabilidad {recovery_rate:.0%})",
    'DEFENSIVE_COLLAPSE': "🛡️ {team} permitiendo {diff:.1f} pts mas de lo usual - posible colapso defensivo",
    'PACE_SHIFT': "⚡ Ritmo de juego {direction} ({current_pace:.1f} vs promedio {avg_pace:.1f} posesiones/48min)",
    'SECOND_HALF_SURGE': "💪 {team} promedia {surge:.1f} pts mas en segunda mitad - considerar ajustes",
    'CLOSING_STRENGTH': "🎯 {team} muy fuerte en cuartos finales (promedio Q4: {q4_avg:.1f} pts)",
    'GAME_UNBALANCED': "⚖️ Juego muy desigual ({lead:.0f} pts) - posible impacto en Q4",
    'INTENSITY_DROP': "📉 Caida de intensidad detectada - ritmo bajando {drop:.1%}",
    'BLOWOUT_MOMENTUM': "🏃 Momentum de paliza - considerar garbage time en Q4"
}

# ⚖️ UMBRALES PARA DETECTAR ANOMALIAS
ALERT_THRESHOLDS = {
    'SIGNIFICANT_DIFF': 4.0,
    'ANOMALY_THRESHOLD': 1.5,
    'STREAK_MIN': 2,
    'PACE_DIFF_THRESHOLD': 5.0,
    'RECOVERY_THRESHOLD': 0.6
}

# 🏀 VALORES POR DEFECTO PARA ESTADISTICAS
DEFAULT_STATS = {
    'field_goals_attempted': 85, 'field_goals_made': 35,
    'free_throws_attempted': 20, 'free_throws_made': 15,
    'offensive_rebounds': 10, 'defensive_rebounds': 30,
    'turnovers': 14, 'assists': 22,
    '3point_field_goals_made': 8, '2point_field_goals_made': 27
}

# 🎨 CONFIGURACION DE INTERFAZ
UI_MESSAGES = {
    'welcome': "🏀 BASCKET MLG v2.1 - ALERTAS INTELIGENTES EDITION",
    'features': "✨ Nuevas caracteristicas: Analisis contextual experto y patrones historicos",
    'training_start': "🔄 Iniciando entrenamiento con metricas mejoradas...",
    'prediction_mode': "🎯 MODO DE PREDICCION CON ALERTAS ACTIVADO...",
    'live_mode_title': "--- MODO EN VIVO - CON PACE, FOUR FACTORS & ALERTAS INTELIGENTES ---",
    'live_mode_subtitle': "🏀 Sistema mejorado con analisis contextual experto"
}
