# ===========================================
# Archivo: config.py
# Configuraciones y constantes centralizadas
# ===========================================

# 📁 CONFIGURACIONES DE CARPETAS
DATA_FOLDER = './leagues'  # 🆕 NUEVA CARPETA
MODELS_FOLDER = './models'
PROCESSED_FILES_PATH = './processed_files.json'

# 📊 MÉTRICAS DE MOMENTUM CRÍTICAS (con EMA)
MOMENTUM_STATS_COLS = [
    'win', 'plus_minus', 'win_rate', 'avg_plus_minus', 
    'scoring_efficiency', 'defensive_stops', 'clutch_performance'
]

# 📈 ESTADÍSTICAS AVANZADAS (con promedio simple)
ADVANCED_STATS_COLS = [
    'possessions', 'ortg', 'drtg',
    'efg_percentage', 'tov_percentage', 'oreb_percentage', 'ft_rate',
    'pace', 'offensive_efficiency', 'defensive_efficiency'
]

# 🎯 MÉTRICAS DE PERFORMANCE CONTEXTUAL
PERFORMANCE_CONTEXT_COLS = [
    'home_advantage_factor', 'comeback_ability', 'consistency_index'
]

# 🆕 MÉTRICAS POR CUARTO ESPECÍFICO (para alertas)
QUARTER_SPECIFIC_COLS = [
    'q1_points', 'q2_points', 'q3_points', 'q4_points',
    'first_half_points', 'second_half_points', 'second_half_surge'
]

# ⚡ RANGOS DE EMA PARA DIFERENTES ASPECTOS DEL MOMENTUM
EMA_RANGES = {
    'short_term': 3,   # Últimos 3 partidos - momentum inmediato
    'medium_term': 7,  # Últimos 7 partidos - forma reciente  
    'long_term': 15    # Últimos 15 partidos - tendencia estacional
}

# 🔧 CARACTERÍSTICAS PRE-PARTIDO BÁSICAS
PRE_GAME_FEATURES_BASIC = [
    'home_avg_pts_scored_last_5', 'home_avg_pts_allowed_last_5', 'home_avg_total_pts_last_5',
    'away_avg_pts_scored_last_5', 'away_avg_pts_allowed_last_5', 'away_avg_total_pts_last_5',
    'diff_avg_pts_scored_last_5', 'diff_avg_pts_allowed_last_5',
]

# 🎪 CARACTERÍSTICAS DE MOMENTUM (múltiples rangos EMA)
PRE_GAME_FEATURES_MOMENTUM = []
for stat in MOMENTUM_STATS_COLS:
    for ema_name, ema_period in EMA_RANGES.items():
        PRE_GAME_FEATURES_MOMENTUM.extend([
            f'home_ema_{stat}_{ema_name}_{ema_period}',
            f'away_ema_{stat}_{ema_name}_{ema_period}',
            f'diff_ema_{stat}_{ema_name}_{ema_period}'
        ])

# 📊 CARACTERÍSTICAS AVANZADAS
PRE_GAME_FEATURES_ADVANCED = []
for stat in ADVANCED_STATS_COLS:
    PRE_GAME_FEATURES_ADVANCED.extend([
        f'home_avg_{stat}_last_5',
        f'away_avg_{stat}_last_5',
        f'diff_avg_{stat}_last_5'
    ])

# 🏠 CARACTERÍSTICAS DE CONTEXTO DE PERFORMANCE
PRE_GAME_FEATURES_CONTEXT = []
for stat in PERFORMANCE_CONTEXT_COLS:
    PRE_GAME_FEATURES_CONTEXT.extend([
        f'home_avg_{stat}_last_10',
        f'away_avg_{stat}_last_10',
        f'diff_avg_{stat}_last_10'
    ])

# 🔴 CARACTERÍSTICAS EN VIVO
LIVE_GAME_FEATURES = [
    'q1_total', 'q2_total', 'q3_total', 'halftime_total', 'q3_end_total',
    'q1_diff', 'q2_diff', 'q3_diff', 'q2_trend', 'q3_trend', 'quarter_variance',
    'live_pace_estimate', 'live_efficiency_home', 'live_efficiency_away',
    'live_momentum_shift', 'quarter_consistency', 'comeback_indicator'
]

# 🎯 TODAS LAS CARACTERÍSTICAS COMBINADAS (INCLUYENDO BALANCE)
PRE_GAME_FEATURES = (PRE_GAME_FEATURES_BASIC + PRE_GAME_FEATURES_MOMENTUM + 
                    PRE_GAME_FEATURES_ADVANCED + PRE_GAME_FEATURES_CONTEXT)

FEATURES_TO_USE = PRE_GAME_FEATURES + LIVE_GAME_FEATURES + BALANCE_FEATURES

# 🔄 MAPEO DE NORMALIZACIÓN DE CLAVES
KEY_MAP = {
    'field_goals_': 'field_goals_percentage',
    '2-point_field_g_attempted': '2point_field_goals_attempted',
    '2point_field_goals_': '2point_field_goals_percentage',
    '3-point_field_g_attempted': '3point_field_goals_attempted',
    '3-point_field_goals_': '3point_field_goals_percentage',
    'free_throws_': 'free_throws_percentage'
}

# 🚨 CONFIGURACIÓN DEL SISTEMA DE ALERTAS
ALERT_TYPES = {
    'UNDER_PERFORMANCE': "⚠️ {team} anotando {diff:.1f} puntos menos que su promedio en {quarter} ({current:.1f} vs {avg:.1f})",
    'OVER_PERFORMANCE': "🔥 {team} anotando {diff:.1f} puntos más que su promedio en {quarter} ({current:.1f} vs {avg:.1f})",
    'SURGE_EXPECTED': "🚀 {team} históricamente tiene repunte en {period} (promedio: +{surge:.1f} pts)",
    'COLD_STREAK': "🧊 {team} en racha fría: {consecutive} cuartos consecutivos por debajo del promedio",
    'HOT_STREAK': "🔥 {team} en racha caliente: {consecutive} cuartos consecutivos por encima del promedio",
    'SLOW_START_RECOVERY': "📈 {team} suele recuperarse tras inicios lentos (probabilidad {recovery_rate:.0%})",
    'DEFENSIVE_COLLAPSE': "🛡️ {team} permitiendo {diff:.1f} pts más de lo usual - posible colapso defensivo",
    'PACE_SHIFT': "⚡ Ritmo de juego {direction} ({current_pace:.1f} vs promedio {avg_pace:.1f} posesiones/48min)",
    'SECOND_HALF_SURGE': "💪 {team} promedia {surge:.1f} pts más en segunda mitad - considerar ajustes",
    'CLOSING_STRENGTH': "🎯 {team} muy fuerte en cuartos finales (promedio Q4: {q4_avg:.1f} pts)"
}

# ⚖️ UMBRALES PARA DETECTAR ANOMALÍAS
ALERT_THRESHOLDS = {
    'SIGNIFICANT_DIFF': 4.0,  # Diferencia significativa en puntos
    'ANOMALY_THRESHOLD': 1.5,  # Desviaciones estándar para anomalías
    'STREAK_MIN': 2,  # Mínimo de cuartos para considerar racha
    'PACE_DIFF_THRESHOLD': 5.0,  # Diferencia significativa en pace
    'RECOVERY_THRESHOLD': 0.6  # Umbral para probabilidad de recuperación
}

# 🏀 VALORES POR DEFECTO PARA ESTADÍSTICAS
DEFAULT_STATS = {
    'field_goals_attempted': 85, 'field_goals_made': 35,
    'free_throws_attempted': 20, 'free_throws_made': 15,
    'offensive_rebounds': 10, 'defensive_rebounds': 30,
    'turnovers': 14, 'assists': 22,
    '3point_field_goals_made': 8, '2point_field_goals_made': 27
}

# 🎨 CONFIGURACIÓN DE INTERFAZ
UI_MESSAGES = {
    'welcome': "🏀 BASCKET MLG v2.1 - ALERTAS INTELIGENTES EDITION",
    'features': "✨ Nuevas características: Análisis contextual experto y patrones históricos",
    'training_start': "🔄 Iniciando entrenamiento con métricas mejoradas...",
    'prediction_mode': "🎯 MODO DE PREDICCIÓN CON ALERTAS ACTIVADO...",
    'live_mode_title': "--- MODO EN VIVO - CON PACE, FOUR FACTORS & ALERTAS INTELIGENTES ---",
    'live_mode_subtitle': "🏀 Sistema mejorado con análisis contextual experto"
}