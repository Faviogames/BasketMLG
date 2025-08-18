# ===========================================
# Archivo: config.py (v2.1 - FEATURES MUERTAS ELIMINADAS)
# Configuraciones y constantes centralizadas con alertas de balance
# ✅ ELIMINADO: 30 features muertas (0.0000% importancia)
# ✅ OPTIMIZADO: ADVANCED_STATS_COLS reducido de 10 → 3 estadísticas útiles
# ✅ ESPERADO: Mejora MAE 7.1 → 6.8-6.9
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
# ✅ CAMBIO CRÍTICO: Activar filtrado para mejor rendimiento
ENABLE_FEATURE_FILTERING = True  # Era False - AHORA True
print(f"Lista de features muertas integrada: {len(UNIVERSALLY_USELESS_FEATURES)} features")

# 📁 CONFIGURACIONES DE CARPETAS
DATA_FOLDER = './leagues'
MODELS_FOLDER = './models'
PROCESSED_FILES_PATH = './processed_files.json'

# ✅ MEJORADO: METRICAS DE MOMENTUM CRITICAS (con EMA + 8 optimizadas)
MOMENTUM_STATS_COLS = [
    # Features básicas (compatibilidad)
    'win', 'plus_minus', 'win_rate', 'avg_plus_minus', 
    'scoring_efficiency', 'defensive_stops', 'clutch_performance',
    # 🆕 8 FEATURES OPTIMIZADAS OVER/UNDER AÑADIDAS:
    'scoring_consistency',          # Consistencia en puntos anotados
    'pace_stability',              # Estabilidad del ritmo de juego
    'offensive_efficiency_trend',   # Tendencia ofensiva reciente vs histórica
    'quarter_scoring_pattern',      # Patrón de scoring por cuartos (Q1→Q4)
    'clutch_efficiency',           # Rendimiento en juegos de alto scoring
    'defensive_fatigue',           # Fatiga defensiva (más puntos permitidos)
    'shooting_rhythm',             # Ritmo de tiro (consistencia FG%)
    'game_flow_indicator'          # Indicador de fluidez del juego
]

# 🗑️ OPTIMIZADO: ESTADISTICAS AVANZADAS (LIMPIADAS - ELIMINADAS 7 ESTADÍSTICAS MUERTAS)
ADVANCED_STATS_COLS = [
    # ✅ SOLO MANTENER ESTAS 3 ESTADÍSTICAS ÚTILES:
    'possessions',  # Útil para pace calculations
    'pace',         # Complementa pace_stability de features optimizadas  
    'ortg'          # Útil como backup de offensive metrics 
    # 🗑️ ELIMINADAS: drtg, efg_percentage, tov_percentage, oreb_percentage, 
    #                ft_rate, offensive_efficiency, defensive_efficiency
    # 🎯 RAZÓN: Las 8 features optimizadas Over/Under las reemplazan completamente
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
    'medium_term': 8,  # Ultimos 8 partidos - forma reciente  
    'long_term': 12    # Ultimos 12 partidos - tendencia estacional
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

# 📊 CARACTERISTICAS AVANZADAS - OPTIMIZADAS (REDUCIDAS DE 30 → 9 FEATURES)
# 🎯 NOTA: Ahora genera solo 9 features (3 stats × 3 variants) en lugar de 30
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
        print(f"✅ Features filtradas: {len(FEATURES_TO_USE)} -> {len(filtered)} (eliminadas: {len(FEATURES_TO_USE) - len(filtered)})")
        return filtered
    else:
        print(f"⚠️ Filtrado desactivado: usando {len(FEATURES_TO_USE)} features sin filtrar")
        return FEATURES_TO_USE

# Nueva variable para usar en training
FILTERED_FEATURES_TO_USE = get_filtered_features_to_use()

# 📄 MAPEO DE NORMALIZACION DE CLAVES
KEY_MAP = {
    'field_goals_': 'field_goals_percentage',
    '2-point_field_g_attempted': '2point_field_goals_attempted',
    '2point_field_goals_': '2point_field_goals_percentage',
    '3-point_field_g_attempted': '3point_field_goals_attempted',
    '3-point_field_goals_': '3point_field_goals_percentage',
    'free_throws_': 'free_throws_percentage'
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
    'welcome': "🏀 BASCKET MLG v2.3 - DEAD FEATURES ELIMINATED EDITION",
    'features': "✨ OPTIMIZADO: 30 features muertas eliminadas + MAE mejorado",
    'training_start': "🔄 Iniciando entrenamiento con features limpias optimizadas...",
    'prediction_mode': "🎯 MODO DE PREDICCION OPTIMIZADO (SIN FEATURES MUERTAS)...",
    'live_mode_title': "--- MODO EN VIVO - FEATURES OPTIMIZADAS & SIN RUIDO ---",
    'live_mode_subtitle': "🏀 Sistema limpio: 8 features Over/Under + 21 features menos ruido"
}

# ===========================================
# 🆕 FUNCIONES DE OPTIMIZACIÓN Y VERIFICACIÓN
# ===========================================

def get_optimized_momentum_stats_cols():
    """Retorna la lista completa de columnas de momentum optimizadas"""
    return MOMENTUM_STATS_COLS

def print_optimized_feature_summary():
    """Muestra resumen optimizado de features (VERSIÓN COMPACTA)"""
    print(f"\n🔧 FEATURES OPTIMIZADAS:")
    print(f"   💪 Momentum: {len(MOMENTUM_STATS_COLS)} (incluye 8 optimizadas)")
    print(f"   📊 Advanced: {len(ADVANCED_STATS_COLS)} (reducido 10→3, eliminadas 7 muertas)")
    print(f"   🎯 Context: {len(PERFORMANCE_CONTEXT_COLS)}")
    print(f"   🔴 Live: {len(LIVE_GAME_FEATURES)}")
    print(f"   ⚖️ Balance: {len(BALANCE_FEATURES)}")
    print(f"   🗑️ Eliminadas: 21 features avanzadas muertas")
    print(f"   ✅ Total final: ~{len(FILTERED_FEATURES_TO_USE)} features")

def is_optuna_enabled():
    """Función de compatibilidad para optimización"""
    return False

def get_optuna_config():
    """Función de compatibilidad para configuración de optimización"""
    return {}

# 🎯 VERIFICACIÓN RÁPIDA (SOLO SI SE EJECUTA DIRECTAMENTE)
if __name__ == "__main__":
    print("🔍 VERIFICACIÓN RÁPIDA:")
    print(f"✅ ADVANCED_STATS_COLS: {len(ADVANCED_STATS_COLS)} stats (era 10)")
    print(f"✅ PRE_GAME_FEATURES_ADVANCED: {len(PRE_GAME_FEATURES_ADVANCED)} features (era 30)")
    print(f"📈 Mejora esperada MAE: 7.1 → 6.8-6.9")
    print_optimized_feature_summary()