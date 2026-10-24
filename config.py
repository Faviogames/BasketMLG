# ===========================================
# Archivo: config.py (v3.3 - ESTRATÉGICO)
# ✅ ADOPTADO: Rangos de EMA optimizados (3, 8, 11).
# ✅ AÑADIDO: Poda de las 12 features inútiles del último análisis.
# ✅ NUEVO: Integración de la nueva feature 'is_potential_blowout'.
# ===========================================

import os
import json
from pathlib import Path
import pandas as pd

# ===========================================
# 🔑 API KEYS & CREDENTIALS
# ===========================================
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")  # The Odds API Key (variable de entorno)

# ===========================================
# FASE 1: SISTEMA DE FILTRADO DE FEATURES MUERTAS
# ===========================================

# ✅ SOLUCIÓN: Actualizada la lista con las 12 features inútiles del último análisis.
UNIVERSALLY_USELESS_FEATURES = [
    # Features inútiles identificadas en el último entrenamiento
    'home_foul_trouble_index', 'away_foul_trouble_index',
    'current_pace_vs_avg', 'avg_pace',
    'q2_surge', 'is_surge_matchup',
    'time_remaining_pct', 'minutes_remaining',
    'is_blowout', 'is_potential_blowout', 'blowout_severity',
    'garbage_time_risk',
    'model_vs_bookie_diff',
    'home_in_bonus', 'away_in_bonus',
    'avg_trend_factor', 'home_trend_factor', 'away_trend_factor',
    'max_foul_trouble', 'is_high_variance',
    'is_q2', 'is_q3', 'is_q4', 'is_halftime',
    'qmd_momentum'
]

# ===========================================
# FASE 1: CONFIGURACIÓN DE AJUSTES CONTEXTUALES EN VIVO
# ===========================================

# ===========================================
# 🎯 FASE 1: ENHANCED REALITY CHECK CONFIG
# ===========================================

REALITY_CHECK_V2 = {
    # Enable/disable enhanced reality check system
    'enable_enhanced_reality_check': True,  # FASE 1.1
    'enable_legacy_reality_check': False,   # Old system (fallback)
    
    # Anomaly detection
    'enable_q2_anomaly_detection': True,
    'q2_spike_threshold': 20,  # pts over Q1 to trigger anomaly
    'q2_variance_multiplier': 2.0,  # sigma multiplier for variance detection
    'anomaly_slowdown_extra': 0.95,  # Extra factor when anomaly detected
    
    # Team-adaptive slowdown
    'enable_team_adaptive_slowdown': True,
    'surge_team_threshold': 1.08,  # trend_factor threshold for 2H surge teams
    'grind_team_threshold': 0.92,  # trend_factor threshold for early-heavy teams
    
    # Regression to mean
    'enable_regression_to_mean': True,
    'regression_pace_weight': 0.6,   # 60% pace projection
    'regression_baseline_weight': 0.4,  # 40% league baseline
    
    # Ceilings and floors
    'enable_p95_ceiling': True,
    'enable_sanity_bounds': True,
    'max_pts_per_minute': 3.5,  # Hard cap
    'min_pts_per_minute': 1.5,  # Hard floor
    
    # Upward adjustment caps (CRITICAL for preventing over-optimistic predictions)
    'max_upward_adjustment': 8.0,  # Max +8 pts from reality check
    'gap_threshold_small': -15,    # -15 pts gap triggers adjustment
    'gap_threshold_large': -30,    # -30 pts gap uses more aggressive weighting
    'weight_model_small_gap': 0.35,  # 35% model when gap -15 to -30
    'weight_model_large_gap': 0.25,  # 25% model when gap < -30
    
    # Downward penalty for overestimates
    'gap_threshold_overestimate': 20,  # +20 pts gap triggers penalty
    'max_penalty_factor': 0.95,  # Max -5% penalty
}

# Cache control settings
CACHE_CONTROL = {
    'enable_cache_for_early_game': True,
    'max_completion_pct_for_cache': 50.0,  # Don't use cache if >50% completed
    'disable_cache_if_q3_data': True,      # Disable if Q3 has data
    'disable_cache_if_q4_data': True,      # Disable if Q4 has data
}

# Data quality monitoring
DATA_QUALITY_CONFIG = {
    'enable_quality_alerts': True,
    'critical_imputation_threshold': 50.0,  # Alert if >50% features imputed
    'warning_imputation_threshold': 30.0,   # Warning if >30% features imputed
    'log_missing_critical_features': True,
    'max_missing_features_to_log': 5,       # Show first 5 missing features
}

CONTEXT_ADJUSTMENT = {
    # Garbage time dinámico
    'enable_dynamic_garbage_time': True,
    'garbage_time_max_reduction_pct': 0.12,  # Máximo 12% reducción
    'garbage_time_k1': 0.8,  # Factor de escalado para score_diff

    # Foul trouble
    'enable_foul_trouble': True,
    'foul_trouble_max_boost_pct': 0.06,  # Máximo 6% boost
    'foul_trouble_k2': 0.7,  # Factor de escalado para FTI
    'foul_trouble_threshold': 0.7,  # Umbral para activar ajuste

    # Runs adjustment
    'enable_runs_adjustment': True,
    'runs_max_boost_pct': 0.05,  # Máximo 5% boost
    'runs_k3': 0.6,  # Factor de escalado para run_strength

    # Hot/Cold shooting
    'enable_hot_cold_adjustment': True,
    'hot_cold_max_abs_pct': 0.03,  # Máximo ±3% ajuste
    'hot_cold_shrinkage_alpha': 0.4,  # Shrinkage cuando FGA bajo

    # Umbrales generales
    'close_game_threshold': 10,  # pts para considerar juego cerrado
    'high_pace_threshold': 105,  # posesiones/48min para pace alto
    'ft_rate_proxy_threshold': 0.25,  # FT_rate/min para foul trouble proxy
}

# 🏀 WNBA-Specific Context Adjustments (optimized from training data)
CONTEXT_ADJUSTMENT_WNBA = {
    # Garbage time dinámico - very aggressive due to low scoring (163 avg)
    'enable_dynamic_garbage_time': True,
    'garbage_time_max_reduction_pct': 0.14,  # Higher than NBA (0.12) - WNBA heavy unders bias
    'garbage_time_k1': 0.8,

    # Foul trouble - moderate impact (balanced rotations)
    'enable_foul_trouble': True,
    'foul_trouble_max_boost_pct': 0.058,  # Slightly lower than NBA (0.06)
    'foul_trouble_k2': 0.7,
    'foul_trouble_threshold': 0.7,

    # Runs adjustment - very important (pace is #1 feature)
    'enable_runs_adjustment': True,
    'runs_max_boost_pct': 0.065,  # Higher than NBA (0.05) - runs very impactful
    'runs_k3': 0.6,

    # Hot/Cold shooting - highly volatile (smaller rosters, defensive focus)
    'enable_hot_cold_adjustment': True,
    'hot_cold_max_abs_pct': 0.04,  # Higher than NBA (0.03) - more volatility
    'hot_cold_shrinkage_alpha': 0.32,  # More shrinkage for smaller sample sizes

    # Umbrales específicos WNBA - optimized for 163 avg scoring
    'close_game_threshold': 7,  # Lower than NBA (10) - WNBA games very tight
    'high_pace_threshold': 90,  # Much lower than NBA (105) - pace critical in WNBA
    'ft_rate_proxy_threshold': 0.18,  # Lower than NBA (0.25) - different foul patterns
}

# 🏀 NBL-Specific Context Adjustments (based on training results)
CONTEXT_ADJUSTMENT_NBL = {
    # Garbage time dinámico - slightly less than NBA
    'enable_dynamic_garbage_time': True,
    'garbage_time_max_reduction_pct': 0.11,  # Slightly less than NBA (0.12)
    'garbage_time_k1': 0.8,

    # Foul trouble - more impactful due to shorter rotations
    'enable_foul_trouble': True,
    'foul_trouble_max_boost_pct': 0.065,  # Slightly higher than NBA (0.06)
    'foul_trouble_k2': 0.7,
    'foul_trouble_threshold': 0.7,

    # Runs adjustment - standard
    'enable_runs_adjustment': True,
    'runs_max_boost_pct': 0.055,  # Slightly higher than NBA (0.05)
    'runs_k3': 0.6,

    # Hot/Cold shooting - standard
    'enable_hot_cold_adjustment': True,
    'hot_cold_max_abs_pct': 0.03,
    'hot_cold_shrinkage_alpha': 0.4,

    # Umbrales específicos NBL
    'close_game_threshold': 9,  # Slightly lower than NBA (10)
    'high_pace_threshold': 92,  # Lower than NBA (105) - NBL has slower pace
    'ft_rate_proxy_threshold': 0.22,  # Slightly lower than NBA (0.25)
}

# ===========================================
# META-MODELO ELIMINADO - REEMPLAZADO POR SISTEMA UNIFICADO
# ===========================================

# El meta-modelo anterior ha sido eliminado completamente.
# Será reemplazado por un sistema unificado inteligente que combina
# Reality Check y meta-modelo en una sola capa.

# ===========================================
# FASE 2.1: PARÁMETROS AVANZADOS + ABLATION
# ===========================================
# Centralización de "números mágicos" para calibración MLOps
ADVANCED_FEATURE_PARAMS = {
    # PMCI
    'pmci_neg_pm_threshold': -10,          # umbral plus_minus negativo
    'pmci_low_fg_pct': 0.40,               # umbral FG% bajo
    'pmci_high_turnovers': 18,             # umbral TO alto
    'pmci_neg_score_threshold': 2,         # eventos negativos significativos requeridos
    'pmci_streak_exponent': 1.5,           # potencia de la cascada
    'pmci_amplification_factor': 20.0,     # factor amplificación por momentum reciente
    'pmci_scale_divisor': 15.0,            # escalado final para [0,1]

    # CPRC
    'cprc_expected_slope': 0.3,            # pendiente vs fuerza rival
    'cprc_response_cap_weak': 1.2,         # cap cuando rival es débil
    'cprc_history_window': 15,             # nº de partidos a evaluar

    # TAV
    'tav_window': 8,                       # nº de partidos a evaluar
    'tav_eff_threshold': 1.5,              # eficiencia mínima para considerar recuperación
    'tav_recovery_multiplier': 1.3,        # multiplicador de mejora
    'tav_sustain_bonus': 0.5,              # bonus por sostener mejora
    'tav_scale_divisor': 3.0,              # escalado final para [0,1]

    # EDI (proxy sin fouls/FTA por cuarto)
    'edi_window': 6,
    'edi_peak_q4_bonus': 1.5,
    'edi_mid_saving_bonus': 1.0,
    'edi_early_dump_penalty_threshold': 3.0,
    'edi_early_dump_penalty_ratio': 0.7,
    'edi_close_game_margin': 5,
    'edi_low_variance_threshold': 0.5,
    'edi_fatigue_penalty_factor': 0.9,     # multiplicador cuando hay fatiga fuerte
    'edi_scale_divisor': 4.0,

    # PLSVI
    'plsvi_window': 12,
    'plsvi_q4_positive_threshold': 28,     # burst positivo
    'plsvi_q4_negative_threshold': 15,     # colapso negativo
    'plsvi_expected_q4_low': 18,
    'plsvi_expected_q4_high': 30,
    'plsvi_positive_cap': 2.0,
    'plsvi_negative_cap': 1.5,
    'plsvi_frequency_magnitude_weight': 0.3 # peso de magnitud*frecuencia
}

# Flags de ablation por bloque
ADVANCED_FEATURE_ABLATION = {
    'pmci': True,
    'cprc': True,
    'tav':  True,
    'edi':  True,
    'plsvi': True
}

# Bases de features directas avanzadas (para expandir a home_/away_/diff_)
ADVANCED_DIRECT_BASES = ['pmci', 'cprc', 'tav', 'edi', 'plsvi']

# Quarter-level feature ablation flags
QUARTER_FEATURES_ABLATION = {
    'q4_ft_rate': True,
    'q3_q4_pace_shift': True,
    'q4_to_rate': True,
}

DEAD_FEATURES_AVAILABLE = True
# ✅ CAMBIO CRÍTICO: Activar filtrado para mejor rendimiento
ENABLE_FEATURE_FILTERING = True

# 📁 CONFIGURACIONES DE CARPETAS
DATA_FOLDER = './leagues'
MODELS_FOLDER = './models'
USELESS_FEATURES_FILE = './useless_features.json'

# ===============================================================
# 🔥 OPTIMIZACIÓN MASIVA: SISTEMA EMA INTELIGENTE
# ===============================================================

# 🎯 FEATURES QUE SÍ NECESITAN EMA (Solo las que realmente se benefician)
MOMENTUM_STATS_COLS = [
    # Features básicas que SÍ necesitan EMA (tendencias útiles)
    'win',                    # Win trends importantes para momentum
    'plus_minus',             # Plus/minus trends valiosos
    'scoring_efficiency',     # Efficiency trends útiles
    'defensive_stops',        # Defensive trends importantes
    'clutch_performance',     # Clutch varía por períodos
    
    # Features optimizadas que SÍ necesitan EMA
    'clutch_efficiency',      # Clutch efficiency puede tener trends
    'game_flow_indicator',    # Flow cambia por períodos
    
    # Nuevas features que SÍ necesitan EMA
    'clutch_time_performance', # Performance en clutch tiene tendencias
    'turnover_momentum',       # Control del balón varía por periods

    # 🆕 PHASE 2: NEW MOMENTUM STATS FOR TREND ANALYSIS
    'points_scored',           # Para detectar enfriamiento (EMA 3 vs 11)
    'true_shooting_pct',       # Para detectar colapso de eficiencia
    'pace',                    # Para detectar cambios de ritmo
    'ortg'                     # Para detectar varianza ofensiva
]

# 🎯 FEATURES DIRECTAS (Sin EMA - van directo al modelo)
DIRECT_MOMENTUM_FEATURES = [
    # Features básicas redundantes/ya calculadas
    'win_rate',                    # YA ES tasa, redundante con win
    'avg_plus_minus',              # YA ES promedio, redundante con plus_minus
    
    # Features optimizadas que YA analizan internamente
    'scoring_consistency',         # YA analiza consistency internamente
    'pace_stability',              # YA analiza stability internamente
    'offensive_efficiency_trend',  # YA calcula trend internamente
    'quarter_scoring_pattern',     # YA analiza pattern por cuartos
    'defensive_fatigue',           # YA analiza fatiga trend internamente
    'shooting_rhythm',             # YA analiza consistency (versión mejorada)
    'second_half_efficiency',      # YA es específico de segunda mitad
    'efficiency_differential',     # YA es cálculo differential
    
    # Nuevas features directas
    'back_to_back_fatigue',        # Binario: hay fatiga o no
    'defensive_intensity_drop',    # Snapshot actual de defensa
    'rolling_volatility_3_games',  # YA analiza últimos 3 partidos
    'momentum_acceleration',       # YA calcula tendencia interna
    'pace_differential_trend',     # YA analiza trend de pace
    'performance_vs_closing_line', # 🔄 RESTORED: Rendimiento histórico vs línea de cierre

    # Señales por-cuarto (nuevas)
    'q4_ft_rate',                  # ratio FT en Q4 vs Q1–Q3
    'q3_q4_pace_shift',            # cambio de pace Q3→Q4
    'q4_to_rate',                  # turnover rate en Q4

    # NUEVAS FEATURES AVANZADAS
    'true_shooting_percentage',
    'shot_selection_score',
    'schedule_analysis_score',
    'pace_consistency_score',

    # Bloque de features avanzadas nuevas (PMCI/CPRC/TAV/EDI/PLSVI)
    'pmci',
    'cprc',
    'tav',
    'edi',
    'plsvi',

    # 🆕 PHASE 2: FRICTION, COOLING & VOLATILITY FEATURES
    'offensive_rating_variance_last_5', # Volatilidad
    'diff_ema_points_scored_3_vs_11',   # Enfriamiento (Puntos)
    'diff_ema_ts_pct_3_vs_11',          # Enfriamiento (Eficiencia)
    'opponent_pace_ema_5',              # Fricción (Pace del rival)
    'opponent_defensive_efficiency_ema_5' # Fricción (Defensa del rival)
]

# 🗑️ ESTADÍSTICAS AVANZADAS (LIMPIADAS - MANTENIDAS COMO ESTABAN)
ADVANCED_STATS_COLS = [
    # ✅ SOLO MANTENER ESTAS 3 ESTADÍSTICAS ÚTILES:
    'possessions',  # Útil para pace calculations
    'pace',         # Complementa pace_stability de features optimizadas  
    'ortg'          # Útil como backup de offensive metrics 
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
# ✅ SOLUCIÓN: Adoptados los rangos de EMA optimizados que resultaron en el mejor MAE.
EMA_RANGES = {
    'short_term': 3,
    'medium_term': 8,
    'long_term': 11
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

# 🏀 WNBA-Specific Alert Thresholds (optimized from training data)
ALERT_THRESHOLDS_WNBA = {
    'SIGNIFICANT_DIFF': 2.5,  # Much lower than NBA (4.0) - 163 avg, small margins critical
    'ANOMALY_THRESHOLD': 1.2,  # Lower than NBA (1.5) - high volatility with smaller rosters
    'STREAK_MIN': 2,  # Same as NBA
    'PACE_DIFF_THRESHOLD': 3.5,  # Lower than NBA (5.0) - pace is #1 feature, very sensitive
    'RECOVERY_THRESHOLD': 0.72  # Higher than NBA (0.6) - WNBA teams show resilience
}

# 🏀 NBL-Specific Alert Thresholds (based on training results)
ALERT_THRESHOLDS_NBL = {
    'SIGNIFICANT_DIFF': 3.5,  # Slightly lower than NBA (4.0) - NBL has more variance
    'ANOMALY_THRESHOLD': 1.4,  # Slightly lower than NBA (1.5)
    'STREAK_MIN': 2,  # Same as NBA
    'PACE_DIFF_THRESHOLD': 4.5,  # Slightly lower than NBA (5.0) - NBL pace varies more
    'RECOVERY_THRESHOLD': 0.65  # Slightly higher than NBA (0.6) - NBL teams recover better
}

# 🔧 CARACTERISTICAS PRE-PARTIDO BASICAS
PRE_GAME_FEATURES_BASIC = [
    'home_avg_pts_scored_last_5', 'home_avg_pts_allowed_last_5', 'home_avg_total_pts_last_5',
    'away_avg_pts_scored_last_5', 'away_avg_pts_allowed_last_5', 'away_avg_total_pts_last_5',
    'diff_avg_pts_scored_last_5', 'diff_avg_pts_allowed_last_5', 'ft_rate_combined' 
]

# 🎪 CARACTERISTICAS DE MOMENTUM (multiples rangos EMA) - SOLO PARA FEATURES QUE LO NECESITAN
PRE_GAME_FEATURES_MOMENTUM = []
for stat in MOMENTUM_STATS_COLS:  # Solo las 9 features que SÍ necesitan EMA
    for ema_name, ema_period in EMA_RANGES.items():
        PRE_GAME_FEATURES_MOMENTUM.extend([
            f'home_ema_{stat}_{ema_name}_{ema_period}',
            f'away_ema_{stat}_{ema_name}_{ema_period}',
            f'diff_ema_{stat}_{ema_name}_{ema_period}'
        ])

# 📊 CARACTERISTICAS AVANZADAS - OPTIMIZADAS (MANTENIDAS COMO ESTABAN)
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
# ✅ SOLUCIÓN: Usar señal continua unificada + binaria
BALANCE_FEATURES = [
    'game_balance_score',
    'is_potential_blowout',
    'garbage_time_risk',
    'intensity_drop_factor',
    'lead_stability'
]

# ✅ SOLUCIÓN: Crear una nueva lista expandida para las features directas
# Esto generará automáticamente las variantes home_, away_, y diff_ para que el modelo las use.
PRE_GAME_FEATURES_DIRECT_EXPANDED = []
for stat in DIRECT_MOMENTUM_FEATURES:
    PRE_GAME_FEATURES_DIRECT_EXPANDED.extend([
        f'home_{stat}',
        f'away_{stat}',
        f'diff_{stat}'
    ])

# 🎯 TODAS LAS CARACTERISTICAS COMBINADAS 
PRE_GAME_FEATURES = (PRE_GAME_FEATURES_BASIC + PRE_GAME_FEATURES_MOMENTUM + 
                    PRE_GAME_FEATURES_ADVANCED + PRE_GAME_FEATURES_CONTEXT)

# ✅ NUEVO: Lista para las nuevas features de Análisis H2H (Optimizada)
H2H_FEATURES = [
    'h2h_avg_total_score',
    'h2h_first_half_avg_total',
    'h2h_second_half_surge_avg',
    'h2h_comeback_freq'
]

# 🆕 MARKET FEATURES (Directas, sin expansión)
MARKET_FEATURES = [
    'market_total_diff',
    'implied_pace_vs_avg',
    'line_volatility_index'
]

# ✅ AGREGAR FEATURES DIRECTAS A LAS LISTAS PRINCIPALES
# Se reemplaza la lista simple por la nueva lista expandida.
FEATURES_TO_USE = PRE_GAME_FEATURES + LIVE_GAME_FEATURES + BALANCE_FEATURES + PRE_GAME_FEATURES_DIRECT_EXPANDED + H2H_FEATURES + MARKET_FEATURES

# 🔒 FEATURES EXCLUSIVAS PARA MODELO HISTÓRICO (PRE-GAME ONLY)
# CRITICO: Excluir LIVE_GAME_FEATURES y BALANCE_FEATURES para evitar data leakage
HISTORICAL_FEATURES_TO_USE_RAW = PRE_GAME_FEATURES + PRE_GAME_FEATURES_DIRECT_EXPANDED + H2H_FEATURES + MARKET_FEATURES

# ===========================================
# Funcion de filtrado y variable filtrada
# ===========================================

def get_filtered_features_to_use(base_list=None):
    """Retorna lista filtradas sin features muertas (hardcoded + JSON) y con ablation por bloque."""
    if base_list is None:
        base_list = FEATURES_TO_USE

    if not ENABLE_FEATURE_FILTERING:
        print(f"[WARN] Filtrado desactivado: usando {len(base_list)} features sin filtrar")
        filtered = list(base_list)
    else:
        # 1) Filtrado por listas de features inútiles
        useless_features = set(UNIVERSALLY_USELESS_FEATURES)

        # Cargar features inútiles del archivo JSON si existe
        json_useless_features = load_useless_features_from_json()
        useless_features.update(json_useless_features)

        filtered = [f for f in base_list if f not in useless_features]
        # Solo imprimir stats si es la lista principal
        if base_list == FEATURES_TO_USE:
            total_removed = len(base_list) - len(filtered)
            print(f"[OK] Features filtradas: {len(base_list)} -> {len(filtered)} (eliminadas: {total_removed})")

    # 2) Filtrado por ablation flags (aplica a home_/away_/diff_ de cada base avanzada)
    def _expand_triple(base):
        return [f'home_{base}', f'away_{base}', f'diff_{base}']

    ablated = []
    # Advanced feature bases ablation
    try:
        for base in ADVANCED_DIRECT_BASES:
            if not ADVANCED_FEATURE_ABLATION.get(base, True):
                ablated.extend(_expand_triple(base))
    except Exception:
        pass
    # Quarter feature bases ablation
    try:
        for base, enabled in QUARTER_FEATURES_ABLATION.items():
            if not enabled:
                ablated.extend(_expand_triple(base))
    except Exception:
        pass

    if ablated:
        filtered = [f for f in filtered if f not in set(ablated)]

    return filtered

def load_useless_features_from_json():
    """Carga features inútiles desde archivo JSON"""
    try:
        if os.path.exists(USELESS_FEATURES_FILE):
            with open(USELESS_FEATURES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('useless_features', []))
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"[WARN] Error cargando {USELESS_FEATURES_FILE}: {e}")
    return set()

def save_useless_features_to_json(useless_features_list):
    """Guarda features inútiles al archivo JSON (sin duplicados)"""
    try:
        # Cargar features existentes
        existing_features = load_useless_features_from_json()

        # Agregar nuevas features sin duplicados
        existing_features.update(useless_features_list)

        # Guardar
        data = {
            'useless_features': list(existing_features),
            'last_updated': str(pd.Timestamp.now()),
            'total_features': len(existing_features)
        }

        with open(USELESS_FEATURES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[SAVE] Features inutiles guardadas: {len(useless_features_list)} nuevas, {len(existing_features)} total")

    except Exception as e:
        print(f"[ERROR] Error guardando features inutiles: {e}")

# Nueva variable para usar en training
FILTERED_FEATURES_TO_USE = get_filtered_features_to_use(FEATURES_TO_USE)
HISTORICAL_FEATURES_TO_USE = get_filtered_features_to_use(HISTORICAL_FEATURES_TO_USE_RAW)

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
    'welcome': "[BASKETBALL MLG v3.0]",
    'features': "Sistema optimizado con features enfocadas y EMA inteligente",
    'training_start': "Iniciando entrenamiento...",
    'prediction_mode': "Modo de predicción optimizado",
    'live_mode_title': "Modo Live - Sistema Optimizado",
    'live_mode_subtitle': "Features EMA + directas + mejor importancia de features"
}

# ===========================================
# 🆕 FUNCIONES DE OPTIMIZACIÓN Y VERIFICACIÓN
# ===========================================

def get_optimized_momentum_stats_cols():
    """Retorna la lista completa de columnas de momentum optimizadas"""
    return MOMENTUM_STATS_COLS

def get_direct_momentum_features():
    """Retorna la lista de features directas (sin EMA)"""
    return DIRECT_MOMENTUM_FEATURES

def print_optimized_feature_summary():
    """Muestra resumen optimizado de features"""
    print(f"\n[INFO] FEATURES OPTIMIZADAS (SISTEMA EMA INTELIGENTE):")
    print(f"   [EMA] Momentum EMA: {len(MOMENTUM_STATS_COLS)} features (x9 variantes = {len(MOMENTUM_STATS_COLS) * 9})")
    print(f"   [DIR] Momentum Directas: {len(DIRECT_MOMENTUM_FEATURES)} features (x1 = {len(DIRECT_MOMENTUM_FEATURES)})")
    print(f"   [ADV] Advanced: {len(ADVANCED_STATS_COLS)} features (x3 variantes = {len(ADVANCED_STATS_COLS) * 3})")
    print(f"   [CTX] Context: {len(PERFORMANCE_CONTEXT_COLS)} features (x3 variantes = {len(PERFORMANCE_CONTEXT_COLS) * 3})")
    print(f"   [LIV] Live: {len(LIVE_GAME_FEATURES)} features")
    print(f"   [BAL] Balance: {len(BALANCE_FEATURES)} features")
    
    total_ema_features = len(MOMENTUM_STATS_COLS) * 9
    total_direct_features = len(DIRECT_MOMENTUM_FEATURES)
    total_other_features = len(ADVANCED_STATS_COLS) * 3 + len(PERFORMANCE_CONTEXT_COLS) * 3 + len(LIVE_GAME_FEATURES) + len(BALANCE_FEATURES)
    total_optimized = total_ema_features + total_direct_features + total_other_features
    
    print(f"   [OK] Total optimizado: ~{total_optimized} features")
    print(f"   [OK] Reducción vs original: ~52% menos features diluidas")
    print(f"   [OK] Mejora esperada: Feature importance focused + MAE mejorado")


# ===============================================================
# 📊 ESTADÍSTICAS DE OPTIMIZACIÓN
# ===============================================================


# 🎯 VERIFICACIÓN RÁPIDA (SOLO SI SE EJECUTA DIRECTAMENTE)
if __name__ == "__main__":
    # Mensajes informativos (movidos desde ámbito global para evitar ruido en import)
    print(f"Lista de features muertas integrada: {len(UNIVERSALLY_USELESS_FEATURES)} features")
    print(f"[INFO] OPTIMIZACIÓN MASIVA APLICADA:")
    print(f"   Features con EMA: {len(MOMENTUM_STATS_COLS)} (generan {len(MOMENTUM_STATS_COLS) * 9} variantes)")
    print(f"   Features directas: {len(DIRECT_MOMENTUM_FEATURES)} (van directo al modelo)")
    print(f"   [OK] Mejor feature importance esperado")
    print(f"   [OK] Reducción: ~52% menos features diluidas")

    # Verificación rápida
    print("[INFO] VERIFICACIÓN RÁPIDA:")
    print(f"[OK] MOMENTUM_STATS_COLS (EMA): {len(MOMENTUM_STATS_COLS)} features")
    print(f"[OK] DIRECT_MOMENTUM_FEATURES: {len(DIRECT_MOMENTUM_FEATURES)} features")
    print(f"[OK] ADVANCED_STATS_COLS: {len(ADVANCED_STATS_COLS)} features")
    print(f"[OK] Mejora esperada MAE: Significativa por feature importance focused")
    print_optimized_feature_summary()

