# ===========================================
# Archivo: core/features.py (v2.5 - FASE 1A CORE MEJORADO)
# Sistema EMA avanzado + Features optimizadas integradas
# ✅ FIXED: Función duplicada, detección inteligente de columnas, limitadores de seguridad
# ✅ NUEVO: 6 estadísticas basketball, features Over/Under optimizadas sin imports externos
# ✅ CORREGIDO: calculate_momentum_metrics función unificada completa
# ✅ SOLUCIÓN PACE/POSSESSIONS IMPLEMENTADA
# ===========================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import copy

# 🆕 IMPORTS DESDE CONFIG CENTRALIZADO (UPDATED)
from config import (
    MOMENTUM_STATS_COLS, ADVANCED_STATS_COLS, PERFORMANCE_CONTEXT_COLS,
    QUARTER_SPECIFIC_COLS, EMA_RANGES, FILTERED_FEATURES_TO_USE, PRE_GAME_FEATURES,
    ADVANCED_FEATURE_PARAMS, ADVANCED_FEATURE_ABLATION, QUARTER_FEATURES_ABLATION
)

# 🆕 IMPORTS DESDE DATA_PROCESSING
from core.data_processing import (
    safe_int, safe_float, normalize_stats_keys,
    extract_quarter_data_for_alerts, process_raw_matches,
    impute_missing_stats, calculate_league_averages,
    calculate_team_league_averages
)

def safe_mean(series):
    """Calcula mean evitando warnings de NumPy"""
    if hasattr(series, 'dropna'):
        clean_data = series.dropna()
        return clean_data.mean() if len(clean_data) > 0 else 0.0
    return 0.0

def safe_divide(numerator, denominator, default=0.0):
    """Safe division that prevents division by zero and handles edge cases"""
    try:
        if denominator is None or denominator == 0 or pd.isna(denominator):
            return default
        if numerator is None or pd.isna(numerator):
            return default
        result = numerator / denominator
        # Check for infinity or NaN
        if pd.isna(result) or np.isinf(result):
            return default
        return result
    except (ZeroDivisionError, TypeError, ValueError):
        return default

def calculate_possessions(team_stats, opponent_stats):
    """Estima las posesiones de un equipo usando la fórmula estándar."""
    try:
        fga = safe_float(team_stats.get('field_goals_attempted', 0))
        orb = safe_float(team_stats.get('offensive_rebounds', 0))
        tov = safe_float(team_stats.get('turnovers', 0))
        fta = safe_float(team_stats.get('free_throws_attempted', 0))
        
        if fga == 0 and tov == 0:
            return np.nan
        
        possessions = fga - orb + tov + (0.44 * fta)
        return possessions if possessions > 0 else np.nan
    except:
        return np.nan
    
def calculate_pace(home_poss, away_poss, minutes_played=48):
    """Calcula el ritmo de juego (posesiones por 48 minutos)."""
    try:
        # Convertir inputs a números seguros
        home_poss = safe_float(home_poss) if home_poss is not None else 0
        away_poss = safe_float(away_poss) if away_poss is not None else 0
        
        # Validar que tenemos datos válidos
        if home_poss <= 0 and away_poss <= 0:
            return 90  # Default pace si no hay datos
        
        # Para partidos completos de NBA (48 minutos)
        # El pace ES directamente el promedio de posesiones del partido
        avg_possessions = (home_poss + away_poss) / 2
        
        # Como tus datos JSON son partidos completos, no necesitamos conversión temporal
        pace = avg_possessions
        
        # Validar rango razonable NBA (80-120 posesiones por partido)
        if pace < 80:
            pace = 80  # Mínimo razonable
        elif pace > 120:
            pace = 120  # Máximo razonable
        
        return pace
        
    except Exception as e:
        # Si hay cualquier error, retornar pace típico NBA
        return 90

def calculate_momentum_metrics(team_history_df):
    """
    🎯 FUNCIÓN UNIFICADA OPTIMIZADA - Integra features básicas + optimizadas
    ✅ OPTIMIZACIÓN: Solo features que necesitan EMA usan EMA
    ✅ DIRECTAS: Features que ya calculan tendencias van directo
    """
    if team_history_df.empty:
        # ✅ CAMBIAR: Retornar np.nan en lugar de valores constantes
        return {
        # Features EMA (serán procesadas por el sistema EMA)
        'win': 0.5, 'plus_minus': 0.0, 'scoring_efficiency': 1.0,
        'defensive_stops': 1.0, 'clutch_performance': 0.5,
        'clutch_efficiency': 1.0, 'game_flow_indicator': 0.5,
        'clutch_time_performance': 0.5, 'turnover_momentum': 0.5,
        
        # Features directas (van directo al modelo)
        'win_rate': 0.5, 'avg_plus_minus': 0.0,
        'scoring_consistency': 0.5, 'pace_stability': 0.5,
        'offensive_efficiency_trend': 1.0, 'quarter_scoring_pattern': 0.0,
        'defensive_fatigue': 0.0, 'shooting_rhythm': 0.5,
        'second_half_efficiency': 0.5, 'efficiency_differential': 0.0,
        'back_to_back_fatigue': 1.0, 'defensive_intensity_drop': 0.5,
        'rolling_volatility_3_games': 0.5, 'momentum_acceleration': 0.5,
        'pace_differential_trend': 0.5
        }

    metrics = {}
    
    # ==========================================
    # 🎯 FEATURES QUE SÍ NECESITAN EMA (el sistema las procesará)
    # ==========================================
    
    # Win (EMA útil para tendencias)
    if 'win' in team_history_df.columns and not team_history_df['win'].isna().all():
        metrics['win'] = safe_mean(team_history_df['win'])
    elif 'points_scored' in team_history_df.columns and 'points_allowed' in team_history_df.columns:
        wins = (team_history_df['points_scored'] > team_history_df['points_allowed']).astype(int)
        metrics['win'] = safe_mean(wins)
    else:
        metrics['win'] = 0.5
    
    # Plus/Minus (EMA útil para momentum)
    if 'plus_minus' in team_history_df.columns and not team_history_df['plus_minus'].isna().all():
        metrics['plus_minus'] = safe_mean(team_history_df['plus_minus'])
    elif 'points_scored' in team_history_df.columns and 'points_allowed' in team_history_df.columns:
        plus_minus = team_history_df['points_scored'] - team_history_df['points_allowed']
        metrics['plus_minus'] = safe_mean(plus_minus)
    else:
        metrics['plus_minus'] = 0.0
    
    # Scoring Efficiency (EMA útil para trends)
    if 'points_scored' in team_history_df.columns:
        recent_avg = safe_mean(team_history_df['points_scored'].tail(3))
        season_avg = safe_mean(team_history_df['points_scored'])
        metrics['scoring_efficiency'] = safe_divide(recent_avg, season_avg, 1.0)
    else:
        metrics['scoring_efficiency'] = 1.0
    
    # Defensive Stops (EMA útil para defensive trends)
    if 'points_allowed' in team_history_df.columns:
        recent_def = safe_mean(team_history_df['points_allowed'].tail(3))
        season_def = safe_mean(team_history_df['points_allowed'])
        metrics['defensive_stops'] = safe_divide(season_def, recent_def, 1.0)
    else:
        metrics['defensive_stops'] = 1.0
    
    # Clutch Performance (EMA útil)
    if len(team_history_df) >= 5:
        close_games = team_history_df[abs(team_history_df.get('plus_minus', 0)) <= 10]
        if len(close_games) > 0:
            metrics['clutch_performance'] = safe_mean(close_games.get('win', pd.Series([0.5])))
        else:
            metrics['clutch_performance'] = metrics['win']
    else:
        metrics['clutch_performance'] = 0.5
    
    # Clutch Efficiency (EMA útil)
    if 'total_score' in team_history_df.columns and 'points_scored' in team_history_df.columns:
        valid_games = team_history_df.dropna(subset=['total_score', 'points_scored'])
        if len(valid_games) >= 5:
            median_total = valid_games['total_score'].median()
            high_scoring_games = valid_games[valid_games['total_score'] > median_total]
            if len(high_scoring_games) >= 3:
                clutch_avg = high_scoring_games['points_scored'].mean()
                normal_avg = valid_games['points_scored'].mean()
                metrics['clutch_efficiency'] = clutch_avg / max(normal_avg, 1)
            else:
                metrics['clutch_efficiency'] = 1.0
        else:
            metrics['clutch_efficiency'] = 1.0
    else:
        metrics['clutch_efficiency'] = 1.0
    
    # Game Flow Indicator (EMA útil - USAR IMPLEMENTACIÓN EXISTENTE)
    # 🔧 USAR LA IMPLEMENTACIÓN COMPLETA QUE YA EXISTE
    has_turnovers = 'turnovers' in team_history_df.columns
    has_total_score = 'total_score' in team_history_df.columns
    
    if has_turnovers and has_total_score:
        # Método principal: turnovers + total_score
        recent_games = team_history_df.tail(5)
        if len(recent_games) >= 3:
            avg_turnovers = recent_games['turnovers'].mean()
            avg_total_score = recent_games['total_score'].mean()
            
            if avg_turnovers > 0:
                flow_score = safe_divide(avg_total_score, avg_turnovers * 8, 25.0)
                metrics['game_flow_indicator'] = min(flow_score / 25, 1.0)
            else:
                metrics['game_flow_indicator'] = 0.7  # 0 turnovers = buen flow
        else:
            metrics['game_flow_indicator'] = 0.5
    elif 'assists' in team_history_df.columns and has_total_score:
        # Fallback 1: assists + total_score
        recent_games = team_history_df.tail(5)
        if len(recent_games) >= 3:
            avg_assists = recent_games['assists'].mean()
            avg_total_score = recent_games['total_score'].mean()
            
            if avg_assists > 0:
                flow_score = (avg_assists / 25) * (avg_total_score / 200)
                metrics['game_flow_indicator'] = min(flow_score, 1.0)
            else:
                metrics['game_flow_indicator'] = 0.3
        else:
            metrics['game_flow_indicator'] = 0.5
    elif has_total_score:
        # Fallback 2: variabilidad de scoring como proxy
        recent_games = team_history_df.tail(5)
        if len(recent_games) >= 3:
            score_variance = recent_games['total_score'].std()
            score_mean = recent_games['total_score'].mean()
            
            if score_mean > 0:
                cv = score_variance / score_mean
                flow_proxy = 1 / (1 + cv * 2)
                metrics['game_flow_indicator'] = min(flow_proxy, 1.0)
            else:
                metrics['game_flow_indicator'] = 0.5
        else:
            metrics['game_flow_indicator'] = 0.5
    else:
        # Fallback 3: estrategia híbrida inteligente basada en contexto
        if 'points_scored' in team_history_df.columns:
            avg_points = team_history_df['points_scored'].tail(5).mean()
            
            if avg_points > 110:      # Alto scoring = buen flow
                base_flow = 0.7
            elif avg_points > 100:
                base_flow = 0.6
            elif avg_points > 90:
                base_flow = 0.5
            else:
                base_flow = 0.4      # Bajo scoring = flow pobre
            
            # Ajuste por tipo de liga detectado automáticamente
            if avg_points > 115:     # Estilo NBA
                league_adjustment = 0.61
            elif avg_points < 85:    # Estilo más defensivo/europeo
                league_adjustment = 0.47
            else:                    # Liga intermedia
                league_adjustment = 0.54
            
            # Promedio inteligente
            metrics['game_flow_indicator'] = (base_flow * 0.6) + (league_adjustment * 0.4)
        else:
            # Último recurso: default por tamaño de liga
            total_games = len(team_history_df) if not team_history_df.empty else 0
            
            if total_games > 50:     # Liga grande, probablemente profesional
                metrics['game_flow_indicator'] = 0.58
            elif total_games > 20:   # Liga mediana
                metrics['game_flow_indicator'] = 0.54
            else:                    # Liga pequeña o pocos datos
                metrics['game_flow_indicator'] = 0.52
    
    # Nuevas features que SÍ necesitan EMA
    metrics['clutch_time_performance'] = calculate_clutch_time_performance(team_history_df)
    metrics['turnover_momentum'] = calculate_turnover_momentum(team_history_df)
    
    # ==========================================
    # 🎯 FEATURES DIRECTAS (van directo, sin EMA)
    # ==========================================
    
    # Features básicas directas (redundantes optimizadas)
    metrics['win_rate'] = metrics['win']  # Redundante con win
    metrics['avg_plus_minus'] = metrics['plus_minus']  # Redundante con plus_minus
    
    # Features optimizadas directas (USAR IMPLEMENTACIONES EXISTENTES)
    
    # 1. Scoring Consistency (ya existe en el código)
    if 'points_scored' in team_history_df.columns:
        points = team_history_df['points_scored'].dropna()
        if len(points) >= 5:
            cv = points.std() / max(points.mean(), 1)
            metrics['scoring_consistency'] = 1 / (1 + cv)
        else:
            metrics['scoring_consistency'] = 0.5
    else:
        metrics['scoring_consistency'] = 0.5
    
    # 2. Pace Stability (ya existe en el código)
    if 'total_score' in team_history_df.columns:
        total_scores = team_history_df['total_score'].tail(8).dropna()
        if len(total_scores) >= 5:
            pace_cv = total_scores.std() / max(total_scores.mean(), 1)
            metrics['pace_stability'] = 1 / (1 + pace_cv)
        else:
            metrics['pace_stability'] = 0.5
    else:
        metrics['pace_stability'] = 0.5
    
    # 3. Offensive Efficiency Trend (ya existe en el código)
    if 'points_scored' in team_history_df.columns and len(team_history_df) >= 6:
        recent_scoring = team_history_df['points_scored'].tail(3).mean()
        historical_scoring = team_history_df['points_scored'].head(-3).mean()
        if historical_scoring > 0:
            metrics['offensive_efficiency_trend'] = recent_scoring / historical_scoring
        else:
            metrics['offensive_efficiency_trend'] = 1.0
    else:
        metrics['offensive_efficiency_trend'] = 1.0
    
    # 4. Quarter Scoring Pattern (ya existe en el código)
    quarter_columns = [f'q{i}_points' for i in range(1, 5)]
    available_quarters = [col for col in quarter_columns if col in team_history_df.columns]
    
    if len(available_quarters) >= 2:
        recent_games = team_history_df.tail(5)
        if len(recent_games) >= 3:
            # Prioridad: Q1 vs Q4 para mejor patrón
            if 'q1_points' in team_history_df.columns and 'q4_points' in team_history_df.columns:
                q1_avg = recent_games['q1_points'].mean()
                q4_avg = recent_games['q4_points'].mean()
                if q1_avg > 0:
                    quarter_pattern = safe_divide(q4_avg - q1_avg, q1_avg, 0.0)
                    metrics['quarter_scoring_pattern'] = max(-0.5, min(0.5, quarter_pattern))
                else:
                    metrics['quarter_scoring_pattern'] = 0.0
            else:
                metrics['quarter_scoring_pattern'] = 0.0
        else:
            metrics['quarter_scoring_pattern'] = 0.0
    else:
        metrics['quarter_scoring_pattern'] = 0.0
    
    # 5. Defensive Fatigue (ya existe en el código)
    if 'points_allowed' in team_history_df.columns and len(team_history_df) >= 8:
        recent_defense = team_history_df['points_allowed'].tail(3).mean()
        historical_defense = team_history_df['points_allowed'].head(-3).mean()
        if historical_defense > 0:
            metrics['defensive_fatigue'] = (recent_defense - historical_defense) / historical_defense
        else:
            metrics['defensive_fatigue'] = 0.0
    else:
        metrics['defensive_fatigue'] = 0.0
    
    # 6. Shooting Rhythm (versión mejorada)
    metrics['shooting_rhythm'] = calculate_enhanced_shooting_rhythm(team_history_df)
    
    # 7. Second Half Efficiency (ya existe en el código)
    metrics['second_half_efficiency'] = calculate_second_half_efficiency(team_history_df)
    
    # 8. Efficiency Differential (ya existe en el código)
    metrics['efficiency_differential'] = calculate_efficiency_differential(team_history_df)
    
    # Nuevas features directas
    metrics['back_to_back_fatigue'] = calculate_back_to_back_fatigue(team_history_df)
    metrics['defensive_intensity_drop'] = calculate_defensive_intensity_drop(team_history_df)
    metrics['rolling_volatility_3_games'] = calculate_rolling_volatility_3_games(team_history_df)
    metrics['momentum_acceleration'] = calculate_momentum_acceleration(team_history_df)
    metrics['pace_differential_trend'] = calculate_pace_differential_trend(team_history_df)
    
    # ✅ SOLUCIÓN: Se eliminan las 4 nuevas features de aquí. Se calcularán directamente en el bucle principal.
    
    return metrics
    
def calculate_performance_context(team_history_df, is_home_team=True):
    """Calcula métricas de contexto de performance."""
    if team_history_df.empty:
        return {'home_advantage_factor': 1.0, 'comeback_ability': 0.5, 'consistency_index': 0.5}
    
    metrics = {}
    
    # 🏠 Home Advantage Factor (solo para equipos locales)
    if is_home_team and len(team_history_df) >= 3:
        recent_performance = safe_mean(team_history_df.tail(5).get('plus_minus', pd.Series([0])))
        metrics['home_advantage_factor'] = 1.0 + (recent_performance / 100)  # Normalizado
    else:
        metrics['home_advantage_factor'] = 1.0
    
    # 🔄 Comeback Ability (capacidad de remontar)
    if 'plus_minus' in team_history_df.columns and len(team_history_df) >= 5:
        comeback_games = sum(1 for pm in team_history_df['plus_minus'] if pm > 5)
        total_games = len(team_history_df)
        metrics['comeback_ability'] = comeback_games / max(total_games, 1)
    else:
        metrics['comeback_ability'] = 0.5
    
    # 📈 Consistency Index (qué tan consistentes son los resultados)
    if 'total_score' in team_history_df.columns and len(team_history_df) >= 5:
        score_std = team_history_df['total_score'].std()
        score_mean = safe_mean(team_history_df['total_score'])
        cv = score_std / max(score_mean, 1)  # Coeficiente de variación
        metrics['consistency_index'] = 1 / (1 + cv)  # Invertido: mayor consistencia = menor variación
    else:
        metrics['consistency_index'] = 0.5
    
    return metrics

def get_rolling_stats(team_history, N, cols_to_average):
    """Calcula las estadísticas promedio simple para los últimos N partidos."""
    if len(team_history) < N:
        return {f'avg_{col}_last_{N}': np.nan for col in cols_to_average}
    
    last_n_games = team_history.tail(N)
    stats = {}
    for col in cols_to_average:
        if col in last_n_games.columns and not last_n_games[col].isnull().all():
            stats[f'avg_{col}_last_{N}'] = safe_mean(last_n_games[col])
        else:
            stats[f'avg_{col}_last_{N}'] = np.nan
    return stats

def get_enhanced_ema_stats(team_history, cols_to_average):
    """
    🚀 SISTEMA EMA PROFESIONAL - Múltiples rangos de tiempo
    Calcula EMA para diferentes horizontes temporales (corto, medio, largo plazo)
    """
    if team_history.empty:
        return {}
    
    ema_stats = {}
    
    for ema_name, ema_period in EMA_RANGES.items():
        if len(team_history) < ema_period:
            # Si no hay suficientes datos, usar los disponibles
            effective_period = min(len(team_history), ema_period)
        else:
            effective_period = ema_period
            
        if effective_period <= 0:
            continue
            
        for col in cols_to_average:
            if col in team_history.columns and not team_history[col].isnull().all():
                # Calcular EMA con ajuste dinámico
                ema_series = team_history[col].ewm(
                    span=effective_period, 
                    adjust=False,
                    min_periods=1
                ).mean()
                
                if not ema_series.empty:
                    ema_stats[f'ema_{col}_{ema_name}_{ema_period}'] = ema_series.iloc[-1]
                else:
                    ema_stats[f'ema_{col}_{ema_name}_{ema_period}'] = np.nan
            else:
                ema_stats[f'ema_{col}_{ema_name}_{ema_period}'] = np.nan
    
    return ema_stats

def get_ema_stats(team_history, N, cols_to_average):
    """Versión legacy para compatibilidad - usa EMA simple."""
    if len(team_history) < N:
        return {f'ema_{col}_last_{N}': np.nan for col in cols_to_average}
    
    stats = {}
    for col in cols_to_average:
        if col in team_history.columns and not team_history[col].isnull().all():
            stats[f'ema_{col}_last_{N}'] = team_history[col].ewm(span=N, adjust=False).mean().iloc[-1]
        else:
            stats[f'ema_{col}_last_{N}'] = np.nan
    return stats

def calculate_live_pace_metrics(q_scores, quarter_stage, team_trends=None):
    """
    Calcula métricas de pace y momentum en tiempo real con análisis mejorado.
    MODIFICADO: Ahora acepta tendencias de equipo para proyecciones mejoradas.
    """
    # Mapear minutos jugados según el estado real del cuarto
    if quarter_stage == 'halftime':
        minutes_played = 24
    elif quarter_stage == 'q3_progress':
        minutes_played = 30  # Q3 en progreso (~30 min jugados)
    else:
        minutes_played = 36  # Fin de Q3 u otros casos
    
    # Inferir cuartos jugados para lógica posterior (momentum/comebacks)
    quarters_played = 3 if (q_scores.get('q3_home', 0) > 0 or q_scores.get('q3_away', 0) > 0) else 2

    total_points = sum(q_scores.values())

    # Estimación de pace más precisa usando possessions reales
    # Calcular possessions usando stats disponibles si es posible
    team_stats = {
        'field_goals_attempted': max(80, total_points * 0.8),  # Estimación conservadora
        'offensive_rebounds': max(8, total_points * 0.05),
        'turnovers': max(12, total_points * 0.08),
        'free_throws_attempted': max(16, total_points * 0.12)
    }

    # Usar función existente para calcular possessions
    estimated_possessions = calculate_possessions(team_stats, {})

    # Validar rango razonable
    if estimated_possessions < 60:
        estimated_possessions = 60  # Mínimo razonable
    elif estimated_possessions > 140:
        estimated_possessions = 140  # Máximo razonable

    # Pace = possessions por 48 minutos (ya está normalizado)
    live_pace_estimate = estimated_possessions
    
    # Eficiencias relativas
    home_points = q_scores.get('q1_home', 0) + q_scores.get('q2_home', 0) + q_scores.get('q3_home', 0)
    away_points = q_scores.get('q1_away', 0) + q_scores.get('q2_away', 0) + q_scores.get('q3_away', 0)
    
    total_current = home_points + away_points
    
    live_efficiency_home = (home_points / total_current) if total_current > 0 else 0.5
    live_efficiency_away = (away_points / total_current) if total_current > 0 else 0.5
    
    # Momentum Shift (cambio de momentum entre cuartos)
    live_momentum_shift = 0
    if quarters_played >= 2:
        q1_diff = q_scores.get('q1_home', 0) - q_scores.get('q1_away', 0)
        q2_diff = q_scores.get('q2_home', 0) - q_scores.get('q2_away', 0)
        live_momentum_shift = abs(q2_diff - q1_diff)
        
        if quarters_played >= 3:
            q3_diff = q_scores.get('q3_home', 0) - q_scores.get('q3_away', 0)
            recent_shift = abs(q3_diff - q2_diff)
            live_momentum_shift = (live_momentum_shift + recent_shift) / 2
    
    # Quarter Consistency (qué tan consistentes son los cuartos)
    quarter_totals = []
    if q_scores.get('q1_home', 0) + q_scores.get('q1_away', 0) > 0:
        quarter_totals.append(q_scores['q1_home'] + q_scores['q1_away'])
    if q_scores.get('q2_home', 0) + q_scores.get('q2_away', 0) > 0:
        quarter_totals.append(q_scores['q2_home'] + q_scores['q2_away'])
    if q_scores.get('q3_home', 0) + q_scores.get('q3_away', 0) > 0:
        quarter_totals.append(q_scores['q3_home'] + q_scores['q3_away'])
    
    quarter_consistency = 1 / (1 + np.std(quarter_totals)) if len(quarter_totals) > 1 else 0.5
    
    # Comeback Indicator (indicador de remontada)
    comeback_indicator = 0
    if quarters_played >= 2:
        q1_leader_home = q_scores.get('q1_home', 0) > q_scores.get('q1_away', 0)
        current_leader_home = home_points > away_points
        comeback_indicator = 1 if q1_leader_home != current_leader_home else 0

    # 🆕 APLICAR MEJORAS CORE
    enhanced_pace = apply_enhanced_pace_projection(
        live_pace_estimate, 
        minutes_played, 
        team_trends
    )

    return {
        'live_pace_estimate': live_pace_estimate,           # Original
        'enhanced_pace_estimate': enhanced_pace,            # 🆕 MEJORADO
        'live_efficiency_home': live_efficiency_home,
        'live_efficiency_away': live_efficiency_away,
        'live_momentum_shift': live_momentum_shift,
        'quarter_consistency': quarter_consistency,
        'comeback_indicator': comeback_indicator
    }

# ===================================================================
# 🚀🚀🚀 NUEVAS FUNCIONES DE MEJORA DE PACE SYSTEM 🚀🚀🚀
# ===================================================================

def calculate_team_quarter_trends(team_history_df):
    """
    🎯 FUNCIÓN CORREGIDA CON DETECCIÓN INTELIGENTE DE COLUMNAS
    ✅ FIXED: Maneja correctamente múltiples formatos de columnas
    ✅ NUEVO: Sistema inteligente que detecta el formato disponible
    """
    if team_history_df is None or team_history_df.empty:
        return {'trend_factor': 1.0}  # Factor neutro

    # 🔧 SISTEMA INTELIGENTE DE DETECCIÓN DE COLUMNAS
    available_columns = list(team_history_df.columns)
    
    # Opción 1: Formato directo esperado
    direct_format_cols = ['q1_points', 'q2_points', 'q3_points', 'q4_points']
    
    # Opción 2: Formato con prefijo (home_q1_points, away_q1_points, etc.)
    prefix_formats = []
    for prefix in ['home_', 'away_', '']:
        prefix_cols = [f'{prefix}q{i}_points' for i in range(1, 5)]
        if all(col in available_columns for col in prefix_cols):
            prefix_formats.append(prefix_cols)
    
    # Seleccionar el mejor formato disponible
    if all(col in available_columns for col in direct_format_cols):
        q_cols = direct_format_cols
    elif prefix_formats:
        # Usar el primer formato válido encontrado
        q_cols = prefix_formats[0]
    else:
        # Búsqueda flexible de cualquier columna de cuarto
        quarter_info_cols = [col for col in available_columns 
                           if any(f'q{i}' in col.lower() and 'points' in col.lower() 
                                 for i in range(1, 5))]
        
        if len(quarter_info_cols) < 4:
            return {'trend_factor': 1.0}
        
        # Mapear a formato estándar
        q_cols = []
        for quarter_num in range(1, 5):
            quarter_col = None
            for col in quarter_info_cols:
                if f'q{quarter_num}' in col.lower() and 'points' in col.lower():
                    quarter_col = col
                    break
            
            if quarter_col:
                q_cols.append(quarter_col)
            else:
                return {'trend_factor': 1.0}
    
    # Validar que tenemos datos válidos
    valid_data_cols = []
    for col in q_cols:
        if col in team_history_df.columns:
            non_null_data = team_history_df[col].dropna()
            if len(non_null_data) > 0 and non_null_data.sum() > 0:
                valid_data_cols.append(col)

    if len(valid_data_cols) < 4:
        return {'trend_factor': 1.0}

    try:
        # Calcular promedios por cuarto
        quarter_averages = []
        for col in valid_data_cols:
            avg = safe_mean(team_history_df[col])
            if not pd.isna(avg) and avg >= 0:
                quarter_averages.append(avg)

        if len(quarter_averages) < 4:
            return {'trend_factor': 1.0}

        # Asumir que están en orden Q1, Q2, Q3, Q4
        q1_avg, q2_avg, q3_avg, q4_avg = quarter_averages[:4]

        # Calcular promedio de primera y segunda mitad
        first_half_avg = q1_avg + q2_avg
        second_half_avg = q3_avg + q4_avg

        # Evitar división por cero
        if first_half_avg <= 0:
            return {'trend_factor': 1.0}

        # Calcular factor de tendencia
        trend_factor = safe_divide(second_half_avg, first_half_avg, 1.0)
        
        # 🛡️ LIMITADORES DE SEGURIDAD: Limitar factor a rangos razonables (±15%)
        trend_factor = max(0.85, min(1.15, trend_factor))
        
        return {'trend_factor': trend_factor}
        
    except Exception as e:
        print(f"⚠️ Error calculando tendencias de cuarto: {e}")
        return {'trend_factor': 1.0}


def apply_enhanced_pace_projection(current_pace, minutes_played, team_trends):
    """
    🚀 PROYECTA EL PACE A 48 MINUTOS, AJUSTADO POR TENDENCIAS HISTÓRICAS.
    Aplica el factor de tendencia de los equipos para una proyección más inteligente.
    """
    # Si no hay tendencias, devolver el pace actual sin ajustar
    if team_trends is None or 'home' not in team_trends or 'away' not in team_trends:
        return current_pace

    # Extraer factores de tendencia (con valor por defecto de 1.0 si no existen)
    home_factor = team_trends.get('home', {}).get('trend_factor', 1.0)
    away_factor = team_trends.get('away', {}).get('trend_factor', 1.0)

    # Promediar el factor de tendencia de ambos equipos
    avg_trend_factor = (home_factor + away_factor) / 2.0

    # 🛡️ LIMITADORES DE SEGURIDAD: Limitar el ajuste a un máximo de ±15% para evitar sobreajustes extremos
    clamped_trend_factor = min(max(avg_trend_factor, 0.85), 1.15)
    
    # La proyección base ya está calculada en `current_pace` (a 48 min)
    # Se aplica el factor de ajuste a esta proyección
    adjusted_projection = safe_divide(current_pace * clamped_trend_factor, 1.0, current_pace)
    
    return adjusted_projection


def apply_blowout_adjustment(prediction, score_diff, time_remaining_pct):
    """
    🛡️ AJUSTE POR PALIZA (BLOWOUT) CONSERVADOR.
    - Limita el impacto máximo al 4% de la predicción.
    - Solo aplica en tramo final (<= 25% restante) con diferencia claramente grande (>20).
    """
    if score_diff > 20 and time_remaining_pct <= 0.25:
        # Crecer suavemente a partir de 20 pts de diferencia, cap 4%
        # Ej: 22 -> 1%, 24 -> 2%, 28+ -> 4% (cap)
        progressive = max(0.0, (score_diff - 20) / 200.0)
        reduction_factor = min(0.04, progressive)
        return prediction * (1 - reduction_factor)
    return prediction

# ===================================================================
# FUNCIONES PRINCIPALES MANTENIDAS DEL BACKUP FUNCIONAL
# ===================================================================

def calculate_advanced_stats_for_match(match_data, home_team_name, away_team_name, all_matches=None, league_averages=None):
    """Calcula Four Factors, Pace y eficiencias con sistema robusto de imputación."""
    if 'quarter_stats' not in match_data or not match_data['quarter_stats']:
        return None

    home_totals, away_totals = {}, {}
    
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    if all(q in match_data['quarter_stats'] for q in quarters):
        for q in quarters:
            q_stats = match_data['quarter_stats'][q]
            if home_team_name in q_stats and away_team_name in q_stats:
                home_norm = normalize_stats_keys(q_stats[home_team_name])
                away_norm = normalize_stats_keys(q_stats[away_team_name])
                for stat_key, value in home_norm.items():
                    home_totals[stat_key] = home_totals.get(stat_key, 0) + safe_int(value)
                for stat_key, value in away_norm.items():
                    away_totals[stat_key] = away_totals.get(stat_key, 0) + safe_int(value)
    
    elif 'MATCH' in match_data['quarter_stats']:
        match_stats = match_data['quarter_stats']['MATCH']
        if home_team_name in match_stats and away_team_name in match_stats:
            home_totals = normalize_stats_keys({k: safe_int(v) for k, v in match_stats[home_team_name].items()})
            away_totals = normalize_stats_keys({k: safe_int(v) for k, v in match_stats[away_team_name].items()})

    if not home_totals or not away_totals:
        return None

    if all_matches and league_averages is not None:
        # 🛡️ FIX: Pass match_date to prevent data leakage
        match_date = match.get('date')
        home_totals = impute_missing_stats(home_totals, home_team_name, all_matches, league_averages, match_date)
        away_totals = impute_missing_stats(away_totals, away_team_name, all_matches, league_averages, match_date)

    home_poss = calculate_possessions(home_totals, away_totals)
    away_poss = calculate_possessions(away_totals, home_totals)
    avg_poss = (home_poss + away_poss) / 2
    
    pace = calculate_pace(home_poss, away_poss)
    
    home_pts = safe_int(match_data.get('home_score') or match_data.get('final_score', {}).get('home'))
    away_pts = safe_int(match_data.get('away_score') or match_data.get('final_score', {}).get('away'))

    stats = {'home': {}, 'away': {}}
    
    # Métricas básicas
    stats['home']['possessions'] = avg_poss
    stats['away']['possessions'] = avg_poss
    stats['home']['pace'] = pace
    stats['away']['pace'] = pace
    
    stats['home']['ortg'] = (home_pts / avg_poss) * 100 if avg_poss > 0 else 0
    stats['away']['ortg'] = (away_pts / avg_poss) * 100 if avg_poss > 0 else 0
    stats['home']['drtg'] = stats['away']['ortg']
    stats['away']['drtg'] = stats['home']['ortg']

    # Four Factors
    home_fg_made = home_totals.get('field_goals_made', 0)
    home_3pt_made = home_totals.get('3point_field_goals_made', 0)
    home_fga = max(home_totals.get('field_goals_attempted', 1), 1)
    
    away_fg_made = away_totals.get('field_goals_made', 0)
    away_3pt_made = away_totals.get('3point_field_goals_made', 0)
    away_fga = max(away_totals.get('field_goals_attempted', 1), 1)
    
    stats['home']['efg_percentage'] = (home_fg_made + 0.5 * home_3pt_made) / home_fga
    stats['away']['efg_percentage'] = (away_fg_made + 0.5 * away_3pt_made) / away_fga

    stats['home']['tov_percentage'] = (home_totals.get('turnovers', 0) / home_poss) * 100 if home_poss > 0 else 0
    stats['away']['tov_percentage'] = (away_totals.get('turnovers', 0) / away_poss) * 100 if away_poss > 0 else 0

    home_oreb = home_totals.get('offensive_rebounds', 0)
    away_dreb = away_totals.get('defensive_rebounds', 0)
    home_oreb_chances = home_oreb + away_dreb
    stats['home']['oreb_percentage'] = (home_oreb / home_oreb_chances) * 100 if home_oreb_chances > 0 else 0
    
    away_oreb = away_totals.get('offensive_rebounds', 0)
    home_dreb = home_totals.get('defensive_rebounds', 0)
    away_oreb_chances = away_oreb + home_dreb
    stats['away']['oreb_percentage'] = (away_oreb / away_oreb_chances) * 100 if away_oreb_chances > 0 else 0

    stats['home']['ft_rate'] = home_totals.get('free_throws_attempted', 0) / home_fga
    stats['away']['ft_rate'] = away_totals.get('free_throws_attempted', 0) / away_fga

    # Eficiencias mejoradas
    stats['home']['offensive_efficiency'] = (
        home_fg_made * 2 + home_3pt_made + home_totals.get('free_throws_made', 0)
    ) / max(home_fga, 1)
    
    stats['away']['offensive_efficiency'] = (
        away_fg_made * 2 + away_3pt_made + away_totals.get('free_throws_made', 0)
    ) / max(away_fga, 1)
    
    stats['home']['defensive_efficiency'] = (
        home_totals.get('defensive_rebounds', 0) + home_totals.get('blocks', 0)
    ) / max(away_fga, 1)
    
    stats['away']['defensive_efficiency'] = (
        away_totals.get('defensive_rebounds', 0) + away_totals.get('blocks', 0)
    ) / max(home_fga, 1)

    flat_stats = {}
    for team_type, team_stats in stats.items():
        for stat_key, value in team_stats.items():
            flat_stats[f'{team_type}_{stat_key}'] = value
    
    return flat_stats

def calculate_real_balance_features(q_scores, quarter_stage, team_names=None):
    """
    🎯 CALCULA BALANCE FEATURES REALES (versión con is_potential_blowout)
    """
    home_points = sum(q_scores.get(f'q{i}_home', 0) for i in range(1, 4))
    away_points = sum(q_scores.get(f'q{i}_away', 0) for i in range(1, 4))
    current_lead = abs(home_points - away_points)
    
    quarters_played = 2 if quarter_stage == 'halftime' else 3
    
    balance_score = 1 / (1 + (current_lead / (10 * quarters_played)))
    
    # Mantenemos 'is_unbalanced' para 'expected_q4_drop' pero ya no se usará como feature principal
    lead_threshold = 12 if quarter_stage == 'halftime' else 18
    is_unbalanced_flag = 1 if current_lead > lead_threshold else 0
    
    quarter_totals = [
        q_scores.get('q1_home', 0) + q_scores.get('q1_away', 0),
        q_scores.get('q2_home', 0) + q_scores.get('q2_away', 0)
    ]
    if quarter_stage == 'q3_end':
        quarter_totals.append(q_scores.get('q3_home', 0) + q_scores.get('q3_away', 0))
    
    quarter_totals = [q for q in quarter_totals if q > 0]
    
    mean_q_total = np.mean(quarter_totals) if quarter_totals else 0
    std_q_total = np.std(quarter_totals) if quarter_totals else 0
    
    if mean_q_total > 0:
        consistency = 1 / (1 + std_q_total / mean_q_total)
    else:
        consistency = 1.0 
    
    intensity_drop_factor = consistency
    
    expected_q4_drop = (current_lead / 100) * is_unbalanced_flag

    q1_leader_home = q_scores.get('q1_home', 0) > q_scores.get('q1_away', 0)
    current_leader_home = home_points > away_points
    
    lead_stability = 1.0
    if q1_leader_home != current_leader_home:
        lead_stability = 0.0
    elif quarter_stage == 'q3_end':
        q2_total_home = q_scores.get('q1_home', 0) + q_scores.get('q2_home', 0)
        q2_total_away = q_scores.get('q1_away', 0) + q_scores.get('q2_away', 0)
        q2_leader_home = q2_total_home > q2_total_away
        if q2_leader_home != current_leader_home:
            lead_stability = 0.5

    # ✅ NUEVA LÓGICA: Feature binaria clara y directa
    is_potential_blowout = 0
    if quarter_stage == 'q3_end':
        q3_end_diff = abs(home_points - away_points)
        if q3_end_diff > 15:  # Si la diferencia al final del Q3 es > 15
            is_potential_blowout = 1

    return {
        'game_balance_score': balance_score,
        'current_lead': current_lead,
        'is_game_unbalanced': is_unbalanced_flag,
        'is_potential_blowout': is_potential_blowout,
        'intensity_drop_factor': intensity_drop_factor,
        'expected_q4_drop': expected_q4_drop,
        'lead_stability': lead_stability
    }

def calculate_garbage_time_risk(q_scores, quarter_stage, lead_stability, quarter_variance=None, is_potential_blowout=0):
    """
    Señal continua [0,1] de riesgo de garbage time según ventaja, estabilidad, variabilidad por cuartos y blowout.
    quarter_stage: 'halftime' | 'q3_progress' | 'q3_end'
    """
    import math
    try:
        # 1) Puntos acumulados y tiempo restante aproximado
        if quarter_stage == 'halftime':
            home = q_scores.get('q1_home', 0) + q_scores.get('q2_home', 0)
            away = q_scores.get('q1_away', 0) + q_scores.get('q2_away', 0)
            t = 0.5
            L0, Ls = 12.0, 4.0  # thresholds más conservadores en HT
        elif quarter_stage == 'q3_progress':
            home = q_scores.get('q1_home', 0) + q_scores.get('q2_home', 0) + q_scores.get('q3_home', 0)
            away = q_scores.get('q1_away', 0) + q_scores.get('q2_away', 0) + q_scores.get('q3_away', 0)
            t = 1.0/3.0  # ~30 min jugados (entre HT y fin Q3)
            L0, Ls = 14.0, 4.5  # umbrales intermedios
        else:
            home = q_scores.get('q1_home', 0) + q_scores.get('q2_home', 0) + q_scores.get('q3_home', 0)
            away = q_scores.get('q1_away', 0) + q_scores.get('q2_away', 0) + q_scores.get('q3_away', 0)
            t = 0.25
            L0, Ls = 15.0, 5.0  # tramo final (fin Q3 en adelante)

        # 2) Componente del lead con logística y ponderación temporal
        L = abs(float(home) - float(away))
        lead_risk = 1.0 / (1.0 + math.exp(-(L - L0) / max(Ls, 1e-6)))

        t_clamped = max(0.0, min(1.0, t))
        time_scale = max(0.0, min(1.0, 1.0 - math.sqrt(t_clamped)))
        lead_component = lead_risk * time_scale

        # 3) Estabilidad del liderazgo
        S = 0.5
        try:
            if lead_stability is not None and not pd.isna(lead_stability):
                S = float(lead_stability)
        except Exception:
            pass
        S = max(0.0, min(1.0, S))

        # 4) Variabilidad entre cuartos via CV normalizado (robusto a escala)
        q_totals = []
        for i in [1, 2, 3]:
            h = q_scores.get(f'q{i}_home', 0)
            a = q_scores.get(f'q{i}_away', 0)
            total = (h or 0) + (a or 0)
            if total > 0:
                if quarter_stage == 'halftime' and i > 2:
                    continue
                q_totals.append(total)

        if len(q_totals) >= 2:
            mean_q = float(np.mean(q_totals))
            std_q = float(np.std(q_totals))
            cv_q = std_q / max(mean_q, 1.0)
            cv0 = 0.15  # umbral típico; tunear por liga si es necesario
            Vp = 1.0 - min(1.0, cv_q / max(cv0, 1e-6))
        else:
            Vp = 0.5

        # 5) Bonus por blowout binario
        B = 1.0 if int(is_potential_blowout or 0) == 1 else 0.0

        # 6) Combinar con pesos y clamp
        w1, w2, w3, w4 = 0.45, 0.25, 0.20, 0.10
        risk = w1 * lead_component + w2 * S + w3 * Vp + w4 * B
        risk = max(0.0, min(1.0, float(risk)))
        return risk
    except Exception:
        return np.nan

def calculate_h2h_features(home_team, away_team, full_df):
    """
    Calcula métricas basadas en los enfrentamientos directos (H2H)
    entre dos equipos. Requiere un mínimo de 5 partidos históricos.
    ROBUSTO: No asume la existencia de columnas por cuarto; usa fallbacks seguros.
    """
    h2h_feature_names = [
        'h2h_avg_total_score', 'h2h_pace_avg', 'h2h_first_half_avg_total',
        'h2h_second_half_surge_avg', 'h2h_comeback_freq'
    ]
    default_h2h_stats = {key: np.nan for key in h2h_feature_names}

    try:
        if full_df is None or len(full_df) == 0:
            return default_h2h_stats

        # Filtrar historial H2H
        h2h_history = full_df[
            ((full_df['home_team'] == home_team) & (full_df['away_team'] == away_team)) |
            ((full_df['home_team'] == away_team) & (full_df['away_team'] == home_team))
        ].copy()

        if len(h2h_history) < 5:
            return default_h2h_stats

        # Helper: columna segura (Serie de ceros si falta)
        def _col(df, name):
            if name in df.columns:
                s = pd.to_numeric(df[name], errors='coerce').fillna(0)
                s.index = df.index
                return s
            return pd.Series(0, index=df.index, dtype=float)

        # 1) Promedio de puntos totales
        avg_total_score = float(pd.to_numeric(h2h_history.get('total_score', pd.Series(np.nan)), errors='coerce').mean())

        # 2) Pace promedio (usa possessions si existen; si no, aproxima)
        h2h_paces = []
        for _, game in h2h_history.iterrows():
            home_poss = game.get('home_possessions', np.nan)
            away_poss = game.get('away_possessions', np.nan)
            if pd.isna(home_poss) or pd.isna(away_poss):
                # Fallback: aproximar con total_score si existe
                total = game.get('total_score', np.nan)
                if not pd.isna(total) and total > 0:
                    # Conversión aproximada usada en otras partes del sistema
                    est_pace = total / 2.1
                    h2h_paces.append(est_pace)
                else:
                    h2h_paces.append(95.0)  # default razonable
            else:
                h2h_paces.append((float(home_poss) + float(away_poss)) / 2.0)
        avg_pace = float(safe_mean(pd.Series(h2h_paces))) if h2h_paces else np.nan

        # 3) Promedio de puntos en primera mitad (columas seguras)
        hq1 = _col(h2h_history, 'home_q1_points')
        aq1 = _col(h2h_history, 'away_q1_points')
        hq2 = _col(h2h_history, 'home_q2_points')
        aq2 = _col(h2h_history, 'away_q2_points')
        first_half_totals = hq1 + aq1 + hq2 + aq2
        avg_first_half_total = float(first_half_totals.mean()) if len(first_half_totals) else np.nan

        # 4) Surge en segunda mitad
        hq3 = _col(h2h_history, 'home_q3_points')
        aq3 = _col(h2h_history, 'away_q3_points')
        hq4 = _col(h2h_history, 'home_q4_points')
        aq4 = _col(h2h_history, 'away_q4_points')
        second_half_totals = hq3 + aq3 + hq4 + aq4
        # Si first_half_totals son todos ceros y no hay columnas, resultará 0; mantén coherencia
        surge = float((second_half_totals - first_half_totals).mean()) if len(second_half_totals) else np.nan

        # 5) Frecuencia de remontadas (robusta a columnas faltantes)
        comebacks = 0
        total_games = len(h2h_history)
        for _, game in h2h_history.iterrows():
            fh_home_score = float(game.get('home_q1_points', 0) or 0) + float(game.get('home_q2_points', 0) or 0)
            fh_away_score = float(game.get('away_q1_points', 0) or 0) + float(game.get('away_q2_points', 0) or 0)

            # Si no hay datos de mitades, no considerar como remontada
            if fh_home_score == 0 and fh_away_score == 0:
                continue

            if fh_home_score == fh_away_score:
                continue  # empates al descanso no cuentan

            leader_at_halftime = 'home' if fh_home_score > fh_away_score else 'away'
            home_final = float(game.get('home_score', 0) or 0)
            away_final = float(game.get('away_score', 0) or 0)
            # Si tampoco hay score final, saltar
            if home_final == 0 and away_final == 0:
                continue
            winner_of_game = 'home' if home_final > away_final else 'away'

            if leader_at_halftime != winner_of_game:
                comebacks += 1

        comeback_freq = (comebacks / total_games) if total_games > 0 else 0.0

        return {
            'h2h_avg_total_score': avg_total_score,
            'h2h_pace_avg': avg_pace,
            'h2h_first_half_avg_total': avg_first_half_total,
            'h2h_second_half_surge_avg': surge,
            'h2h_comeback_freq': comeback_freq
        }
    except Exception:
        # Fallback absoluto
        return default_h2h_stats

def calculate_features(league_data_raw):
    """
    🎯 FUNCIÓN PRINCIPAL MEJORADA - FASE 1A CORE
    ✅ ACTIVADO: FILTERED_FEATURES_TO_USE
    ✅ NUEVO: 6 estadísticas basketball integradas al historial
    ✅ FIXED: Sistema inteligente de detección de columnas
    ✅ OPTIMIZADO: Features Over/Under integradas directamente
    ✅ SOLUCIÓN PACE/POSSESSIONS IMPLEMENTADA
    """
    processed_matches, league_averages = process_raw_matches(league_data_raw)
    
    if not processed_matches:
        return pd.DataFrame()

    df = pd.DataFrame(processed_matches).sort_values(by='date').reset_index(drop=True)
    print(f"📊 Partidos procesados: {len(df)}")

    all_features = []
    # MODIFICADO: Añadir possessions, pace, ortg, true_shooting_pct a las columnas a promediar
    cols_to_avg = ['points_scored', 'points_allowed', 'total_score', 'possessions', 'pace', 'ortg', 'true_shooting_pct'] + ADVANCED_STATS_COLS
    cols_to_ema = MOMENTUM_STATS_COLS

    print("🚀 Calculando features con sistema EMA avanzado + optimizaciones Over/Under...")
    
    for index, match in df.iterrows():
        if index % 50 == 0:
            print(f"   Procesando partido {index + 1}/{len(df)}...")
            
        home_team, away_team = match['home_team'], match['away_team']
        
        if index < 5:
            continue
            
        home_history = df.iloc[:index][(df.iloc[:index]['home_team'] == home_team) | (df.iloc[:index]['away_team'] == home_team)]
        away_history = df.iloc[:index][(df.iloc[:index]['home_team'] == away_team) | (df.iloc[:index]['away_team'] == away_team)]
        
        if len(home_history) < 2 or len(away_history) < 2:
            continue
        
        # 🏠 CREAR DATAFRAME DE HISTORIAL PARA EQUIPO LOCAL - ✅ MEJORADO CON 8 ESTADÍSTICAS
        home_history_list = []
        for _, r in home_history.iterrows():
            is_home = r['home_team'] == home_team
            team_hist_data = {
                'date': r.get('date'), # ✅ SOLUCIÓN: Añadir fecha para análisis de calendario
                'opponent_team': (r.get('away_team') if is_home else r.get('home_team')),
                'points_scored': r['home_score'] if is_home else r['away_score'],
                'points_allowed': r['away_score'] if is_home else r['home_score'],
                'total_score': r['total_score'],
                # 🆕 NUEVAS ESTADÍSTICAS (AHORA 8)
                'turnovers': r.get('home_turnovers' if is_home else 'away_turnovers', 0),
                'assists': r.get('home_assists' if is_home else 'away_assists', 0),
                'field_goals_made': r.get('home_field_goals_made' if is_home else 'away_field_goals_made', 0),
                'field_goals_attempted': r.get('home_field_goals_attempted' if is_home else 'away_field_goals_attempted', 0),
                'steals': r.get('home_steals' if is_home else 'away_steals', 0),
                'personal_fouls': r.get('home_personal_fouls' if is_home else 'away_personal_fouls', 0),
                'offensive_rebounds': r.get('home_offensive_rebounds' if is_home else 'away_offensive_rebounds', 0), # REQUERIDO PARA POSSESSIONS
                'free_throws_attempted': r.get('home_free_throws_attempted' if is_home else 'away_free_throws_attempted', 0), # REQUERIDO PARA POSSESSIONS
                # ✅ SOLUCIÓN DEFINITIVA: Mapear nombres canónicos con fallback a claves antiguas
                '2point_field_goals_made': r.get(('home_2point_field_goals_made' if is_home else 'away_2point_field_goals_made'),
                                                 r.get('home_2_point_field_goals_made' if is_home else 'away_2_point_field_goals_made', 0)),
                '2point_field_goals_attempted': r.get(('home_2point_field_goals_attempted' if is_home else 'away_2point_field_goals_attempted'),
                                                      r.get('home_2_point_field_g_attempted' if is_home else 'away_2_point_field_g_attempted', 0)),
                '3point_field_goals_made': r.get(('home_3point_field_goals_made' if is_home else 'away_3point_field_goals_made'),
                                                 r.get('home_3_point_field_goals_made' if is_home else 'away_3_point_field_goals_made', 0)),
                '3point_field_goals_attempted': r.get(('home_3point_field_goals_attempted' if is_home else 'away_3point_field_goals_attempted'),
                                                      r.get('home_3_point_field_g_attempted' if is_home else 'away_3_point_field_g_attempted', 0)),
                # Columnas de cuarto existentes
                'q1_points': r.get('home_q1_points' if is_home else 'away_q1_points', 0),
                'q2_points': r.get('home_q2_points' if is_home else 'away_q2_points', 0),
                'q3_points': r.get('home_q3_points' if is_home else 'away_q3_points', 0),
                'q4_points': r.get('home_q4_points' if is_home else 'away_q4_points', 0),

                # 🆕 Quarter box (FGA, OREB, FTA, TOV) si están disponibles (para posesiones por cuarto)
                'q1_fga': r.get('home_q1_field_goals_attempted' if is_home else 'away_q1_field_goals_attempted', np.nan),
                'q2_fga': r.get('home_q2_field_goals_attempted' if is_home else 'away_q2_field_goals_attempted', np.nan),
                'q3_fga': r.get('home_q3_field_goals_attempted' if is_home else 'away_q3_field_goals_attempted', np.nan),
                'q4_fga': r.get('home_q4_field_goals_attempted' if is_home else 'away_q4_field_goals_attempted', np.nan),

                'q1_oreb': r.get('home_q1_offensive_rebounds' if is_home else 'away_q1_offensive_rebounds', np.nan),
                'q2_oreb': r.get('home_q2_offensive_rebounds' if is_home else 'away_q2_offensive_rebounds', np.nan),
                'q3_oreb': r.get('home_q3_offensive_rebounds' if is_home else 'away_q3_offensive_rebounds', np.nan),
                'q4_oreb': r.get('home_q4_offensive_rebounds' if is_home else 'away_q4_offensive_rebounds', np.nan),

                'q1_fta': r.get('home_q1_free_throws_attempted' if is_home else 'away_q1_free_throws_attempted', np.nan),
                'q2_fta': r.get('home_q2_free_throws_attempted' if is_home else 'away_q2_free_throws_attempted', np.nan),
                'q3_fta': r.get('home_q3_free_throws_attempted' if is_home else 'away_q3_free_throws_attempted', np.nan),
                'q4_fta': r.get('home_q4_free_throws_attempted' if is_home else 'away_q4_free_throws_attempted', np.nan),

                'q1_tov': r.get('home_q1_turnovers' if is_home else 'away_q1_turnovers', np.nan),
                'q2_tov': r.get('home_q2_turnovers' if is_home else 'away_q2_turnovers', np.nan),
                'q3_tov': r.get('home_q3_turnovers' if is_home else 'away_q3_turnovers', np.nan),
                'q4_tov': r.get('home_q4_turnovers' if is_home else 'away_q4_turnovers', np.nan),

                **{stat: r.get(f'{"home" if is_home else "away"}_{stat}', np.nan) for stat in ADVANCED_STATS_COLS}
            }
            home_history_list.append(team_hist_data)
        home_history_df = pd.DataFrame(home_history_list)
        
        # 🚗 CREAR DATAFRAME DE HISTORIAL PARA EQUIPO VISITANTE - ✅ MEJORADO CON 8 ESTADÍSTICAS
        away_history_list = []
        for _, r in away_history.iterrows():
            is_home = r['home_team'] == away_team
            team_hist_data = {
                'date': r.get('date'), # ✅ SOLUCIÓN: Añadir fecha para análisis de calendario
                'opponent_team': (r.get('home_team') if not is_home else r.get('away_team')),
                'points_scored': r['away_score'] if not is_home else r['home_score'],
                'points_allowed': r['home_score'] if not is_home else r['away_score'],
                'total_score': r['total_score'],
                # 🆕 NUEVAS ESTADÍSTICAS (AHORA 8)
                'turnovers': r.get('away_turnovers' if not is_home else 'home_turnovers', 0),
                'assists': r.get('away_assists' if not is_home else 'home_assists', 0),
                'field_goals_made': r.get('away_field_goals_made' if not is_home else 'home_field_goals_made', 0),
                'field_goals_attempted': r.get('away_field_goals_attempted' if not is_home else 'home_field_goals_attempted', 0),
                'steals': r.get('away_steals' if not is_home else 'home_steals', 0),
                'personal_fouls': r.get('away_personal_fouls' if not is_home else 'home_personal_fouls', 0),
                'offensive_rebounds': r.get('away_offensive_rebounds' if not is_home else 'home_offensive_rebounds', 0), # REQUERIDO PARA POSSESSIONS
                'free_throws_attempted': r.get('away_free_throws_attempted' if not is_home else 'home_free_throws_attempted', 0), # REQUERIDO PARA POSSESSIONS
                # ✅ SOLUCIÓN DEFINITIVA: Mapear nombres canónicos con fallback a claves antiguas
                '2point_field_goals_made': r.get(('away_2point_field_goals_made' if not is_home else 'home_2point_field_goals_made'),
                                                 r.get('away_2_point_field_goals_made' if not is_home else 'home_2_point_field_goals_made', 0)),
                '2point_field_goals_attempted': r.get(('away_2point_field_goals_attempted' if not is_home else 'home_2point_field_goals_attempted'),
                                                      r.get('away_2_point_field_g_attempted' if not is_home else 'home_2_point_field_g_attempted', 0)),
                '3point_field_goals_made': r.get(('away_3point_field_goals_made' if not is_home else 'home_3point_field_goals_made'),
                                                 r.get('away_3_point_field_goals_made' if not is_home else 'home_3_point_field_goals_made', 0)),
                '3point_field_goals_attempted': r.get(('away_3point_field_goals_attempted' if not is_home else 'home_3point_field_goals_attempted'),
                                                      r.get('away_3_point_field_g_attempted' if not is_home else 'home_3_point_field_g_attempted', 0)),
                # Columnas de cuarto
                'q1_points': r.get('away_q1_points' if not is_home else 'home_q1_points', 0),
                'q2_points': r.get('away_q2_points' if not is_home else 'home_q2_points', 0),
                'q3_points': r.get('away_q3_points' if not is_home else 'home_q3_points', 0),
                'q4_points': r.get('away_q4_points' if not is_home else 'home_q4_points', 0),

                # 🆕 Quarter box (FGA, OREB, FTA, TOV) si están disponibles (para posesiones por cuarto)
                'q1_fga': r.get('away_q1_field_goals_attempted' if not is_home else 'home_q1_field_goals_attempted', np.nan),
                'q2_fga': r.get('away_q2_field_goals_attempted' if not is_home else 'home_q2_field_goals_attempted', np.nan),
                'q3_fga': r.get('away_q3_field_goals_attempted' if not is_home else 'home_q3_field_goals_attempted', np.nan),
                'q4_fga': r.get('away_q4_field_goals_attempted' if not is_home else 'home_q4_field_goals_attempted', np.nan),

                'q1_oreb': r.get('away_q1_offensive_rebounds' if not is_home else 'home_q1_offensive_rebounds', np.nan),
                'q2_oreb': r.get('away_q2_offensive_rebounds' if not is_home else 'home_q2_offensive_rebounds', np.nan),
                'q3_oreb': r.get('away_q3_offensive_rebounds' if not is_home else 'home_q3_offensive_rebounds', np.nan),
                'q4_oreb': r.get('away_q4_offensive_rebounds' if not is_home else 'home_q4_offensive_rebounds', np.nan),

                'q1_fta': r.get('away_q1_free_throws_attempted' if not is_home else 'home_q1_free_throws_attempted', np.nan),
                'q2_fta': r.get('away_q2_free_throws_attempted' if not is_home else 'home_q2_free_throws_attempted', np.nan),
                'q3_fta': r.get('away_q3_free_throws_attempted' if not is_home else 'home_q3_free_throws_attempted', np.nan),
                'q4_fta': r.get('away_q4_free_throws_attempted' if not is_home else 'home_q4_free_throws_attempted', np.nan),

                'q1_tov': r.get('away_q1_turnovers' if not is_home else 'home_q1_turnovers', np.nan),
                'q2_tov': r.get('away_q2_turnovers' if not is_home else 'home_q2_turnovers', np.nan),
                'q3_tov': r.get('away_q3_turnovers' if not is_home else 'home_q3_turnovers', np.nan),
                'q4_tov': r.get('away_q4_turnovers' if not is_home else 'home_q4_turnovers', np.nan),

                **{stat: r.get(f'{"away" if not is_home else "home"}_{stat}', np.nan) for stat in ADVANCED_STATS_COLS}
            }
            away_history_list.append(team_hist_data)
        away_history_df = pd.DataFrame(away_history_list)

        # ============================================
        # 🔧 SOLUCIÓN: AGREGAR ESTADÍSTICAS AVANZADAS AL HISTORIAL
        # Este código calcula possessions, pace y ortg para cada partido del historial
        # ============================================
        
        # PARTE 1: Calcular para el equipo LOCAL (home)
        if not home_history_df.empty:
            required_cols = ['field_goals_attempted', 'offensive_rebounds', 'turnovers', 'free_throws_attempted']
            if all(col in home_history_df.columns for col in required_cols):
                home_history_df['possessions'] = (
                    home_history_df['field_goals_attempted'] - 
                    home_history_df['offensive_rebounds'] + 
                    home_history_df['turnovers'] + 
                    (0.44 * home_history_df['free_throws_attempted'])
                )
                home_history_df['pace'] = home_history_df['possessions'] * 2
                if 'points_scored' in home_history_df.columns:
                    mask = home_history_df['possessions'] > 0
                    home_history_df.loc[mask, 'ortg'] = (home_history_df.loc[mask, 'points_scored'] / home_history_df.loc[mask, 'possessions']) * 100
                    home_history_df.loc[~mask, 'ortg'] = np.nan
                home_history_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                home_history_df.loc[home_history_df['possessions'] < 0, 'possessions'] = np.nan
                home_history_df.loc[home_history_df['pace'] < 0, 'pace'] = np.nan
            
            # AGREGAR TRUE SHOOTING %
            if all(col in home_history_df.columns for col in ['points_scored', 'field_goals_attempted', 'free_throws_attempted']):
                # Calcular True Shooting % - métrica de eficiencia de tiro
                fga = home_history_df['field_goals_attempted']
                fta = home_history_df['free_throws_attempted']
                pts = home_history_df['points_scored']
                        
                # Evitar división por cero
                tsa = 2 * (fga + 0.44 * fta)  # True Shooting Attempts
                mask = tsa > 0
                        
                home_history_df.loc[mask, 'true_shooting_pct'] = pts[mask] / tsa[mask]
                home_history_df.loc[~mask, 'true_shooting_pct'] = np.nan
                        
                # Limpiar valores no razonables (TS% típico está entre 0.4 y 0.7)
                home_history_df.loc[home_history_df['true_shooting_pct'] > 1, 'true_shooting_pct'] = np.nan
                home_history_df.loc[home_history_df['true_shooting_pct'] < 0, 'true_shooting_pct'] = np.nan
        
        # PARTE 2: Calcular para el equipo VISITANTE (away)
        if not away_history_df.empty:
            required_cols = ['field_goals_attempted', 'offensive_rebounds', 'turnovers', 'free_throws_attempted']
            if all(col in away_history_df.columns for col in required_cols):
                away_history_df['possessions'] = (
                    away_history_df['field_goals_attempted'] - 
                    away_history_df['offensive_rebounds'] + 
                    away_history_df['turnovers'] + 
                    (0.44 * away_history_df['free_throws_attempted'])
                )
                away_history_df['pace'] = away_history_df['possessions'] * 2
                if 'points_scored' in away_history_df.columns:
                    mask = away_history_df['possessions'] > 0
                    away_history_df.loc[mask, 'ortg'] = (away_history_df.loc[mask, 'points_scored'] / away_history_df.loc[mask, 'possessions']) * 100
                    away_history_df.loc[~mask, 'ortg'] = np.nan
                away_history_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                away_history_df.loc[away_history_df['possessions'] < 0, 'possessions'] = np.nan
                away_history_df.loc[away_history_df['pace'] < 0, 'pace'] = np.nan
                
            # AGREGAR TRUE SHOOTING %
            if all(col in away_history_df.columns for col in ['points_scored', 'field_goals_attempted', 'free_throws_attempted']):
                fga = away_history_df['field_goals_attempted']
                fta = away_history_df['free_throws_attempted']
                pts = away_history_df['points_scored']
                        
                tsa = 2 * (fga + 0.44 * fta)
                mask = tsa > 0
                        
                away_history_df.loc[mask, 'true_shooting_pct'] = pts[mask] / tsa[mask]
                away_history_df.loc[~mask, 'true_shooting_pct'] = np.nan
                        
                away_history_df.loc[away_history_df['true_shooting_pct'] > 1, 'true_shooting_pct'] = np.nan
                away_history_df.loc[away_history_df['true_shooting_pct'] < 0, 'true_shooting_pct'] = np.nan
        
        # ============================================
        # FIN DE LA SOLUCIÓN
        # ============================================
        
        # Añadir métricas calculadas (usando la función ÚNICA mejorada)
        if not home_history_df.empty:
            home_history_df['win'] = (home_history_df['points_scored'] > home_history_df['points_allowed']).astype(int)
            home_history_df['plus_minus'] = home_history_df['points_scored'] - home_history_df['points_allowed']
            # ✅ USAR FUNCIÓN ÚNICA MEJORADA CON FEATURES OPTIMIZADAS INTEGRADAS
            momentum_metrics = calculate_momentum_metrics(home_history_df)
            for metric, value in momentum_metrics.items():
                home_history_df[metric] = value
            context_metrics = calculate_performance_context(home_history_df, is_home_team=True)
            for metric, value in context_metrics.items():
                home_history_df[metric] = value

        if not away_history_df.empty:
            away_history_df['win'] = (away_history_df['points_scored'] > away_history_df['points_allowed']).astype(int)
            away_history_df['plus_minus'] = away_history_df['points_scored'] - away_history_df['points_allowed']
            # ✅ USAR FUNCIÓN ÚNICA MEJORADA CON FEATURES OPTIMIZADAS INTEGRADAS
            momentum_metrics = calculate_momentum_metrics(away_history_df)
            for metric, value in momentum_metrics.items():
                away_history_df[metric] = value
            context_metrics = calculate_performance_context(away_history_df, is_home_team=False)
            for metric, value in context_metrics.items():
                away_history_df[metric] = value

        # Calcular estadísticas rolling y EMA
        home_stats_5 = get_rolling_stats(home_history_df, 5, cols_to_avg)
        away_stats_5 = get_rolling_stats(away_history_df, 5, cols_to_avg)
        
        home_ema_advanced = get_enhanced_ema_stats(home_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)
        away_ema_advanced = get_enhanced_ema_stats(away_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)
        
        home_ema_5 = get_ema_stats(home_history_df, 5, cols_to_ema)
        away_ema_5 = get_ema_stats(away_history_df, 5, cols_to_ema)
        
        # Crear features pre-juego
        pre_game_features = {}
        pre_game_features.update({'home_' + k: v for k, v in home_stats_5.items()})
        pre_game_features.update({'away_' + k: v for k, v in away_stats_5.items()})
        pre_game_features.update({'home_' + k: v for k, v in home_ema_5.items()})
        pre_game_features.update({'away_' + k: v for k, v in away_ema_5.items()})
        pre_game_features.update({'home_' + k: v for k, v in home_ema_advanced.items()})
        pre_game_features.update({'away_' + k: v for k, v in away_ema_advanced.items()})

        # ✅ SOLUCIÓN: Calcular e inyectar las 4 nuevas features directamente
        home_true_shooting = calculate_true_shooting_percentage(home_history_df)
        away_true_shooting = calculate_true_shooting_percentage(away_history_df)
        pre_game_features['home_true_shooting_percentage'] = home_true_shooting
        pre_game_features['away_true_shooting_percentage'] = away_true_shooting
        pre_game_features['diff_true_shooting_percentage'] = home_true_shooting - away_true_shooting
        
        home_shot_selection = calculate_shot_selection_intelligence(home_history_df)
        away_shot_selection = calculate_shot_selection_intelligence(away_history_df)
        pre_game_features['home_shot_selection_score'] = home_shot_selection
        pre_game_features['away_shot_selection_score'] = away_shot_selection
        pre_game_features['diff_shot_selection_score'] = home_shot_selection - away_shot_selection

        home_schedule_analysis = calculate_enhanced_schedule_analysis(home_history_df)
        away_schedule_analysis = calculate_enhanced_schedule_analysis(away_history_df)
        pre_game_features['home_schedule_analysis_score'] = home_schedule_analysis
        pre_game_features['away_schedule_analysis_score'] = away_schedule_analysis
        pre_game_features['diff_schedule_analysis_score'] = home_schedule_analysis - away_schedule_analysis

        home_pace_consistency = calculate_pace_consistency_tracking(home_history_df)
        away_pace_consistency = calculate_pace_consistency_tracking(away_history_df)
        pre_game_features['home_pace_consistency_score'] = home_pace_consistency
        pre_game_features['away_pace_consistency_score'] = away_pace_consistency
        pre_game_features['diff_pace_consistency_score'] = home_pace_consistency - away_pace_consistency

        # ✅ NUEVAS FEATURES AVANZADAS (PMCI, CPRC, TAV, EDI, PLSVI)
        try:
            # Precalcular strength del rival con slice histórico para anti-leakage
            team_strengths_current = _compute_team_strengths(df.iloc[:index]) if (index > 0 and ADVANCED_FEATURE_ABLATION.get('cprc', True)) else {}
        except Exception:
            team_strengths_current = {}

        # PMCI
        if ADVANCED_FEATURE_ABLATION.get('pmci', True):
            home_pmci = calculate_psychological_momentum_cascade(home_history_df)
            away_pmci = calculate_psychological_momentum_cascade(away_history_df)
            pre_game_features['home_pmci'] = home_pmci
            pre_game_features['away_pmci'] = away_pmci
            pre_game_features['diff_pmci'] = (home_pmci - away_pmci) if not pd.isna(home_pmci) and not pd.isna(away_pmci) else np.nan

        # CPRC
        if ADVANCED_FEATURE_ABLATION.get('cprc', True):
            home_cprc = calculate_competitive_pressure_response(home_history_df, team_strengths_current)
            away_cprc = calculate_competitive_pressure_response(away_history_df, team_strengths_current)
            pre_game_features['home_cprc'] = home_cprc
            pre_game_features['away_cprc'] = away_cprc
            pre_game_features['diff_cprc'] = (home_cprc - away_cprc) if not pd.isna(home_cprc) and not pd.isna(away_cprc) else np.nan

        # TAV
        if ADVANCED_FEATURE_ABLATION.get('tav', True):
            home_tav = calculate_tactical_adaptation_velocity(home_history_df)
            away_tav = calculate_tactical_adaptation_velocity(away_history_df)
            pre_game_features['home_tav'] = home_tav
            pre_game_features['away_tav'] = away_tav
            pre_game_features['diff_tav'] = (home_tav - away_tav) if not pd.isna(home_tav) and not pd.isna(away_tav) else np.nan

        # EDI (proxy sin fouls/FTA por cuarto)
        if ADVANCED_FEATURE_ABLATION.get('edi', True):
            home_edi = calculate_energy_distribution_intelligence(home_history_df)
            away_edi = calculate_energy_distribution_intelligence(away_history_df)
            pre_game_features['home_edi'] = home_edi
            pre_game_features['away_edi'] = away_edi
            pre_game_features['diff_edi'] = (home_edi - away_edi) if not pd.isna(home_edi) and not pd.isna(away_edi) else np.nan

        # PLSVI
        if ADVANCED_FEATURE_ABLATION.get('plsvi', True):
            home_plsvi = calculate_power_law_scoring_volatility(home_history_df)
            away_plsvi = calculate_power_law_scoring_volatility(away_history_df)
            pre_game_features['home_plsvi'] = home_plsvi
            pre_game_features['away_plsvi'] = away_plsvi
            pre_game_features['diff_plsvi'] = (home_plsvi - away_plsvi) if not pd.isna(home_plsvi) and not pd.isna(away_plsvi) else np.nan

        # 🆕 CALCULAR FT_RATE_COMBINED (requiere ambos equipos)
        ft_rate_combined = calculate_ft_rate_combined(home_history_df, away_history_df)
        pre_game_features['ft_rate_combined'] = ft_rate_combined

        # 🆕 NUEVAS FEATURES POR CUARTO (si hay datos por cuarto)
        try:
            home_q4_ft = calculate_q4_ft_rate(home_history_df)
            away_q4_ft = calculate_q4_ft_rate(away_history_df)
            pre_game_features['home_q4_ft_rate'] = home_q4_ft
            pre_game_features['away_q4_ft_rate'] = away_q4_ft
            pre_game_features['diff_q4_ft_rate'] = (home_q4_ft - away_q4_ft) if not pd.isna(home_q4_ft) and not pd.isna(away_q4_ft) else np.nan
        except Exception:
            pre_game_features['home_q4_ft_rate'] = np.nan
            pre_game_features['away_q4_ft_rate'] = np.nan
            pre_game_features['diff_q4_ft_rate'] = np.nan

        try:
            home_pace_shift = calculate_q3_q4_pace_shift(home_history_df)
            away_pace_shift = calculate_q3_q4_pace_shift(away_history_df)
            pre_game_features['home_q3_q4_pace_shift'] = home_pace_shift
            pre_game_features['away_q3_q4_pace_shift'] = away_pace_shift
            pre_game_features['diff_q3_q4_pace_shift'] = (home_pace_shift - away_pace_shift) if not pd.isna(home_pace_shift) and not pd.isna(away_pace_shift) else np.nan
        except Exception:
            pre_game_features['home_q3_q4_pace_shift'] = np.nan
            pre_game_features['away_q3_q4_pace_shift'] = np.nan
            pre_game_features['diff_q3_q4_pace_shift'] = np.nan

        try:
            home_q4_tor = calculate_q4_turnover_rate(home_history_df)
            away_q4_tor = calculate_q4_turnover_rate(away_history_df)
            pre_game_features['home_q4_to_rate'] = home_q4_tor
            pre_game_features['away_q4_to_rate'] = away_q4_tor
            pre_game_features['diff_q4_to_rate'] = (home_q4_tor - away_q4_tor) if not pd.isna(home_q4_tor) and not pd.isna(away_q4_tor) else np.nan
        except Exception:
            pre_game_features['home_q4_to_rate'] = np.nan
            pre_game_features['away_q4_to_rate'] = np.nan
            pre_game_features['diff_q4_to_rate'] = np.nan
        
        # ✅ NUEVO PASO: Calcular las Features H2H
        # Pasamos el DataFrame completo (df.iloc[:index]) para que busque el historial
        h2h_stats = calculate_h2h_features(home_team, away_team, df.iloc[:index])
        pre_game_features.update(h2h_stats)
        
        # Calcular diferencias
        for stat in cols_to_avg:
            pre_game_features[f'diff_avg_{stat}_last_5'] = (
                pre_game_features.get(f'home_avg_{stat}_last_5', np.nan) - 
                pre_game_features.get(f'away_avg_{stat}_last_5', np.nan)
            )
        
        for stat in cols_to_ema:
            pre_game_features[f'diff_ema_{stat}_last_5'] = (
                pre_game_features.get(f'home_ema_{stat}_last_5', np.nan) - 
                pre_game_features.get(f'away_ema_{stat}_last_5', np.nan)
            )
            
        for ema_name, ema_period in EMA_RANGES.items():
            for stat in cols_to_ema:
                home_key = f'home_ema_{stat}_{ema_name}_{ema_period}'
                away_key = f'away_ema_{stat}_{ema_name}_{ema_period}'
                diff_key = f'diff_ema_{stat}_{ema_name}_{ema_period}'
                
                if home_key in pre_game_features and away_key in pre_game_features:
                    pre_game_features[diff_key] = (
                        pre_game_features[home_key] - pre_game_features[away_key]
                    )
                else:
                    pre_game_features[diff_key] = np.nan

        # Extraer datos del partido
        raw_match = match['raw_match']
        
        q1_scores = raw_match.get('quarter_scores', {}).get('Q1', {})
        q2_scores = raw_match.get('quarter_scores', {}).get('Q2', {})
        q3_scores = raw_match.get('quarter_scores', {}).get('Q3', {})
        
        q1_home, q1_away = safe_int(q1_scores.get('home_score', 0)), safe_int(q1_scores.get('away_score', 0))
        q2_home, q2_away = safe_int(q2_scores.get('home_score', 0)), safe_int(q2_scores.get('away_score', 0))
        q3_home, q3_away = safe_int(q3_scores.get('home_score', 0)), safe_int(q3_scores.get('away_score', 0))

        base_info = {
            'home_score': match['home_score'], 'away_score': match['away_score'],
            'total_score': match['total_score'], 'final_total_score': match['total_score'],
            'home_team': home_team, 'away_team': away_team, 'raw_match': raw_match
        }

        # 🎯 CÁLCULO DE TENDENCIAS CON SISTEMA INTELIGENTE MEJORADO
        home_trends = calculate_team_quarter_trends(home_history_df)
        away_trends = calculate_team_quarter_trends(away_history_df)
        team_trends = {'home': home_trends, 'away': away_trends}

        # GENERAR FEATURES PARA Q3
        features_q3 = copy.deepcopy(pre_game_features)
        q_scores_q3 = {'q1_home': q1_home, 'q1_away': q1_away, 'q2_home': q2_home, 'q2_away': q2_away, 'q3_home': q3_home, 'q3_away': q3_away}
        live_metrics_q3 = calculate_live_pace_metrics(q_scores_q3, 'q3_end', team_trends)
        features_q3.update({
            'q1_total': q1_home + q1_away, 'q2_total': q2_home + q2_away, 'q3_total': q3_home + q3_away,
            'halftime_total': (q1_home + q1_away) + (q2_home + q2_away),
            'q3_end_total': (q1_home + q1_away) + (q2_home + q2_away) + (q3_home + q3_away),
            'q1_diff': q1_home - q1_away, 'q2_diff': q2_home - q2_away, 'q3_diff': q3_home - q3_away,
            'q2_trend': (q2_home + q2_away) - (q1_home + q1_away),
            'q3_trend': (q3_home + q3_away) - (q2_home + q2_away),
            'quarter_variance': np.std([q1_home + q1_away, q2_home + q2_away, q3_home + q3_away])
        })
        features_q3.update(live_metrics_q3)
        features_q3.update(base_info)
        real_balance_q3 = calculate_real_balance_features(q_scores_q3, 'q3_end')
        features_q3.update(real_balance_q3)
        try:
            features_q3['garbage_time_risk'] = calculate_garbage_time_risk(
                q_scores_q3, 'q3_end',
                lead_stability=real_balance_q3.get('lead_stability', 0.5),
                quarter_variance=features_q3.get('quarter_variance', np.nan),
                is_potential_blowout=real_balance_q3.get('is_potential_blowout', 0)
            )
        except Exception:
            features_q3['garbage_time_risk'] = np.nan
        all_features.append(features_q3)
        
        # GENERAR FEATURES PARA Q2
        features_q2 = copy.deepcopy(pre_game_features)
        q_scores_q2 = {'q1_home': q1_home, 'q1_away': q1_away, 'q2_home': q2_home, 'q2_away': q2_away, 'q3_home': 0, 'q3_away': 0}
        live_metrics_q2 = calculate_live_pace_metrics(q_scores_q2, 'halftime', team_trends)
        features_q2.update({
            'q1_total': q1_home + q1_away, 'q2_total': q2_home + q2_away, 'q3_total': 0,
            'halftime_total': (q1_home + q1_away) + (q2_home + q2_away),
            'q3_end_total': (q1_home + q1_away) + (q2_home + q2_away),
            'q1_diff': q1_home - q1_away, 'q2_diff': q2_home - q2_away, 'q3_diff': 0,
            'q2_trend': (q2_home + q2_away) - (q1_home + q1_away), 'q3_trend': 0,
            'quarter_variance': np.std([q1_home + q1_away, q2_home + q2_away, 0])
        })
        features_q2.update(live_metrics_q2)
        features_q2.update(base_info)
        real_balance_q2 = calculate_real_balance_features(q_scores_q2, 'halftime')
        features_q2.update(real_balance_q2)
        try:
            features_q2['garbage_time_risk'] = calculate_garbage_time_risk(
                q_scores_q2, 'halftime',
                lead_stability=real_balance_q2.get('lead_stability', 0.5),
                quarter_variance=features_q2.get('quarter_variance', np.nan),
                is_potential_blowout=real_balance_q2.get('is_potential_blowout', 0)
            )
        except Exception:
            features_q2['garbage_time_risk'] = np.nan
        all_features.append(features_q2)

    final_df_temp = pd.DataFrame(all_features)
    
    if final_df_temp.empty:
        print("❌ No se generaron características válidas")
        return pd.DataFrame()
    
    critical_features = [
        'home_avg_points_scored_last_5', 'away_avg_points_scored_last_5',
        'home_avg_total_score_last_5', 'away_avg_total_score_last_5'
    ]

    existing_critical = [f for f in critical_features if f in final_df_temp.columns]
    if existing_critical:
        print(f"🔍 Aplicando filtro con {len(existing_critical)} características críticas...")
        final_df = final_df_temp.dropna(subset=existing_critical, how='all').reset_index(drop=True)
    else:
        print("⚠️ No se encontraron características críticas - manteniendo todos los datos")
        final_df = final_df_temp.copy()
    
    # ✅ ACTIVAR FILTERED_FEATURES_TO_USE - NUEVO EN FASE 1A
    available_features = [f for f in FILTERED_FEATURES_TO_USE if f in final_df.columns]
    
    print(f"✅ Procesamiento FASE 1A completado!")
    print(f"📊 Total de data points válidos: {len(final_df)}")
    print(f"🎯 Features incluidas: {len(available_features)} características (filtradas)")
    print(f"🚀 Mejoras FASE 1A integradas:")
    print(f"   - ✅ Features Over/Under optimizadas integradas directamente")
    print(f"   - ✅ Sistema inteligente de detección de columnas")
    print(f"   - ✅ 6 nuevas estadísticas basketball (turnovers, assists, FG, steals, fouls)")
    print(f"   - ✅ Limitadores de seguridad para trend_factor (0.85-1.15)")
    print(f"   - ✅ FILTERED_FEATURES_TO_USE activado")
    print(f"   - ✅ Función calculate_momentum_metrics única y optimizada")
    print(f"   - ✅ Momentum multirango: {len(EMA_RANGES)} escalas temporales")
    print(f"   - ✅ Métricas de contexto: Home Advantage, Comeback Ability, Consistency")
    print(f"   - ✅ Live metrics avanzadas: Momentum Shift, Quarter Consistency")
    
    # 🔍 VERIFICACIÓN NUEVAS FEATURES
    new_features_check = [
        'home_true_shooting_percentage',
        'home_shot_selection_score',
        'home_schedule_analysis_score',
        'home_pace_consistency_score',
        # Quarter-based direct features (home side as proxy; away/diff mirror cobertura)
        'home_q4_ft_rate',
        'home_q3_q4_pace_shift',
        'home_q4_to_rate'
    ]
    
    print(f"\n🔍 VERIFICACIÓN NUEVAS FEATURES:")
    for feat in new_features_check:
        if feat in final_df.columns:
            non_null_count = int(final_df[feat].notna().sum())
            total = len(final_df)
            if non_null_count > 0:
                coverage = non_null_count / max(total, 1)
                mean_val = float(final_df[feat].dropna().mean()) if non_null_count > 0 else float('nan')
                print(f"   ✅ {feat}: {non_null_count}/{total} válidos ({coverage:.1%}) | mean={mean_val:.4f}")
            else:
                print(f"   ⚠️ {feat}: ENCONTRADA pero sin valores válidos (todos NaN)")
        else:
            print(f"   ❌ {feat}: NO ENCONTRADA EN DATASET")
    
    # 🔍 VERIFICACIÓN COBERTURA DE BOX POR CUARTO (inputs crudos)
    try:
        quarter_keys = ['field_goals_attempted', 'offensive_rebounds', 'turnovers', 'free_throws_attempted']
        for q in range(1, 5):
            avail = 0
            total_rows = len(df)
            for key in quarter_keys:
                home_col = f'home_q{q}_{key}'
                away_col = f'away_q{q}_{key}'
                home_ok = (home_col in df.columns) and df[home_col].notna().any()
                away_ok = (away_col in df.columns) and df[away_col].notna().any()
                if home_ok and away_ok:
                    avail += 1
            print(f"   📦 Q{q} coverage (home+away ambos presentes) {avail}/{len(quarter_keys)} métricas base")
    except Exception as _diag_err:
        print(f"   ⚠️ Diagnóstico de cobertura por cuarto falló: {_diag_err}")
    
    return final_df
# ADVANCED FEATURES: PMCI, CPRC, TAV, EDI, PLSVI
# ===========================================

def _compute_team_strengths(df):
    """
    Precalcula un rating de fuerza por equipo usando sólo datos previos (anti-leakage).
    Estrategia:
      - Para cada equipo: promedio de puntos anotados y recibidos en el dataset pasado (df).
      - Calcula 'net rating' simple: off - def.
      - Normaliza por Z-score y mapea a escala [0,100] con 50 como media: 50 + 10*z.
    """
    try:
        if df is None or df.empty:
            return {}

        # Acumular puntos por equipo
        team_points_scored = {}
        team_points_allowed = {}
        team_counts = {}

        for _, row in df.iterrows():
            try:
                home = row.get('home_team')
                away = row.get('away_team')
                hs = safe_float(row.get('home_score', 0))
                as_ = safe_float(row.get('away_score', 0))

                if home:
                    team_points_scored[home] = team_points_scored.get(home, 0.0) + hs
                    team_points_allowed[home] = team_points_allowed.get(home, 0.0) + as_
                    team_counts[home] = team_counts.get(home, 0) + 1
                if away:
                    team_points_scored[away] = team_points_scored.get(away, 0.0) + as_
                    team_points_allowed[away] = team_points_allowed.get(away, 0.0) + hs
                    team_counts[away] = team_counts.get(away, 0) + 1
            except Exception:
                continue

        teams = list(team_counts.keys())
        if not teams:
            return {}

        off_means = []
        def_means = []
        for t in teams:
            c = max(1, team_counts.get(t, 1))
            off_means.append(team_points_scored.get(t, 0.0) / c)
            def_means.append(team_points_allowed.get(t, 0.0) / c)

        off_means = np.array(off_means, dtype=float)
        def_means = np.array(def_means, dtype=float)
        net = off_means - def_means

        mu = float(np.mean(net))
        sigma = float(np.std(net))
        if sigma <= 1e-6:
            sigma = 1.0

        z = (net - mu) / sigma
        strengths = 50.0 + 10.0 * z
        strengths = np.clip(strengths, 0.0, 100.0)

        return {team: float(strengths[i]) for i, team in enumerate(teams)}
    except Exception:
        return {}

def calculate_psychological_momentum_cascade(team_history_df):
    """
    PMCI [0,1]: cascadas de momentum negativo con asimetría.
    Usa thresholds configurables en ADVANCED_FEATURE_PARAMS.
    Neutral: 0.5 si datos insuficientes.
    """
    try:
        if team_history_df is None or team_history_df.empty:
            return 0.5

        pm_th = ADVANCED_FEATURE_PARAMS.get('pmci_neg_pm_threshold', -10)
        low_fg = ADVANCED_FEATURE_PARAMS.get('pmci_low_fg_pct', 0.40)
        high_to = ADVANCED_FEATURE_PARAMS.get('pmci_high_turnovers', 18)
        neg_thr = ADVANCED_FEATURE_PARAMS.get('pmci_neg_score_threshold', 2)
        expn = ADVANCED_FEATURE_PARAMS.get('pmci_streak_exponent', 1.5)
        amp_div = ADVANCED_FEATURE_PARAMS.get('pmci_amplification_factor', 20.0)
        scale_div = ADVANCED_FEATURE_PARAMS.get('pmci_scale_divisor', 15.0)

        recent = team_history_df.tail(10)
        if recent.empty:
            return 0.5

        negative_events = []
        for _, game in recent.iterrows():
            neg_score = 0
            pm = safe_float(game.get('plus_minus', 0.0))
            fg_made = safe_float(game.get('field_goals_made', 0.0))
            fg_att = safe_float(game.get('field_goals_attempted', 0.0))
            to = safe_float(game.get('turnovers', 0.0))
            fg_pct = fg_made / fg_att if fg_att > 0 else 0.0

            if pm < pm_th:
                neg_score += 2
            if fg_pct < low_fg:
                neg_score += 1
            if to > high_to:
                neg_score += 1
            negative_events.append(neg_score)

        cascade_strength = 0.0
        current_streak = 0
        for ev in negative_events:
            if ev >= neg_thr:
                current_streak += 1
                cascade_strength += (current_streak ** expn)
            else:
                current_streak = 0

        recent_momentum = safe_mean(team_history_df.get('plus_minus', pd.Series([0])).tail(3))
        amplification = 1.0 + max(0.0, -recent_momentum / max(amp_div, 1e-6))

        pmci = (cascade_strength * amplification) / max(scale_div, 1e-6)
        return float(max(0.0, min(1.0, pmci)))
    except Exception:
        return 0.5

def calculate_competitive_pressure_response(team_history_df, team_strength_ratings):
    """
    CPRC [0,1]: qué tan bien responde el equipo al nivel de oposición.
    - expected_perf = 100 + (opp_strength - 50) * slope
    - actual_perf = 100 * points_scored / baseline_points
    - response = actual/expected (cap si oponente débil)
    Neutral: 0.5 si datos insuficientes o sin opponent_team.
    """
    try:
        if team_history_df is None or team_history_df.empty:
            return 0.5

        slope = ADVANCED_FEATURE_PARAMS.get('cprc_expected_slope', 0.3)
        cap_weak = ADVANCED_FEATURE_PARAMS.get('cprc_response_cap_weak', 1.2)
        window = int(ADVANCED_FEATURE_PARAMS.get('cprc_history_window', 15))

        hist = team_history_df.tail(window)
        if hist.empty or 'opponent_team' not in hist.columns:
            return 0.5

        # Baseline de puntos propios para normalizar performance
        baseline_points = safe_mean(team_history_df.get('points_scored', pd.Series([0])))
        if baseline_points <= 0:
            return 0.5

        responses = []
        for _, game in hist.iterrows():
            opp = game.get('opponent_team', None)
            opp_strength = float(team_strength_ratings.get(opp, 50.0)) if opp else 50.0

            expected_perf = 100.0 + (opp_strength - 50.0) * slope
            points_scored = safe_float(game.get('points_scored', 0.0))
            actual_perf = 100.0 * safe_divide(points_scored, baseline_points, 1.0)

            resp = safe_divide(actual_perf, expected_perf, 1.0)
            if opp_strength <= 60.0:
                resp = min(cap_weak, resp)
            responses.append(resp)

        if not responses:
            return 0.5

        avg_response = float(np.mean(responses))
        response_std = float(np.std(responses))
        consistency_factor = 1.0 / (1.0 + response_std)

        cprc = avg_response * consistency_factor
        return float(max(0.0, min(1.0, cprc)))
    except Exception:
        return 0.5

def calculate_tactical_adaptation_velocity(team_history_df):
    """
    TAV [0,1]: velocidad de adaptación táctica mid-game.
    Usa puntos/posesión por cuarto si hay box por cuarto; si no, puntos/minuto (fallback).
    Neutral: 0.5 si datos insuficientes.
    """
    try:
        if team_history_df is None or team_history_df.empty:
            return 0.5

        window = int(ADVANCED_FEATURE_PARAMS.get('tav_window', 8))
        eff_thr = ADVANCED_FEATURE_PARAMS.get('tav_eff_threshold', 1.5)
        rec_mul = ADVANCED_FEATURE_PARAMS.get('tav_recovery_multiplier', 1.3)
        sustain_bonus = ADVANCED_FEATURE_PARAMS.get('tav_sustain_bonus', 0.5)
        scale_div = ADVANCED_FEATURE_PARAMS.get('tav_scale_divisor', 3.0)

        def _q_eff(game, q):
            # Intentar puntos/posesión
            fga = safe_float(game.get(f'q{q}_fga', np.nan))
            oreb = safe_float(game.get(f'q{q}_oreb', np.nan))
            tov = safe_float(game.get(f'q{q}_tov', np.nan))
            fta = safe_float(game.get(f'q{q}_fta', np.nan))
            pts = safe_float(game.get(f'q{q}_points', 0.0))
            if not pd.isna(fga) and not pd.isna(oreb) and not pd.isna(tov) and not pd.isna(fta):
                poss = fga - oreb + tov + 0.44 * fta
                if poss and poss > 0:
                    return pts / poss
            # Fallback: puntos por minuto
            return safe_divide(pts, 12.0, 0.0)

        recent = team_history_df.tail(window)
        if recent.empty:
            return 0.5

        if not all(col in recent.columns for col in ['q1_points','q2_points','q3_points','q4_points']):
            return 0.5

        scores = []
        for _, game in recent.iterrows():
            q1_eff = _q_eff(game, 1)
            q2_eff = _q_eff(game, 2)
            q3_eff = _q_eff(game, 3)
            q4_eff = _q_eff(game, 4)
            quarters = [q1_eff, q2_eff, q3_eff, q4_eff]

            adaptations = 0
            sustain = 0.0

            if q1_eff < eff_thr and q2_eff > q1_eff * rec_mul:
                adaptations += 1
            if q2_eff < eff_thr and q3_eff > q2_eff * rec_mul:
                adaptations += 1
            if q3_eff < eff_thr and q4_eff > q3_eff * rec_mul:
                adaptations += 1

            for i in range(1, 4):
                prev = quarters[i-1]
                curr = quarters[i]
                if prev > 0 and curr > prev * rec_mul:
                    if i < 3 and quarters[i+1] >= curr * 0.9:
                        sustain += sustain_bonus

            scores.append(adaptations + sustain)

        if not scores:
            return 0.5

        avg_score = float(np.mean(scores))
        tav = avg_score / max(scale_div, 1e-6)
        return float(max(0.0, min(1.0, tav)))
    except Exception:
        return 0.5

def calculate_energy_distribution_intelligence(team_history_df):
    """
    EDI [0,1]: inteligencia de distribución de energía a lo largo del juego.
    Usa puntos/posesión por cuarto si hay box por cuarto; añade señal por spike de FT en Q4.
    Neutral: 0.5 si datos insuficientes.
    """
    try:
        if team_history_df is None or team_history_df.empty:
            return 0.5

        window = int(ADVANCED_FEATURE_PARAMS.get('edi_window', 6))
        peak_q4_bonus = ADVANCED_FEATURE_PARAMS.get('edi_peak_q4_bonus', 1.5)
        mid_bonus = ADVANCED_FEATURE_PARAMS.get('edi_mid_saving_bonus', 1.0)
        early_dump_thr = ADVANCED_FEATURE_PARAMS.get('edi_early_dump_penalty_threshold', 3.0)
        early_dump_ratio = ADVANCED_FEATURE_PARAMS.get('edi_early_dump_penalty_ratio', 0.7)
        close_margin = ADVANCED_FEATURE_PARAMS.get('edi_close_game_margin', 5)
        low_var_thr = ADVANCED_FEATURE_PARAMS.get('edi_low_variance_threshold', 0.5)
        fatigue_penalty = ADVANCED_FEATURE_PARAMS.get('edi_fatigue_penalty_factor', 0.9)
        scale_div = ADVANCED_FEATURE_PARAMS.get('edi_scale_divisor', 4.0)

        recent = team_history_df.tail(window)
        if recent.empty or not all(col in recent.columns for col in ['q1_points','q2_points','q3_points','q4_points']):
            return 0.5

        def _q_eff(game, q):
            fga = safe_float(game.get(f'q{q}_fga', np.nan))
            oreb = safe_float(game.get(f'q{q}_oreb', np.nan))
            tov = safe_float(game.get(f'q{q}_tov', np.nan))
            fta = safe_float(game.get(f'q{q}_fta', np.nan))
            pts = safe_float(game.get(f'q{q}_points', 0.0))
            if not pd.isna(fga) and not pd.isna(oreb) and not pd.isna(tov) and not pd.isna(fta):
                poss = fga - oreb + tov + 0.44 * fta
                if poss and poss > 0:
                    return pts / poss
            return safe_divide(pts, 12.0, 0.0)

        scores = []
        for _, game in recent.iterrows():
            q1 = _q_eff(game, 1)
            q2 = _q_eff(game, 2)
            q3 = _q_eff(game, 3)
            q4 = _q_eff(game, 4)
            intens = [q1, q2, q3, q4]

            score = 0.0
            # Ahorro medio hacia final
            if q2 < (q1 + q4) / 2.0:
                score += mid_bonus
            if q3 < (q2 + q4) / 2.0:
                score += mid_bonus

            # Peak en clutch (Q4)
            if q4 == max(intens):
                score += peak_q4_bonus

            # Evitar dump temprano insostenible (cuando solo hay ritmo temprano)
            if q1 > early_dump_thr and q4 < q1 * early_dump_ratio:
                score -= 1.0

            # 🆕 Spike de FT en Q4 (indicador de ejecución/estrategia en cierre)
            q4_fta = safe_float(game.get('q4_fta', np.nan))
            q13_fta = [safe_float(game.get('q1_fta', np.nan)),
                       safe_float(game.get('q2_fta', np.nan)),
                       safe_float(game.get('q3_fta', np.nan))]
            q13_fta = [x for x in q13_fta if not pd.isna(x)]
            if not pd.isna(q4_fta) and q13_fta:
                base_fta = np.mean(q13_fta)
                if base_fta > 0:
                    spike_ratio = q4_fta / base_fta
                    if spike_ratio >= 1.3:
                        score += 0.5  # bonus moderado por presión/ataque al aro en cierre

            # Consistencia en juegos cerrados
            pm = safe_float(game.get('plus_minus', 0.0))
            if abs(pm) <= close_margin:
                m = np.mean(intens) if np.mean(intens) > 0 else 0.0
                if m > 0:
                    cv = (np.std(intens) / m) if m > 0 else 1.0
                    if cv < low_var_thr:
                        score += 1.0

            # Ajuste por fatiga acumulada
            try:
                fat = calculate_back_to_back_fatigue(team_history_df)
                if fat < 0.9:
                    score *= fatigue_penalty
            except Exception:
                pass

            scores.append(max(0.0, score))

        if not scores:
            return 0.5

        avg = float(np.mean(scores))
        edi = avg / max(scale_div, 1e-6)
        return float(max(0.0, min(1.0, edi)))
    except Exception:
        return 0.5

def calculate_power_law_scoring_volatility(team_history_df):
    """
    PLSVI [0,1] (0.5 = neutral): tendencia a eventos explosivos o colapsos en Q4.
    Detecta bursts/collapses contra un esperado normal estabilizado y combina magnitud y frecuencia.
    Neutral: 0.5 si no hay eventos detectados.
    """
    try:
        if team_history_df is None or team_history_df.empty:
            return 0.5

        window = int(ADVANCED_FEATURE_PARAMS.get('plsvi_window', 12))
        thr_pos = ADVANCED_FEATURE_PARAMS.get('plsvi_q4_positive_threshold', 28)
        thr_neg = ADVANCED_FEATURE_PARAMS.get('plsvi_q4_negative_threshold', 15)
        exp_low = ADVANCED_FEATURE_PARAMS.get('plsvi_expected_q4_low', 18)
        exp_high = ADVANCED_FEATURE_PARAMS.get('plsvi_expected_q4_high', 30)
        pos_cap = ADVANCED_FEATURE_PARAMS.get('plsvi_positive_cap', 2.0)
        neg_cap = ADVANCED_FEATURE_PARAMS.get('plsvi_negative_cap', 1.5)
        weight = ADVANCED_FEATURE_PARAMS.get('plsvi_frequency_magnitude_weight', 0.3)

        recent = team_history_df.tail(window)
        if recent.empty or 'q4_points' not in team_history_df.columns:
            return 0.5

        # Baseline esperado para Q4: promedio histórico clamp a [exp_low, exp_high]
        base_q4 = team_history_df['q4_points'].dropna()
        if len(base_q4) >= 3:
            expected_q4 = float(np.clip(base_q4.mean(), exp_low, exp_high))
        else:
            expected_q4 = float((exp_low + exp_high) / 2.0)

        expected_min = 0.85 * expected_q4

        events = []
        for _, game in recent.iterrows():
            q4 = safe_float(game.get('q4_points', 0.0))
            if q4 <= 0:
                continue

            # Positive burst
            if q4 > thr_pos and q4 > 1.3 * expected_q4:
                mag = (q4 - expected_q4) / max(expected_q4, 1e-6)
                events.append(min(pos_cap, mag))

            # Negative collapse
            if q4 < thr_neg and q4 < expected_min:
                mag_neg = (expected_min - q4) / max(expected_min, 1e-6)
                events.append(-min(neg_cap, mag_neg))

        if not events:
            return 0.5

        avg_mag = float(np.mean([abs(e) for e in events]))
        freq = float(len(events)) / max(1, len(recent))
        plsvi = 0.5 + weight * avg_mag * freq
        return float(max(0.0, min(1.0, plsvi)))
    except Exception:
        return 0.5
    
def calculate_second_half_efficiency(team_history_df):
    """
    🎯 NUEVA FEATURE 1: Eficiencia en segunda mitad
    Mide qué tan bien convierte un equipo en Q3+Q4 vs primera mitad
    """
    if team_history_df.empty or len(team_history_df) < 3:
        return 0.5  # Neutral default
    
    try:
        # Verificar que tenemos datos de cuartos
        quarter_cols = ['q3_points', 'q4_points', 'q1_points', 'q2_points']
        available_cols = [col for col in quarter_cols if col in team_history_df.columns]
        
        if len(available_cols) < 4:
            return 0.5
        
        recent_games = team_history_df.tail(5)  # Últimos 5 juegos
        
        second_half_totals = []
        first_half_totals = []
        
        for _, game in recent_games.iterrows():
            try:
                q1_pts = safe_float(game.get('q1_points', 0))
                q2_pts = safe_float(game.get('q2_points', 0))
                q3_pts = safe_float(game.get('q3_points', 0))
                q4_pts = safe_float(game.get('q4_points', 0))
                
                first_half = q1_pts + q2_pts
                second_half = q3_pts + q4_pts
                
                # Solo incluir si ambas mitades tienen datos válidos
                if first_half > 0 and second_half > 0:
                    first_half_totals.append(first_half)
                    second_half_totals.append(second_half)
            except Exception as game_error:
                continue  # Saltar este juego si hay errores
        
        # Calcular eficiencia si tenemos suficientes datos
        if len(first_half_totals) >= 2 and len(second_half_totals) >= 2:
            avg_first_half = safe_mean(first_half_totals)
            avg_second_half = safe_mean(second_half_totals)
            
            if avg_first_half > 0:
                efficiency = safe_divide(avg_second_half, avg_first_half, 1.0)
                # Normalizar entre 0 y 1 (0.5 = mismo rendimiento)
                return min(max(efficiency / 2, 0.0), 1.0)
        
        return 0.5  # Default neutral
        
    except Exception as e:
        return 0.5

def calculate_ft_rate_combined(home_history_df, away_history_df):
    """
    🎯 NUEVA FEATURE 2: Tasa combinada de tiros libres en situaciones cerradas
    Predice cuántos FT extra habrá por fouling strategy en juegos cerrados
    """
    try:
        # Obtener datos de FT de ambos equipos
        home_ft_data = []
        away_ft_data = []
        
        # Buscar columnas de FT
        ft_made_cols = ['free_throws_made', 'ft_made', 'ftm']
        ft_att_cols = ['free_throws_attempted', 'ft_attempted', 'fta']
        
        home_ft_made_col = None
        home_ft_att_col = None
        
        for col in ft_made_cols:
            if col in home_history_df.columns:
                home_ft_made_col = col
                break
        
        for col in ft_att_cols:
            if col in home_history_df.columns:
                home_ft_att_col = col
                break
        
        # Si no hay datos de FT, usar proxy con total de puntos
        if not home_ft_made_col or not home_ft_att_col:
            # Proxy: asumir que 15% de puntos vienen de FT en juegos cerrados
            home_avg_pts = safe_mean(home_history_df.get('points_scored', pd.Series([100])))
            away_avg_pts = safe_mean(away_history_df.get('points_scored', pd.Series([100])))
            
            # Si equipos son similares en scoring → más FT esperados en juegos cerrados
            pts_similarity = 1 - abs(home_avg_pts - away_avg_pts) / max(home_avg_pts, away_avg_pts, 100)
            
            # Retornar tasa estimada basada en similitud
            return 0.15 + (pts_similarity * 0.10)  # 15-25% de puntos por FT
        
        # Cálculo real con datos de FT
        for _, game in home_history_df.tail(5).iterrows():
            ft_made = game.get(home_ft_made_col, 0)
            ft_att = game.get(home_ft_att_col, 0)
            if ft_att > 0:
                home_ft_data.append(ft_made / ft_att)
        
        for _, game in away_history_df.tail(5).iterrows():
            ft_made = game.get(home_ft_made_col, 0)  # Mismo formato
            ft_att = game.get(home_ft_att_col, 0)
            if ft_att > 0:
                away_ft_data.append(ft_made / ft_att)
        
        if not home_ft_data and not away_ft_data:
            return 0.20  # Default NBA average
        
        # Combinar tasas de ambos equipos
        all_ft_rates = home_ft_data + away_ft_data
        combined_rate = np.mean(all_ft_rates) if all_ft_rates else 0.20
        
        # Ajustar por factor de juego cerrado (más FT intentados)
        return min(0.35, combined_rate * 1.2)  # Max 35% boost
        
    except Exception as e:
        print(f"⚠️ Error calculando ft_rate_combined: {e}")
        return 0.20  # Default fallback

def calculate_q4_ft_rate(team_history_df):
    """
    Promedio móvil de FT intentados en Q4 relativo al promedio de Q1–Q3.
    Devuelve un ratio (>=0). Si faltan datos, retorna NaN.
    """
    try:
        recent = team_history_df.tail(int(ADVANCED_FEATURE_PARAMS.get('edi_window', 6)))
        if recent.empty:
            return np.nan
        ratios = []
        for _, g in recent.iterrows():
            q4_fta = safe_float(g.get('q4_fta', np.nan))
            q13 = [safe_float(g.get('q1_fta', np.nan)), safe_float(g.get('q2_fta', np.nan)), safe_float(g.get('q3_fta', np.nan))]
            q13 = [x for x in q13 if not pd.isna(x)]
            if not pd.isna(q4_fta) and q13 and np.mean(q13) > 0:
                ratios.append(q4_fta / np.mean(q13))
        return float(np.mean(ratios)) if ratios else np.nan
    except Exception:
        return np.nan

def calculate_q3_q4_pace_shift(team_history_df):
    """
    Cambio de pace (posesiones) de Q3 a Q4.
    Usa (Q4_poss - Q3_poss)/max(Q3_poss,1) promedio en ventana.
    """
    try:
        recent = team_history_df.tail(int(ADVANCED_FEATURE_PARAMS.get('edi_window', 6)))
        if recent.empty:
            return np.nan
        shifts = []
        for _, g in recent.iterrows():
            def _poss(q):
                fga = safe_float(g.get(f'q{q}_fga', np.nan))
                oreb = safe_float(g.get(f'q{q}_oreb', np.nan))
                tov = safe_float(g.get(f'q{q}_tov', np.nan))
                fta = safe_float(g.get(f'q{q}_fta', np.nan))
                if not pd.isna(fga) and not pd.isna(oreb) and not pd.isna(tov) and not pd.isna(fta):
                    return fga - oreb + tov + 0.44 * fta
                return np.nan
            p3, p4 = _poss(3), _poss(4)
            if not pd.isna(p3) and not pd.isna(p4) and p3 > 0:
                shifts.append((p4 - p3) / p3)
        return float(np.mean(shifts)) if shifts else np.nan
    except Exception:
        return np.nan

def calculate_q4_turnover_rate(team_history_df):
    """
    Tasa de pérdidas en Q4: TOV / (FGA - OREB + TOV + 0.44*FTA) promedio en ventana.
    """
    try:
        recent = team_history_df.tail(int(ADVANCED_FEATURE_PARAMS.get('edi_window', 6)))
        if recent.empty:
            return np.nan
        rates = []
        for _, g in recent.iterrows():
            fga = safe_float(g.get('q4_fga', np.nan))
            oreb = safe_float(g.get('q4_oreb', np.nan))
            tov = safe_float(g.get('q4_tov', np.nan))
            fta = safe_float(g.get('q4_fta', np.nan))
            if not pd.isna(fga) and not pd.isna(oreb) and not pd.isna(tov) and not pd.isna(fta):
                poss = fga - oreb + tov + 0.44 * fta
                if poss > 0:
                    rates.append(tov / poss)
        return float(np.mean(rates)) if rates else np.nan
    except Exception:
        return np.nan

def calculate_efficiency_differential(team_history_df):
    """
    🎯 NUEVA FEATURE 3: Diferencial de eficiencia contextualizado
    """
    if team_history_df.empty or len(team_history_df) < 3:
        return 0.0
    
    try:
        # Verificar columnas necesarias
        required_cols = ['points_scored', 'points_allowed', 'total_score']
        available_cols = [col for col in required_cols if col in team_history_df.columns]
        
        if len(available_cols) < 3:
            return 0.0
        
        recent_games = team_history_df.tail(5)  # Últimos 5 juegos
        
        efficiency_diffs = []
        for _, game in recent_games.iterrows():
            points_scored = game.get('points_scored', 0)
            points_allowed = game.get('points_allowed', 0)
            total_score = game.get('total_score', 0)
            
            if total_score > 0:  # Evitar división por cero
                plus_minus = points_scored - points_allowed
                efficiency_diff = safe_divide(plus_minus, total_score, 0.0)
                efficiency_diffs.append(efficiency_diff)
        
        if len(efficiency_diffs) >= 3:
            return np.mean(efficiency_diffs)
        else:
            return 0.0
            
    except Exception as e:
        print(f"⚠️ Error calculando efficiency_differential: {e}")
        return 0.0

def calculate_back_to_back_fatigue(team_history_df):
    """
    🎯 FEATURE 1: Detecta fatiga por juegos back-to-back
    
    ¿Cómo funciona?
    1. Mira las fechas de los últimos 2-3 partidos del equipo
    2. Si jugaron ayer o antier = están cansados
    3. Reduce su rendimiento esperado proporcionalmente
    
    ¿Qué retorna?
    - 1.0 = Sin fatiga (rindió normal)
    - 0.95 = Fatiga ligera (-5% rendimiento) 
    - 0.85 = Fatiga severa (-15% rendimiento)
    - 0.80 = Fatiga extrema (-20% rendimiento)
    """
    try:
        # Verificar que tenemos suficientes partidos
        if len(team_history_df) < 2:
            return 1.0  # Sin fatiga si no hay historial
        
        # Verificar que tenemos columna de fechas
        if 'date' not in team_history_df.columns:
            return 1.0
        
        # Obtener las últimas 3 fechas de partidos (para mejor análisis)
        last_dates = team_history_df['date'].tail(3).tolist()
        
        if len(last_dates) < 2:
            return 1.0  # Necesitamos al menos 2 fechas
        
        # Calcular días entre partidos recientes
        try:
            # Convertir a datetime si es necesario
            parsed_dates = []
            for date in last_dates:
                if isinstance(date, str):
                    # Manejar formato "23.06.2025 02:00"
                    for fmt in ('%d.%m.%Y %H:%M', '%d.%m.%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
                        try:
                            parsed_date = datetime.strptime(date, fmt)
                            parsed_dates.append(parsed_date)
                            break
                        except ValueError:
                            continue
                elif hasattr(date, 'date'):  # datetime object
                    parsed_dates.append(date)
                else:
                    continue
            
            if len(parsed_dates) < 2:
                return 1.0
            
            # Ordenar fechas (más reciente primero)
            parsed_dates.sort(reverse=True)
            
            # Calcular diferencias en días
            days_since_last = (parsed_dates[0] - parsed_dates[1]).days
            
            # Factor de fatiga basado en días de descanso
            if days_since_last <= 1:
                # Back-to-back o mismo día
                fatigue_factor = 0.80  # -20% rendimiento
            elif days_since_last == 2:
                # 1 día de descanso
                fatigue_factor = 0.85  # -15% rendimiento  
            elif days_since_last == 3:
                # 2 días de descanso
                fatigue_factor = 0.92  # -8% rendimiento
            elif days_since_last == 4:
                # 3 días de descanso
                fatigue_factor = 0.95  # -5% rendimiento
            else:
                # 4+ días de descanso = sin fatiga
                fatigue_factor = 1.0
            
            # Si tenemos 3+ fechas, considerar fatiga acumulada
            if len(parsed_dates) >= 3:
                days_between_2nd_3rd = (parsed_dates[1] - parsed_dates[2]).days
                
                # Si también el partido anterior fue back-to-back
                if days_between_2nd_3rd <= 2:
                    fatigue_factor *= 0.95  # Fatiga acumulada adicional
            
            return fatigue_factor
            
        except Exception as e:
            return 1.0
            
    except Exception as e:
        return 1.0

def calculate_defensive_intensity_drop(team_history_df):
    """
    🎯 FEATURE 2: Detecta colapso defensivo en tiempo real
    
    ¿Cómo funciona?
    1. Analiza trend de points allowed últimos 3-5 partidos
    2. Analiza trend de stats defensivas (rebounds, blocks, steals)
    3. Detecta si defensa está colapsando vs su promedio
    
    ¿Qué retorna?
    - 0.0 = Defensa mejorando
    - 0.5 = Defensa estable
    - 1.0 = Defensa colapsando
    """
    try:
        if team_history_df.empty or len(team_history_df) < 3:
            return 0.5  # Neutral si no hay suficientes datos
        
        # Obtener últimos 5 partidos para análisis
        recent_games = team_history_df.tail(5)
        
        # ANÁLISIS 1: Points Allowed Trend
        points_allowed_trend = 0.0
        if 'points_allowed' in recent_games.columns:
            points_allowed = recent_games['points_allowed'].dropna()
            
            if len(points_allowed) >= 3:
                # Comparar últimos 2 vs primeros 2 partidos
                recent_avg = points_allowed.tail(2).mean()
                earlier_avg = points_allowed.head(2).mean()
                
                if earlier_avg > 0:
                    # Si recent_avg > earlier_avg = defensa empeorando
                    points_trend = (recent_avg - earlier_avg) / earlier_avg
                    points_allowed_trend = min(1.0, max(0.0, points_trend + 0.5))
        
        # ANÁLISIS 2: Defensive Stats Trend
        defensive_stats_trend = 0.0
        defensive_stats = ['defensive_rebounds', 'blocks', 'steals']
        available_def_stats = [stat for stat in defensive_stats if stat in recent_games.columns]
        
        if available_def_stats:
            def_trends = []
            
            for stat in available_def_stats:
                stat_values = recent_games[stat].dropna()
                
                if len(stat_values) >= 3:
                    recent_stat_avg = stat_values.tail(2).mean()
                    earlier_stat_avg = stat_values.head(2).mean()
                    
                    if earlier_stat_avg > 0:
                        # Para stats defensivas, menos = peor defensa
                        stat_trend = (earlier_stat_avg - recent_stat_avg) / earlier_stat_avg
                        def_trends.append(max(0.0, stat_trend))
            
            if def_trends:
                defensive_stats_trend = min(1.0, np.mean(def_trends))
        
        # ANÁLISIS 3: Consistency Check
        consistency_penalty = 0.0
        if 'points_allowed' in recent_games.columns:
            points_allowed = recent_games['points_allowed'].dropna()
            
            if len(points_allowed) >= 3:
                # Alta variabilidad = defensa inconsistente
                cv = points_allowed.std() / max(points_allowed.mean(), 1)
                consistency_penalty = min(0.3, cv / 3)  # Max 30% penalty
        
        # COMBINAR ANÁLISIS
        if points_allowed_trend > 0 or defensive_stats_trend > 0:
            # Hay evidencia de problemas defensivos
            intensity_drop = (points_allowed_trend * 0.6 + 
                            defensive_stats_trend * 0.3 + 
                            consistency_penalty * 0.1)
        else:
            # Sin datos suficientes, usar solo consistency
            intensity_drop = consistency_penalty * 2  # Scale up
        
        # Normalizar entre 0.0 y 1.0
        return min(1.0, max(0.0, intensity_drop))
        
    except Exception as e:
        return 0.5

def calculate_rolling_volatility_3_games(team_history_df):
    """
    🎯 FEATURE 3: Calcula volatilidad en últimos 3 partidos
    
    ¿Cómo funciona?
    1. Toma points scored de últimos 3 partidos
    2. Calcula coeficiente de variación (CV = std/mean)
    3. Volatilidad alta = equipo impredecible
    4. Volatilidad baja = equipo consistente
    
    ¿Qué retorna?
    - 0.0 = Muy consistente (predictible)
    - 0.5 = Volatilidad normal
    - 1.0 = Muy volátil (impredecible)
    """
    try:
        if team_history_df.empty or len(team_history_df) < 3:
            return 0.5  # Volatilidad neutral si no hay datos
        
        # Obtener últimos 3 partidos
        last_3_games = team_history_df.tail(3)
        
        # ANÁLISIS 1: Points Scored Volatility
        points_volatility = 0.5
        if 'points_scored' in last_3_games.columns:
            points = last_3_games['points_scored'].dropna()
            
            if len(points) == 3:
                mean_points = points.mean()
                std_points = points.std()
                
                if mean_points > 0:
                    cv_points = std_points / mean_points
                    # Normalizar CV a escala 0-1
                    # CV típico para NBA: 0.05-0.20
                    points_volatility = min(1.0, cv_points / 0.20)
        
        # ANÁLISIS 2: Total Score Volatility (si disponible)
        total_volatility = 0.5
        if 'total_score' in last_3_games.columns:
            totals = last_3_games['total_score'].dropna()
            
            if len(totals) == 3:
                mean_total = totals.mean()
                std_total = totals.std()
                
                if mean_total > 0:
                    cv_total = std_total / mean_total
                    # CV típico para totales: 0.03-0.15
                    total_volatility = min(1.0, cv_total / 0.15)
        
        # ANÁLISIS 3: Plus/Minus Volatility
        pm_volatility = 0.5
        if 'plus_minus' in last_3_games.columns:
            plus_minus = last_3_games['plus_minus'].dropna()
            
            if len(plus_minus) == 3:
                # Para plus/minus, usamos rango en lugar de CV
                pm_range = plus_minus.max() - plus_minus.min()
                # Rango típico: 5-40 puntos
                pm_volatility = min(1.0, pm_range / 40)
        
        # COMBINAR ANÁLISIS
        # Priorizar points scored, luego total, luego plus/minus
        if 'points_scored' in last_3_games.columns and not last_3_games['points_scored'].isna().all():
            volatility = points_volatility * 0.7 + total_volatility * 0.2 + pm_volatility * 0.1
        elif 'total_score' in last_3_games.columns and not last_3_games['total_score'].isna().all():
            volatility = total_volatility * 0.8 + pm_volatility * 0.2
        else:
            volatility = pm_volatility
        
        # ANÁLISIS 4: Bonus - Shooting Volatility
        shooting_volatility = 0.0
        shooting_stats = ['field_goals_made', 'field_goals_attempted']
        
        if all(stat in last_3_games.columns for stat in shooting_stats):
            fg_made = last_3_games['field_goals_made'].dropna()
            fg_att = last_3_games['field_goals_attempted'].dropna()
            
            if len(fg_made) == 3 and len(fg_att) == 3 and (fg_att > 0).all():
                fg_percentages = fg_made / fg_att
                fg_cv = fg_percentages.std() / max(fg_percentages.mean(), 0.01)
                shooting_volatility = min(1.0, fg_cv / 0.30)  # CV típico FG%: 0.1-0.3
                
                # Incorporar shooting volatility como bonus
                volatility = volatility * 0.9 + shooting_volatility * 0.1
        
        return min(1.0, max(0.0, volatility))
        
    except Exception as e:
        return 0.5
    
def calculate_momentum_acceleration(team_history_df):
    """
    🎯 FEATURE 4: Detecta aceleración/desaceleración del momentum
    
    ¿Cómo funciona?
    1. Analiza scoring trend últimos 5 partidos
    2. Calcula si están acelerando (mejorando) o desacelerando
    3. Considera shooting efficiency trend también
    
    ¿Qué retorna?
    - 0.0 = Desacelerando fuertemente (momentum bajando)
    - 0.5 = Momentum estable
    - 1.0 = Acelerando fuertemente (momentum subiendo)
    """
    try:
        if team_history_df.empty or len(team_history_df) < 4:
            return 0.5  # Neutral si no hay suficientes datos
        
        # Obtener últimos 5 partidos para análisis de tendencia
        recent_games = team_history_df.tail(5)
        
        # ANÁLISIS 1: Scoring Acceleration
        scoring_acceleration = 0.5
        if 'points_scored' in recent_games.columns:
            points = recent_games['points_scored'].dropna()
            
            if len(points) >= 4:
                # Dividir en dos mitades para ver tendencia
                first_half = points.head(len(points)//2).mean()
                second_half = points.tail(len(points)//2).mean()
                
                if first_half > 0:
                    # Calcular aceleración porcentual
                    acceleration = (second_half - first_half) / first_half
                    # Normalizar: -20% a +20% = 0.0 a 1.0
                    scoring_acceleration = max(0.0, min(1.0, (acceleration + 0.2) / 0.4))
        
        # ANÁLISIS 2: Shooting Efficiency Acceleration
        shooting_acceleration = 0.5
        shooting_stats = ['field_goals_made', 'field_goals_attempted']
        
        if all(stat in recent_games.columns for stat in shooting_stats):
            fg_made = recent_games['field_goals_made'].dropna()
            fg_att = recent_games['field_goals_attempted'].dropna()
            
            if len(fg_made) >= 4 and len(fg_att) >= 4 and (fg_att > 0).all():
                fg_percentages = fg_made / fg_att
                
                # Calcular tendencia en shooting percentage
                first_half_pct = fg_percentages.head(len(fg_percentages)//2).mean()
                second_half_pct = fg_percentages.tail(len(fg_percentages)//2).mean()
                
                if first_half_pct > 0:
                    shooting_change = (second_half_pct - first_half_pct) / first_half_pct
                    # Normalizar: -15% a +15% = 0.0 a 1.0
                    shooting_acceleration = max(0.0, min(1.0, (shooting_change + 0.15) / 0.30))
        
        # ANÁLISIS 3: Win Rate Acceleration (si disponible)
        win_acceleration = 0.5
        if 'win' in recent_games.columns:
            wins = recent_games['win'].dropna()
            
            if len(wins) >= 4:
                first_half_wins = wins.head(len(wins)//2).mean()
                second_half_wins = wins.tail(len(wins)//2).mean()
                
                # Diferencia directa en win rate
                win_rate_change = second_half_wins - first_half_wins
                # Normalizar: -1.0 a +1.0 = 0.0 a 1.0
                win_acceleration = max(0.0, min(1.0, (win_rate_change + 1.0) / 2.0))
        
        # ANÁLISIS 4: Plus/Minus Acceleration
        pm_acceleration = 0.5
        if 'plus_minus' in recent_games.columns:
            plus_minus = recent_games['plus_minus'].dropna()
            
            if len(plus_minus) >= 4:
                first_half_pm = plus_minus.head(len(plus_minus)//2).mean()
                second_half_pm = plus_minus.tail(len(plus_minus)//2).mean()
                
                # Cambio en plus/minus
                pm_change = second_half_pm - first_half_pm
                # Normalizar: -30 a +30 = 0.0 a 1.0
                pm_acceleration = max(0.0, min(1.0, (pm_change + 30) / 60))
        
        # COMBINAR ANÁLISIS con pesos
        momentum_acceleration = (
            scoring_acceleration * 0.4 +      # Más peso a scoring
            shooting_acceleration * 0.25 +    # Shooting efficiency importante
            win_acceleration * 0.2 +          # Wins matter
            pm_acceleration * 0.15            # Plus/minus como contexto
        )
        
        return max(0.0, min(1.0, momentum_acceleration))
        
    except Exception as e:
        return 0.5

def calculate_pace_differential_trend(team_history_df):
    """
    🎯 FEATURE 5: Analiza trend en differential de pace del equipo
    
    ¿Cómo funciona?
    1. Calcula pace real de cada partido usando FGA + TO
    2. Analiza si el equipo está jugando más rápido/lento últimamente
    3. Considera impact en totales (pace alto = más puntos)
    
    ¿Qué retorna?
    - 0.0 = Pace bajando (menos possessions, menos puntos)
    - 0.5 = Pace estable  
    - 1.0 = Pace subiendo (más possessions, más puntos)
    """
    try:
        if team_history_df.empty or len(team_history_df) < 4:
            return 0.5  # Neutral si no hay suficientes datos
        
        # Obtener últimos 5 partidos
        recent_games = team_history_df.tail(5)
        
        # ANÁLISIS 1: Pace Real Calculation
        pace_values = []
        
        # Método preferido: usar FGA + TO para calcular possessions
        if all(col in recent_games.columns for col in ['field_goals_attempted', 'turnovers']):
            for _, game in recent_games.iterrows():
                try:
                    fga = game.get('field_goals_attempted', 0)
                    to = game.get('turnovers', 0)
                    
                    # Estimación de possessions del equipo
                    team_possessions = fga + to
                    
                    # Estimar possessions del oponente (asumiendo similar)
                    # Para pace real necesitaríamos datos del oponente, pero podemos estimar
                    estimated_total_possessions = team_possessions * 2
                    
                    # Pace = possessions por 48 minutos (normalizado)
                    pace = (estimated_total_possessions / 48) * 48  # Ya está en 48 min
                    
                    if pace > 0:
                        pace_values.append(pace)
                        
                except:
                    continue
        
        # Método fallback: usar total_score como proxy
        elif 'total_score' in recent_games.columns:
            total_scores = recent_games['total_score'].dropna()
            for total in total_scores:
                if total > 0:
                    # Conversión aproximada: total score ≈ pace × 2.1
                    estimated_pace = total / 2.1
                    pace_values.append(estimated_pace)
        
        if len(pace_values) < 4:
            return 0.5  # No suficientes datos para trend
        
        # ANÁLISIS 2: Pace Trend Calculation
        pace_trend = 0.5
        
        # Dividir en primera y segunda mitad para ver tendencia
        first_half_pace = np.mean(pace_values[:len(pace_values)//2])
        second_half_pace = np.mean(pace_values[len(pace_values)//2:])
        
        if first_half_pace > 0:
            pace_change = (second_half_pace - first_half_pace) / first_half_pace
            
            # Normalizar cambio de pace: -10% a +10% = 0.0 a 1.0
            pace_trend = max(0.0, min(1.0, (pace_change + 0.1) / 0.2))
        
        # ANÁLISIS 3: Shooting Tempo (bonus analysis)
        tempo_adjustment = 0.0
        
        if 'field_goals_attempted' in recent_games.columns:
            fga_values = recent_games['field_goals_attempted'].dropna()
            
            if len(fga_values) >= 4:
                first_half_fga = fga_values.head(len(fga_values)//2).mean()
                second_half_fga = fga_values.tail(len(fga_values)//2).mean()
                
                if first_half_fga > 0:
                    fga_change = (second_half_fga - first_half_fga) / first_half_fga
                    # FGA trend como ajuste menor
                    tempo_adjustment = max(-0.1, min(0.1, fga_change))
        
        # ANÁLISIS 4: Turnover Rate Impact
        to_impact = 0.0
        
        if 'turnovers' in recent_games.columns:
            to_values = recent_games['turnovers'].dropna()
            
            if len(to_values) >= 4:
                first_half_to = to_values.head(len(to_values)//2).mean()
                second_half_to = to_values.tail(len(to_values)//2).mean()
                
                # Más turnovers = pace más rápido (más possessions)
                if first_half_to > 0:
                    to_change = (second_half_to - first_half_to) / first_half_to
                    to_impact = max(-0.05, min(0.05, to_change * 0.5))
        
        # COMBINAR ANÁLISIS
        final_pace_trend = pace_trend + tempo_adjustment + to_impact
        
        return max(0.0, min(1.0, final_pace_trend))
        
    except Exception as e:
        return 0.5

def calculate_clutch_time_performance(team_history_df):
    """
    🎯 FEATURE 6: Analiza performance en clutch time (situaciones cerradas)
    
    ¿Cómo funciona?
    1. Identifica partidos donde entraron al Q4 con <10 pts diferencia
    2. Analiza cómo rindieron en esos Q4 clutch vs su promedio normal
    3. Factor clave para Over/Under en juegos cerrados
    
    ¿Qué retorna?
    - 0.0 = Mal en clutch (underperform en juegos cerrados)
    - 0.5 = Performance normal en clutch
    - 1.0 = Excelente en clutch (overperform en juegos cerrados)
    """
    try:
        if team_history_df.empty or len(team_history_df) < 3:
            return 0.5  # Neutral si no hay suficientes datos
        
        # Verificar que tenemos datos de cuartos
        quarter_cols = ['q1_points', 'q2_points', 'q3_points', 'q4_points']
        available_quarters = [col for col in quarter_cols if col in team_history_df.columns]
        
        if len(available_quarters) < 4:
            return 0.5  # No hay datos de cuartos suficientes
        
        clutch_performances = []
        normal_q4_performances = []
        
        # Analizar cada partido en el historial
        for _, game in team_history_df.iterrows():
            try:
                q1_pts = game.get('q1_points', 0)
                q2_pts = game.get('q2_points', 0) 
                q3_pts = game.get('q3_points', 0)
                q4_pts = game.get('q4_points', 0)
                
                # Calcular situación entrando al Q4
                q3_total_team = q1_pts + q2_pts + q3_pts
                
                # Necesitamos estimar puntos del oponente para determinar si fue clutch
                # Método 1: usar plus_minus si está disponible
                if 'plus_minus' in game.index and not pd.isna(game['plus_minus']):
                    total_team_points = game.get('points_scored', q1_pts + q2_pts + q3_pts + q4_pts)
                    plus_minus = game['plus_minus']
                    opponent_total = total_team_points - plus_minus
                    opponent_q3 = opponent_total - q4_pts  # Estimar oponente en Q3
                    
                    q3_differential = abs(q3_total_team - opponent_q3)
                    
                elif 'points_allowed' in game.index and not pd.isna(game['points_allowed']):
                    # Método 2: usar points_allowed
                    opponent_total = game['points_allowed']
                    opponent_q4 = opponent_total - (opponent_total * 0.75)  # Estimar Q4 oponente
                    opponent_q3 = opponent_total - opponent_q4
                    
                    q3_differential = abs(q3_total_team - opponent_q3)
                    
                else:
                    # Método 3: asumir distribución típica de cuartos
                    # Q4 típicamente es 25% del total
                    estimated_total_game = (q1_pts + q2_pts + q3_pts + q4_pts) * 2
                    estimated_q3_total = estimated_total_game * 0.75
                    estimated_opponent_q3 = estimated_total_game - q3_total_team
                    
                    q3_differential = abs(q3_total_team - estimated_opponent_q3)
                
                # Determinar si fue situación clutch (diferencia <10 pts entrando Q4)
                if q3_differential < 10 and q4_pts > 0:
                    clutch_performances.append(q4_pts)
                elif q4_pts > 0:
                    normal_q4_performances.append(q4_pts)
                    
            except Exception as e:
                continue
        
        # ANÁLISIS: Comparar performance clutch vs normal
        if len(clutch_performances) >= 2 and len(normal_q4_performances) >= 2:
            clutch_avg = np.mean(clutch_performances)
            normal_avg = np.mean(normal_q4_performances)
            
            if normal_avg > 0:
                # Ratio de performance en clutch vs normal
                clutch_ratio = clutch_avg / normal_avg
                
                # Normalizar: 0.7 a 1.3 ratio = 0.0 a 1.0
                clutch_performance = max(0.0, min(1.0, (clutch_ratio - 0.7) / 0.6))
                
                return clutch_performance
        
        # FALLBACK: Si no hay suficientes datos clutch, usar Q4 consistency
        elif len(normal_q4_performances) >= 3:
            q4_values = normal_q4_performances
            q4_avg = np.mean(q4_values)
            q4_std = np.std(q4_values)
            
            # Consistency como proxy de clutch performance
            if q4_avg > 0:
                cv = q4_std / q4_avg
                # Menos variabilidad = mejor clutch performer
                clutch_proxy = max(0.0, min(1.0, 1 - (cv / 0.4)))
                return clutch_proxy
        
        return 0.5  # Default neutral
        
    except Exception as e:
        return 0.5

def calculate_enhanced_shooting_rhythm(team_history_df):
    """
    🎯 FEATURE 7: Shooting rhythm MEJORADO (reemplaza la versión existente)
    
    ¿Cómo funciona?
    1. Analiza consistency en FG% últimos 5 partidos
    2. Analiza consistency en 3P% últimos 5 partidos  
    3. Detecta hot/cold streaks
    4. Considera volume de tiros (más tiros = más confiable)
    
    ¿Qué retorna?
    - 0.0 = Shooting muy inconsistente/frío
    - 0.5 = Shooting rhythm normal
    - 1.0 = Shooting muy consistente/caliente
    """
    try:
        if team_history_df.empty or len(team_history_df) < 3:
            return 0.5  # Neutral si no hay suficientes datos
        
        recent_games = team_history_df.tail(5)
        
        # ANÁLISIS 1: Field Goal Consistency
        fg_rhythm = 0.5
        fg_stats = ['field_goals_made', 'field_goals_attempted']
        
        if all(stat in recent_games.columns for stat in fg_stats):
            fg_made = recent_games['field_goals_made'].dropna()
            fg_att = recent_games['field_goals_attempted'].dropna()
            
            if len(fg_made) >= 3 and len(fg_att) >= 3 and (fg_att > 0).all():
                fg_percentages = fg_made / fg_att
                
                # Consistency (menor variabilidad = mejor rhythm)
                fg_mean = fg_percentages.mean()
                fg_std = fg_percentages.std()
                
                if fg_mean > 0:
                    cv = fg_std / fg_mean
                    # Normalizar CV: 0.4 a 0.1 = 0.0 a 1.0 (menos CV = mejor)
                    fg_consistency = max(0.0, min(1.0, (0.4 - cv) / 0.3))
                    
                    # Bonus por shooting percentage alto
                    fg_level_bonus = max(0.0, min(0.3, (fg_mean - 0.35) / 0.2))
                    
                    fg_rhythm = fg_consistency * 0.7 + fg_level_bonus
        
        # ANÁLISIS 2: Three-Point Consistency  
        three_rhythm = 0.5
        three_stats = ['3_point_field_goals_made', '3_point_field_g_attempted']
        
        if all(stat in recent_games.columns for stat in three_stats):
            three_made = recent_games['3_point_field_goals_made'].dropna()
            three_att = recent_games['3_point_field_g_attempted'].dropna()
            
            if len(three_made) >= 3 and len(three_att) >= 3 and (three_att > 0).all():
                three_percentages = three_made / three_att
                
                three_mean = three_percentages.mean()
                three_std = three_percentages.std()
                
                if three_mean > 0:
                    cv_three = three_std / three_mean
                    three_consistency = max(0.0, min(1.0, (0.5 - cv_three) / 0.4))
                    
                    # Bonus por 3P% alto
                    three_level_bonus = max(0.0, min(0.3, (three_mean - 0.3) / 0.2))
                    
                    three_rhythm = three_consistency * 0.7 + three_level_bonus
        
        # ANÁLISIS 3: Hot/Cold Streak Detection
        streak_factor = 0.5
        
        if all(stat in recent_games.columns for stat in fg_stats):
            fg_made = recent_games['field_goals_made'].dropna()
            fg_att = recent_games['field_goals_attempted'].dropna()
            
            if len(fg_made) >= 3:
                # Últimos 2 vs primeros 2 partidos
                recent_fg_pct = (fg_made.tail(2) / fg_att.tail(2)).mean()
                earlier_fg_pct = (fg_made.head(2) / fg_att.head(2)).mean()
                
                if earlier_fg_pct > 0:
                    streak_change = (recent_fg_pct - earlier_fg_pct) / earlier_fg_pct
                    # Normalizar: -20% a +20% = 0.0 a 1.0
                    streak_factor = max(0.0, min(1.0, (streak_change + 0.2) / 0.4))
        
        # ANÁLISIS 4: Volume Reliability Weight
        volume_weight = 1.0
        
        if 'field_goals_attempted' in recent_games.columns:
            avg_fga = recent_games['field_goals_attempted'].mean()
            
            # Más volume = más confiable la métrica
            if avg_fga < 60:      # Low volume
                volume_weight = 0.8
            elif avg_fga > 85:    # High volume  
                volume_weight = 1.1
        
        # COMBINAR ANÁLISIS
        shooting_rhythm = (
            fg_rhythm * 0.6 +           # FG rhythm más importante
            three_rhythm * 0.25 +       # 3P rhythm significativo
            streak_factor * 0.15        # Momentum reciente
        ) * volume_weight
        
        return max(0.0, min(1.0, shooting_rhythm))
        
    except Exception as e:
        return 0.5

def calculate_turnover_momentum(team_history_df):
    """
    🎯 FEATURE 8: Analiza momentum de turnovers
    
    ¿Cómo funciona?
    1. Analiza trend de turnovers últimos 5 partidos
    2. Menos turnovers = mejor control, más possessions efectivas
    3. Más turnovers = pace más rápido pero menos eficiente
    4. Impact directo en pace y totales del juego
    
    ¿Qué retorna?
    - 0.0 = Turnovers aumentando (perdiendo control)
    - 0.5 = Turnovers estables
    - 1.0 = Turnovers bajando (mejor control)
    """
    try:
        if team_history_df.empty or len(team_history_df) < 3:
            return 0.5  # Neutral si no hay suficientes datos
        
        recent_games = team_history_df.tail(5)
        
        # ANÁLISIS 1: Turnover Trend
        to_trend = 0.5
        
        if 'turnovers' in recent_games.columns:
            turnovers = recent_games['turnovers'].dropna()
            
            if len(turnovers) >= 4:
                # Comparar primera vs segunda mitad
                first_half_to = turnovers.head(len(turnovers)//2).mean()
                second_half_to = turnovers.tail(len(turnovers)//2).mean()
                
                if first_half_to > 0:
                    # Menos turnovers en segunda mitad = mejor (trend positivo)
                    to_change = (first_half_to - second_half_to) / first_half_to
                    # Normalizar: -50% a +50% = 0.0 a 1.0
                    to_trend = max(0.0, min(1.0, (to_change + 0.5) / 1.0))
        
        # ANÁLISIS 2: Turnover Rate vs League Average
        to_rate_factor = 0.5
        
        if 'turnovers' in recent_games.columns and 'field_goals_attempted' in recent_games.columns:
            turnovers = recent_games['turnovers'].dropna()
            fga = recent_games['field_goals_attempted'].dropna()
            
            if len(turnovers) >= 3 and len(fga) >= 3:
                # Calcular turnover rate aproximado
                avg_to = turnovers.mean()
                avg_fga = fga.mean()
                
                # Estimar possessions (FGA + TO aproximadamente)
                estimated_possessions = avg_fga + avg_to
                
                if estimated_possessions > 0:
                    to_rate = avg_to / estimated_possessions
                    
                    # NBA average TO rate ≈ 14%
                    # Mejor que promedio = factor alto
                    if to_rate < 0.12:        # Excelente control
                        to_rate_factor = 0.8
                    elif to_rate < 0.14:      # Buen control
                        to_rate_factor = 0.65
                    elif to_rate < 0.16:      # Promedio
                        to_rate_factor = 0.5
                    elif to_rate < 0.18:      # Por debajo del promedio
                        to_rate_factor = 0.35
                    else:                     # Mal control
                        to_rate_factor = 0.2
        
        # ANÁLISIS 3: Turnover Consistency
        to_consistency = 0.5
        
        if 'turnovers' in recent_games.columns:
            turnovers = recent_games['turnovers'].dropna()
            
            if len(turnovers) >= 3:
                to_mean = turnovers.mean()
                to_std = turnovers.std()
                
                if to_mean > 0:
                    cv = to_std / to_mean
                    # Menos variabilidad = más consistente = mejor
                    to_consistency = max(0.0, min(1.0, (0.6 - cv) / 0.6))
        
        # ANÁLISIS 4: Impact on Pace
        pace_impact = 0.5
        
        if all(col in recent_games.columns for col in ['turnovers', 'total_score']):
            turnovers = recent_games['turnovers'].dropna()
            totals = recent_games['total_score'].dropna()
            
            if len(turnovers) >= 3 and len(totals) >= 3:
                # Correlación entre turnovers y pace (total score como proxy)
                # Más turnovers típicamente = más pace = más puntos
                avg_to = turnovers.mean()
                avg_total = totals.mean()
                
                # Ajustar basado en si los turnovers están ayudando al pace
                if avg_total > 200:  # High scoring games
                    if avg_to > 15:  # High turnovers pero high scoring
                        pace_impact = 0.7  # Turnovers ayudando al pace
                    else:  # Low turnovers, high scoring  
                        pace_impact = 0.8  # Efficient offense
                else:  # Lower scoring games
                    if avg_to > 15:  # High turnovers, low scoring
                        pace_impact = 0.3  # Turnovers hurt efficiency
                    else:  # Low turnovers, low scoring
                        pace_impact = 0.5  # Slow pace game
        
        # COMBINAR ANÁLISIS
        turnover_momentum = (
            to_trend * 0.4 +           # Trend más importante
            to_rate_factor * 0.3 +     # Rate vs league average
            to_consistency * 0.2 +     # Consistency matters
            pace_impact * 0.1          # Context de pace
        )
        
        return max(0.0, min(1.0, turnover_momentum))
        
    except Exception as e:
        return 0.5

def calculate_true_shooting_percentage(team_history_df):
    """
    🎯 True Shooting % - Métrica estándar basketball analytics
    
    TS% = Points / (2 * (FGA + 0.44 * FTA))
    
    Mide eficiencia real de scoring considerando:
    - Field goals (2P + 3P)
    - Free throws
    - Valor extra de triples
    """
    try:
        if team_history_df.empty or len(team_history_df) < 3:
            return np.nan
        
        recent_games = team_history_df.tail(5)
        ts_values = []
        
        for _, game in recent_games.iterrows():
            points = safe_float(game.get('points_scored', 0))
            fga = safe_float(game.get('field_goals_attempted', 0))
            fta = safe_float(game.get('free_throws_attempted', 0))
            
            if fga == 0 and fta == 0: continue

            # Fórmula estándar NBA/WNBA
            true_shooting_attempts = 2 * (fga + 0.44 * fta)

            if true_shooting_attempts > 0:
                ts_percentage = safe_divide(points, true_shooting_attempts, np.nan)
                
                # Validar rango razonable (25%-70%)
                if 0.25 <= ts_percentage <= 0.70:
                    ts_values.append(ts_percentage)
        
        if len(ts_values) >= 2:
            return safe_mean(pd.Series(ts_values))
        else:
            return np.nan
            
    except Exception as e:
        print(f"⚠️ Error calculando True Shooting %: {e}")
        return np.nan

def calculate_shot_selection_intelligence(team_history_df):
    """
    🎯 Shot Selection Intelligence - Aprovecha tus datos 2P/3P separados
    
    Analiza:
    - 3-Point Rate (¿toman buenos triples?)
    - 3-Point Efficiency (¿los hacen?)
    - 2-Point Efficiency (¿son buenos cerca del aro?)
    - Shot Distribution (¿distribución moderna?)
    
    USA TUS DATOS EXACTOS:
    - 2_point_field_goals_made/attempted
    - 3_point_field_goals_made/attempted
    """
    try:
        if team_history_df.empty or len(team_history_df) < 3:
            return np.nan
        
        recent_games = team_history_df.tail(5)
        
        three_rates = []      # % de tiros que son triples
        three_effs = []       # Eficiencia en triples
        two_effs = []         # Eficiencia en dos puntos
        
        for _, game in recent_games.iterrows():
            # Usar TUS datos separados exactos
            total_fga = safe_float(game.get('field_goals_attempted', 1))
            
            # 3-Point stats
            # ✅ SOLUCIÓN: Usar el nombre canónico y limpio que mapeamos antes.
            three_att = safe_float(game.get('3point_field_goals_attempted', 0))
            three_made = safe_float(game.get('3point_field_goals_made', 0))
            
            # 2-Point stats
            # ✅ SOLUCIÓN: Usar el nombre canónico y limpio que mapeamos antes.
            two_att = safe_float(game.get('2point_field_goals_attempted', 0))
            two_made = safe_float(game.get('2point_field_goals_made', 0))
            
            # 3-Point Rate
            if total_fga > 0:
                three_rate = three_att / total_fga
                three_rates.append(three_rate)
            
            # 3-Point Efficiency
            if three_att > 0:
                three_eff = three_made / three_att
                three_effs.append(three_eff)
            
            # 2-Point Efficiency
            if two_att > 0:
                two_eff = two_made / two_att
                two_effs.append(two_eff)
        
        # Calcular scores
        scores = []
        
        # 1. 3-Point Rate Score (óptimo WNBA ~35%)
        if three_rates:
            avg_three_rate = safe_mean(pd.Series(three_rates))
            optimal_rate = 0.35
            rate_score = 1.0 - abs(avg_three_rate - optimal_rate) / optimal_rate
            rate_score = max(0.0, min(1.0, rate_score))
            scores.append(rate_score * 0.3)
        
        # 2. 3-Point Efficiency Score (benchmark 33%)
        if three_effs:
            avg_three_eff = safe_mean(pd.Series(three_effs))
            three_eff_score = min(avg_three_eff / 0.33, 1.0)
            scores.append(three_eff_score * 0.25)
        
        # 3. 2-Point Efficiency Score (benchmark 50%)
        if two_effs:
            avg_two_eff = safe_mean(pd.Series(two_effs))
            two_eff_score = min(avg_two_eff / 0.50, 1.0)
            scores.append(two_eff_score * 0.35)
        
        # 4. Shot Consistency Bonus
        if len(three_effs) >= 3:
            three_consistency = 1 / (1 + np.std(three_effs) / max(np.mean(three_effs), 0.01))
            scores.append(three_consistency * 0.1)
        
        # Combinar scores
        if scores:
            final_score = sum(scores)
            return max(0.0, min(1.0, final_score))
        else:
            return np.nan
            
    except Exception as e:
        print(f"⚠️ Error calculando Shot Selection Intelligence: {e}")
        return np.nan

def calculate_enhanced_schedule_analysis(team_history_df):
    """
    📅 Schedule Analysis Mejorado - Usa tus fechas disponibles
    
    Analiza:
    - Días desde último juego
    - Back-to-back games
    - Schedule density (juegos recientes)
    - Fatigue accumulation
    
    USA TU FORMATO: "16.08.2025 04:00"
    """
    try:
        if team_history_df.empty or 'date' not in team_history_df.columns:
            return np.nan
        
        # ✅ SOLUCIÓN: Eliminar la conversión a string. Trabajar directamente con los objetos datetime
        # que ya vienen procesados desde data_processing.py.
        dates_in_history = team_history_df['date'].dropna().tail(10)

        if len(dates_in_history) < 2:
            return np.nan
        
        # Ordenar fechas (más reciente primero)
        dates_converted = sorted(dates_in_history, reverse=True)
        
        schedule_factors = []
        
        # 1. Rest Analysis (últimos 3 juegos)
        rest_scores = []
        for i in range(1, min(len(dates_converted), 4)):
            # Asegurarse de que los elementos son datetimes
            if isinstance(dates_converted[i-1], datetime) and isinstance(dates_converted[i], datetime):
                days_rest = (dates_converted[i-1] - dates_converted[i]).days
            else:
                continue # Saltar si no son objetos de fecha válidos
            
            # Score basado en días de descanso
            if days_rest <= 1:
                rest_score = 0.2  # Back-to-back (malo)
            elif days_rest == 2:
                rest_score = 0.5  # 1 día descanso (regular)
            elif days_rest == 3:
                rest_score = 0.8  # 2 días (bueno)
            else:
                rest_score = 1.0  # 3+ días (óptimo)
            
            rest_scores.append(rest_score)
        
        if rest_scores:
            avg_rest_score = safe_mean(pd.Series(rest_scores))
            schedule_factors.append(avg_rest_score * 0.4)
        
        # 2. Schedule Density (últimos 7 días)
        if len(dates_converted) >= 3:
            week_ago = dates_converted[0] - timedelta(days=7)
            games_last_week = sum(1 for date in dates_converted if isinstance(date, datetime) and date >= week_ago)
            
            # Score basado en densidad
            if games_last_week <= 2:
                density_score = 1.0  # Light schedule
            elif games_last_week == 3:
                density_score = 0.7  # Normal
            elif games_last_week == 4:
                density_score = 0.4  # Heavy
            else:
                density_score = 0.2  # Very heavy
            
            schedule_factors.append(density_score * 0.35)
        
        # 3. Back-to-Back Detection
        b2b_count = 0
        for i in range(1, min(len(dates_converted), 6)):
             if isinstance(dates_converted[i-1], datetime) and isinstance(dates_converted[i], datetime):
                days_between = (dates_converted[i-1] - dates_converted[i]).days
                if days_between <= 1:
                    b2b_count += 1
        
        b2b_penalty = min(b2b_count * 0.1, 0.3)  # Max 30% penalty
        b2b_score = 1.0 - b2b_penalty
        schedule_factors.append(b2b_score * 0.25)
        
        # Combinar factores
        if schedule_factors:
            final_score = sum(schedule_factors)
            return max(0.0, min(1.0, final_score))
        else:
            return np.nan
            
    except Exception as e:
        # print(f"⚠️ Error calculando Schedule Analysis: {e}")
        return np.nan

def calculate_pace_consistency_tracking(team_history_df):
    """
    ⚡ Pace Consistency Tracking - Complementa tu sistema actual
    
    Analiza:
    - Consistency en possessions por juego
    - Pace trend (¿acelera o desacelera?)
    - Tempo stability
    - Possession efficiency consistency
    
    USA TUS FUNCIONES EXISTENTES: calculate_possessions()
    """
    try:
        if team_history_df.empty or len(team_history_df) < 3:
            return np.nan
        
        recent_games = team_history_df.tail(8)
        
        pace_values = []
        efficiency_values = []
        
        for _, game in recent_games.iterrows():
            # Usar tu función existente calculate_possessions
            try:
                team_stats = {
                    'field_goals_attempted': game.get('field_goals_attempted', 80),
                    'offensive_rebounds': game.get('offensive_rebounds', 10),
                    'turnovers': game.get('turnovers', 14),
                    'free_throws_attempted': game.get('free_throws_attempted', 20)
                }
                
                opponent_stats = {}  # Placeholder para tu función
                
                possessions = calculate_possessions(team_stats, opponent_stats)
                
                if possessions and possessions > 0:
                    pace_values.append(possessions)
                    
                    # Efficiency: puntos por posesión
                    points = safe_float(game.get('points_scored', 0))
                    if possessions > 0:
                        efficiency = safe_divide(points, possessions, 0.0)
                        efficiency_values.append(efficiency)
                        
            except:
                # Fallback si calculate_possessions falla
                total_score = safe_float(game.get('total_score', 160))
                estimated_pace = total_score / 2.1  # Estimación aproximada
                pace_values.append(estimated_pace)
        
        consistency_scores = []
        
        # 1. Pace Consistency
        if len(pace_values) >= 3:
            pace_mean = np.mean(pace_values)
            pace_std = np.std(pace_values)
            
            if pace_mean > 0:
                pace_cv = safe_divide(pace_std, pace_mean, 0.0)
                pace_consistency = 1 / (1 + pace_cv)
                consistency_scores.append(pace_consistency * 0.4)
        
        # 2. Pace Trend Analysis
        if len(pace_values) >= 6:
            recent_3 = pace_values[-3:]
            earlier_3 = pace_values[-6:-3]
            
            recent_avg = np.mean(recent_3)
            earlier_avg = np.mean(earlier_3)
            
            if earlier_avg > 0:
                trend_ratio = safe_divide(recent_avg, earlier_avg, 1.0)

                # Score favorece consistency (ratio cerca de 1.0)
                trend_consistency = 1.0 - abs(trend_ratio - 1.0)
                trend_consistency = max(0.0, min(1.0, trend_consistency))
                consistency_scores.append(trend_consistency * 0.3)
        
        # 3. Efficiency Consistency
        if len(efficiency_values) >= 3:
            eff_mean = np.mean(efficiency_values)
            eff_std = np.std(efficiency_values)
            
            if eff_mean > 0:
                eff_cv = safe_divide(eff_std, eff_mean, 0.0)
                eff_consistency = 1 / (1 + eff_cv)
                consistency_scores.append(eff_consistency * 0.3)
        
        # Combinar scores
        if consistency_scores:
            final_score = sum(consistency_scores)
            return max(0.0, min(1.0, final_score))
        else:
            return np.nan
            
    except Exception as e:
        print(f"⚠️ Error calculando Pace Consistency: {e}")
        return np.nan

