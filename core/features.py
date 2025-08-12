# ===========================================
# Archivo: core/features.py (v2.3) - MEJORAS CORE AL PACE SYSTEM
# Sistema EMA avanzado - Pace proyectado mejorado y ajuste por paliza
# ===========================================
import pandas as pd
import numpy as np
from datetime import datetime
import copy

# 🆕 IMPORTS DESDE CONFIG CENTRALIZADO
from config import (
    MOMENTUM_STATS_COLS, ADVANCED_STATS_COLS, PERFORMANCE_CONTEXT_COLS,
    QUARTER_SPECIFIC_COLS, EMA_RANGES, FEATURES_TO_USE, PRE_GAME_FEATURES
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

def calculate_possessions(team_stats, opponent_stats):
    """Estima las posesiones de un equipo usando la fórmula estándar."""
    fga = team_stats.get('field_goals_attempted', 0)
    orb = team_stats.get('offensive_rebounds', 0)
    tov = team_stats.get('turnovers', 0)
    fta = team_stats.get('free_throws_attempted', 0)
    
    possessions = fga - orb + tov + (0.44 * fta)
    return max(possessions, 1)

def calculate_pace(home_poss, away_poss, minutes_played=48):
    """Calcula el ritmo de juego (posesiones por 48 minutos)."""
    avg_poss = (home_poss + away_poss) / 2
    return (avg_poss / minutes_played) * 48

def calculate_momentum_metrics(team_history_df):
    """Calcula métricas avanzadas de momentum para un equipo."""
    if team_history_df.empty:
        return {
            'win_rate': 0.5,
            'avg_plus_minus': 0.0, 
            'scoring_efficiency': 1.0,
            'defensive_stops': 1.0,
            'clutch_performance': 0.5
        }
    
    metrics = {}
    
    # 🎯 Win Rate (tasa de victorias) - CORREGIDO
    if 'win' in team_history_df.columns and not team_history_df['win'].isna().all():
        metrics['win_rate'] = safe_mean(team_history_df['win'])
    elif 'points_scored' in team_history_df.columns and 'points_allowed' in team_history_df.columns:
        # Calcular win si no existe
        wins = (team_history_df['points_scored'] > team_history_df['points_allowed']).astype(int)
        metrics['win_rate'] = safe_mean(wins)
    else:
        metrics['win_rate'] = 0.5
        
    # 📊 Average Plus/Minus - CORREGIDO
    if 'plus_minus' in team_history_df.columns and not team_history_df['plus_minus'].isna().all():
        metrics['avg_plus_minus'] = safe_mean(team_history_df['plus_minus'])
    elif 'points_scored' in team_history_df.columns and 'points_allowed' in team_history_df.columns:
        # Calcular plus_minus si no existe
        plus_minus = team_history_df['points_scored'] - team_history_df['points_allowed']
        metrics['avg_plus_minus'] = safe_mean(plus_minus)
    else:
        metrics['avg_plus_minus'] = 0.0
    
    # 🔥 Scoring Efficiency Trend
    if 'points_scored' in team_history_df.columns:
        recent_avg = safe_mean(team_history_df['points_scored'].tail(3))
        season_avg = safe_mean(team_history_df['points_scored'])
        metrics['scoring_efficiency'] = recent_avg / max(season_avg, 1)
    else:
        metrics['scoring_efficiency'] = 1.0
    
    # 🛡️ Defensive Stops (basado en puntos permitidos)
    if 'points_allowed' in team_history_df.columns:
        recent_def = safe_mean(team_history_df['points_allowed'].tail(3))
        season_def = safe_mean(team_history_df['points_allowed'])
        metrics['defensive_stops'] = max(season_def, 1) / max(recent_def, 1)
    else:
        metrics['defensive_stops'] = 1.0
    
    # 🎪 Clutch Performance (consistencia en victorias ajustadas)
    if len(team_history_df) >= 5:
        close_games = team_history_df[abs(team_history_df.get('plus_minus', 0)) <= 10]
        if len(close_games) > 0:
            metrics['clutch_performance'] = safe_mean(close_games.get('win', pd.Series([0.5])))
        else:
            metrics['clutch_performance'] = metrics['win_rate']
    else:
        metrics['clutch_performance'] = 0.5
    
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
    quarters_played = 2 if quarter_stage == 'halftime' else 3
    minutes_played = quarters_played * 12
    
    total_points = sum(q_scores.values())
    
    # Estimación de pace más sofisticada
    estimated_possessions = total_points / 1.08  # Ajuste más preciso
    live_pace_estimate = (estimated_possessions / minutes_played) * 48 if minutes_played > 0 else 0
    
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
    🎯 CALCULA TENDENCIAS HISTÓRICAS POR CUARTO DE UN EQUIPO.
    Retorna un factor de ajuste que indica si el equipo tiende a acelerar
    o desacelerar en las segundas mitades.
    """
    if team_history_df is None or team_history_df.empty:
        return {'trend_factor': 1.0} # Factor neutro

    # Columnas de puntos por cuarto requeridas
    q_cols = ['q1_points', 'q2_points', 'q3_points', 'q4_points']
    
    # Verificar si las columnas existen y tienen datos válidos
    if not all(col in team_history_df.columns for col in q_cols):
        return {'trend_factor': 1.0}

    # Calcular promedios por cuarto, manejando NaNs
    q1_avg = safe_mean(team_history_df['q1_points'])
    q2_avg = safe_mean(team_history_df['q2_points'])
    q3_avg = safe_mean(team_history_df['q3_points'])
    q4_avg = safe_mean(team_history_df['q4_points'])

    # Si algún promedio es NaN (por falta de datos), no se puede calcular la tendencia
    if any(pd.isna(avg) for avg in [q1_avg, q2_avg, q3_avg, q4_avg]):
        return {'trend_factor': 1.0}

    # Calcular promedio de primera y segunda mitad
    first_half_avg = q1_avg + q2_avg
    second_half_avg = q3_avg + q4_avg

    # Evitar división por cero
    if first_half_avg == 0:
        return {'trend_factor': 1.0}

    # Calcular factor de tendencia
    trend_factor = second_half_avg / first_half_avg
    
    return {'trend_factor': trend_factor}


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

    # Limitar el ajuste a un máximo de ±15% para evitar sobreajustes extremos
    clamped_trend_factor = min(max(avg_trend_factor, 0.85), 1.15)
    
    # La proyección base ya está calculada en `current_pace` (a 48 min)
    # Se aplica el factor de ajuste a esta proyección
    adjusted_projection = current_pace * clamped_trend_factor
    
    return adjusted_projection


def apply_blowout_adjustment(prediction, score_diff, time_remaining_pct):
    """
    🛡️ AJUSTE POR PALIZA (BLOWOUT).
    Reduce la predicción de puntos totales en situaciones de "garbage time"
    cuando un partido está decidido.
    """
    # Aplicar ajuste solo si la diferencia es grande y queda poco tiempo (ej: último cuarto)
    if score_diff > 20 and time_remaining_pct <= 0.25: # >20 pts de diferencia, <= 25% de partido restante
        # El factor de reducción aumenta con la diferencia, pero está limitado a un máximo del 8%
        reduction_factor = min(score_diff / 100, 0.08)
        
        # Aplicar la reducción a la predicción original
        return prediction * (1 - reduction_factor)
    
    # Si no se cumplen las condiciones de paliza, devolver la predicción sin cambios
    return prediction

# ===================================================================
# ===================================================================
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
        home_totals = impute_missing_stats(home_totals, home_team_name, all_matches, league_averages)
        away_totals = impute_missing_stats(away_totals, away_team_name, all_matches, league_averages)

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
    🎯 CALCULA BALANCE FEATURES REALES
    """
    home_points = sum(q_scores.get(f'q{i}_home', 0) for i in range(1, 4))
    away_points = sum(q_scores.get(f'q{i}_away', 0) for i in range(1, 4))
    current_lead = abs(home_points - away_points)
    
    quarters_played = 2 if quarter_stage == 'halftime' else 3
    
    balance_score = 1 / (1 + (current_lead / (10 * quarters_played)))
    
    lead_threshold = 12 if quarter_stage == 'halftime' else 18
    is_unbalanced = 1 if current_lead > lead_threshold else 0
    
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
    
    blowout_momentum = 0.0
    if is_unbalanced:
        if quarter_stage == 'halftime':
            q1_diff = q_scores.get('q1_home', 0) - q_scores.get('q1_away', 0)
            q2_diff = q_scores.get('q2_home', 0) - q_scores.get('q2_away', 0)
            if np.sign(q2_diff) == np.sign(q1_diff) and abs(q2_diff) > abs(q1_diff):
                 blowout_momentum = (abs(q2_diff) - abs(q1_diff)) / 10
        elif quarter_stage == 'q3_end':
            q2_total_diff = (q_scores.get('q1_home', 0) + q_scores.get('q2_home', 0)) - (q_scores.get('q1_away', 0) + q_scores.get('q2_away', 0))
            q3_total_diff = home_points - away_points
            if np.sign(q3_total_diff) == np.sign(q2_total_diff) and abs(q3_total_diff) > abs(q2_total_diff):
                 blowout_momentum = (abs(q3_total_diff) - abs(q2_total_diff)) / 10

    expected_q4_drop = (current_lead / 100) * is_unbalanced

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

    return {
        'game_balance_score': balance_score,
        'is_game_unbalanced': is_unbalanced,
        'intensity_drop_factor': intensity_drop_factor,
        'blowout_momentum': blowout_momentum,
        'expected_q4_drop': expected_q4_drop,
        'lead_stability': lead_stability
    }

def calculate_features(league_data_raw):
    """
    🎯 FUNCIÓN PRINCIPAL - Calcula características enriquecidas con EMA profesional
    """
    processed_matches, league_averages = process_raw_matches(league_data_raw)
    
    if not processed_matches:
        return pd.DataFrame()

    df = pd.DataFrame(processed_matches).sort_values(by='date').reset_index(drop=True)
    print(f"📊 Partidos procesados: {len(df)}")

    all_features = []
    cols_to_avg = ['points_scored', 'points_allowed', 'total_score'] + ADVANCED_STATS_COLS
    cols_to_ema = MOMENTUM_STATS_COLS

    print("🚀 Calculando features con sistema EMA avanzado...")
    
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
        
        # 🏠 CREAR DATAFRAME DE HISTORIAL PARA EQUIPO LOCAL
        # 🛠️ AJUSTE: Añadir datos por cuarto al historial para calcular tendencias
        home_history_list = []
        for _, r in home_history.iterrows():
            is_home = r['home_team'] == home_team
            team_hist_data = {
                'points_scored': r['home_score'] if is_home else r['away_score'],
                'points_allowed': r['away_score'] if is_home else r['home_score'],
                'total_score': r['total_score'],
                'q1_points': r.get('home_q1_points' if is_home else 'away_q1_points', 0),
                'q2_points': r.get('home_q2_points' if is_home else 'away_q2_points', 0),
                'q3_points': r.get('home_q3_points' if is_home else 'away_q3_points', 0),
                'q4_points': r.get('home_q4_points' if is_home else 'away_q4_points', 0),
                **{stat: r.get(f'{"home" if is_home else "away"}_{stat}', np.nan) for stat in ADVANCED_STATS_COLS}
            }
            home_history_list.append(team_hist_data)
        home_history_df = pd.DataFrame(home_history_list)
        
        if not home_history_df.empty:
            home_history_df['win'] = (home_history_df['points_scored'] > home_history_df['points_allowed']).astype(int)
            home_history_df['plus_minus'] = home_history_df['points_scored'] - home_history_df['points_allowed']
            momentum_metrics = calculate_momentum_metrics(home_history_df)
            for metric, value in momentum_metrics.items():
                home_history_df[metric] = value
            context_metrics = calculate_performance_context(home_history_df, is_home_team=True)
            for metric, value in context_metrics.items():
                home_history_df[metric] = value

        # 🚗 CREAR DATAFRAME DE HISTORIAL PARA EQUIPO VISITANTE
        # 🛠️ AJUSTE: Añadir datos por cuarto al historial para calcular tendencias
        away_history_list = []
        for _, r in away_history.iterrows():
            is_home = r['home_team'] == away_team # Note: logic is inverted here
            team_hist_data = {
                'points_scored': r['away_score'] if not is_home else r['home_score'],
                'points_allowed': r['home_score'] if not is_home else r['away_score'],
                'total_score': r['total_score'],
                'q1_points': r.get('away_q1_points' if not is_home else 'home_q1_points', 0),
                'q2_points': r.get('away_q2_points' if not is_home else 'home_q2_points', 0),
                'q3_points': r.get('away_q3_points' if not is_home else 'home_q3_points', 0),
                'q4_points': r.get('away_q4_points' if not is_home else 'home_q4_points', 0),
                **{stat: r.get(f'{"away" if not is_home else "home"}_{stat}', np.nan) for stat in ADVANCED_STATS_COLS}
            }
            away_history_list.append(team_hist_data)
        away_history_df = pd.DataFrame(away_history_list)
        
        if not away_history_df.empty:
            away_history_df['win'] = (away_history_df['points_scored'] > away_history_df['points_allowed']).astype(int)
            away_history_df['plus_minus'] = away_history_df['points_scored'] - away_history_df['points_allowed']
            momentum_metrics = calculate_momentum_metrics(away_history_df)
            for metric, value in momentum_metrics.items():
                away_history_df[metric] = value
            context_metrics = calculate_performance_context(away_history_df, is_home_team=False)
            for metric, value in context_metrics.items():
                away_history_df[metric] = value

        home_stats_5 = get_rolling_stats(home_history_df, 5, cols_to_avg)
        away_stats_5 = get_rolling_stats(away_history_df, 5, cols_to_avg)
        
        home_ema_advanced = get_enhanced_ema_stats(home_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)
        away_ema_advanced = get_enhanced_ema_stats(away_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)
        
        home_ema_5 = get_ema_stats(home_history_df, 5, cols_to_ema)
        away_ema_5 = get_ema_stats(away_history_df, 5, cols_to_ema)
        
        pre_game_features = {}
        pre_game_features.update({'home_' + k: v for k, v in home_stats_5.items()})
        pre_game_features.update({'away_' + k: v for k, v in away_stats_5.items()})
        pre_game_features.update({'home_' + k: v for k, v in home_ema_5.items()})
        pre_game_features.update({'away_' + k: v for k, v in away_ema_5.items()})
        pre_game_features.update({'home_' + k: v for k, v in home_ema_advanced.items()})
        pre_game_features.update({'away_' + k: v for k, v in away_ema_advanced.items()})

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

        # Calcular tendencias para pasarlas a las métricas en vivo
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
    
    available_features = [f for f in FEATURES_TO_USE if f in final_df.columns]
    
    print(f"✅ Procesamiento EMA completado!")
    print(f"📊 Total de data points válidos: {len(final_df)}")
    print(f"🎯 Features incluidas: {len(available_features)} características")
    print(f"🚀 Nuevas métricas EMA añadidas:")
    print(f"   - Momentum multirango: {len(EMA_RANGES)} escalas temporales")
    print(f"   - Win Rate & Plus/Minus con EMA profesional")
    print(f"   - Métricas de contexto: Home Advantage, Comeback Ability, Consistency")
    print(f"   - Live metrics avanzadas: Momentum Shift, Quarter Consistency")
    
    return final_df
