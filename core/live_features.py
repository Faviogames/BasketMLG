# ===========================================
# Archivo: core/live_features.py (v1.0 - LIVE FEATURES MODULE)
# Features que requieren datos/estado live, rolling, señales dinámicas
# ✅ LIVE: Rolling del partido actual, señales en tiempo real, runs, momentum, clutch actual
# ===========================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

# IMPORTS DESDE FEATURES PARA FUNCIONES COMPARTIDAS
from core.features import (
    safe_mean, safe_divide, calculate_possessions, calculate_pace,
    calculate_team_quarter_trends, apply_enhanced_pace_projection,
    apply_blowout_adjustment, calculate_clutch_time_performance
)

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

def calculate_live_interaction_features(live_pace_metrics, balance_features, team_history_df=None):
    """
    🎯 CALCULATE LIVE INTERACTION FEATURES - FASE 1: INTERACTIONS FEATURE ENGINEERING

    Calcula las 4 categorías principales de interacciones para live predictions:
    1. Pace × Efficiency Interactions
    2. Momentum × Stability Interactions
    3. Context × Performance Interactions
    4. Turnover × Pace Interactions

    Args:
        live_pace_metrics: Dict con métricas de pace en vivo
        balance_features: Dict con métricas de balance del partido
        team_history_df: DataFrame opcional con historial del equipo

    Returns:
        Dict con todas las features de interacción calculadas
    """
    try:
        interaction_features = {}

        # ==========================================
        # 1. PACE × EFFICIENCY INTERACTIONS
        # ==========================================

        # A) pace_shooting_interaction = pace_estimate × shooting_efficiency
        pace_estimate = live_pace_metrics.get('enhanced_pace_estimate', 90)
        live_efficiency = (live_pace_metrics.get('live_efficiency_home', 0.5) +
                          live_pace_metrics.get('live_efficiency_away', 0.5)) / 2

        # Calcular shooting efficiency desde historial si disponible
        shooting_efficiency = 1.0
        if team_history_df is not None and not team_history_df.empty:
            recent_games = team_history_df.tail(3)
            if 'field_goals_made' in recent_games.columns and 'field_goals_attempted' in recent_games.columns:
                fg_made = recent_games['field_goals_made'].sum()
                fg_att = recent_games['field_goals_attempted'].sum()
                if fg_att > 0:
                    shooting_efficiency = fg_made / fg_att

        interaction_features['pace_shooting_interaction'] = pace_estimate * shooting_efficiency

        # B) pace × eFG%, pace × TS%, pace × FT%
        # Calcular eFG% desde historial
        efg_pct = 0.5
        ts_pct = 0.5
        ft_pct = 0.75

        if team_history_df is not None and not team_history_df.empty:
            recent_games = team_history_df.tail(3)

            # eFG%
            if all(col in recent_games.columns for col in ['field_goals_made', 'field_goals_attempted', '3point_field_goals_made']):
                fg_made = recent_games['field_goals_made'].sum()
                fg_att = recent_games['field_goals_attempted'].sum()
                three_made = recent_games['3point_field_goals_made'].sum()
                if fg_att > 0:
                    efg_pct = (fg_made + 0.5 * three_made) / fg_att

            # TS%
            ts_values = []
            for _, game in recent_games.iterrows():
                points = safe_float(game.get('points_scored', 0))
                fga = safe_float(game.get('field_goals_attempted', 0))
                fta = safe_float(game.get('free_throws_attempted', 0))

                if fga > 0 or fta > 0:
                    tsa = 2 * (fga + 0.44 * fta)
                    if tsa > 0:
                        ts_values.append(points / tsa)

            if ts_values:
                ts_pct = np.mean(ts_values)

            # FT%
            if 'free_throws_made' in recent_games.columns and 'free_throws_attempted' in recent_games.columns:
                ft_made = recent_games['free_throws_made'].sum()
                ft_att = recent_games['free_throws_attempted'].sum()
                if ft_att > 0:
                    ft_pct = ft_made / ft_att

        interaction_features['pace_efg_interaction'] = pace_estimate * efg_pct
        interaction_features['pace_ts_interaction'] = pace_estimate * ts_pct
        interaction_features['pace_ft_interaction'] = pace_estimate * ft_pct

        # Threshold check: >1.2σ sobre media histórica → +15% over probability
        # (Esta lógica se implementará en el modelo de predicción)

        # ==========================================
        # 2. MOMENTUM × STABILITY INTERACTIONS
        # ==========================================

        # A) momentum_stability_index = momentum_score × (1 - quarter_variance)
        momentum_shift = live_pace_metrics.get('live_momentum_shift', 0)
        quarter_variance = balance_features.get('quarter_variance', 0.5)

        interaction_features['momentum_stability_index'] = momentum_shift * (1 - quarter_variance)

        # B) momentum × intensity_drop, momentum × garbage_time_risk
        intensity_drop = balance_features.get('intensity_drop_factor', 0.5)
        garbage_time_risk = balance_features.get('garbage_time_risk', 0.0)

        interaction_features['momentum_intensity_interaction'] = momentum_shift * intensity_drop
        interaction_features['momentum_garbage_interaction'] = momentum_shift * garbage_time_risk

        # ==========================================
        # 3. CONTEXT × PERFORMANCE INTERACTIONS
        # ==========================================

        # A) clutch_garbage_interaction = clutch_performance × garbage_time_risk
        clutch_performance = 0.5
        if team_history_df is not None and not team_history_df.empty:
            clutch_performance = calculate_clutch_time_performance(team_history_df)

        interaction_features['clutch_garbage_interaction'] = clutch_performance * garbage_time_risk

        # B) clutch × lead_size, clutch × time_remaining
        current_lead = balance_features.get('current_lead', 0)
        # time_remaining_pct se infiere del quarter_stage
        quarter_stage = balance_features.get('quarter_stage', 'q3_end')
        if quarter_stage == 'halftime':
            time_remaining_pct = 0.5
        elif quarter_stage == 'q3_progress':
            time_remaining_pct = 0.25
        else:  # q3_end
            time_remaining_pct = 0.1

        interaction_features['clutch_lead_interaction'] = clutch_performance * current_lead
        interaction_features['clutch_time_interaction'] = clutch_performance * time_remaining_pct

        # ==========================================
        # 4. TURNOVER × PACE INTERACTIONS
        # ==========================================

        # A) turnover_pace_efficiency = turnovers_per_minute × pace_estimate
        # Estimar turnovers por minuto desde historial
        turnovers_per_minute = 0.0
        if team_history_df is not None and not team_history_df.empty:
            recent_to = team_history_df['turnovers'].tail(3).mean() if 'turnovers' in team_history_df.columns else 0
            # Asumiendo ~48 minutos por partido
            turnovers_per_minute = recent_to / 48

        interaction_features['turnover_pace_efficiency'] = turnovers_per_minute * pace_estimate

        # B) TO% × possessions_per_minute, TO × defensive_stops
        possessions_per_minute = pace_estimate / 48  # pace = possessions por 48 min

        to_percentage = 0.0
        if team_history_df is not None and not team_history_df.empty:
            if 'turnovers' in team_history_df.columns and 'field_goals_attempted' in team_history_df.columns:
                recent_to = team_history_df['turnovers'].tail(3).sum()
                recent_fga = team_history_df['field_goals_attempted'].tail(3).sum()
                if recent_fga > 0:
                    to_percentage = recent_to / (recent_fga + recent_to)  # TO% aproximado

        interaction_features['to_pct_possessions_interaction'] = to_percentage * possessions_per_minute

        # Defensive stops desde historial
        defensive_stops = calculate_defensive_stops(team_history_df)

        interaction_features['to_defensive_stops_interaction'] = turnovers_per_minute * defensive_stops

        return interaction_features

    except Exception as e:
        print(f"⚠️ Error calculando live interaction features: {e}")
        # Retornar valores neutros por defecto
        return {
            'pace_shooting_interaction': 90.0,
            'pace_efg_interaction': 45.0,
            'pace_ts_interaction': 45.0,
            'pace_ft_interaction': 67.5,
            'momentum_stability_index': 0.0,
            'momentum_intensity_interaction': 0.0,
            'momentum_garbage_interaction': 0.0,
            'clutch_garbage_interaction': 0.25,
            'clutch_lead_interaction': 0.0,
            'clutch_time_interaction': 0.25,
            'turnover_pace_efficiency': 0.0,
            'to_pct_possessions_interaction': 0.0,
            'to_defensive_stops_interaction': 0.0
        }

def calculate_enhanced_live_features(live_pace_metrics, balance_features, team_history_df, time_info=None):
    """
    🎯 FASE 1: ENHANCED LIVE FEATURES - Only available live, not in pre-game
    Features that depend on current game state and rolling windows.

    Returns 8 new live-specific features for O/U predictions.
    """
    features = {}

    try:
        # 🏀 1. LIVE PACE × LIVE EFFICIENCY (rolling window)
        # Uses current live pace × efficiency in rolling window (last 3-5 games + current quarter)
        live_pace_efficiency = _calculate_live_pace_efficiency_interaction(
            live_pace_metrics, balance_features, team_history_df
        )
        features.update(live_pace_efficiency)

        # 🏀 2. LIVE MOMENTUM × LIVE VARIANCE
        # Momentum changes with real quarter-by-quarter variance
        live_momentum_variance = _calculate_live_momentum_variance_interaction(
            live_pace_metrics, balance_features, team_history_df
        )
        features.update(live_momentum_variance)

        # 🏀 3. CLUTCH-FACTOR MÓVIL (ventana móvil)
        # Clutch performance in last X minutes, not global average
        clutch_mobile = _calculate_mobile_clutch_factor(
            live_pace_metrics, balance_features, team_history_df, time_info
        )
        features.update(clutch_mobile)

        # 🏀 4. LEAD SWING + FREE THROWS RUN
        # Current lead changes + FT clustering detection
        lead_ft_features = _calculate_lead_swing_ft_run(
            live_pace_metrics, balance_features, team_history_df
        )
        features.update(lead_ft_features)

        # 🏀 5. LIVE VOLATILITY ACCELERATION
        # Rolling volatility in scoring patterns during live game
        live_volatility = _calculate_live_volatility_acceleration(
            live_pace_metrics, balance_features, team_history_df
        )
        features.update(live_volatility)

    except Exception as e:
        print(f"⚠️ Error calculating enhanced live features: {e}")
        # Return neutral defaults
        features.update({
            'live_pace_efficiency_interaction': 0.5,
            'live_momentum_variance_index': 0.0,
            'mobile_clutch_factor': 0.5,
            'lead_swing_intensity': 0.0,
            'ft_run_active': False,
            'ft_run_strength': 0.0,
            'ft_clustering_score': 0.0,
            'live_volatility_acceleration': 0.5
        })

    return features

def _calculate_live_pace_efficiency_interaction(live_pace_metrics, balance_features, team_history_df):
    """
    Live Pace × Live Efficiency: Current pace × efficiency in rolling window
    Superior to pre-game because it captures real-time scoring bursts.
    """
    try:
        # Get current live pace
        current_pace = live_pace_metrics.get('live_pace_estimate', 90.0)

        # Calculate live efficiency from recent games + current quarter
        if team_history_df is not None and not team_history_df.empty:
            # Recent scoring efficiency (last 3 games)
            recent_scoring = team_history_df['points_scored'].tail(3).mean() if 'points_scored' in team_history_df.columns else 100.0

            # Current quarter efficiency proxy
            current_efficiency = balance_features.get('live_efficiency_home', 0.5)
            if current_efficiency < 0.3:  # Very low efficiency
                current_efficiency = 0.3

            # Rolling efficiency: blend recent historical + current quarter
            rolling_efficiency = (recent_scoring / 100.0) * 0.7 + current_efficiency * 0.3
        else:
            rolling_efficiency = 0.5

        # Calculate interaction
        pace_efficiency_interaction = current_pace * rolling_efficiency

        # Normalize to 0-1 scale (typical range 45-135)
        normalized_interaction = max(0.0, min(1.0, (pace_efficiency_interaction - 45) / 90))

        return {
            'live_pace_efficiency_interaction': normalized_interaction
        }

    except Exception as e:
        return {'live_pace_efficiency_interaction': 0.5}

def _calculate_live_momentum_variance_interaction(live_pace_metrics, balance_features, team_history_df):
    """
    Live Momentum × Live Variance: Momentum changes with real quarter variance
    Detects explosive games with strong momentum but high variance.
    """
    try:
        # Current momentum shift
        momentum_shift = abs(live_pace_metrics.get('live_momentum_shift', 0.0))

        # Live variance from quarter scores
        quarter_variance = balance_features.get('quarter_variance', 0.0)

        # Normalize variance (typical range 0-100)
        normalized_variance = min(1.0, quarter_variance / 50.0)

        # Interaction: momentum × (1 - variance) - high momentum + low variance = explosive
        momentum_variance_index = momentum_shift * (1 - normalized_variance)

        # Scale to reasonable range
        momentum_variance_index = max(-1.0, min(1.0, momentum_variance_index / 10.0))

        return {
            'live_momentum_variance_index': momentum_variance_index
        }

    except Exception as e:
        return {'live_momentum_variance_index': 0.0}

def _calculate_mobile_clutch_factor(live_pace_metrics, balance_features, team_history_df, time_info=None):
    """
    Clutch-Factor Móvil: Clutch performance in rolling window (last X minutes)
    Not global clutch rating, but recent clutch performance.
    """
    try:
        # Estimate time played for window calculation
        if time_info and 'minutes_played' in time_info:
            minutes_played = time_info['minutes_played']
        else:
            # Estimate from quarter stage
            quarter_stage = balance_features.get('quarter_stage', 'q3_end')
            if quarter_stage == 'halftime':
                minutes_played = 24
            elif quarter_stage == 'q3_progress':
                minutes_played = 30
            else:  # q3_end
                minutes_played = 36

        # Window size based on time played (last 12-18 minutes)
        window_minutes = max(12, min(18, minutes_played * 0.5))

        # Calculate clutch factor from recent close games in history
        if team_history_df is not None and not team_history_df.empty:
            # Find close games in recent history
            close_games = team_history_df[abs(team_history_df.get('plus_minus', 0)) <= 8] if 'plus_minus' in team_history_df.columns else team_history_df.tail(3)

            if len(close_games) >= 2:
                # Clutch performance in close games
                clutch_wins = close_games['win'].mean() if 'win' in close_games.columns else 0.5
                clutch_efficiency = close_games['points_scored'].mean() / max(close_games['points_allowed'].mean(), 1) if 'points_scored' in close_games.columns and 'points_allowed' in close_games.columns else 1.0

                mobile_clutch = (clutch_wins + min(clutch_efficiency, 1.5) - 0.5) / 2
            else:
                mobile_clutch = 0.5
        else:
            mobile_clutch = 0.5

        # Adjust by current game pressure (score difference)
        current_lead = abs(balance_features.get('current_lead', 0))
        pressure_multiplier = 1.0 + (current_lead / 20.0)  # More pressure in close games

        mobile_clutch = min(1.0, mobile_clutch * pressure_multiplier)

        return {
            'mobile_clutch_factor': mobile_clutch
        }

    except Exception as e:
        return {'mobile_clutch_factor': 0.5}

def _calculate_lead_swing_ft_run(live_pace_metrics, balance_features, team_history_df):
    """
    Lead Swing + Free Throws Run: Current lead changes + FT clustering detection
    Detects momentum swings and fouling strategy activation.
    """
    try:
        # Lead swing intensity (how much lead has changed recently)
        current_lead = balance_features.get('current_lead', 0)

        # Estimate lead swing (simplified - in production would use time-series)
        # For now, use quarter consistency as proxy for stability
        quarter_consistency = balance_features.get('quarter_consistency', 0.5)
        lead_stability = balance_features.get('lead_stability', 0.5)

        # Lead swing = 1 - stability (more unstable = more swings)
        lead_swing_intensity = 1.0 - ((quarter_consistency + lead_stability) / 2.0)
        lead_swing_intensity = max(0.0, min(1.0, lead_swing_intensity))

        # Free throws run detection
        ft_run_active = False
        ft_run_strength = 0.0
        ft_clustering_score = 0.0

        # Simple FT run detection based on current lead and game flow
        # In production, this would analyze FT sequences over time
        if abs(current_lead) <= 5:  # Close game
            # Higher chance of FT runs in close games
            ft_clustering_score = 0.6 + (abs(current_lead) / 10.0)

            if ft_clustering_score > 0.7:
                ft_run_active = True
                ft_run_strength = min(1.0, ft_clustering_score - 0.5)
        else:
            # Blowout games have lower FT clustering
            ft_clustering_score = 0.2

        return {
            'lead_swing_intensity': lead_swing_intensity,
            'ft_run_active': ft_run_active,
            'ft_run_strength': ft_run_strength,
            'ft_clustering_score': ft_clustering_score
        }

    except Exception as e:
        return {
            'lead_swing_intensity': 0.0,
            'ft_run_active': False,
            'ft_run_strength': 0.0,
            'ft_clustering_score': 0.0
        }

def _calculate_live_volatility_acceleration(live_pace_metrics, balance_features, team_history_df):
    """
    Live Volatility Acceleration: Rolling volatility in scoring patterns during live game
    Measures how scoring volatility has changed in recent games vs historical baseline.
    """
    try:
        # Calculate current game volatility from quarter scores
        quarter_variance = balance_features.get('quarter_variance', 0.0)

        # Historical volatility from team history
        historical_volatility = 0.5
        if team_history_df is not None and not team_history_df.empty:
            if 'points_scored' in team_history_df.columns:
                recent_scores = team_history_df['points_scored'].tail(5)
                if len(recent_scores) >= 3:
                    cv = recent_scores.std() / max(recent_scores.mean(), 1)
                    historical_volatility = min(1.0, cv / 0.25)  # Normalize CV

        # Acceleration: how much more volatile is current game vs history
        volatility_acceleration = quarter_variance / max(historical_volatility, 0.1)
        volatility_acceleration = max(0.0, min(2.0, volatility_acceleration))  # Cap at 2x

        # Normalize to 0-1 scale
        normalized_acceleration = volatility_acceleration / 2.0

        return {
            'live_volatility_acceleration': normalized_acceleration
        }

    except Exception as e:
        return {'live_volatility_acceleration': 0.5}

def calculate_defensive_stops(team_history_df):
    """
    Helper function to calculate defensive stops metric
    """
    if team_history_df.empty:
        return 1.0

    try:
        if 'points_allowed' in team_history_df.columns:
            recent_defense = team_history_df['points_allowed'].tail(3).mean()
            season_defense = team_history_df['points_allowed'].mean()
            if season_defense > 0:
                return season_defense / recent_defense
    except:
        pass

    return 1.0