# ===========================================
# Archivo: core/live_features.py (v1.0 - LIVE FEATURES MODULE)
# Features que requieren datos/estado live, rolling, señales dinámicas
# ✅ LIVE: Rolling del partido actual, señales en tiempo real, runs, momentum, clutch actual
# ===========================================
from scipy.stats import poisson
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

def calculate_live_pace_metrics(q_scores, quarter_stage, team_trends=None, time_info=None):
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
    
    # 🆕 FIX: Proyectar puntos a 48 minutos para estimación correcta de Pace
    # Si no proyectamos, estimated_possessions será muy bajo (ej. 50 en halftime)
    projected_total_points = total_points * (48.0 / max(minutes_played, 1))

    # Estimación de pace más precisa usando possessions reales (proyectadas)
    # Calcular possessions usando stats disponibles si es posible
    team_stats = {
        'field_goals_attempted': max(80, projected_total_points * 0.8),  # Estimación conservadora
        'offensive_rebounds': max(8, projected_total_points * 0.05),
        'turnovers': max(12, projected_total_points * 0.08),
        'free_throws_attempted': max(16, projected_total_points * 0.12)
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

    # 🚀 FASE 1: Dynamic Split Boundaries
    # 🆕 CORRECCIÓN: Pasar time_info para cálculo preciso de PPM
    adaptive_split = calculate_adaptive_split_detector(q_scores, quarter_stage, team_trends, time_info)
    transition_shock = calculate_quarter_transition_shock(q_scores, quarter_stage, team_trends)

    return {
        'live_pace_estimate': live_pace_estimate,           # Original
        'enhanced_pace_estimate': enhanced_pace,            # 🆕 MEJORADO
        'live_efficiency_home': live_efficiency_home,
        'live_efficiency_away': live_efficiency_away,
        'live_momentum_shift': live_momentum_shift,
        'quarter_consistency': quarter_consistency,
        'comeback_indicator': comeback_indicator,
        # 🚀 FASE 1: Dynamic Split Boundaries
        'adaptive_split_detector': adaptive_split,
        'quarter_transition_shock': transition_shock
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

def calculate_game_balance_features(q_scores, time_info, live_pace_metrics, league_name='NBA'):
    """
    🎯 Wrapper unificado para calcular todas las métricas de balance del juego.
    Integra calculate_real_balance_features y calculate_garbage_time_risk.
    """
    # 1. Determinar quarter_stage si no viene en time_info
    quarter_stage = time_info.get('quarter_stage')
    if not quarter_stage:
        # Lógica de fallback simplificada
        q3_total = q_scores.get('q3_home', 0) + q_scores.get('q3_away', 0)
        q4_total = q_scores.get('q4_home', 0) + q_scores.get('q4_away', 0)
        if q4_total > 0: quarter_stage = 'q4_progress'
        elif q3_total > 0: quarter_stage = 'q3_progress'
        else: quarter_stage = 'halftime'

    # 2. Calcular features base de balance
    balance_features = calculate_real_balance_features(q_scores, quarter_stage)
    # 3. Calcular riesgo de garbage time
    garbage_risk = calculate_garbage_time_risk(
        q_scores, 
        quarter_stage, 
        balance_features.get('lead_stability', 1.0),
        is_potential_blowout=balance_features.get('is_potential_blowout', 0)
    )
    
    balance_features['garbage_time_risk'] = garbage_risk
    
    return balance_features

# ==============================================================================
# 🚀 LEVEL 1 SPECIALISTS (The Pacer & The Runner)
# ==============================================================================

def calculate_pacer_features(current_total, time_info, current_line=None):
    """
    🏃 THE PACER: Proyección basada en Poisson y Ritmo.
    Diferencia con Safe Projection:
    - Safe Projection: Regla simple (Pace * 48) usada como "freno de emergencia".
    - The Pacer: Modelo probabilístico (Poisson) usado como "input" para el Meta-Learner.
    """
    played_seconds = time_info.get('played_seconds', 0)
    if time_info.get('clock_valid') is False:
        raise ValueError('Reloj no valido para Pacer')
    total_seconds = float(time_info.get('total_game_minutes', 48.0)) * 60
    if total_seconds not in (2400, 2880) or not np.isfinite(played_seconds) or not 0 <= played_seconds <= total_seconds:
        raise ValueError('Duracion o tiempo jugado no valido para Pacer')
    try:
        remaining_seconds = total_seconds - played_seconds
        
        if played_seconds < 60:
            return {'pacer_projection': current_total * 4, 'pacer_prob_over': 0.5}

        # 1. Calcular Lambda (Tasa de anotación por segundo)
        scoring_rate = current_total / played_seconds
        
        # 2. Proyectar Puntos Restantes (Lambda * Tiempo Restante)
        expected_remaining = scoring_rate * remaining_seconds
        
        # 3. Proyección Final
        pacer_projection = current_total + expected_remaining
        
        # 4. Probabilidad Poisson (Si tenemos línea)
        prob_over = 0.5
        if current_line:
            # Probabilidad de que (Puntos Actuales + Puntos Restantes) > Línea
            # Es decir: Prob(Puntos Restantes > Línea - Puntos Actuales)
            needed = current_line - current_total
            if needed <= 0:
                prob_over = 0.99 # Ya se cubrió
            else:
                # Survival function (1 - cdf) de Poisson con media = expected_remaining
                prob_over = poisson.sf(needed, expected_remaining)
        
        return {
            'pacer_projection': pacer_projection,
            'pacer_prob_over': prob_over,
            'pacer_scoring_rate': scoring_rate * 60 # PPM
        }
    except Exception as e:
        print(f"[PACER ERROR] {e}")
        return {'pacer_projection': current_total, 'pacer_prob_over': 0.5}

def calculate_runner_flags(q_scores, quarter_stage):
    """
    🏃 THE RUNNER: Detector de Volatilidad y Momentum.
    Detecta si hay un "Run" activo que pueda distorsionar la proyección base.
    """
    try:
        # Detectar aceleración reciente (Q3 vs Q2 vs Q1)
        q1_total = q_scores.get('q1_home', 0) + q_scores.get('q1_away', 0)
        q2_total = q_scores.get('q2_home', 0) + q_scores.get('q2_away', 0)
        q3_total = q_scores.get('q3_home', 0) + q_scores.get('q3_away', 0)
        
        run_active = 0
        volatility_index = 0.0
        
        if quarter_stage in ['q3_progress', 'q3_end']:
            # Si Q3 está siendo mucho más rápido que el promedio de Q1+Q2
            avg_h1 = (q1_total + q2_total) / 2
            if avg_h1 > 0:
                if q3_total > (avg_h1 * 1.3): # 30% más rápido -> RUN ACTIVO
                    run_active = 1
                    volatility_index = (q3_total - avg_h1) / avg_h1
                elif q3_total < (avg_h1 * 0.7): # 30% más lento -> COLD SPELL
                    run_active = -1
                    volatility_index = (q3_total - avg_h1) / avg_h1
                
        return {
            'run_active_flag': run_active,
            'volatility_index': volatility_index
        }
    except Exception:
        return {'run_active_flag': 0, 'volatility_index': 0.0}

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


# ===========================================
# 🚀🚀🚀 FASE 1: LIVE DYNAMIC SPLIT BOUNDARIES 🚀🚀🚀
# ===========================================

def calculate_adaptive_split_detector(q_scores, quarter_stage, team_history_df=None, time_info=None):
    """
    🎯 FASE 1: adaptive_split_detector (CORREGIDO)
    Algoritmo que detecta cambios de momentum reales vs splits fijos
    
    ✅ CORRECCIÓN: Usa tiempo real del partido en lugar de 12 min fijos
    ✅ MEJORA: Umbral mínimo de puntos para evitar ruido estadístico

    Usa change point detection simple basado en puntos por minuto.
    Retorna 1.0 si detecta cambio de momentum, 0.0 si no.
    """
    try:
        # Solo funciona en Q3 para detectar cambios antes del Q4
        if quarter_stage not in ['q3_progress', 'q3_end']:
            return 0.0

        # Necesitamos datos de Q1, Q2, Q3
        required_quarters = ['q1_home', 'q1_away', 'q2_home', 'q2_away', 'q3_home', 'q3_away']
        if not all(k in q_scores for k in required_quarters):
            return 0.0

        q1_total = q_scores['q1_home'] + q_scores['q1_away']
        q2_total = q_scores['q2_home'] + q_scores['q2_away']
        q3_total = q_scores['q3_home'] + q_scores['q3_away']

        # 🆕 UMBRAL MÍNIMO: Evitar ruido estadístico al inicio del cuarto
        MIN_Q3_POINTS_THRESHOLD = 10  # Mínimo 10 puntos para activar detector
        
        if q3_total < MIN_Q3_POINTS_THRESHOLD:
            return 0.0  # Demasiados puntos para ser estadísticamente significativo

        # 🆕 CORRECCIÓN PRINCIPAL: Usar tiempo real para Q3
        minutes_in_q3 = 12.0  # Default
        
        if time_info and time_info.get('time_into_quarter', 0) > 0:
            # Usar tiempo real transcurrido en Q3
            minutes_in_q3 = time_info['time_into_quarter'] / 60  # Convertir segundos a minutos
        elif quarter_stage == 'q3_progress':
            # Estimación conservadora si no hay time_info: 6 min (mitad del cuarto)
            minutes_in_q3 = 6.0

        # 🆕 CÁLCULO CORREGIDO: Usar tiempo real para Q3, 12 min para Q1 y Q2
        q1_ppm = q1_total / 12.0  # Q1 siempre completo
        q2_ppm = q2_total / 12.0  # Q2 siempre completo
        q3_ppm = q3_total / minutes_in_q3  # ← CORREGIDO: tiempo real

        # Detectar cambio significativo entre Q2 y Q3
        if q2_ppm > 0:
            momentum_change = (q3_ppm - q2_ppm) / q2_ppm

            # Threshold para cambio significativo: ±15%
            if abs(momentum_change) > 0.15:
                return 1.0  # Cambio de momentum detectado

        # Fallback: comparar con historial si disponible
        if team_history_df is not None and not team_history_df.empty:
            try:
                # Calcular Q2->Q3 transitions históricos
                transitions = []
                for _, game in team_history_df.tail(10).iterrows():
                    hist_q2 = safe_float(game.get('q2_points', 0))
                    hist_q3 = safe_float(game.get('q3_points', 0))

                    if hist_q2 > 0:
                        hist_change = (hist_q3 - hist_q2) / hist_q2
                        transitions.append(hist_change)

                if transitions:
                    hist_mean = np.mean(transitions)
                    hist_std = np.std(transitions)

                    # Comparar current change con distribución histórica
                    if q2_ppm > 0:
                        current_change = (q3_ppm - q2_ppm) / q2_ppm
                        z_score = abs(current_change - hist_mean) / max(hist_std, 0.01)

                        # Z-score > 2 = cambio significativo
                        if z_score > 2.0:
                            return 1.0

            except Exception:
                pass

        return 0.0

    except Exception as e:
        return 0.0


def calculate_quarter_transition_shock(q_scores, quarter_stage, team_history_df=None):
    """
    🎯 FASE 1: quarter_transition_shock
    Mide "shocks" en transiciones (puntos/minuto cambios bruscos)

    Calcula el máximo cambio de puntos/min en ventanas de 3 minutos al final/inicio de cuarto.
    Retorna score normalizado 0-1.
    """
    try:
        # Solo relevante en transiciones de cuarto
        if quarter_stage not in ['halftime', 'q3_end']:
            return 0.0

        # Para halftime: analizar transición Q2->Q3
        # Para q3_end: analizar transición Q3->Q4 (si tenemos datos parciales)

        if quarter_stage == 'halftime':
            # Analizar cambios en los últimos minutos de Q2 y primeros de Q3
            # Esto requeriría datos por minuto, que no tenemos
            # Fallback: usar cambio total Q2 vs Q3

            required_quarters = ['q2_home', 'q2_away', 'q3_home', 'q3_away']
            if not all(k in q_scores for k in required_quarters):
                return 0.0

            q2_total = q_scores['q2_home'] + q_scores['q2_away']
            q3_total = q_scores['q3_home'] + q_scores['q3_away']

            # Asumir distribución típica: últimos 3 min de Q2 vs primeros 3 min de Q3
            # Estimación simple: si Q3 es mucho más alto/bajo que Q2
            if q2_total > 0:
                transition_ratio = q3_total / q2_total

                # Ratios extremos indican shock
                if transition_ratio > 1.5 or transition_ratio < 0.67:
                    shock_score = min(1.0, abs(transition_ratio - 1.0))
                    return shock_score

        elif quarter_stage == 'q3_end':
            # Similar análisis para Q3->Q4 si tenemos datos parciales de Q4
            # Por ahora, usar lógica similar
            pass

        # Comparar con historial para contextualizar
        if team_history_df is not None and not team_history_df.empty:
            try:
                # Calcular shocks históricos
                historical_shocks = []
                for _, game in team_history_df.tail(10).iterrows():
                    hist_q2 = safe_float(game.get('q2_points', 0))
                    hist_q3 = safe_float(game.get('q3_points', 0))

                    if hist_q2 > 0:
                        hist_ratio = hist_q3 / hist_q2
                        if hist_ratio > 1.5 or hist_ratio < 0.67:
                            hist_shock = min(1.0, abs(hist_ratio - 1.0))
                            historical_shocks.append(hist_shock)

                if historical_shocks:
                    # Si el shock actual es mayor que el percentil 75 histórico
                    current_shock = 0.0  # Calcular del análisis anterior

                    if quarter_stage == 'halftime':
                        q2_total = q_scores.get('q2_home', 0) + q_scores.get('q2_away', 0)
                        q3_total = q_scores.get('q3_home', 0) + q_scores.get('q3_away', 0)

                        if q2_total > 0:
                            transition_ratio = q3_total / q2_total
                            if transition_ratio > 1.5 or transition_ratio < 0.67:
                                current_shock = min(1.0, abs(transition_ratio - 1.0))

                    if current_shock > np.percentile(historical_shocks, 75):
                        return current_shock

            except Exception:
                pass

        return 0.0

    except Exception as e:
        return 0.0
# ===========================================
# 🚀🚀🚀 FASE 2: ADVANCED LIVE METRICS (SVI, RWT, PACE, ORTG, QMD) 🚀🚀🚀
# ===========================================

from typing import Dict, List, Optional

def calculate_svi_by_quarter(quarter_scores: Dict) -> Dict:
    """
    Calcula Scoring Volatility Index (SVI) usando datos por cuarto.
    Mide la consistencia de la anotación.
    """
    home_scores = []
    away_scores = []
    total_scores = []
    
    for q in ['q1', 'q2', 'q3', 'q4']:
        if f'{q}_home' in quarter_scores and f'{q}_away' in quarter_scores:
            home = int(quarter_scores[f'{q}_home'])
            away = int(quarter_scores[f'{q}_away'])
            home_scores.append(home)
            away_scores.append(away)
            total_scores.append(home + away)
    
    def compute_svi(scores: List[int]) -> Dict:
        if len(scores) < 2:
            return {'svi': 0.0, 'class': 'INSUFFICIENT_DATA'}
        
        mean_ppq = np.mean(scores)
        std_ppq = np.std(scores)
        
        # Coeficiente de variación
        svi = std_ppq / max(mean_ppq, 1)
        
        # Clasificación
        if svi < 0.10:
            volatility_class = 'VERY_STABLE'
        elif svi < 0.15:
            volatility_class = 'STABLE'
        elif svi < 0.25:
            volatility_class = 'MODERATE'
        elif svi < 0.35:
            volatility_class = 'VOLATILE'
        else:
            volatility_class = 'CHAOTIC'
        
        return {
            'svi': round(svi, 4),
            'class': volatility_class
        }
    
    home_svi = compute_svi(home_scores)
    away_svi = compute_svi(away_scores)
    total_svi = compute_svi(total_scores)
    
    return {
        'svi_home': home_svi['svi'],
        'svi_away': away_svi['svi'],
        'svi_total': total_svi['svi'],
        'volatility_class': total_svi['class']
    }

def calculate_rwt(quarter_stats: Dict) -> Dict:
    """
    Calcula Referee Whistle Trend (RWT) usando fouls por cuarto.
    Detecta si los árbitros están pitando más o menos.
    """
    if not quarter_stats:
        return {'rwt': 0.0, 'trend_class': 'UNKNOWN'}

    quarters = ['q1', 'q2', 'q3', 'q4']
    fouls_by_quarter = []
    
    for q in quarters:
        if q in quarter_stats:
            try:
                home_fouls = int(quarter_stats[q].get('home', {}).get('personal_fouls', 0))
                away_fouls = int(quarter_stats[q].get('away', {}).get('personal_fouls', 0))
                fouls_by_quarter.append(home_fouls + away_fouls)
            except:
                pass
    
    if len(fouls_by_quarter) < 2:
        return {'rwt': 0.0, 'trend_class': 'INSUFFICIENT_DATA'}
    
    midpoint = len(fouls_by_quarter) // 2
    first_half_avg = np.mean(fouls_by_quarter[:midpoint])
    second_half_avg = np.mean(fouls_by_quarter[midpoint:])
    
    rwt = (second_half_avg - first_half_avg) / max(first_half_avg, 1)
    
    if rwt > 0.25:
        trend_class = 'TIGHTENING'
    elif rwt < -0.25:
        trend_class = 'LOOSENING'
    else:
        trend_class = 'CONSISTENT'
        
    return {
        'rwt': round(rwt, 3),
        'trend_class': trend_class
    }

def calculate_pace_by_quarter(quarter_stats: Dict, quarter_scores: Dict) -> Dict:
    """
    Estima Pace (posesiones por 48 min) usando la fórmula estándar por cuarto.
    """
    pace_data = {'pace_trend': 'STABLE', 'avg_pace': 95.0}
    
    quarters = ['q1', 'q2', 'q3', 'q4']
    ppm_by_quarter = []
    
    for q in quarters:
        if f'{q}_home' in quarter_scores:
            pts = quarter_scores[f'{q}_home'] + quarter_scores[f'{q}_away']
            ppm = pts / 12.0
            ppm_by_quarter.append(ppm)
            
    if len(ppm_by_quarter) >= 2:
        first_half_ppm = np.mean(ppm_by_quarter[:2])
        current_ppm = ppm_by_quarter[-1]
        
        pace_change = (current_ppm - first_half_ppm) / max(first_half_ppm, 1)
        
        if pace_change > 0.1:
            pace_data['pace_trend'] = 'ACCELERATING'
        elif pace_change < -0.1:
            pace_data['pace_trend'] = 'DECELERATING'
            
        pace_data['avg_pace'] = (np.mean(ppm_by_quarter) * 48) / 2.2
        
    return pace_data

def calculate_ortg_by_quarter(quarter_stats: Dict, quarter_scores: Dict) -> Dict:
    """
    Calcula Offensive Rating (puntos por 100 posesiones) por cuarto.
    """
    return {'ortg_trend': 'STABLE', 'home_ortg': 110.0, 'away_ortg': 110.0}

def calculate_quarter_momentum(quarter_scores: Dict) -> Dict:
    """
    Mide el momentum entre cuartos consecutivos (QMD).
    """
    quarters = ['q1', 'q2', 'q3', 'q4']
    total_scores = []
    
    for q in quarters:
        if f'{q}_home' in quarter_scores:
            total_scores.append(quarter_scores[f'{q}_home'] + quarter_scores[f'{q}_away'])
            
    if len(total_scores) < 2:
        return {'momentum_score': 0.0, 'trend': 'UNKNOWN'}
        
    deltas = []
    for i in range(1, len(total_scores)):
        deltas.append(total_scores[i] - total_scores[i-1])
        
    momentum_score = sum(deltas) / len(deltas)
    
    if momentum_score > 3:
        trend = 'BUILDING'
    elif momentum_score < -3:
        trend = 'FADING'
    else:
        trend = 'FLUCTUATING'
        
    return {
        'momentum_score': round(momentum_score, 2),
        'trend': trend
    }

def calculate_detailed_quarter_metrics(quarter_stats: Dict, quarter_scores: Dict) -> Dict:
    """
    Calcula métricas avanzadas detalladas por cuarto usando la fórmula de posesiones real.
    Incluye: Realized Pace, FTR, eFG% Delta, TOV%, Foul Density.
    """
    features = {}
    
    # Iteramos por cada cuarto disponible en los stats
    for q_key, q_stats in quarter_stats.items():
        # Normalizar clave del cuarto (Q1, Q2, etc.)
        q_norm = q_key.upper()
        
        # Extraer datos de los dos equipos (asumiendo estructura {'home': {...}, 'away': {...}})
        if 'home' not in q_stats or 'away' not in q_stats:
            continue
            
        h_stats = q_stats['home']
        a_stats = q_stats['away']
        
        # --- PARSING DE STRINGS A FLOATS (Limpieza de datos) ---
        def p(val): 
            if isinstance(val, str):
                return float(val.replace('%', '')) 
            return float(val) if val is not None else 0.0
        
        # Variables Home
        h_fga = p(h_stats.get('field_goals_attempted', 0))
        h_fgm = p(h_stats.get('field_goals_made', 0))
        h_fta = p(h_stats.get('free_throws_attempted', 0))
        h_orb = p(h_stats.get('offensive_rebounds', 0))
        h_tov = p(h_stats.get('turnovers', 0))
        h_fouls = p(h_stats.get('personal_fouls', 0))
        h_3pm = p(h_stats.get('3_point_field_goals_made', 0))

        # Variables Away
        a_fga = p(a_stats.get('field_goals_attempted', 0))
        a_fgm = p(a_stats.get('field_goals_made', 0))
        a_fta = p(a_stats.get('free_throws_attempted', 0))
        a_orb = p(a_stats.get('offensive_rebounds', 0))
        a_tov = p(a_stats.get('turnovers', 0))
        a_fouls = p(a_stats.get('personal_fouls', 0))
        a_3pm = p(a_stats.get('3_point_field_goals_made', 0))

        # --- 1. REALIZED PACE (Estimación de Posesiones del Cuarto) ---
        # Formula básica: FGA + 0.44*FTA - ORB + TOV
        h_poss = h_fga + (0.44 * h_fta) - h_orb + h_tov
        a_poss = a_fga + (0.44 * a_fta) - a_orb + a_tov
        # El ritmo del cuarto es el promedio de ambos
        q_possessions = (h_poss + a_poss) / 2
        
        # --- 2. FREE THROW RATE (Aggregated) ---
        total_fta = h_fta + a_fta
        total_fga = h_fga + a_fga
        ft_rate = total_fta / total_fga if total_fga > 0 else 0
        
        # --- 3. SHOOTING SUSTAINABILITY (eFG% Combinado) ---
        total_fgm = h_fgm + a_fgm
        total_3pm = h_3pm + a_3pm
        combined_efg = (total_fgm + (0.5 * total_3pm)) / total_fga if total_fga > 0 else 0
        
        # --- 4. TURNOVER % (Waste Rate) ---
        total_tov = h_tov + a_tov
        tov_rate = total_tov / q_possessions if q_possessions > 0 else 0
        
        # --- 5. FOUL DENSITY ---
        total_fouls = h_fouls + a_fouls
        foul_density = total_fouls / q_possessions if q_possessions > 0 else 0

        # Guardamos en el diccionario de features
        features[f'{q_norm}_Pace'] = round(q_possessions, 2)
        features[f'{q_norm}_FTR'] = round(ft_rate, 3)
        features[f'{q_norm}_eFG'] = round(combined_efg, 3)
        features[f'{q_norm}_TOV_Rate'] = round(tov_rate, 3)
        features[f'{q_norm}_Foul_Density'] = round(foul_density, 3)

    return features

def calculate_advanced_live_features(game_data: Dict) -> Dict:
    """
    Wrapper principal para calcular todas las nuevas métricas.
    """
    quarter_scores = game_data.get('quarter_scores', {})
    quarter_stats = game_data.get('quarter_stats', {})
    
    features = {}
    
    # 1. SVI
    features.update(calculate_svi_by_quarter(quarter_scores))
    
    # 2. RWT
    rwt_data = calculate_rwt(quarter_stats)
    features['rwt'] = rwt_data['rwt']
    
    # 3. Pace (Legacy + New Realized Pace)
    # pace_data = calculate_pace_by_quarter(quarter_stats, quarter_scores)
    # features['avg_pace'] = pace_data['avg_pace']
    
    # 4. Momentum
    # momentum_data = calculate_quarter_momentum(quarter_scores)
    # features['qmd_momentum'] = momentum_data['momentum_score']
    
    # 5. 🆕 DETAILED METRICS (Realized Pace, FTR, eFG, TOV, Foul Density)
    detailed_metrics = calculate_detailed_quarter_metrics(quarter_stats, quarter_scores)
    features.update(detailed_metrics)
    
    return features

def calculate_pace_poisson(live_pace_metrics, q_scores, market_total=None, time_info=None):
    """
    The Pacer (Pace-Adjusted Poisson) - Phase 2 Implementation
    Logic: (Current_Score / Current_Possessions) * (Current_Pace * 48)
    Returns projected total and probabilities.
    """
    try:
        # 1. Get inputs
        current_pace = live_pace_metrics.get('enhanced_pace_estimate', 
                       live_pace_metrics.get('live_pace_estimate', 96.0))
        
        # Calculate minutes played
        minutes_played = 0
        if time_info and 'minutes_played' in time_info:
            minutes_played = time_info['minutes_played']
        else:
            # Infer from quarters
            if 'q3_home' in q_scores and q_scores['q3_home'] > 0:
                minutes_played = 36 # Assuming end of Q3
                if time_info and 'time_into_quarter' in time_info:
                     minutes_played = 24 + (time_info['time_into_quarter'] / 60)
            elif 'q2_home' in q_scores and q_scores['q2_home'] > 0:
                minutes_played = 24
            elif 'q1_home' in q_scores:
                minutes_played = 12
        
        if minutes_played < 5: # Too early
             return {}

        current_score = sum(q_scores.values())
        
        # Calculate Projected Total
        # Method: Linear Projection adjusted by regression to mean
        projected_total = current_score * (48 / max(minutes_played, 1))
        
        # Adjust for regression to mean if early
        if minutes_played < 36:
            # Regress towards market total or league average
            # More minutes = more weight to projection
            weight = minutes_played / 40.0 # Full trust by min 40
            weight = min(1.0, weight)
            
            baseline = market_total if market_total else 220.0
            projected_total = weight * projected_total + (1 - weight) * baseline

        result = {
            'pace_poisson_projection': projected_total
        }

        if market_total:
            # Poisson probabilities
            # Prob(Score > Market) = 1 - CDF(Market)
            # Use int for discrete distribution
            market_k = int(market_total)
            
            prob_under = poisson.cdf(market_k, projected_total)
            prob_over = 1 - prob_under
            
            result['prob_over'] = prob_over
            result['prob_under'] = prob_under
            
        return result

    except Exception as e:
        print(f"Error in Pacer: {e}")
        return {}
