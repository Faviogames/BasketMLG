# ===========================================
# analysis/context_features.py
# Calcula features de context para el Metamodel
# Estas features reemplazan los ajustes manuales
# ===========================================

from typing import Dict, Any
import numpy as np

def calculate_context_features(
    q_scores: Dict,
    balance_features: Dict,
    live_pace_metrics: Dict,
    time_info: Dict,
    live_signals: Dict,
    team_trends: Dict,
    bookie_line: float = None
) -> Dict[str, float]:
    """
    Calcula features de context para el Metamodel.
    
    Estas features reemplazan ajustes manuales como:
    - Blowout adjustment
    - Context adjustment  
    - Reality check
    
    Returns:
        Dict con ~15-20 features de context
    """
    
    features = {}
    
    # ============================================
    # GRUPO 1: BLOWOUT & GAME STATE FEATURES
    # ============================================
    home_score = q_scores.get('q1_home', 0) + q_scores.get('q2_home', 0) + q_scores.get('q3_home', 0)
    away_score = q_scores.get('q1_away', 0) + q_scores.get('q2_away', 0) + q_scores.get('q3_away', 0)
    
    features['score_diff_abs'] = abs(home_score - away_score)
    features['is_blowout'] = 1 if abs(home_score - away_score) > 15 else 0
    features['is_close_game'] = balance_features.get('is_close_game', 0)
    
    # Time remaining
    remaining_minutes = time_info.get('total_remaining_minutes', 48.0)
    features['time_remaining_pct'] = remaining_minutes / 48.0
    features['minutes_remaining'] = remaining_minutes
    
    # Blowout severity (score diff per minute remaining)
    features['blowout_severity'] = features['score_diff_abs'] / max(1, remaining_minutes)
    
    # ============================================
    # GRUPO 2: GARBAGE TIME & INTENSITY FEATURES
    # ============================================
    features['garbage_time_risk'] = balance_features.get('garbage_time_risk', 0.0)
    features['intensity_drop_factor'] = balance_features.get('intensity_drop_factor', 1.0)
    features['is_potential_blowout'] = balance_features.get('is_potential_blowout', 0)
    
    # ============================================
    # GRUPO 3: PACE FEATURES
    # ============================================
    current_total = balance_features.get('current_total', 0)
    current_pace = time_info.get('current_pace_per_minute', 0.0)
    
    # Pace projection (simple)
    projected_remaining = remaining_minutes * current_pace if current_pace > 0 else 0
    features['pace_projection'] = current_total + projected_remaining
    
    # Pace vs league average
    league_avg_pace = 2.0  # ~2 pts/min típico NBA
    features['current_pace_vs_avg'] = current_pace - league_avg_pace if current_pace > 0 else 0
    
    # Pace acceleration Q2 vs Q1
    q1_total = q_scores.get('q1_home', 0) + q_scores.get('q1_away', 0)
    q2_total = q_scores.get('q2_home', 0) + q_scores.get('q2_away', 0)
    
    if q1_total > 0:
        features['pace_acceleration'] = (q2_total - q1_total) / q1_total
    else:
        features['pace_acceleration'] = 0.0
    
    # ============================================
    # GRUPO 4: MOMENTUM & SURGE FEATURES
    # ============================================
    features['q2_surge'] = q2_total - q1_total
    features['momentum_score'] = q2_total - q1_total  # Same as halftime momentum
    
    # ============================================
    # GRUPO 5: FOUL FEATURES
    # ============================================
    home_fti = live_signals.get('home_fti', 0.0) if live_signals else 0.0
    away_fti = live_signals.get('away_fti', 0.0) if live_signals else 0.0
    
    features['home_foul_trouble_index'] = home_fti
    features['away_foul_trouble_index'] = away_fti
    features['max_foul_trouble'] = max(home_fti, away_fti)
    features['home_in_bonus'] = 1 if home_fti > 0.8 else 0
    features['away_in_bonus'] = 1 if away_fti > 0.8 else 0
    
    # ============================================
    # GRUPO 6: TEAM TREND FEATURES
    # ============================================
    home_trend = team_trends.get('home', {}).get('trend_factor', 1.0) if team_trends else 1.0
    away_trend = team_trends.get('away', {}).get('trend_factor', 1.0) if team_trends else 1.0
    
    features['home_trend_factor'] = home_trend
    features['away_trend_factor'] = away_trend
    features['avg_trend_factor'] = (home_trend + away_trend) / 2.0
    features['is_surge_matchup'] = 1 if (home_trend > 1.08 and away_trend > 1.08) else 0
    
    # ============================================
    # GRUPO 7: MARKET FEATURES (si disponible)
    # ============================================
    if bookie_line and bookie_line > 0:
        features['bookie_line'] = bookie_line
        features['model_vs_bookie_diff'] = 0  # Will be calculated later with base_prediction
    else:
        features['bookie_line'] = 0.0
        features['model_vs_bookie_diff'] = 0.0
    
    # ============================================
    # GRUPO 8: ANOMALY & VARIANCE FEATURES
    # ============================================
    # Quarter variance (if available)
    q3_total = q_scores.get('q3_home', 0) + q_scores.get('q3_away', 0)
    quarters = [q1_total, q2_total, q3_total]
    quarters_nonzero = [q for q in quarters if q > 0]
    
    if len(quarters_nonzero) >= 2:
        features['quarter_variance'] = float(np.std(quarters_nonzero))
        features['is_high_variance'] = 1 if np.std(quarters_nonzero) > 8 else 0
    else:
        features['quarter_variance'] = 0.0
        features['is_high_variance'] = 0
    
    # ============================================
    # GRUPO 9: QUARTER STAGE (categorical → numeric)
    # ============================================
    quarter_stage = time_info.get('quarter_stage', 'unknown')
    
    # One-hot encoding simplificado
    features['is_q2'] = 1 if 'q2' in quarter_stage else 0
    features['is_q3'] = 1 if 'q3' in quarter_stage else 0
    features['is_q4'] = 1 if 'q4' in quarter_stage else 0
    features['is_halftime'] = 1 if 'halftime' in quarter_stage else 0
    
    return features


def calculate_advanced_metrics(
    stats: Dict[str, Any],
    time_info: Dict[str, Any]
) -> Dict[str, float]:
    """
    🚀 VISIONARY ROADMAP: PHASE 1 (The Pulse Engine)
    Calcula métricas avanzadas sintéticas usando datos live reales.
    
    Metrics:
    - Possessions (Synthetic)
    - Offensive Rating (ORTG)
    - True Pace (Possessions / 48min)
    """
    metrics = {}
    
    try:
        # 1. Extraer componentes básicos (con fallback a 0)
        # Nota: live_api.py devuelve keys como 'home_field_goals_attempted'
        home_fga = stats.get('home_field_goals_attempted', 0)
        away_fga = stats.get('away_field_goals_attempted', 0)
        
        home_fta = stats.get('home_free_throws_attempted', 0)
        away_fta = stats.get('away_free_throws_attempted', 0)
        
        home_orb = stats.get('home_offensive_rebounds', 0)
        away_orb = stats.get('away_offensive_rebounds', 0)
        
        home_to = stats.get('home_turnovers', 0)
        away_to = stats.get('away_turnovers', 0)
        
        home_score = stats.get('home_score', 0)
        away_score = stats.get('away_score', 0)
        
        # 2. Calcular Posesiones Sintéticas (Fórmula Dean Oliver)
        # Poss = 0.96 * (FGA + TO + 0.44 * FTA - ORB)
        home_poss = 0.96 * (home_fga + home_to + 0.44 * home_fta - home_orb)
        away_poss = 0.96 * (away_fga + away_to + 0.44 * away_fta - away_orb)
        
        avg_poss = (home_poss + away_poss) / 2.0
        metrics['possessions_home'] = home_poss
        metrics['possessions_away'] = away_poss
        metrics['possessions_avg'] = avg_poss
        
        # 3. Calcular Offensive Rating (Pts / 100 Poss)
        metrics['ortg_home'] = (home_score / home_poss * 100) if home_poss > 0 else 0.0
        metrics['ortg_away'] = (away_score / away_poss * 100) if away_poss > 0 else 0.0
        
        # 4. Calcular True Pace (Poss / Minutos Jugados * 48)
        minutes_played = time_info.get('total_minutes_played', 0.0)
        if minutes_played > 0:
            metrics['true_pace'] = avg_poss / minutes_played * 48.0
        else:
            metrics['true_pace'] = 0.0
            
        # 5. Efficiency Gap (Diferencia de eficiencia)
        metrics['efficiency_gap'] = metrics['ortg_home'] - metrics['ortg_away']

        # 6. True Shooting Percentage (TS%)
        # Formula: Pts / (2 * (FGA + 0.44 * FTA))
        if (home_fga + 0.44 * home_fta) > 0:
            metrics['ts_home'] = home_score / (2 * (home_fga + 0.44 * home_fta))
        else:
            metrics['ts_home'] = 0.0
            
        if (away_fga + 0.44 * away_fta) > 0:
            metrics['ts_away'] = away_score / (2 * (away_fga + 0.44 * away_fta))
        else:
            metrics['ts_away'] = 0.0
        
        return metrics
        
    except Exception as e:
        print(f"[WARNING] Error calculating advanced metrics: {e}")
        return {
            'possessions_avg': 0.0,
            'ortg_home': 0.0,
            'ortg_away': 0.0,
            'true_pace': 0.0,
            'efficiency_gap': 0.0
        }


def calculate_model_vs_pace_gap(base_prediction: float, pace_projection: float) -> float:
    """
    Calcula gap entre predicción del modelo y proyección basada en pace.
    
    Feature derivada que se usa para detectar:
    - Model overestimating (gap > 20)
    - Model underestimating (gap < -15)
    """
    return base_prediction - pace_projection


def calculate_model_vs_bookie_diff(base_prediction: float, bookie_line: float) -> float:
    """
    Calcula diferencia entre modelo base y línea de casa.
    
    Positivo: Modelo más optimista que casa
    Negativo: Modelo más conservador que casa
    """
    if bookie_line and bookie_line > 0:
        return base_prediction - bookie_line
    return 0.0
