# ===========================================
# Archivo: analysis/live_mode.py (v3.0)
# ===========================================
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional
import traceback
import os
import joblib
import json
from core.game_clock import league_clock_settings

# Imports del sistema
from config import (
    ADVANCED_STATS_COLS, MOMENTUM_STATS_COLS, PERFORMANCE_CONTEXT_COLS,
    CONTEXT_ADJUSTMENT, MODELS_FOLDER,
    REALITY_CHECK_V2, CACHE_CONTROL, DATA_QUALITY_CONFIG
)
# Removed metamodel import - no longer needed

def _canonicalize_league_name(name: str) -> str:
    """
    Normaliza el nombre de la liga para estabilizar artefactos entre entrenamiento y live.
    - Quita sufijos de etapa después de ' - ' (e.g., 'Eurobasket - SEMI-FINALS' -> 'Eurobasket')
    - Unifica variantes conocidas (wnba -> WNBA, euro* -> EUROBASKET, nbl -> NBL)
    - Mantiene el resto si no coincide con reglas conocidas
    """
    n = (name or '').strip()
    if ' - ' in n:
        n = n.split(' - ', 1)[0].strip()
    low = n.lower()
    if 'wnba' in low:
        return 'WNBA'
    if 'nbl' in low or 'national_basketball' in low or 'aus' in low:
        return 'NBL'
    if 'euro' in low:
        return 'EUROBASKET'
    return n

# Imports robustos con fallbacks:
try:
    from core.features import (
        get_rolling_stats, get_ema_stats, get_enhanced_ema_stats,
        calculate_momentum_metrics,
        calculate_team_quarter_trends,
        apply_blowout_adjustment, safe_mean, safe_divide
    )
    from core.live_features import (
        calculate_live_pace_metrics, calculate_real_balance_features,
        calculate_garbage_time_risk
    )
    FEATURES_IMPORTS_OK = True
except ImportError as e:
    print(f"[WARNING] Error importando core.features o core.live_features: {e}")
    FEATURES_IMPORTS_OK = False

    # Definir funciones fallback básicas
    def safe_mean(series):
        if hasattr(series, 'dropna'):
            clean_data = series.dropna()
            return clean_data.mean() if len(clean_data) > 0 else 0.0
        return 0.0

    def safe_divide(numerator, denominator, default=0.0):
        """Safe division with fallback for zero denominator"""
        try:
            if denominator == 0 or denominator is None:
                return default
            return numerator / denominator
        except (ZeroDivisionError, TypeError):
            return default

# Sistema de alertas eliminado - integrado en el meta-modelo
ALERTS_AVAILABLE = False
print("[INFO] Sistema de alertas integrado en el meta-modelo")

# ===================================================================
# SISTEMA DE AJUSTES POR CONTEXTO (MANTENIDO)
# ===================================================================

def _get_league_scoring_params(league_name: str = None) -> Dict[str, float]:
    """Get league-specific scoring parameters"""
    if not league_name:
        # Default conservative parameters
        return {
            'q4_conservative_reduction': 0.96,  # -4% for Q4 close games
            'pace_multiplier': 1.0      # Base pace multiplier
        }

    league_lower = league_name.lower()

    if 'nba' in league_lower:
        # NBA: Higher scoring potential, faster pace
        return {
            'q4_conservative_reduction': 0.96,  # Same -4% reduction
            'pace_multiplier': 1.2      # NBA has faster pace
        }
    elif 'wnba' in league_lower:
        # WNBA: Lower scoring, shorter quarters (10m)
        return {
            'q4_conservative_reduction': 0.98,  # Less reduction
            'pace_multiplier': 0.95     # Slower pace
        }
    return {
        'q4_conservative_reduction': 0.96,
        'pace_multiplier': 1.0
    }

def _calculate_safe_projection(current_total: float, minutes_played: float, quarter_stage: str, current_pace: float) -> float:
    """
    Calcula una proyección 'segura' basada en lo que ya ha ocurrido + ritmo actual.
    Fórmula: Puntos_Actuales + (Minutos_Restantes * Ritmo_Actual)
    """
    if minutes_played <= 0:
        return 0.0
        
    # Determinar duración total del partido
    total_game_minutes = 48.0 # Default NBA
    if 'wnba' in quarter_stage.lower() or 'fiba' in quarter_stage.lower(): # Simplificación, idealmente pasar liga
        total_game_minutes = 40.0
        
    minutes_remaining = max(0, total_game_minutes - minutes_played)
    
    # Si estamos en Q3/Q4, el ritmo actual es muy fiable
    if minutes_played >= 24.0:
        # Usar el ritmo actual del partido
        safe_projection = current_total + (minutes_remaining * current_pace)
        return safe_projection
    else:
        # En primera mitad, mezclar ritmo actual con promedio de liga (aprox 2.0 pts/min)
        blended_pace = (current_pace * 0.7) + (2.0 * 0.3)
        safe_projection = current_total + (minutes_remaining * blended_pace)
        return safe_projection


def _get_league_context_adjustment(league_name: str = None) -> Dict[str, Any]:
    """Get league-specific context adjustment parameters"""
    if not league_name:
        # Default global parameters
        from config import CONTEXT_ADJUSTMENT
        return CONTEXT_ADJUSTMENT

    league_lower = league_name.lower()

    if 'wnba' in league_lower:
        # WNBA-specific context adjustments (very different from NBA)
        try:
            from config import CONTEXT_ADJUSTMENT_WNBA
            return CONTEXT_ADJUSTMENT_WNBA
        except ImportError:
            print(f"[WARNING] WNBA context adjustments not found, using default")
            from config import CONTEXT_ADJUSTMENT
            return CONTEXT_ADJUSTMENT
    elif 'nbl' in league_lower:
        # NBL-specific context adjustments
        try:
            from config import CONTEXT_ADJUSTMENT_NBL
            return CONTEXT_ADJUSTMENT_NBL
        except ImportError:
            print(f"[WARNING] NBL context adjustments not found, using default")
            from config import CONTEXT_ADJUSTMENT
            return CONTEXT_ADJUSTMENT
    else:
        # Default global parameters for NBA and other leagues
        from config import CONTEXT_ADJUSTMENT
        return CONTEXT_ADJUSTMENT

def _get_live_pace_high_threshold(league_name: str = None) -> float:
    """
    Define un umbral 'alto' para live_pace_estimate/enhanced_pace_estimate en la escala actual
    usada por el sistema (que típicamente produce valores ~110–190 en HT/Q3).
    Estos valores están calibrados a la metodología de cálculo de pace actual.
    """
    if not league_name:
        return 140.0
    low = league_name.lower()
    if 'wnba' in low:
        return 140.0
    if 'nbl' in low:
        return 155.0
    if 'nba' in low:
        return 150.0
    return 145.0

# ===========================================
# ENHANCED META-MODEL WITH LIVE SIGNAL LEARNING
# ===========================================

def calculate_signal_strength(live_signals: Dict[str, float]) -> float:
    """
    Calculate overall signal strength based on active signals and their magnitudes.
    Returns a value between 0.0 (no signals) and 1.0 (maximum signal impact).
    """
    if not live_signals:
        return 0.0

    signal_weights = {
        # Primary O/U signals (higher weight)
        'diff_ato_ratio': 0.25,      # Assist/Turnover ratio differential
        'home_fti': 0.20,            # Foul trouble index
        'away_fti': 0.20,
        'diff_ts_live': 0.15,        # True shooting differential
        'run_strength': 0.10,        # Run detector strength

        # Secondary signals (lower weight)
        'home_treb_diff': 0.05,      # Rebound differential
        'away_treb_diff': 0.05,
    }

    total_strength = 0.0
    max_possible_strength = sum(signal_weights.values())

    for signal_name, weight in signal_weights.items():
        if signal_name in live_signals:
            signal_value = abs(live_signals[signal_name])

            # Normalize signal value to 0-1 scale based on typical thresholds
            if 'ato_ratio' in signal_name:
                normalized_value = min(1.0, signal_value / 0.5)  # 0.25 threshold
            elif 'fti' in signal_name:
                normalized_value = min(1.0, max(0.0, (signal_value - 0.5) / 0.3))  # 0.7-1.0 range
            elif 'ts_live' in signal_name:
                normalized_value = min(1.0, signal_value / 0.3)  # 0.15 threshold
            elif 'run_strength' in signal_name:
                normalized_value = min(1.0, signal_value / 0.5)  # 0.3 threshold
            elif 'treb_diff' in signal_name:
                normalized_value = min(1.0, signal_value / 0.8)  # 0.4 threshold
            else:
                normalized_value = min(1.0, signal_value)

            total_strength += weight * normalized_value

    return min(1.0, total_strength / max_possible_strength) if max_possible_strength > 0 else 0.0

def count_active_signals(live_signals: Dict[str, float]) -> int:
    """
    Count how many signals are actively triggering adjustments.
    """
    if not live_signals:
        return 0

    active_count = 0

    # Define activation thresholds for each signal
    thresholds = {
        'diff_ato_ratio': 0.25,
        'home_fti': 0.7,
        'away_fti': 0.7,
        'diff_ts_live': 0.15,
        'run_active': 0.5,  # For run_strength
        'home_treb_diff': 0.4,
        'away_treb_diff': 0.4,
    }

    for signal_name, threshold in thresholds.items():
        if signal_name in live_signals:
            signal_value = live_signals[signal_name]

            if signal_name == 'run_active':
                # Special case for run detector
                if live_signals.get('run_active', False) and live_signals.get('run_strength', 0) > 0.3:
                    active_count += 1
            elif signal_name == 'home_fti' or signal_name == 'away_fti':
                # FTI signals are active if above threshold
                if signal_value > threshold:
                    active_count += 1
            else:
                # Other signals active if absolute value above threshold
                if abs(signal_value) > threshold:
                    active_count += 1

    return active_count

# ===================================================================
# INDIVIDUAL CONTEXT ADJUSTMENT FUNCTIONS
# ===================================================================

def _apply_garbage_time_adjustment(gt_enabled: bool, garbage_time_risk: float,
                                  quarter_stage: str, home_fti: float, away_fti: float,
                                  context_adjustment: Dict) -> Tuple[float, str]:
    """
    Apply garbage time adjustment based on risk level.

    Args:
        gt_enabled: Whether garbage time adjustment is enabled
        garbage_time_risk: Risk level (0-1)
        quarter_stage: Current quarter stage
        home_fti: Home team foul trouble index
        away_fti: Away team foul trouble index
        context_adjustment: Context adjustment configuration

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if not (gt_enabled and garbage_time_risk is not None and
            quarter_stage in ['q3_just_ended', 'q4_progress'] and garbage_time_risk >= 0.6):
        return 1.0, ""

    # Reducción suave acotada y atenuada por foul-trouble
    max_red_pct = context_adjustment.get('garbage_time_max_reduction_pct', 0.12)
    foul_trouble_boost = 1 + (max(home_fti, away_fti) * 0.1)  # +10% máx
    raw_reduction = min(max_red_pct, garbage_time_risk * max_red_pct)
    adjusted_reduction = (1 - raw_reduction) * foul_trouble_boost

    description = f"Garbage time dinámico: -{((1-adjusted_reduction)*100):.0f}% (risk: {garbage_time_risk:.2f}, FT boost: {foul_trouble_boost:.2f})"
    return adjusted_reduction, description


def _apply_close_game_high_pace_adjustment(is_close_game: bool, is_high_pace: bool,
                                          is_late_game: bool, quarter_stage: str,
                                          home_ts_live: float, away_ts_live: float,
                                          enhanced_pace: float, league_name: str) -> Tuple[float, str]:
    """
    Apply adjustment for close games with high pace.

    Args:
        is_close_game: Whether the game is close
        is_high_pace: Whether pace is high
        is_late_game: Whether it's late in the game
        quarter_stage: Current quarter stage
        home_ts_live: Home team true shooting %
        away_ts_live: Away team true shooting %
        enhanced_pace: Enhanced pace estimate
        league_name: League name for logging

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if not (is_close_game and is_high_pace and is_late_game):
        return 1.0, ""

    # More conservative logic for all late-game scenarios
    if quarter_stage == 'q4_progress':
        # En Q4 de partidos cerrados, equipos suelen frenar el ritmo significativamente
        reduction_pct = 6.0  # -6% (increased from -4%)
        description = f"Q4 partido cerrado (conservador): -{reduction_pct:.0f}% (pace alto pero Q4 defensivo - {league_name or 'Unknown'})"
        return 1.0 - (reduction_pct / 100), description
    elif quarter_stage == 'q3_just_ended':
        # Just ended Q3, entering Q4 - be very conservative
        reduction_pct = 3.0  # -3%
        description = f"Fin Q3 partido cerrado: -{reduction_pct:.0f}% (transición a Q4 defensivo)"
        return 1.0 - (reduction_pct / 100), description
    else:
        # Para Q3, mantener lógica original pero más conservadora
        shooting_boost = 1 + (max(home_ts_live, away_ts_live) - 0.5) * 0.05  # +5% max (reduced from +10%)
        adjustment_factor = 1.01 * shooting_boost  # +1% base (reduced from +2%)
        description = f"Partido cerrado + pace alto (conservador): +{((adjustment_factor-1)*100):.1f}% (pace: {enhanced_pace:.1f}, TS boost: {shooting_boost:.2f})"
        return adjustment_factor, description


def _apply_intensity_drop_adjustment(intensity_drop: float, run_strength: float) -> Tuple[float, str]:
    """
    Apply adjustment for games with declining intensity.

    Args:
        intensity_drop: Intensity drop factor
        run_strength: Run detector strength

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if intensity_drop >= 0.85:
        return 1.0, ""

    # Escalar por run detector (run activa = más puntos)
    run_boost = 1 + (run_strength * 0.15)  # +15% max por run strength
    adjusted_reduction = 0.96 * run_boost
    description = f"Caída de intensidad: -{((1-adjusted_reduction)*100):.0f}% (factor: {intensity_drop:.3f}, run boost: {run_boost:.2f})"
    return adjusted_reduction, description


def _apply_fouling_strategy_adjustment(is_close_game: bool, quarter_stage: str,
                                       enhanced_pace: float, home_fti: float, away_fti: float) -> Tuple[float, str]:
    """
    Apply adjustment for fouling strategy in close Q4 games.
    More conservative to prevent unrealistic predictions.

    Args:
        is_close_game: Whether the game is close
        quarter_stage: Current quarter stage
        enhanced_pace: Enhanced pace estimate
        home_fti: Home team foul trouble index
        away_fti: Away team foul trouble index

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if not (is_close_game and quarter_stage == 'q4_progress' and enhanced_pace > 110):
        return 1.0, ""

    # More conservative: cap the foul trouble boost and base adjustment
    max_fti = max(home_fti, away_fti)
    if max_fti < 0.8:  # Require significant foul trouble
        return 1.0, ""

    # Reduced boost: +15% max instead of +25%
    ft_boost = 1 + ((max_fti - 0.8) * 0.1875)  # Scales from 0% to +15%
    # Reduced base: +4% instead of +8%
    adjustment_factor = 1.04 * ft_boost

    # Cap total adjustment to +8% max for Q4 fouling
    adjustment_factor = min(adjustment_factor, 1.08)

    description = f"Estrategia fouling Q4 (conservador): +{((adjustment_factor-1)*100):.1f}% (pace: {enhanced_pace:.1f}, FTI: {max_fti:.2f})"
    return adjustment_factor, description


def _apply_run_detector_adjustment(run_active: bool, run_strength: float, run_side: str) -> Tuple[float, str]:
    """
    Apply adjustment when run detector is active.

    Args:
        run_active: Whether run detector is active
        run_strength: Run detector strength
        run_side: Which side has the run

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if not (run_active and run_strength > 0.3):
        return 1.0, ""

    run_boost = 1 + (run_strength * 0.2)  # +20% max por run strength
    description = f"Run detectado ({run_side}): +{((run_boost-1)*100):.0f}% (strength: {run_strength:.2f})"
    return run_boost, description


def _apply_foul_trouble_adjustment(home_fti: float, away_fti: float) -> Tuple[float, str]:
    """
    Apply adjustment for significant foul trouble.

    Args:
        home_fti: Home team foul trouble index
        away_fti: Away team foul trouble index

    Returns:
        Tuple of (adjustment_factor, description)
    """
    max_fti = max(home_fti, away_fti)
    if max_fti <= 0.7:
        return 1.0, ""

    ft_boost = 1 + ((max_fti - 0.7) * 0.3)  # +30% max por foul trouble extremo
    description = f"Foul trouble alto: +{((ft_boost-1)*100):.0f}% (FTI: {max_fti:.2f})"
    return ft_boost, description


def _apply_shooting_efficiency_adjustment(diff_ts_live: float) -> Tuple[float, str]:
    """
    Apply adjustment for extreme shooting efficiency differential.

    Args:
        diff_ts_live: True shooting % differential

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if abs(diff_ts_live) <= 0.15:
        return 1.0, ""

    shooting_boost = 1 + (abs(diff_ts_live) * 0.15)  # +15% max por diferencia TS%
    description = f"Shooting efficiency diferencial: +{((shooting_boost-1)*100):.0f}% (TS diff: {diff_ts_live:.2f})"
    return shooting_boost, description


def _apply_assist_turnover_adjustment(diff_ato_ratio: float) -> Tuple[float, str]:
    """
    Apply adjustment for extreme assist-to-turnover ratio differential.

    Args:
        diff_ato_ratio: Assist-to-turnover ratio differential

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if abs(diff_ato_ratio) <= 0.25:
        return 1.0, ""

    ato_boost = 1 + (abs(diff_ato_ratio) * 0.15)  # +15% max por diferencia A/TO
    direction = "superior" if diff_ato_ratio > 0 else "inferior"
    description = f"Control de balón {direction}: +{((ato_boost-1)*100):.0f}% (A/TO diff: {diff_ato_ratio:.2f})"
    return ato_boost, description


def _apply_rebound_efficiency_adjustment(home_treb_diff: float) -> Tuple[float, str]:
    """
    Apply adjustment for extreme rebound efficiency differential.

    Args:
        home_treb_diff: Home team rebound differential

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if abs(home_treb_diff) <= 0.4:
        return 1.0, ""

    reb_boost = 1 + (abs(home_treb_diff) * 0.12)  # +12% max por ventaja en rebotes
    direction = "ventaja" if home_treb_diff > 0 else "desventaja"
    description = f"Rebotes {direction}: +{((reb_boost-1)*100):.0f}% (TREB diff: {home_treb_diff:.2f})"
    return reb_boost, description


def _apply_time_remaining_adjustment(remaining_minutes: float) -> Tuple[float, str]:
    """
    Apply adjustment based on remaining time.

    Args:
        remaining_minutes: Minutes remaining in the game

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if remaining_minutes <= 0:
        return 1.0, ""

    # Ajuste por tiempo restante: más tiempo = más puntos posibles
    time_factor = min(1.0, remaining_minutes / 48.0)  # Normalizar vs juego completo
    time_adjustment = 1 + (time_factor * 0.05)  # +5% max por tiempo restante
    description = f"Tiempo restante ({remaining_minutes:.1f}min): +{((time_adjustment-1)*100):.1f}% (más tiempo = más puntos)"
    return time_adjustment, description


def _apply_pace_expectation_adjustment(current_pace: float, remaining_minutes: float,
                                      completion_percentage: float) -> Tuple[float, str]:
    """
    Apply adjustment based on current pace vs time remaining.

    Args:
        current_pace: Current pace (points per minute)
        remaining_minutes: Minutes remaining
        completion_percentage: Game completion percentage

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if current_pace <= 0 or completion_percentage <= 10:
        return 1.0, ""

    # Si el pace actual es alto pero queda poco tiempo, reducir expectativa
    pace_expectation = current_pace * (remaining_minutes / 60)  # Puntos esperados restantes
    if pace_expectation < 10 and remaining_minutes < 20:
        pace_reduction = 0.95  # -5% si pace bajo con poco tiempo
        description = f"Pace actual bajo ({current_pace:.1f}pts/min) con poco tiempo: -5%"
        return pace_reduction, description

    return 1.0, ""


def _apply_overtime_adjustment(is_overtime: bool) -> Tuple[float, str]:
    """
    Apply adjustment for overtime games.

    Args:
        is_overtime: Whether the game is in overtime

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if not is_overtime:
        return 1.0, ""

    # En overtime, los equipos suelen frenar el ritmo
    ot_adjustment = 0.92  # -8% en overtime
    description = "Overtime: -8% (ritmo más lento en prórroga)"
    return ot_adjustment, description


def _apply_end_game_adjustment(remaining_minutes: float, is_overtime: bool) -> Tuple[float, str]:
    """
    Apply adjustment for end game situations.

    Args:
        remaining_minutes: Minutes remaining
        is_overtime: Whether the game is in overtime

    Returns:
        Tuple of (adjustment_factor, description)
    """
    if remaining_minutes > 6 or is_overtime:
        return 1.0, ""

    # En los últimos minutos, menos puntos esperados
    final_adjustment = 0.96  # -4% en garbage time
    description = "Final del partido: -4% (menos puntos en garbage time)"
    return final_adjustment, description


def calculate_signal_adjustment_magnitude(context_info: Dict) -> float:
    """
    Calculate the total magnitude of signal-based adjustments applied.
    """
    applied_adjustments = context_info.get('applied_adjustments', [])

    total_magnitude = 0.0

    for adjustment in applied_adjustments:
        # Extract percentage changes from adjustment descriptions
        if '%' in adjustment:
            try:
                # Find the percentage value in the string
                import re
                pct_match = re.search(r'([+-]?\d+\.?\d*)%', adjustment)
                if pct_match:
                    pct_value = float(pct_match.group(1))
                    total_magnitude += abs(pct_value)
            except:
                continue

    return total_magnitude




def _calculate_pace_based_projection(current_total: float, remaining_minutes: float,
                                   current_pace: float, league_name: str = None) -> float:
    """
    LEGACY FUNCTION - Mantiene compatibilidad con código existente
    Usa versión simple sin team trends
    """
    if current_pace <= 0 or remaining_minutes <= 0:
        return current_total

    # Project remaining points based on current pace
    projected_remaining = remaining_minutes * current_pace

    # Apply conservative scaling for late game (teams tend to slow down)
    if remaining_minutes < 6:  # Last 6 minutes
        projected_remaining *= 0.80  # -20% very conservative (MEJORADO)
    elif remaining_minutes < 12:  # Last 12 minutes
        projected_remaining *= 0.85  # -15% conservative (MEJORADO)
    elif remaining_minutes < 24:  # Q3+Q4
        projected_remaining *= 0.92  # -8% moderate (NUEVO)

    return current_total + projected_remaining


def _apply_reality_check_penalty(prediction: float, realistic_projection: float,
                                quarter_stage: str, applied_adjustments: list) -> Tuple[float, str]:
    """
    Apply penalty when prediction deviates significantly from pace-based projection.
    """
    deviation = prediction - realistic_projection

    # Only penalize upward deviations in Q4 (over-optimistic predictions)
    if deviation > 15 and quarter_stage == 'q4_progress':
        # Scale penalty based on deviation magnitude
        penalty_factor = min(0.95, 1.0 - (deviation - 15) * 0.01)  # Max -5% penalty
        description = f"Reality check Q4: -{((1-penalty_factor)*100):.1f}% (pred: {prediction:.0f} vs pace proj: {realistic_projection:.0f})"
        return penalty_factor, description
    elif deviation > 25:  # Extreme deviation in any quarter
        penalty_factor = min(0.97, 1.0 - (deviation - 25) * 0.005)  # Max -3% penalty
        description = f"Reality check: -{((1-penalty_factor)*100):.1f}% (extreme deviation: {deviation:.1f} pts)"
        return penalty_factor, description

    return 1.0, ""


def apply_context_adjustment(prediction: float, balance_features: Dict[str, float],
                              live_pace_metrics: Dict[str, float], quarter_stage: str,
                              live_signals: Dict[str, float] = None,
                              league_name: str = None) -> Tuple[float, Dict[str, Any]]:
    """
    Sistema de ajustes por contexto - CON REALITY CHECK PARA PREDICCIONES REALISTAS
    """
    adjustment_factor = 1.0
    applied_adjustments = []

    try:
        # Get league-specific parameters
        league_params = _get_league_scoring_params(league_name)

        # Get league-specific context adjustment parameters
        context_adjustment = _get_league_context_adjustment(league_name)

        # Extraer contexto del partido con valores por defecto seguros
        blowout_momentum = balance_features.get('blowout_momentum', 0.0)
        intensity_drop = balance_features.get('intensity_drop_factor', 1.0)
        current_lead = balance_features.get('current_lead', None)
        is_unbalanced_flag = balance_features.get('is_game_unbalanced', 0) == 1

        # [NEW] Extraer información de tiempo precisa para ajustes más inteligentes
        time_info = live_signals.get('time_info', {}) if live_signals else {}
        remaining_minutes = time_info.get('total_remaining_minutes', 48.0)
        current_pace = time_info.get('current_pace_per_minute', 0.0)
        completion_percentage = time_info.get('completion_percentage', 0.0)
        is_overtime = time_info.get('is_overtime', False)

        # Extraer pace context con valores por defecto seguros
        pace_estimate = live_pace_metrics.get('live_pace_estimate', 100)
        enhanced_pace = live_pace_metrics.get('enhanced_pace_estimate', pace_estimate) * league_params['pace_multiplier']

        # Determinar contextos base (partido cerrado = diferencia <= 10)
        is_close_game = (current_lead is not None and current_lead <= 10)
        # Umbral de "pace alto" ajustado a la escala actual de live pace
        pace_high_threshold = _get_live_pace_high_threshold(league_name)
        is_high_pace = enhanced_pace > pace_high_threshold
        is_late_game = quarter_stage in ['q3_just_ended', 'q4_progress']

        # [NEW] FASE 2: Extraer señales live enfocadas en O/U con valores por defecto seguros
        live_signals = live_signals or {}
        home_fti = live_signals.get('home_fti', 0.5)
        away_fti = live_signals.get('away_fti', 0.5)
        diff_fti = live_signals.get('diff_fti', 0.0)
        home_ts_live = live_signals.get('home_ts_live', 0.5)
        away_ts_live = live_signals.get('away_ts_live', 0.5)
        diff_ts_live = live_signals.get('diff_ts_live', 0.0)
        run_active = live_signals.get('run_active', False)
        run_side = live_signals.get('run_side', 'none')
        run_strength = live_signals.get('run_strength', 0.0)

        # [NEW] FASE 2: Señales O/U específicas usando stats completos de SofaScore
        home_ato_ratio = live_signals.get('home_ato_ratio', 1.0)
        away_ato_ratio = live_signals.get('away_ato_ratio', 1.0)
        diff_ato_ratio = live_signals.get('diff_ato_ratio', 0.0)
        home_oreb_pct = live_signals.get('home_oreb_pct', 0.25)
        away_oreb_pct = live_signals.get('away_oreb_pct', 0.25)
        diff_oreb_pct = live_signals.get('diff_oreb_pct', 0.0)
        home_dreb_pct = live_signals.get('home_dreb_pct', 0.75)
        away_dreb_pct = live_signals.get('away_dreb_pct', 0.75)
        diff_dreb_pct = live_signals.get('diff_dreb_pct', 0.0)
        home_treb_diff = live_signals.get('home_treb_diff', 0.0)
        away_treb_diff = live_signals.get('away_treb_diff', 0.0)

        # Apply individual context adjustments in priority order
        adjustments_to_apply = [
            # Garbage time (highest priority - mutually exclusive with others)
            lambda: _apply_garbage_time_adjustment(
                context_adjustment.get('enable_dynamic_garbage_time', True),
                balance_features.get('garbage_time_risk', None),
                quarter_stage, home_fti, away_fti, context_adjustment
            ),

            # Close game + high pace
            lambda: _apply_close_game_high_pace_adjustment(
                is_close_game, is_high_pace, is_late_game, quarter_stage,
                home_ts_live, away_ts_live, enhanced_pace, league_name
            ),

            # Intensity drop
            lambda: _apply_intensity_drop_adjustment(intensity_drop, run_strength),

            # Fouling strategy
            lambda: _apply_fouling_strategy_adjustment(
                is_close_game, quarter_stage, enhanced_pace, home_fti, away_fti
            ),

            # Run detector
            lambda: _apply_run_detector_adjustment(run_active, run_strength, run_side),

            # Foul trouble
            lambda: _apply_foul_trouble_adjustment(home_fti, away_fti),

            # Shooting efficiency
            lambda: _apply_shooting_efficiency_adjustment(diff_ts_live),

            # Assist-to-turnover ratio
            lambda: _apply_assist_turnover_adjustment(diff_ato_ratio),

            # Rebound efficiency
            lambda: _apply_rebound_efficiency_adjustment(home_treb_diff),
        ]

        # Apply first matching adjustment (mutually exclusive)
        for adjustment_func in adjustments_to_apply:
            adj_factor, description = adjustment_func()
            if description:  # Only apply if there's a description (meaning condition was met)
                adjustment_factor *= adj_factor
                applied_adjustments.append(description)
                break  # Only apply one adjustment from the mutually exclusive set

        # Apply time-based adjustments (can be combined)
        time_adj_factor, time_description = _apply_time_remaining_adjustment(remaining_minutes)
        if time_description:
            adjustment_factor *= time_adj_factor
            applied_adjustments.append(time_description)

        pace_adj_factor, pace_description = _apply_pace_expectation_adjustment(
            current_pace, remaining_minutes, completion_percentage
        )
        if pace_description:
            adjustment_factor *= pace_adj_factor
            applied_adjustments.append(pace_description)

        # Apply overtime adjustment
        ot_adj_factor, ot_description = _apply_overtime_adjustment(is_overtime)
        if ot_description:
            adjustment_factor *= ot_adj_factor
            applied_adjustments.append(ot_description)

        # Apply end game adjustment
        end_adj_factor, end_description = _apply_end_game_adjustment(remaining_minutes, is_overtime)
        if end_description:
            adjustment_factor *= end_adj_factor
            applied_adjustments.append(end_description)

        # Default case: balanced game
        if not applied_adjustments or all("Sin ajustes" not in adj for adj in applied_adjustments):
            if is_close_game and 95 <= enhanced_pace <= 105:
                applied_adjustments.append("Partido equilibrado: Sin ajustes")

        # [NEW] REALITY CHECK: Apply pace-based penalty for unrealistic predictions
        try:
            current_total = balance_features.get('current_total', 0)
            if current_total > 0 and remaining_minutes > 0 and current_pace > 0:
                realistic_projection = _calculate_pace_based_projection(
                    current_total, remaining_minutes, current_pace, league_name
                )

                # Calculate what the adjusted prediction would be
                pre_reality_check_prediction = prediction * adjustment_factor

                reality_penalty, reality_description = _apply_reality_check_penalty(
                    pre_reality_check_prediction, realistic_projection, quarter_stage, applied_adjustments
                )

                if reality_description:
                    adjustment_factor *= reality_penalty
                    applied_adjustments.append(reality_description)

                    # Add warning for extreme deviations
                    deviation = pre_reality_check_prediction - realistic_projection
                    if deviation > 30:
                        applied_adjustments.append(f"[WARNING] ALERTA: Predicción {deviation:.1f} pts por encima de proyección pace realista")
        except Exception as e:
            # Don't break the system if reality check fails
            pass

    except Exception as e:
        print(f"[WARNING] Error en ajuste por contexto: {e}")
        import traceback
        traceback.print_exc()  # Add stack trace for debugging
        applied_adjustments.append("Error en análisis - usando predicción base")
        adjustment_factor = 1.0
    
    # Calcular adjusted_prediction con factor de contexto
    adjusted_prediction = prediction * adjustment_factor
    
    # Aplicar límite conservador final (±2 en Q4 close, ±4 otros)
    try:
        pre_clamp_prediction = adjusted_prediction
        if quarter_stage == 'q4_progress' and current_lead is not None and current_lead <= 10:
            max_delta = 2.0
        else:
            max_delta = 4.0

        upper = prediction + max_delta
        lower = prediction - max_delta

        if adjusted_prediction > upper:
            adjusted_prediction = upper
            applied_adjustments.append(f"Clamp final aplicado: superior (de {pre_clamp_prediction:.0f} a {adjusted_prediction:.0f}, bound=+{max_delta:.0f})")
        elif adjusted_prediction < lower:
            adjusted_prediction = lower
            applied_adjustments.append(f"Clamp final aplicado: inferior (de {pre_clamp_prediction:.0f} a {adjusted_prediction:.0f}, bound=-{max_delta:.0f})")
    except Exception:
        # Si algo falla en el clamp final, continuar con el valor actual
        pass

    # Asegurar que adjusted_prediction esté en el diccionario
    context_info = {
        'original_prediction': prediction,
        'adjusted_prediction': adjusted_prediction,
        'adjustment_factor': adjustment_factor,
        'adjustment_percentage': (adjustment_factor - 1) * 100,
        'applied_adjustments': applied_adjustments,
        'contexts_detected': {
            'is_close_game': is_close_game,
            'is_high_pace': is_high_pace,
            'is_late_game': is_late_game,
            'blowout_momentum': blowout_momentum,
            'garbage_time_risk': garbage_time_risk if 'garbage_time_risk' in locals() else balance_features.get('garbage_time_risk', None),
            'intensity_drop': intensity_drop,
            'current_lead': current_lead,
            'is_unbalanced': is_unbalanced_flag
        },
        'live_signals_used': {
            'home_fti': home_fti,
            'away_fti': away_fti,
            'diff_fti': diff_fti,
            'home_ts_live': home_ts_live,
            'away_ts_live': away_ts_live,
            'diff_ts_live': diff_ts_live,
            'run_active': run_active,
            'run_side': run_side,
            'run_strength': run_strength,
            # [NEW] FASE 2: Señales O/U específicas
            'home_ato_ratio': home_ato_ratio,
            'away_ato_ratio': away_ato_ratio,
            'diff_ato_ratio': diff_ato_ratio,
            'home_oreb_pct': home_oreb_pct,
            'away_oreb_pct': away_oreb_pct,
            'diff_oreb_pct': diff_oreb_pct,
            'home_dreb_pct': home_dreb_pct,
            'away_dreb_pct': away_dreb_pct,
            'diff_dreb_pct': diff_dreb_pct,
            'home_treb_diff': home_treb_diff,
            'away_treb_diff': away_treb_diff,
            # [NEW] TIME-AWARE: Información temporal precisa
            'time_info': time_info
        }
    }
    
    return adjusted_prediction, context_info
def build_pre_game_cache(home_team_name: str, away_team_name: str, trained_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    [OPTIMIZATION] PRE-GAME CACHE BUILDER - Computa features estáticas una sola vez por partido
    [OK] OPTIMIZACIÓN: Reduce CPU en live updates al ~70%
    [OK] INCLUYE: H2H features, team trends, alerts system
    """
    print("[BUILD] Construyendo caché pre-partido...")

    historical_df = trained_data['historical_df']
    features_used_in_model = trained_data['features_used']

    # Crear alerts system una sola vez (reutilizable)
    alerts_system = None
    # Sistema de alertas eliminado - funcionalidad integrada en el meta-modelo
    alerts_system = None
    pre_game_alerts = []

    # Filtrar historial de equipos (computación estática)
    home_history = historical_df[
        (historical_df['home_team'] == home_team_name) |
        (historical_df['away_team'] == home_team_name)
    ]
    away_history = historical_df[
        (historical_df['home_team'] == away_team_name) |
        (historical_df['away_team'] == away_team_name)
    ]

    cols_to_avg = ['points_scored', 'points_allowed', 'total_score'] + ADVANCED_STATS_COLS
    cols_to_ema = MOMENTUM_STATS_COLS

    # ==========================================
    # COMPUTACIÓN ESTÁTICA DE FEATURES PRE-GAME
    # ==========================================

    # Crear DataFrames de historial para cada equipo
    home_history_list = []
    for _, r in home_history.iterrows():
        is_home = r['home_team'] == home_team_name
        team_hist_data = {
            'points_scored': r['home_score'] if is_home else r['away_score'],
            'points_allowed': r['away_score'] if is_home else r['home_score'],
            'total_score': r['total_score'],
            'q1_points': 0,
            'q2_points': 0,
            'q3_points': 0,
            'q4_points': 0,
            **{stat: r.get(f'{"home" if is_home else "away"}_{stat}', np.nan) for stat in ADVANCED_STATS_COLS}
        }
        home_history_list.append(team_hist_data)
    home_history_df = pd.DataFrame(home_history_list)

    if not home_history_df.empty:
        home_history_df['win'] = (home_history_df['points_scored'] > home_history_df['points_allowed']).astype(int)
        home_history_df['plus_minus'] = home_history_df['points_scored'] - home_history_df['points_allowed']
        if FEATURES_IMPORTS_OK:
            momentum_metrics = calculate_momentum_metrics(home_history_df)
            for metric, value in momentum_metrics.items():
                home_history_df[metric] = value

    away_history_list = []
    for _, r in away_history.iterrows():
        is_home = r['home_team'] == away_team_name
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
        if FEATURES_IMPORTS_OK:
            momentum_metrics = calculate_momentum_metrics(away_history_df)
            for metric, value in momentum_metrics.items():
                away_history_df[metric] = value

    # Computar estadísticas rolling y EMA (ESTÁTICAS)
    if FEATURES_IMPORTS_OK:
        home_stats_5 = get_rolling_stats(home_history_df, 5, cols_to_avg)
        away_stats_5 = get_rolling_stats(away_history_df, 5, cols_to_avg)

        home_ema_advanced = get_enhanced_ema_stats(home_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)
        away_ema_advanced = get_enhanced_ema_stats(away_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)

        home_ema_5 = get_ema_stats(home_history_df, 5, cols_to_ema)
        away_ema_5 = get_ema_stats(away_history_df, 5, cols_to_ema)
    else:
        # Fallback básico si no hay funciones
        home_stats_5 = {col: safe_mean(home_history_df[col]) if col in home_history_df.columns else 0 for col in cols_to_avg}
        away_stats_5 = {col: safe_mean(away_history_df[col]) if col in away_history_df.columns else 0 for col in cols_to_avg}
        home_ema_5 = {col: safe_mean(home_history_df[col]) if col in home_history_df.columns else 0 for col in cols_to_ema}
        away_ema_5 = {col: safe_mean(away_history_df[col]) if col in away_history_df.columns else 0 for col in cols_to_ema}
        home_ema_advanced = {}
        away_ema_advanced = {}

    # Ensamblar features estáticas
    features_static = {}
    features_static.update({'home_' + k: v for k, v in home_stats_5.items()})
    features_static.update({'away_' + k: v for k, v in away_stats_5.items()})
    features_static.update({'home_' + k: v for k, v in home_ema_5.items()})
    features_static.update({'away_' + k: v for k, v in away_ema_5.items()})
    features_static.update({'home_' + k: v for k, v in home_ema_advanced.items()})
    features_static.update({'away_' + k: v for k, v in away_ema_advanced.items()})

    # Calcular diferencias estáticas
    for stat in cols_to_avg:
        features_static[f'diff_avg_{stat}_last_5'] = features_static.get(f'home_avg_{stat}_last_5', np.nan) - features_static.get(f'away_avg_{stat}_last_5', np.nan)
    for stat in cols_to_ema:
        features_static[f'diff_ema_{stat}_last_5'] = features_static.get(f'home_ema_{stat}_last_5', np.nan) - features_static.get(f'away_ema_{stat}_last_5', np.nan)

    # Calcular team trends para pace projection (ESTÁTICOS)
    team_trends = {'home': {}, 'away': {}}
    if FEATURES_IMPORTS_OK:
        try:
            home_trends = calculate_team_quarter_trends(home_history_df)
            away_trends = calculate_team_quarter_trends(away_history_df)
            team_trends = {'home': home_trends, 'away': away_trends}
        except Exception as e:
            print(f"[WARNING] Error calculando team trends: {e}")

    # Calcular H2H features (ESTÁTICOS)
    h2h_stats = {}
    try:
        from core.features import calculate_h2h_features
        h2h_stats = calculate_h2h_features(home_team_name, away_team_name, historical_df)
    except Exception as e:
        print(f"[WARNING] Error calculando H2H features: {e}")
        h2h_stats = {k: np.nan for k in ['h2h_avg_total_score', 'h2h_pace_avg', 'h2h_first_half_avg_total', 'h2h_second_half_surge_avg', 'h2h_comeback_freq']}

    # Agregar H2H al cache
    features_static.update(h2h_stats)

    # Crear cache completo
    pre_game_cache = {
        'features_static': features_static,
        'team_trends': team_trends,
        'h2h_stats': h2h_stats,
        'alerts_system': alerts_system,
        'pre_game_alerts': pre_game_alerts,
        'home_history_df': home_history_df,
        'away_history_df': away_history_df,
        'cache_timestamp': pd.Timestamp.now(),
        'home_team': home_team_name,
        'away_team': away_team_name
    }

    print(f"[OK] Caché pre-partido construido: {len(features_static)} features estáticas")
    return pre_game_cache


# ===========================================
# INTEGRACIÓN DEL META-MODELO UNIFICADO
# ===========================================


# Función original renombrada para evitar conflicto



def get_predictions_with_alerts(home_team_name: str, away_team_name: str,
                                q_scores: Dict[str, int], trained_data: Dict[str, Any],
                                pre_game_cache: Dict[str, Any] = None,
                                live_signals: Dict[str, float] = None,
                                silent_mode: bool = False) -> Tuple[float, List[Dict], Dict[str, Any]]:
    """
    Calcula predicciones Over/Under CON SISTEMA DE ALERTAS - SIN BETTING PROTECTION
    🚀 OPTIMIZADO: Soporta caché pre-partido para ~70% menos CPU en live updates
    """
    # Initialize variables to ensure they're always defined
    pre_game_alerts = []
    alerts_system = None

    model = trained_data['model']
    std_dev = trained_data['std_dev']
    historical_df = trained_data['historical_df']
    features_used_in_model = trained_data['features_used']

    # 🚀 FASE 1.2: VALIDAR CACHÉ APROPIADO PARA QUARTER STAGE
    # Determinar completion percentage para validar uso de caché
    time_info = dict(live_signals.get('time_info', {}) if live_signals else {})
    if time_info.get('clock_valid') is False:
        raise ValueError('Reloj live no valido: ' + time_info.get('clock_error', 'sin detalle'))
    period_seconds, _ = league_clock_settings(trained_data.get('league_name', 'NBA'))
    total_game_minutes = period_seconds * 4 / 60.0
    if time_info.get('total_game_minutes', total_game_minutes) != total_game_minutes:
        raise ValueError('El reloj no coincide con la liga seleccionada')
    time_info['total_game_minutes'] = total_game_minutes
    time_info['period_length'] = period_seconds
    completion_pct_early = time_info.get('completion_percentage', 0.0)
    
    # Determinar quarter stage preliminar para cache validation
    q3_total_check = q_scores.get('q3_home', 0) + q_scores.get('q3_away', 0)
    q4_total_check = q_scores.get('q4_home', 0) + q_scores.get('q4_away', 0)
    
    # [FIX] Determine quarter_stage explicitly to ensure availability
    if 'quarter_stage' in time_info:
        quarter_stage = time_info['quarter_stage']
    else:
        # Fallback logic if not provided in time_info
        if q4_total_check > 0:
            quarter_stage = 'q4_progress'
        elif q3_total_check > 0:
            quarter_stage = 'q3_progress'
        elif completion_pct_early >= 50:
            quarter_stage = 'halftime'
        else:
            quarter_stage = 'q1_or_q2'
    
    # Disable cache si estamos en Q3+ (>50% juego completado) o si Q3 tiene datos
    use_cache = False
    cache_disabled_reason = None
    
    if pre_game_cache is not None:
        if completion_pct_early >= 50:
            cache_disabled_reason = f"completion >= 50% ({completion_pct_early:.1f}%)"
        elif q3_total_check > 0:
            cache_disabled_reason = "Q3 data present (late game)"
        elif q4_total_check > 0:
            cache_disabled_reason = "Q4 data present (very late game)"
        else:
            use_cache = True
    
    # [FIX] Initialize base features from cache if available, regardless of use_cache flag
    # This ensures we have the static pre-game features even if we are doing a fresh live update
    if pre_game_cache is not None:
        features = pre_game_cache.get('features_static', {}).copy()
        home_history_df = pre_game_cache.get('home_history_df')
        away_history_df = pre_game_cache.get('away_history_df')
        team_trends = pre_game_cache.get('team_trends', {'home': {}, 'away': {}})
        h2h_stats = pre_game_cache.get('h2h_stats', {})
    else:
        features = {}
        home_history_df = None
        away_history_df = None
        team_trends = {'home': {}, 'away': {}}
        h2h_stats = {}

    if use_cache and pre_game_cache is not None:
        print("[CACHE] Usando caché pre-partido - computación optimizada")
        alerts_system = pre_game_cache.get('alerts_system')
        pre_game_alerts = pre_game_cache.get('pre_game_alerts', [])

    # [NEW] FASE 0: CALCULAR TIEMPO JUGADO (CRÍTICO PARA PULSE ENGINE)
    # 🆕 USAR MINUTOS EXACTOS DE LA API EN LUGAR DE ESTIMACIONES
    # Priorizar minutes_played de time_info si está disponible
    if time_info and 'minutes_played' in time_info:
        minutes_assumed = time_info['minutes_played']
        # print(f"[OK] Using API time data: {minutes_assumed:.1f} minutes played")
    else:
        # Fallback: Calcular minutos jugados asumidos por liga (NBA=12, WNBA/NBL/EURO=10 min/cuarto)
        try:
            ln_canon = _canonicalize_league_name(trained_data.get('league_name', 'NBA'))
        except Exception:
            ln_canon = 'NBA'
        qlen = 10 if any(k in ln_canon.upper() for k in ['WNBA','NBL','EURO']) else 12
        if quarter_stage == 'halftime':
            minutes_assumed = qlen * 2  # 24 min played, 24 min left
        elif quarter_stage == 'q3_progress':
            minutes_assumed = int(qlen * 2.5)  # ~30 min played, ~6-12 min left
        elif quarter_stage == 'q3_just_ended':
            minutes_assumed = qlen * 3  # 36 min played, 12 min left (1 quarter)
        elif quarter_stage == 'q4_progress':
            minutes_assumed = int(qlen * 3.5)  # 36+ min played, <12 min left
        else:
            minutes_assumed = qlen * 2
        # print(f"[WARNING] Using estimated time (no API data): {minutes_assumed} minutes")

    # [NEW] FASE 0.5: CALCULAR PACE & BALANCE METRICS (Requerido para Interaction Features)
    live_pace_metrics = {}
    balance_features = {}
    try:
        from core.live_features import calculate_live_pace_metrics, calculate_game_balance_features
        
        # Calcular métricas de ritmo
        live_pace_metrics = calculate_live_pace_metrics(
            q_scores, quarter_stage, team_trends, time_info
        )
        
        # Calcular métricas de balance
        balance_features = calculate_game_balance_features(
            q_scores, time_info,
            live_pace_metrics,
            trained_data.get('league_name', 'NBA')
        )
        
        features.update(live_pace_metrics)
        features.update(balance_features)
        
    except Exception as e:
        print(f"[WARNING] Error calculando Pace/Balance metrics: {e}")
        live_pace_metrics = {'live_pace_estimate': 100.0, 'enhanced_pace_estimate': 100.0}
        balance_features = {'game_balance_score': 0.5, 'is_potential_blowout': 0}
    
    # Initialize container for live stats
    api_stats_to_add = {}

    # Reutilizar alerts system si existe
    api_stats_to_add['home_field_goals_made'] = live_signals.get('current_stats', {}).get('home_field_goals_made', 0)
    api_stats_to_add['away_field_goals_made'] = live_signals.get('current_stats', {}).get('away_field_goals_made', 0)
    api_stats_to_add['home_field_goals_attempted'] = live_signals.get('current_stats', {}).get('home_field_goals_attempted', 0)
    api_stats_to_add['away_field_goals_attempted'] = live_signals.get('current_stats', {}).get('away_field_goals_attempted', 0)
    
    api_stats_to_add['home_2_point_field_goals_made'] = live_signals.get('current_stats', {}).get('home_2_point_field_goals_made', 0)
    api_stats_to_add['away_2_point_field_goals_made'] = live_signals.get('current_stats', {}).get('away_2_point_field_goals_made', 0)
    
    api_stats_to_add['home_3_point_field_goals_made'] = live_signals.get('current_stats', {}).get('home_3_point_field_goals_made', 0)
    api_stats_to_add['away_3_point_field_goals_made'] = live_signals.get('current_stats', {}).get('away_3_point_field_goals_made', 0)
    
    api_stats_to_add['home_free_throws_made'] = live_signals.get('current_stats', {}).get('home_free_throws_made', 0)
    api_stats_to_add['away_free_throws_made'] = live_signals.get('current_stats', {}).get('away_free_throws_made', 0)
    api_stats_to_add['home_free_throws_attempted'] = live_signals.get('current_stats', {}).get('home_free_throws_attempted', 0)
    api_stats_to_add['away_free_throws_attempted'] = live_signals.get('current_stats', {}).get('away_free_throws_attempted', 0)
    
    # Estadísticas de rebotes
    api_stats_to_add['home_total_rebounds'] = live_signals.get('current_stats', {}).get('home_total_rebounds', 0)
    api_stats_to_add['away_total_rebounds'] = live_signals.get('current_stats', {}).get('away_total_rebounds', 0)
    api_stats_to_add['home_offensive_rebounds'] = live_signals.get('current_stats', {}).get('home_offensive_rebounds', 0)
    api_stats_to_add['away_offensive_rebounds'] = live_signals.get('current_stats', {}).get('away_offensive_rebounds', 0)
    api_stats_to_add['home_defensive_rebounds'] = live_signals.get('current_stats', {}).get('home_defensive_rebounds', 0)
    api_stats_to_add['away_defensive_rebounds'] = live_signals.get('current_stats', {}).get('away_defensive_rebounds', 0)
    
    # Estadísticas de juego
    api_stats_to_add['home_assists'] = live_signals.get('current_stats', {}).get('home_assists', 0)
    api_stats_to_add['away_assists'] = live_signals.get('current_stats', {}).get('away_assists', 0)
    api_stats_to_add['home_turnovers'] = live_signals.get('current_stats', {}).get('home_turnovers', 0)
    api_stats_to_add['away_turnovers'] = live_signals.get('current_stats', {}).get('away_turnovers', 0)
    api_stats_to_add['home_steals'] = live_signals.get('current_stats', {}).get('home_steals', 0)
    api_stats_to_add['away_steals'] = live_signals.get('current_stats', {}).get('away_steals', 0)
    api_stats_to_add['home_blocks'] = live_signals.get('current_stats', {}).get('home_blocks', 0)
    api_stats_to_add['away_blocks'] = live_signals.get('current_stats', {}).get('away_blocks', 0)
    api_stats_to_add['home_personal_fouls'] = live_signals.get('current_stats', {}).get('home_personal_fouls', 0)
    api_stats_to_add['away_personal_fouls'] = live_signals.get('current_stats', {}).get('away_personal_fouls', 0)
    
    # [NEW] AGREGAR CARACTERÍSTICAS CRÍTICAS QUE EL SISTEMA ESPERA EXACTAMENTE
    # Estas son las características que el sistema busca en CRITICAL_FEATURES
    current_stats_data = live_signals.get('current_stats', {})
    
    # Características de pace (ya vienen de live_pace_metrics pero las aseguramos)
    if 'live_pace_estimate' not in features:
        api_stats_to_add['live_pace_estimate'] = live_pace_metrics.get('live_pace_estimate', 100)
    if 'enhanced_pace_estimate' not in features:
        api_stats_to_add['enhanced_pace_estimate'] = live_pace_metrics.get('enhanced_pace_estimate', 100)
    
    # Características de balance y contexto
    if 'game_balance_score' not in features:
        api_stats_to_add['game_balance_score'] = balance_features.get('game_balance_score', 0.5)
    if 'is_potential_blowout' not in features:
        api_stats_to_add['is_potential_blowout'] = balance_features.get('is_potential_blowout', 0)
    if 'garbage_time_risk' not in features:
        api_stats_to_add['garbage_time_risk'] = balance_features.get('garbage_time_risk', 0.0)
    if 'intensity_drop_factor' not in features:
        api_stats_to_add['intensity_drop_factor'] = balance_features.get('intensity_drop_factor', 1.0)
    
    # [NEW] CARACTERÍSTICAS ESPECÍFICAS QUE FALTAN (según el log)
    # El sistema busca exactamente estos nombres en CRITICAL_FEATURES
    if 'personal_fouls' not in features:
        # Agregar faltas combinadas (el sistema espera esta característica)
        total_fouls = current_stats_data.get('home_personal_fouls', 0) + current_stats_data.get('away_personal_fouls', 0)
        api_stats_to_add['personal_fouls'] = total_fouls
    
    if 'field_goals_made' not in features:
        # Agregar field goals combinados
        total_fg_made = current_stats_data.get('home_field_goals_made', 0) + current_stats_data.get('away_field_goals_made', 0)
        api_stats_to_add['field_goals_made'] = total_fg_made
        
    if 'field_goals_attempted' not in features:
        # Agregar field goals intentados combinados
        total_fg_attempted = current_stats_data.get('home_field_goals_attempted', 0) + current_stats_data.get('away_field_goals_attempted', 0)
        api_stats_to_add['field_goals_attempted'] = total_fg_attempted
        
    if 'free_throws_made' not in features:
        # Agregar free throws combinados
        total_ft_made = current_stats_data.get('home_free_throws_made', 0) + current_stats_data.get('away_free_throws_made', 0)
        api_stats_to_add['free_throws_made'] = total_ft_made
        
    if 'free_throws_attempted' not in features:
        # Agregar free throws intentados combinados
        total_ft_attempted = current_stats_data.get('home_free_throws_attempted', 0) + current_stats_data.get('away_free_throws_attempted', 0)
        api_stats_to_add['free_throws_attempted'] = total_ft_attempted
        
    if 'total_rebounds' not in features:
        # Agregar rebotes combinados
        total_rebounds = current_stats_data.get('home_total_rebounds', 0) + current_stats_data.get('away_total_rebounds', 0)
        api_stats_to_add['total_rebounds'] = total_rebounds
        
    if 'assists' not in features:
        # Agregar asistencias combinadas
        total_assists = current_stats_data.get('home_assists', 0) + current_stats_data.get('away_assists', 0)
        api_stats_to_add['assists'] = total_assists
        
    if 'turnovers' not in features:
        # Agregar pérdidas combinadas
        total_turnovers = current_stats_data.get('home_turnovers', 0) + current_stats_data.get('away_turnovers', 0)
        api_stats_to_add['turnovers'] = total_turnovers
        
    if 'steals' not in features:
        # Agregar robos combinados
        total_steals = current_stats_data.get('home_steals', 0) + current_stats_data.get('away_steals', 0)
        api_stats_to_add['steals'] = total_steals
        
    if 'blocks' not in features:
        # Agregar bloqueos combinados
        total_blocks = current_stats_data.get('home_blocks', 0) + current_stats_data.get('away_blocks', 0)
        api_stats_to_add['blocks'] = total_blocks
    
    # Agregar estadísticas de la API a las características del modelo
    features.update(api_stats_to_add)
    
    # Logging de integración exitosa
    non_zero_stats = {k: v for k, v in api_stats_to_add.items() if v > 0}
    critical_added = {k: v for k, v in api_stats_to_add.items() if k in [
        'personal_fouls', 'field_goals_made', 'field_goals_attempted', 'free_throws_made',
        'free_throws_attempted', 'total_rebounds', 'assists', 'turnovers', 'steals', 'blocks',
        'live_pace_estimate', 'enhanced_pace_estimate', 'game_balance_score', 'garbage_time_risk'
    ] and v > 0}
    
    if non_zero_stats:
        pass
        # print(f"[OK] [API INTEGRATION] {len(non_zero_stats)} estadísticas de la API integradas en características del modelo")
        # print(f"[OK] [CRITICAL FEATURES] {len(critical_added)} características críticas agregadas")
        # print(f"   Ejemplo: FG {api_stats_to_add.get('field_goals_made', 0)}, "
        #       f"FT {api_stats_to_add.get('free_throws_made', 0)}, "
        #       f"REB {api_stats_to_add.get('total_rebounds', 0)}, "
        #       f"PF {api_stats_to_add.get('personal_fouls', 0)}")
    else:
        print(f"[WARNING] [API INTEGRATION] Las estadísticas de la API están en cero - posible problema de extracción")



    # [NEW] FASE 1: CALCULAR ADVANCED METRICS (PULSE ENGINE)
    # Inicializar adv_metrics para evitar UnboundLocalError
    adv_metrics = {}
    try:
        from analysis.context_features import calculate_advanced_metrics
        # [FIX] Usar current_stats_data (datos raw con splits) para métricas avanzadas
        # api_stats_to_add solo tiene totales combinados, lo cual rompía el cálculo de TS%
        stats_source = current_stats_data if 'current_stats_data' in locals() and current_stats_data else live_signals.get('current_stats', {})
        
        # Asegurar time_info_dict
        if 'time_info_dict' not in locals():
             time_info_dict = {
                'total_remaining_minutes': total_game_minutes - minutes_assumed,
                'total_game_minutes': total_game_minutes,
                'period_length': period_seconds,
                'current_pace_per_minute': time_info.get('current_pace_per_minute', 0.0),
                'quarter_stage': quarter_stage,
                'total_minutes_played': float(minutes_assumed)
            }

        adv_metrics = calculate_advanced_metrics(stats_source, time_info_dict)
        features.update(adv_metrics)
        # print(f"[OK] Pulse Engine metrics calculadas: {len(adv_metrics)} features añadidas")
    except Exception as e:
        print(f"[WARNING] Error calculando Pulse Engine metrics: {e}")
        # Fallback values
        adv_metrics = {
            'possessions_avg': 100.0,
            'ortg_home': 110.0,
            'ortg_away': 110.0,
            'true_pace': 100.0,
            'efficiency_gap': 0.0,
            'ts_home': 0.55,
            'ts_away': 0.55
        }
        features.update(adv_metrics)

    # [NEW] FASE 1: CALCULAR INTERACTION FEATURES LIVE
    try:
        from core.live_features import calculate_live_interaction_features
        interaction_features = calculate_live_interaction_features(
            live_pace_metrics, balance_features, home_history_df
        )
        features.update(interaction_features)
        # print(f"[OK] Interaction features calculadas: {len(interaction_features)} features añadidas")
    except Exception as e:
        print(f"[WARNING] Error calculando interaction features: {e}")
        # Fallback: añadir valores neutros por defecto
        default_interactions = {
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
            'lead_swing_intensity': 0.0,
            'ft_run_active': False,
            'ft_run_strength': 0.0,
            'ft_clustering_score': 0.0,
            'live_volatility_acceleration': 0.5
        }
        features.update(default_enhanced)

    # Establecer current_total real para salvaguarda de ritmo
    try:
        if quarter_stage == 'halftime':
            balance_features['current_total'] = features.get('halftime_total', 0)
        elif quarter_stage == 'q3_progress':
            balance_features['current_total'] = features.get('halftime_total', 0) + features.get('q3_total', 0)
        elif quarter_stage == 'q3_just_ended':
            balance_features['current_total'] = features.get('q3_end_total', 0)
        elif quarter_stage == 'q4_progress':
            balance_features['current_total'] = features.get('q3_end_total', 0) + q4_total_now
        else:
            # Fallback conservador si surge otro estado
            balance_features['current_total'] = features.get('q3_end_total', features.get('halftime_total', 0))
    except Exception:
        # No bloquear si algo falla
        pass

    # Initialize variables that might be used in return statement
    live_alerts = []
    home_summary = {'avg_by_quarter': {'q1': 25, 'q2': 25, 'q3': 25, 'q4': 25}, 'second_half_tendency': 0, 'recovery_ability': 0.5}
    away_summary = {'avg_by_quarter': {'q1': 25, 'q2': 25, 'q3': 25, 'q4': 25}, 'second_half_tendency': 0, 'recovery_ability': 0.5}

    # [DEBUG] Telemetría crítica Q2+ para diagnóstico de dips en Q3 en progreso
    try:
        q1_t = features['q1_total']; q2_t = features['q2_total']; q3_t = features['q3_total']
        halftime_t = features['halftime_total']; q3_end_t = features['q3_end_total']
        q4_home = q_scores.get('q4_home', 0); q4_away = q_scores.get('q4_away', 0)
        q4_t = q4_home + q4_away
        q3_in_progress = (q3_t > 0) and (q4_t == 0)
        q3_in_progress = (q3_t > 0) and (q4_t == 0)
        # (minutes_assumed calculation moved to FASE 0)
        quarters_completed_for_alerts = ['q1','q2'] if (quarter_stage in ['halftime', 'q3_progress']) else ['q1','q2','q3']

        debug_logs = {
            'quarter_stage': quarter_stage,
            'minutes_played_assumed': minutes_assumed,
            'q_scores_snapshot': dict(q_scores),
            'totals': {
                'q1_total': q1_t,
                'q2_total': q2_t,
                'q3_total': q3_t,
                'halftime_total': halftime_t,
                'q3_end_total': q3_end_t,
                'quarter_variance': float(features.get('quarter_variance', np.nan))
            },
            'live_pace_metrics': dict(live_pace_metrics),
            'balance_signals': {
                'current_lead': balance_features.get('current_lead', None),
                'is_game_unbalanced': balance_features.get('is_game_unbalanced', 0),
                'lead_stability': balance_features.get('lead_stability', None),
                'is_potential_blowout': balance_features.get('is_potential_blowout', None),
                'intensity_drop_factor': balance_features.get('intensity_drop_factor', None),
                'garbage_time_risk': balance_features.get('garbage_time_risk', None)
            },
            'std_dev': float(std_dev),
            'quarters_completed_for_alerts': quarters_completed_for_alerts,
            'q3_in_progress_flag': q3_in_progress
        }

        # Imprimir solo desde Q2+
        # if q2_t > 0:
            # print("\nDEBUG LIVE (Q2+):")
            # print(f"   • Stage: {quarter_stage} | min asumidos: {minutes_assumed}")
            # print(f"   • Totales: Q1={q1_t}, Q2={q2_t}, Q3={q3_t} | HT={halftime_t} | VarCuartos={features.get('quarter_variance', np.nan):.3f}")
            # lp = live_pace_metrics.get('live_pace_estimate', 'n/a')
            # ep = live_pace_metrics.get('enhanced_pace_estimate', live_pace_metrics.get('live_pace_estimate', 'n/a'))
            # print(f"   • Pace: live={lp} | enhanced={ep}")
            # print(f"   • Balance: lead={balance_features.get('current_lead', 'n/a')}, unbalanced={balance_features.get('is_game_unbalanced', 'n/a')}, gtrisk={balance_features.get('garbage_time_risk', 'n/a')}")
            # print(f"   • Qtrs para alertas: {quarters_completed_for_alerts} | σ={std_dev:.2f}")
            # if quarter_stage == 'q3_progress':
            #     print(f"   [INFO] Q3 en progreso — asumiendo {minutes_assumed}m para pace/ajustes (sin activar lógica de fin de Q3)")
    except Exception as _dbg_err:
        # print(f"Error en debug live: {_dbg_err}")

        # Sistema de alertas eliminado - funcionalidad integrada en el meta-modelo
        live_alerts = []
        home_summary = {'avg_by_quarter': {'q1': 25, 'q2': 25, 'q3': 25, 'q4': 25}, 'second_half_tendency': 0, 'recovery_ability': 0.5}
        away_summary = {'avg_by_quarter': {'q1': 25, 'q2': 25, 'q3': 25, 'q4': 25}, 'second_half_tendency': 0, 'recovery_ability': 0.5}

    # [NEW] FASE 2: Telemetría de señales live enfocadas en O/U
    if live_signals:
        print("\n[SEÑALES LIVE O/U (FASE 2)]:")
        print(f"   [FTI] - {home_team_name}: {live_signals.get('home_fti', 0.5):.3f}")
        print(f"   [FTI] - {away_team_name}: {live_signals.get('away_fti', 0.5):.3f}")
        
        # Use robust TS% from advanced metrics if available, fallback to live_signals
        ts_home = adv_metrics.get('ts_home', 0)
        if ts_home == 0: ts_home = live_signals.get('home_ts_live', 0.5)
        
        ts_away = adv_metrics.get('ts_away', 0)
        if ts_away == 0: ts_away = live_signals.get('away_ts_live', 0.5)
        print(f"   [TS%] - {home_team_name}: {ts_home:.3f}")
        print(f"   [TS%] - {away_team_name}: {ts_away:.3f}")
        
        print(f"   [Run detector]: {'ACTIVO' if live_signals.get('run_active', False) else 'Inactivo'}")
        if live_signals.get('run_active', False):
            print(f"      Lado: {live_signals.get('run_side', 'none')} | Fuerza: {live_signals.get('run_strength', 0.0):.2f}")

        # [NEW] FASE 2: Mostrar señales O/U específicas
        print(f"   [A/TO] - {home_team_name}: {live_signals.get('home_ato_ratio', 1.0):.2f}")
        print(f"   [A/TO] - {away_team_name}: {live_signals.get('away_ato_ratio', 1.0):.2f}")
        print(f"   [TREB diff] - {home_team_name}: {live_signals.get('home_treb_diff', 0.0):.2f}")
        print(f"   [OREB%] - {home_team_name}: {live_signals.get('home_oreb_pct', 0.25):.1%}")
        print(f"   [OREB%] - {away_team_name}: {live_signals.get('away_oreb_pct', 0.25):.1%}")

    if not silent_mode:
        if balance_features.get('is_game_unbalanced', 0) == 1:
            print(f"\n[ALERT] ALERTA DE DESBALANCE:")
            print(f"   Juego muy desigual - posible impacto en total final")
            print(f"   Balance Score: {balance_features['game_balance_score']:.3f}")

        if balance_features.get('intensity_drop_factor', 1.0) < 0.8:
            print(f"\n[ALERT] ALERTA DE INTENSIDAD:")
            print(f"   Caída de intensidad detectada")
            print(f"   Factor: {balance_features['intensity_drop_factor']:.3f}")

    # Preparar datos para el modelo
    X_pred = pd.DataFrame([features], columns=features_used_in_model)
    
    impute_values = trained_data.get('impute_values', {})
    
    # [NEW] IMPROVEMENT 2: Monitor missing features and reduce confidence
    CRITICAL_FEATURES = [
        'live_pace_estimate', 'enhanced_pace_estimate',
        'game_balance_score', 'is_potential_blowout',
        'garbage_time_risk', 'intensity_drop_factor',
        # [NEW] ESTADÍSTICAS BÁSICAS CRÍTICAS DESDE API
        'personal_fouls', 'field_goals_made', 'field_goals_attempted',
        'free_throws_made', 'free_throws_attempted', 'total_rebounds',
        'assists', 'turnovers', 'steals', 'blocks'
    ]

    print(f"\n[Verificando {len(X_pred.columns)} caracteristicas...]")
    missing_count = 0
    missing_features = []
    critical_missing = 0

    for col in X_pred.columns:
        if X_pred[col].isnull().any():
            missing_count += 1
            missing_features.append(col)
            if col in CRITICAL_FEATURES:
                critical_missing += 1

            if col in impute_values and not np.isnan(impute_values[col]):
                X_pred[col] = X_pred[col].fillna(impute_values[col])
            else:
                if col in trained_data['historical_df'].columns:
                    fallback = trained_data['historical_df'][col].mean()
                    if np.isnan(fallback):
                        fallback = 0.0
                    X_pred[col] = X_pred[col].fillna(fallback)
                else:
                    X_pred[col] = X_pred[col].fillna(0.0)

    # Calculate confidence reduction based on missing critical features
    confidence_multiplier = 1.0
    if critical_missing > 0:
        # Reduce confidence by 10% per critical missing feature, max 50%
        confidence_reduction = min(0.5, critical_missing * 0.1)
        confidence_multiplier = 1.0 - confidence_reduction
        print(f"[Confianza reducida: {confidence_multiplier:.1f} - {critical_missing} features críticas faltantes]")

    # [PHASE 1.3] STATS VALIDATION ALERT
    if missing_count > 0:
        imputation_percentage = (missing_count / len(X_pred.columns)) * 100
        print(f"[Total de caracteristicas imputadas: {missing_count}/{len(X_pred.columns)} ({imputation_percentage:.1f}%)]")
        
        # [NEW] MEJORA: Logging detallado de características críticas faltantes
        missing_critical_features = [f for f in missing_features if f in CRITICAL_FEATURES]
        if missing_critical_features:
            print(f"[CRITICAL DEBUG] Características críticas faltantes: {missing_critical_features}")
            
            # Identificar qué categoría de características críticas falta más
            pace_missing = [f for f in missing_critical_features if 'pace' in f]
            balance_missing = [f for f in missing_critical_features if 'balance' in f or 'blowout' in f or 'garbage' in f]
            intensity_missing = [f for f in missing_critical_features if 'intensity' in f or 'drop' in f]
            
            if pace_missing:
                print(f"[PACE DEBUG] Faltan {len(pace_missing)} características de pace: {pace_missing}")
            if balance_missing:
                print(f"[BALANCE DEBUG] Faltan {len(balance_missing)} características de balance: {balance_missing}")
            if intensity_missing:
                print(f"[INTENSITY DEBUG] Faltan {len(intensity_missing)} características de intensidad: {intensity_missing}")
        
        # CRITICAL: Alert si >50% features imputadas
        if imputation_percentage > 50:
            print(f"[CRITICAL] ALERTA CRÍTICA: {imputation_percentage:.1f}% features imputadas (>50% threshold)")
            print(f"   Sistema operando con datos muy limitados - REDUCIR CONFIANZA")
            print(f"   Considerar NO usar esta predicción para decisiones críticas")
            # [NEW] MEJORA: Sugerir modo conservador automáticamente
            print(f"[CONSERVATIVE] MODO CONSERVADOR Activado automáticamente por alta imputación")
        elif imputation_percentage > 30:
            print(f"[WARNING] WARNING: {imputation_percentage:.1f}% features imputadas (>30%)")
            print(f"   Data quality subóptima - usar predicción con cautela")
            # [NEW] MEJORA: Sugerir validación de datos API
            print(f"[API VALIDATION] Se recomienda verificar conexión con SofaScore API")
        
        if critical_missing > 0:
            print(f"[WARNING] Features CRÍTICAS faltantes: {critical_missing}/{len(CRITICAL_FEATURES)} - reducir confianza en predicción]")
            print(f"   Missing: {', '.join([f for f in missing_features if f in CRITICAL_FEATURES][:5])}")
    else:
        print("[Todas las caracteristicas disponibles - sin imputacion necesaria]")

    if X_pred.isnull().values.any():
        print("\n[WARNING] Advertencia: Aún quedan valores NaN después de imputar – rellenando con 0s.")
        X_pred = X_pred.fillna(0.0)

        
    
    # 4. Bookie line (Restored)
    bookie_line = trained_data.get('bookie_line', None)
    if bookie_line is None:
        bookie_line = features.get('bookie_line', None)

    # 5. Team trends (Restored/Ensured)
    # team_trends variable is available from earlier in the function

    # Predicción del modelo BASE (necesaria para el flujo y como input legacy)
    final_total_pred = model.predict(X_pred)[0]
    
    # [NEW] VISIONARY ROADMAP: PHASE 1 (The Pulse Engine)
    # Metrics already calculated above
    
    # Integrar métricas avanzadas en live_signals para que el metamodelo las reciba
    if live_signals is None: live_signals = {}
    live_signals.update(adv_metrics)
    
    # Mostrar métricas en consola
    if not silent_mode:
        print("\n[THE PULSE ENGINE - SYNTHETIC ANALYTICS]:")
        print(f"   [POSS] Home: {adv_metrics.get('possessions_home', 0):.1f} | Away: {adv_metrics.get('possessions_away', 0):.1f}")
        print(f"   [ORTG] Home: {adv_metrics.get('ortg_home', 0):.1f} | Away: {adv_metrics.get('ortg_away', 0):.1f}")
        print(f"   [PACE] True Pace: {adv_metrics.get('true_pace', 0):.1f} (Poss/48m)")
        print(f"   [EFF GAP] {adv_metrics.get('efficiency_gap', 0):+.1f} (Home advantage)")

    print(f"\n[METAMODEL AI v2 - CHRONOS EDITION]")
    print(f"   Invocando Metamodel Chronos con Pulse Engine...")
    
    # Preparar datos para el Metamodel
    q_scores_dict = {
        'q1_home': q_scores.get('q1_home', 0),
        'q1_away': q_scores.get('q1_away', 0),
        'q2_home': q_scores.get('q2_home', 0),
        'q2_away': q_scores.get('q2_away', 0),
        'q3_home': q_scores.get('q3_home', 0),
        'q3_away': q_scores.get('q3_away', 0),
        'q4_home': q_scores.get('q4_home', 0),
        'q4_away': q_scores.get('q4_away', 0)
    }
    
    time_info_dict = {
        'total_remaining_minutes': total_game_minutes - minutes_assumed,
        'total_game_minutes': total_game_minutes,
        'period_length': period_seconds,
        'clock_valid': time_info.get('clock_valid'),
        'current_pace_per_minute': time_info.get('current_pace_per_minute', 0.0),
        'quarter_stage': quarter_stage,
        'total_minutes_played': float(minutes_assumed),  # Now uses API data when available
        'played_seconds': float(minutes_assumed) * 60  # Required for The Pacer
    }
    
    # La predicción final viene SOLO del Metamodel Chronos
    from analysis.metamodel_integration import aplicar_metamodelo_ai
    final_prediction, meta_metadata = aplicar_metamodelo_ai(
        base_prediction=final_total_pred,
        q_scores=q_scores_dict,
        balance_features=balance_features,
        live_pace_metrics=live_pace_metrics,
        time_info=time_info_dict,
        live_signals=live_signals, # Ahora incluye adv_metrics
        team_trends=team_trends,
        bookie_line=bookie_line,
        team_stats={}
    )
    
    # Mostrar resultado del Metamodel
    adjustment = meta_metadata.get('adjustment', 0.0)
    chronos_pred = meta_metadata.get('chronos_prediction', final_prediction)
    sign = "+" if adjustment > 0 else ""
    print(f"   [Chronos Prediction]: {chronos_pred:.1f} pts")
    print(f"   [Diff vs Base]: {sign}{adjustment:.1f} pts")

    # La predicción final es la del Metamodel
    final_prediction = final_prediction
    
    # Context info para logs (simplificado)
    context_info = {
        'applied_adjustments': [f"Metamodel AI adjustment: {sign}{adjustment:.1f} pts"],
        'meta_metadata': meta_metadata
    }

    print(f"   [PREDICCION FINAL]: {final_prediction:.0f} pts")

    # [NEW] SAFE GUARD: "Safe Projection" Logic
    # Calcular una proyección basada puramente en ritmo actual para validar al modelo
    try:
        safe_proj_val = _calculate_safe_projection(
            current_total=balance_features.get('current_total', 0),
            minutes_played=minutes_assumed,
            quarter_stage=quarter_stage,
            current_pace=time_info.get('current_pace_per_minute', 0),
            halftime_total=features.get('halftime_total', None),
            total_game_minutes=total_game_minutes
        )
        
        if safe_proj_val > 0:
            print(f"\n[SAFE PROJECTION] Proyección basada en ritmo: {safe_proj_val:.1f} pts")
            
            # Verificar desviación extrema del Metamodelo
            diff_safe = final_prediction - safe_proj_val
            
            # [FIX] Lógica de salvaguarda dinámica según calidad de datos
            # Si faltan features (missing_count > 0), confiamos MÁS en la proyección segura (matemática)
            # y MENOS en el modelo (que está imputando datos).
            
            threshold = 30.0
            safe_weight = 0.6
            
            if missing_count > 0:
                threshold = 15.0  # Umbral más estricto si faltan datos
                safe_weight = 0.8 # Confiar 80% en la proyección matemática
                print(f"[SAFEGUARD] Modo estricto activado por falta de datos (Threshold: {threshold}pts, Weight: {safe_weight})")

            # Si el modelo se desvía más del umbral
            if abs(diff_safe) > threshold:
                print(f"[WARNING] Desviación extrema detectada ({diff_safe:+.1f} pts vs Safe Projection)")
                print(f"   Activando salvaguarda de estabilidad...")
                
                # Blending dinámico
                blended_pred = (safe_proj_val * safe_weight) + (final_prediction * (1.0 - safe_weight))
                print(f"   [ADJUSTED] Predicción corregida por salvaguarda: {final_prediction:.1f} -> {blended_pred:.1f}")
                
                final_prediction = blended_pred
                
                # Actualizar contexto
                context_info['safeguard_active'] = True
                context_info['original_prediction'] = chronos_pred
                context_info['safe_projection'] = safe_proj_val
    except Exception as e:
        print(f"[WARNING] Error en cálculo de Safe Projection: {e}")
    
    # Recalcular score_diff para el return
    home_score = q_scores.get('q1_home', 0) + q_scores.get('q2_home', 0) + q_scores.get('q3_home', 0) + q_scores.get('q4_home', 0)
    away_score = q_scores.get('q1_away', 0) + q_scores.get('q2_away', 0) + q_scores.get('q3_away', 0) + q_scores.get('q4_away', 0)
    score_diff = abs(home_score - away_score)

    # Generar múltiples líneas con probabilidades (usando std_dev original)
    predictions = []
    center_line = final_prediction
    for i in range(-2, 3):
        line = round(center_line) + i * (std_dev / 4)
        z_score = (line - center_line) / std_dev
        under_prob = stats.norm.cdf(z_score) * 100
        over_prob = 100 - under_prob

        predictions.append({
            'line': line,
            'over_prob': over_prob,
            'under_prob': under_prob
        })

    # [NEW] IMPROVEMENT 4: Analyze and log high edge games + unrealistic predictions
    high_edge_alert = None
    unrealistic_prediction_alert = None
    
    # Reality check legacy eliminado - Metamodel se encarga de la coherencia


    if predictions:
        # Calculate edge for center line (most reliable)
        center_pred = predictions[2]  # Index 2 is center (i=0)
        max_edge = max(abs(center_pred['over_prob'] - 50), abs(center_pred['under_prob'] - 50)) * 2  # Convert to percentage

        if max_edge > 30.0:
            edge_level = "EXTREME" if max_edge > 40.0 else "HIGH"
            stronger_side = "OVER" if center_pred['over_prob'] > center_pred['under_prob'] else "UNDER"

            print(f"[EDGE] {edge_level} EDGE DETECTED: {max_edge:.1f}% edge on {stronger_side}")
            print("   [TRACK] Track this game's actual outcome for performance analysis")

            high_edge_alert = {
                'edge_percentage': max_edge,
                'edge_level': edge_level,
                'recommended_side': stronger_side,
                'tracking_required': True
            }

        # Check for unrealistic predictions (gap between prediction and realistic projection)
        try:
            current_total = balance_features.get('current_total', 0)
            remaining_minutes = time_info.get('total_remaining_minutes', 48.0)
            current_pace = time_info.get('current_pace_per_minute', 0.0)

            if current_total > 0 and remaining_minutes > 0 and current_pace > 0:
                realistic_projection = _calculate_pace_based_projection(
                    current_total, remaining_minutes, current_pace, league_name
                )

                prediction_gap = final_prediction - realistic_projection

                if abs(prediction_gap) > 25:  # Significant deviation
                    gap_level = "EXTREME" if abs(prediction_gap) > 35 else "HIGH"
                    direction = "OVER" if prediction_gap > 0 else "UNDER"

                    print(f"[WARNING] {gap_level} PREDICTION GAP: {abs(prediction_gap):.1f} pts {direction} vs pace projection")
                    print(f"   Pred: {final_prediction:.1f} vs Pace proj: {realistic_projection:.1f} (current: {current_total:.0f}, pace: {current_pace:.2f})")

                    unrealistic_prediction_alert = {
                        'gap_points': prediction_gap,
                        'gap_level': gap_level,
                        'direction': direction,
                        'realistic_projection': realistic_projection,
                        'current_total': current_total,
                        'current_pace': current_pace,
                        'remaining_minutes': remaining_minutes
                    }
        except Exception as e:
            # Don't break if reality check fails
            pass

    return final_prediction, predictions, {
        'pre_game_alerts': pre_game_alerts,
        'live_alerts': live_alerts,
        'team_summaries': {'home': home_summary, 'away': away_summary},
        'raw_prediction': final_total_pred,
        'blowout_adjusted': final_prediction,
        'context_adjusted': final_prediction,
        'final_prediction': final_prediction,
        # 'meta_offset': meta_offset_val,  # META-MODELO ELIMINADO
        'context_info': context_info,
        'score_diff': score_diff,
        'confidence_score': confidence_multiplier if 'confidence_multiplier' in locals() else 1.0,
        'missing_features': missing_features if 'missing_features' in locals() else [],
        'critical_missing_count': critical_missing if 'critical_missing' in locals() else 0,
        'high_edge_alert': high_edge_alert,
        'unrealistic_prediction_alert': unrealistic_prediction_alert,
        'debug_info': debug_logs if 'debug_logs' in locals() else {}
    }


# ===========================================
# METAMODELO AI INTEGRATION
# ===========================================

try:
    from analysis.metamodel_integration import aplicar_metamodelo_ai
    METAMODEL_AI_AVAILABLE = True
except ImportError as e:
    print(f'[WARNING] MetaModeloAI no disponible: {e}')
    METAMODEL_AI_AVAILABLE = False

def apply_metamodel_ai_adjustment(prediccion_base: float, q_scores: dict, 
                                  quarter_stage: str, live_signals: dict = None,
                                  home_team: str = '', away_team: str = '') -> tuple:
    """
    Aplica ajuste del MetaModeloAI a la prediccion base.
    Returns: (prediccion_ajustada, metadata_ai)
    """
    if not METAMODEL_AI_AVAILABLE:
        return prediccion_base, {'error': 'AI not available'}
    
    try:
        return aplicar_metamodelo_ai(
            base_prediction=prediccion_base,
            q_scores=q_scores,
            balance_features={},
            live_pace_metrics={},
            time_info={'quarter_stage': quarter_stage},
            live_signals=live_signals,
            team_trends={},
            bookie_line=None,
            team_stats={}
        )
    except Exception as e:
        print(f'[ERROR] Fallo en MetaModeloAI: {e}')
        return prediccion_base, {'error': str(e)}

def _calculate_safe_projection(current_total: float, minutes_played: float, quarter_stage: str, current_pace: float, halftime_total: float = None, total_game_minutes: float = 48.0) -> float:
    """
    Calcula una proyección segura basada en ritmo, estabilizada con el ritmo del primer tiempo.
    """
    if minutes_played <= 0:
        return 0.0
    
    # 1. Calcular proyección lineal basada en el ritmo actual global
    if current_pace > 0:
        linear_projection = current_pace * total_game_minutes
    else:
        linear_projection = (current_total / minutes_played) * total_game_minutes
        
    # 2. Si estamos en Q3/Q4, estabilizar con el ritmo del halftime (si existe)
    # Esto evita que una racha fría/caliente en Q3 distorsione toda la proyección
    halftime_minutes = total_game_minutes / 2.0
    if halftime_total and minutes_played > halftime_minutes:
        halftime_pace = (halftime_total / halftime_minutes) * total_game_minutes
        
        # Peso dinámico: A medida que avanza el juego, confiamos más en el ritmo actual
        # Min 24: 50% HT, 50% Current (aunque current == HT)
        # Min 36: 30% HT, 70% Current
        # Min 48: 0% HT, 100% Current
        
        minutes_past_ht = minutes_played - halftime_minutes
        weight_current = 0.5 + (0.5 * (minutes_past_ht / halftime_minutes))
        weight_current = min(1.0, max(0.5, weight_current))
        
        stabilized_projection = (linear_projection * weight_current) + (halftime_pace * (1 - weight_current))
        
        # print(f"   [DEBUG] Safe Projection Stabilized: Linear={linear_projection:.1f}, HT_Pace={halftime_pace:.1f}, W_Curr={weight_current:.2f} -> {stabilized_projection:.1f}")
        return stabilized_projection

    return linear_projection
