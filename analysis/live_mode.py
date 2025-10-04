# ===========================================
# Archivo: analysis/live_mode.py (v3.0)
# ===========================================
import numpy as np
import pandas as pd
import scipy.stats as stats
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from typing import Dict, List, Tuple, Any
import traceback
import os
import joblib
import json

# Imports del sistema
from config import (
    ADVANCED_STATS_COLS, MOMENTUM_STATS_COLS, PERFORMANCE_CONTEXT_COLS,
    UI_MESSAGES, CONTEXT_ADJUSTMENT, META_MODEL, MODELS_FOLDER
)

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
        calculate_live_pace_metrics, calculate_momentum_metrics,
        calculate_real_balance_features, calculate_garbage_time_risk,
        calculate_team_quarter_trends,
        apply_blowout_adjustment, safe_mean, safe_divide
    )
    FEATURES_IMPORTS_OK = True
except ImportError as e:
    print(f"⚠️ Error importando core.features: {e}")
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

# Imports de análisis con fallback integrado
try:
    from analysis.alerts import create_alerts_system
    ALERTS_AVAILABLE = True
except ImportError:
    print("⚠️ Sistema de alertas no disponible")
    ALERTS_AVAILABLE = False

# ===================================================================
# SISTEMA DE AJUSTES POR CONTEXTO (MANTENIDO)
# ===================================================================

def _get_league_scoring_params(league_name: str = None) -> Dict[str, float]:
    """Get league-specific scoring parameters for safeguards"""
    if not league_name:
        # Default conservative parameters
        return {
            'max_ppg_threshold': 35.0,  # PPG threshold for unrealistic scoring
            'max_ppg_cap': 30.0,        # Max realistic PPG
            'q4_conservative_reduction': 0.96,  # -4% for Q4 close games
            'pace_multiplier': 1.0      # Base pace multiplier
        }

    league_lower = league_name.lower()

    if 'nba' in league_lower:
        # NBA: Higher scoring potential, faster pace
        return {
            'max_ppg_threshold': 45.0,  # NBA can score 45+ PPG realistically
            'max_ppg_cap': 40.0,        # Cap at 40 PPG max
            'q4_conservative_reduction': 0.96,  # Same -4% reduction
            'pace_multiplier': 1.2      # NBA has faster pace
        }
    elif 'wnba' in league_lower:
        # WNBA: Lower scoring, slower pace
        return {
            'max_ppg_threshold': 35.0,  # WNBA threshold
            'max_ppg_cap': 30.0,        # WNBA cap
            'q4_conservative_reduction': 0.96,  # Same reduction
            'pace_multiplier': 0.8      # WNBA slower pace
        }
    elif 'nbl' in league_lower:
        # NBL: Australian league - moderate scoring, consistent pace
        return {
            'max_ppg_threshold': 41.0,  # NBL realistic threshold (based on 241 max)
            'max_ppg_cap': 37.0,        # NBL conservative cap
            'q4_conservative_reduction': 0.94,  # More conservative than NBA/WNBA
            'pace_multiplier': 0.98     # Slightly slower than NBA baseline
        }
    else:
        # Other leagues: Conservative defaults
        return {
            'max_ppg_threshold': 35.0,
            'max_ppg_cap': 30.0,
            'q4_conservative_reduction': 0.96,
            'pace_multiplier': 1.0
        }

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
            print(f"⚠️ WNBA context adjustments not found, using default")
            from config import CONTEXT_ADJUSTMENT
            return CONTEXT_ADJUSTMENT
    elif 'nbl' in league_lower:
        # NBL-specific context adjustments
        try:
            from config import CONTEXT_ADJUSTMENT_NBL
            return CONTEXT_ADJUSTMENT_NBL
        except ImportError:
            print(f"⚠️ NBL context adjustments not found, using default")
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

def apply_context_adjustment(prediction: float, balance_features: Dict[str, float],
                             live_pace_metrics: Dict[str, float], quarter_stage: str,
                             live_signals: Dict[str, float] = None,
                             league_name: str = None) -> Tuple[float, Dict[str, Any]]:
    """
    Sistema de ajustes por contexto - SIN funcionalidad de betting
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

        # Extraer pace context con valores por defecto seguros
        pace_estimate = live_pace_metrics.get('live_pace_estimate', 100)
        enhanced_pace = live_pace_metrics.get('enhanced_pace_estimate', pace_estimate) * league_params['pace_multiplier']

        # Determinar contextos base (partido cerrado = diferencia <= 10)
        is_close_game = (current_lead is not None and current_lead <= 10)
        # Umbral de "pace alto" ajustado a la escala actual de live pace
        pace_high_threshold = _get_live_pace_high_threshold(league_name)
        is_high_pace = enhanced_pace > pace_high_threshold
        is_late_game = quarter_stage in ['q3_end', 'q4']

        # 🆕 FASE 2: Extraer señales live enfocadas en O/U con valores por defecto seguros
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

        # 🆕 FASE 2: Señales O/U específicas usando stats completos de SofaScore
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

        # CONTEXTO 1: Garbage Time dinámico por riesgo (MEJORADO)
        gt_enabled = context_adjustment.get('enable_dynamic_garbage_time', True)
        garbage_time_risk = balance_features.get('garbage_time_risk', None)
        if gt_enabled and garbage_time_risk is not None and quarter_stage in ['q3_end', 'q4'] and garbage_time_risk >= 0.6:
            # Reducción suave acotada y atenuada por foul-trouble (más FT => menos reducción efectiva)
            max_red_pct = context_adjustment.get('garbage_time_max_reduction_pct', 0.12)
            foul_trouble_boost = 1 + (max(home_fti, away_fti) * 0.1)  # +10% máx
            raw_reduction = min(max_red_pct, garbage_time_risk * max_red_pct)
            adjusted_reduction = (1 - raw_reduction) * foul_trouble_boost
            adjustment_factor *= adjusted_reduction
            applied_adjustments.append(f"Garbage time dinámico: -{((1-adjusted_reduction)*100):.0f}% (risk: {garbage_time_risk:.2f}, FT boost: {foul_trouble_boost:.2f})")

        # CONTEXTO 2: Partidos cerrados + pace alto (REFINADO - MÁS CONSERVADOR EN Q4)
        elif is_close_game and is_high_pace and is_late_game:
            # Lógica más conservadora para Q4: reducir en lugar de aumentar
            if quarter_stage == 'q4':
                # En Q4 de partidos cerrados, equipos suelen frenar el ritmo
                adjustment_factor *= league_params['q4_conservative_reduction']
                reduction_pct = ((1-league_params['q4_conservative_reduction'])*100)
                applied_adjustments.append(f"Q4 partido cerrado: -{reduction_pct:.0f}% (pace alto pero Q4 defensivo - {league_name or 'Unknown'})")
            else:
                # Para Q3, mantener lógica original pero más conservadora
                shooting_boost = 1 + (max(home_ts_live, away_ts_live) - 0.5) * 0.1  # +10% max (reducido)
                adjustment_factor *= 1.02 * shooting_boost  # +2% base (reducido)
                applied_adjustments.append(f"Partido cerrado + pace alto: +{((1.02*shooting_boost-1)*100):.0f}% (pace: {enhanced_pace:.1f}, TS boost: {shooting_boost:.2f})")

        # CONTEXTO 3: Intensidad en declive (REFINADO)
        elif intensity_drop < 0.85:
            # Escalar por run detector (run activa = más puntos)
            run_boost = 1 + (run_strength * 0.15)  # +15% max por run strength
            adjusted_reduction = 0.96 * run_boost
            adjustment_factor *= adjusted_reduction
            applied_adjustments.append(f"Caída de intensidad: -{((1-adjusted_reduction)*100):.0f}% (factor: {intensity_drop:.3f}, run boost: {run_boost:.2f})")

        # CONTEXTO 4: Fouling strategy (REFINADO)
        elif is_close_game and quarter_stage == 'q4' and enhanced_pace > 110:
            # Escalar por foul trouble directo (más FT = más puntos)
            ft_boost = 1 + (max(home_fti, away_fti) * 0.25)  # +25% max por foul trouble
            adjustment_factor *= 1.08 * ft_boost
            applied_adjustments.append(f"Estrategia fouling Q4: +{((1.08*ft_boost-1)*100):.0f}% (pace: {enhanced_pace:.1f}, FT boost: {ft_boost:.2f})")

        # CONTEXTO 5: Run detector activo (NUEVO)
        elif run_active and run_strength > 0.3:
            run_boost = 1 + (run_strength * 0.2)  # +20% max por run strength
            adjustment_factor *= run_boost
            applied_adjustments.append(f"Run detectado ({run_side}): +{((run_boost-1)*100):.0f}% (strength: {run_strength:.2f})")

        # CONTEXTO 6: Foul trouble significativo (NUEVO)
        elif max(home_fti, away_fti) > 0.7:
            ft_boost = 1 + ((max(home_fti, away_fti) - 0.7) * 0.3)  # +30% max por foul trouble extremo
            adjustment_factor *= ft_boost
            applied_adjustments.append(f"Foul trouble alto: +{((ft_boost-1)*100):.0f}% (FTI: {max(home_fti, away_fti):.2f})")

        # CONTEXTO 7: Shooting efficiency extrema (NUEVO)
        elif abs(diff_ts_live) > 0.15:
            shooting_boost = 1 + (abs(diff_ts_live) * 0.15)  # +15% max por diferencia TS%
            adjustment_factor *= shooting_boost
            applied_adjustments.append(f"Shooting efficiency diferencial: +{((shooting_boost-1)*100):.0f}% (TS diff: {diff_ts_live:.2f})")

        # 🆕 FASE 2: CONTEXTO 8: Assist-to-Turnover Ratio extrema (O/U IMPACT)
        elif abs(diff_ato_ratio) > 0.25:
            ato_boost = 1 + (abs(diff_ato_ratio) * 0.15)  # +15% max por diferencia A/TO
            adjustment_factor *= ato_boost
            direction = "superior" if diff_ato_ratio > 0 else "inferior"
            applied_adjustments.append(f"Control de balón {direction}: +{((ato_boost-1)*100):.0f}% (A/TO diff: {diff_ato_ratio:.2f})")

        # 🆕 FASE 2: CONTEXTO 9: Rebound efficiency diferencial (O/U IMPACT)
        elif abs(home_treb_diff) > 0.4:
            reb_boost = 1 + (abs(home_treb_diff) * 0.12)  # +12% max por ventaja en rebotes
            adjustment_factor *= reb_boost
            direction = "ventaja" if home_treb_diff > 0 else "desventaja"
            applied_adjustments.append(f"Rebotes {direction}: +{((reb_boost-1)*100):.0f}% (TREB diff: {home_treb_diff:.2f})")

        # CONTEXTO 10: Partido equilibrado (ORIGINAL)
        elif is_close_game and 95 <= enhanced_pace <= 105:
            applied_adjustments.append("Partido equilibrado: Sin ajustes")

    except Exception as e:
        print(f"⚠️ Error en ajuste por contexto: {e}")
        import traceback
        traceback.print_exc()  # Add stack trace for debugging
        applied_adjustments.append("Error en análisis - usando predicción base")
        adjustment_factor = 1.0
    
    # Calcular adjusted_prediction con factor de contexto (clamp se aplicará DESPUÉS del safeguard)
    adjusted_prediction = prediction * adjustment_factor
    # Flags/telemetría para clamp y safeguard
    safeguard_triggered = False
    pre_safeguard_prediction = adjusted_prediction
    try:
        # 🆕 SAFEGUARD: Verificar ritmo de puntuación realista
        # Calcular puntos restantes y tiempo restante
        # Necesitamos calcular el total actual desde balance_features o usar valores por defecto
        try:
            # Intentar obtener el total actual de balance_features si está disponible
            current_total = balance_features.get('current_total', 0)
            if current_total == 0:
                # Fallback: estimar basado en quarter_stage
                if quarter_stage in ['q3_end', 'q4']:
                    current_total = 100  # Valor conservador por defecto
                elif quarter_stage == 'q3_progress':
                    current_total = 80   # Valor conservador para Q3 en progreso
                else:
                    current_total = 60   # Valor conservador para halftime
        except:
            current_total = 0

        if current_total > 0 and prediction > current_total:
            points_needed = prediction - current_total
            # Estimar tiempo restante en minutos (AJUSTADO POR LIGA: NBA=12, WNBA/NBL/EURO=10 min/cuarto)
            qlen = 12
            try:
                if league_name:
                    low = league_name.lower()
                    if ('wnba' in low) or ('nbl' in low) or ('euro' in low):
                        qlen = 10
            except Exception:
                pass

            if quarter_stage == 'q4':
                time_remaining = qlen
            elif quarter_stage == 'q3_end':
                time_remaining = qlen
            elif quarter_stage == 'q3_progress':
                time_remaining = int(qlen * 1.5)
            elif quarter_stage == 'halftime':
                time_remaining = qlen * 2
            else:
                time_remaining = qlen * 2

            if time_remaining > 0:
                # Normalizar a "puntos por cuarto" usando la duración real del cuarto
                required_ppg = points_needed / (time_remaining / qlen)

            # Usar league-specific scoring thresholds
                if required_ppg > league_params['max_ppg_threshold']:
                    max_realistic = current_total + (time_remaining / 12 * league_params['max_ppg_cap'])
                    if adjusted_prediction > max_realistic:
                        adjusted_prediction = max_realistic
                        safeguard_triggered = True
                        applied_adjustments.append(f"Safeguard ritmo aplicado: reducido a {max_realistic:.0f} pts (ritmo requerido: {required_ppg:.1f} PPG, liga: {league_name or 'Unknown'})")

    except Exception:
        # Si algo falla, mantener valor calculado
        pass
    
    # Aplicar límite conservador final (±2 en Q4 close, ±4 otros) DESPUÉS de safeguards
    try:
        pre_clamp_prediction = adjusted_prediction
        if quarter_stage == 'q4' and current_lead is not None and current_lead <= 10:
            max_delta = 2.0
        else:
            max_delta = 4.0

        upper = prediction + max_delta
        lower = prediction - max_delta

        if adjusted_prediction > upper:
            adjusted_prediction = upper
            applied_adjustments.append(f"Clamp final aplicado: superior (de {pre_clamp_prediction:.0f} a {adjusted_prediction:.0f}, bound=+{max_delta:.0f})")
        elif adjusted_prediction < lower:
            if 'safeguard_triggered' in locals() and safeguard_triggered:
                # No elevar por encima del valor limitado por safeguard
                applied_adjustments.append(f"Clamp omitido por safeguard: límite inferior {lower:.0f} ignorado (valor={adjusted_prediction:.0f})")
            else:
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
            # 🆕 FASE 2: Señales O/U específicas
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
            'away_treb_diff': away_treb_diff
        }
    }
    
    return adjusted_prediction, context_info
def build_pre_game_cache(home_team_name: str, away_team_name: str, trained_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    🚀 PRE-GAME CACHE BUILDER - Computa features estáticas una sola vez por partido
    ✅ OPTIMIZACIÓN: Reduce CPU en live updates al ~70%
    ✅ INCLUYE: H2H features, team trends, alerts system
    """
    print("🔧 Construyendo caché pre-partido...")

    historical_df = trained_data['historical_df']
    features_used_in_model = trained_data['features_used']

    # Crear alerts system una sola vez (reutilizable)
    alerts_system = None
    if ALERTS_AVAILABLE:
        alerts_system = create_alerts_system(historical_df)
        pre_game_alerts = alerts_system.generate_pre_game_alerts(home_team_name, away_team_name)
    else:
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
            print(f"⚠️ Error calculando team trends: {e}")

    # Calcular H2H features (ESTÁTICOS)
    h2h_stats = {}
    try:
        from core.features import calculate_h2h_features
        h2h_stats = calculate_h2h_features(home_team_name, away_team_name, historical_df)
    except Exception as e:
        print(f"⚠️ Error calculando H2H features: {e}")
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

    print(f"✅ Caché pre-partido construido: {len(features_static)} features estáticas")
    return pre_game_cache

def get_predictions_with_alerts(home_team_name: str, away_team_name: str,
                                q_scores: Dict[str, int], trained_data: Dict[str, Any],
                                pre_game_cache: Dict[str, Any] = None,
                                live_signals: Dict[str, float] = None,
                                silent_mode: bool = False) -> Tuple[float, List[Dict], Dict[str, Any]]:
    """
    Calcula predicciones Over/Under CON SISTEMA DE ALERTAS - SIN BETTING PROTECTION
    🚀 OPTIMIZADO: Soporta caché pre-partido para ~70% menos CPU en live updates
    """
    model = trained_data['model']
    std_dev = trained_data['std_dev']
    historical_df = trained_data['historical_df']
    features_used_in_model = trained_data['features_used']

    # 🚀 USAR CACHÉ SI ESTÁ DISPONIBLE
    if pre_game_cache is not None:
        print("📦 Usando caché pre-partido - computación optimizada")
        alerts_system = pre_game_cache.get('alerts_system')
        pre_game_alerts = pre_game_cache.get('pre_game_alerts', [])
        home_history_df = pre_game_cache.get('home_history_df')
        away_history_df = pre_game_cache.get('away_history_df')
        team_trends = pre_game_cache.get('team_trends', {'home': {}, 'away': {}})
        h2h_stats = pre_game_cache.get('h2h_stats', {})

        # Usar features estáticas del caché
        features = pre_game_cache['features_static'].copy()

        # Reutilizar alerts system si existe
        if alerts_system is None and ALERTS_AVAILABLE:
            alerts_system = create_alerts_system(historical_df)
            pre_game_alerts = alerts_system.generate_pre_game_alerts(home_team_name, away_team_name)
    else:
        # 🔄 COMPUTACIÓN TRADICIONAL (sin caché)
        print("Computacion tradicional - sin cache")
        if ALERTS_AVAILABLE:
            alerts_system = create_alerts_system(historical_df)
            pre_game_alerts = alerts_system.generate_pre_game_alerts(home_team_name, away_team_name)
        else:
            pre_game_alerts = []
            alerts_system = None
    
    # 🔄 COMPUTACIÓN TRADICIONAL (solo si no hay caché)
    if pre_game_cache is None:
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

        features = {}
        features.update({'home_' + k: v for k, v in home_stats_5.items()})
        features.update({'away_' + k: v for k, v in away_stats_5.items()})
        features.update({'home_' + k: v for k, v in home_ema_5.items()})
        features.update({'away_' + k: v for k, v in away_ema_5.items()})
        features.update({'home_' + k: v for k, v in home_ema_advanced.items()})
        features.update({'away_' + k: v for k, v in away_ema_advanced.items()})

        for stat in cols_to_avg:
            features[f'diff_avg_{stat}_last_5'] = features.get(f'home_avg_{stat}_last_5', np.nan) - features.get(f'away_avg_{stat}_last_5', np.nan)
        for stat in cols_to_ema:
            features[f'diff_ema_{stat}_last_5'] = features.get(f'home_ema_{stat}_last_5', np.nan) - features.get(f'away_ema_{stat}_last_5', np.nan)

        # Calcular team trends para pace projection
        team_trends = {'home': {}, 'away': {}}
        if FEATURES_IMPORTS_OK:
            try:
                home_trends = calculate_team_quarter_trends(home_history_df)
                away_trends = calculate_team_quarter_trends(away_history_df)
                team_trends = {'home': home_trends, 'away': away_trends}
            except Exception as e:
                print(f"⚠️ Error calculando team trends: {e}")

        # Calcular H2H features
        h2h_stats = {}
        try:
            from core.features import calculate_h2h_features
            h2h_stats = calculate_h2h_features(home_team_name, away_team_name, historical_df)
        except Exception as e:
            print(f"⚠️ Error calculando H2H features: {e}")
            h2h_stats = {k: np.nan for k in ['h2h_avg_total_score', 'h2h_pace_avg', 'h2h_first_half_avg_total', 'h2h_second_half_surge_avg', 'h2h_comeback_freq']}

        # Agregar H2H al features
        features.update(h2h_stats)

    features['q1_total'] = q_scores['q1_home'] + q_scores['q1_away']
    features['q2_total'] = q_scores['q2_home'] + q_scores['q2_away']
    features['q3_total'] = q_scores['q3_home'] + q_scores['q3_away']
    features['halftime_total'] = features['q1_total'] + features['q2_total']
    features['q3_end_total'] = features['halftime_total'] + features['q3_total']
    features['q1_diff'] = q_scores['q1_home'] - q_scores['q1_away']
    features['q2_diff'] = q_scores['q2_home'] - q_scores['q2_away']
    features['q3_diff'] = q_scores['q3_home'] - q_scores['q3_away']
    features['q2_trend'] = features['q2_total'] - features['q1_total']
    features['q3_trend'] = features['q3_total'] - features['q2_total']
    features['quarter_variance'] = np.std([features['q1_total'], features['q2_total'], features['q3_total']])

    # Calcular métricas avanzadas si están disponibles
    # Detectar progreso real de Q3 para no asumir fin de cuarto prematuramente
    q4_total_now = q_scores.get('q4_home', 0) + q_scores.get('q4_away', 0)
    if features['q3_total'] > 0 and q4_total_now == 0:
        quarter_stage = 'q3_progress'
    elif features['q3_total'] > 0 and q4_total_now > 0:
        quarter_stage = 'q3_end'
    else:
        quarter_stage = 'halftime'

    if FEATURES_IMPORTS_OK:
        try:
            # Usar team_trends del caché si está disponible, sino calcular
            if pre_game_cache is None:
                home_trends = calculate_team_quarter_trends(home_history_df)
                away_trends = calculate_team_quarter_trends(away_history_df)
                team_trends = {'home': home_trends, 'away': away_trends}
            # team_trends ya está definido arriba desde el caché

            live_pace_metrics = calculate_live_pace_metrics(q_scores, quarter_stage, team_trends)
            balance_features = calculate_real_balance_features(q_scores, quarter_stage, (home_team_name, away_team_name))
            # Calcular garbage_time_risk en vivo y añadir a balance_features
            try:
                gtr = calculate_garbage_time_risk(
                    q_scores, quarter_stage,
                    lead_stability=balance_features.get('lead_stability', 0.5),
                    quarter_variance=features.get('quarter_variance', np.nan),
                    is_potential_blowout=balance_features.get('is_potential_blowout', 0)
                )
                balance_features['garbage_time_risk'] = gtr
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ Error en métricas avanzadas: {e}")
            live_pace_metrics = {'live_pace_estimate': 100, 'live_efficiency_home': 1.0, 'live_efficiency_away': 1.0}
            balance_features = {'is_game_unbalanced': 0, 'game_balance_score': 0.5}
    else:
        live_pace_metrics = {'live_pace_estimate': 100, 'live_efficiency_home': 1.0, 'live_efficiency_away': 1.0}
        balance_features = {'is_game_unbalanced': 0, 'game_balance_score': 0.5}

    features.update(live_pace_metrics)
    features.update(balance_features)

    # Establecer current_total real para salvaguarda de ritmo
    try:
        if quarter_stage == 'halftime':
            balance_features['current_total'] = features.get('halftime_total', 0)
        elif quarter_stage == 'q3_progress':
            balance_features['current_total'] = features.get('halftime_total', 0) + features.get('q3_total', 0)
        elif quarter_stage == 'q3_end':
            balance_features['current_total'] = features.get('q3_end_total', 0)
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

    # 🧩 DEBUG: Telemetría crítica Q2+ para diagnóstico de dips en Q3 en progreso
    try:
        q1_t = features['q1_total']; q2_t = features['q2_total']; q3_t = features['q3_total']
        halftime_t = features['halftime_total']; q3_end_t = features['q3_end_total']
        q4_home = q_scores.get('q4_home', 0); q4_away = q_scores.get('q4_away', 0)
        q4_t = q4_home + q4_away
        q3_in_progress = (q3_t > 0) and (q4_t == 0)
        # Calcular minutos jugados asumidos por liga (NBA=12, WNBA/NBL/EURO=10 min/cuarto)
        try:
            ln_canon = _canonicalize_league_name(trained_data.get('league_name', 'NBA'))
        except Exception:
            ln_canon = 'NBA'
        qlen = 10 if any(k in ln_canon.upper() for k in ['WNBA','NBL','EURO']) else 12
        if quarter_stage == 'halftime':
            minutes_assumed = qlen * 2
        elif quarter_stage == 'q3_progress':
            minutes_assumed = int(qlen * 2.5)
        elif quarter_stage == 'q3_end':
            minutes_assumed = qlen * 3
        else:
            minutes_assumed = qlen * 2
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
        if q2_t > 0:
            # print("\nDEBUG LIVE (Q2+):")
            print(f"   • Stage: {quarter_stage} | min asumidos: {minutes_assumed}")
            print(f"   • Totales: Q1={q1_t}, Q2={q2_t}, Q3={q3_t} | HT={halftime_t} | VarCuartos={features.get('quarter_variance', np.nan):.3f}")
            lp = live_pace_metrics.get('live_pace_estimate', 'n/a')
            ep = live_pace_metrics.get('enhanced_pace_estimate', live_pace_metrics.get('live_pace_estimate', 'n/a'))
            print(f"   • Pace: live={lp} | enhanced={ep}")
            print(f"   • Balance: lead={balance_features.get('current_lead', 'n/a')}, unbalanced={balance_features.get('is_game_unbalanced', 'n/a')}, gtrisk={balance_features.get('garbage_time_risk', 'n/a')}")
            print(f"   • Qtrs para alertas: {quarters_completed_for_alerts} | σ={std_dev:.2f}")
            if quarter_stage == 'q3_progress':
                print(f"   🟡 Q3 en progreso — asumiendo {minutes_assumed}m para pace/ajustes (sin activar lógica de fin de Q3)")
    except Exception as _dbg_err:
        # print(f"Error en debug live: {_dbg_err}")

        if ALERTS_AVAILABLE:
            live_alerts = alerts_system.analyze_live_performance(
                home_team_name, away_team_name, q_scores, quarter_stage, balance_features
            )
            home_summary = alerts_system.get_team_summary(home_team_name)
            away_summary = alerts_system.get_team_summary(away_team_name)

    # 🆕 FASE 2: Telemetría de señales live enfocadas en O/U
    if live_signals:
        print("\n[SEÑALES LIVE O/U (FASE 2)]:")
        print(f"   [FTI] - {home_team_name}: {live_signals.get('home_fti', 0.5):.3f}")
        print(f"   [FTI] - {away_team_name}: {live_signals.get('away_fti', 0.5):.3f}")
        print(f"   [TS%] - {home_team_name}: {live_signals.get('home_ts_live', 0.5):.3f}")
        print(f"   [TS%] - {away_team_name}: {live_signals.get('away_ts_live', 0.5):.3f}")
        print(f"   [Run detector]: {'ACTIVO' if live_signals.get('run_active', False) else 'Inactivo'}")
        if live_signals.get('run_active', False):
            print(f"      Lado: {live_signals.get('run_side', 'none')} | Fuerza: {live_signals.get('run_strength', 0.0):.2f}")

        # 🆕 FASE 2: Mostrar señales O/U específicas
        print(f"   [A/TO] - {home_team_name}: {live_signals.get('home_ato_ratio', 1.0):.2f}")
        print(f"   [A/TO] - {away_team_name}: {live_signals.get('away_ato_ratio', 1.0):.2f}")
        print(f"   [TREB diff] - {home_team_name}: {live_signals.get('home_treb_diff', 0.0):.2f}")
        print(f"   [OREB%] - {home_team_name}: {live_signals.get('home_oreb_pct', 0.25):.1%}")
        print(f"   [OREB%] - {away_team_name}: {live_signals.get('away_oreb_pct', 0.25):.1%}")

    if not silent_mode:
        if balance_features.get('is_game_unbalanced', 0) == 1:
            print(f"\n🚨 ALERTA DE DESBALANCE:")
            print(f"   Juego muy desigual - posible impacto en total final")
            print(f"   Balance Score: {balance_features['game_balance_score']:.3f}")

        if balance_features.get('intensity_drop_factor', 1.0) < 0.8:
            print(f"\n📉 ALERTA DE INTENSIDAD:")
            print(f"   Caída de intensidad detectada")
            print(f"   Factor: {balance_features['intensity_drop_factor']:.3f}")

    # Preparar datos para el modelo
    X_pred = pd.DataFrame([features], columns=features_used_in_model)
    
    impute_values = trained_data.get('impute_values', {})
    
    print(f"\n[Verificando {len(X_pred.columns)} caracteristicas...]")
    missing_count = 0

    for col in X_pred.columns:
        if X_pred[col].isnull().any():
            missing_count += 1
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

    if missing_count > 0:
        print(f"[Total de caracteristicas imputadas: {missing_count}/{len(X_pred.columns)}]")
    else:
        print("[Todas las caracteristicas disponibles - sin imputacion necesaria]")

    if X_pred.isnull().values.any():
        print("\n⚠️ Advertencia: Aún quedan valores NaN después de imputar – rellenando con 0s.")
        X_pred = X_pred.fillna(0.0)

    # Predicción del modelo
    final_total_pred = model.predict(X_pred)[0]
    
    # Aplicar ajuste por paliza (original)
    home_score = q_scores['q1_home'] + q_scores['q2_home'] + q_scores['q3_home']
    away_score = q_scores['q1_away'] + q_scores['q2_away'] + q_scores['q3_away']
    score_diff = abs(home_score - away_score)

    quarters_played = 3 if q_scores['q3_home'] > 0 or q_scores['q3_away'] > 0 else 2
    time_remaining_pct = (4 - quarters_played) / 4

    try:
        if FEATURES_IMPORTS_OK:
            blowout_adjusted_prediction = apply_blowout_adjustment(final_total_pred, score_diff, time_remaining_pct)
        else:
            blowout_adjusted_prediction = final_total_pred
    except:
        blowout_adjusted_prediction = final_total_pred

    # Aplicar ajuste por contexto con señales live avanzadas
    # Canonicalizar liga para asegurar parámetros correctos (NBA/WNBA/NBL)
    league_name_raw = trained_data.get('league_name', None)
    league_name = _canonicalize_league_name(league_name_raw or 'NBA')
    try:
        # Reflejar la liga canónica en el payload (fuente de verdad para pasos siguientes)
        trained_data['league_name'] = league_name
    except Exception:
        pass

    context_adjusted_prediction, context_info = apply_context_adjustment(
        blowout_adjusted_prediction, balance_features, live_pace_metrics, quarter_stage,
        live_signals=live_signals, league_name=league_name
    )

    # Meta-model correction (stacking) - enabled by config, applied only if per-league artifact exists
    final_prediction = context_adjusted_prediction
    meta_offset_val = 0.0
    try:
        if META_MODEL.get('enable_meta_correction', False):
            league_name = trained_data.get('league_name', 'LEAGUE')
            # Resolver candidatos robustos: 1) nombre canónico 2) nombre original
            canonical = _canonicalize_league_name(league_name)
            safe_canonical = canonical.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            safe_original = league_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            candidates = []
            for nm in [safe_canonical, safe_original]:
                if nm not in candidates:
                    candidates.append(nm)

            # 1) Intentar bundle JSON único <LEAGUE>.json con sección 'meta'
            meta_payload = None
            bundle_path = None
            for nm in candidates:
                p = os.path.join(MODELS_FOLDER, f"{nm}.json")
                if os.path.exists(p):
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        # Bundle expected: {'artifact_type':'bundle', 'base':{...}, 'meta':{...}}
                        if isinstance(data, dict) and ('meta' in data) and isinstance(data['meta'], dict):
                            meta_payload = data['meta']
                            bundle_path = p
                            break
                    except Exception:
                        continue

            # 2) Fallback a meta JSON legacy <LEAGUE>_meta.json
            if meta_payload is None:
                for nm in candidates:
                    p = os.path.join(MODELS_FOLDER, f"{nm}_meta.json")
                    if os.path.exists(p):
                        try:
                            with open(p, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            if isinstance(data, dict) and data.get('artifact_type') == 'meta_model':
                                meta_payload = data
                                bundle_path = p
                                break
                        except Exception:
                            continue

            # Meta model disabled or not available

            # Solo Ridge regression - calcular manualmente con coeficientes
            if meta_payload is not None and isinstance(meta_payload, dict):
                try:
                    model_type = meta_payload.get('model_type', '').lower()
                    if model_type in ['ridge', 'ridge_linear'] or ('coef' in meta_payload and 'intercept' in meta_payload):
                        feature_names = meta_payload.get('feature_names', [])
                        impute_values = meta_payload.get('impute_values', {})
                        coef = meta_payload.get('coef', [])
                        intercept = float(meta_payload.get('intercept', 0.0))
                        clip_val = float(meta_payload.get('clip', META_MODEL.get('meta_offset_clip', 8.0)))

                        # Construir vector ordenado con señales live enhanced
                        row_vals = []
                        for f in feature_names:
                            if f == 'base_pred':
                                row_vals.append(float(final_total_pred))
                            elif f == 'context_adjusted_pred':
                                # Usar la predicción ajustada por señales como feature
                                row_vals.append(float(context_adjusted_prediction))
                            elif f == 'signal_strength':
                                # Fuerza total de señales activas
                                signal_strength = calculate_signal_strength(live_signals or {})
                                row_vals.append(float(signal_strength))
                            elif f == 'active_signals':
                                # Número de señales activas
                                active_signals = count_active_signals(live_signals or {})
                                row_vals.append(float(active_signals))
                            elif f == 'adjustment_magnitude':
                                # Magnitud del ajuste aplicado por señales
                                adjustment_magnitude = calculate_signal_adjustment_magnitude(context_info)
                                row_vals.append(float(adjustment_magnitude))
                            else:
                                v = features.get(f, np.nan)
                                if v is None or (isinstance(v, float) and np.isnan(v)):
                                    v = impute_values.get(f, 0.0)
                                row_vals.append(float(v))
                        # Dot product + intercept
                        meta_offset = float(np.dot(np.array(row_vals, dtype=float), np.array(coef, dtype=float)) + intercept)
                        if abs(meta_offset) > clip_val:
                            meta_offset = clip_val if meta_offset > 0 else -clip_val
                        meta_offset_val = meta_offset
                        final_prediction = context_adjusted_prediction + meta_offset
                        print(f"Meta-model adjustment: {meta_offset:+.1f} pts (clip ±{clip_val:.1f})")
                    else:
                        print(f"⚠️ Meta-modelo no es Ridge: {model_type} - ignorando")
                except Exception as e:
                    print(f"❌ Error aplicando meta-modelo Ridge: {e}")
            # else: silently skip if no artifact found
    except Exception as _meta_err:
        print(f"⚠️ Meta-modelo deshabilitado o no disponible: {_meta_err}")
        final_prediction = context_adjusted_prediction

    # Mostrar información de ajustes profesional
    print(f"\n[Evolucion de la Prediccion:]")
    print(f"   [Prediccion base del modelo]: {final_total_pred:.0f} pts")

    if abs(blowout_adjusted_prediction - final_total_pred) > 0.1:
        blowout_reduction = final_total_pred - blowout_adjusted_prediction
        if score_diff > 15:
            print(f"   [Ajuste por paliza]: -{blowout_reduction:.0f} pts (diferencia {score_diff:.0f} pts)")
            print(f"       -> Equipos pueden relajarse en el final")

    if abs(context_adjusted_prediction - blowout_adjusted_prediction) > 0.1:
        context_change = context_adjusted_prediction - blowout_adjusted_prediction
        sign = "+" if context_change > 0 else ""
        print(f"   [Ajuste por situacion]: {sign}{context_change:.0f} pts")

        # Explicar el ajuste
        for adjustment in context_info['applied_adjustments']:
            if "garbage time" in adjustment.lower():
                print(f"       -> Garbage time detectado - equipos aflojando")
            elif "cerrado + pace alto" in adjustment.lower():
                print(f"       -> Partido cerrado y rapido - mas fouls esperados")
            elif "caída de intensidad" in adjustment.lower():
                print(f"       -> Intensidad bajando - menos puntos esperados")
            elif "fouling" in adjustment.lower():
                print(f"       -> Estrategia de fouls en Q4")
            elif "safeguard ritmo aplicado" in adjustment.lower():
                print(f"       -> Salvaguarda de ritmo: {adjustment}")
            elif "clamp final aplicado" in adjustment.lower():
                print(f"       -> {adjustment}")
            elif "clamp omitido por safeguard" in adjustment.lower():
                print(f"       -> {adjustment}")

    print(f"   [PREDICCION FINAL]: {final_prediction:.0f} pts")
    # Generar múltiples líneas con probabilidades
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
        
    return final_prediction, predictions, {
        'pre_game_alerts': pre_game_alerts,
        'live_alerts': live_alerts,
        'team_summaries': {'home': home_summary, 'away': away_summary},
        'raw_prediction': final_total_pred,
        'blowout_adjusted': blowout_adjusted_prediction,
        'context_adjusted': context_adjusted_prediction,
        'final_prediction': final_prediction,
        'meta_offset': meta_offset_val,
        'context_info': context_info,
        'score_diff': score_diff,
        'debug_info': debug_logs if 'debug_logs' in locals() else {}
    }

def live_mode_with_alerts(trained_data: Dict[str, Any]) -> None:
    """Función principal: Modo en vivo CON ALERTAS - SIN BETTING PROTECTION"""
    print(f"\n{UI_MESSAGES['live_mode_title']}")
    print("🎯 Sistema simplificado: Solo predicciones + alertas contextuales")
    
    team_completer = WordCompleter(trained_data['team_names'], ignore_case=True)
    home_team_name = prompt("Equipo Local: ", completer=team_completer)
    away_team_name = prompt("Equipo Visitante: ", completer=team_completer)
    
    if home_team_name not in trained_data['team_names'] or away_team_name not in trained_data['team_names']:
        print("\nError: Uno o ambos equipos no se encuentran en los datos de la liga seleccionada.")
        return

    try:
        print(f"\n-- Puntuación por Cuarto: {home_team_name} vs {away_team_name} --")
        q_scores = {
            'q1_home': int(prompt(f"Puntos de {home_team_name} en Q1: ")),
            'q1_away': int(prompt(f"Puntos de {away_team_name} en Q1: ")),
            'q2_home': int(prompt(f"Puntos de {home_team_name} en Q2: ")),
            'q2_away': int(prompt(f"Puntos de {away_team_name} en Q2: ")),
            'q3_home': 0, 'q3_away': 0
        }

        add_q3 = prompt("¿Desea agregar los datos del Q3? (si/no): ").lower().strip()
        if add_q3 == 'si':
            q_scores['q3_home'] = int(prompt(f"Puntos de {home_team_name} en Q3: "))
            q_scores['q3_away'] = int(prompt(f"Puntos de {away_team_name} en Q3: "))

        live_score_home = q_scores['q1_home'] + q_scores['q2_home'] + q_scores['q3_home']
        live_score_away = q_scores['q1_away'] + q_scores['q2_away'] + q_scores['q3_away']
        
        # ANÁLISIS SIMPLIFICADO - Sin betting protection
        adjusted_pred, predictions, alerts_data = get_predictions_with_alerts(
            home_team_name, away_team_name, q_scores, trained_data
        )
        
        print(f"\n📋 Información del Modelo:")
        print(f"   Liga: {trained_data['league_name']}")
        print(f"   Partidos en historial: {len(trained_data['historical_df'])}")
        print(f"   Características usadas: {len(trained_data['features_used'])}")
        print(f"   Desviación estándar: {trained_data['std_dev']:.2f}")

        # ANÁLISIS BÁSICO DE LÍNEA (opcional)
        try:
            bookie_line_input = prompt("\nLínea de la casa (Enter para omitir): ").strip()
            if bookie_line_input:
                bookie_line = float(bookie_line_input)
                diff = adjusted_pred - bookie_line
                print(f"\n🎯 Análisis de Línea:")
                print(f"   Línea Casa de Apuestas: {bookie_line}")
                print(f"   Predicción del Modelo: {adjusted_pred:.1f}")
                print(f"   Diferencia: {diff:+.1f} puntos")

                # Análisis básico sin Kelly
                z_score = (bookie_line - adjusted_pred) / trained_data['std_dev']
                model_under_prob = stats.norm.cdf(z_score) * 100
                model_over_prob = 100 - model_under_prob

                print(f"\n📊 Probabilidades del Modelo:")
                print(f"   Over {bookie_line}: {model_over_prob:.1f}%")
                print(f"   Under {bookie_line}: {model_under_prob:.1f}%")

                stronger_side = "OVER" if adjusted_pred > bookie_line else "UNDER"
                edge_magnitude = abs(diff) / bookie_line * 100 if bookie_line > 0 else 0

                if edge_magnitude > 3:
                    print(f"\n🎯 RECOMENDACIÓN: {stronger_side} ({edge_magnitude:.1f}% edge)")
                elif edge_magnitude > 1:
                    print(f"\n⚖️ Ligera ventaja: {stronger_side} ({edge_magnitude:.1f}% edge)")
                else:
                    print(f"\n🤷 Sin ventaja clara - diferencia mínima")
        except:
            pass  # Continuar sin análisis de línea si hay error

        print("\n" + "="*60)
        if add_q3 == 'si':
            print("🏀 ANÁLISIS DESPUÉS DEL TERCER CUARTO")
        else:
            print("🏀 ANÁLISIS AL DESCANSO")

        print(f"Marcador Actual: {home_team_name} {live_score_home} - {away_team_name} {live_score_away}")
        print(f"Total Actual: {live_score_home + live_score_away} puntos")

        if adjusted_pred is not None:
            remaining_quarters = 1 if add_q3 == 'si' else 2
            predicted_remaining = adjusted_pred - (live_score_home + live_score_away)
            print(f"Predicción Final (Ajustada): {adjusted_pred:.1f} puntos")
            print(f"Puntos Restantes Estimados: {predicted_remaining:.1f} ({remaining_quarters} cuarto{'s' if remaining_quarters > 1 else ''})")

        print("=" * 60 + "\n")

        if predictions:
            print("📊 PROBABILIDADES OVER/UNDER - MÚLTIPLES LÍNEAS:\n")
            for i, pred in enumerate(predictions):
                confidence = "🔥" if max(pred['over_prob'], pred['under_prob']) > 65 else "📈" if max(pred['over_prob'], pred['under_prob']) > 55 else "⚖️"
                print(f"  {confidence} Línea {pred['line']:.1f}: Over {pred['over_prob']:.1f}% | Under {pred['under_prob']:.1f}%")

            print(f"\n🎯 Línea Central (más confiable): {predictions[2]['line']:.1f}")

            print("\n" + "="*60)
            print("🚨 RESUMEN DE ALERTAS CLAVE:")

            key_alerts = []
            if alerts_data.get('live_alerts'):
                key_alerts.extend(alerts_data['live_alerts'][:3])
            if alerts_data.get('pre_game_alerts'):
                key_alerts.extend([alert for alert in alerts_data['pre_game_alerts'] if 'surge' in alert.lower() or 'closing' in alert.lower()][:2])

            if key_alerts:
                for alert in key_alerts:
                    print(f"   • {alert}")
            else:
                print("   ✅ No se detectaron anomalías significativas")

            print("=" * 60)
        else:
            print("❌ No se pudieron generar predicciones.")

    except (ValueError, EOFError):
        print("\nEntrada inválida o saliendo del programa.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()