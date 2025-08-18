# ===========================================
# Archivo: analysis/live_mode.py (v2.5 - FASE 2B LIVE MODE AVANZADO)
# SISTEMA COMPLETO: Alertas + Ajustes por Contexto + UI Profesional + Protección Betting
# ✅ FIXED: Series ambiguous, XGBoost dtype, KeyError context - TODOS LOS BUGS CRÍTICOS
# ===========================================
import numpy as np
import pandas as pd
import scipy.stats as stats
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from typing import Dict, List, Tuple, Any
import traceback

# Imports del sistema
from config import (
    ADVANCED_STATS_COLS, MOMENTUM_STATS_COLS, PERFORMANCE_CONTEXT_COLS,
    UI_MESSAGES
)

# ✅ SOLUCIÓN - Imports robustos con fallbacks:
try:
    from core.features import (
        get_rolling_stats, get_ema_stats, get_enhanced_ema_stats,
        calculate_live_pace_metrics, calculate_momentum_metrics,
        calculate_real_balance_features,
        calculate_team_quarter_trends,
        apply_blowout_adjustment, safe_mean
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

# Imports de análisis con fallback integrado
try:
    from analysis.alerts import create_alerts_system
    ALERTS_AVAILABLE = True
except ImportError:
    print("⚠️ Sistema de alertas no disponible")
    ALERTS_AVAILABLE = False

# ===================================================================
# 🆕 SISTEMA DE PROTECCIÓN BETTING BÁSICO INTEGRADO
# ===================================================================

class BasicBettingProtection:
    """Sistema básico de protección betting integrado"""
    
    def analyze_betting_opportunity(self, model_prediction: float, model_std: float, 
                                  bookie_line: float, odds_input: str = None) -> Dict[str, Any]:
        """Analiza oportunidad de apuesta con protección básica"""
        # Calcular probabilidades del modelo
        z_score_over = (bookie_line - model_prediction) / model_std
        model_under_prob = stats.norm.cdf(z_score_over)
        model_over_prob = 1 - model_under_prob
        
        # Determinar lado más fuerte
        stronger_side = "OVER" if model_prediction > bookie_line else "UNDER"
        stronger_prob = model_over_prob if stronger_side == "OVER" else model_under_prob
        
        # Calcular edge real
        real_edge = abs(model_prediction - bookie_line)
        real_edge_percentage = (real_edge / bookie_line) * 100 if bookie_line > 0 else 0
        
        # Análisis de cuotas si se proporciona
        betting_analysis = {
            'model_over_prob': model_over_prob,
            'model_under_prob': model_under_prob,
            'stronger_side': stronger_side,
            'stronger_prob': stronger_prob,
            'real_edge_percentage': real_edge_percentage,
            'bookie_line': bookie_line,
            'model_prediction': model_prediction
        }
        
        if odds_input and odds_input.strip():
            odds_analysis = self._analyze_odds(odds_input, stronger_prob)
            betting_analysis.update(odds_analysis)
            
            # Generar recomendación
            if 'kelly_percentage' in betting_analysis and betting_analysis['kelly_percentage'] > 1:
                if real_edge_percentage > 5:
                    betting_analysis['recommendation'] = f"APOSTAR AL {stronger_side}"
                    betting_analysis['confidence_level'] = "HIGH"
                elif real_edge_percentage > 2:
                    betting_analysis['recommendation'] = f"Apostar al {stronger_side.lower()}"
                    betting_analysis['confidence_level'] = "MEDIUM"
                else:
                    betting_analysis['recommendation'] = "NO APOSTAR"
                    betting_analysis['confidence_level'] = "LOW"
            else:
                betting_analysis['recommendation'] = "NO APOSTAR"
                betting_analysis['confidence_level'] = "LOW"
        else:
            # Sin cuotas, solo análisis básico
            if real_edge_percentage > 5:
                betting_analysis['recommendation'] = f"STRONG {stronger_side}"
                betting_analysis['confidence_level'] = "HIGH"
            elif real_edge_percentage > 2:
                betting_analysis['recommendation'] = f"Moderate {stronger_side.lower()}"
                betting_analysis['confidence_level'] = "MEDIUM"
            else:
                betting_analysis['recommendation'] = "NO CLEAR EDGE"
                betting_analysis['confidence_level'] = "LOW"
        
        return betting_analysis
    
    def _analyze_odds(self, odds_input: str, model_prob: float) -> Dict[str, Any]:
        """Analiza cuotas y calcula Kelly"""
        try:
            # Detectar formato de cuota
            odds_str = odds_input.strip()
            
            if odds_str.startswith(('+', '-')):
                # Formato americano
                american_odds = int(odds_str)
                if american_odds > 0:
                    decimal_odds = (american_odds / 100) + 1
                else:
                    decimal_odds = (100 / abs(american_odds)) + 1
                odds_format = "americana"
            else:
                # Formato decimal/europeo
                decimal_odds = float(odds_str)
                if decimal_odds > 0:
                    american_odds = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                else:
                    return {'odds_error': 'Cuota inválida'}
                odds_format = "europea"
            
            # Calcular probabilidad implícita de la casa
            bookmaker_prob = 1 / decimal_odds
            
            # Calcular Kelly Criterion
            b = decimal_odds - 1
            p = model_prob
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            kelly_percentage = kelly_fraction * 100
            
            # Stake recomendado por cada 100 unidades
            stake_per_100 = max(0, min(kelly_percentage, 15))  # Límite de 15% por seguridad
            
            return {
                'american_odds': american_odds,
                'decimal_odds': decimal_odds,
                'odds_format_detected': odds_format,
                'bookmaker_implied_prob': bookmaker_prob,
                'kelly_percentage': kelly_percentage,
                'stake_per_100': stake_per_100,
                'negative_edge_alert': kelly_percentage <= 0,
                'risk_warning': kelly_percentage > 10
            }
            
        except (ValueError, ZeroDivisionError) as e:
            return {'odds_error': f'Error procesando cuota: {str(e)}'}
    
    def generate_protection_alerts(self, betting_analysis: Dict[str, Any]) -> List[str]:
        """Genera alertas de protección"""
        alerts = []
        
        # Kelly muy alto
        if betting_analysis.get('kelly_percentage', 0) > 15:
            alerts.append("⚠️ CUIDADO: Kelly muy alto - considera reducir stake")
        
        # Edge negativo
        if betting_analysis.get('negative_edge_alert', False):
            alerts.append("🚫 SIN VENTAJA: La casa está favorecida")
        
        # Confianza baja
        if betting_analysis.get('confidence_level') == 'LOW':
            alerts.append("📊 CONFIANZA BAJA: Oportunidad marginal")
        
        # Recomendación de no apostar
        if 'NO APOSTAR' in betting_analysis.get('recommendation', ''):
            alerts.append("🛡️ PROTECCIÓN: No se recomienda apostar")
        
        return alerts

# Crear instancia global de protección betting
betting_protection_system = BasicBettingProtection()
BETTING_PROTECTION_AVAILABLE = True

# ===================================================================
# 🆕 SISTEMA DE AJUSTES POR CONTEXTO (NUEVA FUNCIONALIDAD)
# ===================================================================

def apply_context_adjustment(prediction: float, balance_features: Dict[str, float], 
                           live_pace_metrics: Dict[str, float], quarter_stage: str) -> Tuple[float, Dict[str, Any]]:
    """
    🎯 SISTEMA DE AJUSTES POR CONTEXTO - CORREGIDO
    ✅ FIX: Corrige KeyError 'adjusted_prediction'
    ✅ NUEVO: Detecta 5 situaciones de contexto y ajusta la predicción
    """
    adjustment_factor = 1.0
    applied_adjustments = []
    
    try:
        # Extraer contexto del partido con valores por defecto seguros
        blowout_momentum = balance_features.get('blowout_momentum', 0.0)
        is_unbalanced = balance_features.get('is_game_unbalanced', 0) == 1
        intensity_drop = balance_features.get('intensity_drop_factor', 1.0)
        
        # Extraer pace context con valores por defecto seguros
        pace_estimate = live_pace_metrics.get('live_pace_estimate', 100)
        enhanced_pace = live_pace_metrics.get('enhanced_pace_estimate', pace_estimate)
        
        # Determinar contextos
        is_close_game = not is_unbalanced
        is_high_pace = enhanced_pace > 105
        is_late_game = quarter_stage in ['q3_end', 'q4']
        
        # CONTEXTO 1: Garbage Time en partidos desbalanceados
        if blowout_momentum > 0.7 and is_unbalanced:
            if quarter_stage == 'q3_end':
                adjustment_factor *= 0.94
                applied_adjustments.append(f"Garbage time Q3: -6% (blowout: {blowout_momentum:.2f})")
            elif quarter_stage == 'q4':
                adjustment_factor *= 0.88
                applied_adjustments.append(f"Garbage time Q4: -12% (blowout: {blowout_momentum:.2f})")
        
        # CONTEXTO 2: Partidos cerrados + pace alto
        elif is_close_game and is_high_pace and is_late_game:
            adjustment_factor *= 1.05
            applied_adjustments.append(f"Partido cerrado + pace alto: +5% (pace: {enhanced_pace:.1f})")
        
        # CONTEXTO 3: Intensidad en declive
        elif intensity_drop < 0.85:
            adjustment_factor *= 0.96
            applied_adjustments.append(f"Caída de intensidad: -4% (factor: {intensity_drop:.3f})")
        
        # CONTEXTO 4: Fouling strategy
        elif is_close_game and quarter_stage == 'q4' and enhanced_pace > 110:
            adjustment_factor *= 1.08
            applied_adjustments.append(f"Estrategia fouling Q4: +8% (pace: {enhanced_pace:.1f})")
        
        # CONTEXTO 5: Partido equilibrado
        elif is_close_game and 95 <= enhanced_pace <= 105:
            applied_adjustments.append("Partido equilibrado: Sin ajustes")
    
    except Exception as e:
        print(f"⚠️ Error en ajuste por contexto: {e}")
        applied_adjustments.append("Error en análisis - usando predicción base")
        adjustment_factor = 1.0
    
    # 🔧 CLAVE CRÍTICA: Calcular adjusted_prediction
    adjusted_prediction = prediction * adjustment_factor
    
    # 🔧 CLAVE CRÍTICA: Asegurar que adjusted_prediction esté en el diccionario
    context_info = {
        'original_prediction': prediction,
        'adjusted_prediction': adjusted_prediction,  # ← ESTA ERA LA CLAVE FALTANTE
        'adjustment_factor': adjustment_factor,
        'adjustment_percentage': (adjustment_factor - 1) * 100,
        'applied_adjustments': applied_adjustments,
        'contexts_detected': {
            'is_close_game': is_close_game,
            'is_high_pace': is_high_pace,
            'is_late_game': is_late_game,
            'blowout_momentum': blowout_momentum,
            'intensity_drop': intensity_drop
        }
    }
    
    return adjusted_prediction, context_info
def get_predictions_with_alerts(home_team_name: str, away_team_name: str, 
                               q_scores: Dict[str, int], trained_data: Dict[str, Any]) -> Tuple[float, List[Dict], Dict[str, Any]]:
    """
    🚀 Calcula predicciones Over/Under CON SISTEMA DE ALERTAS INTELIGENTES Y PACE MEJORADO
    """
    model = trained_data['model']
    std_dev = trained_data['std_dev']
    historical_df = trained_data['historical_df']
    features_used_in_model = trained_data['features_used']
    
    if ALERTS_AVAILABLE:
        alerts_system = create_alerts_system(historical_df)
        pre_game_alerts = alerts_system.generate_pre_game_alerts(home_team_name, away_team_name)
    else:
        pre_game_alerts = []
    
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
    quarter_stage = 'q3_end' if features['q3_total'] > 0 else 'halftime'
    
    if FEATURES_IMPORTS_OK:
        try:
            home_trends = calculate_team_quarter_trends(home_history_df)
            away_trends = calculate_team_quarter_trends(away_history_df)
            team_trends = {'home': home_trends, 'away': away_trends}
            
            live_pace_metrics = calculate_live_pace_metrics(q_scores, quarter_stage, team_trends)
            balance_features = calculate_real_balance_features(q_scores, quarter_stage, (home_team_name, away_team_name))
        except Exception as e:
            print(f"⚠️ Error en métricas avanzadas: {e}")
            live_pace_metrics = {'live_pace_estimate': 100, 'live_efficiency_home': 1.0, 'live_efficiency_away': 1.0}
            balance_features = {'is_game_unbalanced': 0, 'game_balance_score': 0.5}
    else:
        live_pace_metrics = {'live_pace_estimate': 100, 'live_efficiency_home': 1.0, 'live_efficiency_away': 1.0}
        balance_features = {'is_game_unbalanced': 0, 'game_balance_score': 0.5}

    features.update(live_pace_metrics)
    features.update(balance_features)

    if ALERTS_AVAILABLE:
        live_alerts = alerts_system.analyze_live_performance(
            home_team_name, away_team_name, q_scores, quarter_stage, balance_features
        )
        home_summary = alerts_system.get_team_summary(home_team_name)
        away_summary = alerts_system.get_team_summary(away_team_name)
    else:
        live_alerts = []
        home_summary = {'avg_by_quarter': {'q1': 25, 'q2': 25, 'q3': 25, 'q4': 25}, 'second_half_tendency': 0, 'recovery_ability': 0.5}
        away_summary = {'avg_by_quarter': {'q1': 25, 'q2': 25, 'q3': 25, 'q4': 25}, 'second_half_tendency': 0, 'recovery_ability': 0.5}

    # 🚀 MOSTRAR MÉTRICAS EN TIEMPO REAL
    print(f"\n📈 Métricas en Tiempo Real:")
    print(f"   Pace Estimado Original: {live_pace_metrics['live_pace_estimate']:.1f} posesiones/48min")
    if 'enhanced_pace_estimate' in live_pace_metrics:
        enhanced = live_pace_metrics['enhanced_pace_estimate']
        print(f"   Pace Proyectado (48min): {enhanced:.1f} posesiones/48min")
        improvement = enhanced - live_pace_metrics['live_pace_estimate']
        if abs(improvement) > 1:
            trend = "↗️" if improvement > 0 else "↘️"
            print(f"   Ajuste por tendencias: {trend} {improvement:+.1f}")

    print(f"   Eficiencia {home_team_name}: {live_pace_metrics['live_efficiency_home']:.3f}")
    print(f"   Eficiencia {away_team_name}: {live_pace_metrics['live_efficiency_away']:.3f}")
    print(f"   Momentum Shift: {live_pace_metrics.get('live_momentum_shift', 0):.2f}")
    print(f"   Quarter Consistency: {live_pace_metrics.get('quarter_consistency', 0.5):.3f}")
    
    if balance_features.get('is_game_unbalanced', 0) == 1:
        print(f"\n🚨 ALERTA DE DESBALANCE:")
        print(f"   Juego muy desigual - posible impacto en total final")
        print(f"   Balance Score: {balance_features['game_balance_score']:.3f}")
    
    if balance_features.get('intensity_drop_factor', 1.0) < 0.8:
        print(f"\n📉 ALERTA DE INTENSIDAD:")
        print(f"   Caída de intensidad detectada")
        print(f"   Factor: {balance_features['intensity_drop_factor']:.3f}")

    if pre_game_alerts:
        print(f"\n🎯 Alertas Pre-Partido:")
        for alert in pre_game_alerts:
            print(f"   {alert}")
    
    if live_alerts:
        print(f"\n🚨 Alertas en Tiempo Real:")
        for alert in live_alerts:
            print(f"   {alert}")

    # Preparar datos para el modelo
    X_pred = pd.DataFrame([features], columns=features_used_in_model)
    
    impute_values = trained_data.get('impute_values', {})
    
    print(f"\n🔧 Verificando {len(X_pred.columns)} características...")
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
        print(f"📋 Total de características imputadas: {missing_count}/{len(X_pred.columns)}")
    else:
        print("✅ Todas las características disponibles - sin imputación necesaria")

    if X_pred.isnull().values.any():
        print("\n⚠️ Advertencia: Aún quedan valores NaN después de imputar — rellenando con 0s.")
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

    # 🆕 APLICAR AJUSTE POR CONTEXTO
    context_adjusted_prediction, context_info = apply_context_adjustment(
        blowout_adjusted_prediction, balance_features, live_pace_metrics, quarter_stage
    )

    # 🆕 MOSTRAR INFORMACIÓN DE AJUSTES PROFESIONAL
    print(f"\n🎯 Evolución de la Predicción (Para Apostadores):")
    print(f"   🤖 Predicción base del modelo: {final_total_pred:.0f} pts")

    if abs(blowout_adjusted_prediction - final_total_pred) > 0.1:
        blowout_reduction = final_total_pred - blowout_adjusted_prediction
        if score_diff > 15:
            print(f"   🃏 Ajuste por paliza: -{blowout_reduction:.0f} pts (diferencia {score_diff:.0f} pts)")
            print(f"       ↳ Equipos pueden relajarse en el final")

    if abs(context_adjusted_prediction - blowout_adjusted_prediction) > 0.1:
        context_change = context_adjusted_prediction - blowout_adjusted_prediction
        sign = "+" if context_change > 0 else ""
        print(f"   🎯 Ajuste por situación: {sign}{context_change:.0f} pts")
        
        # Explicar el ajuste en lenguaje apostador
        for adjustment in context_info['applied_adjustments']:
            if "garbage time" in adjustment.lower():
                print(f"       ↳ 🥱 Garbage time detectado - equipos aflojando")
            elif "cerrado + pace alto" in adjustment.lower():
                print(f"       ↳ 📈 Partido cerrado y rápido - más fouls esperados")
            elif "caída de intensidad" in adjustment.lower():
                print(f"       ↳ 😴 Intensidad bajando - menos puntos esperados")
            elif "fouling" in adjustment.lower():
                print(f"       ↳ ⏰ Estrategia de fouls en Q4")

    print(f"   🏆 PREDICCIÓN FINAL: {context_adjusted_prediction:.0f} pts")

    # Generar múltiples líneas con probabilidades
    predictions = []
    center_line = context_adjusted_prediction 
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
        
    return context_adjusted_prediction, predictions, {
        'pre_game_alerts': pre_game_alerts,
        'live_alerts': live_alerts,
        'team_summaries': {'home': home_summary, 'away': away_summary},
        'raw_prediction': final_total_pred,
        'blowout_adjusted': blowout_adjusted_prediction,
        'context_adjusted': context_adjusted_prediction,
        'context_info': context_info,
        'score_diff': score_diff
    }

def live_mode_with_alerts(trained_data: Dict[str, Any]) -> None:
    """🚀 FUNCIÓN PRINCIPAL: Modo en vivo CON ALERTAS INTELIGENTES Y PACE MEJORADO"""
    print(f"\n{UI_MESSAGES['live_mode_title']}")
    print(UI_MESSAGES['live_mode_subtitle'])
    
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
        
        start_live = prompt("¿Empezar predicción con análisis avanzado y alertas? (si/no): ").lower().strip()
        if start_live == 'si':
            adjusted_pred, predictions, alerts_data = get_predictions_with_alerts(
                home_team_name, away_team_name, q_scores, trained_data
            )
            
            print(f"\n📋 Información del Modelo:")
            print(f"   Liga: {trained_data['league_name']}")
            print(f"   Partidos en historial: {len(trained_data['historical_df'])}")
            print(f"   Características usadas: {len(trained_data['features_used'])}")
            print(f"   Desviación estándar: {trained_data['std_dev']:.2f}")
            
        else:
            bookie_line = float(prompt("Introduce la línea de la casa de apuestas: "))
            adjusted_pred, predictions, alerts_data = get_predictions_with_alerts(
                home_team_name, away_team_name, q_scores, trained_data
            )
            
            diff = adjusted_pred - bookie_line
            print(f"\n🎯 Análisis de Línea:")
            print(f"   Línea Casa de Apuestas: {bookie_line}")
            print(f"   Predicción del Modelo (Final): {adjusted_pred:.1f}")
            print(f"   Diferencia: {diff:+.1f} puntos")
            
            # 🆕 SISTEMA DE PROTECCIÓN BETTING INTEGRADO
            if BETTING_PROTECTION_AVAILABLE:
                print(f"\n🛡️ ANÁLISIS DE PROTECCIÓN BETTING:")
                
                # Solicitar cuota
                try:
                    odds_input = prompt("Cuota (ej: -110, +150, 1.91, 2.50): ").strip()
                except (EOFError, KeyboardInterrupt):
                    odds_input = None
                
                # Análisis completo de protección
                try:
                    betting_analysis = betting_protection_system.analyze_betting_opportunity(
                        adjusted_pred, 
                        trained_data['std_dev'], 
                        bookie_line, 
                        odds_input
                    )
                    
                    # Mostrar análisis
                    print(f"\n📊 Probabilidades del Modelo:")
                    print(f"   Over: {betting_analysis['model_over_prob']:.1%}")
                    print(f"   Under: {betting_analysis['model_under_prob']:.1%}")
                    print(f"   Lado más fuerte: {betting_analysis['stronger_side']}")
                    
                    if odds_input and odds_input.strip():
                        if 'odds_error' in betting_analysis:
                            print(f"\n❌ Error en cuota: {betting_analysis['odds_error']}")
                        else:
                            print(f"\n💰 Análisis de Cuotas:")
                            print(f"   Cuota ingresada: {odds_input}")
                            print(f"   Edge real: {betting_analysis['real_edge_percentage']:+.1f}%")
                            
                            if 'kelly_percentage' in betting_analysis:
                                print(f"\n🎯 Kelly Criterion:")
                                print(f"   Kelly: {betting_analysis['kelly_percentage']:.1f}%")
                                print(f"   Stake recomendado: {betting_analysis.get('stake_per_100', 0):.0f} de cada 100")
                    
                    print(f"\n🚨 RECOMENDACIÓN: {betting_analysis['recommendation']}")
                    print(f"📈 Confianza: {betting_analysis['confidence_level']}")
                    
                    # Guardar para uso posterior
                    alerts_data['betting_analysis'] = betting_analysis
                    
                except Exception as e:
                    print(f"⚠️ Error en sistema de protección: {e}")

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