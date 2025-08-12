# ===========================================
# Archivo: analysis/live_mode.py (v1.2)
# Lógica del modo en vivo con PACE PROYECTADO MEJORADO y AJUSTE POR PALIZA
# ===========================================
import numpy as np
import pandas as pd
import scipy.stats as stats
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from typing import Dict, List, Tuple, Any

from config import (
    ADVANCED_STATS_COLS, MOMENTUM_STATS_COLS, PERFORMANCE_CONTEXT_COLS,
    UI_MESSAGES
)
from core.features import (
    get_rolling_stats, get_ema_stats, get_enhanced_ema_stats,
    calculate_live_pace_metrics, calculate_momentum_metrics,
    calculate_real_balance_features,
    # 🆕 IMPORTAR LAS NUEVAS FUNCIONES DE MEJORA
    calculate_team_quarter_trends,
    apply_blowout_adjustment
)
from analysis.alerts import create_alerts_system

def get_predictions_with_alerts(home_team_name: str, away_team_name: str, 
                               q_scores: Dict[str, int], trained_data: Dict[str, Any]) -> Tuple[float, List[Dict], Dict[str, Any]]:
    """
    🚀 Calcula predicciones Over/Under CON SISTEMA DE ALERTAS INTELIGENTES Y PACE MEJORADO
    """
    model = trained_data['model']
    std_dev = trained_data['std_dev']
    historical_df = trained_data['historical_df']
    features_used_in_model = trained_data['features_used']
    
    alerts_system = create_alerts_system(historical_df)
    
    pre_game_alerts = alerts_system.generate_pre_game_alerts(home_team_name, away_team_name)
    
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

    # Crear DataFrames de historial para cada equipo (asegurando que los datos por cuarto existan)
    home_history_list = []
    for _, r in home_history.iterrows():
        is_home = r['home_team'] == home_team_name
        team_hist_data = {
            'points_scored': r['home_score'] if is_home else r['away_score'],
            'points_allowed': r['away_score'] if is_home else r['home_score'],
            'total_score': r['total_score'],
            'q1_points': 0,  # Temporal - no tenemos datos por cuarto en el historial
            'q2_points': 0,  # Temporal - no tenemos datos por cuarto en el historial  
            'q3_points': 0,  # Temporal - no tenemos datos por cuarto en el historial
            'q4_points': 0,  # Temporal - no tenemos datos por cuarto en el historial
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
        momentum_metrics = calculate_momentum_metrics(away_history_df)
        for metric, value in momentum_metrics.items():
            away_history_df[metric] = value

    home_stats_5 = get_rolling_stats(home_history_df, 5, cols_to_avg)
    away_stats_5 = get_rolling_stats(away_history_df, 5, cols_to_avg)
    
    home_ema_advanced = get_enhanced_ema_stats(home_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)
    away_ema_advanced = get_enhanced_ema_stats(away_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)
    
    home_ema_5 = get_ema_stats(home_history_df, 5, cols_to_ema)
    away_ema_5 = get_ema_stats(away_history_df, 5, cols_to_ema)

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

    # 🚀 PASO 4: INTEGRAR CÁLCULO DE TENDENCIAS Y PACE MEJORADO
    quarter_stage = 'q3_end' if features['q3_total'] > 0 else 'halftime'
    
    # Calcular tendencias históricas de los equipos
    home_trends = calculate_team_quarter_trends(home_history_df)
    away_trends = calculate_team_quarter_trends(away_history_df)
    team_trends = {'home': home_trends, 'away': away_trends}
    
    # Calcular métricas de pace, ahora con las tendencias
    live_pace_metrics = calculate_live_pace_metrics(q_scores, quarter_stage, team_trends)
    features.update(live_pace_metrics)

    balance_features = calculate_real_balance_features(q_scores, quarter_stage, (home_team_name, away_team_name))
    features.update(balance_features)

    live_alerts = alerts_system.analyze_live_performance(
        home_team_name, away_team_name, q_scores, quarter_stage, balance_features
    )
    
    # 🚀 PASO 6: MOSTRAR MEJORAS EN LA UI (primera parte)
    print(f"\n📈 Métricas en Tiempo Real:")
    print(f"   Pace Estimado Original: {live_pace_metrics['live_pace_estimate']:.1f} posesiones/48min")
    if 'enhanced_pace_estimate' in live_pace_metrics:
        enhanced = live_pace_metrics['enhanced_pace_estimate']
        print(f"   Pace Proyectado (48min): {enhanced:.1f} posesiones/48min")
        improvement = enhanced - live_pace_metrics['live_pace_estimate']
        if abs(improvement) > 1: # Mostrar solo si el ajuste es notable
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
    
    print(f"\n🎯 Momentum Analysis (EMA):")
    if not home_history_df.empty and not away_history_df.empty:
        home_win_rate = features.get('home_ema_win_rate_short_term_3', features.get('home_avg_win_rate_last_5', 0.5))
        away_win_rate = features.get('away_ema_win_rate_short_term_3', features.get('away_avg_win_rate_last_5', 0.5))
        print(f"   {home_team_name} Win Rate (reciente): {home_win_rate:.1%}")
        print(f"   {away_team_name} Win Rate (reciente): {away_win_rate:.1%}")
        
        home_pm = features.get('home_ema_avg_plus_minus_short_term_3', 0)
        away_pm = features.get('away_ema_avg_plus_minus_short_term_3', 0)
        print(f"   {home_team_name} Plus/Minus Trend: {home_pm:+.1f}")
        print(f"   {away_team_name} Plus/Minus Trend: {away_pm:+.1f}")
    else:
        print("   Insuficiente historial para análisis de momentum")

    print(f"\n📊 Patrones Históricos:")
    home_summary = alerts_system.get_team_summary(home_team_name)
    away_summary = alerts_system.get_team_summary(away_team_name)
    
    print(f"   {home_team_name}:")
    print(f"     • Promedio por cuartos: Q1:{home_summary['avg_by_quarter']['q1']:.1f} Q2:{home_summary['avg_by_quarter']['q2']:.1f} Q3:{home_summary['avg_by_quarter']['q3']:.1f} Q4:{home_summary['avg_by_quarter']['q4']:.1f}")
    print(f"     • Tendencia 2ª mitad: {home_summary['second_half_tendency']:+.1f} pts")
    print(f"     • Capacidad recuperación: {home_summary['recovery_ability']:.1%}")
    
    print(f"   {away_team_name}:")
    print(f"     • Promedio por cuartos: Q1:{away_summary['avg_by_quarter']['q1']:.1f} Q2:{away_summary['avg_by_quarter']['q2']:.1f} Q3:{away_summary['avg_by_quarter']['q3']:.1f} Q4:{away_summary['avg_by_quarter']['q4']:.1f}")
    print(f"     • Tendencia 2ª mitad: {away_summary['second_half_tendency']:+.1f} pts")
    print(f"     • Capacidad recuperación: {away_summary['recovery_ability']:.1%}")

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
    
    # 🚀 PASO 5: APLICAR AJUSTE POR PALIZA
    home_score = q_scores['q1_home'] + q_scores['q2_home'] + q_scores['q3_home']
    away_score = q_scores['q1_away'] + q_scores['q2_away'] + q_scores['q3_away']
    score_diff = abs(home_score - away_score)
    
    quarters_played = 3 if q_scores['q3_home'] > 0 or q_scores['q3_away'] > 0 else 2
    time_remaining_pct = (4 - quarters_played) / 4
    
    adjusted_prediction = apply_blowout_adjustment(final_total_pred, score_diff, time_remaining_pct)
    
    # Generar múltiples líneas con probabilidades usando la predicción AJUSTADA
    predictions = []
    # Usar la predicción ajustada como el centro para las líneas de probabilidad
    center_line = adjusted_prediction 
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
        
    return adjusted_prediction, predictions, {
        'pre_game_alerts': pre_game_alerts,
        'live_alerts': live_alerts,
        'team_summaries': {'home': home_summary, 'away': away_summary},
        'raw_prediction': final_total_pred,
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
            print(f"   Predicción del Modelo (Ajustada): {adjusted_pred:.1f}")
            print(f"   Diferencia: {diff:+.1f} puntos")
            
            if abs(diff) >= 3:
                recommendation = "OVER" if diff > 0 else "UNDER"
                print(f"   🚨 RECOMENDACIÓN FUERTE: {recommendation}")
            elif abs(diff) >= 1.5:
                recommendation = "Over" if diff > 0 else "Under"
                print(f"   💡 Recomendación moderada: {recommendation}")
            else:
                print(f"   ⚖️ Línea ajustada - Sin recomendación clara")
            
            print(f"\n🔍 Análisis Contextual:")
            if alerts_data['live_alerts']:
                supporting_alerts = []
                contradicting_alerts = []
                
                for alert in alerts_data['live_alerts']:
                    if 'menos' in alert.lower() or 'cold' in alert.lower() or 'fría' in alert.lower():
                        if diff < 0:
                            supporting_alerts.append(alert)
                        else:
                            contradicting_alerts.append(alert)
                    elif 'más' in alert.lower() or 'hot' in alert.lower() or 'surge' in alert.lower():
                        if diff > 0:
                            supporting_alerts.append(alert)
                        else:
                            contradicting_alerts.append(alert)
                
                if supporting_alerts:
                    print(f"   ✅ Alertas que APOYAN la recomendación:")
                    for alert in supporting_alerts[:2]:
                        print(f"      • {alert}")
                
                if contradicting_alerts:
                    print(f"   ⚠️ Alertas CONTRARIAS a considerar:")
                    for alert in contradicting_alerts[:2]:
                        print(f"      • {alert}")

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
            # 🚀 PASO 6: MOSTRAR MEJORAS EN LA UI (segunda parte)
            if 'raw_prediction' in alerts_data:
                score_diff = alerts_data.get('score_diff', 0)
                raw_pred = alerts_data['raw_prediction']
                reduction = raw_pred - adjusted_pred
                if reduction > 0.1: # Mostrar solo si hay ajuste
                    print(f"   🛡️ Ajuste por paliza: -{reduction:.1f} pts (diferencia de {score_diff:.0f} pts)")

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
