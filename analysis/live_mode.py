# ===========================================
# Archivo: analysis/live_mode.py
# Lógica del modo en vivo con alertas inteligentes
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
    calculate_live_pace_metrics, calculate_momentum_metrics
)
from analysis.alerts import create_alerts_system

def get_predictions_with_alerts(home_team_name: str, away_team_name: str, 
                               q_scores: Dict[str, int], trained_data: Dict[str, Any]) -> Tuple[float, List[Dict], Dict[str, Any]]:
    """
    🚀 Calcula predicciones Over/Under CON SISTEMA DE ALERTAS INTELIGENTES
    """
    model = trained_data['model']
    std_dev = trained_data['std_dev']
    historical_df = trained_data['historical_df']
    features_used_in_model = trained_data['features_used']
    
    # 🆕 CREAR SISTEMA DE ALERTAS
    alerts_system = create_alerts_system(historical_df)
    
    # 🔍 GENERAR ALERTAS PRE-PARTIDO
    pre_game_alerts = alerts_system.generate_pre_game_alerts(home_team_name, away_team_name)
    
    # Obtener historial de ambos equipos
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

    # Crear DataFrames de historial para cada equipo con métricas de momentum
    home_history_df = pd.DataFrame([{
        'points_scored': r['home_score'] if r['home_team'] == home_team_name else r['away_score'],
        'points_allowed': r['away_score'] if r['home_team'] == home_team_name else r['home_score'],
        'total_score': r['total_score'],
        **{stat: r.get(f'{"home" if r["home_team"] == home_team_name else "away"}_{stat}', np.nan) for stat in ADVANCED_STATS_COLS}
    } for _, r in home_history.iterrows()])
    
    if not home_history_df.empty:
        home_history_df['win'] = (home_history_df['points_scored'] > home_history_df['points_allowed']).astype(int)
        home_history_df['plus_minus'] = home_history_df['points_scored'] - home_history_df['points_allowed']
        momentum_metrics = calculate_momentum_metrics(home_history_df)
        for metric, value in momentum_metrics.items():
            home_history_df[metric] = value

    away_history_df = pd.DataFrame([{
        'points_scored': r['away_score'] if r['away_team'] == away_team_name else r['home_score'],
        'points_allowed': r['home_score'] if r['away_team'] == away_team_name else r['away_score'],
        'total_score': r['total_score'],
        **{stat: r.get(f'{"away" if r["away_team"] == away_team_name else "home"}_{stat}', np.nan) for stat in ADVANCED_STATS_COLS}
    } for _, r in away_history.iterrows()])
    
    if not away_history_df.empty:
        away_history_df['win'] = (away_history_df['points_scored'] > away_history_df['points_allowed']).astype(int)
        away_history_df['plus_minus'] = away_history_df['points_scored'] - away_history_df['points_allowed']
        momentum_metrics = calculate_momentum_metrics(away_history_df)
        for metric, value in momentum_metrics.items():
            away_history_df[metric] = value

    # Calcular estadísticas históricas
    home_stats_5 = get_rolling_stats(home_history_df, 5, cols_to_avg)
    away_stats_5 = get_rolling_stats(away_history_df, 5, cols_to_avg)
    
    # EMA avanzado multi-rango
    home_ema_advanced = get_enhanced_ema_stats(home_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)
    away_ema_advanced = get_enhanced_ema_stats(away_history_df, cols_to_ema + PERFORMANCE_CONTEXT_COLS)
    
    # EMA legacy (compatibilidad)
    home_ema_5 = get_ema_stats(home_history_df, 5, cols_to_ema)
    away_ema_5 = get_ema_stats(away_history_df, 5, cols_to_ema)

    # Combinar todas las características
    features = {}
    features.update({'home_' + k: v for k, v in home_stats_5.items()})
    features.update({'away_' + k: v for k, v in away_stats_5.items()})
    features.update({'home_' + k: v for k, v in home_ema_5.items()})
    features.update({'away_' + k: v for k, v in away_ema_5.items()})
    features.update({'home_' + k: v for k, v in home_ema_advanced.items()})
    features.update({'away_' + k: v for k, v in away_ema_advanced.items()})

    # Diferencias entre equipos
    for stat in cols_to_avg:
        features[f'diff_avg_{stat}_last_5'] = features.get(f'home_avg_{stat}_last_5', np.nan) - features.get(f'away_avg_{stat}_last_5', np.nan)
    for stat in cols_to_ema:
        features[f'diff_ema_{stat}_last_5'] = features.get(f'home_ema_{stat}_last_5', np.nan) - features.get(f'away_ema_{stat}_last_5', np.nan)

    # Características en vivo (por cuartos)
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

    # Nuevas métricas de pace en tiempo real
    quarter_stage = 'q3_end' if features['q3_total'] > 0 else 'halftime'
    live_pace_metrics = calculate_live_pace_metrics(q_scores, quarter_stage)
    features.update(live_pace_metrics)

    # 🆕 GENERAR ALERTAS EN VIVO
    live_alerts = alerts_system.analyze_live_performance(
        home_team_name, away_team_name, q_scores, quarter_stage
    )

    # 🤖 OBTENER CARACTERÍSTICAS DE BALANCE PARA EL MODELO ML (si está disponible)
    balance_features = {}
    try:
        balance_features = alerts_system.get_balance_features_for_model(
            home_team_name, away_team_name, q_scores, quarter_stage
        )
        features.update(balance_features)
    except AttributeError:
        # El sistema de balance aún no está implementado, continuar sin él
        pass
    
    # Mostrar métricas calculadas con EMA avanzado
    print(f"\n📈 Métricas en Tiempo Real:")
    print(f"   Pace Estimado: {live_pace_metrics['live_pace_estimate']:.1f} posesiones/48min")
    print(f"   Eficiencia {home_team_name}: {live_pace_metrics['live_efficiency_home']:.3f}")
    print(f"   Eficiencia {away_team_name}: {live_pace_metrics['live_efficiency_away']:.3f}")
    print(f"   Momentum Shift: {live_pace_metrics.get('live_momentum_shift', 0):.2f}")
    print(f"   Quarter Consistency: {live_pace_metrics.get('quarter_consistency', 0.5):.3f}")
    
    # 🆕 MOSTRAR MÉTRICAS DE BALANCE (si están disponibles)
    if balance_features and balance_features.get('is_game_unbalanced', 0) > 0:
        print(f"\n🚨 DETECCIÓN DE DESBALANCE:")
        print(f"   Balance Score: {balance_features['game_balance_score']:.3f}")
        print(f"   Caída de Intensidad: {balance_features['intensity_drop_factor']:.3f}")
        print(f"   Momentum de Paliza: {balance_features['blowout_momentum']:.3f}")
        print(f"   Caída Q4 Esperada: {balance_features['expected_q4_drop']:.1%}")
    elif balance_features:
        print(f"\n✅ Partido Balanceado:")
        print(f"   Balance Score: {balance_features.get('game_balance_score', 0):.3f} (Equilibrado)")
        print(f"   Intensidad Sostenida: Sin caídas significativas detectadas")
    
    # 🆕 MOSTRAR ALERTAS PRE-PARTIDO
    if pre_game_alerts:
        print(f"\n🎯 Alertas Pre-Partido:")
        for alert in pre_game_alerts:
            print(f"   {alert}")
    
    # 🆕 MOSTRAR ALERTAS EN VIVO
    if live_alerts:
        print(f"\n🚨 Alertas en Tiempo Real:")
        for alert in live_alerts:
            print(f"   {alert}")
    
    # Mostrar momentum analysis EMA
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

    # 🆕 MOSTRAR RESUMEN DE PATRONES POR EQUIPO
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

    # Imputación robusta
    X_pred = pd.DataFrame([features], columns=features_used_in_model)
    
    impute_values = trained_data.get('impute_values', {})
    
    print(f"\n🔧 Verificando {len(X_pred.columns)} características...")
    missing_count = 0
    
    for col in X_pred.columns:
        if X_pred[col].isnull().any():
            missing_count += 1
            if col in impute_values and not np.isnan(impute_values[col]):
                X_pred[col] = X_pred[col].fillna(impute_values[col])
                print(f"   ✅ Imputado {col}: {impute_values[col]:.3f} (promedio entrenamiento)")
            else:
                if col in trained_data['historical_df'].columns:
                    fallback = trained_data['historical_df'][col].mean()
                    if np.isnan(fallback):
                        fallback = 0.0
                    X_pred[col] = X_pred[col].fillna(fallback)
                    print(f"   🔄 Imputado {col}: {fallback:.3f} (fallback histórico)")
                else:
                    X_pred[col] = X_pred[col].fillna(0.0)
                    print(f"   ⚠️ Imputado {col}: 0.0 (valor por defecto)")

    if missing_count > 0:
        print(f"📋 Total de características imputadas: {missing_count}/{len(X_pred.columns)}")
    else:
        print("✅ Todas las características disponibles - sin imputación necesaria")

    if X_pred.isnull().values.any():
        print("\n⚠️ Advertencia: Aún quedan valores NaN después de imputar — rellenando con 0s.")
        X_pred = X_pred.fillna(0.0)

    # Predicción final
    final_total_pred = model.predict(X_pred)[0]
    
    # Generar múltiples líneas con probabilidades
    predictions = []
    for i in range(-2, 3):
        line = round(final_total_pred) + i * (std_dev / 4)
        z_score = (line - final_total_pred) / std_dev
        under_prob = stats.norm.cdf(z_score) * 100
        over_prob = 100 - under_prob
        
        predictions.append({
            'line': line, 
            'over_prob': over_prob, 
            'under_prob': under_prob
        })
        
    # Retornar también las alertas
    return final_total_pred, predictions, {
        'pre_game_alerts': pre_game_alerts,
        'live_alerts': live_alerts,
        'team_summaries': {'home': home_summary, 'away': away_summary}
    }

def live_mode_with_alerts(trained_data: Dict[str, Any]) -> None:
    """🚀 FUNCIÓN PRINCIPAL: Modo en vivo CON ALERTAS INTELIGENTES"""
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
            # Usar la nueva función con alertas
            final_pred, predictions, alerts_data = get_predictions_with_alerts(
                home_team_name, away_team_name, q_scores, trained_data
            )
            
            print(f"\n📋 Información del Modelo:")
            print(f"   Liga: {trained_data['league_name']}")
            print(f"   Partidos en historial: {len(trained_data['historical_df'])}")
            print(f"   Características usadas: {len(trained_data['features_used'])}")
            print(f"   Desviación estándar: {trained_data['std_dev']:.2f}")
            
        else:
            bookie_line = float(prompt("Introduce la línea de la casa de apuestas: "))
            final_pred, predictions, alerts_data = get_predictions_with_alerts(
                home_team_name, away_team_name, q_scores, trained_data
            )
            
            # Análisis de la línea vs predicción
            diff = final_pred - bookie_line
            print(f"\n🎯 Análisis de Línea:")
            print(f"   Línea Casa de Apuestas: {bookie_line}")
            print(f"   Predicción del Modelo: {final_pred:.1f}")
            print(f"   Diferencia: {diff:+.1f} puntos")
            
            if abs(diff) >= 3:
                recommendation = "OVER" if diff > 0 else "UNDER"
                print(f"   🚨 RECOMENDACIÓN FUERTE: {recommendation}")
            elif abs(diff) >= 1.5:
                recommendation = "Over" if diff > 0 else "Under"
                print(f"   💡 Recomendación moderada: {recommendation}")
            else:
                print(f"   ⚖️ Línea ajustada - Sin recomendación clara")
            
            # 🆕 ANÁLISIS CONTEXTUAL DE LA LÍNEA CON ALERTAS
            print(f"\n🔍 Análisis Contextual:")
            if alerts_data['live_alerts']:
                supporting_alerts = []
                contradicting_alerts = []
                
                for alert in alerts_data['live_alerts']:
                    if 'menos' in alert.lower() or 'cold' in alert.lower() or 'fría' in alert.lower():
                        if diff < 0:  # Sugiere UNDER
                            supporting_alerts.append(alert)
                        else:
                            contradicting_alerts.append(alert)
                    elif 'más' in alert.lower() or 'hot' in alert.lower() or 'surge' in alert.lower():
                        if diff > 0:  # Sugiere OVER
                            supporting_alerts.append(alert)
                        else:
                            contradicting_alerts.append(alert)
                
                if supporting_alerts:
                    print(f"   ✅ Alertas que APOYAN la recomendación:")
                    for alert in supporting_alerts[:2]:  # Máximo 2
                        print(f"      • {alert}")
                
                if contradicting_alerts:
                    print(f"   ⚠️ Alertas CONTRARIAS a considerar:")
                    for alert in contradicting_alerts[:2]:  # Máximo 2
                        print(f"      • {alert}")

        print("\n" + "="*60)
        if add_q3 == 'si':
            print("🏀 ANÁLISIS DESPUÉS DEL TERCER CUARTO")
        else:
            print("🏀 ANÁLISIS AL DESCANSO")
            
        print(f"Marcador Actual: {home_team_name} {live_score_home} - {away_team_name} {live_score_away}")
        print(f"Total Actual: {live_score_home + live_score_away} puntos")
        
        if final_pred is not None:
            remaining_quarters = 1 if add_q3 == 'si' else 2
            predicted_remaining = final_pred - (live_score_home + live_score_away)
            print(f"Predicción Final: {final_pred:.1f} puntos")
            print(f"Puntos Restantes Estimados: {predicted_remaining:.1f} ({remaining_quarters} cuarto{'s' if remaining_quarters > 1 else ''})")
            
        print("=" * 60 + "\n")
        
        if predictions:
            print("📊 PROBABILIDADES OVER/UNDER - MÚLTIPLES LÍNEAS:\n")
            for i, pred in enumerate(predictions):
                confidence = "🔥" if max(pred['over_prob'], pred['under_prob']) > 65 else "📈" if max(pred['over_prob'], pred['under_prob']) > 55 else "⚖️"
                print(f"  {confidence} Línea {pred['line']:.1f}: Over {pred['over_prob']:.1f}% | Under {pred['under_prob']:.1f}%")
            
            print(f"\n🎯 Línea Central (más confiable): {predictions[2]['line']:.1f}")
            
            # 🆕 MOSTRAR ALERTAS CLAVE AL FINAL
            print("\n" + "="*60)
            print("🚨 RESUMEN DE ALERTAS CLAVE:")
            
            key_alerts = []
            if alerts_data.get('live_alerts'):
                key_alerts.extend(alerts_data['live_alerts'][:3])  # Top 3 alertas live
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