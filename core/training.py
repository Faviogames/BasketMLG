# ===========================================
# Archivo: core/training.py (v2.0) - CON AUDITORÍA DE FEATURES
# Ahora incluye análisis de importancia y detección de features inútiles
# ===========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json
import os
import numpy as np

# 🆕 IMPORTS ACTUALIZADOS PARA LA NUEVA ESTRUCTURA
from config import DATA_FOLDER, MODELS_FOLDER, PROCESSED_FILES_PATH, FEATURES_TO_USE
from core.features import calculate_features
from core.data_processing import load_league_data

def load_processed_files():
    """Carga la lista de archivos ya procesados."""
    if os.path.exists(PROCESSED_FILES_PATH):
        try:
            with open(PROCESSED_FILES_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_processed_files(files_list):
    """Guarda la lista actualizada de archivos procesados."""
    with open(PROCESSED_FILES_PATH, 'w', encoding='utf-8') as f:
        json.dump(files_list, f)

def analyze_feature_importance(model, feature_names, league_name):
    """
    🔍 NUEVA FUNCIÓN: Analiza qué features realmente usa el modelo
    """
    print(f"\n{'='*60}")
    print(f"🔍 AUDITORÍA DE FEATURES - Liga: {league_name}")
    print(f"{'='*60}")
    
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    
    # Ordenar por importancia
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # 🎯 TOP FEATURES MÁS IMPORTANTES
    print(f"\n🎯 TOP 15 FEATURES MÁS IMPORTANTES:")
    total_importance_top15 = 0
    for i, (feature, importance) in enumerate(feature_importance[:15]):
        total_importance_top15 += importance
        print(f"{i+1:2d}. {feature:<40} {importance:.4f} ({importance*100:.1f}%)")
    
    print(f"\n📊 Las top 15 features representan {total_importance_top15*100:.1f}% del poder predictivo")
    
    # ❌ BOTTOM FEATURES MENOS ÚTILES  
    print(f"\n❌ BOTTOM 15 FEATURES MENOS ÚTILES:")
    bottom_15 = feature_importance[-15:]
    total_importance_bottom15 = 0
    for i, (feature, importance) in enumerate(bottom_15):
        total_importance_bottom15 += importance
        rank = len(feature_importance) - 14 + i
        print(f"{rank:2d}. {feature:<40} {importance:.4f} ({importance*100:.1f}%)")
    
    print(f"\n📉 Las bottom 15 features solo aportan {total_importance_bottom15*100:.1f}% del poder predictivo")
    
    # 🗑️ FEATURES POSIBLEMENTE INÚTILES
    useless_threshold = 0.001  # Menos de 0.1% de importancia
    useless_features = [f for f, imp in feature_importance if imp < useless_threshold]
    
    if useless_features:
        print(f"\n🗑️ FEATURES POSIBLEMENTE INÚTILES ({len(useless_features)} features con <0.1% importancia):")
        for i, feature in enumerate(useless_features, 1):
            importance = next(imp for f, imp in feature_importance if f == feature)
            print(f"{i:2d}. {feature:<40} {importance:.6f}")
        
        print(f"\n💡 RECOMENDACIÓN: Considera eliminar estas {len(useless_features)} features para:")
        print(f"   ✅ Reducir tiempo de entrenamiento")
        print(f"   ✅ Acelerar predicciones")
        print(f"   ✅ Reducir overfitting")
        print(f"   ✅ Simplificar el modelo")
    else:
        print(f"\n✅ Todas las features tienen importancia significativa (>0.1%)")
    
    # 📊 ESTADÍSTICAS GENERALES
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   Total features: {len(feature_names)}")
    print(f"   Features útiles (>0.1%): {len(feature_names) - len(useless_features)}")
    print(f"   Features posiblemente inútiles: {len(useless_features)}")
    print(f"   Concentración top 10: {sum(imp for _, imp in feature_importance[:10])*100:.1f}%")
    print(f"   Concentración top 25: {sum(imp for _, imp in feature_importance[:25])*100:.1f}%")
    
    return {
        'feature_importance': feature_importance,
        'useless_features': useless_features,
        'top_15_importance': total_importance_top15,
        'bottom_15_importance': total_importance_bottom15,
        'stats': {
            'total_features': len(feature_names),
            'useful_features': len(feature_names) - len(useless_features),
            'useless_features': len(useless_features),
            'top_10_concentration': sum(imp for _, imp in feature_importance[:10]),
            'top_25_concentration': sum(imp for _, imp in feature_importance[:25])
        }
    }

def test_model_without_features(X, y, features_to_remove, league_name):
    """
    🧪 NUEVA FUNCIÓN: Prueba precisión sin ciertas features
    """
    if not features_to_remove:
        print(f"\n✅ No hay features para eliminar en {league_name}")
        return None
    
    print(f"\n{'='*50}")
    print(f"🧪 PRUEBA DE ELIMINACIÓN - {league_name}")
    print(f"{'='*50}")
    
    # Modelo original
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_original = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_original.fit(X_train, y_train)
    y_pred_original = model_original.predict(X_test)
    mae_original = mean_absolute_error(y_test, y_pred_original)
    
    # Modelo sin features sospechosas
    features_available = [f for f in features_to_remove if f in X.columns]
    if not features_available:
        print(f"⚠️ Ninguna de las features a eliminar está presente en el dataset")
        return None
        
    X_reduced = X.drop(columns=features_available, errors='ignore')
    
    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    model_reduced = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_reduced.fit(X_train_red, y_train_red)
    y_pred_reduced = model_reduced.predict(X_test_red)
    mae_reduced = mean_absolute_error(y_test_red, y_pred_reduced)
    
    # Comparar resultados
    mae_difference = mae_reduced - mae_original
    percentage_change = (mae_difference / mae_original) * 100
    
    print(f"📊 RESULTADOS DE LA PRUEBA:")
    print(f"   Features eliminadas: {len(features_available)}")
    print(f"   Features restantes: {len(X_reduced.columns)}")
    print(f"   MAE original: {mae_original:.3f}")
    print(f"   MAE sin features: {mae_reduced:.3f}")
    print(f"   Diferencia: {mae_difference:+.3f} ({percentage_change:+.1f}%)")
    
    if mae_difference <= 0:
        print(f"   🎉 RESULTADO: Modelo MEJORA o se mantiene igual sin estas features!")
        print(f"   💡 RECOMENDACIÓN: ELIMINAR estas features del modelo")
    elif percentage_change < 2:
        print(f"   ✅ RESULTADO: Pérdida mínima (<2%) - Considerar eliminar por simplicidad")
    elif percentage_change < 5:
        print(f"   🤔 RESULTADO: Pérdida moderada (2-5%) - Evaluar trade-off velocidad vs precisión")
    else:
        print(f"   ❌ RESULTADO: Pérdida significativa (>5%) - MANTENER estas features")
    
    return {
        'features_removed': features_available,
        'features_remaining': len(X_reduced.columns),
        'mae_original': mae_original,
        'mae_reduced': mae_reduced,
        'mae_difference': mae_difference,
        'percentage_change': percentage_change,
        'recommendation': 'ELIMINAR' if mae_difference <= 0 else 'MANTENER' if percentage_change > 5 else 'EVALUAR'
    }

def train_models_by_league():
    """Carga los datos, entrena un modelo por liga y los guarda CON AUDITORÍA."""
    print("🔄 MODO DE ENTRENAMIENTO CON AUDITORÍA ACTIVADO...")
    
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    processed_files = load_processed_files()
    
    # Usar el nuevo sistema de carga de datos
    all_data = load_league_data(DATA_FOLDER)
    
    # Actualizar archivos procesados
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".json") and filename not in processed_files:
            processed_files.append(filename)
                
    save_processed_files(processed_files)

    # 📊 RESULTADOS DE AUDITORÍA GLOBAL
    global_audit_results = {}

    for league, league_data in all_data.items():
        print(f"\n{'🏀'*20}")
        print(f"🏀 PROCESANDO Y ENTRENANDO: '{league}'")
        print(f"{'🏀'*20}")
        
        total_matches_found = len(league_data)
        print(f"📊 Total de partidos encontrados: {total_matches_found}")

        df = calculate_features(league_data)
        
        num_valid_data_points = len(df)
        print(f"📊 Data points válidos para entrenamiento: {num_valid_data_points}")
        
        if df.empty or num_valid_data_points < 20:
            print(f"❌ No hay suficientes datos válidos. Se necesitan al menos 20 partidos. Saltando.")
            continue

        features_for_this_model = [f for f in FEATURES_TO_USE if f in df.columns]
        X = df[features_for_this_model]
        y = df['final_total_score']
        
        print(f"🎯 Features incluidas en el modelo: {len(features_for_this_model)}")
        
        # 🔄 ENTRENAMIENTO NORMAL
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        residuals = y_test - y_pred
        std_dev = np.std(residuals)
        
        print(f"✅ Entrenamiento completado:")
        print(f"   MAE: {mae:.2f}")
        print(f"   Std Dev: {std_dev:.2f}")
        
        # 🔍 AUDITORÍA DE FEATURES
        audit_results = analyze_feature_importance(model, features_for_this_model, league)
        
        # 🧪 PRUEBA DE ELIMINACIÓN (solo si hay features inútiles)
        if audit_results['useless_features']:
            elimination_results = test_model_without_features(
                X, y, audit_results['useless_features'], league
            )
            audit_results['elimination_test'] = elimination_results
        else:
            audit_results['elimination_test'] = None
        
        # Guardar resultados de auditoría
        global_audit_results[league] = audit_results
        
        # 💾 GUARDAR MODELO (incluyendo auditoría)
        team_names = list(set(df['home_team'].unique().tolist() + df['away_team'].unique().tolist()))
        
        # Crear valores de imputación
        impute_values = {}
        if not df[features_for_this_model].empty:
            impute_values = df[features_for_this_model].mean().to_dict()
        else:
            impute_values = {f: 0.0 for f in features_for_this_model}

        trained_data = {
            'model': model,
            'features_used': features_for_this_model,
            'std_dev': std_dev,
            'historical_df': df,
            'team_names': team_names,
            'league_name': league,
            'impute_values': impute_values,
            'audit_results': audit_results  # 🆕 INCLUIR AUDITORÍA
        }
        
        safe_league_name = league.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        model_filename = os.path.join(MODELS_FOLDER, f"{safe_league_name}.joblib")
        joblib.dump(trained_data, model_filename)
        
        print(f"💾 Modelo guardado: {model_filename}")
        print(f"📊 Equipos: {len(team_names)} | Partidos: {len(df)} | Features: {len(features_for_this_model)}")
    
    # 📋 RESUMEN GLOBAL DE AUDITORÍA
    print(f"\n{'='*80}")
    print(f"📋 RESUMEN GLOBAL DE AUDITORÍA")
    print(f"{'='*80}")
    
    total_leagues = len(global_audit_results)
    total_useless_features = sum(len(results['useless_features']) for results in global_audit_results.values())
    avg_feature_count = np.mean([results['stats']['total_features'] for results in global_audit_results.values()])
    avg_concentration = np.mean([results['stats']['top_10_concentration'] for results in global_audit_results.values()])
    
    print(f"🏆 Ligas procesadas: {total_leagues}")
    print(f"📊 Promedio de features por liga: {avg_feature_count:.1f}")
    print(f"🎯 Concentración promedio top 10: {avg_concentration*100:.1f}%")
    print(f"🗑️ Total features potencialmente inútiles: {total_useless_features}")
    
    if total_useless_features > 0:
        print(f"\n💡 RECOMENDACIÓN GLOBAL:")
        print(f"   Considera crear una versión optimizada eliminando features inútiles")
        print(f"   Beneficios esperados: +25-30% velocidad, menos overfitting")
    else:
        print(f"\n✅ EXCELENTE: Todas las features parecen contribuir al modelo")
    
    print(f"\n🎉 AUDITORÍA COMPLETA! Revisa los resultados arriba para optimizar tu modelo.")

if __name__ == "__main__":
    train_models_by_league()