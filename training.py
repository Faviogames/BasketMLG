# ===========================================
# Archivo: core/training.py (v3.1) - SISTEMA DUAL COMPLETO
# Entrena RF y XGBoost simultáneamente con selección de modelo
# ===========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json
import os
import numpy as np

# Imports del proyecto
from config import DATA_FOLDER, MODELS_FOLDER, PROCESSED_FILES_PATH, FEATURES_TO_USE
from core.features import calculate_features
from core.data_processing import load_league_data

# Import del sistema dual
try:
    from core.dual_models import DualModelSystem
    DUAL_SYSTEM_AVAILABLE = True
    print("✅ Sistema dual cargado correctamente")
except ImportError as e:
    DUAL_SYSTEM_AVAILABLE = False
    print(f"⚠️ Sistema dual no disponible: {e}")
    print("🔄 Continuando con Random Forest solamente...")

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
    🔍 Analiza qué features realmente usa el modelo
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️ El modelo {type(model).__name__} no tiene 'feature_importances_'. Saltando auditoría.")
        return {
            'feature_importance': [],
            'useless_features': [],
            'stats': {
                'total_features': len(feature_names),
                'top_10_concentration': 0.0,
            }
        }

    print(f"\n{'='*60}")
    print(f"🔍 AUDITORÍA DE FEATURES - {league_name}")
    print(f"{'='*60}")
    
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # TOP 15 features más importantes
    print(f"\n🎯 TOP 15 FEATURES MÁS IMPORTANTES:")
    total_importance_top15 = 0
    for i, (feature, importance) in enumerate(feature_importance[:15]):
        total_importance_top15 += importance
        print(f"{i+1:2d}. {feature:<40} {importance:.4f} ({importance*100:.1f}%)")
    
    print(f"\n📊 Las top 15 features representan {total_importance_top15*100:.1f}% del poder predictivo")
    
    # Features inútiles
    useless_threshold = 0.001
    useless_features = [f for f, imp in feature_importance if imp < useless_threshold]
    
    if useless_features:
        print(f"\n🗑️ FEATURES POTENCIALMENTE INÚTILES ({len(useless_features)} features con <0.1% importancia):")
        for i, feature in enumerate(useless_features[:10]):  # Solo primeros 10
            importance = next(imp for f, imp in feature_importance if f == feature)
            print(f"{i+1:2d}. {feature:<40} {importance:.6f}")
        
        if len(useless_features) > 10:
            print(f"... y {len(useless_features) - 10} más")
    else:
        print(f"\n✅ Todas las features tienen importancia significativa (>0.1%)")
    
    return {
        'feature_importance': feature_importance,
        'useless_features': useless_features,
        'stats': {
            'total_features': len(feature_names),
            'useful_features': len(feature_names) - len(useless_features),
            'useless_features': len(useless_features),
            'top_10_concentration': sum(imp for _, imp in feature_importance[:10]),
            'top_25_concentration': sum(imp for _, imp in feature_importance[:25])
        }
    }

def train_single_model_legacy(X_train, y_train, X_test, y_test):
    """
    Entrenamiento legacy con Random Forest (fallback)
    """
    print("🌲 Entrenando Random Forest (modo legacy)...")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    residuals = y_test - y_pred
    std_dev = np.std(residuals)
    
    print(f"✅ Random Forest: MAE = {mae:.3f}, Std Dev = {std_dev:.3f}")
    
    return {
        'model': model,
        'mae': mae,
        'std_dev': std_dev,
        'model_type': 'random_forest'
    }

def train_models_by_league():
    """Carga los datos, entrena modelos por liga y los guarda."""
    print("🔄 INICIANDO ENTRENAMIENTO...")
    
    if DUAL_SYSTEM_AVAILABLE:
        print("🤖 MODO: Sistema Dual (RF + XGBoost)")
    else:
        print("🌲 MODO: Random Forest solamente")
    
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    processed_files = load_processed_files()
    all_data = load_league_data(DATA_FOLDER)
    
    if not all_data:
        print("❌ No se encontraron datos para procesar")
        return
    
    # Actualizar archivos procesados
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".json") and filename not in processed_files:
            processed_files.append(filename)
    save_processed_files(processed_files)

    # 📊 Resultados de auditoría global
    global_audit_results = {}

    for league, league_data in all_data.items():
        print(f"\n{'🏀'*20}")
        print(f"🏀 PROCESANDO Y ENTRENANDO: '{league}'")
        print(f"{'🏀'*20}")
        
        total_matches_found = len(league_data)
        print(f"📊 Total de partidos encontrados: {total_matches_found}")

        # Calcular features
        try:
            df = calculate_features(league_data)
        except Exception as e:
            print(f"❌ Error calculando features para {league}: {e}")
            continue
        
        num_valid_data_points = len(df)
        print(f"📊 Data points válidos para entrenamiento: {num_valid_data_points}")
        
        if df.empty or num_valid_data_points < 20:
            print(f"❌ No hay suficientes datos válidos para '{league}'. Se necesitan al menos 20. Saltando.")
            continue

        # Preparar datos
        features_for_this_model = [f for f in FEATURES_TO_USE if f in df.columns]
        X = df[features_for_this_model]
        y = df['final_total_score']
        
        print(f"🎯 Features incluidas en el modelo: {len(features_for_this_model)}")
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- ENTRENAMIENTO ---
        if DUAL_SYSTEM_AVAILABLE:
            print(f"\n🚀 INICIANDO SISTEMA DUAL...")
            
            try:
                # Crear sistema dual
                dual_system = DualModelSystem()
                
                # Verificar modelos disponibles
                available_models = dual_system.get_available_models()
                print(f"📊 Modelos disponibles: {available_models}")
                
                if len(available_models) < 2:
                    print(f"⚠️ Solo {len(available_models)} modelo(s) disponible(s). Usando entrenamiento simple.")
                    model_result = train_single_model_legacy(X_train, y_train, X_test, y_test)
                    selected_model_type = 'random_forest'
                else:
                    # Pasar datos adicionales al sistema
                    dual_system.features_used = features_for_this_model
                    dual_system.historical_df = df
                    dual_system.team_names = list(set(df['home_team'].unique().tolist() + df['away_team'].unique().tolist()))
                    dual_system.impute_values = df[features_for_this_model].mean().to_dict()
                    
                    # Entrenar todos los modelos
                    print(f"🚀 Entrenando {len(available_models)} modelos...")
                    all_results = dual_system.train_all_models(X_train, y_train, X_test, y_test)
                    
                    # --- SELECCIÓN DE MODELO ---
                    # 🎯 CONFIGURA AQUÍ TU PREFERENCIA
                    model_preference = "interactive"  # Opciones: "auto", "random_forest", "xgboost", "interactive"
                    
                    print(f"\n🎯 Método de selección: {model_preference}")
                    
                    # Obtener modelo según preferencia
                    selected_model_type, model_result = dual_system.get_model_by_preference(model_preference)
                    
                    if not model_result:
                        print(f"❌ No se pudo seleccionar un modelo para {league}. Usando Random Forest.")
                        model_result = train_single_model_legacy(X_train, y_train, X_test, y_test)
                        selected_model_type = 'random_forest'
                    else:
                        print(f"\n🎯 MODELO SELECCIONADO PARA {league}: {selected_model_type.upper()}")
                
            except Exception as e:
                print(f"❌ Error en sistema dual: {e}")
                print("🔄 Fallback a entrenamiento legacy...")
                model_result = train_single_model_legacy(X_train, y_train, X_test, y_test)
                selected_model_type = 'random_forest'
        else:
            # Entrenamiento legacy
            model_result = train_single_model_legacy(X_train, y_train, X_test, y_test)
            selected_model_type = 'random_forest'
        
        # Extraer datos del resultado
        model = model_result['model']
        mae = model_result['mae']
        std_dev = model_result['std_dev']
        
        print(f"✅ Entrenamiento completado:")
        print(f"   Modelo: {selected_model_type.upper()}")
        print(f"   MAE: {mae:.2f}")
        print(f"   Std Dev: {std_dev:.2f}")
        
        # 🔍 AUDITORÍA DE FEATURES
        audit_results = analyze_feature_importance(model, features_for_this_model, f"{league} ({selected_model_type})")
        global_audit_results[f"{league}_{selected_model_type}"] = audit_results
        
        # --- GUARDADO DE MODELOS ---
        team_names = list(set(df['home_team'].unique().tolist() + df['away_team'].unique().tolist()))
        
        # Crear valores de imputación
        impute_values = {}
        if not df[features_for_this_model].empty:
            impute_values = df[features_for_this_model].mean().to_dict()
        else:
            impute_values = {f: 0.0 for f in features_for_this_model}

        # Datos del modelo principal (compatibilidad con código existente)
        trained_data = {
            'model': model,
            'model_type': selected_model_type,
            'features_used': features_for_this_model,
            'std_dev': std_dev,
            'historical_df': df,
            'team_names': team_names,
            'league_name': league,
            'impute_values': impute_values,
            'audit_results': audit_results,
            'mae': mae
        }
        
        # Guardar modelo principal
        safe_league_name = league.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        model_filename = os.path.join(MODELS_FOLDER, f"{safe_league_name}.joblib")
        joblib.dump(trained_data, model_filename)
        
        print(f"💾 Modelo principal guardado: {safe_league_name}.joblib")
        
        # Guardar modelos duales si están disponibles
        if DUAL_SYSTEM_AVAILABLE and 'dual_system' in locals() and hasattr(dual_system, 'results') and dual_system.results:
            try:
                saved_files, primary_model = dual_system.save_models(league, MODELS_FOLDER, selected_model_type)
                print(f"💾 Modelos duales guardados: {len(saved_files)} archivos")
            except Exception as e:
                print(f"⚠️ Error guardando modelos duales: {e}")
        
        print(f"📊 Resumen: {len(team_names)} equipos | {len(df)} partidos | {len(features_for_this_model)} features")

    # 📋 RESUMEN GLOBAL DE AUDITORÍA
    print(f"\n{'='*80}")
    print(f"📋 RESUMEN GLOBAL DE AUDITORÍA")
    print(f"{'='*80}")
    
    total_leagues = len(global_audit_results)
    print(f"🏆 Ligas procesadas: {total_leagues}")
    
    if global_audit_results:
        # Estadísticas promedio
        avg_feature_count = np.mean([results['stats']['total_features'] for results in global_audit_results.values() if results.get('stats')])
        avg_concentration = np.mean([results['stats']['top_10_concentration'] for results in global_audit_results.values() if results.get('stats')])
        total_useless_features = sum(len(results['useless_features']) for results in global_audit_results.values())
        
        print(f"📊 Promedio de features por liga: {avg_feature_count:.1f}")
        print(f"🎯 Concentración promedio top 10: {avg_concentration*100:.1f}%")
        print(f"🗑️ Total features potencialmente inútiles: {total_useless_features}")
        
        if total_useless_features > 0:
            print(f"\n💡 RECOMENDACIÓN GLOBAL:")
            print(f"   Considera crear una versión optimizada eliminando features inútiles")
            print(f"   Beneficios esperados: +25-30% velocidad, menos overfitting")
        else:
            print(f"\n✅ EXCELENTE: Todas las features parecen contribuir al modelo")
    
    print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
    
    if DUAL_SYSTEM_AVAILABLE:
        print(f"🤖 Sistema dual utilizado con éxito")
    else:
        print(f"🌲 Sistema legacy Random Forest utilizado")

if __name__ == "__main__":
    train_models_by_league()