"""
SHAP Analysis + Model Comparison
- Analizar importancia con SHAP values
- Comparar XGBoost y LightGBM
- Encontrar número óptimo de features
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import warnings
import os
warnings.filterwarnings('ignore')

def compare_models_and_select_features():
    """
    1. Entrenar 2 modelos (XGBoost, LightGBM)
    2. SHAP analysis para feature selection
    3. Recomendar top N features
    """
    
    print("="*70)
    print("SHAP Analysis + Model Comparison (XGBoost vs LightGBM)")
    print("="*70)
    
    # ==========================================
    # 1. CARGAR DATOS
    # ==========================================
    print("\n[1/5] Cargando datos...")
    if not os.path.exists('data/training_dataset_live_v4_stacked.csv'):
        print("Error: No se encuentra data/training_dataset_live_v4_stacked.csv")
        return
        
    df = pd.read_csv('data/training_dataset_live_v4_stacked.csv')
    
    feature_cols = [col for col in df.columns if col != 'final_total']
    # Filtrar solo columnas numéricas
    X = df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = X.columns.tolist()
    y = df['final_total']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Features: {len(feature_cols)}")
    
    # ==========================================
    # 2. ENTRENAR MODELOS
    # ==========================================
    print("\n[2/5] Entrenando modelos...")
    
    models = {}
    results = {}
    
    # XGBoost
    print("   Entrenando XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'MAE': mean_absolute_error(y_test, xgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred))
    }
    
    # LightGBM
    print("   Entrenando LightGBM...")
    lgb_model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    models['LightGBM'] = lgb_model
    results['LightGBM'] = {
        'MAE': mean_absolute_error(y_test, lgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lgb_pred))
    }
    
    # ==========================================
    # 3. COMPARAR RESULTADOS
    # ==========================================
    print("\n[3/5] Comparando modelos...")
    print(f"\n{'='*70}")
    print(f"{'Modelo':<15} {'MAE (pts)':<15} {'RMSE (pts)':<15} {'Mejor?':<10}")
    print(f"{'='*70}")
    
    best_mae = min(results[m]['MAE'] for m in results)
    for model_name, metrics in results.items():
        is_best = " <-- MEJOR" if metrics['MAE'] == best_mae else ""
        print(f"{model_name:<15} {metrics['MAE']:<15.2f} {metrics['RMSE']:<15.2f} {is_best}")
    print(f"{'='*70}")
    
    # Elegir mejor modelo
    best_model_name = min(results, key=lambda x: results[x]['MAE'])
    best_model = models[best_model_name]
    print(f"\n[OK] Mejor modelo: {best_model_name} (MAE: {results[best_model_name]['MAE']:.2f})")
    
    # ==========================================
    # 4. SHAP ANALYSIS
    # ==========================================
    print(f"\n[4/5] Calculando SHAP values ({best_model_name})...")
    
    # Crear explainer (sample para rapidez)
    sample_size = min(100, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)
    
    # Calcular importancia promedio absoluta
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': feature_importance
    }).sort_values('shap_importance', ascending=False)
    
    print("\nTop 20 Features por SHAP:")
    print(f"{'='*70}")
    for idx, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:<35} {row['shap_importance']:>8.4f}")
    
    # ==========================================
    # 5. FEATURE SELECTION
    # ==========================================
    print(f"\n[5/5] Recomendación de features...")
    
    # Probar con diferentes cantidades
    print(f"\n{'='*70}")
    print("Testando con diferentes cantidades de features:")
    print(f"{'='*70}")
    print(f"{'N Features':<15} {'MAE (pts)':<15} {'Mejora vs 41':<15}")
    print(f"{'-'*70}")
    
    baseline_mae = results[best_model_name]['MAE']
    
    for n_features in [10, 15, 20, 25, 30, 35]:
        top_features = importance_df.head(n_features)['feature'].tolist()
        X_train_subset = X_train[top_features]
        X_test_subset = X_test[top_features]
        
        # Re-entrenar con subset
        if best_model_name == 'XGBoost':
            model_temp = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)
        else:  # LightGBM
            model_temp = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
        
        model_temp.fit(X_train_subset, y_train)
        pred_temp = model_temp.predict(X_test_subset)
        mae_temp = mean_absolute_error(y_test, pred_temp)
        
        mejora = baseline_mae - mae_temp
        print(f"{n_features:<15} {mae_temp:<15.2f} {mejora:+.2f} pts")
    
    print(f"{'='*70}")
    
    # ==========================================
    # RECOMENDACIÓN FINAL
    # ==========================================
    print(f"\n{'='*70}")
    print("RECOMENDACION FINAL:")
    print(f"{'='*70}")
    
    # Encontrar sweet spot (entre 15-25 features típicamente)
    recommended_n = 20
    top_features_final = importance_df.head(recommended_n)['feature'].tolist()
    
    print(f"\n1. MEJOR ALGORITMO: {best_model_name}")
    print(f"   MAE: {results[best_model_name]['MAE']:.2f} pts")
    
    print(f"\n2. FEATURES RECOMENDADAS: {recommended_n}")
    print(f"\n   Top {recommended_n} features:")
    for idx, feat in enumerate(top_features_final, 1):
        print(f"   {idx:2d}. {feat}")
    
    print(f"\n{'='*70}")
    
    # Guardar resultados
    os.makedirs('analysis', exist_ok=True)
    importance_df.to_csv('analysis/shap_feature_importance.csv', index=False)
    print(f"\n[OK] SHAP importance guardada: analysis/shap_feature_importance.csv")
    
    return {
        'best_model': best_model_name,
        'results': results,
        'top_features': top_features_final,
        'importance_df': importance_df
    }

if __name__ == "__main__":
    compare_models_and_select_features()
