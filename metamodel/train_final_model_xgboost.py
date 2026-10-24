"""
Entrenar Metamodel FINAL con 41 features
XGBoost - Versión Producción
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import json

def train_final_metamodel_xgboost():
    """
    Entrena modelo FINAL para producción con:
    - Algoritmo: XGBoost (compatible con Py3.14)
    - Features: 41 (todas las generadas)
    """
    
    print("="*70)
    print("ENTRENAMIENTO FINAL - Metamodel v2 (XGBoost)")
    print("="*70)
    
    # 1. Cargar dataset
    print("\n[1/5] Cargando dataset...")
    df = pd.read_csv('data/training_dataset_live_v2.csv')
    print(f"   Registros: {len(df)}")
    print(f"   Features: {len(df.columns) - 1}")
    
    # 2. Preparar datos
    print("\n[2/5] Preparando datos...")
    feature_cols = [col for col in df.columns if col != 'final_total']
    X = df[feature_cols]
    y = df['final_total']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train)} juegos")
    print(f"   Test: {len(X_test)} juegos")
    
    # 3. Entrenar modelo FINAL
    print("\n[3/5] Entrenando XGBoost FINAL...")
    
    # Tuning ligero para compensar vs CatBoost
    final_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,  # Más bajo para mejor generalización
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    # Entrenar con eval set
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # 4. Evaluar modelo
    print("\n[4/5] Evaluando modelo...")
    
    # Predicciones
    y_pred_train = final_model.predict(X_train)
    y_pred_test = final_model.predict(X_test)
    
    # Métricas
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    print(f"\n{'='*70}")
    print(f"METRICAS FINALES (XGBoost):")
    print(f"{'='*70}")
    print(f"{'Set':<15} {'MAE (pts)':<15} {'RMSE (pts)':<15}")
    print(f"{'-'*70}")
    print(f"{'Train':<15} {mae_train:<15.2f} {rmse_train:<15.2f}")
    print(f"{'Test':<15} {mae_test:<15.2f} {rmse_test:<15.2f}")
    print(f"{'-'*70}")
    print(f"MAPE Test: {mape_test:.2f}%")
    print(f"{'='*70}")
    
    # Feature importance
    print(f"\nTop 10 Features:")
    feature_importance = final_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:6.4f}")
    
    # 5. Guardar modelo y metadatos
    print(f"\n[5/5] Guardando modelo...")
    
    os.makedirs('metamodel/models', exist_ok=True)
    
    # Guardar modelo (formato JSON es más compatible)
    model_path = 'metamodel/models/xgboost_live_v2_final.json'
    final_model.save_model(model_path)
    print(f"   Modelo guardado: {model_path}")
    
    # Guardar feature list
    features_path = 'metamodel/models/features_v2_final.txt'
    with open(features_path, 'w') as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")
    print(f"   Features guardadas: {features_path}")
    
    # Guardar metadatos
    metadata = {
        'algorithm': 'XGBoost',
        'version': 'v2_final_xgb',
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'mae_test': float(mae_test),
        'rmse_test': float(rmse_test),
        'mape_test': float(mape_test)
    }
    
    metadata_path = 'metamodel/models/metadata_v2_final_xgb.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata guardada: {metadata_path}")
    
    print(f"\n{'='*70}")
    print(f"[OK] MODELO XGBOOST LISTO PARA PRODUCCION")
    print(f"{'='*70}")
    
    return {
        'model': final_model,
        'mae_test': mae_test
    }

if __name__ == "__main__":
    train_final_metamodel_xgboost()
