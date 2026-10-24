"""
Entrenar Metamodel FINAL con LightGBM
Versión Producción
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import json

def train_final_metamodel_lightgbm():
    """
    Entrena modelo FINAL para producción con:
    - Algoritmo: LightGBM
    - Features: Todas las disponibles en dataset v3
    """
    
    print("="*70)
    print("ENTRENAMIENTO FINAL - Metamodel v2 (LightGBM)")
    print("="*70)
    
    # 1. Cargar dataset
    print("\n[1/5] Cargando dataset...")
    dataset_path = 'data/training_dataset_live_v4_stacked.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: No se encuentra {dataset_path}")
        return

"""
Entrenar Metamodel FINAL con LightGBM
Versión Producción
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import json

def train_final_metamodel_lightgbm():
    """
    Entrena modelo FINAL para producción con:
    - Algoritmo: LightGBM
    - Features: Todas las disponibles en dataset v3
    """
    
    print("="*70)
    print("ENTRENAMIENTO FINAL - Metamodel ADJUSTER (LightGBM)")
    print("="*70)
    
    # 1. Cargar dataset
    print("\n[1/5] Cargando dataset...")
    dataset_path = 'data/training_dataset_live_v4_stacked.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: No se encuentra {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    print(f"   Registros: {len(df)}")
    print(f"   Features: {len(df.columns) - 2}") # Excluyendo match_id y final_total
    
    # 2. Preparar datos
    print("\n[2/5] Preparando datos...")
    
    # CALCULAR TARGET: RESIDUAL
    # Residual = Real - Base_Prediction
    # El modelo aprenderá a corregir el error del modelo base
    if 'base_model_prediction' not in df.columns:
        print("Error: 'base_model_prediction' no encontrado en dataset")
        return

    df['residual'] = df['final_total'] - df['base_model_prediction']
    print(f"   Target: Residual (Mean: {df['residual'].mean():.2f}, Std: {df['residual'].std():.2f})")

    # Excluir columnas no numéricas o identificadores
    # MANTENER base_model_prediction como feature!
    exclude_cols = ['match_id', 'final_total', 'residual', 'volatility_class', 'rwt_trend', 'pace_trend', 'qmd_trend']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Asegurar que solo usamos columnas numéricas
    X = df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = X.columns.tolist()
    
    # Verificar que base_model_prediction está en features
    if 'base_model_prediction' not in feature_cols:
        print("⚠️ Warning: base_model_prediction no estaba en features numéricas, intentando añadir...")
        if 'base_model_prediction' in df.columns:
             X['base_model_prediction'] = df['base_model_prediction']
             feature_cols.append('base_model_prediction')

    y = df['residual']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train)} juegos")
    print(f"   Test: {len(X_test)} juegos")
    
    # 3. Entrenar modelo FINAL
    print("\n[3/5] Entrenando LightGBM ADJUSTER (Residual)...")
    
    final_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03, # Lower LR for residual
        max_depth=5,
        num_leaves=20, # Smaller trees to prevent overfitting on noise
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # Entrenar con eval set
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='mae'
    )
    
    # 4. Evaluar modelo
    print("\n[4/5] Evaluando modelo...")
    
    # Predicciones (del Residual)
    resid_pred_train = final_model.predict(X_train)
    resid_pred_test = final_model.predict(X_test)
    
    # Reconstruir Totales para métricas reales
    # Total_Pred = Base + Predicted_Residual
    base_train = X_train['base_model_prediction']
    base_test = X_test['base_model_prediction']
    
    y_train_total = base_train + y_train # This reconstructs the actual total because y_train IS the residual
    y_test_total = base_test + y_test
    
    pred_train_total = base_train + resid_pred_train
    pred_test_total = base_test + resid_pred_test
    
    # Métricas del Residual
    mae_resid = mean_absolute_error(y_test, resid_pred_test)
    
    # Métricas del Total Final
    mae_total = mean_absolute_error(y_test_total, pred_test_total)
    rmse_total = np.sqrt(mean_squared_error(y_test_total, pred_test_total))
    
    # Comparar con Base Model solo
    mae_base = mean_absolute_error(y_test_total, base_test)
    
    print(f"\n{'='*70}")
    print(f"METRICAS ADJUSTER (LightGBM Residual):")
    print(f"{'='*70}")
    print(f"Residual MAE: {mae_resid:.2f}")
    print(f"{'-'*70}")
    print(f"MEJORA DEL MODELO:")
    print(f"Base Model MAE:     {mae_base:.2f}")
    print(f"Adjusted Model MAE: {mae_total:.2f}")
    print(f"Improvement:        {mae_base - mae_total:.2f} pts")
    print(f"{'='*70}")
    
    # Feature importance
    print(f"\nTop 10 Features:")
    feature_importance = final_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:6.2f}")
    
    # 5. Guardar modelo y metadatos
    print(f"\n[5/5] Guardando modelo...")
    
    os.makedirs('metamodel/models', exist_ok=True)
    
    # Guardar modelo
    model_path = 'metamodel/models/lightgbm_adjuster_v1.txt'
    final_model.booster_.save_model(model_path)
    print(f"   Modelo guardado: {model_path}")
    
    # Guardar feature list
    features_path = 'metamodel/models/features_adjuster_v1.txt'
    with open(features_path, 'w') as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")
    print(f"   Features guardadas: {features_path}")
    
    # Guardar metadatos
    metadata = {
        'algorithm': 'LightGBM Residual Adjuster',
        'version': 'v1_adjuster',
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'mae_residual': float(mae_resid),
        'mae_total_adjusted': float(mae_total),
        'mae_base_model': float(mae_base),
        'improvement': float(mae_base - mae_total)
    }
    
    with open('metamodel/models/metadata_adjuster_v1.json', 'w') as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    train_final_metamodel_lightgbm()
