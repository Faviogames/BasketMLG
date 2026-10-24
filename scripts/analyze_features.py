#!/usr/bin/env python3
"""
Analizador de Importancia de Features del Modelo Entrenado
Carga el modelo XGBoost entrenado y analiza el peso de todas las features
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODELS_FOLDER

def analyze_model_features(league_name="NBA"):
    """Analiza la importancia de features del modelo entrenado"""
    
    print(f"\n{'='*80}")
    print(f"[ANALISIS] IMPORTANCIA DE FEATURES - {league_name}")
    print(f"{'='*80}\n")
    
    # Cargar modelo
    model_path = os.path.join(MODELS_FOLDER, f"{league_name}.joblib")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encontro el modelo en {model_path}")
        print("   Ejecuta primero: python -m core.training")
        return
    
    print(f"[INFO] Cargando modelo desde: {model_path}")
    bundle = joblib.load(model_path)
    
    # Extract model from bundle
    if isinstance(bundle, dict) and 'model' in bundle:
        model = bundle['model']
        print(f"[INFO] Tipo de modelo: {bundle.get('model_type', 'unknown')}")
    else:
        model = bundle
    
    # Extraer importancia de features
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        # Try to get feature names from bundle first, then from model
        if isinstance(bundle, dict) and 'features_used' in bundle:
            feature_names = bundle['features_used']
        elif hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            feature_names = [f"f{i}" for i in range(len(importance))]
    else:
        print("[ERROR] El modelo no tiene informacion de importancia de features")
        return
    
    # Crear DataFrame
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Añadir ranking
    df_importance['Rank'] = range(1, len(df_importance) + 1)
    
    # Mostrar estadísticas generales
    print(f"\n[ESTADISTICAS] GENERALES:")
    print(f"   Total de features: {len(df_importance)}")
    print(f"   Features con importancia > 0: {(df_importance['Importance'] > 0).sum()}")
    print(f"   Features con importancia = 0: {(df_importance['Importance'] == 0).sum()}")
    print(f"   Importancia máxima: {df_importance['Importance'].max():.6f}")
    print(f"   Importancia promedio: {df_importance['Importance'].mean():.6f}")
    
    # TOP 20 Features
    print(f"\n{'='*80}")
    print("[TOP 20] FEATURES MAS IMPORTANTES")
    print(f"{'='*80}")
    print(f"\n{'Rank':<6} {'Feature':<50} {'Importance':>12}")
    print("-" * 80)
    
    for idx, row in df_importance.head(20).iterrows():
        print(f"{row['Rank']:<6} {row['Feature']:<50} {row['Importance']:>12.6f}")
    
    # Analizar nuevas features de mercado
    print(f"\n{'='*80}")
    print("[MERCADO] ANALISIS DE FEATURES DE MERCADO (NUEVAS)")
    print(f"{'='*80}\n")
    
    market_features = ['market_total_diff', 'implied_pace_vs_avg', 'line_volatility_index']
    
    for feat in market_features:
        if feat in df_importance['Feature'].values:
            row = df_importance[df_importance['Feature'] == feat].iloc[0]
            rank = row['Rank']
            importance = row['Importance']
            
            # Categorizar por importancia
            if importance > 0.01:
                status = "[OK] ALTA"
            elif importance > 0.001:
                status = "[WARN] MEDIA"
            elif importance > 0:
                status = "[LOW] BAJA"
            else:
                status = "[NONE] NULA"
            
            print(f"  {feat}:")
            print(f"    Ranking: #{rank} de {len(df_importance)}")
            print(f"    Importancia: {importance:.6f}")
            print(f"    Status: {status}")
            print()
        else:
            print(f"  {feat}:")
            print(f"    [WARN] NO ENCONTRADA en el modelo")
            print(f"    (Posibles razones: feature constante, todos NaN, o no incluida en entrenamiento)")
            print()
    
    # Features con importancia cercana a cero
    print(f"\n{'='*80}")
    print("[BAJA] FEATURES CON IMPORTANCIA MUY BAJA (< 0.0001)")
    print(f"{'='*80}\n")
    
    low_importance = df_importance[df_importance['Importance'] < 0.0001]
    print(f"Total: {len(low_importance)} features")
    
    if len(low_importance) > 0:
        print("\nPrimeras 30:")
        for idx, row in low_importance.head(30).iterrows():
            print(f"  - {row['Feature']} (Importancia: {row['Importance']:.8f})")
    
    # Guardar análisis completo
    output_file = os.path.join(MODELS_FOLDER, f"{league_name}_feature_analysis.csv")
    df_importance.to_csv(output_file, index=False)
    print(f"\n[SAVE] Analisis completo guardado en: {output_file}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analizar importancia de features del modelo")
    parser.add_argument("--league", default="NBA", help="Nombre de la liga (default: NBA)")
    
    args = parser.parse_args()
    
    analyze_model_features(args.league)
