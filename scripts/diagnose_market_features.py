#!/usr/bin/env python3
"""
Script para diagnosticar por qué las features de mercado tienen importancia 0
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_FOLDER
from core.data_processing import load_league_data
from core.features import calculate_features

def diagnose_market_features():
    print("\n" + "="*80)
    print("[DIAGNOSTICO] FEATURES DE MERCADO")
    print("="*80 + "\n")
    
    # Cargar datos
    all_data = load_league_data(DATA_FOLDER)
    league_data = all_data.get('NBA', [])
    
    print(f"[INFO] Total de partidos: {len(league_data)}")
    
    # Verificar cuántos tienen line_odds
    has_line_odds = sum(1 for m in league_data if 'line_odds' in m and m['line_odds'])
    print(f"[INFO] Partidos con line_odds: {has_line_odds} ({has_line_odds/len(league_data)*100:.1f}%)")
    
    # Muestra de line_odds
    print("\n[MUESTRA] Primeros 3 line_odds:")
    for i, match in enumerate(league_data[:3]):
        if 'line_odds' in match:
            print(f"  Partido {i+1}: {match.get('line_odds')}")
    
    # Calcular features
    print("\n[INFO] Calculando features...")
    df = calculate_features(league_data)
    
    # Analizar las features de mercado
    market_features = [
        'home_market_total_diff', 'away_market_total_diff', 'diff_market_total_diff',
        'home_implied_pace_vs_avg', 'away_implied_pace_vs_avg', 'diff_implied_pace_vs_avg',
        'home_line_volatility_index', 'away_line_volatility_index', 'diff_line_volatility_index'
    ]
    
    print(f"\n{'='*80}")
    print("[ANALISIS] VALORES DE FEATURES DE MERCADO")
    print("="*80 + "\n")
    
    for feat in market_features:
        if feat in df.columns:
            values = df[feat]
            non_zero = (values != 0).sum()
            unique_vals = values.nunique()
            
            print(f"{feat}:")
            print(f"  Valores != 0: {non_zero} ({non_zero/len(df)*100:.1f}%)")
            print(f"  Valores unicos: {unique_vals}")
            print(f"  Min: {values.min():.4f}, Max: {values.max():.4f}")
            print(f"  Mean: {values.mean():.4f}, Std: {values.std():.4f}")
            print(f"  NaN: {values.isna().sum()}")
            
            if unique_vals <= 5:
                print(f"  Valores: {sorted(values.unique())}")
            print()
        else:
            print(f"{feat}: [WARN] NO ENCONTRADA\n")

if __name__ == "__main__":
    diagnose_market_features()
