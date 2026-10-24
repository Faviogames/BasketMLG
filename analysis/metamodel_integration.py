# ===========================================
# analysis/metamodel_integration.py (v5.0 HUBER META-LEARNER)
# ===========================================
# Integración del Meta-Learner Huber (The Judge)
# ===========================================

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Any, Optional
from datetime import datetime
from core.live_features import calculate_pacer_features, calculate_runner_flags

# Rutas a los modelos (Actualizadas para v5.0)
MODEL_PATH = os.path.join('metamodel', 'models', 'meta_learner_v1_final.joblib')

# Variables globales para caché
_METAMODEL = None

def cargar_metamodelo():
    """
    Carga el Meta-Learner Huber.
    Usa caché para no recargar en cada llamada.
    """
    global _METAMODEL
    
    if _METAMODEL is not None:
        return _METAMODEL

    try:
        print(f"[METAMODEL] Cargando modelo desde: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encuentra el modelo en {MODEL_PATH}")
            
        # Cargar modelo Joblib (HuberRegressor)
        _METAMODEL = joblib.load(MODEL_PATH)
        
        print(f"[METAMODEL] Modelo Huber cargado exitosamente.")
        return _METAMODEL

    except Exception as e:
        print(f"[ERROR] Error crítico cargando metamodelo: {e}")
        return None

def aplicar_metamodelo_ai(base_prediction: float, 
                         q_scores: Dict, 
                         balance_features: Dict, 
                         live_pace_metrics: Dict, 
                         time_info: Dict,
                         live_signals: Dict = None,
                         team_trends: Dict = None,
                         bookie_line: float = None,
                         team_stats: Dict = None) -> Tuple[float, Dict]:
    """
    Aplica el Meta-Learner Huber para combinar Base + Pacer.
    
    Args:
        base_prediction: Predicción del modelo base
        q_scores: Diccionario con scores por cuarto
        balance_features: Features de balance del juego
        live_pace_metrics: Métricas de ritmo
        time_info: Información de tiempo y etapa
        live_signals: Señales live avanzadas (Pulse Engine)
        team_trends: Tendencias de equipos (opcional)
        bookie_line: Línea de la casa de apuestas (opcional)
        team_stats: Estadísticas base de los equipos (opcional)
        
    Returns:
        Tuple(prediccion_final, metadatos_ajuste)
    """
    model = cargar_metamodelo()
    
    if model is None:
        return base_prediction, {'error': 'Model not loaded'}

    try:
        # 1. Calcular The Pacer (Poisson Projection)
        current_total = q_scores.get('q1_home', 0) + q_scores.get('q1_away', 0) + \
                       q_scores.get('q2_home', 0) + q_scores.get('q2_away', 0) + \
                       q_scores.get('q3_home', 0) + q_scores.get('q3_away', 0) + \
                       q_scores.get('q4_home', 0) + q_scores.get('q4_away', 0)
                       
        pacer_result = calculate_pacer_features(
            current_total=current_total,
            time_info=time_info,
            current_line=bookie_line
        )
        
        pace_poisson_projection = pacer_result.get('pacer_projection', base_prediction)
        
        # 2. Calcular The Runner (Volatility Flag)
        quarter_stage = time_info.get('quarter_stage', 'unknown')
        runner_result = calculate_runner_flags(q_scores, quarter_stage)
        run_active_flag = runner_result.get('run_active_flag', 0)
        
        # 3. Preparar input para Meta-Learner
        # Features: ['base_model_prediction', 'pace_poisson_projection']
        # NOTA: El modelo actual solo usa estas 2. En el futuro añadiremos run_active_flag.
        X_input = pd.DataFrame([{
            'base_model_prediction': base_prediction,
            'pace_poisson_projection': pace_poisson_projection
        }])
        
        # 4. Predecir
        final_prediction = model.predict(X_input)[0]
        
        # 5. Calcular ajuste
        adjustment = final_prediction - base_prediction
        
        metadata_out = {
            'chronos_prediction': float(final_prediction),
            'adjustment': float(adjustment),
            'model_version': 'v5.0_huber',
            'components': {
                'base_prediction': float(base_prediction),
                'pacer_projection': float(pace_poisson_projection),
                'run_active_flag': int(run_active_flag), # Logueado para debug
                'volatility_index': float(runner_result.get('volatility_index', 0.0))
            },
            'pacer_prob_over': pacer_result.get('pacer_prob_over', 0.5),
            'pacer_prob_under': 1.0 - pacer_result.get('pacer_prob_over', 0.5)
        }
        
        return float(final_prediction), metadata_out

    except Exception as e:
        print(f"[ERROR] Fallo en predicción Metamodelo: {e}")
        import traceback
        traceback.print_exc()
        return base_prediction, {'error': str(e)}
