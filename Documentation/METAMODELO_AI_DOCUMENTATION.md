> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](auditoria/CONTEXTO_DOCUMENTATION.md) y [estado vigente](auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Documentación del MetaModeloAI - Sistema de Predicción Avanzado

## Resumen Ejecutivo

El MetaModeloAI es un componente avanzado del sistema Basketball MLG v2 que implementa un metamodelo basado en CatBoost para corregir y refinar predicciones en tiempo real. Este sistema utiliza características específicas del contexto del juego para aplicar ajustes inteligentes con límites de seguridad.

## Arquitectura del Sistema

### Componentes Principales

1. **MetaModeloAI Class**: Clase principal que encapsula toda la funcionalidad
2. **CatBoostRegressor**: Modelo de machine learning para predicciones robustas
3. **Safety Rails System**: Mecanismo de protección contra predicciones extremas
4. **Feature Calculator**: Sistema de cálculo de características en tiempo real

### Flujo de Procesamiento

```
Estado del Juego → Cálculo de Features → Predicción CatBoost → Safety Rails → Predicción Final
```

## Características Detalladas

### 1. Inicialización y Configuración

```python
class MetaModeloAI:
    def __init__(self, league_name: str = "NBA"):
        self.league_name = league_name
        self.model = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'catboost_live_v1.cbm')
        self.features_list = [
            'halftime_total', 'pace_halftime', 'efg_home', 'efg_away',
            'to_rate_home', 'to_rate_away', 'ft_rate_home', 'ft_rate_away',
            'home_lead', 'is_close_game', 'momentum_score'
        ]
        self._load_model()
```

#### Parámetros de Configuración

- **league_name**: Nombre de la liga (NBA, WNBA, NBL)
- **model_path**: Ruta al modelo CatBoost entrenado
- **features_list**: Lista de características requeridas para el modelo

### 2. Sistema de Carga de Modelo

#### _load_model()
- **Propósito**: Cargar el modelo CatBoost entrenado desde disco
- **Funcionalidad**:
  - Verificación de existencia del archivo del modelo
  - Manejo robusto de errores
  - Registro de estado del modelo

```python
def _load_model(self):
    """Carga el modelo CatBoost entrenado."""
    if os.path.exists(self.model_path):
        try:
            self.model = CatBoostRegressor()
            self.model.load_model(self.model_path)
            print(f"🤖 MetaModeloAI (CatBoost) cargado correctamente desde {self.model_path}")
        except Exception as e:
            print(f"❌ Error cargando MetaModeloAI: {e}")
            self.model = None
    else:
        print(f"⚠️ No se encontró el modelo en {self.model_path}")
        self.model = None
```

### 3. Sistema de Predicción

#### predict(game_state)
- **Propósito**: Generar predicción basada en el estado del juego
- **Entrada**: Diccionario con estado del juego
- **Salida**: Tupla (predicción_final, metadata)

```python
def predict(self, game_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Genera una predicción basada en el estado del juego.
    Retorna: (prediccion_final, metadata)
    """
    if self.model is None:
        return game_state.get('current_total_projection', 0.0), {'error': 'Model not loaded'}

    # Calcular features
    features_df = self._calculate_features(game_state)
    
    # Predicción
    try:
        prediction = self.model.predict(features_df)[0]
    except Exception as e:
        print(f"Error en predicción AI: {e}")
        return game_state.get('current_total_projection', 0.0), {'error': str(e)}

    # Safety Rails
    final_pred, adjustment_info = self._apply_safety_rails(prediction, game_state)

    metadata = {
        'raw_prediction': float(prediction),
        'final_prediction': float(final_pred),
        'features': features_df.to_dict(orient='records')[0],
        'adjustments': adjustment_info
    }

    return final_pred, metadata
```

### 4. Sistema de Cálculo de Features

#### _calculate_features(state)
- **Propósito**: Calcular características necesarias para el modelo
- **Características Calculadas**:
  - **halftime_total**: Puntos totales al medio tiempo
  - **pace_halftime**: Ritmo estimado al medio tiempo
  - **efg_home/away**: Eficiencia de tiro efectivo
  - **to_rate_home/away**: Tasa de turnovers
  - **ft_rate_home/away**: Tasa de tiros libres
  - **home_lead**: Ventaja del equipo local
  - **is_close_game**: Indicador de juego cerrado
  - **momentum_score**: Puntuación de momentum

```python
def _calculate_features(self, state: Dict[str, Any]) -> pd.DataFrame:
    """Calcula las features necesarias para el modelo."""
    # Extraer datos básicos
    q1_home = state.get('q1_home', 0)
    q1_away = state.get('q1_away', 0)
    q2_home = state.get('q2_home', 0)
    q2_away = state.get('q2_away', 0)
    
    halftime_home = q1_home + q2_home
    halftime_away = q1_away + q2_away
    halftime_total = halftime_home + halftime_away
    
    # Pace (estimado si no hay datos detallados)
    pace = state.get('pace', 95.0)
    
    # Advanced stats (usar live_signals o estimar)
    stats = state.get('team_stats', {})
    
    # Construir vector de features
    data = {
        'halftime_total': halftime_total,
        'pace_halftime': pace,
        'efg_home': stats.get('home_efg', 0.5),
        'efg_away': stats.get('away_efg', 0.5),
        'to_rate_home': stats.get('home_to_rate', 0.13),
        'to_rate_away': stats.get('away_to_rate', 0.13),
        'ft_rate_home': stats.get('home_ft_rate', 0.20),
        'ft_rate_away': stats.get('away_ft_rate', 0.20),
        'home_lead': halftime_home - halftime_away,
        'is_close_game': 1 if abs(halftime_home - halftime_away) <= 10 else 0,
        'momentum_score': (q2_home + q2_away) - (q1_home + q1_away) # Simple momentum
    }
    
    return pd.DataFrame([data], columns=self.features_list)
```

### 5. Sistema de Safety Rails

#### _apply_safety_rails(prediction, state)
- **Propósito**: Aplicar límites de seguridad a predicciones extremas
- **Mecanismos de Seguridad**:
  - Límites globales (150-280 puntos)
  - Restricción de desviación de proyección base
  - Soft clamp para ajustes extremos

```python
def _apply_safety_rails(self, prediction: float, state: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Aplica límites de seguridad a la predicción."""
    adjustments = []
    final_pred = prediction
    
    # 1. Global Bounds
    min_total = 150
    max_total = 280
    if final_pred < min_total:
        final_pred = min_total
        adjustments.append(f"Capped at min {min_total}")
    elif final_pred > max_total:
        final_pred = max_total
        adjustments.append(f"Capped at max {max_total}")
        
    # 2. Deviation from base projection (if available)
    base_proj = state.get('current_total_projection')
    if base_proj:
        max_dev = 25.0
        if abs(final_pred - base_proj) > max_dev:
            # Soft clamp
            direction = 1 if final_pred > base_proj else -1
            final_pred = base_proj + (direction * max_dev) + (final_pred - (base_proj + direction * max_dev)) * 0.5
            adjustments.append(f"Soft clamped deviation from base {base_proj:.1f}")

    return final_pred, adjustments
```

## Integración con el Sistema Principal

### 1. Flujo de Integración

```python
# En analysis/live_mode.py
from metamodel.meta_model_ai import MetaModeloAI

# Inicialización
metamodelo = MetaModeloAI(league_name="NBA")

# Uso en predicción
game_state = {
    'current_total_projection': context_adjusted_prediction,
    'halftime_total': halftime_total,
    'pace': pace_estimate,
    'team_stats': live_signals
}

final_prediction, metadata = metamodelo.predict(game_state)
```

### 2. Punto de Integración en el Pipeline

El MetaModeloAI se integra después del ajuste contextual pero antes de la predicción final:

1. Predicción base del modelo XGBoost
2. Ajustes contextuales (señales en vivo)
3. **Corrección del MetaModeloAI**
4. Predicción final

## Características Avanzadas

### 1. Manejo Robusto de Errores

- Detección de modelo no cargado
- Manejo de excepciones en predicción
- Valores por defecto seguros

### 2. Metadata Completa

El sistema proporciona metadata detallada:
- Predicción cruda del modelo
- Predicción final después de safety rails
- Features utilizadas
- Ajustes aplicados

### 3. Diseño Modular

- Separación clara de responsabilidades
- Métodos privados para operaciones internas
- Interfaz pública simple y limpia

## Configuración y Personalización

### 1. Parámetros Ajustables

- **Límites globales**: min_total, max_total
- **Desviación máxima**: max_dev
- **Umbral de juego cerrado**: close_game_threshold

### 2. Personalización por Liga

El sistema soporta diferentes configuraciones por liga:
- NBA: Límites ajustados para alto scoring
- WNBA: Límites ajustados para bajo scoring
- NBL: Configuración intermedia

## Rendimiento y Optimización

### 1. Eficiencia

- Carga perezosa del modelo
- Cálculo optimizado de features
- Uso eficiente de memoria

### 2. Escalabilidad

- Diseño para múltiples juegos simultáneos
- Reutilización de instancia del modelo
- Procesamiento paralelo posible

## Mantenimiento y Monitoreo

### 1. Registro de Eventos

- Estado de carga del modelo
- Errores de predicción
- Ajustes de safety rails aplicados

### 2. Métricas de Rendimiento

- Tiempo de respuesta de predicción
- Frecuencia de ajustes de safety rails
- Precisión de predicciones

## Mejoras Futuras

### 1. Extensiones Planeadas

- Incorporación de más features contextuales
- Sistema de safety rails adaptativo
- Integración con más fuentes de datos

### 2. Optimizaciones

- Caching de cálculos de features
- Paralelización de predicciones
- Reducción de latencia

## Conclusión

El MetaModeloAI representa un componente sofisticado del sistema Basketball MLG v2 que proporciona correcciones inteligentes a predicciones en tiempo real. Su diseño robusto, sistema de safety rails y capacidad de integración lo hacen ideal para producción, mientras que su modularidad facilita el mantenimiento y la extensión futura.

La combinación de CatBoost, features específicas del contexto del juego y límites de seguridad asegura predicciones precisas y realistas que mejoran significativamente la calidad del sistema general de predicción.