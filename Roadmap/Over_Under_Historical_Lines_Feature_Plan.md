> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_ROADMAP.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Over/Under Historical Lines Feature Plan

## Resumen Ejecutivo

**Objetivo:** Implementar features basadas en líneas históricas de apuestas Over/Under para mejorar la precisión del modelo predictivo de totales de basketball, potencialmente reduciendo el MAE y aumentando el acierto en predicciones.

**Estado:** Plan diseñado y aprobado para futura implementación.

## Análisis del Sistema Actual

### Arquitectura Existente
- **Modelo:** XGBoost + Ridge meta-modelo para predicción de `final_total_score`
- **Features:** 120+ incluyendo momentum, estadísticas avanzadas, señales live, H2H
- **Integración:** API de cuotas ya implementada para líneas live
- **Tracking:** Sistema de seguimiento de predicciones con líneas históricas

### Evaluación de Beneficios
**Potenciales mejoras:**
- ✅ Incorpora sabiduría colectiva del mercado de apuestas
- ✅ Captura análisis de expertos y percepción pública
- ✅ Identifica patrones de precisión del mercado
- ✅ Proporciona contexto adicional no capturado por estadísticas raw

**Riesgos considerados:**
- Correlación con totales reales (ya usados en el modelo)
- Disponibilidad de datos históricos de líneas
- Posible introducción de ruido si las líneas no son predictivas

**Conclusión:** Sí, podría ayudar significativamente a reducir MAE aprendiendo de la eficiencia del mercado.

## Plan de Implementación Detallado

### Fase 1: Diseño de Features (✅ Completado)

**Features Propuestas:**
- `avg_historical_line`: Promedio de líneas históricas por equipo
- `line_vs_actual_diff`: Diferencia entre línea histórica y total real
- `market_bias_indicator`: Tendencia del mercado a over/under para equipos específicos
- `line_volatility`: Variabilidad de líneas históricas
- `line_accuracy_trend`: Tendencia de precisión del mercado
- `team_line_strength`: Fortaleza relativa basada en líneas históricas
- `league_line_trends`: Tendencias de líneas por liga

### Fase 2: Nuevo Módulo `core/ou_line_features.py`

**Arquitectura del Módulo:**
```python
# core/ou_line_features.py
def calculate_historical_line_features(team_history_df, league_historical_lines):
    """Calcula features basadas en líneas históricas de apuestas"""

def get_team_line_history(team_name, historical_lines_db):
    """Obtiene historial de líneas para un equipo"""

def calculate_market_bias_indicators(team_history_df):
    """Calcula indicadores de sesgo del mercado"""

def calculate_line_volatility_metrics(team_history_df):
    """Métricas de volatilidad de líneas"""

def calculate_line_accuracy_trends(team_history_df):
    """Tendencias de precisión del mercado"""
```

**Beneficios del módulo separado:**
- Mantenibilidad: `core/features.py` ya tiene 3184 líneas
- Modularidad: Separación clara de responsabilidades
- Testing: Más fácil de testear unitariamente
- Reutilización: Features específicas de líneas en un solo lugar

### Fase 3: Recolección de Datos

**Sistema de Almacenamiento:**
- Extender `storage/prediction_tracker.py` para almacenar líneas históricas
- Base de datos dedicada para líneas históricas por partido
- Integración con API de cuotas para datos históricos (si disponible)

**Estructura de Datos:**
```json
{
  "game_id": "unique_game_identifier",
  "home_team": "Team A",
  "away_team": "Team B",
  "date": "2024-01-15",
  "bookmaker": "DraftKings",
  "totals_line": 215.5,
  "over_odds": 1.95,
  "under_odds": 1.87,
  "actual_total": 218.0
}
```

### Fase 4: Integración con Pipeline de Entrenamiento

**Modificaciones Requeridas:**
1. **data_processing.py:** Merge de datos de líneas con datos de partidos
2. **features.py:** Importar y llamar funciones del nuevo módulo
3. **training.py:** Incluir nuevas features en el conjunto de entrenamiento
4. **config.py:** Actualizar listas de features permitidas

**Integración con Sistema EMA:**
- Aplicar cálculos EMA a features de líneas
- Rangos temporales: corto (5 juegos), medio (10 juegos), largo (20 juegos)
- Features EMA: `ema_line_accuracy_short_5`, `ema_market_bias_medium_10`, etc.

### Fase 5: Testing y Validación

**Estrategia de Testing:**
- **A/B Testing:** Entrenar modelo con/sin features de líneas
- **Métricas:** Comparar MAE, R², precisión de over/under
- **Validación Cruzada:** Asegurar que mejoras no son overfitting
- **Análisis de Features:** Importancia de nuevas features vs existentes

**Métricas de Éxito:**
- Reducción de MAE ≥ 2%
- Mejora en precisión de predicciones ≥ 3%
- Features de líneas en top 20 de importancia

## Timeline Estimado

### Fase 1: Diseño y Planificación (✅ Completado)
- Diseño de features
- Análisis de arquitectura
- Plan de integración

### Fase 2: Implementación Core (2-3 semanas)
- Crear módulo `core/ou_line_features.py`
- Implementar funciones de feature engineering
- Sistema de recolección de datos históricos

### Fase 3: Integración (1 semana)
- Modificar pipeline de entrenamiento
- Actualizar data processing
- Testing de integración

### Fase 4: Validación y Optimización (1-2 semanas)
- A/B testing
- Análisis de resultados
- Optimización de features

## Riesgos y Mitigaciones

### Riesgos Técnicos
- **Disponibilidad de datos:** Mitigación - usar datos disponibles inicialmente, expandir gradualmente
- **Calidad de datos:** Mitigación - validación y limpieza de datos de líneas
- **Overfitting:** Mitigación - validación cruzada y regularización

### Riesgos de Negocio
- **Complejidad añadida:** Mitigación - módulo separado mantiene simplicidad
- **Mantenimiento:** Mitigación - documentación clara y tests automatizados
- **Performance:** Mitigación - optimización de cálculos y caching

## Conclusión

Esta feature tiene alto potencial para mejorar el rendimiento del modelo al incorporar información del mercado de apuestas. La implementación modular asegura mantenibilidad mientras se integra perfectamente con la arquitectura existente.

**Estado:** Plan aprobado y guardado para futura implementación cuando se considere prioritario.