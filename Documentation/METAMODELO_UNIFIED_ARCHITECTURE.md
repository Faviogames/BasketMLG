> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](auditoria/CONTEXTO_DOCUMENTATION.md) y [estado vigente](auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Arquitectura Unificada del MetaModeloInteligente

## Overview

El MetaModeloInteligente ahora integra funcionalidades de tracking y aprendizaje en una sola clase unificada, simplificando la arquitectura y mejorando la eficiencia del sistema.

## Cambios Arquitectónicos

### 1. Integración de PredictionTracker

**Antes:**
- PredictionTracker en `storage/prediction_tracker.py`
- Imports dispersos en múltiples archivos
- Comunicación separada entre tracking y metamodelo

**Ahora:**
- PredictionTracker integrado en `metamodel/prediction_tracker.py`
- Métodos unificados en MetaModeloInteligente
- Comunicación directa y eficiente

### 2. Métodos Unificados

Se han agregado 4 métodos principales que combinan tracking + aprendizaje:

#### `procesar_prediccion_con_tracking()`
```python
def procesar_prediccion_con_tracking(
    self,
    home_team: str,
    away_team: str,
    prediccion_base: float,
    contexto_partido: Dict[str, Any],
    odds_line: Optional[float] = None,
    predictions_list: List[Dict] = None,
    confidence: float = 0.5,
    recommended_side: str = "OVER",
    edge_magnitude: float = 0.0,
    senales_live: Dict[str, Any] = None,
    q_scores: Dict[str, int] = None,
    quarter_stage: str = None,
    league: str = "NBA",
    prediction_timing: str = "pre_game"
) -> Tuple[str, float, Dict[str, Any]]:
```
- Procesa predicción con metamodelo
- Registra automáticamente con tracking
- Retorna: (prediction_id, prediccion_ajustada, info_ajuste)

#### `registrar_resultado_y_aprender()`
```python
def registrar_resultado_y_aprender(
    self,
    prediction_id: str,
    resultado_real: float,
    info_prediccion: Dict = None
) -> Dict[str, Any]:
```
- Actualiza resultado en tracking
- Aplica aprendizaje automático
- Combina ambos procesos en una sola llamada

#### `obtener_predicciones_pendientes_aprendizaje()`
```python
def obtener_predicciones_pendientes_aprendizaje(self) -> List[Dict]:
```
- Obtiene predicciones pendientes de resultado
- Muestra información formateada
- Facilita el registro manual

#### `generar_reporte_integrado()`
```python
def generar_reporte_integrado(self, output_file: str = None) -> str:
```
- Genera reporte combinado de tracking + aprendizaje
- Incluye métricas de rendimiento
- Guarda en formato JSON estructurado

### 3. Métodos Delegados

Se mantienen todos los métodos del PredictionTracker como delegados:

```python
def track_prediction(self, home_team: str, away_team: str, **kwargs) -> str:
def update_prediction_result(self, prediction_id: str, actual_total: float, result_notes: str = None) -> bool:
def get_all_predictions(self) -> List[Dict]:
def get_pending_predictions(self) -> List[Dict]:
def calculate_performance_metrics(self) -> Dict[str, Any]:
def update_seasonal_file(self, prediction_data: Dict):
def generate_monthly_report(self, league: str, year: str, month: str) -> str:
def generate_season_analysis(self, league: str, season_year: str) -> str:
```

## Flujo de Trabajo Unificado

### 1. Procesamiento de Predicciones

```python
# Antes (2 pasos)
prediccion_ajustada, info = metamodelo.procesar_prediccion(...)
prediction_id = tracker.track_prediction(...)

# Ahora (1 paso)
prediction_id, prediccion_ajustada, info = metamodelo.procesar_prediccion_con_tracking(...)
```

### 2. Registro de Resultados

```python
# Antes (2 pasos)
tracker.update_prediction_result(prediction_id, resultado_real)
metamodelo.registrar_resultado_manual(prediction_id, resultado_real)

# Ahora (1 paso)
resultado = metamodelo.registrar_resultado_y_aprender(prediction_id, resultado_real)
```

### 3. Análisis Integrado

```python
# Antes (múltiples fuentes)
tracking_metrics = tracker.calculate_performance_metrics()
metamodelo_stats = metamodelo.obtener_estadisticas()

# Ahora (1 llamada)
reporte_file = metamodelo.generar_reporte_integrado()
```

## Beneficios de la Arquitectura Unificada

### 1. Simplificación del Código
- Menos líneas de código para operaciones comunes
- API más intuitiva y coherente
- Reducción de complejidad para usuarios

### 2. Mejora de Rendimiento
- Menos llamadas a métodos
- Compartición de estado interno
- Reducción de overhead de comunicación

### 3. Mejor Mantenimiento
- Código centralizado en una clase
- Menos dependencias entre módulos
- Más fácil de extender y modificar

### 4. Consistencia de Datos
- Estado compartido entre tracking y aprendizaje
- Menos riesgo de inconsistencias
- Integridad de datos garantizada

## Actualización de Imports

### Cambios Realizados

**Archivos actualizados:**
- `metamodel/meta_model.py`
- `core/live_api.py`
- `analysis/live_mode.py`
- `demo_hybrid_storage_simple.py`
- `demo_hybrid_storage.py`
- `tools/hybrid_analysis_tools.py`
- Múltiples archivos de tests

**Cambio de import:**
```python
# Antes
from storage.prediction_tracker import PredictionTracker

# Ahora
from metamodel.prediction_tracker import PredictionTracker
```

## Sistema de Almacenamiento Híbrido

La arquitectura unificada mantiene el sistema de almacenamiento híbrido:

### 1. Archivos Diarios
- `prediction_records/daily/YYYY-MM-DD_predictions.json`
- Predicciones del día
- Acceso rápido para operaciones diarias

### 2. Archivos de Temporada
- `prediction_records/seasonal/NBA_YYYY-YYYY_season.json`
- Todas las predicciones de la temporada
- Optimizado para aprendizaje

### 3. Índices Mensuales
- `prediction_records/monthly_indices/NBA_YYYY-MM_index.json`
- Referencias rápidas a predicciones
- Acceso eficiente por fechas

## Ejemplo de Uso Completo

```python
from metamodel.meta_model import MetaModeloInteligente

# 1. Inicializar metamodelo unificado
metamodelo = MetaModeloInteligente(league_name="NBA")

# 2. Procesar predicción con tracking automático
contexto = {
    "minutos_restantes": 24,
    "puntos_actuales": 91,
    "quarter": 2
}

prediction_id, prediccion_ajustada, info = metamodelo.procesar_prediccion_con_tracking(
    home_team="Cleveland Cavaliers",
    away_team="Memphis Grizzlies",
    prediccion_base=206.0,
    contexto_partido=contexto,
    odds_line=232.5,
    confidence=0.65,
    recommended_side="UNDER"
)

# 3. Cuando el partido termina, registrar resultado y aprender
resultado = metamodelo.registrar_resultado_y_aprender(
    prediction_id=prediction_id,
    resultado_real=225.0
)

# 4. Generar reporte integrado
reporte_file = metamodelo.generar_reporte_integrado()
print(f"Reporte generado en: {reporte_file}")

# 5. Ver predicciones pendientes
pendientes = metamodelo.obtener_predicciones_pendientes_aprendizaje()
print(f"Predicciones pendientes: {len(pendientes)}")
```

## Consideraciones Técnicas

### 1. Compatibilidad hacia Atrás
- Todos los métodos originales siguen disponibles
- El código existente continúa funcionando
- Migración gradual posible

### 2. Manejo de Errores
- Validación robusta de datos de entrada
- Manejo de formatos variables en predicciones
- Mensajes de error claros y descriptivos

### 3. Performance
- Lazy loading del PredictionTracker
- Caching de instancias para mejorar rendimiento
- Optimización de operaciones frecuentes

## Próximos Pasos

1. **Migración de Scripts Existente**
   - Actualizar todos los scripts de demostración
   - Migrar código de producción a nueva API
   - Documentar mejores prácticas

2. **Extensión de Funcionalidades**
   - Agregar más métodos unificados
   - Implementar análisis avanzados
   - Mejorar visualización de datos

3. **Optimización de Rendimiento**
   - Perfilado de operaciones críticas
   - Optimización de acceso a datos
   - Implementar caching inteligente

## Conclusión

La arquitectura unificada del MetaModeloInteligente representa un avance significativo en la simplificación y eficiencia del sistema. Al integrar tracking y aprendizaje en una sola clase, se reduce la complejidad, se mejora el rendimiento y se facilita el mantenimiento del código.

Esta nueva arquitectura está lista para producción y proporciona una base sólida para futuras mejoras y extensiones del sistema.