> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](auditoria/CONTEXTO_DOCUMENTATION.md) y [estado vigente](auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Documentación del Sistema de Tracking de Predicciones

## Resumen Ejecutivo

El PredictionTracker es un sistema integral de almacenamiento y análisis de predicciones que implementa un enfoque híbrido para gestionar datos de predicciones de baloncesto. Proporciona funcionalidades completas de tracking, análisis de rendimiento y generación de reportes para el sistema Basketball MLG v2.

## Arquitectura del Sistema

### Componentes Principales

1. **PredictionTracker Class**: Clase principal que gestiona todo el ciclo de vida de predicciones
2. **Hybrid Storage System**: Sistema de almacenamiento multi-formato (diario, mensual, por temporada)
3. **Performance Analytics**: Motor de análisis de métricas de rendimiento
4. **Report Generator**: Sistema de generación de reportes automáticos

### Estructura de Almacenamiento

```
prediction_records/
├── daily/                    # Predicciones diarias
│   └── YYYY-MM-DD_predictions.json
├── monthly_indices/           # Índices mensuales
│   └── NBA_YYYY-MM_index.json
├── seasonal/                 # Datos de temporada
│   └── NBA_YYYY-YYYY_season.json
└── exports/                  # Reportes exportados
```

## Características Detalladas

### 1. Inicialización y Configuración

```python
class PredictionTracker:
    def __init__(self, storage_dir: str = "prediction_records"):
        self.storage_dir = storage_dir
        self.ensure_storage_structure()
        self.predicciones_history = []
        self.ajustes_aplicados = []
        self.ultima_actualizacion = datetime.now()
```

#### Configuración de Almacenamiento

- **storage_dir**: Directorio base para almacenamiento
- **Estructura automática**: Creación automática de subdirectorios
- **Estado en memoria**: Historial y caché de operaciones

### 2. Sistema de Almacenamiento Híbrido

#### ensure_storage_structure()
- **Propósito**: Crear estructura de directorios necesaria
- **Directorios creados**:
  - `daily/`: Para predicciones diarias
  - `monthly_indices/`: Para índices de referencia rápida
  - `seasonal/`: Para datos completos de temporada
  - `exports/`: Para reportes generados

```python
def ensure_storage_structure(self):
    """Asegura que la estructura de directorios exista."""
    os.makedirs(f"{self.storage_dir}/daily", exist_ok=True)
    os.makedirs(f"{self.storage_dir}/monthly_indices", exist_ok=True)
    os.makedirs(f"{self.storage_dir}/seasonal", exist_ok=True)
    os.makedirs(f"{self.storage_dir}/exports", exist_ok=True)
```

### 3. Registro de Predicciones

#### track_prediction()
- **Propósito**: Registrar nueva predicción con metadatos completos
- **Características**:
  - Generación automática de ID único
  - Validación de datos de entrada
  - Almacenamiento en múltiples formatos
  - Timestamp automático

```python
def track_prediction(self, home_team: str, away_team: str, **kwargs) -> str:
    """
    Registra una nueva predicción y retorna el ID generado.
    
    Args:
        home_team: Equipo local
        away_team: Equipo visitante
        **kwargs: Parámetros adicionales de la predicción
        
    Returns:
        str: ID único de la predicción
    """
    # Generar ID único
    prediction_id = self._generate_prediction_id(home_team, away_team)
    
    # Crear estructura de datos
    prediction_data = {
        'prediction_id': prediction_id,
        'timestamp': datetime.now().isoformat(),
        'home_team': home_team,
        'away_team': away_team,
        'league': kwargs.get('league', 'NBA'),
        'status': 'pending',
        **kwargs
    }
    
    # Almacenar en múltiples formatos
    self._save_daily_prediction(prediction_data)
    self._update_seasonal_file(prediction_data)
    self._update_monthly_index(prediction_data)
    
    # Actualizar historial en memoria
    self.predicciones_history.append(prediction_data)
    
    return prediction_id
```

#### _generate_prediction_id()
- **Propósito**: Generar ID único basado en equipos y timestamp
- **Formato**: `{HASH}_{HOME_TEAM}_vs_{AWAY_TEAM}_{DATE}`

```python
def _generate_prediction_id(self, home_team: str, away_team: str) -> str:
    """Genera un ID único para la predicción."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    team_hash = hash(f"{home_team}_{away_team}") % 10000
    return f"{team_hash:04d}_{home_team}_vs_{away_team}_{timestamp}"
```

### 4. Actualización de Resultados

#### update_prediction_result()
- **Propósito**: Actualizar resultado de predicción y calcular métricas
- **Características**:
  - Actualización atómica de registros
  - Cálculo automático de métricas
  - Actualización de todos los formatos de almacenamiento

```python
def update_prediction_result(self, prediction_id: str, actual_total: float, result_notes: str = None) -> bool:
    """
    Actualiza el resultado de una predicción existente.
    
    Args:
        prediction_id: ID de la predicción a actualizar
        actual_total: Resultado real del partido
        result_notes: Notas adicionales sobre el resultado
        
    Returns:
        bool: True si se actualizó correctamente, False si no se encontró
    """
    # Buscar predicción en archivos diarios
    daily_file = self._get_daily_file_for_prediction(prediction_id)
    if not daily_file:
        return False
    
    # Cargar y actualizar
    predictions = self._load_daily_predictions(daily_file)
    for pred in predictions:
        if pred['prediction_id'] == prediction_id:
            pred['status'] = 'completed'
            pred['actual_total'] = actual_total
            pred['result_notes'] = result_notes
            pred['completed_timestamp'] = datetime.now().isoformat()
            
            # Calcular métricas
            if 'predicted_total' in pred:
                pred['prediction_error'] = abs(pred['predicted_total'] - actual_total)
                pred['prediction_direction'] = 'over' if actual_total > pred['predicted_total'] else 'under'
            
            break
    
    # Guardar actualizaciones
    self._save_daily_predictions(daily_file, predictions)
    self._update_seasonal_with_result(prediction_id, actual_total, result_notes)
    
    return True
```

### 5. Sistema de Almacenamiento Diario

#### _save_daily_prediction()
- **Propósito**: Guardar predicción en archivo diario
- **Formato**: `YYYY-MM-DD_predictions.json`

```python
def _save_daily_prediction(self, prediction_data: Dict):
    """Guarda predicción en archivo diario."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{self.storage_dir}/daily/{date_str}_predictions.json"
    
    # Cargar predicciones existentes
    existing = []
    if os.path.exists(filename):
        existing = self._load_daily_predictions(filename)
    
    # Agregar nueva predicción
    existing.append(prediction_data)
    
    # Guardar
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
```

#### _load_daily_predictions()
- **Propósito**: Cargar predicciones desde archivo diario
- **Manejo de errores**: Robusto contra archivos corruptos

```python
def _load_daily_predictions(self, filename: str) -> List[Dict]:
    """Carga predicciones desde archivo diario."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
```

### 6. Sistema de Almacenamiento por Temporada

#### update_seasonal_file()
- **Propósito**: Mantener archivo consolidado por temporada
- **Formato**: `NBA_YYYY-YYYY_season.json`

```python
def update_seasonal_file(self, prediction_data: Dict):
    """Actualiza archivo de temporada con nueva predicción."""
    league = prediction_data.get('league', 'NBA')
    season_year = self._get_season_year(prediction_data['timestamp'])
    filename = f"{self.storage_dir}/seasonal/{league}_{season_year}_season.json"
    
    # Cargar datos existentes
    seasonal_data = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            seasonal_data = json.load(f)
    
    # Agregar nueva predicción
    seasonal_data.append(prediction_data)
    
    # Guardar
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(seasonal_data, f, indent=2, ensure_ascii=False)
```

### 7. Sistema de Índices Mensuales

#### update_monthly_index()
- **Propósito**: Mantener índice rápido de referencias mensuales
- **Formato**: `NBA_YYYY-MM_index.json`

```python
def update_monthly_index(self, prediction_data: Dict):
    """Actualiza índice mensual con referencia a predicción."""
    league = prediction_data.get('league', 'NBA')
    timestamp = prediction_data['timestamp']
    date_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    month_str = date_obj.strftime("%Y-%m")
    
    filename = f"{self.storage_dir}/monthly_indices/{league}_{month_str}_index.json"
    
    # Cargar índice existente
    index_data = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    
    # Agregar referencia
    index_entry = {
        'prediction_id': prediction_data['prediction_id'],
        'date': timestamp,
        'home_team': prediction_data['home_team'],
        'away_team': prediction_data['away_team'],
        'status': prediction_data['status']
    }
    index_data.append(index_entry)
    
    # Guardar
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
```

### 8. Análisis de Rendimiento

#### calculate_performance_metrics()
- **Propósito**: Calcular métricas completas de rendimiento
- **Métricas calculadas**:
  - Precisión general
  - Error promedio absoluto (MAE)
  - Precisión por dirección (over/under)
  - Distribución de errores

```python
def calculate_performance_metrics(self) -> Dict[str, Any]:
    """
    Calcula métricas de rendimiento completas.
    
    Returns:
        Dict: Métricas de rendimiento calculadas
    """
    all_predictions = self.get_all_predictions()
    completed_predictions = [p for p in all_predictions if p['status'] == 'completed']
    
    if not completed_predictions:
        return {'error': 'No completed predictions found'}
    
    # Métricas básicas
    total_predictions = len(completed_predictions)
    errors = [p['prediction_error'] for p in completed_predictions if 'prediction_error' in p]
    
    metrics = {
        'total_predictions': total_predictions,
        'mean_absolute_error': sum(errors) / len(errors) if errors else 0,
        'median_absolute_error': sorted(errors)[len(errors)//2] if errors else 0,
        'max_error': max(errors) if errors else 0,
        'min_error': min(errors) if errors else 0,
    }
    
    # Precisión por dirección
    over_correct = sum(1 for p in completed_predictions 
                      if p.get('prediction_direction') == 'over' and p.get('actual_total', 0) > p.get('predicted_total', 0))
    under_correct = sum(1 for p in completed_predictions 
                       if p.get('prediction_direction') == 'under' and p.get('actual_total', 0) <= p.get('predicted_total', 0))
    
    metrics.update({
        'over_accuracy': over_correct / total_predictions,
        'under_accuracy': under_correct / total_predictions,
        'overall_accuracy': (over_correct + under_correct) / total_predictions
    })
    
    return metrics
```

### 9. Generación de Reportes

#### generate_monthly_report()
- **Propósito**: Generar reporte mensual de rendimiento
- **Contenido**:
  - Resumen de predicciones del mes
  - Análisis de precisión por categoría
  - Identificación de patrones

```python
def generate_monthly_report(self, league: str, year: str, month: str) -> str:
    """
    Genera reporte mensual de rendimiento.
    
    Args:
        league: Liga (NBA, WNBA, etc.)
        year: Año del reporte
        month: Mes del reporte
        
    Returns:
        str: Ruta del archivo de reporte generado
    """
    # Obtener predicciones del mes
    month_predictions = self.get_month_predictions(league, year, month)
    
    # Calcular métricas
    metrics = self._calculate_month_metrics(month_predictions)
    
    # Generar reporte
    report = {
        'report_type': 'monthly',
        'league': league,
        'period': f"{year}-{month}",
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_predictions': len(month_predictions),
            'completed_predictions': len([p for p in month_predictions if p['status'] == 'completed'])
        },
        'metrics': metrics,
        'predictions': month_predictions
    }
    
    # Guardar reporte
    filename = f"{self.storage_dir}/exports/{league}_{year}_{month}_report.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return filename
```

#### generate_season_analysis()
- **Propósito**: Generar análisis completo de temporada
- **Características**:
  - Tendencias de rendimiento
  - Comparación vs temporadas anteriores
  - Insights estratégicos

```python
def generate_season_analysis(self, league: str, season_year: str) -> str:
    """
    Genera análisis completo de temporada.
    
    Args:
        league: Liga (NBA, WNBA, etc.)
        season_year: Año de temporada (ej: "2024-2025")
        
    Returns:
        str: Ruta del archivo de análisis generado
    """
    # Obtener predicciones de temporada
    season_predictions = self.get_season_predictions(league, season_year)
    
    # Análisis avanzado
    analysis = {
        'analysis_type': 'season',
        'league': league,
        'season': season_year,
        'generated_at': datetime.now().isoformat(),
        'performance_trends': self._analyze_performance_trends(season_predictions),
        'team_performance': self._analyze_team_performance(season_predictions),
        'prediction_patterns': self._analyze_prediction_patterns(season_predictions),
        'recommendations': self._generate_recommendations(season_predictions)
    }
    
    # Guardar análisis
    filename = f"{self.storage_dir}/exports/{league}_{season_year}_analysis.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    return filename
```

### 10. Consultas y Recuperación

#### get_all_predictions()
- **Propósito**: Obtener todas las predicciones del sistema
- **Optimización**: Carga eficiente desde archivos de temporada

```python
def get_all_predictions(self) -> List[Dict]:
    """Obtiene todas las predicciones del sistema."""
    all_predictions = []
    
    # Cargar desde archivos de temporada (más eficiente)
    seasonal_dir = f"{self.storage_dir}/seasonal"
    if os.path.exists(seasonal_dir):
        for filename in os.listdir(seasonal_dir):
            if filename.endswith('.json'):
                with open(os.path.join(seasonal_dir, filename), 'r', encoding='utf-8') as f:
                    season_data = json.load(f)
                    all_predictions.extend(season_data)
    
    return all_predictions
```

#### get_pending_predictions()
- **Propósito**: Obtener predicciones pendientes de resultado
- **Uso**: Facilitar registro manual de resultados

```python
def get_pending_predictions(self) -> List[Dict]:
    """Obtiene predicciones pendientes de resultado."""
    all_predictions = self.get_all_predictions()
    return [p for p in all_predictions if p['status'] == 'pending']
```

## Integración con el Sistema Principal

### 1. Flujo de Integración

```python
# En analysis/live_mode.py
from metamodel.prediction_tracker import PredictionTracker

# Inicialización
tracker = PredictionTracker()

# Registro de predicción
prediction_id = tracker.track_prediction(
    home_team="Lakers",
    away_team="Celtics",
    predicted_total=220.5,
    confidence=0.75,
    recommended_side="OVER"
)

# Actualización de resultado (cuando termina el partido)
tracker.update_prediction_result(
    prediction_id=prediction_id,
    actual_total=225.0,
    result_notes="Good prediction, game was high-paced"
)
```

### 2. Puntos de Integración

1. **Registro de Predicción**: Después de generar predicción final
2. **Actualización de Resultado**: Cuando termina el partido
3. **Análisis de Rendimiento**: Para evaluación continua
4. **Generación de Reportes**: Para análisis estratégico

## Características Avanzadas

### 1. Manejo Robusto de Errores

- Validación de datos de entrada
- Manejo de archivos corruptos
- Recuperación automática de datos
- Logging detallado de errores

### 2. Optimización de Rendimiento

- Caching inteligente de datos
- Índices para acceso rápido
- Carga diferida de datos
- Compresión de archivos históricos

### 3. Escalabilidad

- Diseño para millones de predicciones
- Particionamiento automático de datos
- Consultas eficientes por período
- Archivos de tamaño manejable

## Configuración y Personalización

### 1. Parámetros Configurables

- **storage_dir**: Directorio de almacenamiento
- **retention_days**: Días de retención de archivos
- **compression_enabled**: Habilitar compresión de archivos antiguos
- **backup_enabled**: Habilitar sistema de backup

### 2. Personalización por Liga

- Métricas específicas por liga
- Formatos de reporte personalizados
- Análisis adaptados a características de liga

## Mantenimiento y Monitoreo

### 1. Mantenimiento Automático

- Limpieza de archivos antiguos
- Compresión de datos históricos
- Reindexación de datos
- Verificación de integridad

### 2. Monitoreo de Sistema

- Métricas de uso del sistema
- Alertas de errores críticos
- Monitorización de espacio en disco
- Análisis de rendimiento

## Mejoras Futuras

### 1. Extensiones Planeadas

- Integración con bases de datos SQL
- Sistema de notificaciones en tiempo real
- Dashboard web de análisis
- API REST para acceso externo

### 2. Optimizaciones

- Caching distribuido
- Procesamiento paralelo
- Algoritmos de compresión avanzados
- Consultas optimizadas con índices

## Conclusión

El PredictionTracker es un sistema completo y robusto que proporciona capacidades integrales de tracking, almacenamiento y análisis de predicciones. Su arquitectura híbrida, diseño modular y características avanzadas lo hacen ideal para producción a escala, mientras que su sistema de reportes y análisis proporciona insights valiosos para la mejora continua del sistema de predicción.

La combinación de almacenamiento eficiente, análisis de rendimiento detallado y generación de reportes automáticos asegura que el sistema no solo registre datos, sino que también genere conocimiento accionable para optimizar las predicciones futuras.