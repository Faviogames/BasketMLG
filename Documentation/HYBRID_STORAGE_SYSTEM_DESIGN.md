> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](auditoria/CONTEXTO_DOCUMENTATION.md) y [estado vigente](auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Diseño del Sistema de Almacenamiento Híbrido para Predicciones

## 🎯 Objetivo

Implementar un sistema de almacenamiento híbrido que combine archivos por temporada con índices mensuales para optimizar el aprendizaje del metamodelo y el acceso rápido a datos.

## 📁 Estructura de Archivos Propuesta

```
prediction_records/
├── daily/                    # Archivos individuales (como ahora)
│   ├── 2025-11-16_predictions.json
│   └── 40C8D1A5_Cavs_vs_Grizzlies_2025-11-16.json
├── seasonal/                  # 🆕 Archivos por temporada (para aprendizaje)
│   ├── NBA_2024-25_season.json
│   ├── NBA_2025-26_season.json
│   └── WNBA_2025_season.json
├── monthly_indices/           # 🆕 Índices mensuales (acceso rápido)
│   ├── NBA_2024-11_index.json
│   ├── NBA_2024-12_index.json
│   └── NBA_2025-11_index.json
└── analytics/                 # Reportes y análisis
    ├── monthly_reports/
    └── season_analysis/
```

## 🔧 Modificaciones a PredictionTracker

### 1. Nuevos Directorios en `ensure_storage_structure()`

```python
def ensure_storage_structure(self):
    """Crea la estructura de directorios necesaria"""
    os.makedirs(self.storage_dir, exist_ok=True)
    os.makedirs(f"{self.storage_dir}/daily", exist_ok=True)
    os.makedirs(f"{self.storage_dir}/seasonal", exist_ok=True)      # 🆕
    os.makedirs(f"{self.storage_dir}/monthly_indices", exist_ok=True) # 🆕
    os.makedirs(f"{self.storage_dir}/analytics", exist_ok=True)
    os.makedirs(f"{self.storage_dir}/analytics/monthly_reports", exist_ok=True)  # 🆕
    os.makedirs(f"{self.storage_dir}/analytics/season_analysis", exist_ok=True)  # 🆕
```

### 2. Nuevos Métodos en PredictionTracker

#### `get_season_file_path(league, season_year)`
```python
def get_season_file_path(self, league: str, season_year: str) -> str:
    """Genera la ruta del archivo de temporada"""
    return f"{self.storage_dir}/seasonal/{league}_{season_year}_season.json"
```

#### `get_monthly_index_path(league, year, month)`
```python
def get_monthly_index_path(self, league: str, year: str, month: str) -> str:
    """Genera la ruta del índice mensual"""
    return f"{self.storage_dir}/monthly_indices/{league}_{year}-{month}_index.json"
```

#### `update_seasonal_file(prediction_data)`
```python
def update_seasonal_file(self, prediction_data: Dict):
    """Actualiza el archivo de temporada con una nueva predicción"""
    league = prediction_data['matchup']['league']
    date_str = prediction_data['metadata']['date']
    season_year = self._get_season_year(date_str)
    
    season_file = self.get_season_file_path(league, season_year)
    
    # Cargar predicciones existentes
    if os.path.exists(season_file):
        with open(season_file, 'r', encoding='utf-8') as f:
            season_predictions = json.load(f)
    else:
        season_predictions = {
            "season_metadata": {
                "league": league,
                "season_year": season_year,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_predictions": 0
            },
            "predictions": []
        }
    
    # Agregar nueva predicción
    season_predictions["predictions"].append(prediction_data)
    season_predictions["season_metadata"]["total_predictions"] += 1
    season_predictions["season_metadata"]["last_updated"] = datetime.now().isoformat()
    
    # Guardar archivo actualizado
    with open(season_file, 'w', encoding='utf-8') as f:
        f.write(json_dumps_safe(season_predictions))
```

#### `update_monthly_index(prediction_data)`
```python
def update_monthly_index(self, prediction_data: Dict):
    """Actualiza el índice mensual con una referencia a la predicción"""
    league = prediction_data['matchup']['league']
    date_str = prediction_data['metadata']['date']
    year, month = date_str.split('-')
    
    index_file = self.get_monthly_index_path(league, year, month)
    
    # Cargar índice existente
    if os.path.exists(index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            monthly_index = json.load(f)
    else:
        monthly_index = {
            "index_metadata": {
                "league": league,
                "year": year,
                "month": month,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_predictions": 0
            },
            "prediction_references": []
        }
    
    # Agregar referencia (sin datos completos para mantener ligero)
    prediction_ref = {
        "prediction_id": prediction_data["prediction_id"],
        "date": date_str,
        "home_team": prediction_data["matchup"]["home_team"],
        "away_team": prediction_data["matchup"]["away_team"],
        "final_prediction": prediction_data["predictions"]["final_prediction"],
        "actual_total": prediction_data["result_tracking"]["actual_total"],
        "status": prediction_data["result_tracking"]["status"],
        "accuracy": prediction_data["result_tracking"]["prediction_accuracy"]
    }
    
    monthly_index["prediction_references"].append(prediction_ref)
    monthly_index["index_metadata"]["total_predictions"] += 1
    monthly_index["index_metadata"]["last_updated"] = datetime.now().isoformat()
    
    # Guardar índice actualizado
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(json_dumps_safe(monthly_index))
```

#### `get_season_predictions(league, season_year)`
```python
def get_season_predictions(self, league: str, season_year: str) -> List[Dict]:
    """Obtiene todas las predicciones de una temporada específica"""
    season_file = self.get_season_file_path(league, season_year)
    
    if os.path.exists(season_file):
        with open(season_file, 'r', encoding='utf-8') as f:
            season_data = json.load(f)
            return season_data.get("predictions", [])
    
    return []
```

#### `get_monthly_predictions(league, year, month)`
```python
def get_monthly_predictions(self, league: str, year: str, month: str) -> List[Dict]:
    """Obtiene referencias de predicciones de un mes específico"""
    index_file = self.get_monthly_index_path(league, year, month)
    
    if os.path.exists(index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            monthly_data = json.load(f)
            return monthly_data.get("prediction_references", [])
    
    return []
```

### 3. Método Auxiliar

#### `_get_season_year(date_str)`
```python
def _get_season_year(self, date_str: str) -> str:
    """Determina el año de temporada basado en la fecha"""
    year = int(date_str.split('-')[0])
    month = int(date_str.split('-')[1])
    
    # Las temporadas de NBA típicamente empiezan en octubre
    if month >= 10:
        return f"{year}-{year+1}"
    else:
        return f"{year-1}-{year}"
```

## 🔄 Modificación a `track_prediction()`

```python
def track_prediction(self, ...):
    # ... código existente ...
    
    # Guardar en archivo diario (como antes)
    daily_file = f"{self.storage_dir}/daily/{date_str}_predictions.json"
    daily_predictions = self._load_daily_predictions(date_str)
    daily_predictions.append(prediction_data)
    self._save_daily_predictions(date_str, daily_predictions)
    
    # Guardar archivo individual (como antes)
    individual_file = self._get_individual_file_path(prediction_id, home_team, away_team, date_str)
    with open(individual_file, 'w', encoding='utf-8') as f:
        f.write(json_dumps_safe(prediction_data))
    
    # 🆕 Actualizar archivo de temporada
    self.update_seasonal_file(prediction_data)
    
    # 🆕 Actualizar índice mensual
    self.update_monthly_index(prediction_data)
    
    print(f"Predicción registrada: {prediction_id}")
    print(f"  Archivo diario: {daily_file}")
    print(f"  Archivo individual: {individual_file}")
    print(f"  Temporada: {self.get_season_file_path(league, self._get_season_year(date_str))}")
    print(f"  Índice mensual: {self.get_monthly_index_path(league, date_str.split('-')[0], date_str.split('-')[1])}")
    
    return prediction_id
```

## 🎯 Formato Plano para Exportación

### Formato Individual (como actualmente)
```json
{
  "prediction_id": "40C8D1A5",
  "metadata": {
    "timestamp": "2025-11-16T20:15:30",
    "date": "2025-11-16",
    "version": "1.1"
  },
  "matchup": {
    "home_team": "Cleveland Cavaliers",
    "away_team": "Memphis Grizzlies",
    "league": "NBA"
  },
  "odds_information": {
    "totals_line": 232.5,
    "bookmaker": "API Auto"
  },
  "predictions": {
    "base_model_prediction": 219.7,
    "context_adjusted_prediction": 208.7,
    "final_prediction": 208.7,
    "adjustments": {
      "blowout_adjustment": -11.0,
      "meta_model_offset": 0.0
    }
  },
  "probabilities": [
    {"line": 228.8, "over_prob": 45.2, "under_prob": 54.8},
    {"line": 230.5, "over_prob": 50.0, "under_prob": 50.0},
    {"line": 232.5, "over_prob": 54.8, "under_prob": 45.2}
  ],
  "analysis": {
    "confidence_score": 0.75,
    "recommended_bet": "UNDER",
    "edge_percentage": 23.7
  },
  "game_context": {
    "quarter_scores": {"home": [28, 26, 22, 32], "away": [32, 29, 23, 16]},
    "prediction_timing": "HT"
  },
  "result_tracking": {
    "status": "Completado",
    "actual_total": 208,
    "actual_result": "UNDER",
    "prediction_accuracy": "Acierto"
  }
}
```

## 🚀 Integración con el Metamodelo

### Modificación a `MetaModeloInteligente`

```python
def cargar_historico_temporada(self, league: str = "NBA", season_year: str = None) -> List[Dict]:
    """Carga todas las predicciones de una temporada para aprendizaje"""
    if not season_year:
        season_year = self._get_current_season_year()
    
    tracker = PredictionTracker()
    season_predictions = tracker.get_season_predictions(league, season_year)
    
    # Filtrar solo predicciones completadas
    completed_predictions = [
        p for p in season_predictions 
        if p['result_tracking']['status'] == 'completed'
    ]
    
    print(f"Cargadas {len(completed_predictions)} predicciones completadas de la temporada {season_year}")
    return completed_predictions
```

## 📊 Herramientas de Análisis

### 1. Análisis Mensual
```python
def generate_monthly_report(league: str, year: str, month: str):
    """Genera reporte mensual de rendimiento"""
    tracker = PredictionTracker()
    monthly_refs = tracker.get_monthly_predictions(league, year, month)
    
    # Calcular métricas del mes
    # Generar visualizaciones
    # Guardar reporte en analytics/monthly_reports/
```

### 2. Análisis de Temporada
```python
def generate_season_analysis(league: str, season_year: str):
    """Genera análisis completo de temporada"""
    tracker = PredictionTracker()
    season_predictions = tracker.get_season_predictions(league, season_year)
    
    # Análisis de tendencias
    # Evolución del rendimiento
    # Patrones de ajuste del metamodelo
    # Guardar análisis en analytics/season_analysis/
```

## 🎯 Beneficios del Sistema Híbrido

1. **Aprendizaje Óptimo**: El metamodelo accede al archivo completo de temporada
2. **Acceso Rápido**: Índices mensuales para consultas específicas
3. **Escalabilidad**: Maneja fácilmente 1000+ predicciones por temporada
4. **Flexibilidad**: Permite análisis por mes, trimestre o temporada completa
5. **Backup Natural**: Múltiples niveles de almacenamiento
6. **Rendimiento**: Archivos individuales para acceso inmediato

## 🔄 Flujo de Trabajo

1. **Predicción Nueva**: Se guarda en 3 lugares (daily, individual, seasonal + index)
2. **Actualización de Resultado**: Se actualiza en todos los archivos
3. **Aprendizaje del Metamodelo**: Lee archivo de temporada completo
4. **Análisis Rápido**: Usa índices mensuales
5. **Reportes**: Genera desde datos agregados

## 📈 Implementación

La implementación se realizará en los siguientes pasos:
1. Modificar `PredictionTracker` para agregar nuevos métodos
2. Actualizar `track_prediction()` para usar sistema híbrido
3. Modificar `update_prediction_result()` para actualizar todos los archivos
4. Integrar con metamodelo para aprendizaje
5. Crear herramientas de análisis
6. Documentar y probar el sistema completo