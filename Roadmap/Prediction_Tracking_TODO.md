> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_ROADMAP.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# 📋 TODO: SISTEMA DE SEGUIMIENTO DE PREDICCIONES

## 🎯 OBJETIVO
Implementar un sistema completo de tracking que registre predicciones automáticamente y permita análisis posterior de aciertos/fallos.

## 📝 PLAN DE IMPLEMENTACIÓN

### ETAPA 1: BACKEND BÁSICO (FUNDACIÓN)
- [ ] **TASK 1.1**: Crear estructura de directorios
  - Crear `prediction_records/`
  - Crear subdirectorios: `daily_predictions/`, `export_logs/`
  - Crear `tracking_config.json`

- [ ] **TASK 1.2**: Implementar `core/prediction_tracker.py`
  - Clase `PredictionTracker` con métodos básicos
  - Función `record_prediction()` 
  - Función `load_predictions()`
  - Función `save_predictions()`

- [ ] **TASK 1.3**: Diseñar schema de datos
  - Definir estructura JSON completa
  - Validación de datos
  - Manejo de IDs únicos

- [ ] **TASK 1.4**: Integrar en `analysis/live_mode.py`
  - Import del tracker
  - Llamar registro después de predicción
  - Manejo de errores de registro

- [ ] **TASK 1.5**: Integrar en `analysis/live_system.py`
  - Integrar tracker en modo live
  - Registro automático de predicciones live

### ETAPA 2: SISTEMA DE RESULTADOS
- [ ] **TASK 2.1**: Crear `core/result_updater.py`
  - Función `search_match_results()`
  - Función `update_prediction_result()`
  - Función `mark_pending_results()`

- [ ] **TASK 2.2**: Implementar búsqueda de resultados
  - Integración con SofaScore API (usar datos existentes)
  - Búsqueda por equipos y fecha
  - Manejo de múltiples fuentes

- [ ] **TASK 2.3**: Sistema de actualización manual
  - Comando para marcar resultados manualmente
  - Interfaz para corregir resultados erróneos
  - Validación de datos de entrada

- [ ] **TASK 2.4**: Estado de seguimiento
  - Estados: `pending`, `completed`, `cancelled`
  - Timestamps de actualización
  - Audit trail de cambios

### ETAPA 3: ANÁLISIS Y MÉTRICAS
- [ ] **TASK 3.1**: Crear `analysis/prediction_analyzer.py`
  - Clase `PredictionAnalyzer`
  - Función `calculate_accuracy()`
  - Función `calculate_roi()`
  - Función `analyze_by_context()`

- [ ] **TASK 3.2**: Implementar métricas básicas
  - Overall accuracy
  - Over/Under accuracy por separado
  - Average edge magnitude
  - Success rate por rango de odds

- [ ] **TASK 3.3**: Implementar métricas avanzadas
  - Context-based analysis (Q3, Q4, etc.)
  - Team-based performance
  - Time-based analysis (hora/día)
  - Streak tracking

- [ ] **TASK 3.4**: Crear sistema de reportes
  - Función `generate_daily_report()`
  - Función `generate_weekly_report()`
  - Función `generate_monthly_report()`

### ETAPA 4: EXPORTACIÓN Y UI
- [ ] **TASK 4.1**: Crear `core/data_exporter.py`
  - Clase `DataExporter`
  - Export a JSON
  - Export a CSV
  - Export a Excel
  - Export a PDF

- [ ] **TASK 4.2**: Modificar `main.py`
  - Nuevos comandos CLI:
    - `--export-predictions`
    - `--show-stats`
    - `--update-results`
    - `--track-results`
  - Parámetros de filtrado (fecha, equipo, etc.)

- [ ] **TASK 4.3**: Interfaz de seguimiento
  - Menú en modo manual para ver estadísticas
  - Menú en modo live para seguimiento
  - Opciones de exportación integradas

- [ ] **TASK 4.4**: Sistema de backup
  - Backup automático diario
  - Función de recuperación
  - Validación de integridad

### ETAPA 5: TESTING Y VALIDACIÓN
- [ ] **TASK 5.1**: Crear tests unitarios
  - Test del PredictionTracker
  - Test del ResultUpdater
  - Test del PredictionAnalyzer
  - Test del DataExporter

- [ ] **TASK 5.2**: Crear tests de integración
  - Test de flujo completo
  - Test de exportación
  - Test de recuperación de errores

- [ ] **TASK 5.3**: Validación con datos reales
  - Test con predicciones existentes
  - Verificación de cálculos de accuracy
  - Validación de exportación

## 🔧 COMANDOS FINALES DEL SISTEMA

### Para el Usuario Final:
```bash
# Predicción normal (con tracking automático)
python main.py
python main.py --live

# Ver estadísticas
python main.py --show-stats
python main.py --show-stats --last-week
python main.py --show-stats --team "GS Warriors"

# Exportar datos
python main.py --export-predictions --format csv
python main.py --export-predictions --date 2025-11-10 --format excel
python main.py --export-predictions --last-month --format pdf

# Gestión de resultados
python main.py --update-results  # Buscar resultados pendientes
python main.py --mark-result --prediction-id "uuid" --actual-total 225.5

# Análisis avanzado
python main.py --analyze-performance --context q4
python main.py --analyze-roi --format detailed
```

## 📊 EJEMPLO DE FLUJO DE TRABAJO

### 1. PREDICCIÓN CON TRACKING
```
Usuario ejecuta: python main.py
Selecciona: NBA → GS Warriors vs Indiana Pacers
Sistema obtiene línea: 221.5 automáticamente
Sistema hace predicción: 224.8
[ NUEVO ] Sistema registra predicción automáticamente
Sistema muestra: "Recomendación: OVER 221.5 (3.0% edge)"
```

### 2. ACTUALIZACIÓN DE RESULTADOS
```
Sistema ejecuta cada 6h: python main.py --update-results
Encuentra resultado: GS Warriors 118, Indiana Pacers 112 = Total 230
Actualiza registro: actual_total = 230, was_correct = true
```

### 3. ANÁLISIS Y EXPORTACIÓN
```
Usuario ejecuta: python main.py --show-stats
Sistema muestra:
- Overall Accuracy: 73.2%
- OVER Accuracy: 71.5%
- UNDER Accuracy: 75.0%
- Average Edge: 2.8%

Usuario ejecuta: python main.py --export-predictions --last-week --format csv
Sistema genera: predictions_2025-11-04_to_2025-11-10.csv
```

## 🎯 CRITERIOS DE ÉXITO

### ETAPA 1 COMPLETADA CUANDO:
- ✅ Predicciones se registran automáticamente
- ✅ Datos se guardan en formato estructurado
- ✅ No interfiere con funcionalidad existente

### ETAPA 2 COMPLETADA CUANDO:
- ✅ Sistema puede buscar resultados automáticamente
- ✅ Usuario puede marcar resultados manualmente
- ✅ Estados de tracking funcionan correctamente

### ETAPA 3 COMPLETADA CUANDO:
- ✅ Métricas de rendimiento se calculan correctamente
- ✅ Reportes se generan sin errores
- ✅ Análisis por contexto funciona

### ETAPA 4 COMPLETADA CUANDO:
- ✅ Exportación en todos los formatos funciona
- ✅ Comandos CLI están implementados
- ✅ Interfaz de usuario integrada

### SISTEMA COMPLETO CUANDO:
- ✅ Flujo completo funciona end-to-end
- ✅ Tests pasan sin errores
- ✅ Documentación está completa
- ✅ Usuario puede trackear predicciones profesionalmente

## 🚀 ORDEN DE IMPLEMENTACIÓN SUGERIDO

1. **Semana 1**: ETAPA 1 (Backend básico)
2. **Semana 2**: ETAPA 2 (Sistema de resultados)  
3. **Semana 3**: ETAPA 3 (Análisis y métricas)
4. **Semana 4**: ETAPA 4 (Exportación y UI)
5. **Semana 5**: ETAPA 5 (Testing y refinamiento)

Cada etapa debe completarse y probarse antes de proceder a la siguiente.