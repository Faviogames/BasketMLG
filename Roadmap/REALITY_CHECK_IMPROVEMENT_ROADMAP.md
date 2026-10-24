> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_ROADMAP.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Roadmap de Mejoras del Sistema de Reality Check

## 📋 Resumen Ejecutivo

Basado en el análisis exhaustivo de datos de predicciones con y sin Reality Check, este roadmap detalla un plan estratégico para mejorar la precisión del sistema de predicciones de baloncesto, manteniendo los beneficios del mecanismo de control de calidad mientras se minimizan sus fallos.

### 🎯 Objetivo Principal
Evolucionar el Reality Check de un sistema de ajuste fijo a un sistema inteligente y contextual que pueda distinguir entre optimismo sistemático y predicciones legítimamente altas, con el objetivo de aumentar la tasa de aciertos general por encima del 75%.

## 🏗️ ARQUITECTURA ACTUAL VS PROPUESTA

### Arquitectura Actual (Problemas Identificados)
```
Modelo Base → Ajuste por Contexto → Reality Check → Meta-modelo
     ↓              ↓                ↓           ↓
  232.0         230.5            218.5       218.5
```

**Problemas:**
1. **Meta-modelo pasivo**: Solo aplica pequeños ajustes (+/- 8 pts)
2. **Reality Check duplicado**: Aplica penalizaciones que el meta-modelo debería manejar
3. **Contexto separado**: El ajuste por contexto no se integra con el meta-modelo
4. **Falta de aprendizaje**: El meta-modelo no aprende de los errores del RC

### Arquitectura Propuesta (Integrada)
```
Modelo Base → Meta-modelo Inteligente (Integrado)
     ↓                    ↓
  232.0               218.5
     ↳ Contexto dinámico
     ↳ Reality check integrado
     ↳ Detección de outliers
     ↳ Aprendizaje continuo
```

**Ventajas:**
1. **Meta-modelo activo**: Toma decisiones inteligentes basadas en contexto
2. **Eliminación de duplicación**: Una sola capa de ajuste inteligente
3. **Aprendizaje continuo**: El meta-modelo aprende de sus errores
4. **Contexto unificado**: Todas las señales se procesan juntas

## 📊 Análisis de Datos Actuales

### Rendimiento Comparativo
| Métrica | Sin Reality Check | Con Reality Check | Impacto |
|---------|-------------------|-------------------|---------|
| Tasa de Aciertos | 66.7% (4/6) | 66.7% (2/3) | Neutral |
| Fallos por Optimismo | 2 casos | 0 casos | ✅ Mejora |
| Falsos Negativos | 0 casos | 1 caso | ❌ Problema |
| Balance Neto | - | +1 acierto | ✅ Positivo |

### Patrones Identificados
1. **Optimismo Sistemático del Modelo Base**: Confirmado en ambos grupos
2. **Efectividad del RC**: Convierte fallos en aciertos en 66.7% de los casos
3. **Problema Principal**: Penalización de outliers legítimos (caso Knicks/Heat)

## 🚀 Plan de Mejoras Detallado

### 🎯 ANÁLISIS DEL PROBLEMA ACTUAL

#### Problema Identificado: Meta-modelo Pasivo
El análisis del código revela que el meta-modelo actual tiene una participación muy limitada:

```python
# Líneas 2096-2137 en live_mode.py
if model_type in ['ridge', 'ridge_linear'] or ('coef' in meta_payload and 'intercept' in meta_payload):
    # Solo aplica pequeños ajustes (+/- 8 pts)
    meta_offset = float(np.dot(np.array(row_vals, dtype=float), np.array(coef, dtype=float)) + intercept)
    if abs(meta_offset) > clip_val:
        meta_offset = clip_val if meta_offset > 0 else -clip_val
    final_prediction = context_adjusted_prediction + meta_offset
```

**Problemas:**
1. **Ajustes mínimos**: Limitado a +/- 8 puntos
2. **Sin aprendizaje**: No actualiza sus parámetros basados en resultados
3. **Duplicación de lógica**: Reality Check y meta-modelo aplican ajustes similares
4. **Falta de contexto**: No considera el contexto específico del partido

### Fase 1: Integración de Meta-modelo y Reality Check (Semanas 1-2)

#### 1.1 Meta-modelo Unificado Inteligente

**Objetivo**: Unificar el Reality Check y el meta-modelo en una sola capa inteligente.

**Arquitectura Propuesta**:
```python
class MetaModeloInteligente:
    """
    Meta-modelo unificado que reemplaza Reality Check + meta-modelo
    """
    def __init__(self):
        self.feature_weights = {
            'base_prediction': 0.4,      # Predicción del modelo base
            'context_signals': 0.3,        # Señales de contexto (pace, fouls, etc.)
            'historical_patterns': 0.2,     # Patrones históricos (H2H, team trends)
            'outlier_detection': 0.1        # Detección de outliers legítimos
        }
        self.learning_rate = 0.01
        self.performance_history = []
    
    def predecir_ajuste_inteligente(self, base_prediction, contexto, live_signals):
        """
        Predicción unificada que reemplaza todas las capas de ajuste
        """
        # 1. Análisis de contexto unificado
        context_score = self.analizar_contexto(contexto, live_signals)
        
        # 2. Detección de outliers legítimos
        outlier_score = self.detectar_outlier_legitimo(base_prediction, contexto)
        
        # 3. Análisis de patrones históricos
        pattern_score = self.analizar_patrones_historicos(contexto)
        
        # 4. Cálculo de ajuste unificado
        ajuste_total = (
            self.feature_weights['base_prediction'] * base_prediction +
            self.feature_weights['context_signals'] * context_score +
            self.feature_weights['historical_patterns'] * pattern_score +
            self.feature_weights['outlier_detection'] * outlier_score
        )
        
        return ajuste_total
    
    def aprender_de_resultado(self, prediccion, resultado_real, contexto):
        """
        Aprendizaje continuo basado en resultados
        """
        error = abs(prediccion - resultado_real)
        
        # Actualizar pesos basándose en el error
        if error > 15:  # Error significativo
            self.adjust_weights(contexto, error)
        
        # Guardar para análisis
        self.performance_history.append({
            'prediccion': prediccion,
            'resultado': resultado_real,
            'error': error,
            'contexto': contexto,
            'timestamp': datetime.now()
        })
    
    def adjust_weights(self, contexto, error):
        """
        Ajuste dinámico de pesos basado en rendimiento
        """
        # Lógica de gradient descent simple
        for feature in self.feature_weights:
            if contexto.get(feature + '_importance', 0) > 0:
                # Aumentar peso si fue importante, reducir si no
                adjustment = self.learning_rate * (error / 100.0)
                self.feature_weights[feature] += adjustment
        
        # Normalizar pesos para que sumen 1.0
        total_weight = sum(self.feature_weights.values())
        for feature in self.feature_weights:
            self.feature_weights[feature] /= total_weight
```

#### 1.2 Eliminación de Duplicación

**Objetivo**: Remover la lógica duplicada entre Reality Check y meta-modelo.

**Cambios Propuestos**:
```python
# ELIMINAR: _apply_reality_check_penalty() en live_mode.py
# ELIMINAR: Lógica de meta-modelo actual (líneas 2046-2156)
# REEMPLAZAR: Por meta_modelo_inteligente.predecir_ajuste_inteligente()

def get_predictions_with_unified_meta(home_team_name, away_team_name, q_scores, trained_data, pre_game_cache=None, live_signals=None, silent_mode=False):
    """
    Nueva función unificada que reemplaza el sistema actual
    """
    # ... código existente para features ...
    
    # Predicción base del modelo
    base_prediction = model.predict(X_pred)[0]
    
    # 🆕 META-MODELO UNIFICADO INTELIGENTE
    meta_modelo = MetaModeloInteligente()
    
    # Cargar modelo entrenado si existe
    if os.path.exists(MODELS_FOLDER + "/meta_modelo_inteligente.joblib"):
        meta_modelo = joblib.load(MODELS_FOLDER + "/meta_modelo_inteligente.joblib")
    
    # Contexto unificado
    contexto = {
        'base_prediction': base_prediction,
        'q_scores': q_scores,
        'balance_features': balance_features,
        'live_pace_metrics': live_pace_metrics,
        'quarter_stage': quarter_stage,
        'league_name': league_name,
        'team_trends': team_trends
    }
    
    # Predicción final unificada
    final_prediction = meta_modelo.predecir_ajuste_inteligente(
        base_prediction, contexto, live_signals
    )
    
    # Guardar para aprendizaje futuro
    prediction_record = {
        'base_prediction': base_prediction,
        'final_prediction': final_prediction,
        'contexto': contexto,
        'timestamp': datetime.now()
    }
    
    return final_prediction, predictions, {
        'base_prediction': base_prediction,
        'final_prediction': final_prediction,
        'meta_modelo_info': meta_modelo.get_debug_info(),
        'prediction_record': prediction_record
    }
```

#### 1.3 Sistema de Aprendizaje Continuo

**Objetivo**: Implementar aprendizaje automático del meta-modelo.

**Implementación**:
```python
class SistemaAprendizajeContinuo:
    """
    Sistema que actualiza el meta-modelo basándose en resultados reales
    """
    def __init__(self):
        self.predicciones_pendientes = {}
        self.meta_modelo = MetaModeloInteligente()
    
    def registrar_prediccion(self, prediction_id, prediccion, contexto):
        """
        Registra predicción para seguimiento futuro
        """
        self.predicciones_pendientes[prediction_id] = {
            'prediccion': prediccion,
            'contexto': contexto,
            'timestamp': datetime.now()
        }
    
    def actualizar_con_resultado(self, prediction_id, resultado_real):
        """
        Actualiza meta-modelo cuando se conoce el resultado
        """
        if prediction_id not in self.predicciones_pendientes:
            return
        
        prediccion_info = self.predicciones_pendientes[prediction_id]
        
        # Aprendizaje del meta-modelo
        self.meta_modelo.aprender_de_resultado(
            prediccion_info['prediccion'],
            resultado_real,
            prediccion_info['contexto']
        )
        
        # Guardar modelo actualizado
        joblib.dump(self.meta_modelo, MODELS_FOLDER + "/meta_modelo_inteligente.joblib")
        
        # Limpiar predicción procesada
        del self.predicciones_pendientes[prediction_id]
        
        print(f"✅ Meta-modelo actualizado con resultado: error={abs(prediccion_info['prediccion'] - resultado_real):.1f}")
    
    def generar_reporte_aprendizaje(self):
        """
        Genera reporte de progreso del aprendizaje
        """
        if not self.meta_modelo.performance_history:
            return "Sin datos de aprendizaje aún"
        
        errores_recientes = [p['error'] for p in self.meta_modelo.performance_history[-10:]]
        error_promedio = sum(errores_recientes) / len(errores_recientes)
        
        return {
            'predicciones_aprendidas': len(self.meta_modelo.performance_history),
            'error_promedio_reciente': error_promedio,
            'pesos_actuales': self.meta_modelo.feature_weights,
            'mejora_rendimiento': self.calcular_mejora_rendimiento()
        }
```

### Fase 2: Mejoras Intermedias (Semanas 3-4)

#### 2.1 Meta-modelo Contextual

**Objetivo**: Reemplazar ajustes fijos por un modelo que aprende del contexto.

**Arquitectura del Meta-modelo**:
```python
class MetaModeloContextual:
    def __init__(self):
        self.pesos_contexto = {
            "temporal": 0.3,      # Rendimiento reciente
            "h2h": 0.25,          # Historial enfrentamientos
            "linea_apuestas": 0.2, # Desviación línea oficial
            "factores_especiales": 0.15, # Lesiones, back-to-back
            "regimen_partido": 0.1  # Tipo de partido
        }
    
    def predecir_ajuste(self, prediccion_base, contexto):
        """
        Predice el ajuste óptimo basado en contexto
        """
        score_contextual = self.calcular_score_contextual(contexto)
        ajuste_recomendado = self.mapear_score_a_ajuste(score_contextual)
        
        return ajuste_recomendado
    
    def actualizar_pesos(self, historial_resultados):
        """
        Actualiza pesos basándose en rendimiento reciente
        """
        # Lógica de aprendizaje automático
        pass
```

#### 2.2 Sistema de Detección de Regímenes

**Objetivo**: Identificar diferentes tipos de partidos y aplicar ajustes específicos.

**Clasificación de Regímenes**:
```python
class DetectorRegimenes:
    def __init__(self):
        self.regimenes = {
            "alto_scoring_ofensivo": {
                "caracteristicas": ["ppg_alto", "tempo_rapido", "defensa_debil"],
                "ajuste_rc": "minimo"
            },
            "bajo_scoring_defensivo": {
                "caracteristicas": ["ppg_bajo", "tempo_lento", "defensa_fuerte"],
                "ajuste_rc": "moderado"
            },
            "irregular_volatil": {
                "caracteristicas": ["varianza_alta", "rendimiento_inconsistente"],
                "ajuste_rc": "conservador"
            },
            "estandar_equilibrado": {
                "caracteristicas": ["metricas_normales", "consistencia_media"],
                "ajuste_rc": "estandar"
            }
        }
    
    def clasificar_partido(self, contexto):
        """
        Clasifica el partido en un régimen específico
        """
        # Lógica de clasificación
        pass
```

### Fase 3: Mejoras Avanzadas (Semanas 5-8)

#### 3.1 Ensemble de Reality Checks

**Objetivo**: Múltiples enfoques de RC con votación ponderada.

**Arquitectura Ensemble**:
```python
class EnsembleRC:
    def __init__(self):
        self.rcs = {
            "estadistico": RC_Estadistico(),
            "contextual": RC_Contextual(),
            "aprendizaje": RC_Aprendizaje(),
            "hibrido": RC_Hibrido()
        }
        self.pesos_votacion = {
            "estadistico": 0.25,
            "contextual": 0.30,
            "aprendizaje": 0.25,
            "hibrido": 0.20
        }
    
    def votar_ajuste(self, prediccion_base, contexto):
        """
        Sistema de votación ponderada para ajuste final
        """
        votos = {}
        for nombre, rc in self.rcs.items():
            ajuste = rc.calcular_ajuste(prediccion_base, contexto)
            votos[nombre] = ajuste
        
        ajuste_final = self.ponderar_votos(votos)
        return ajuste_final
```

#### 3.2 Sistema de Retroalimentación Continua

**Objetivo**: Actualización automática de parámetros basada en resultados.

**Implementación**:
```python
class SistemaRetroalimentacion:
    def __init__(self):
        self.historial_resultados = []
        self.parametros_optimos = {}
        self.tasa_aprendizaje = 0.01
    
    def procesar_resultado(self, prediccion, resultado_real):
        """
        Procesa cada resultado para actualizar parámetros
        """
        error = abs(prediccion.final - resultado_real)
        impacto_rc = self.evaluar_impacto_rc(prediccion, resultado_real)
        
        self.actualizar_parametros(error, impacto_rc)
        self.historial_resultados.append({
            "prediccion": prediccion,
            "resultado": resultado_real,
            "error": error,
            "impacto_rc": impacto_rc
        })
    
    def optimizar_parametros(self):
        """
        Optimización automática de parámetros
        """
        # Algoritmo de optimización (ej: gradient descent)
        pass
```

## 📈 Métricas de Éxito y KPIs

### Métricas Cuantitativas
| Métrica | Actual | Objetivo Fase 1 | Objetivo Final |
|---------|--------|-----------------|----------------|
| Tasa de Aciertos General | 66.7% | 70% | 75% |
| Falsos Negativos | 33.3% | < 20% | < 10% |
| Balance Neto RC | +1 | +3 | +5 |
| Sesgo UNDER/OVER | 89/11 | 75/25 | 65/35 |

### Métricas Cualitativas
- **Consistencia**: Rendimiento estable a través de diferentes tipos de partidos
- **Adaptabilidad**: Capacidad de ajustarse a cambios en patrones de la liga
- **Interpretabilidad**: Capacidad de explicar por qué se aplicó cada ajuste

## 🔄 Flujo de Trabajo de Implementación

### Semana 1-2: Implementación Crítica
- [ ] Desarrollar sistema de detección de outliers
- [ ] Implementar ajuste dinámico del RC
- [ ] Crear dashboard de monitoreo
- [ ] Tests unitarios y de integración

### Semana 3-4: Mejoras Contextuales
- [ ] Desarrollar meta-modelo contextual
- [ ] Implementar detector de regímenes
- [ ] Integrar con sistema existente
- [ ] Validación con datos históricos

### Semana 5-8: Optimización Avanzada
- [ ] Desarrollar ensemble de RCs
- [ ] Implementar retroalimentación continua
- [ ] Optimización de hiperparámetros
- [ ] Documentación completa

## 🎯 Casos de Uso Específicos

### Caso 1: Knicks vs Heat (Fallo Actual del RC)
**Problema**: Predicción base 270.0 → RC -26.3 → Final 243.2 (Fallo)
**Solución Propuesta**: 
- Detección de outlier legítimo (desviación extrema)
- Análisis de contexto: equipos alto-scoring
- Ajuste reducido o nulo para este tipo de partidos

### Caso 2: Utah vs Atlanta (Fallo sin RC)
**Problema**: Predicción 285.2 → Real 254 (Fallo por optimismo)
**Solución Propuesta**:
- Detección de optimismo sistemático
- Ajuste estándar aplicado correctamente
- Monitoreo de efectividad

### Caso 3: Pistons vs 76ers (Éxito del RC)
**Problema**: Base 232.0 → RC -13.5 → Final 218.5 (Acierto)
**Solución Propuesta**:
- Mantener lógica actual para estos casos
- Refinar umbral de detección
- Optimizar magnitud del ajuste

## 🚨 Riesgos y Mitigación

### Riesgos Identificados
1. **Sobre-ajuste**: Demasiada complejidad en el meta-modelo
2. **Latencia**: Tiempo de procesamiento aumentado
3. **Sobre-optimización**: Adaptación excesiva a datos históricos

### Estrategias de Mitigación
1. **Validación Cruzada**: Tests con múltiples períodos temporales
2. **Monitoreo Continuo**: Alertas para degradación de rendimiento
3. **Rollback Automático**: Reversión a versión anterior si falla

## 📋 Próximos Pasos

### Inmediatos (Esta semana)
1. **Validación del análisis**: Revisión de conclusiones con equipo
2. **Definición de prioridades**: Ajuste de roadmap según feedback
3. **Asignación de recursos**: Determinar equipo y tiempo disponible

### Corto Plazo (Próximas 2 semanas)
1. **Implementación Fase 1**: Desarrollo de detección de outliers
2. **Tests iniciales**: Validación con datos conocidos
3. **Ajuste de parámetros**: Optimización inicial

### Mediano Plazo (Próximo mes)
1. **Implementación completa**: Todas las fases del roadmap
2. **Validación extensiva**: Tests con múltiples datasets
3. **Documentación**: Guías de uso y mantenimiento

## 📞 Puntos para Discusión

### Dudas Técnicas
1. **Umbral de detección de outliers**: ¿Qué nivel de sensibilidad es óptimo?
2. **Frecuencia de actualización**: ¿Con qué periodicidad actualizar los parámetros?
3. **Integración con modelo base**: ¿Cómo afecta al entrenamiento del modelo principal?

### Decisiones de Negocio
1. **Balance riesgo-recompensa**: ¿Preferir más aciertos o menos fallos?
2. **Interpretabilidad vs precisión**: ¿Hasta qué punto sacrificar explicabilidad?
3. **Inversión vs retorno**: ¿Justifica el beneficio el costo de implementación?

### Sugerencias de Mejora
1. **Datos adicionales**: ¿Qué otra información podría ser útil?
2. **Métricas alternativas**: ¿Hay otros KPIs que deberíamos considerar?
3. **Integración externa**: ¿Deberíamos considerar datos de terceros?

---

## 📝 Notas para Reunión de Discusión

1. **Preparar datos**: Exportar dataset completo para análisis conjunto
2. **Demostración**: Preparar demo visual del sistema actual vs propuesto
3. **Tiempo estimado**: Desglose detallado de esfuerzo por fase
4. **Métricas de éxito**: Definir cómo mediremos el éxito del proyecto
5. **Plan de rollback**: Estrategia si las mejoras no funcionan como esperado

---

*Última actualización: 15 de noviembre de 2025*
*Versión: 1.0*
*Estado: Propuesta para discusión*