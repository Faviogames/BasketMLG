> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Análisis de Predicciones de la Semana - Noviembre 2025

## 📊 Resumen Ejecutivo

Análisis detallado de las predicciones realizadas esta semana con el sistema de predicciones de baloncesto, enfocado en identificar patrones de fallo del modelo base y la efectividad de las correcciones del meta-modelo de reality check.

## 🎯 Estadísticas Generales

### Predicciones Analizadas
- **Total de predicciones**: 13 partidos
- **Predicciones completadas**: 12
- **Predicciones pendientes**: 1 (Houston vs Orlando)
- **Período analizado**: 10-17 de noviembre de 2025

### Rendimiento General
| Métrica | Valor | Porcentaje |
|---------|-------|------------|
| **Predicciones correctas** | 7/12 | 58.3% |
| **Predicciones incorrectas** | 5/12 | 41.7% |
| **Promedio de ajuste RC** | -7.2 pts | - |
| **Desviación promedio** | 15.8 pts | - |

## 🔍 Análisis Detallado por Partido

### Predicciones CORRECTAS (7/12)

1. **Cleveland vs Memphis** (Nov 16)
   - Base: 219.5 → RC: -11.0 → Final: 208.5
   - Real: 208 (UNDER) ✅
   - **Ajuste acertado**: RC detectó optimismo y corrigió adecuadamente

2. **Philadelphia vs Detroit** (Nov 15)
   - Base: 232.5 → RC: -13.5 → Final: 218.5
   - Real: 219 (UNDER) ✅
   - **Ajuste acertado**: RC redujo predicción optimista

3. **Brooklyn vs Orlando** (Nov 15)
   - Base: 232.5 → RC: -18.9 → Final: 213.5
   - Real: 203 (UNDER) ✅
   - **Ajuste acertado**: RC aplicó corrección fuerte y acertada

4. **Dallas vs Milwaukee** (Nov 11)
   - Base: 230.5 → RC: 0.0 → Final: 230.5
   - Real: 230 (OVER) ✅
   - **Sin ajuste necesario**: Modelo base acertó sin intervención

5. **Cleveland vs Toronto** (Nov 14)
   - Base: 235.5 → RC: 0.0 → Final: 235.5
   - Real: 239 (UNDER) ✅
   - **Sin ajuste necesario**: Predicción base muy precisa

6. **Phoenix vs Indiana** (Nov 14)
   - Base: 234.5 → RC: 0.0 → Final: 234.5
   - Real: 231 (UNDER) ✅
   - **Sin ajuste necesario**: Modelo base acertó

7. **Memphis vs Oklahoma City** (Nov 10)
   - Base: 229.5 → RC: 0.0 → Final: 229.5
   - Real: 218 (OVER) ✅
   - **Sin ajuste necesario**: Predicción base correcta

### Predicciones INCORRECTAS (5/12)

1. **Miami vs New York** (Nov 15) - ⚠️ **Fallo del RC**
   - Base: 270.5 → RC: -26.3 → Final: 243.5
   - Real: 272 (OVER) ❌
   - **Problema**: RC penalizó excesivamente un outlier legítimo

2. **Utah vs Atlanta** (Nov 14) - ⚠️ **Fallo del Modelo Base**
   - Base: 285.5 → RC: 0.0 → Final: 285.5
   - Real: 254 (UNDER) ❌
   - **Problema**: Optimismo extremo del modelo base no detectado

3. **Boston vs LA Clippers** (Nov 16) - ⚠️ **Fallo del RC**
   - Base: 222.5 → RC: -3.6 → Final: 208.5
   - Real: 239 (OVER) ❌
   - **Problema**: RC aplicó corrección innecesaria

4. **Milwaukee vs LA Lakers** (Nov 16) - ⚠️ **Fallo del RC**
   - Base: 214.5 → RC: -4.0 → Final: 199.5
   - Real: 214 (OVER) ❌
   - **Problema**: RC aplicó corrección incorrecta

5. **Phoenix vs Atlanta** (Nov 17) - ⚠️ **Fallo del RC**
   - Base: 230.6 → RC: -3.7 → Final: 215.6
   - Real: 246 (OVER) ❌
   - **Problema**: RC aplicó corrección innecesaria

## 📈 Patrones Identificados

### 1. Optimismo Sistemático del Modelo Base
- **Evidencia**: Predicciones base consistentemente altas
- **Promedio de predicciones base**: 236.8 pts
- **Promedio de resultados reales**: 230.3 pts
- **Sesgo**: +6.5 pts (hacia OVER)

### 2. Efectividad del Reality Check
| Situación | Casos | Tasa Éxito |
|-----------|-------|------------|
| **Ajustes aplicados** | 7 | 42.9% (3/7) |
| **Sin ajustes** | 5 | 80.0% (4/5) |
| **Total** | 12 | 58.3% (7/12) |

### 3. Patrones de Fallo del Reality Check

#### Falsos Negativos (RC aplicó corrección incorrecta)
- **Boston vs LA Clippers**: Predijo UNDER, fue OVER (+30.5 pts)
- **Milwaukee vs LA Lakers**: Predijo UNDER, fue OVER (+14.5 pts)
- **Phoenix vs Atlanta**: Predijo UNDER, fue OVER (+30.4 pts)
- **Miami vs New York**: Predijo UNDER, fue OVER (+28.5 pts)

#### Características comunes de los falsos negativos:
- Predicciones base moderadas (214-270 pts)
- Ajustes RC pequeños a moderados (-3.6 a -26.3 pts)
- Resultados reales consistentemente OVER
- Diferencia promedio: +26.0 pts

### 4. Predicciones Extremas del Modelo Base

#### Casos con predicciones > 260 pts:
1. **Miami vs New York**: 270.5 pts (Real: 272) - ✅ Correcta
2. **Utah vs Atlanta**: 285.5 pts (Real: 254) - ❌ Incorrecta

#### Análisis:
- El modelo base tiende a predicciones muy altas en partidos con potencial ofensivo
- El Reality Check es inconsistente en manejar estos casos
- Necesidad de mejor discriminación entre outliers legítimos y optimismo

## 🎯 Análisis por Etapa del Partido

### Timing de Predicciones
| Etapa | Casos | Tasa Éxito | Observaciones |
|-------|-------|------------|---------------|
| **HT (Half-Time)** | 9 | 55.6% (5/9) | Mayoría de las predicciones |
| **q2** | 1 | 0% (0/1) | Partido de Dallas vs Portland |
| **pre_game** | 2 | 100% (2/2) | Predicciones más precisas |

### Insights por Etapa:
- Las predicciones en **pre_game** tienen mejor rendimiento (100%)
- Las predicciones en **HT** son más volátiles (55.6%)
- Las predicciones en **q2** necesitan más datos para conclusión

## 📊 Relación Confianza vs Resultados

### Análisis por Nivel de Confianza
| Rango Confianza | Casos | Tasa Éxito | Observaciones |
|-----------------|-------|------------|---------------|
| **Alta (>0.9)** | 1 | 0% (0/1) | Miami vs NY (fallo del RC) |
| **Media (0.7-0.9)** | 4 | 75.0% (3/4) | Buen rendimiento |
| **Baja-Media (0.5-0.7)** | 1 | 100% (1/1) | Philadelphia vs Detroit |
| **Estándar (0.75)** | 6 | 66.7% (4/6) | Predicciones sin RC |

### Hallazgos Clave:
- **Alta confianza no garantiza acierto** (caso Miami-NY)
- **Confianza media tiene mejor balance**
- **Predicciones estándar sin RC son consistentes**

## 🏀 Análisis por Rangos de Puntuación

### Predicciones Bajas (< 220 pts)
- **Casos**: 3
- **Tasa éxito**: 66.7% (2/3)
- **Observación**: Predicciones más conservadoras, mejor rendimiento

### Predicciones Medias (220-240 pts)
- **Casos**: 7
- **Tasa éxito**: 57.1% (4/7)
- **Observación**: Rango más común, rendimiento moderado

### Predicciones Altas (> 240 pts)
- **Casos**: 2
- **Tasa éxito**: 50.0% (1/2)
- **Observación**: Mayor volatilidad, necesidad de ajustes específicos

## 🚨 Problemas Críticos Identificados

### 1. Over-corrección del Reality Check
- **Problema**: RC aplicando correcciones innecesarias
- **Impacto**: 4 de 5 fallos involucran correcciones incorrectas
- **Solución**: Mejorar detección de cuándo NO intervenir

### 2. Fallos en Partidos de Alto Scoring
- **Problema**: Dificultad distinguir outliers legítimos
- **Casos**: Miami-NY, Utah-Atlanta
- **Solución**: Contexto específico para equipos alto-scoring

### 3. Inconsistencia en Predicciones de Half-Time
- **Problema**: Mayor tasa de fallos en HT
- **Causa**: Posible sobre-ajuste basado en rendimiento parcial
- **Solución**: Ajustes más conservadores en HT

## 💡 Recomendaciones de Mejora

### 1. Mejorar Detección de Outliers Legítimos
```python
# Implementar análisis específico para equipos alto-scoring
def es_outlier_legitimo_alto_scoring(equipos, prediccion_base):
    equipos_alto_scoring = ['Miami Heat', 'New York Knicks', 'Phoenix Suns']
    if any(equipo in equipos_alto_scoring for equipo in equipos):
        if prediccion_base > 260:
            return True, "outlier_legitimo_alto_scoring"
    return False, None
```

### 2. Ajustar Umbrales del Reality Check
- **Actual**: Umbral de corrección muy sensible
- **Propuesto**: Aumentar umbral para intervenciones
- **Implementación**: Requerir más evidencia antes de aplicar correcciones

### 3. Sistema de Contexto por Equipos
- **Desarrollar**: Perfiles específicos por equipos
- **Factores**: Estilos de juego, tendencias históricas
- **Aplicación**: Ajustes personalizados según combinación de equipos

### 4. Mejorar Predicciones en Half-Time
- **Problema**: Sobre-ajuste basado en rendimiento parcial
- **Solución**: Ajustes más graduales en HT
- **Implementación**: Factor de conservadurismo para predicciones HT

### 5. Sistema de Aprendizaje Continuo
- **Registrar**: Todos los fallos y aciertos del RC
- **Analizar**: Patrones de fallo sistemáticos
- **Actualizar**: Parámetros del RC basados en resultados

## 📋 Próximos Pasos

### Inmediatos (Esta semana)
1. **Implementar detección mejorada de outliers legítimos**
2. **Ajustar umbrales de intervención del RC**
3. **Crear perfiles de equipos alto-scoring**

### Corto Plazo (Próximas 2 semanas)
1. **Desarrollar sistema de contexto por equipos**
2. **Implementar ajustes conservadores en HT**
3. **Iniciar sistema de aprendizaje continuo**

### Mediano Plazo (Próximo mes)
1. **Sistema ensemble de múltiples Reality Checks**
2. **Integración con datos históricos más extensos**
3. **Dashboard de monitoreo de rendimiento del RC**

## 📊 Métricas de Éxito Propuestas

### Objetivos de Mejora
| Métrica | Actual | Objetivo (4 semanas) | Objetivo (8 semanas) |
|---------|--------|---------------------|---------------------|
| **Tasa de aciertos general** | 58.3% | 65% | 75% |
| **Falsos negativos del RC** | 33.3% | < 20% | < 10% |
| **Precisión en partidos alto-scoring** | 50% | 70% | 80% |
| **Precisión en predicciones HT** | 55.6% | 70% | 80% |

---

*Análisis realizado: 17 de noviembre de 2025*
*Próxima actualización: 24 de noviembre de 2025*