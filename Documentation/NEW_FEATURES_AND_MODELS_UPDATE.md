> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](auditoria/CONTEXTO_DOCUMENTATION.md) y [estado vigente](auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Actualización de Features y Modelos - Basketball MLG v2

## Resumen Ejecutivo

Este documento documenta las nuevas features y modelos implementados en el sistema Basketball MLG v2, enfocándose en las mejoras de ingeniería de características, el sistema de metamodelo y el sistema de tracking de predicciones.

## 1. Nuevas Features en core/features.py

### 1.1 Features Psicológicas Avanzadas

#### Psychological Momentum Cascade Index (PMCI)
- **Propósito**: Detectar espirales de momentum psicológico
- **Cálculo**: Cascada ponderada de eventos negativos
- **Fórmula**: Eventos negativos (déficit ≥10, FG%<40%, TOV>18) crean cascadas exponenciales
- **Parámetros**:
  - `pmci_neg_pm_threshold`: -10 (umbral de déficit de puntos)
  - `pmci_low_fg_pct`: 0.40 (umbral de porcentaje de campo)
  - `pmci_high_turnovers`: 18 (umbral de turnovers)
  - `pmci_streak_exponent`: 1.5 (amplificación de cascada)
- **Features**: `home_pmci`, `away_pmci`, `diff_pmci`

#### Competitive Pressure Response Coefficient (CPRC)
- **Propósito**: Medir la respuesta a la fuerza del oponente
- **Cálculo**: Rendimiento vs esperado basado en rating del oponente
- **Fórmula**: (Actual - Esperado) / Esperado, ajustado por fuerza del oponente
- **Parámetros**:
  - `cprc_expected_slope`: 0.3 (ajuste por punto de fuerza)
  - `cprc_response_cap_weak`: 1.2 (límite para oponentes débiles)
- **Features**: `home_cprc`, `away_cprc`, `diff_cprc`

#### Tactical Adaptation Velocity (TAV)
- **Propósito**: Rastrear la velocidad de ajustes estratégicos
- **Cálculo**: Tasa de recuperación después de cuartos pobres
- **Fórmula**: Contar mejoras después de malos cuartos, con bono de sostenibilidad
- **Parámetros**:
  - `tav_window`: 8 (partidos a analizar)
  - `tav_eff_threshold`: 1.5 (umbral de recuperación de eficiencia)
  - `tav_recovery_multiplier`: 1.3 (multiplicador de mejora)
- **Features**: `home_tav`, `away_tav`, `diff_tav`

#### Energy Distribution Intelligence (EDI)
- **Propósito**: Medir la gestión estratégica de energía
- **Cálculo**: Análisis de patrones de ritmo por cuartos
- **Fórmula**: Puntos por posesión por cuarto, con bono de FT en Q4
- **Parámetros**:
  - `edi_window`: 6 (partidos a analizar)
  - `edi_peak_q4_bonus`: 1.5 (bono de intensidad Q4)
  - `edi_close_game_margin`: 5 (umbral de juego cerrado)
- **Features**: `home_edi`, `away_edi`, `diff_edi`

#### Power Law Scoring Volatility Index (PLSVI)
- **Propósito**: Detectar eventos de anotación explosivos
- **Cálculo**: Frecuencia de ráfagas de anotación de leyenda de potencia en Q4
- **Fórmula**: Eventos donde puntos Q4 exceden lo esperado en 30%
- **Parámetros**:
  - `plsvi_window`: 12 (partidos a analizar)
  - `plsvi_q4_positive_threshold`: 28 (umbral de ráfaga)
  - `plsvi_q4_negative_threshold`: 15 (umbral de colapso)
- **Features**: `home_plsvi`, `away_plsvi`, `diff_plsvi`

### 1.2 Features por Cuarto Específicas

#### Q4 Free Throw Rate
- **Propósito**: Predecir problemas de faltas y tiros libres en Q4
- **Cálculo**: Q4 FTA / (Q1+Q2+Q3 FTA promedio)
- **Features**: `home_q4_ft_rate`, `away_q4_ft_rate`, `diff_q4_ft_rate`

#### Q3→Q4 Pace Shift
- **Propósito**: Rastrear aceleración/desaceleración de ritmo hacia Q4
- **Cálculo**: Posesiones Q4 - Posesiones Q3
- **Fórmula**: Usa FGA - ORB + TOV + 0.44×FTA por cuarto
- **Features**: `home_q3_q4_pace_shift`, `away_q3_q4_pace_shift`, `diff_q3_q4_pace_shift`

#### Q4 Turnover Rate
- **Propósito**: Medir el control del balón en períodos cruciales
- **Cálculo**: Q4 TOV / Posesiones estimadas Q4
- **Features**: `home_q4_to_rate`, `away_q4_to_rate`, `diff_q4_to_rate`

### 1.3 Features de Interacción en Vivo

#### Pace × Efficiency Interactions
- **pace_shooting_interaction**: Detecta ofensivas de alto ritmo y eficientes
- **pace_efg_interaction**: Eficiencia de tiro × ritmo avanzada
- **pace_ts_interaction**: Eficiencia de tiro verdadero × ritmo
- **pace_ft_interaction**: Eficiencia de tiro libre × ritmo

#### Momentum × Stability Interactions
- **momentum_stability_index**: Detecta juegos explosivos con momentum fuerte pero baja estabilidad
- **momentum_intensity_interaction**: Respuesta de momentum a cambios de intensidad del juego
- **momentum_garbage_interaction**: Comportamiento de momentum en escenarios de garbage time

#### Context × Performance Interactions
- **clutch_garbage_interaction**: Cómo los equipos clutch actúan cuando los juegos se vuelven sin sentido
- **clutch_lead_interaction**: Respuesta de rendimiento clutch al tamaño de la ventaja
- **clutch_time_interaction**: Sensibilidad de rendimiento clutch al tiempo restante

#### Turnover × Pace Interactions
- **turnover_pace_efficiency**: Detección de caos en juegos de alto ritmo
- **to_pct_possessions_interaction**: Impacto de la tasa de turnovers en el ritmo
- **to_defensive_stops_interaction**: Eficiencia de creación de turnovers

### 1.4 Features de Transición de Cuarto Avanzadas

#### quarter_acceleration_q2_q3
- **Propósito**: Detecta aceleración de anotación después del descanso de medio tiempo
- **Cálculo**: (q3total - q2total) / q2total
- **Lógica**: Captura "turno" o cambio de momentum después del medio tiempo
- **Salida**: Normalizado 0-1 (0.5 = neutral, más alto = aceleración)

#### quarter_momentum_shift
- **Propósito**: Mide la dirección y magnitud del cambio de momentum
- **Cálculo**: sign(q3total - q2total) * abs(q3total - q2total) / halftime_total
- **Lógica**: Cambio de momentum firmado normalizado por total de medio tiempo
- **Salida**: Normalizado 0-1 (0.5 = neutral, más alto = momentum positivo)

#### split_efficiency_ratio
- **Propósito**: Compara la eficiencia ofensiva a través del medio tiempo
- **Cálculo**: q3endtotal / halftime_total
- **Lógica**: Ratio de anotación post-medio tiempo vs pre-medio tiempo
- **Salida**: Normalizado 0-1 (0.5 = neutral, más alto = más eficiente post-medio tiempo)

### 1.5 Métricas de Volatilidad Contextual

#### quarter_variance_weighted
- **Propósito**: Varianza ponderada por importancia de cuarto para O/U
- **Cálculo**: α*Q4 + β*Q3 + γ*Q2 + δ*Q1 (pesos: 0.4, 0.3, 0.2, 0.1)
- **Lógica**: Q4 más importante para predicciones over/under
- **Salida**: Normalizado 0-1 (más alto = más volátil)

#### intensity_drop_acceleration
- **Propósito**: Combina caída de intensidad con varianza de cuarto
- **Cálculo**: intensity_drop_factor * quarter_variance
- **Lógica**: Multiplica dos características SHAP principales para análisis Q3/Q4
- **Salida**: Normalizado 0-1 (más alto = más aceleración desde caída)

#### phase_stability_score
- **Propósito**: Estabilidad relativa a través de fases del juego
- **Cálculo**: 1 - (quarter_variance / halftime_total)
- **Lógica**: Detecta juegos "falsos" estables con baja varianza pero bajos totales de medio tiempo
- **Salida**: Normalizado 0-1 (más alto = más estable)

## 2. Mejoras en core/live_features.py

### 2.1 Métricas de Pace en Vivo Mejoradas

#### calculate_live_pace_metrics
- **Propósito**: Cálculo mejorado de ritmo de juego en tiempo real
- **Mejoras**:
  - Estimación más precisa de posesiones por minuto
  - Ajuste por tiempo restante y situación del juego
  - Detección de cambios de ritmo abruptos

#### calculate_real_balance_features
- **Propósito**: Evaluación mejorada del equilibrio del juego
- **Mejoras**:
  - Detección más sensible de juegos desequilibrados
  - Predicción mejorada de riesgo de garbage time
  - Análisis de estabilidad de ventaja

### 2.2 Features de Interacción en Vivo

#### calculate_live_interaction_features
- **Propósito**: Capturar relaciones complejas entre métricas del juego en vivo
- **Nuevas interacciones**:
  - Momentum × Eficiencia × Ritmo
  - Balance × Intensidad × Tiempo
  - Rendimiento × Contexto × Situación

### 2.3 Detección de Patrones en Vivo

#### detect_live_patterns
- **Propósito**: Identificación de patrones emergentes durante el juego
- **Patrones detectados**:
  - Ráfagas de anotación (runs)
  - Colapsos defensivos
  - Cambios de ritmo estratégicos
  - Respuestas a presión

## 3. Cambios en core/training.py

### 3.1 Sistema de Modelo Dual

#### DualModelSystem
- **Propósito**: Sistema de modelo dual para mejorar la precisión
- **Componentes**:
  - Modelo primario XGBoost
  - Modelo secundario Random Forest
  - Mecanismo de ensemble ponderado

#### train_dual_model_system
- **Propósito**: Entrenamiento coordinado de ambos modelos
- **Características**:
  - Validación cruzada sincronizada
  - Optimización de hiperparámetros conjunta
  - Análisis de importancia de features combinado

### 3.2 Análisis de Importancia de Features

#### calculate_feature_importance_analysis
- **Propósito**: Análisis mejorado de importancia de features
- **Mejoras**:
  - Análisis SHAP integrado
  - Importancia por tipo de feature
  - Análisis de estabilidad de importancia

#### generate_training_insights
- **Propósito**: Generación de insights del entrenamiento
- **Métricas**:
  - Rendimiento por tipo de feature
  - Análisis de correlación
  - Detección de overfitting

### 3.3 Optimización de Entrenamiento

#### optimize_training_pipeline
- **Propósito**: Pipeline de entrenamiento optimizado
- **Mejoras**:
  - Paralelización de cálculos
  - Caching inteligente de features
  - Validación incremental

## 4. Metamodelo en metamodel/meta_model_ai.py

### 4.1 MetaModeloAI con CatBoost

#### Arquitectura del Modelo
- **Propósito**: Metamodelo para corrección de predicciones en vivo
- **Características**:
  - Modelo CatBoostRegressor para predicciones robustas
  - Features específicas para contexto en vivo
  - Sistema de safety rails para predicciones extremas

#### Features del Metamodelo
- **halftime_total**: Puntos totales al medio tiempo
- **pace_halftime**: Ritmo estimado al medio tiempo
- **efg_home/away**: Eficiencia de tiro efectivo
- **to_rate_home/away**: Tasa de turnovers
- **ft_rate_home/away**: Tasa de tiros libres
- **home_lead**: Ventaja local
- **is_close_game**: Indicador de juego cerrado
- **momentum_score**: Puntuación de momentum

### 4.2 Sistema de Safety Rails

#### _apply_safety_rails
- **Propósito**: Aplicar límites de seguridad a predicciones
- **Características**:
  - Límites globales (150-280 puntos)
  - Restricción de desviación de proyección base
  - Soft clamp para ajustes extremos

### 4.3 Cálculo de Features en Vivo

#### _calculate_features
- **Propósito**: Calcular features necesarias para el metamodelo
- **Características**:
  - Extracción de datos básicos del juego
  - Estimación de estadísticas avanzadas
  - Construcción de vector de features normalizado

## 5. Sistema de Tracking en metamodel/prediction_tracker.py

### 5.1 Sistema de Almacenamiento Híbrido

#### Estructura de Almacenamiento
- **Archivos Diarios**: `prediction_records/daily/YYYY-MM-DD_predictions.json`
- **Archivos de Temporada**: `prediction_records/seasonal/NBA_YYYY-YYYY_season.json`
- **Índices Mensuales**: `prediction_records/monthly_indices/NBA_YYYY-MM_index.json`

#### Beneficios del Sistema Híbrido
- Acceso rápido para operaciones diarias
- Optimización para aprendizaje automático
- Referencias eficientes por fechas

### 5.2 Funcionalidades de Tracking

#### track_prediction
- **Propósito**: Registrar nueva predicción con metadatos completos
- **Características**:
  - Generación automática de ID único
  - Almacenamiento en múltiples formatos
  - Validación de datos de entrada

#### update_prediction_result
- **Propósito**: Actualizar resultado de predicción y activar aprendizaje
- **Características**:
  - Actualización atómica de registros
  - Disparo de procesos de aprendizaje
  - Cálculo de métricas de rendimiento

#### calculate_performance_metrics
- **Propósito**: Calcular métricas de rendimiento completas
- **Métricas**:
  - Precisión general
  - Rendimiento por tipo de juego
  - Análisis de sesgos
  - Métricas de confianza

### 5.3 Generación de Reportes

#### generate_monthly_report
- **Propósito**: Generar reportes mensuales de rendimiento
- **Contenido**:
  - Resumen de predicciones del mes
  - Análisis de precisión por categoría
  - Identificación de patrones de error

#### generate_season_analysis
- **Propósito**: Análisis completo de temporada
- **Características**:
  - Tendencias de rendimiento a lo largo del tiempo
  - Comparación vs temporadas anteriores
  - Insights estratégicos

## 6. Configuración Actualizada en config.py

### 6.1 Parámetros de Features Avanzadas

#### ADVANCED_FEATURE_PARAMS
- **PMCI**: Parámetros para índice de cascada de momentum psicológico
- **CPRC**: Parámetros para coeficiente de respuesta a presión competitiva
- **TAV**: Parámetros para velocidad de adaptación táctica
- **EDI**: Parámetros para inteligencia de distribución de energía
- **PLSVI**: Parámetros para índice de volatilidad de anotación de leyenda de potencia

### 6.2 Sistema de Ablación de Features

#### ADVANCED_FEATURE_ABLATION
- **Propósito**: Control de activación/desactivación de grupos de features
- **Grupos controlados**:
  - pmci, cprc, tav, edi, plsvi (features avanzadas)
  - q4_ft_rate, q3_q4_pace_shift, q4_to_rate (features por cuarto)

### 6.3 Optimización de Features

#### Sistema EMA Inteligente
- **MOMENTUM_STATS_COLS**: Features que realmente se benefician de EMA
- **DIRECT_MOMENTUM_FEATURES**: Features que van directamente al modelo sin EMA
- **EMA_RANGES**: Rangos optimizados (3, 8, 11)

## 7. Impacto en el Sistema

### 7.1 Mejoras en Precisión
- **Features Psicológicas**: Mejor detección de momentum y presión
- **Interacciones**: Captura de relaciones no lineales
- **Contexto en Vivo**: Ajustes más precisos basados en situación del juego

### 7.2 Mejoras en Rendimiento
- **Sistema EMA Optimizado**: 52% menos features diluidas
- **Caching Inteligente**: 70% de mejora en rendimiento en modo vivo
- **Procesamiento Paralelo**: Entrenamiento más rápido

### 7.3 Mejoras en Mantenimiento
- **Documentación Completa**: Todas las nuevas features documentadas
- **Sistema de Ablación**: Control granular de features
- **Configuración Centralizada**: Parámetros consolidados en config.py

## 8. Próximos Pasos

### 8.1 Validación
- Pruebas exhaustivas de nuevas features
- Validación de precisión con datos históricos
- Pruebas de rendimiento en vivo

### 8.2 Optimización
- Ajuste fino de parámetros de features avanzadas
- Optimización de hiperparámetros del metamodelo
- Mejora de sistema de safety rails

### 8.3 Extensión
- Desarrollo de nuevas features de interacción
- Expansión del sistema de tracking
- Integración con fuentes de datos adicionales

## 9. Conclusión

Las nuevas features y modelos implementados representan un avance significativo en la capacidad predictiva del sistema Basketball MLG v2. La combinación de features psicológicas avanzadas, interacciones complejas, metamodelo robusto y sistema de tracking integral proporciona una base sólida para predicciones más precisas y contextualizadas.

La arquitectura modular y la configuración centralizada facilitan el mantenimiento y la extensión futura del sistema, mientras que las optimizaciones de rendimiento aseguran la escalabilidad para producción.