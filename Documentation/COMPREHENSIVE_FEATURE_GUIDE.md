> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](auditoria/CONTEXTO_DOCUMENTATION.md) y [estado vigente](auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Guía Completa de Features - Basketball MLG v2

## Resumen Ejecutivo

Esta guía proporciona una visión completa de todas las características implementadas en el sistema Basketball MLG v2, organizadas por categorías funcionales y con explicaciones detalladas de su propósito, cálculo e impacto predictivo.

## Tabla de Contenidos

1. [Features de Momentum (EMA-Based)](#1-features-de-momentum-ema-based)
2. [Features de Estadísticas Avanzadas](#2-features-de-estadísticas-avanzadas)
3. [Features de Contexto de Rendimiento](#3-features-de-contexto-de-rendimiento)
4. [Features de Juego en Vivo](#4-features-de-juego-en-vivo)
5. [Features de Balance](#5-features-de-balance)
6. [Features de Interacción en Vivo](#6-features-de-interacción-en-vivo)
7. [Features Psicológicas Avanzadas](#7-features-psicológicas-avanzadas)
8. [Features de Transición de Cuarto](#8-features-de-transición-de-cuarto)
9. [Features de Head-to-Head](#9-features-de-head-to-head)
10. [Features de Mercado](#10-features-de-mercado)
11. [Features del Metamodelo](#11-features-del-metamodelo)
12. [Configuración y Parámetros](#12-configuración-y-parámetros)

---

## 1. Features de Momentum (EMA-Based)

### 1.1 Features de Tasa de Victorias
- **Propósito**: Rastrear consistencia y momentum de victorias
- **Cálculo**: Media Móvil Exponencial (EMA) de porcentaje de victorias
- **Rangos**: Corto plazo (3 juegos), Mediano plazo (8 juegos), Largo plazo (11 juegos)
- **Features**:
  - `ema_win_short_term_3`
  - `ema_win_medium_term_8`
  - `ema_win_long_term_11`

### 1.2 Features de Plus/Minus
- **Propósito**: Medir tendencias de eficiencia ofensiva/defensiva
- **Cálculo**: EMA de diferencial de puntos por juego
- **Features**:
  - `ema_plus_minus_short_term_3`
  - `ema_plus_minus_medium_term_8`
  - `ema_plus_minus_long_term_11`

### 1.3 Features de Eficiencia de Anotación
- **Propósito**: Rastrear tendencias ofensivas
- **Cálculo**: EMA de puntos anotados por juego
- **Features**:
  - `ema_scoring_efficiency_short_term_3`
  - `ema_scoring_efficiency_medium_term_8`
  - `ema_scoring_efficiency_long_term_11`

### 1.4 Features de Paradas Defensivas
- **Propósito**: Rastrear tendencias de rendimiento defensivo
- **Cálculo**: EMA de puntos permitidos por juego
- **Features**:
  - `ema_defensive_stops_short_term_3`
  - `ema_defensive_stops_medium_term_8`
  - `ema_defensive_stops_long_term_11`

### 1.5 Features de Rendimiento Clutch
- **Propósito**: Medir rendimiento en juegos cerrados
- **Cálculo**: EMA de tasa de victorias en juegos decididos por ≤10 puntos
- **Features**:
  - `ema_clutch_performance_short_term_3`
  - `ema_clutch_performance_medium_term_8`
  - `ema_clutch_performance_long_term_11`

### 1.6 Features de Momentum de Turnovers
- **Propósito**: Rastrear control del balón y tendencias de ritmo
- **Cálculo**: EMA de tendencias de tasa de turnovers
- **Features**:
  - `ema_turnover_momentum_short_term_3`
  - `ema_turnover_momentum_medium_term_8`
  - `ema_turnover_momentum_long_term_11`

---

## 2. Features de Estadísticas Avanzadas

### 2.1 Features de Posesiones
- **Propósito**: Rastrear ritmo y tempo
- **Cálculo**: Promedio móvil de posesiones estimadas por juego
- **Fórmula**: FGA - ORB + TOV + (0.44 × FTA)
- **Features**:
  - `home_avg_possessions_last_5`
  - `away_avg_possessions_last_5`
  - `diff_avg_possessions_last_5`

### 2.2 Features de Ritmo (Pace)
- **Propósito**: Medir tempo del juego
- **Cálculo**: Posesiones normalizadas a 48 minutos
- **Features**:
  - `home_avg_pace_last_5`
  - `away_avg_pace_last_5`
  - `diff_avg_pace_last_5`

### 2.3 Features de Rating Ofensivo
- **Propósito**: Rastrear eficiencia ofensiva
- **Cálculo**: Puntos por 100 posesiones
- **Fórmula**: (PTS / POSS) × 100
- **Features**:
  - `home_avg_ortg_last_5`
  - `away_avg_ortg_last_5`
  - `diff_avg_ortg_last_5`

### 2.4 Features de Porcentaje de Tiro Verdadero (TS%)
- **Propósito**: Métrica avanzada de eficiencia de tiro
- **Cálculo**: Promedio móvil de TS%
- **Fórmula**: PTS / (2 × (FGA + 0.44 × FTA))
- **Features**:
  - `home_true_shooting_percentage`
  - `away_true_shooting_percentage`
  - `diff_true_shooting_percentage`

---

## 3. Features de Contexto de Rendimiento

### 3.1 Factor de Ventaja Local
- **Propósito**: Medir ventaja de cancha local
- **Cálculo**: Ajuste de rendimiento reciente (1.0 + PM/100)
- **Features**:
  - `home_home_advantage_factor`
  - `away_home_advantage_factor`
  - `diff_home_advantage_factor`

### 3.2 Capacidad de Remontada
- **Propósito**: Rastrear habilidad para superar déficits
- **Cálculo**: Frecuencia de victorias después de estar perdiendo
- **Features**:
  - `home_comeback_ability`
  - `away_comeback_ability`
  - `diff_comeback_ability`

### 3.3 Índice de Consistencia
- **Propósito**: Medir consistencia de anotación
- **Cálculo**: 1 / (1 + CV) donde CV es coeficiente de variación
- **Features**:
  - `home_consistency_index`
  - `away_consistency_index`
  - `diff_consistency_index`

---

## 4. Features de Juego en Vivo

### 4.1 Totales por Cuarto
- **Propósito**: Anotación actual del juego por período
- **Cálculo**: Suma de puntos anotados en cada cuarto
- **Features**:
  - `q1_total`, `q2_total`, `q3_total`
  - `halftime_total`, `q3_end_total`

### 4.2 Límites de Split Dinámicos (PHASE 1)
- **Propósito**: Detectar cambios reales de momentum vs límites de cuarto fijos
- **Cálculo**: Algoritmos avanzados para análisis de estado del juego en vivo

#### adaptive_split_detector
- **Propósito**: Detecta cambios reales de momentum vs límites de cuarto fijos
- **Cálculo**: Detección de puntos de cambio basada en puntos por minuto
- **Lógica**: Compara transición Q2→Q3 contra patrones históricos
- **Salida**: Flag binario (1.0 = cambio de momentum detectado, 0.0 = sin cambio)
- **Features**:
  - `adaptive_split_detector`

#### quarter_transition_shock
- **Propósito**: Mide eventos de "shock" en transiciones de cuarto
- **Cálculo**: Cambio máximo de puntos/minuto en ventanas de 3 minutos
- **Lógica**: Detecta cambios abruptos de ritmo de anotación en límites de cuarto
- **Salida**: Puntuación normalizada 0-1 (más alto = transición más impactante)
- **Features**:
  - `quarter_transition_shock`

### 4.3 Métricas de Eficiencia en Vivo
- **Propósito**: Eficiencia ofensiva/defensiva en tiempo real
- **Cálculo**: Puntos por posesión en juego actual
- **Features**:
  - `live_efficiency_home`
  - `live_efficiency_away`

### 4.4 Métricas de Ritmo en Vivo
- **Propósito**: Tempo actual del juego
- **Cálculo**: Posesiones estimadas por 48 minutos
- **Features**:
  - `live_pace_estimate`
  - `enhanced_pace_estimate`

### 4.5 Indicadores de Momentum
- **Propósito**: Rastrear cambios de momentum de anotación
- **Cálculo**: Cambios de anotación de cuarto a cuarto
- **Features**:
  - `q2_trend`, `q3_trend`
  - `live_momentum_shift`

---

## 5. Features de Balance

### 5.1 Puntuación de Balance del Juego
- **Propósito**: Medir competitividad del juego
- **Cálculo**: 1 / (1 + lead/10) - más alto = más balanceado
- **Features**:
  - `game_balance_score`

### 5.2 Indicador de Paliza Potencial
- **Propósito**: Detectar juegos propensos a convertirse en palizas
- **Cálculo**: Flag binario cuando ventaja Q3 ≥15 puntos
- **Features**:
  - `is_potential_blowout`

### 5.3 Riesgo de Garbage Time
- **Propósito**: Predecir probabilidad de minutos finales sin sentido
- **Cálculo**: Función logística basada en ventaja y tiempo restante
- **Features**:
  - `garbage_time_risk`

### 5.4 Factor de Caída de Intensidad
- **Propósito**: Medir declive en intensidad del juego
- **Cálculo**: Varianza de anotación de cuarto a cuarto
- **Features**:
  - `intensity_drop_factor`

### 5.5 Estabilidad de Ventaja
- **Propósito**: Rastrear cambios de líder durante el juego
- **Cálculo**: Si el líder cambia entre cuartos
- **Features**:
  - `lead_stability`

---

## 6. Features de Interacción en Vivo (FASE 1)

### 6.1 Interacciones Pace × Eficiencia

#### pace_shooting_interaction
- **Propósito**: Detectar ofensivas de alto ritmo y eficientes que llevan a alta anotación
- **Fórmula**: `pace_estimate × shooting_efficiency`
- **Por qué funciona**: Alto ritmo + tiro eficiente = más posesiones × mejor conversión = más puntos
- **Variantes**: pace × eFG%, pace × TS%, pace × FT%
- **Umbral**: >1.2σ por encima de la media histórica dispara +15% de probabilidad over

#### pace_efg_interaction
- **Propósito**: Interacción de eficiencia de tiro avanzada × ritmo
- **Fórmula**: `pace_estimate × eFG%`
- **Por qué funciona**: eFG% cuenta valor extra de triples, combinado con ritmo predice ráfagas de anotación

#### pace_ts_interaction
- **Propósito**: Interacción de eficiencia de tiro verdadero × ritmo
- **Fórmula**: `pace_estimate × TS%`
- **Por qué funciona**: TS% incluye tiros libres, proporcionando imagen completa de tiro

#### pace_ft_interaction
- **Propósito**: Interacción de eficiencia de tiro libre × ritmo
- **Fórmula**: `pace_estimate × FT%`
- **Por qué funciona**: Equipos de alto ritmo con buen tiro libre obtienen más faltas y puntos gratuitos

### 6.2 Interacciones Momentum × Estabilidad

#### momentum_stability_index
- **Propósito**: Detectar juegos explosivos con momentum fuerte pero baja estabilidad
- **Fórmula**: `momentum_score × (1 - quarter_variance)`
- **Por qué funciona**: Momentum fuerte + baja estabilidad = juegos impredecibles, de alta anotación

#### momentum_intensity_interaction
- **Propósito**: Respuesta de momentum a cambios de intensidad del juego
- **Fórmula**: `momentum_shift × intensity_drop_factor`
- **Por qué funciona**: Equipos con momentum pueden superar caídas de intensidad, llevando a remontadas

#### momentum_garbage_interaction
- **Propósito**: Comportamiento de momentum en escenarios de garbage time
- **Fórmula**: `momentum_shift × garbage_time_risk`
- **Por qué funciona**: Equipos con momentum pueden presionar más incluso cuando el juego está decidido

### 6.3 Interacciones Contexto × Rendimiento

#### clutch_garbage_interaction
- **Propósito**: Cómo los equipos clutch actúan cuando los juegos se vuelven sin sentido
- **Fórmula**: `clutch_performance × garbage_time_risk`
- **Por qué funciona**: Identifica equipos que "siguen jugando" vs equipos que "se rinden"

#### clutch_lead_interaction
- **Propósito**: Respuesta de rendimiento clutch al tamaño de ventaja
- **Fórmula**: `clutch_performance × current_lead`
- **Por qué funciona**: Equipos clutch buenos rinden mejor con ventajas grandes (escapan) o pequeñas (cierran)

#### clutch_time_interaction
- **Propósito**: Sensibilidad de rendimiento clutch al tiempo restante
- **Fórmula**: `clutch_performance × time_remaining_pct`
- **Por qué funciona**: Equipos clutch excelen cuando la presión de tiempo aumenta

### 6.4 Interacciones Turnover × Pace

#### turnover_pace_efficiency
- **Propósito**: Detección de caos en juegos de alto ritmo
- **Fórmula**: `turnovers_per_minute × pace_estimate`
- **Por qué funciona**: Alto ritmo + turnovers altos = oportunidades de transición y fast breaks
- **Variantes**: TO% × possessions_per_minute, TO × defensive_stops

#### to_pct_possessions_interaction
- **Propósito**: Impacto de tasa de turnovers en ritmo
- **Fórmula**: `turnover_percentage × possessions_per_minute`
- **Por qué funciona**: Mayores tasas de turnovers significan más posesiones para ambos equipos

#### to_defensive_stops_interaction
- **Propósito**: Eficiencia de creación de turnovers
- **Fórmula**: `turnovers_per_minute × defensive_stops`
- **Por qué funciona**: Equipos defensivos buenos crean más turnovers, llevando a más posesiones

---

## 7. Features Psicológicas Avanzadas

### 7.1 Psychological Momentum Cascade Index (PMCI)
- **Propósito**: Detectar espirales de momentum psicológico
- **Cálculo**: Cascada ponderada de eventos negativos
- **Fórmula**: Eventos negativos (déficit ≥10, FG%<40%, TOV>18) crean cascadas exponenciales
- **Parámetros**:
  - `pmci_neg_pm_threshold`: -10 (umbral de déficit de puntos)
  - `pmci_low_fg_pct`: 0.40 (umbral de porcentaje de campo)
  - `pmci_high_turnovers`: 18 (umbral de turnovers)
  - `pmci_streak_exponent`: 1.5 (amplificación de cascada)
- **Features**:
  - `home_pmci`, `away_pmci`, `diff_pmci`

### 7.2 Competitive Pressure Response Coefficient (CPRC)
- **Propósito**: Medir respuesta a la fuerza del oponente
- **Cálculo**: Rendimiento vs esperado basado en rating del oponente
- **Fórmula**: (Actual - Esperado) / Esperado, ajustado por fuerza del oponente
- **Parámetros**:
  - `cprc_expected_slope`: 0.3 (ajuste por punto de fuerza)
  - `cprc_response_cap_weak`: 1.2 (límite para oponentes débiles)
- **Features**:
  - `home_cprc`, `away_cprc`, `diff_cprc`

### 7.3 Tactical Adaptation Velocity (TAV)
- **Propósito**: Rastrear velocidad de ajustes estratégicos
- **Cálculo**: Tasa de recuperación después de cuartos pobres
- **Fórmula**: Contar mejoras después de malos cuartos, con bono de sostenibilidad
- **Parámetros**:
  - `tav_window`: 8 (partidos a analizar)
  - `tav_eff_threshold`: 1.5 (umbral de recuperación de eficiencia)
  - `tav_recovery_multiplier`: 1.3 (multiplicador de mejora)
- **Features**:
  - `home_tav`, `away_tav`, `diff_tav`

### 7.4 Energy Distribution Intelligence (EDI)
- **Propósito**: Medir gestión estratégica de energía
- **Cálculo**: Análisis de patrones de ritmo por cuartos
- **Fórmula**: Puntos por posesión por cuarto, con bono de FT en Q4
- **Parámetros**:
  - `edi_window`: 6 (partidos a analizar)
  - `edi_peak_q4_bonus`: 1.5 (bono de intensidad Q4)
  - `edi_close_game_margin`: 5 (umbral de juego cerrado)
- **Features**:
  - `home_edi`, `away_edi`, `diff_edi`

### 7.5 Power Law Scoring Volatility Index (PLSVI)
- **Propósito**: Detectar eventos de anotación explosivos
- **Cálculo**: Frecuencia de ráfagas de anotación de leyenda de potencia en Q4
- **Fórmula**: Eventos donde puntos Q4 exceden lo esperado en 30%
- **Parámetros**:
  - `plsvi_window`: 12 (partidos a analizar)
  - `plsvi_q4_positive_threshold`: 28 (umbral de ráfaga)
  - `plsvi_q4_negative_threshold`: 15 (umbral de colapso)
- **Features**:
  - `home_plsvi`, `away_plsvi`, `diff_plsvi`

---

## 8. Features de Transición de Cuarto

### 8.1 Aceleración y Momentum de Cuarto

#### quarter_acceleration_q2_q3
- **Propósito**: Detecta aceleración de anotación después del descanso de medio tiempo
- **Cálculo**: (q3total - q2total) / q2total
- **Lógica**: Captura "turno" o cambio de momentum después del medio tiempo
- **Salida**: Normalizado 0-1 (0.5 = neutral, más alto = aceleración)
- **Features**:
  - `quarter_acceleration_q2_q3`

#### quarter_momentum_shift
- **Propósito**: Mide dirección y magnitud del cambio de momentum
- **Cálculo**: sign(q3total - q2total) * abs(q3total - q2total) / halftime_total
- **Lógica**: Cambio de momentum firmado normalizado por total de medio tiempo
- **Salida**: Normalizado 0-1 (0.5 = neutral, más alto = momentum positivo)
- **Features**:
  - `quarter_momentum_shift`

#### split_efficiency_ratio
- **Propósito**: Compara eficiencia ofensiva a través del medio tiempo
- **Cálculo**: q3endtotal / halftime_total
- **Lógica**: Ratio de anotación post-medio tiempo vs pre-medio tiempo
- **Salida**: Normalizado 0-1 (0.5 = neutral, más alto = más eficiente post-medio tiempo)
- **Features**:
  - `split_efficiency_ratio`

### 8.2 Métricas de Volatilidad Contextual

#### quarter_variance_weighted
- **Propósito**: Varianza ponderada por importancia de cuarto para O/U
- **Cálculo**: α*Q4 + β*Q3 + γ*Q2 + δ*Q1 (pesos: 0.4, 0.3, 0.2, 0.1)
- **Lógica**: Q4 más importante para predicciones over/under
- **Salida**: Normalizado 0-1 (más alto = más volátil)
- **Features**:
  - `quarter_variance_weighted`

#### intensity_drop_acceleration
- **Propósito**: Combina caída de intensidad con varianza de cuarto
- **Cálculo**: intensity_drop_factor * quarter_variance
- **Lógica**: Multiplica dos características SHAP principales para análisis Q3/Q4
- **Salida**: Normalizado 0-1 (más alto = más aceleración desde caída)
- **Features**:
  - `intensity_drop_acceleration`

#### phase_stability_score
- **Propósito**: Estabilidad relativa a través de fases del juego
- **Cálculo**: 1 - (quarter_variance / halftime_total)
- **Lógica**: Detecta juegos "falsos" estables con baja varianza pero bajos totales de medio tiempo
- **Salida**: Normalizado 0-1 (más alto = más estable)
- **Features**:
  - `phase_stability_score`

---

## 9. Features de Head-to-Head (H2H)

### 9.1 Puntuación Total Promedio H2H
- **Propósito**: Anotación histórica en enfrentamientos
- **Cálculo**: Promedio de puntos totales en reuniones previas
- **Features**:
  - `h2h_avg_total_score`

### 9.2 Ritmo Promedio H2H
- **Propósito**: Tempo típico en enfrentamientos
- **Cálculo**: Promedio de ritmo en reuniones previas
- **Features**:
  - `h2h_pace_avg`

### 9.3 Promedio de Primera Mitad H2H
- **Propósito**: Patrones de anotación temprana en enfrentamientos
- **Cálculo**: Promedio de puntos de primera mitad en enfrentamientos
- **Features**:
  - `h2h_first_half_avg_total`

### 9.4 Ráfega de Segunda Mitad H2H
- **Propósito**: Aceleración de anotación tardía en enfrentamientos
- **Cálculo**: Diferencial segunda mitad - primera mitad
- **Features**:
  - `h2h_second_half_surge_avg`

### 9.5 Frecuencia de Remontadas H2H
- **Propósito**: Probabilidad de remontadas en enfrentamientos
- **Cálculo**: Frecuencia de remontadas en reuniones previas
- **Features**:
  - `h2h_comeback_freq`

---

## 10. Features de Mercado

### 10.1 Diferencia de Mercado Total
- **Propósito**: Diferencia entre línea de mercado y predicción del modelo
- **Cálculo**: línea_mercado - predicción_modelo
- **Features**:
  - `market_total_diff`

### 10.2 Ritmo Implícito vs Promedio
- **Propósito**: Ritmo esperado por mercado vs histórico
- **Cálculo**: ritmo_implícito_mercado / ritmo_historico_promedio
- **Features**:
  - `implied_pace_vs_avg`

### 10.3 Índice de Volatilidad de Línea
- **Propósito**: Medir volatilidad y movimiento de líneas de mercado
- **Cálculo**: Desviación estándar de movimientos de línea
- **Features**:
  - `line_volatility_index`

---

## 11. Features del Metamodelo

### 11.1 Features de CatBoost Meta-Model
- **Propósito**: Features específicamente diseñadas para el metamodelo CatBoost
- **Features**:
  - `halftime_total`: Puntos anotados al medio tiempo
  - `pace_halftime`: Ritmo estimado al medio tiempo
  - `efg_home/away`: Porcentaje de tiro efectivo
  - `to_rate_home/away`: Tasa de turnovers
  - `ft_rate_home/away`: Tasa de tiros libres
  - `home_lead`: Ventaja del equipo local
  - `is_close_game`: Indicador binario de juego cerrado
  - `momentum_score`: Puntuación de momentum del juego

### 11.2 Sistema de Safety Rails
- **Propósito**: Prevenir predicciones extremas y asegurar salidas razonables
- **Features**:
  - Límites globales (150-280 puntos)
  - Límites de desviación de proyección base
  - Soft clamp para ajustes extremos

---

## 12. Configuración y Parámetros

### 12.1 Rangos de EMA
```python
EMA_RANGES = {
    'short_term': 3,    # Rendimiento reciente
    'medium_term': 8,   # Tendencias de mediano plazo
    'long_term': 11     # Patrones de temporada completa
}
```

### 12.2 Parámetros de Features Avanzadas
Ubicados en `config.py` bajo `ADVANCED_FEATURE_PARAMS` con valores de calibración para:
- Umbrales de cascada PMCI
- Coeficientes de respuesta CPRC
- Ventanas de adaptación TAV
- Patrones de energía EDI
- Detección de volatilidad PLSVI

### 12.3 Controles de Ablación
- `ADVANCED_FEATURE_ABLATION`: Activar/desactivar features avanzadas
- `QUARTER_FEATURES_ABLATION`: Activar/desactivar features por cuarto
- `ENABLE_FEATURE_FILTERING`: Eliminar features inútiles

### 12.4 Configuración Específica por Liga

#### NBA
- **Línea base de anotación**: 220-230 puntos por juego
- **Multiplicador de ritmo**: 1.2 (liga de ritmo rápido)
- **Umbral de foul trouble**: >0.7 FTI (juego agresivo)
- **Detección de runs**: 7-9 puntos de ráfaga (alta anotación)

#### WNBA
- **Línea base de anotación**: 160-175 puntos por juego
- **Multiplicador de ritmo**: 0.8 (ritmo más lento)
- **Umbral de foul trouble**: >0.6 FTI (juego físico)
- **Detección de runs**: 5-7 puntos de ráfaga (anotación más baja)

#### NBL
- **Línea base de anotación**: 170-185 puntos por juego
- **Multiplicador de ritmo**: 0.98 (ritmo moderado)
- **Umbral de foul trouble**: >0.65 FTI (liga física)
- **Detección de runs**: 6-8 puntos de ráfaga (anotación balanceada)

---

## Impacto Predictivo

### Features Principales por Importancia
Típicamente, las features principales por importancia incluyen:
1. **Estado del juego en vivo** (halftime_total, q1_total)
2. **Rendimiento reciente** (features EMA de corto plazo)
3. **Métricas de ritmo** (live_pace_estimate, possessions)
4. **Features psicológicas avanzadas** (PMCI, CPRC)
5. **Factores de calendario** (schedule_analysis_score)

### Impacto por Categoría

#### Momentum Features
- **Capturan**: Tendencias de rendimiento a corto/largo plazo
- **Impacto**: Alto para juegos con equipos en racha
- **Sensibilidad**: Media a cambios recientes

#### Live Game Features
- **Capturan**: Estado actual y dinámica del juego
- **Impacto**: Muy alto para predicciones en tiempo real
- **Sensibilidad**: Alta a eventos del juego

#### Advanced Psychological Features
- **Capturan**: Factores intangibles y momentum psicológico
- **Impacto**: Moderado a alto en situaciones de presión
- **Sensibilidad**: Media a patrones de equipo

#### Interaction Features
- **Capturan**: Relaciones no lineales entre métricas
- **Impacto**: Alto para escenarios complejos
- **Sensibilidad**: Alta a condiciones específicas del juego

---

## Notas de Ingeniería de Features

### 1. Anti-Filtrado (Anti-Leakage)
- Todas las features pre-juego usan solo datos históricos (df.iloc[:index])
- Las features en vivo usan solo datos disponibles hasta el momento actual
- Estricta separación temporal para evitar sobreajuste

### 2. Normalización
- Features escaladas a [0,1] cuando es aplicable
- Manejo consistente de valores extremos
- Normalización por liga cuando es apropiado

### 3. Robustez
- Divisiones seguras y manejo de NaN en todo el sistema
- Valores por defecto para datos faltantes
- Validación de rangos y límites

### 4. Compatibilidad en Vivo
- Las features funcionan con datos parciales del juego
- Degradación graceful cuando datos están incompletos
- Cálculos eficientes para actualizaciones en tiempo real

### 5. Multi-Liga
- Parámetros ajustados para diferencias NBA/WNBA/NBL
- Configuraciones específicas por liga
- Normalización relativa a características de liga

---

## Conclusión

Este conjunto comprensivo de features captura métricas tradicionales de box score, análisis avanzado, factores psicológicos y dinámicas de juego en vivo para predicciones robustas de Over/Under. La combinación de features basadas en EMA, interacciones complejas y características psicológicas avanzadas proporciona una base sólida para predicciones precisas y contextualizadas.

El diseño modular y la configuración centralizada facilitan el mantenimiento y la extensión futura, mientras que la optimización de rendimiento asegura la escalabilidad para producción.