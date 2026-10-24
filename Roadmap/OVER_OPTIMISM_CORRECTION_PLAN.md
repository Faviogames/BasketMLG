> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_ROADMAP.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Plan de Corrección de Predicciones Optimistas de OVER

## Análisis del Problema

### Situación Actual
El sistema actual presenta un patrón de predicciones excesivamente optimistas para OVER, especialmente en partidos con equipos ofensivos. El MetaModeloInteligente (`core_meta_model.py`) actúa como un "Control de Calidad" que corrige estas predicciones en tiempo real, pero esta corrección es reactiva en lugar de preventiva.

### Patrón Identificado (Patrón A)
- **Modelo Base (XGBoost)**: Predicciones consistentemente optimistas para OVER
- **MetaModeloInteligente**: Correcciones drásticas (hasta -30%) para ajustar las predicciones
- **Resultado**: Sistema funcional pero con inestabilidad y correcciones extremas

## Estrategia Propuesta: Enfoque Dual

### 1. Mejora del Modelo Base (Prevención)
**Objetivo**: Reducir el optimismo en la fuente mediante features de "freno" que enseñen al modelo base a ser más realista.

### 2. Ajuste del MetaModelo (Refinamiento)
**Objetivo**: Convertir el MetaModeloInteligente de un "Corrector Drástico" a un "Refinador Sutil" que realice ajustes menores y más precisos.

## Diseño de Features de "Freno"

### Categoría 1: Features de "Fricción" (El Oponente Importa)

#### `opponent_pace_ema_5`
- **Descripción**: Pace promedio de los oponentes enfrentados en últimos 5 partidos
- **Por qué funciona**: Detecta si el pace de un equipo está "inflado" por oponentes lentos
- **Implementación**: 
```python
def calculate_opponent_pace_ema_5(team_id, games, current_date):
    recent_games = get_recent_games(team_id, games, current_date, 5)
    opponent_paces = []
    for game in recent_games:
        opponent_id = get_opponent_id(team_id, game)
        opponent_pace = get_team_pace(opponent_id, games, current_date)
        opponent_paces.append(opponent_pace)
    return exponential_moving_average(opponent_paces, alpha=0.5)
```

#### `opponent_defensive_efficiency_ema_5`
- **Descripción**: Rating defensivo promedio de oponentes en últimos 5 partidos
- **Por qué funciona**: Enseña al modelo sobre la interacción ataque vs defensa
- **Impacto esperado**: Reducción del optimismo cuando ofensivos fuertes enfrentan defensas élite

### Categoría 2: Features de "Enfriamiento" (Recencia vs. Histórico)

#### `diff_ema_points_scored_3_vs_11`
- **Descripción**: Diferencial entre EMA de 3 partidos vs EMA de 11 partidos
- **Fórmula**: `EMA_3 puntos - EMA_11 puntos`
- **Por qué funciona**: Detecta rachas de mal rendimiento reciente
- **Señal**: Valores negativos indican "enfriamiento" ofensivo

#### `diff_ema_ts_pct_3_vs_11`
- **Descripción**: Diferencial de True Shooting % entre EMA de 3 y 11 partidos
- **Por qué funciona**: Detecta colapsos de eficiencia antes de que se reflejen en puntos
- **Ventaja**: Más sensible que los puntos para detectar problemas ofensivos

### Categoría 3: Features de "Inestabilidad" (Volatilidad)

#### `offensive_rating_variance_last_5`
- **Descripción**: Varianza del rating ofensivo en últimos 5 partidos
- **Por qué funciona**: Equipos volátiles son malos candidatos para OVER
- **Lógica**: Alto promedio + Alta varianza = Predicción neutral

#### `scoring_consistency`
- **Descripción**: Ya existe en config.py (línea 179) pero necesita mayor peso
- **Fórmula**: `1 / (1 + CoeficienteDeVariación)`
- **Acción**: Asegurar alta importancia en training.py

## Plan de Implementación

### Fase 1: Implementación de Features (Core)
1. **Modificar `core/features.py`**:
   - Añadir las 5 nuevas features de "freno"
   - Implementar cálculos EMA y varianza
   - Integrar con el sistema existente de features

2. **Actualizar `config.py`**:
   - Añadir configuraciones para nuevas features
   - Ajustar pesos para dar más importancia a features de consistencia

### Fase 2: Reentrenamiento del Modelo Base
1. **Modificar `core/training.py`**:
   - Incluir nuevas features en el dataset de entrenamiento
   - Ajustar hiperparámetros para optimizar con nuevas features
   - Validar importancia de features para asegurar impacto

2. **Generar nuevo modelo**:
   - Reentrenar XGBoost con dataset enriquecido
   - Comparar importancia de features vs modelo anterior

### Fase 3: Ajuste del MetaModelo
1. **Modificar `metamodel/meta_model.py`**:
   - Reducir umbrales de corrección (de -30% a -10% máximo)
   - Ajustar lógica para correcciones más sutiles
   - Enfocarse en ajustes finos而不是 correcciones drásticas

2. **Actualizar `metamodel/context_analyzer.py`**:
   - Incorporar señales de las nuevas features
   - Refinar lógica de detección de optimismo

### Fase 4: Validación y Pruebas
1. **Pruebas A/B**:
   - Comparar sistema actual vs propuesto
   - Medir reducción en correcciones del MetaModelo
   - Evaluar precisión general de predicciones

2. **Métricas de éxito**:
   - Reducción del 50% en correcciones mayores al 15%
   - Mejora del 10% en precisión de predicciones de OVER
   - Reducción de la volatilidad en predicciones

## Impacto Esperado

### Cambios en el Comportamiento del Sistema

#### Antes (Sistema Actual):
```
Modelo Base: Predicción OVER = 220 puntos (+15%)
MetaModelo: Corrección = -30%
Resultado Final: 154 puntos (subestimado)
```

#### Después (Sistema Propuesto):
```
Modelo Base: Predicción OVER = 200 puntos (+5%)
MetaModelo: Corrección = -5%
Resultado Final: 190 puntos (preciso)
```

### Beneficios Clave:
1. **Estabilidad**: Menos correcciones drásticas
2. **Precisión**: Predicciones más realistas desde el inicio
3. **Eficiencia**: MetaModelo enfocado en ajustes finos
4. **Confiabilidad**: Sistema más predecible y estable

## Archivos a Modificar

### Core del Sistema:
1. `core/features.py` - Implementación de nuevas features
2. `core/training.py` - Reentrenamiento con nuevas features
3. `config.py` - Configuración de pesos y parámetros

### Metamodelo:
1. `metamodel/meta_model.py` - Ajuste de lógica de corrección
2. `metamodel/context_analyzer.py` - Integración de nuevas señales

### Validación:
1. `tests/test_metamodel_quality.py` - Pruebas de calidad
2. `tests/test_reality_check_fix.py` - Validación de correcciones

## Cronograma Sugerido

### Semana 1:
- Implementar features de "freno" en `core/features.py`
- Actualizar configuraciones en `config.py`

### Semana 2:
- Modificar `core/training.py` para nuevas features
- Reentrenar modelo base XGBoost

### Semana 3:
- Ajustar MetaModelo para correcciones más sutiles
- Implementar pruebas de validación

### Semana 4:
- Ejecutar pruebas A/B
- Análisis de resultados y ajustes finales

## Métricas de Monitoreo

### KPIs Principales:
1. **Porcentaje de correcciones > 15%**: Objetivo < 10%
2. **Precisión en predicciones OVER**: Objetivo +10%
3. **Varianza de predicciones**: Objetivo -20%
4. **Importancia de features de "freno"**: Objetivo > 15% del peso total

### Monitoreo Continuo:
- Distribución de magnitudes de corrección
- Importancia de features en modelo base
- Precisión por categorías de partidos

## Conclusión

Este enfoque dual ataca el problema del optimismo desde dos ángulos complementarios:
1. **Prevención**: Features de "freno" en el modelo base
2. **Refinamiento**: MetaModelo más sutil y preciso

El resultado esperado es un sistema más estable, preciso y eficiente que requiere menos intervenciones drásticas y produce predicciones más realistas de manera consistente.