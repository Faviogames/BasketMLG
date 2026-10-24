# Módulo 06: core/features.py

Fecha: 16 de octubre de 2026. Estado: revisión local terminada; sin cambios funcionales. Se leyó el archivo completo. Se ejecutaron 13 escenarios aislados, con funciones reales extraídas mediante AST, pandas/numpy y datos sintéticos. No se entrenó ni cargó ningún modelo ni se consultó SofaScore.

## Responsabilidad y contratos

El módulo prepara el dataset histórico de entrenamiento y contiene cálculos que también utiliza live. No es código prescindible al retirar la interfaz manual.

- `calculate_features(league_data_raw)` delega el procesamiento inicial a `process_raw_matches`, ordena por fecha y construye históricos anteriores a cada fila para ambos equipos. Omite las primeras cinco filas y exige al menos dos partidos por equipo; los rolling de cinco requieren cinco antecedentes.
- Calcula posesiones, ritmo, ORTG, TS%, momentum, contexto, medias/EMA, H2H, mercado y familias avanzadas PMCI/CPRC/TAV/EDI/PLSVI.
- Por partido elegible añade dos muestras: final de Q3 y descanso. Ambas comparten históricos y objetivo final. No genera muestras específicas de Q1 ni de minutos intermedios. Esto no demuestra por sí solo que el sistema deba dejar de cubrir esos momentos; exige contrastar entrenamiento e inferencia.
- Devuelve un DataFrame amplio con características, objetivo `final_total_score`, nombres, resultados finales y `raw_match`. **La presencia del resultado final en ese DataFrame no prueba que entre como predictor**: el consumidor selecciona columnas.
- `core/training.py:833` selecciona la intersección con `HISTORICAL_FEATURES_TO_USE`. El listado `FILTERED_FEATURES_TO_USE` en este módulo sólo calcula un recuento para imprimir; no recorta el DataFrame devuelto.
- `analysis/live_mode.py` comparte helpers rolling/EMA, momentum, H2H y tendencias; `core/live_features.py` también importa cálculos de aquí. Hay importación local desde `calculate_features` para evitar el ciclo.

La entrada procesada necesita fecha, equipos, marcadores, total y `raw_match`. Hay campos opcionales de estadísticas finales, parciales y box por cuarto. Varios campos ausentes pasan a cero, otros a NaN: el contrato de ausencia no es uniforme.

## Hallazgos reproducidos

Actualización documental: `feature_documentation.md` y `NEW_FEATURES_AND_MODELS_UPDATE.md` confirman que separar métricas directas de familias EMA fue una decisión deliberada. **No se propone aplicar EMA a todo.** F06-03 trata de sustituir una serie por un agregado repetido y F06-04 de omitir métricas en el ensamblaje, no de que esa separación sea un error. Las guías de mercado contienen definiciones contradictorias entre sí: conservar fórmula, nombres y versión del artefacto como contrato a comprobar antes de reparar.

### F06-01 — Variables de mercado consultan nombres equivocados

En `calculate_features`, aproximadamente líneas 1203–1226, `home_stats_5` usa claves como `avg_points_scored_last_5` y `avg_pace_last_5`. El bloque de mercado busca `points_scored` y `pace`, por lo que obtiene sus valores de respaldo.

Con línea 215 y antecedentes completos, el resultado es `market_total_diff = -215`, porque suma 0 + 0. `implied_pace_vs_avg` utiliza ritmo 98 aunque haya rolling calculado. No es una ausencia de la cuota en este escenario: el dato de mercado sí existe.

Corregirlo después puede alterar la distribución de variables respecto de modelos guardados; antes de aplicar habrá que comprobar qué columnas usan y si necesitan reentrenamiento. El bloque de mercado histórico sí existe aquí, a diferencia de su ausencia en la caché live registrada en V05-01.

### F06-02 — Ritmo histórico con una escala distinta

En los bloques de enriquecimiento histórico, líneas 1017–1063, `pace = possessions * 2`. Con FGA 80, OREB 10, TOV 12 y FTA 20, las posesiones son 90.8 y el ritmo guardado es 181.6.

En cambio, `calculate_pace(90.8, 90.8)` utiliza el promedio de posesiones, dentro de un rango 80–120. Hay dos significados distintos para el mismo nombre dentro del módulo. Las medias y EMA históricas heredan la escala duplicada. No se ha cuantificado su efecto en los modelos reales.

`calculate_pace` declara posesiones por 48 minutos y acepta `minutes_played`, pero ignora ese argumento; 24 y 48 minutos producen el mismo resultado. Su implementación comenta que recibe partidos completos: no asumir que cada llamada necesita normalización temporal. Antes de cambiarlo, definir si se quiere ritmo normalizado o posesiones por partido y cómo se aplica a NBA/WNBA.

### F06-03 — La EMA de victorias y margen pierde el orden de los partidos

Líneas 1085–1106: se crean `win` y `plus_minus` por partido, pero `calculate_momentum_metrics` devuelve sus medias y el bucle posterior las escribe en todas las filas del historial. Después se calcula EMA sobre esas columnas constantes.

En cinco resultados `[0, 1, 0, 1, 0]`, la EMA corta producida vale 0.4; la EMA real por partido con span 3 vale 0.3125. Varias escalas de estas métricas se vuelven iguales. El defecto también existe en el ensamblaje live revisado antes; no es únicamente una diferencia entre entrenamiento e inferencia.

La sustitución de `plus_minus` afecta además a cálculos posteriores que intentan interpretar márgenes individuales, como PMCI/EDI y contexto. Las demás métricas agregadas pueden estar pensadas como snapshots; no convertir todas a series temporales automáticamente.

### F06-04 — Quince variables directas se calculan y se pierden al ensamblar

`calculate_momentum_metrics` produce, entre otras, `win_rate`, `avg_plus_minus`, consistencia, tendencias, fatiga y aceleración. Se añaden como columnas al historial, pero el ensamblaje copia rolling de otras columnas y EMA de `MOMENTUM_STATS_COLS`/contexto; no inserta esas quince métricas con sus nombres `home_*`, `away_*`, `diff_*`.

El escenario completo sintético confirmó que no aparecen `home_win_rate`, `home_back_to_back_fatigue` ni las otras trece de esa primera familia. El problema también existe en live. **No todas las variables directas faltan**: TS%, selección de tiro, calendario, consistencia de ritmo, familias avanzadas, mercado y señales históricas por cuarto sí tienen bloques explícitos.

Esto explica exclusiones estructurales del entrenamiento actual, pero todavía no identifica columna a columna las 124 imputaciones de la ejecución real ni prueba cómo se creó cada modelo guardado.

### F06-05 — Eficiencia de segunda mitad siempre neutra con entradas válidas

`safe_mean` (línea 27) sólo procesa objetos con `dropna`; para una lista devuelve 0. `calculate_second_half_efficiency` (líneas 2031–2032) le entrega listas de puntos por mitad. Sus dos medias quedan a cero y termina en 0.5.

Cinco partidos con primera mitad 40 y segunda mitad 60 devuelven 0.5; según la fórmula documentada de esa función correspondería 0.75. No se ha corregido el helper compartido: ampliar su contrato requiere revisar consumidores. Esta señal, además, está entre las directas no copiadas al dataset principal; el error del helper y su alcance real son cuestiones distintas.

### F06-06 — Los triples canónicos no llegan al cálculo de ritmo de tiro

`calculate_enhanced_shooting_rhythm` (aproximadamente línea 2854) busca `3_point_field_goals_made` y `3_point_field_g_attempted`, mientras el historial construido aquí usa `3point_field_goals_made` y `3point_field_goals_attempted`.

Con idénticos valores, el formato canónico devuelve 0.8000 y los alias antiguos 0.9250: con el primero el componente de triples usa su respaldo. `calculate_shot_selection_intelligence` sí utiliza las claves canónicas; no trasladar este hallazgo a todas las métricas de tiro.

### F06-07 — Rama opcional de estadísticas avanzadas falla con NameError

En `calculate_advanced_stats_for_match`, línea 651, se lee `match.get('date')` aunque el parámetro se llama `match_data`. Si se pasan `all_matches` y `league_averages` se produce `NameError` antes de imputar.

Se reprodujo con un box `MATCH` válido. No se encontraron llamadas a esta función en los consumidores Python inspeccionados de `core`, `analysis`, `scripts`, `tests` y `main.py`; su fallo no explica por sí solo el flujo live actual. No borrarla sin inventario completo y confirmación de usos externos.

### F06-08 — H2H trata parciales ausentes como mediciones a cero

Con cinco enfrentamientos que sólo tienen marcador final, `calculate_h2h_features` devuelve primera mitad 0, incremento de segunda mitad 0 y frecuencia de remontada 0. La ausencia no queda representada como NaN. Una cobertura parcial también diluye la frecuencia porque su denominador incluye todos los encuentros.

Debe definirse después qué mínimos de cobertura son necesarios y cómo se informa la falta de datos. No inventar parciales para reducir imputación.

## Otras observaciones por inspección y decisiones pendientes

- **Separación de muestras del mismo partido:** dos filas comparten antecedentes y objetivo. El fragmento de `training.py:855` usa división aleatoria por fila. Hay riesgo de que el mismo partido quede a ambos lados y, al seleccionar sólo características históricas, de muestras idénticas entre conjuntos. Revisar entrenamiento completo, OOF y validación antes de cuantificar contaminación o rendimiento. No se ejecutó un entrenamiento ni se evaluaron sus métricas.
- **Descanso previo al partido actual:** fatiga y análisis de calendario reciben únicamente el historial, sin fecha del encuentro a predecir. Calculan intervalos entre partidos anteriores y densidad respecto del último histórico, por lo que no pueden distinguir una vuelta tras un día de una vuelta tras varias semanas con idéntico historial. Confirmar si deben medir carga reciente o descanso actual antes de cambiar el contrato.
- **Umbrales de liga:** TAV/EDI usan 12 minutos en respaldos por puntos/minuto, pero posesiones cuando hay box. Mezclan dos unidades bajo umbrales comunes. Otros benchmarks de tiro/volumen/ritmo son fijos. WNBA dura menos; no equiparar esa diferencia legítima a un error por sí misma. Revisar definición de cada señal y procedencia NBA/WNBA de sus parámetros.
- **Clutch histórico aproximado:** el camino de `calculate_clutch_time_performance` que usa margen final resta los puntos propios de Q4 al total rival. Si el total propio coincide con la suma de parciales, la diferencia estimada al inicio de Q4 se reduce algebraicamente al margen final absoluto. No mide el margen real al terminar Q3. Este código estudia antecedentes de Q4; conservar esa información histórica no equivale a recomendar apuestas en Q4 tardío ni a añadir prórrogas.
- **Proxies con nombres más fuertes que su cálculo:** `comeback_ability` cuenta victorias por margen >5, no remontadas demostradas; `opponent_pace_ema_5` es una media simple del ritmo de partidos propios; `opponent_defensive_efficiency_ema_5` es la media del ORTG propio, reconocido como proxy en comentarios. PLSVI combina magnitud absoluta de colapsos y explosiones: mide eventos extremos, no su dirección. Revisar semántica con el usuario antes de sustituir fórmulas.
- **Disponibilidad de tiros libres:** el historial principal no copia `free_throws_made`; `calculate_ft_rate_combined` suele ir al proxy pese a disponer potencialmente del dato de origen. Si recibe datos completos, calcula acierto FT y lo ajusta/capa, mientras la documentación habla de proporción de puntos/estrategia de faltas. Requiere acordar qué representa la variable.
- **Datos ausentes:** faltas, robos, asistencias y algunos tiros se inicializan a cero; varias señales aceptan esos valores como observados. Las señales por cuarto con inputs NaN suelen devolver NaN, lo cual permite distinguir ausencia. Contrastar cobertura real al revisar `data_processing`.
- **Cronología:** el corte `df.iloc[:index]` excluye la fila actual y posteriores, y H2H/fuerza de equipos también reciben ese corte. No obstante, hay que revisar fechas iguales, integridad del histórico y las medias de liga que entrega el procesamiento inicial. Este módulo no demuestra que esas medias estén libres de información futura.
- **Coste:** reconstruye históricos y vuelve a calcular ratings por cada partido. Existe trabajo repetido; no se midió un cuello de botella y no se propone cachear antes de fijar contratos temporales.

## Código candidato a revisión, sin borrados autorizados

- Imports sin uso en este archivo: `QUARTER_SPECIFIC_COLS`, `PRE_GAME_FEATURES`, `QUARTER_FEATURES_ABLATION`, `extract_quarter_data_for_alerts`, `calculate_league_averages`, `calculate_team_league_averages`; los imports locales `calculate_enhanced_live_features` y `calculate_live_interaction_features` tampoco se llaman desde `calculate_features`.
- El comentario «FILTERED_FEATURES_TO_USE activado» no corresponde a un filtrado de salida; el recuento sí se imprime y entrenamiento hace su propia selección.
- `cols_to_avg` repite `possessions`, `pace` y `ortg` porque se escriben explícitamente y también vienen de `ADVANCED_STATS_COLS`. Produce trabajo redundante con claves de diccionario iguales.
- `get_ema_stats` tiene etiqueta legacy, pero lo usan tanto entrenamiento como caché live. **Conservarlo hasta diseñar una migración del contrato.**
- Helpers de tendencia/proyección y `apply_blowout_adjustment` tienen referencias desde otros módulos: no declararlos obsoletos por no ver una llamada interna.

## Comprobaciones y límites

Script: `.diagnostics/audit_features.py`. Los 13 escenarios reproducen comportamientos actuales, incluidos defectos, y dos controles positivos sobre EMA/cooling y exclusión del resultado actual de su propio rolling. La ejecución del ensamblaje usa todas las funciones reales de este módulo, los conversores `safe_int`/`safe_float` reales, un doble de procesamiento raw y cinco dobles de helpers live. No valida el parser, las fórmulas de esos helpers, los modelos guardados ni la precisión predictiva.

SHA-256 del archivo de producción antes y después: `d38e961d6882805fd083a3503471b91ef8a0b6cf60c5cbff130625017a8fdf4b`. Sin cambios.

Prioridad futura: definir el contrato compartido entrenamiento/live y comprobar artefactos antes de reparar nombres, escalas o fórmulas. Cambiar estas variables puede exigir reentrenamiento y afectar la interpretación de registros anteriores. No se ha autorizado esa intervención.

Siguiente módulo sugerido: `core/data_processing.py`, para conocer qué datos llegan realmente, cómo se normalizan, cómo se imputan y cómo se construyen medias históricas/de liga. Parar aquí hasta que el usuario autorice continuar.
