# Módulo 04: core/live_api.py

Fecha: 16 de octubre de 2026. Estado: revisión local terminada, sin cambios funcionales. Se leyó el archivo completo, distinguiendo cliente Playwright, cliente de extensión y procesador. No constituye una validación empírica de los modelos ni una auditoría de las dependencias delegadas.

## Arquitectura y recorridos

| Componente | Papel actual |
| --- | --- |
| `safe_divide` | Utilidad para divisiones con respaldo |
| `SofaScoreClient` | Cliente Playwright anterior y base que contiene parsers, caché, extracción de reloj y normalización de eventos |
| `BrowserExtensionClient(SofaScoreClient)` | Transporte principal actual; sustituye consultas de navegador por solicitudes a `BrowserBridge` y conserva lógica heredada |
| `LiveProcessor` | Selección por compatibilidad, seguimiento, señales, llamada a inferencia, estabilidad, presentación y exportación |

Recorrido de producción desde el menú: `LiveProcessor` crea **siempre** `BrowserExtensionClient`; `get_available_games` enriquece el listado; `monitor_game` obtiene datos, comprueba si debe recalcular, genera señales y llama a `analysis.live_mode.get_predictions_with_alerts`. Esa función ya devuelve una predicción con metamodelo/salvaguardas. El monitor registra el resultado, aplica estabilidad, lo muestra y espera hasta la siguiente consulta.

Existe un segundo recorrido `process_game_update`, usado por al menos un test, pero no por el menú actual. No equivale al bucle de producción y contiene incompatibilidades concretas.

## Contratos, dependencias y efectos

- `BrowserBridge`: inicializar/cerrar conexión local y `request(path)` async. Su implementación se auditará aparte; esta revisión no repite la instalación ni hace pruebas live.
- `core.sofascore_session`: opciones y persistencia de sesión del cliente Playwright, no de las consultas de extensión.
- `analysis.live_mode`: `build_pre_game_cache`, `get_predictions_with_alerts`, y un wrapper de ajustes contextuales para pruebas. **Retirar la interfaz manual no autoriza borrar estas funciones compartidas.**
- `metamodel.prediction_tracker.PredictionTracker`: persistencia delegada de predicciones; no se llamó al tracker real durante la auditoría.
- Imports adelantados `aplicar_metamodelo_ai` y `calculate_detailed_quarter_metrics`: sin uso interno encontrado, pero importan dependencias antes del import perezoso de la función de predicción. Playwright es opcional por `try/except ImportError`.
- Bundle utilizado: al menos `league_name`, `team_names` en el constructor y el resto del contrato de inferencia en el módulo delegado. Esta clase no valida el bundle completo.
- Evento bruto: equipos, marcadores, `status` y `time`. Estadísticas brutas: periodos → grupos → `statisticsItems`.
- Estadísticas interpretadas: claves planas `home_*`/`away_*`, como `home_field_goals_attempted`; no son el JSON bruto con envoltorio `statistics`.
- `live_data`: `q_scores` Q1–Q4, `game_finished`, contexto, estadísticas y reloj. El reloj añade minutos, periodo, restantes, ritmo y final proyectado.

Efectos reales del módulo: consultas de navegador o puente según cliente, cachés/estado en memoria, sesión persistente solo en el cliente anterior y registro de predicciones mediante tracker. No entrena modelos.

## Hallazgos en la ruta actual, reproducidos sin red

### A04-01 — La decisión de recalcular ignora todo Q4; alcance aclarado posteriormente

Prioridad por concretar para el tramo inicial de Q4. Referencia: `core/live_api.py:2382`.

`_should_update_prediction` suma exclusivamente Q1–Q3 para comparar marcadores. Una vez terminado Q3, los nuevos puntos de Q4 no superan ningún umbral, aunque cambie mucho el total. Con 25 puntos añadidos en Q4 se obtuvo «Cambio mínimo no alcanzado (0 < 3 pts)». El monitor puede seguir recibiendo datos y mantener la predicción sin recalcularla durante Q4.

La omisión también está en run detector, contexto alternativo, modificadores FTI y algunos helpers de momentum. No se corrige todavía: habrá que acordar qué total y estado se usan dentro del tramo permitido, manteniendo las prórrogas fuera del sistema y sin modificar indiscriminadamente reglas de estabilidad.

**Aclaración posterior del usuario:** la cobertura prevista es Q1–Q3 y solo en ocasiones el inicio/primera mitad de Q4; no se recomiendan entradas en Q4 tardío. Por tanto, no se declara un defecto la ausencia de nuevas recomendaciones durante ese tramo excluido. La observación técnica sigue siendo válida para **todo** Q4, incluido el tramo inicial que puede estar permitido: habrá que distinguir si se pretende recalcular allí o conservar una predicción anterior. No ampliar cobertura al corregir o limpiar código.

### A04-02 — Estados fuera de alcance y tiempo restante al final

Referencia: `core/live_api.py:982`, especialmente líneas 999–1002. La ausencia de soporte de prórrogas es intencionada, según aclaración posterior del usuario.

Se limita `current_quarter` a `total_periods` **antes** de comprobar si es mayor, por lo que `is_overtime` no puede resultar verdadero. Con 50 minutos jugados se devuelve Q4 sin OT. En un partido finalizado justo a los 48 minutos, el módulo de la división devuelve cero y el cálculo asigna 12 minutos restantes.

También hay ambigüedad en los límites exactos entre periodos/descanso: se infiere el cuarto exclusivamente a partir del tiempo jugado, sin resolver primero el estado del evento. No se probó contra un partido real en prórroga en esta auditoría.

**Alcance corregido:** no proponer añadir cálculos ni predicciones de prórroga. Separar esa función excluida del reconocimiento de un estado no admitido: presentarlo como Q4 con minutos regulares restantes puede inducir a interpretar que el partido aún está dentro de cobertura. La política al salir del tramo permitido (detener, mostrar solo datos o marcar la predicción como anterior) se decidirá con el usuario antes de modificar comportamiento.

### A04-03 — Error de unidades en el final proyectado y supuestos de duración

Prioridad media/alta para los consumidores de ese campo. Referencia: `core/live_api.py:1030`.

El ritmo ya está expresado en puntos/minuto, pero se multiplica por 48 y se vuelve a dividir por 60. Ejemplo reproducido: 100 puntos en 20 minutos producen `projected_final_score=4`, frente a 240 si se proyecta a 48 minutos. Este campo pasa al contexto `projected_final`; no se afirma que equivalga a la predicción final del modelo ni que todos sus consumidores lo utilicen.

La proyección fija 48 minutos aunque `periodLength` indique periodos de diez minutos. El helper de estimación de minutos (`3171`) también asume 12 por cuarto y 24 al descanso. La calibración por liga y el manejo del tiempo real deben revisarse conjuntamente antes de modificar fórmulas.

### A04-04 — WNBA se clasifica como NBA

Prioridad alta para selección por liga. Referencias: `core/live_api.py:1329`, `core/live_api.py:2680`.

`'NBA' in 'WNBA'` es verdadero y el detector evalúa NBA primero. Una entrada de torneo WNBA se devuelve como NBA. Además, las comprobaciones por subcadena consideran NBA y WNBA plenamente compatibles. Ambos resultados se reprodujeron.

Esto no se resuelve retirando WNBA ni sus modelos. Hay que acordar normalización explícita y conservar la cobertura de ligas.

### A04-05 — La compatibilidad permite equipos que el modelo no conoce

Prioridad media, criterio de producto que requiere decisión. Referencias: `core/live_api.py:1926`, `core/live_api.py:2696`, `core/live_api.py:2763`.

Los equipos desconocidos reciben 0.4 en lugar de exclusión. Con liga NBA y dos equipos ajenos al bundle se reprodujo un score 0.66, por encima del umbral 0.5. Por tanto, «juego válido» no significa que ambos equipos estén en los históricos. El matching por subcadenas también puede producir coincidencias parciales.

No se ha demostrado que este caso explique la imputación del partido mostrado por el usuario. Antes de cambiarlo hay que decidir si los equipos desconocidos se rechazan o se admiten con una política explícita de respaldo.

### A04-06 — `game_info` permanece con el marcador y estado iniciales

Prioridad alta de coherencia. Referencias: `core/live_api.py:2014`, `core/live_api.py:2251`.

`monitor_game` recibe `live_data` nuevos, pero no renueva el objeto inicial con sus marcadores/estado. `_show_live_update` prefiere los `home_score`/`away_score` iniciales; adaptación de intervalos y modificadores por cuarto también consultan ese estado. Se reprodujo que las nuevas puntuaciones de Q3 no cambian un `game_info` que continúa en Q2 y con 50–50.

La extracción del reloj y estadísticas puede estar actualizándose correctamente mientras otra parte del contexto sigue antigua. La alternativa de sumar marcadores en la vista tampoco incluye Q4. No se debe interpretar un marcador visual repetido como prueba de que el proveedor haya dejado de actualizar.

### A04-07 — Exportación anterior a la estabilidad

Prioridad alta para trazabilidad/evaluación. Referencias: `core/live_api.py:2139`, `core/live_api.py:2167`.

Se exporta `final_prediction` antes de llamar a `_apply_prediction_stability`. En una secuencia simulada de resultados 200 y 220, el segundo registro guardó 220, mientras la predicción retenida y enviada a presentación fue 204. Las futuras evaluaciones pueden medir una cifra distinta de la mostrada al usuario.

Antes de repararlo se debe definir qué etapas se registran (salida de metamodelo, salvaguarda, estabilidad, valor mostrado), conservar los registros antiguos y evitar cambiar retrospectivamente su significado.

### A04-08 — La etiqueta de base y la confianza exportada no reflejan sus entradas

Prioridad media/alta de trazabilidad. Referencias: `core/live_api.py:2108`, `core/live_api.py:2129`, `core/live_api.py:2144`, `core/live_api.py:2279`.

La primera salida de `get_predictions_with_alerts` es ya final, pero se llama `base_prediction`, se guarda como `model_prediction` y se presenta como «Base Model (XGBoost)». El resultado original está disponible por contrato en `alerts_data['raw_prediction']`, pero se ignora. `meta_adjustment` se fuerza a cero.

La confianza del tracker se fija en 0.75, sin utilizar el ajuste de calidad recibido. Se reprodujo una salida final 220 con `raw_prediction=180` y `confidence_score=.25`: se registraron base 220 y confianza .75. Esto confirma que el 75 % de registros y el 80 % de listado (L03-02) son respaldos distintos, no validación de precisión.

### A04-09 — La línea elegida no llega a la llamada de inferencia por esa ruta

Prioridad media, contrato de mercado. Referencia: `core/live_api.py:2108`; consumidor consultado: `analysis/live_mode.py:1612`.

El monitor recibe `bookie_line`, la usa al mostrar y registrar, pero no la pasa al predictor ni la añade a `trained_data`. El consumidor busca la línea en `trained_data` o características. Se reprodujo una línea 215 en tracker, ausente en los argumentos/datos de las dos llamadas de inferencia simuladas.

No se afirma que toda predicción deba depender de la cuota. Debe decidirse qué etapas necesitan ese dato y cómo afecta a características de mercado/probabilidades, antes de cambiar los cálculos.

### A04-10 — Validar nombres de estadísticas no valida sus valores

Prioridad media/alta de calidad de datos. Referencias: `core/live_api.py:1415`, `core/live_api.py:1801`.

El filtro de la extensión exige categorías por nombre, pero el parser principal lee `home`/`away`. No utiliza `homeValue`/`awayValue` y `homeTotal`/`awayTotal`, aunque el parser anterior sí tenía soporte de esas claves. Con nombres completos y solo valores numéricos nativos se aceptó una respuesta plana con ceros en vez de los valores suministrados. Incluso nombres sin valores de ningún tipo producen un diccionario no vacío de ceros y pasan el control.

El formato textual fraccional usado en las pruebas previas sí funciona. No se ha demostrado que las respuestas reales del usuario carezcan de él. También hay fallos de recuperación de fracciones mal formadas: tras `ValueError` no se procesa de nuevo el valor, y un tipo inesperado en el campo visitante puede abortar el parser entero. Estos últimos caminos se detectaron por lectura, no con pruebas adicionales de todas las combinaciones.

## Recorridos alternativos/anteriores: defectos separados

### A04-11 — Método duplicado y fallback que no existe

Referencias: `core/live_api.py:830`, `core/live_api.py:1079`, `core/live_api.py:1163`.

`SofaScoreClient.get_game_statistics` se define dos veces; Python conserva la segunda. La primera ruta, que usa `_parse_statistics`, queda eclipsada. La segunda llama a `_apply_intelligent_fallback`, método que no existe en la clase ni sus bases; se reprodujo que esa rama devuelve `None` tras capturar el error.

La extensión sobrescribe este método y no pasa por ese fallback. `_basic_fallback` contiene valores ficticios y reemplaza ceros válidos (se reprodujeron FT 0→5 y pérdidas 0→8), pero su única llamada interna está después de un `return` y es inalcanzable. No atribuir esos números al flujo de extensión actual.

### A04-12 — `process_game_update` usa una variable antes de asignarla

Referencia: `core/live_api.py:2587`.

Cuando encuentra parciales frescos en el listado, hace `q_scores.update` antes de definir `q_scores`. Se reprodujo `UnboundLocalError`, capturado para devolver `None`. El menú actual usa `monitor_game`, por lo que este fallo no invalida el funcionamiento live ya confirmado.

### A04-13 — El recorrido alternativo descarta estadísticas ya parseadas

Referencias: `core/live_api.py:2590`, `core/live_api.py:2593`.

Si se evita la rama anterior, el método espera `current_stats['statistics']`, aunque el cliente devuelve estadísticas planas. Genera `parsed_stats={}` y no pasa `current_stats` a las señales del predictor. Reproducido con una actualización alternativa sin eventos en listado.

Además reconstruye parciales desde `game_info`, ignorando los `q_scores` que ya venían de `live_data`. Su fallback reparte un 25 % del marcador a Q1 y Q2 y deja los demás a cero, de modo que no conserva el total. Antes de mantener o retirar esta API, revisar su consumidor de test (`tests/test_feature_imputation_fix.py`); una prueba sobre esta ruta no certifica la ruta del menú.

### A04-14 — Predictor dummy incompatible con sus llamadores

Referencia: `core/live_api.py:1901`.

Si falla el import del predictor real, el fallback devuelve un total fijo 200, pero su firma solo acepta cuatro argumentos. Ambos llamadores pasan también `pre_game_cache`, `live_signals` y `silent_mode`. Se reprodujo el `TypeError` al usar esos keywords. No es un respaldo operativo ni una predicción válida; el import real funciona en el caso mostrado por el usuario.

## Señales y aproximaciones: decisiones pendientes

Actualización tras leer `unsure`: los planes de Run Detector con ContextAnalyzer y aprendizaje con decay no acreditan su implantación actual. Lo indicado aquí sobre rachas describe el cálculo encontrado, no una regresión demostrada de una versión contextual anterior. El mapeo API→modelo documentado sí está presente; no se reabre aquel bug antiguo ni se le atribuye el porcentaje actual de imputación. Los límites de Q4 tardío/OT son decisiones de producto, no carencias que reparar.

Actualización tras revisar `Documentation`: `live_signals_troubleshooting.md` también acredita la intención del cambio FTI0.5→0.17 y la admisión de listas/dicts en el parser; ambos patrones están presentes. Se consideran cambios previos, no defectos antiguos sin corregir. No resuelven por sí solos la calibración del FTI, el parsing de valores de A04-10 ni la cobertura de columnas del bundle. Mostrar una señal no demuestra que modifique el modelo final.

- **Run detector:** no guarda una ventana temporal real; estima 120/180/300 segundos a partir del texto del estado. Los cocientes temporales se cancelan algebraicamente en gran parte de la comparación. Si ambos equipos suman diez puntos, se selecciona una racha local por orden de ramas; reproducido. Q4 también se omite. No se ha definido aún si «racha» debe significar puntos sin respuesta, diferencial o ritmo reciente.
- **FTI:** el cálculo activo usa `(pf/min - .17)/.25` y modificadores por cuarto/margen. El comentario que equipara 4–5 faltas en 12 minutos a .17–.21 es aritméticamente inconsistente; la calibración empírica no se ha validado. Que se sature en 1.0 no prueba por sí solo un error de extracción. `historical_baseline` y `team_side` no aportan al cálculo actual.
- **A/TO y rebotes:** la ruta activa usa `_calculate_live_ou_signals` con ratios y diferencias brutos. Los helpers posteriores normalizan con otras fórmulas y comparten nombres de salida, pero no se encontraron llamadas. No intercambiarlos como si fueran equivalentes; cambiarían las entradas del modelo. El helper alternativo de OREB% reparte los rebotes ofensivos entre ambos equipos, distinto del denominador de oportunidades usado en la ruta activa.
- **TS%/eFG%:** la ruta activa suma Q1–Q4 y hace shrinkage con el mayor FGA de los dos equipos. No incluye puntos de prórroga en ese total y las estadísticas pueden incluirlos. La corrección de fórmula y calibración por equipo requiere revisión de entrenamiento y consumidores, no solo de logs.
- **Datos temporales ausentes:** `extract_time_info` usa ceros/defaults y puede devolver una estructura completa sin indicar explícitamente que el reloj falta o falló. Los consumidores deben distinguir valor real y estimación; aún no se fija una nueva política.
- **Halftime enforcement:** el criterio actual exige unos campos locales y un total mínimo, no confirma que Q2 haya acabado. El usuario ya lo usó al final de Q2. Es una decisión de etapa permitida, pendiente antes de endurecer/reducir requisitos.
- **Single shot:** la salida tras una sola iteración se ejecuta en caminos de éxito; datos ausentes o excepciones pueden continuar/reintentar. Hay scripts consumidores. No retirar ese parámetro solo porque no esté en el menú.

## Seguridad del cambio y obsolescencia

**No borrar `SofaScoreClient` entero:** `BrowserExtensionClient` hereda su constructor, mapeos, parser, reloj, caché y construcción de `live_data`. Separar transporte y normalización puede ser una mejora futura, con contratos y regresiones concretas. Tampoco borrar funciones compartidas de `analysis/live_mode.py` al retirar su interfaz manual.

Candidatos a limpieza posterior:

- Primera definición eclipsada de `get_game_statistics`; parser numérico antiguo que podría ser útil para corregir el contrato antes de decidir su destino.
- Llamada a `_apply_intelligent_fallback` inexistente y sentencia posterior a `return` en el validador.
- Imports sin uso interno `random`, `aplicar_metamodelo_ai`, `calculate_detailed_quarter_metrics`; no confundir import redundante con eliminación de sus módulos.
- Helpers alternativos sin llamadas encontradas: A/TO normalizado, rebotes alternativos, steals/blocks y momentum de liderazgo. Su presencia no significa que el modelo reciba esas señales.
- Variables sin efecto como `sum_q1_q2`, `abs_change` y metadatos de calidad que no se actualizan. `data_quality_score` se calcula en parser pero no se asigna a `_last_data_quality_score`, que el cliente anterior intenta leer.
- Documentación de mejoras porcentuales de precisión/CPU y funcionalidades «restauradas»: no acreditadas mediante benchmarks en esta revisión.

La validación del cliente anterior también es desigual: el health check y fallback básico aceptan JSON parseable sin exigir siempre `events`, a diferencia del método principal. La señal DOM de sesión solo exige un enlace a un partido, que puede ser un enlace estático. Son riesgos de ese transporte diagnóstico, no pruebas de un fallo actual de la extensión.

## Controles actuales que conviene preservar

- La extensión diferencia lista vacía de error de acceso y comunica la causa del puente.
- Se exige periodo ALL antes de interpretar acumulados en la extensión; no se sustituye por un cuarto individual como permite el parser base genérico.
- Faltan categorías requeridas o falla evento/estadísticas: la actualización de la extensión se rechaza antes de producir `live_data` para predicción, aunque falta reforzar validación de valores.
- IDs de evento de la extensión se limitan a dígitos ASCII.
- La caché tiene TTL de 30 segundos y el bucle limpia caché en cada consulta. Aun así, evento y estadísticas se piden por separado: no se ha acreditado que formen una captura atómica o simultánea.
- El monitor recibe `game_finished` y puede finalizar; errores de una actualización se registran y reintentan según su flujo actual.

## Validación y límites

Reproducción: `.diagnostics/audit_live_api.py`. Se ejecutaron **19 escenarios con aserciones** usando los cuerpos originales de las clases y funciones compilados desde AST. Se sustituyeron puente, navegador, predictor, pregame cache, tracker y esperas por dobles; no se importaron módulos ML ni se calcularon predicciones reales. Se comprobó que el archivo fuente permanecía idéntico al terminar.

No hubo conexiones de red, navegador, servidor local real, carga/escritura de modelos, guardado de sesión ni registros de predicción reales. Las exportaciones del escenario eran objetos en memoria. Algunas observaciones están verificadas por lectura o álgebra y se distinguen de los casos ejecutados.

Las dependencias consultadas puntualmente fueron las firmas/retornos de `analysis/live_mode.py`, campos de `PredictionTracker` y referencias de tests/scripts. Falta auditar sus implementaciones completas. Tampoco se ha determinado la causa concreta de las 124/188 características imputadas del encuentro del usuario: los defectos alternativos no se presentan como explicación de aquel caso.

Siguiente módulo sugerido: `analysis/live_mode.py`, motor de características y llamada a inferencia que comparte la ruta live con la interfaz manual futura a retirar. La revisión se detiene aquí.
