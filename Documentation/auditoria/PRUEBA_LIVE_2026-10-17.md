# Prueba live: Chicago Bulls–Minnesota Timberwolves

17 de octubre de 2026, Europe/Madrid. Ejecución de `main.py --live` con el entorno aislado compatible de la prueba anterior. El usuario conectó su extensión Chrome. Sin reentrenar ni modificar código funcional.

## Resultado observado

La extensión entregó 12 eventos; el sistema mostró cinco encuentros compatibles NBA, todos con estado Q3. Se eligió Chicago–Minnesota, 87–77. Se obtuvieron 34 métricas y se ejecutaron base, Huber y salvaguarda. La consulta de cuotas no se inició porque faltaba `requests` en el entorno auxiliar; se continuó explícitamente sin línea. Esto no demuestra un fallo del proveedor ni de la instalación del usuario.

La primera actualización produjo una incoherencia determinante: el extractor informó **Q2, 12.6 minutos jugados, 35.4 restantes**, pese al estado Q3 y los parciales recibidos. Con total164, el ritmo mostrado fue aproximadamente13 puntos/minuto. Huber proyectó530.3, y la salvaguarda elevó la salida a604.5537192. Se interrumpió antes de una segunda actualización. El puente local se cerró.

No es una estimación válida para evaluar precisión del modelo. El origen exacto de la discrepancia de `time.played` aún no está acreditado con el payload bruto.

## Evidencia persistida por el flujo normal

El monitor exportó el registro **3FE7E4DC** y actualizó sus archivos diario, estacional e índice mensual. No se eliminaron ni corrigieron. Archivo individual:

`prediction_records/daily/3FE7E4DC_Chicago Bulls_vs_Minnesota Timberwolves_2026-10-17.json`

Parciales guardados: Q1 33–28, Q2 34–32, Q3 20–17, Q4 0–0. Q1+Q2=127, Q3=37, total=164. Esto demuestra que **se incluyó Q3 en curso**, no que se descartó por estar incompleto. El registro marca `q3_progress`, guarda604.5537192 y metadatos de meta vacíos, reflejando además los problemas de trazabilidad A04-07/08.

La salida informó128/188 características imputadas, frente a124/188 del escenario sintético anterior. No se equiparan las causas ni la lista de columnas de ambas ejecuciones.

Este registro debe identificarse como prueba con reloj incoherente antes de utilizarlo en evaluación o aprendizaje posterior. No se modificaron sus notas, estados ni los agregados durante esta revisión; acordar tratamiento de registros de prueba al auditar tracking.

## Recorrido de código comprobado

- `core/live_api.py:603`: toma `period3` del evento y lo incorpora a `q3_home/away`, aunque Q3 esté en curso.
- `core/live_api.py:992–998`: `extract_time_info` lee `time.played`, lo considera tiempo acumulado y calcula cuarto mediante división por `periodLength`. No contrasta `status.period`, descripción ni parciales para esa decisión. El objeto status sólo interviene después para live/finished.
- `analysis/live_mode.py:1107`: reconoce etapa Q3 si hay puntos Q3, cuando no llega una etapa explícita.
- `analysis/live_mode.py:1151–1153`: si existe `time_info.minutes_played`, lo prioriza frente al fallback por etapa. Así pueden coexistir `q3_progress` y12.6 minutos de supuesto acumulado.
- Pacer y Safe Projection reciben ese reloj; la salvaguarda no es independiente del error, porque utiliza una extrapolación con la misma información temporal incoherente.

Se ejecutó el método real mediante AST con un evento **sintético** de status Q3, `played=756` y periodos de720s: devuelve Q2/12.6min/35.4min. Esta reproducción acredita la falta de contraste entre campos; **756 no es un valor extraído del payload original**, que no se conservó en bruto.

## Aclaración de la hipótesis del usuario

Puede haber tratamientos deliberados de Q3 parcial en métricas concretas; no se revisan como si todo parcial fuese un cuarto completo. Pero no explican este incidente: los37 puntos Q3 sí entraron y el tiempo recibido fue menor que los24 minutos necesarios para haber completado Q1+Q2 NBA.

## Mejora concreta propuesta, no aplicada

1. Capturar en una consulta controlada el evento bruto: status, objeto time, reloj adicional si existe, parciales y total; sin predicción/exportación.
2. Identificar qué significa realmente cada campo temporal antes de elegir fórmula. No suponer todavía que `played` es tiempo del cuarto ni sumar24 de forma incondicional.
3. Construir un reloj normalizado y contrastarlo con periodo, descansos y duración por liga. Probar NBA/WNBA y transiciones; sin incorporar prórrogas ni recomendaciones de Q4 tardío.
4. Cuando el estado y el reloj se contradigan y no haya resolución fiable, devolver un estado temporal no válido y evitar publicar una nueva predicción; conservar los datos y la última salida identificada como anterior según política a acordar.
5. Repetir la prueba real tras aprobar y validar el cambio. No cambiar pesos, reentrenar ni introducir límites artificiales de puntos para ocultar el problema.

La prioridad demostrada por esta prueba pasa a ser el contrato de reloj antes de atribuir la salida a la imputación o a un fallo de aprendizaje del modelo.

## Segunda fase: respuestas reales del evento y recorrido de liga

Tras la autorización del usuario se ejecutó `.diagnostics/probe_live_clock.py` mediante el mismo puente y la extensión. Se hicieron dos rondas sobre tres encuentros, separadas por una espera de30s. Se guardaron seis respuestas brutas en `.diagnostics/live_clock_probe.json` y comprobaciones de coherencia en `.diagnostics/live_clock_consistency.json`. No se generaron predicciones ni se invocó el tracker. El puente se cerró al terminar.

Los eventos procedían de **NBA Preseason**, no de temporada regular. Las capturas se documentan como prueba de transporte y parsing; no validan el rendimiento del modelo en pretemporada.

| Evento | Estado API en ambas rondas | `time.played` | Cuarto/minutos que calcula el extractor | Límite inferior coherente con Q4 NBA |
| --- | --- | --- | --- | --- |
| Chicago–Minnesota16869058 | 4th quarter | 758s | Q2 /12.6333min | 36min |
| Dallas–Atlanta16685178 | 4th quarter | 1415s | Q2 /23.5833min | 36min |
| Memphis–Charlotte16869060 | 4th quarter | 1925s | Q3 /32.0833min | 36min |

Los tres eventos indican `periodLength=720`, `totalPeriodCount=4`, `defaultPeriodLength=12` y `lastPeriod=period4`. **Las seis observaciones incumplen la coherencia entre el estado y el tiempo acumulado interpretado.** Ya no es sólo un ejemplo sintético. El significado/actualidad del campo sigue sin establecerse; no es suficiente para elegir una fórmula sustitutiva exacta.

En las dos rondas permanecieron idénticos tanto el objeto time como los marcadores de cada evento. No se demuestra por ello que el reloj deba avanzar con el tiempo de pared: puede haber pausas o respuestas repetidas. Tampoco se acredita caché como causa. La extensión utiliza `cache: no-store` en su ruta fetch, pero también dispone de navegación a JSON; esto no prueba ausencia de caché en todas las capas.

`clockRunningLastUpdated` contiene instantes posteriores al reloj UTC de captura de este entorno (por ejemplo01:43Z frente a00:02Z). Hay una discrepancia temporal adicional sin atribución; puede corresponder a reloj del entorno, proveedor u otra diferencia de datos. **No extrapolar automáticamente desde ese timestamp** ni etiquetarlo como antiguo sin resolverla. `timeouts.activeTimeout.clock=6:42` aparece para Chicago, pero pertenece al objeto de un timeout; no se considera por sí solo el reloj actual del encuentro.

### Dónde se conserva y dónde se pierde la duración por liga

1. `analysis/live_system.py` selecciona el bundle y fuerza su `league_name` correctamente.
2. `LiveProcessor` conserva ese bundle, pero construye `BrowserExtensionClient(headless=...)` sin pasarle liga. El extractor de ese cliente no consulta el modelo seleccionado: toma duración de `event.time.periodLength` y usa720 como respaldo.
3. `analysis/live_mode.py:1151` prioriza los minutos recibidos de la API. Sólo si no existen utiliza la estimación por liga: NBA12min/cuarto y WNBA10min/cuarto. Esto confirma el mecanismo recordado por el usuario, pero también por qué no corrige un campo presente e incoherente.
4. `analysis/live_mode.py:1652` vuelve a construir el reloj para Huber con `48 - minutes_assumed`, sin transmitir duración de liga ni periodo.
5. `calculate_pacer_features` fija `48*60`; la versión activa de `_calculate_safe_projection` fija48 y descanso24. Así la distinción parcial NBA/WNBA no llega a todos los consumidores.

No se consultó un partido WNBA real durante esta fase. La prueba de duración WNBA anterior fue con modelo real y marcador sintético. Diferenciar ambas evidencias.

### Próximo contraste necesario

Se pidió al usuario comprobar si la ficha normal de SofaScore muestra minuto/segundo además del cuarto. Si sólo muestra el periodo, no inventar un reloj exacto desde puntos, tiempo de timeout o resta de timestamps incoherentes. Antes de cualquier reparación, decidir cómo representar un intervalo temporal válido o no emitir una nueva predicción cuando no hay reloj fiable.

La posibilidad de seguir mostrando datos fuera del tramo de recomendaciones no autoriza publicar nuevas recomendaciones en Q4 tardío ni añadir prórrogas. Todos los hallazgos de esta fase son observaciones y propuestas; no hay cambios funcionales.

## Tercera fase: contraste directo con el reloj visible de SofaScore

El usuario confirmó que la ficha normal muestra cuarto/minuto/segundo y compartió la URL canónica de Chicago–Minnesota:

`https://www.sofascore.com/basketball/match/minnesota-timberwolves-chicago-bulls/jtbsBtb#id:16869058`

La ruta genérica `/event/16869058` del navegador integrado llevó inicialmente a verificación; la URL canónica compartida sí cargó la ficha normal, sin resolver ni automatizar CAPTCHA. Se leyó directamente el DOM visible mediante las herramientas del navegador, sin inspección de estado privado de la aplicación ni nuevas solicitudes desde evaluate.

La ficha mostraba **Q4 (11:22)** como texto HTML, en un span de estado live. Después se navegó a Baloncesto por el enlace visible y se leyeron tres enlaces de encuentros presentes en el listado:

| Encuentro | Reloj visible DOM | `played` capturado antes | `720 - (played % 720)` | Minutos acumulados derivados de periodo4 + reloj visible |
| --- | --- | --- | --- | --- |
| Chicago–Minnesota | Q4 11:22 | 758 | 682s =11:22 | 36.6333 |
| Dallas–Atlanta | Q4 0:25 | 1415 | 25s =0:25 | 47.5833 |
| Memphis–Charlotte | Q4 3:55 | 1925 | 235s =3:55 | 44.0833 |

Comparación aritmética guardada en `.diagnostics/live_clock_dom_comparison.json`. Los tres restos coinciden con lo visible. La deducción actual del número de cuarto mediante la división completa no coincide: devuelve2,2,3 frente al4 indicado tanto por status como por web. Esto permite una hipótesis de reparación concreta: periodo desde un estado fiable y tiempo dentro del periodo desde un reloj contrastado, conservando la duración por liga.

**Límites:** esta observación de UI no se capturó simultáneamente con las respuestas anteriores; sólo acredita coincidencia entre valores observados. No demuestra por sí sola frescura del cronómetro, actualización continua o semántica oficial de `played`. La discrepancia de timestamps sigue sin atribución. No se debe concluir sin pruebas que basta aplicar módulo a todos los estados: inicio/final exacto, pausas, descanso, cambio de periodo y reloj ausente requieren desambiguación. Un reloj de timeout no sustituye al reloj principal.

La extensión ya tiene permiso `scripting` sobre SofaScore y lee DOM en su modo JSON. Eso hace técnicamente plausible leer el estado visible de la ficha, **pero no se ha añadido ese flujo ni ampliado las rutas del puente**. Antes de elegir DOM como fuente principal o respaldo, evaluar disponibilidad, identidad del evento, formato, frescura y comportamiento durante la navegación actual de la extensión.

## Cuarta fase: ejecución desde el menú después de la reparación

Se ejecutó `main.py --live` mediante `.diagnostics/run_live_test.py`, con los modelos reales y el transporte de extensión. Sólo se redirigió en memoria la persistencia del tracker a `.diagnostics/live_test_predictions.json`; no se sustituyeron predictor, estadísticas, reloj ni estabilidad. Tras fallar el listado inicial por una respuesta sin `events`, se volvió al menú y se seleccionó NBA e ID directo16869058 con autorización del usuario.

La primera actualización obtuvo Chicago122–Minnesota105,34 métricas de estadísticas y Q4/36.6333min jugados. La salida fue283.5216 puntos, con128/188 entradas imputadas. Las cuatro consultas siguientes recuperaron estadísticas sin error y conservaron la predicción por cambio calculado de0 puntos. No se evaluó exactitud ni se obtuvo línea de mercado.

El usuario confirmó que el encuentro estaba en pausa y pidió continuar: mantener el reloj quieto durante esa pausa no demuestra por sí solo un fallo. Queda pendiente contrastarlo al reanudarse. Además, la cabecera mostraba11:22 mientras el historial visible ya contenía una jugada de Q4 a3:23 con el mismo marcador122–105; esa discrepancia entre superficies sigue sin explicar y no permite dar por validada la frescura del reloj recibido.

Por inspección independiente, `_should_update_prediction` sólo suma Q1–Q3 al calcular cambios: no tiene en cuenta puntos nuevos de Q4. Esta observación no demuestra que las consultas durante la pausa recibieran puntos nuevos. Revisar después el comportamiento esperado para el inicio permitido de Q4, sin cambiar la política durante esta prueba.

Al redactar esta nota, el seguimiento continuaba activo por petición del usuario. No se cerró el puente ni se alteró código funcional en esta fase. Las horas02:27–02:31 proceden del log local del proceso; la evidencia JSON registra UTC explícito.

### Continuación en la misma sesión: movimiento confirmado

Tras indicar el usuario que el tiempo había cambiado, se mantuvo la misma sesión y conexión. Actualizaciones7–9: estadísticas recuperadas sin errores, Q4/36.6min jugados y estabilidad con cambio0. La ficha normal, incluso después de recargar, mostraba126–107 en lugar de122–105; la cabecera seguía en11:22 y el historial de jugadas llegaba a2:02. No confundir el tiempo de una jugada con el cronómetro principal exacto.

Se abrió una pestaña temporal del endpoint `/api/v1/event/16869058` en el navegador integrado, sin modificar la conexión de la extensión, y se leyó su JSON visible: `homeScore.current=126`, parciales33/34/31/28; `awayScore.current=107`, parciales28/32/28/19; `time.played=758`, `clockRunning=true`, `clockRunningLastUpdated=2026-10-17T01:43:00+00:00`; `changes.changeTimestamp=1792204471`, con cambios sólo en marcadores actuales/display/normaltime y period4. La pestaña temporal se cerró.

Esto acredita un marcador nuevo y el mismo campo temporal en una consulta independiente. No demuestra qué payload exacto recibió cada actualización del monitor, que no exporta su respuesta bruta. Sí confirma por código que, si recibe esos parciales, el filtro de estabilidad calcula cambio0 porque Q1–Q3 siguen iguales. El reloj fijo durante la pausa no bastaba para concluir un defecto; después de observar movimiento, la fuente temporal sigue sin quedar validada. No extrapolar timestamps sin resolver su sincronización ni sustituirlos por el reloj de timeout.

### Final y simulación posterior

Resultado final Chicago133–Minnesota116 (249 puntos), comunicado por el usuario y confirmado en la ficha como Finished. La actualización11 detectó Q4/48.0min jugados/0 restantes, terminó sin nueva predicción y cerró el puente, volviendo al menú. La única predicción de diagnóstico fue283.5216, error absoluto34.52 puntos; no usarla para estimar precisión general con el reloj activo pendiente de validar. El resumen Total updates:1 cuenta la predicción calculada, frente a11 ciclos de consulta.

A petición del usuario se reconstruyó después el descanso con estadísticas sólo1Q/2Q y reloj24/48min. Resultado250.6082 frente a249 reales (error absoluto1.6082). Ver `SIMULACION_DESCANSO_16869058.md` para método, comparación con extrapolación simple y límites retrospectivos. No se incorporaron estas pruebas al historial normal ni se modificaron modelos.

No se generaron nuevas predicciones para Q4 tardío. Todos los modelos y el código funcional se mantienen intactos; el registro inválido de la primera fase se conserva como evidencia pendiente de clasificación en tracking.
