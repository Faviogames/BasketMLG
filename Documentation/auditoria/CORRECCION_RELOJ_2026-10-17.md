# Corrección autorizada del reloj live

17 de octubre de 2026. El usuario autorizó expresamente aplicar la corrección propuesta tras la prueba live. Ésta es una excepción acotada al acuerdo de auditoría sin modificaciones: reloj, duración por liga, transmisión a proyecciones y bloqueo ante reloj no válido. No autoriza limpieza general, reentrenamiento ni sustitución de modelos.

## Cambio aplicado

Se añadió `core/game_clock.py`, sin dependencias de modelos o navegador. El periodo procede de la descripción/periodo del evento; el tiempo dentro del cuarto utiliza el resto de `time.played` según la duración. En los tres encuentros reales esta combinación coincide con el reloj visible de SofaScore. No se deduce el cuarto del cociente de played y no se extrapola desde timestamps de pared.

Se preservan duraciones separadas: NBA720s/cuarto y48min, WNBA600s/cuarto y40min. La liga seleccionada se transmite al cliente; una duración de evento incompatible se rechaza. Los formatos de10min ya reconocidos para NBL/EURO/FIBA se conservan; no se añadió otra liga ni soporte de prórrogas.

`SofaScoreClient.extract_time_info` delega en este normalizador y conserva las claves temporales consumidas por el predictor. Se incorporan `clock_valid`, motivo de error, fuente, played original, etapa y duración total. `projected_final_score` se expresa en puntos del partido, corrigiendo el divisor60 adicional anterior.

En `get_live_game_data`, el reloj inválido interrumpe la actualización antes de la predicción/exportación. El predictor compartido también rechaza explícitamente un reloj inválido o una duración incompatible, para proteger llamadas directas. Un encuentro reglamentario terminado se reconoce sin emitir otra predicción en el bucle principal.

En `analysis/live_mode.py`, la duración seleccionada llega al diccionario temporal del metamodelo y a la función activa de Safe Projection. Pacer utiliza esa duración; la salvaguarda utiliza también la mitad de la duración para su ancla al descanso. Se mantienen pesos, umbrales, estadística Poisson y esquema de dos entradas del Huber. No se añadieron columnas al modelo.

## Límites y estados ambiguos

- Descanso explícito:20min WNBA o24min NBA, aunque no haya played utilizable.
- Descansos de cuarto explícitos: se usa el final del periodo indicado. `lastPeriod` sólo sirve para identificar el periodo de un descanso explícito, no para adivinar el cuarto de un estado genérico.
- Inicio exacto: se acepta el comienzo coherente de un periodo con reloj corriendo. Un múltiplo positivo del periodo con significado ambiguo se rechaza hasta recibir datos desambiguados.
- Final reglamentario: duración completa y cero restante; fin de Q4 sin estado final fiable no genera nueva predicción.
- Prórroga, formatos no compatibles, campos no finitos, periodo contradictorio, puntos de un periodo futuro o duraciones incompatibles: reloj no válido; sin una nueva proyección basada en ese dato.
- No se usa el reloj de timeout ni se añade una estimación basada en timestamps cuya sincronización no se ha acreditado.

La actualización omitida puede conservar la última predicción ya calculada en memoria, pero no la recalcula con un tiempo supuesto. El modo live imprime la razón y vuelve a intentar en su próximo ciclo. No se modificaron reglas de selección de partidos, el corte pendiente del inicio permitido de Q4 ni la política de recomendaciones fuera de alcance.

## Archivos funcionales

- Nuevo: `core/game_clock.py`.
- `core/live_api.py`: normalización, transmisión de liga y barreras previas a la predicción.
- `analysis/live_mode.py`: validación temporal, duración transmitida y salvaguarda por liga.
- `core/live_features.py`: Pacer con duración del reloj recibido y rechazo de reloj inválido.
- Nuevo: `tests/test_game_clock.py`.

No se modificaron extensión, puente, credenciales, datasets, artefactos ML ni fórmulas históricas de entrenamiento. Las métricas que explícitamente se expresan por48min y los helpers auxiliares anteriores no se convierten automáticamente en duración de partido: su semántica se conserva para evitar cambios ajenos a esta reparación.

## Verificación

**27 pruebas** mediante unittest, sin entrenar, red o exportación. Cubren NBA/WNBA, seis payloads reales, los tres incidentes de cuarto, inicio de periodos, descansos, final, OT excluido, contradicciones y reloj ausente/no finito. Incluyen la función real del cliente y la barrera del predictor compartido mediante extracción AST, sin simular la lógica de esas funciones. La prueba del cliente verifica que un reloj ambiguo se detiene antes de consultar estadísticas o alcanzar el predictor. También se rechaza un reloj al inicio exacto del cuarto cuando ya hay puntos registrados en ese periodo.

**Modelos guardados reales:** se repitieron los tres escenarios sintéticos de la prueba anterior con el mismo entorno compatible. Las dos salidas NBA permanecieron idénticas. En WNBA, con101 puntos y20min, Pacer pasó de242.4 a202; Huber devolvió194.0786 y no se activó la salvaguarda errónea anterior. Esta cifra es un ensayo de coherencia, no un resultado de precisión.

**Reproducción offline de eventos reales capturados**, sin consultas nuevas ni recomendaciones para Q4 tardío:

| Evento | Minutos corregidos | Pacer | Huber | Retorno después de salvaguarda |
| --- | --- | --- | --- | --- |
| Chicago–Minnesota | 36.6333 | 243.7125 | 242.8126 | 242.8126 |
| Dallas–Atlanta | 47.5833 | 178.5499 | 192.7715 | 192.7715 |
| Memphis–Charlotte | 44.0833 | 157.8828 | 177.6935 | 161.8449 |

Estas ejecuciones no incorporaron box real, sólo los parciales/estado/tiempo capturados y los históricos del bundle. No constituyen nuevas predicciones live, medición de MAE ni recomendación de mercado. Se mantuvieron los modelos intactos y no se llamó al tracker.

**Nueve pruebas existentes del detector adaptativo de Q3** ejecutadas contra el módulo real: todas pasaron. No se afirma que haya pasado toda la suite del repositorio.

Evidencias: `.diagnostics/clock_fix_replay.json` y `.log`, `.diagnostics/saved_models_inference.json` y copia previa `_before_clock_fix`. Los hashes de modelos y el inventario/tamaños/mtime de registros no cambiaron durante las verificaciones.

## Pendiente después de esta reparación

Repetir una sesión nueva mediante extensión, en descanso/Q3, y contrastar el reloj con la ficha visible durante varias actualizaciones. La reproducción de respuestas capturadas no sustituye ese seguimiento. La frescura del reloj del proveedor y las transiciones entre respuestas reales siguen siendo límites que vigilar.

Permanecen fuera de esta reparación: imputación, significado de confianza, trazabilidad base/meta/estabilidad en exportaciones, metadatos del artefacto y cobertura de entrenamiento. El registro inválido3FE7E4DC de la primera prueba se conserva sin editar; acordar después cómo excluir o clasificar pruebas en las métricas.

Para usar la corrección basta reiniciar Python con el código actualizado; la extensión no requiere recarga por este cambio. No se reentrenó ni se sobrescribió ningún modelo.
