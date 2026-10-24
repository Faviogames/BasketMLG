# Módulo 05: analysis/live_mode.py

> Actualización20/10/2026: se retiró únicamente `live_mode_with_alerts`, la interfaz interactiva manual, y sus imports exclusivos. Las funciones de cálculo compartidas se conservan sin cambios. La retirada fue autorizada expresamente por el usuario; no reintroducir esa interfaz por las referencias históricas de esta auditoría. Ver `00_ESTADO.md`.

Fecha: 16 de octubre de 2026. Estado: revisión local terminada; sin cambios funcionales. Se leyó el archivo completo y se distinguió la ruta live compartida, la interfaz manual a retirar y los ajustes anteriores. No se cargaron modelos reales ni se auditó por completo la ingeniería de características de otros módulos.

## Papel del módulo y partes que deben separarse

| Bloque | Papel y uso encontrado |
| --- | --- |
| `build_pre_game_cache` | Construye históricos por equipo, rolling/EMA, tendencias y H2H para el procesador live |
| `get_predictions_with_alerts` | Monta el vector, imputa, ejecuta el modelo base, delega al metamodelo, aplica una salvaguarda y devuelve probabilidades/metadatos |
| `live_mode_with_alerts` | Interfaz por prompts; el usuario decidió retirarla después de la auditoría |
| `apply_context_adjustment` y helpers | Sistema anterior de reglas; no lo llama la ruta principal actual de `get_predictions_with_alerts`, pero existe un wrapper en el cliente y referencias de pruebas |
| `calculate_signal_strength`, `count_active_signals`, `calculate_signal_adjustment_magnitude` | Funciones importadas desde entrenamiento; no son eliminables por pertenecer al archivo del modo manual |
| `apply_metamodel_ai_adjustment` | Wrapper defensivo adicional; el predictor principal llama directamente a la integración, sin usarlo |
| `_calculate_safe_projection` | Salvaguarda activa, definida dos veces; Python usa la última |

**La retirada del modo manual debe conservar los cálculos compartidos.** El nombre del archivo no refleja todas sus responsabilidades.

## Recorrido actual

1. El procesador construye la caché con el bundle histórico.
2. El predictor toma `features_static`, determina etapa/minutos y consulta helpers de ritmo/balance.
3. Agrega estadísticas de SofaScore, métricas avanzadas e interacciones.
4. Crea un DataFrame exactamente con `trained_data['features_used']` y rellena NaN con imputaciones guardadas, medias históricas o cero.
5. Ejecuta `trained_data['model'].predict` y entrega esa salida a `analysis.metamodel_integration.aplicar_metamodelo_ai`.
6. Puede mezclar la salida del metamodelo con una proyección de ritmo si la desviación supera el umbral de salvaguarda.
7. Genera cinco líneas sintéticas, devuelve la predicción final y metadatos. No escribe registros en esta ruta: la exportación live se hace en `LiveProcessor`.

Entradas requeridas del bundle: `model`, `std_dev`, `historical_df`, `features_used`; opcionales `impute_values`, `league_name`, `bookie_line`. La caché contiene características, históricos, tendencias, equipos y fecha; las señales contienen estadísticas planas y reloj.

La salida es `(final_prediction, predictions, alerts_data)`. `alerts_data['raw_prediction']` es la salida del modelo base; `context_info['meta_metadata']` conserva datos de la integración; `final_prediction` puede incorporar la salvaguarda posterior. No confundir estas etapas al corregir el registro A04-07/A04-08.

## Hallazgos de la ruta live

### V05-01 — Contrato incompleto entre caché y características esperadas

Prioridad alta. Referencias: `analysis/live_mode.py:938`, `analysis/live_mode.py:1010`, `analysis/live_mode.py:1018`; configuración consultada en la auditoría 02.

Se reprodujeron con helpers rolling/EMA reales extraídos de `core/features.py` y datos controlados:

| Familia | Lo esperado/declarado | Lo que monta la caché |
| --- | --- | --- |
| Puntos básicos | `home_avg_pts_scored_last_5` | `home_avg_points_scored_last_5` |
| EMA multiventana de diferencia | `diff_ema_win_short_term_3`, etc. | Solo se calculan diferencias `diff_ema_win_last_5`; no las multiventana |
| Métricas directas | `home_win_rate`, variantes de otras bases directas | No se copian al diccionario estático como características directas |
| Contexto promedio | Nombres `*_avg_*_last_10` del catálogo | Se pasan bases de contexto a EMA, no se monta el rolling de diez con esos nombres |
| Mercado | `market_total_diff`, `implied_pace_vs_avg`, `line_volatility_index` | No hay ensamblaje de estas características en este módulo ni en los helpers live/avanzados consultados |

La caché sí construye algunas columnas home/away de EMA multiventana. El problema no es que toda la historia esté ausente: faltan o no coinciden familias concretas. Agregar estadísticas live al vector no reconstruye esas familias históricas.

Esto aporta **causas estructurales comprobadas de imputación**, pero no cuantifica las 124/188 entradas del partido del usuario. Para atribuir cada una hay que cruzar `features_used` del bundle concreto, su histórico y la lista de faltantes de esa ejecución. El texto `model_features.txt` tiene 153 líneas y el JSON de metadata NBA anuncia 188 con `features_used: null`; no se tomaron como sustituto probado de la lista actual del bundle.

### V05-02 — Histórico local/visitante asimétrico y agregados repetidos

Prioridad alta de coherencia de características. Referencias: `analysis/live_mode.py:948`, `analysis/live_mode.py:973`.

Para el equipo local se escriben `q1_points`…`q4_points` como cero; para el visitante se toman los parciales históricos. Se reprodujo con filas que contenían los mismos parciales disponibles para ambos equipos. Las tendencias de cuartos pueden recibir información distinta por la posición del equipo en la predicción.

También se calcula un diccionario de momentum para todo el historial y cada agregado se asigna a todas las filas antes de calcular las EMA. Esto puede sobrescribir las series `win` y `plus_minus` ya construidas. Se comprobó el broadcast con un doble de momentum; el helper real consultado devuelve agregados para esas claves. El efecto estadístico exacto frente al entrenamiento requiere su auditoría posterior: no cambiar esta construcción sin comparar la semántica de ambos recorridos.

### V05-03 — Caché opcional que no se reconstruye si falta

Prioridad media/alta. Referencias: `analysis/live_mode.py:1130`, `analysis/live_mode.py:1137`.

Con `pre_game_cache=None`, el predictor inicia `features={}` y no reconstruye características estáticas desde `historical_df`, aunque lo tiene disponible. Se reprodujo que una característica aportada por la caché desaparece y se imputa al quitarla. Conecta con el procesador, que continúa sin caché si falla su construcción.

La caché se reutiliza sin comprobar aquí que sus equipos correspondan a la llamada. Tampoco se ordena ni recorta por fecha el histórico al construirla. No se afirma que el histórico actual esté desordenado o contenga futuro; esa garantía debe establecerse en el contrato de datos y los tests retrospectivos.

### V05-04 — Respaldo de interacciones usa una variable inexistente

Prioridad alta en el camino de error. Referencia: `analysis/live_mode.py:1399`.

Si falla `calculate_live_interaction_features`, se crea `default_interactions` pero se ejecuta `features.update(default_enhanced)`. Se reprodujo el `NameError`, que interrumpe la predicción en vez de aplicar el respaldo. No se modifica ahora el nombre ni se decide si esos valores por defecto son apropiados.

### V05-05 — El argumento opcional `live_signals=None` no es admisible en la práctica

Prioridad media. Referencias: firma en `analysis/live_mode.py:1073`, accesos desde `analysis/live_mode.py:1203`.

Tras algunos accesos protegidos se llama a `live_signals.get` sin normalizar `None`. Invocar la función con su valor predeterminado causa `AttributeError`; se reprodujo. La interfaz manual hace precisamente una llamada sin señales. El procesador live normalmente suministra un diccionario, por lo que este defecto no invalida por sí solo el flujo real ya confirmado.

### V05-06 — Reloj reconstruido como un partido de 48 minutos

Prioridad alta para ligas de distinta duración. Referencias: `analysis/live_mode.py:1347`, `analysis/live_mode.py:1653`, `analysis/live_mode.py:2113`.

Aunque llegan minutos restantes reales, se vuelve a construir el tiempo de las métricas avanzadas y del metamodelo como `48 - minutes_assumed`. Se reprodujo WNBA con 20 minutos jugados y 20 restantes: se enviaron 28 restantes a ambos contratos.

La proyección segura activa también fija 48 minutos y un descanso a 24; no recibe la liga. El fallback sin reloj distingue algunas ligas para estimar minutos jugados, pero luego vuelve a usar 48 al calcular los restantes. No se propone agregar prórrogas: están excluidas por decisión del usuario.

### V05-07 — Totales de contexto ausentes y variables indefinidas silenciadas

Prioridad media/alta. Referencias: `analysis/live_mode.py:1401`, `analysis/live_mode.py:1410`, `analysis/live_mode.py:1793`.

- Los totales `halftime_total`, `q3_total`, `q3_end_total` se leen de `features`, pero este módulo no los monta directamente desde `q_scores`. Los helpers de ritmo/balance consultados tampoco aportan esas claves. Con parciales válidos que suman 100 se reprodujo `balance_features['current_total']=0` bajo esos contratos. Esto no impide que el metamodelo reciba los parciales correctos por otro argumento.
- En Q4 se referencia `q4_total_now` sin definirlo. Se captura el error y puede quedar el total sin actualizar. Reproducido con un inicio de Q4, que puede estar dentro del alcance acordado.
- El chequeo final de desviación usa `league_name` sin asignarlo en la función. La excepción se silencia y no se genera la alerta. Se reprodujo el mismo caso sin alerta; al introducir ese global únicamente en el entorno de auditoría, apareció la alerta esperada. No se aplicó ese cambio al proyecto.

El debug de totales también puede fallar por esas claves ausentes; se captura y desaparece información útil. Un `except: pass` no demuestra que el cálculo omitido sea correcto.

### V05-08 — Calidad, confianza y «modo conservador» no representan lo mismo

Prioridad media/alta. Referencias: `analysis/live_mode.py:1533`, `analysis/live_mode.py:1557`, `analysis/live_mode.py:1587`, `analysis/live_mode.py:1715`.

El contador de imputación mide NaN de columnas del modelo. La confianza solo baja si la columna está además en una lista corta de nombres críticos, muchos de ellos live/totales. Se reprodujo que 20 de 21 características históricas faltaban, la predicción se ejecutaba y `confidence_score` seguía en 1.0. Ese campo no es una probabilidad calibrada de acierto.

Las estadísticas API ausentes se introducen como cero, de modo que no cuentan como faltantes en el control NaN: se reprodujo `home_assists=0` y ninguna imputación registrada sin haber suministrado esa estadística. Esto mezcla un cero medido con un dato desconocido.

El mensaje «modo conservador activado» de más del 50 % no introduce por sí solo un flag o bloqueo. **Sí hay una salvaguarda real posterior**, que cambia de umbral 30/peso 0.6 a umbral 15/peso 0.8 ante cualquier imputación, no solo al superar 50 %. En un escenario controlado, una sola columna ausente cambió el resultado de 220 a 197.6 por esa regla. No debe describirse todo el mecanismo como meramente decorativo ni asumir que la advertencia implica abstención automática.

La política de datos insuficientes, reducción de confianza y permiso para recomendar deberá acordarse con el usuario; no se cambian thresholds ni pesos durante la auditoría.

### V05-09 — Líneas sintéticas y alerta de ventaja desconectadas de la línea de mercado

Prioridad media de interpretación. Referencias: `analysis/live_mode.py:1612`, `analysis/live_mode.py:1753`.

La línea de mercado se busca en bundle/características y se pasa a la integración. Sin embargo, las cinco líneas devueltas se generan alrededor de la predicción, separadas por `std_dev/4`; no son líneas cotizadas obtenidas del proveedor. Se reprodujo que la central coincide con el resultado redondeado y tiene aproximadamente 50/50 para una predicción entera.

La alerta `high_edge_alert` compara probabilidades de esa línea central sintética, no de la línea real. En general mide la distancia del redondeo respecto a la desviación estándar, no una oportunidad de mercado. Las probabilidades reutilizan la desviación estándar del bundle tras metamodelo/salvaguarda; su calibración no ha sido validada aquí. Antes de cambiar la presentación o el significado de los registros, fijar qué se quiere devolver y distinguir proyección, línea de referencia y mercado real.

## Código anterior: defectos que no se atribuyen a la ruta live principal

Actualización tras leer `unsure`: se aclara el propósito histórico de los límites y del Reality Check —evitar sobrecorrecciones y distinguir outliers legítimos—. No se considera esa arquitectura anterior un fallo por coexistir con Huber. Los comportamientos técnicos siguientes se conservan como observaciones de funciones anteriores; no acreditan un defecto grave del recorrido principal actual ni autorizan recuperar, sustituir o borrar esas reglas.

- `_calculate_safe_projection` aparece en líneas 111 y 2113; la segunda definición eclipsa a la primera. Se comprobó qué cuerpo se ejecuta. La definición antigua no es la salvaguarda usada hoy.
- `_get_league_scoring_params` comprueba NBA antes de WNBA y devuelve el multiplicador NBA para WNBA. Reproducido. Este helper lo usa `apply_context_adjustment`, no la llamada principal actual al metamodelo.
- `_apply_pace_expectation_adjustment` divide minutos entre 60 pese a recibir ritmo en puntos/minuto; se reprodujo una reducción con 4 pts/min y 12 minutos por ese criterio.
- `_apply_reality_check_penalty` usa `min` donde los comentarios prometen limitar la reducción: puede devolver un factor negativo con desviación grande. Reproducido con una entrada extrema. `apply_context_adjustment` tiene después un clamp del resultado, por lo que no se afirma que la predicción principal actual se vuelva negativa por este helper.
- El respaldo de `apply_context_adjustment` puede seguir leyendo locales no asignadas después de capturar un error inicial. Reproducido con contexto no válido.
- Algunos ajustes descritos como reducciones pueden convertirse en aumentos al multiplicar boosts. Otros comentarios de porcentajes máximos no coinciden con los coeficientes. Requieren análisis de reglas si se decide mantener esa API, no una recalibración automática.
- `METAMODEL_AI_AVAILABLE` protege el wrapper adicional, pero `get_predictions_with_alerts` importa y llama directamente a la integración; ese flag no garantiza fallback del recorrido principal.

La presencia de helpers de prórroga y final de partido es código histórico, no una autorización para ampliar el alcance. Mantener Q1–Q3 y solo el Q4 inicial acordado; no implementar OT ni recomendaciones tardías como parte de una limpieza.

## Interfaz manual: retirada ya decidida, no reparada

`live_mode_with_alerts` contiene prompts de equipos/parciales, comprobación de descanso, consulta de cuotas, presentación y una exportación propia. Llama al predictor sin señales, afectado por V05-05. Por lectura, además usa `from prediction_tracker` sin la ruta `metamodel` y referencias a `live_pace_metrics`/`balance_features` que no define en ese alcance; se capturan errores de exportación. No se hizo una ejecución manual ni una exportación real para probar esas ramas adicionales.

El usuario ya decidió retirar esta interfaz. No invertir en repararla sin una nueva petición, pero preparar antes una separación de contratos: `build_pre_game_cache`, `get_predictions_with_alerts` y las funciones consumidas por entrenamiento deben seguir disponibles. Los prompts de esta interfaz no equivalen a la introducción opcional de línea o selección de ID del menú live.

## Límites de obsolescencia y otros detalles

Actualización tras revisar `Documentation`: mantener antecedentes estáticos en caché durante Q3 no es un defecto por sí solo. Se descarta como recomendación automática el plan antiguo de poner `pre_game_cache=None` por etapa: V05-03 mostró que no se reconstruye todo al faltar. Las alarmas Knicks–Nets basadas en comparar predicción final con puntos parciales no se usan como evidencia de error final ni para imponer un nuevo orden de pipeline. El recorrido actual comprobado sigue siendo la referencia.

- `apply_context_adjustment` no forma parte del recorrido principal actual, pero tiene consumidores de compatibilidad/pruebas. No declarar todas sus reglas eliminables antes de inventariarlos.
- Las funciones de fuerza/conteo/magnitud de señales tienen un consumidor en `core/training.py:535`. Conservarlas o migrarlas con sus llamadas cuando se retire lo manual.
- La variable `use_cache` cambia reutilización de alertas, pero las características estáticas se copian independientemente de ella; los comentarios de «desactivar caché en Q3» no describen una reconstrucción completa. Conservar estáticas durante un mismo partido puede ser intencionado.
- No se valida aquí el tipo/finitez de todas las columnas o imputaciones; se comprueba NaN. Tampoco se acredita una calibración de los valores de respaldo de Pulse/interacciones. Esos contratos se revisarán con entrenamiento y productores.
- `silent_mode=True` no silencia todos los mensajes; señales, imputación y metamodelo se imprimen igualmente.
- Hay asignaciones repetidas/autorreferentes y resúmenes de equipo con valores fijos. Los comentarios de metamodelo «eliminado» quedan aclarados como referencias a diseños anteriores: no se consideran por sí solos una alarma funcional. No confundir textos antiguos con capacidades actuales verificadas.
- El alias genérico `euro* → EUROBASKET` y otros nombres de liga deberán contrastarse con las competiciones realmente admitidas antes de normalizar archivos/modelos.

## Validación y alcance de las conclusiones

Reproducción: `.diagnostics/audit_live_mode.py`. Se completaron **17 escenarios con aserciones** y se comprobó que el fuente permanecía idéntico.

Se evaluaron los cuerpos originales de las funciones mediante AST con pandas/numpy. Se usaron cinco pequeños helpers rolling/EMA/seguridad originales de `core/features.py`; el agregado momentum, H2H, métricas live, Pulse, modelo y metamodelo se sustituyeron por contratos/dobles explícitos. Por tanto, se validan ensamblaje, errores y transformaciones de este módulo, no exactitud de los helpers no ejecutados ni de predicciones reales. No se importó el módulo ML completo ni se deserializó ningún bundle.

No hubo red, navegador, entrenamiento, cambios de parámetros, registros de predicción reales ni modificaciones del código funcional. Se consultaron nombres de configuración, productores de claves y metadata legible para contrastar contratos. No se ha reproducido la captura exacta del usuario ni se ha demostrado que una única causa explique las 124 características imputadas.

Siguiente módulo sugerido: `core/features.py`, para contrastar construcción histórica/entrenamiento con los nombres y semántica que consume este módulo. La revisión se detiene aquí.
