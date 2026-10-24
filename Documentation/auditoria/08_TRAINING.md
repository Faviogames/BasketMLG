# Módulo 08: core/training.py

Fecha: 16 de octubre de 2026. Revisión local terminada, sin cambios funcionales. Archivo completo leído y 21 escenarios aislados ejecutados sobre funciones/clase reales con datos sintéticos y estimadores simulados. **No se ejecutó entrenamiento ML, no se cargaron joblibs y no se escribieron modelos ni exclusiones reales.**

## Responsabilidad y recorrido actual

1. Carga ligas desde `DATA_FOLDER`; `LEAGUES_INCLUDE` permite filtrarlas por coincidencia exacta de nombre.
2. Ejecuta `calculate_features`; exige 20 filas válidas, no 20 partidos distintos.
3. Selecciona la intersección entre `HISTORICAL_FEATURES_TO_USE` y columnas disponibles. El objetivo es `final_total_score`.
4. Intenta generar `base_pred` OOF mediante bloques de grupos por partido. Si falla, registra una advertencia y continúa.
5. Independientemente de OOF, divide X/y al azar por filas con 20% de test y semilla 42.
6. En el camino normal crea un XGBoost, ajusta con train y calcula MAE/desviación de residuos en test. No vuelve a ajustar el modelo final con todo el dataset.
7. Analiza importancias y puede actualizar el JSON global de exclusiones **antes** de guardar el modelo.
8. Guarda `<liga>.joblib` con predictor, columnas, imputaciones, histórico, equipos, métricas y auditoría. Después intenta guardar `<liga>.json` descriptivo.

El llamado «sistema dual» sólo anuncia XGBoost. La eliminación de RF y la retirada de carga Optuna aparecen explícitamente en comentarios; **no reintroducirlos por encontrar documentación antigua que describe dos modelos**. Las funciones `create_model`, `_create_base_model` y el fallback legacy duplican los parámetros XGBoost; actualmente coinciden.

Este entrenamiento **no entrena el metamodelo Chronos/Huber usado en live**. `meta_info` queda en `None`. Los comentarios «meta-modelo eliminado» describen esta ruta, no autorizan retirar la integración activa ni sus artefactos externos.

**Reconciliación con `unsure` y el usuario:** RF/Optuna y los diseños anteriores de metamodelo forman parte de la evolución del proyecto. Queda cerrado como sospecha que su ausencia actual exija restaurarlos o sea por sí sola un fallo. Los documentos LightGBM/Reality Check no describen necesariamente la integración Huber activa. Se mantiene T08-07 porque trata de una etiqueta incorrecta reproducida, no de la ausencia de RF; las observaciones de validación tampoco se invalidan por esta aclaración histórica.

## Hallazgos comprobados

### T08-01 — La evaluación final separa filas del mismo partido

`core/features.py` produce dos filas por encuentro con el mismo histórico y objetivo. La selección de entrenamiento excluye variables live/balance, por lo que esas dos filas pueden ser idénticas en X. La división de `training.py:855` usa `train_test_split` por fila, sin grupos ni corte temporal.

Con 20 partidos sintéticos y dos filas cada uno, una simulación de la permutación aleatoria con semilla 42 deja **6 de los 7 partidos representados en test también en train**. La simulación utiliza el convenio de permutación de ShuffleSplit; no se ejecutó sklearn en este runtime. La ausencia de agrupación en la llamada real se verifica por inspección del código.

La división aleatoria tampoco garantiza que train sea anterior a test: los rolling históricos de una muestra posterior pueden incorporar resultados de encuentros que quedaron en test. Esto compromete la interpretación de la evaluación como simulación de predicción futura. No se ha medido cuánto cambia el MAE ni demostrado qué versión exacta generó los modelos guardados.

**Control positivo:** la función OOF sí agrupa por partido cuando dispone de identidad suficiente. Esa protección no se traslada a la división final; no afirmar que todo el entrenamiento carece de grupos.

### T08-02 — La columna llamada OOF incluye predicciones dentro de muestra

`_generate_oof_base_predictions` (361–449) usa grupos anteriores para los bloques de validación. El primer bloque se deja sin predicción porque no tiene train anterior. Al final entrena otro modelo con **todo** el dataset y rellena los huecos con predicciones sobre esas mismas filas.

En 10 grupos y cinco bloques, los primeros dos grupos, cuatro filas, reciben predicciones de un estimador que ya vio sus etiquetas. Un doble que memoriza inputs confirma este recorrido. Los cuatro folds posteriores sí entrenan con grupos anteriores de tamaños 4, 8, 12 y 16 filas; la imputación de cada fold se calcula sólo con su train.

El MAE impreso como «OOF base» agrega los dos tipos de predicción. `base_pred` queda en el histórico guardado, pero no se añade a X en esta ruta y aquí no se entrena un meta-learner. Por tanto, este relleno **no es el mecanismo directo del MAE final** de T08-01. Revisar los consumidores posteriores de la columna antes de retirarla o cambiar su formato.

Con menos de dos grupos, el respaldo usa la media de las características de cada fila como `base_pred`: una entrada normalizada a 0 devuelve 0 puntos aunque el objetivo sea 200. Es un valor de contingencia sin unidad predictiva garantizada.

### T08-03 — La identidad de partido tiene respaldos frágiles

`_ensure_match_group_key` prioriza `raw_match.match_id`, después equipos + fecha raw. Si existe `raw_match` pero sus diccionarios no tienen fecha/ID, encuentros repetidos entre los mismos equipos colapsan en un único grupo. Si no existe la columna, empareja filas consecutivas por índice incluso cuando haya otras columnas identificadoras.

Se reprodujeron ambos respaldos. Con IDs válidos, veinte filas quedan correctamente en diez grupos. OOF presupone que el orden de primera aparición de los grupos es cronológico; no ordena fechas ni comprueba su validez por sí mismo. El productor actual ordena previamente, sujeto a las limitaciones de fechas D07-07.

### T08-04 — Entrenamiento e inferencia reciben tratamientos distintos de ausencia

En el camino principal X_train llega a `model.fit` sin rellenar NaN. XGBoost puede tratarlos como datos ausentes. En cambio, `impute_values` se calcula con **todo** `df`, incluidas filas de validación, y se guarda para que live sustituya faltantes por valores numéricos antes de predecir.

El escenario confirmó una media guardada 9.743589 frente a 9.806452 usando sólo train, y NaN en el input recibido por el estimador simulado. No se afirma que esas medias se aplicaran al ajuste principal: **no se aplican allí**. Hay una diferencia de contrato entrenamiento/live y el respaldo guardado no representa un cálculo restringido a train.

No cambiar a imputación numérica universal ni eliminarla sin decidir cómo deben verse los faltantes en ambos recorridos y cómo migrar los artefactos.

### T08-05 — Se guarda un dataset por etapas como historial por partido

El bundle guarda `historical_df = df`, con ambas muestras y también las filas de test. En el escenario, el predictor final ve 32 filas y el histórico guardado contiene 40. Conservar todo el histórico puede ser una decisión válida para inferencia posterior al dataset; **no es por sí solo una fuga de validación**.

El problema de contrato aparece al consumirlo como partidos: `build_pre_game_cache` de `analysis/live_mode.py:907` filtra equipos e itera filas sin deduplicar encuentros. `tail(5)` sobre ese histórico cubre tres partidos distintos en vez de cinco y modifica medias/EMA respecto de una serie por encuentro. Se verificó la diferencia con el histórico sintético guardado y se contrastó el consumidor por inspección.

El dataset de entrenamiento, el histórico base para inferencia y el histórico para reproducir una predicción pasada tienen usos distintos. Antes de separarlos, revisar todos sus consumidores y confirmar una migración que conserve los modelos existentes.

### T08-06 — El contador de partidos cuenta muestras

Tanto el resumen como `base.matches_count` del JSON usan `len(df)`. En el escenario representan 40 muestras de 20 partidos. El mínimo de 20 filas puede corresponder sólo a diez encuentros.

Los JSON existentes anuncian NBA 2968 y WNBA 1046; con este generador esos campos serían recuentos por etapa, pero **no se cargó el histórico de sus joblibs para verificar cuántos encuentros únicos contienen**. No reinterpretar automáticamente todos los números históricos.

### T08-07 — El fallback conserva etiquetas de un modelo retirado

Si falla el camino principal y `train_single_model_legacy` funciona, éste crea XGBoost, pero el llamador asigna `selected_model_type = 'random_forest'`. Se reprodujo provocando un fallo simulado en la creación principal: el bundle declara RF aunque la fábrica utilizada es XGBoost.

Si XGBoost no está instalado, `DUAL_SYSTEM_AVAILABLE` es falso y el fallback también requiere XGBoost: lanza `ImportError`; no hay un segundo backend. No proponer reinstalar RF como reparación implícita.

### T08-08 — Las exclusiones globales se escriben antes de confirmar el artefacto

Con importancia inferior a 0.001 se invoca `save_useless_features_to_json` antes de `joblib.dump`. En un escenario con fallo de almacenamiento simulado, el guardado de exclusiones ya había sido solicitado, aunque no se guardara el nuevo modelo. No hay rollback conjunto.

Esto se suma al alcance global/entre ligas señalado en C02. El código denomina estas variables «potencialmente inútiles» pero persiste su exclusión automáticamente. La importancia baja de un ajuste no demuestra por sí sola que sean prescindibles en todos los modelos/ligas. Confirmar la política anterior del equipo antes de cambiarla.

El guardado principal sobrescribe una ruta fija sin staging/versionado. Un fallo puede interrumpir las ligas siguientes; el JSON descriptivo tiene un `try` independiente y puede quedar desincronizado con el joblib. Se inspeccionó esta secuencia; no se provocaron fallos en archivos reales.

## Informes y metadatos: distinguir medición de etiqueta

- MAE y `std_dev` se calculan con la división final descrita, no con los folds temporales OOF. `std_dev = np.std(y_test - y_pred)` no incluye por sí sola calibración de probabilidades live ni garantiza sesgo cero; no se validaron residuos reales.
- `calculate_training_insights` vuelve a predecir sobre el mismo test. Su `feature_stability_score` es concentración de importancia de las diez primeras variables, **no una medición de estabilidad entre ajustes o periodos**.
- `pace_consistency` en el informe de ritmo es un literal 0.8: se reprodujo que permanece con ritmos 60 y 140. Es un placeholder, no una estadística calculada.
- `calculate_scoring_patterns` necesita `pace_game`/`q4_total`, pero el ensamblaje actual no conserva el primero del procesador raw ni construye el segundo; los helpers live consultados tampoco aportan `pace_game`. Varias secciones pueden omitirse por falta de columnas. No rellenarlas con estimaciones sin acuerdo.
- `calculate_league_scoring_stats` cuenta totales estrictamente mayores que cada línea. Los empates/pushes quedan en el denominador, pero no se reportan aparte. Dos totales de 220 producen over_220=0 y etiqueta `unders_favored`; no interpretar esa etiqueta como probabilidad de ganar una apuesta under.
- Las líneas de informe [200, 210, 220, 230, 240] y el sesgo respecto de 220 son fijos para todas las ligas. Es una política de presentación pendiente, especialmente para WNBA; no cambiar su duración ni fórmulas relacionadas a partir de este informe.
- El JSON omite deliberadamente `features_used` para no crecer. El bundle binario sí conserva la lista. Faltan huellas de código/dataset/dependencias y detalle de partición que permitan atribuir una métrica histórica a una ejecución reproducible.

## Artefactos existentes y límites

Se inspeccionó únicamente metadata JSON y el inventario de archivos. NBA declara creación 2025-11-28, MAE 3.35449, 188 características y 30 equipos; WNBA declara 2025-10-09, MAE 4.52095, 153 características y 13 equipos. **Son cifras registradas, no verificadas por esta auditoría**. El JSON NBA tiene variables de mercado entre las principales: revisar F06-01 y el origen concreto antes de corregirlas o evaluar el modelo.

No se cargaron joblibs, ni se reprodujo el dataset exacto, ni se afirmó que las métricas guardadas provengan de esta versión del módulo. El runtime auxiliar no tiene sklearn, xgboost ni joblib; se usaron doubles que registran entradas/salidas. Ningún `fit` del diagnóstico optimiza un modelo.

## Código candidato a revisión, sin eliminaciones autorizadas

- Cabecera y docstrings RF+XGBoost obsoletos respecto de la implementación actual. `DualModelSystem`, sin embargo, contiene creación/guardado activos: no borrar la clase por su nombre.
- Camino multi-modelo, selector interactivo y comparador quedan normalmente inaccesibles porque `get_available_models` sólo devuelve XGBoost. Pueden tener usos externos; confirmar antes de simplificar.
- En el camino de un solo modelo se guarda el principal directamente, pero no se llena `dual_system.results`; por eso no se generan sus archivos alternativos. No es una prueba de que falte guardar el principal.
- `_simulate_historical_signals` no se llama desde entrenamiento activo; existe referencia en un test cuyo import está comentado. Simula TS% algebraicamente constante para totales positivos y algunos proxies usan resultados finales. No confundirla con señales live reales ni conservarla/eliminarla automáticamente por el nombre.
- `Ridge`, `UNIVERSALLY_USELESS_FEATURES`, `FILTERED_FEATURES_TO_USE` son imports sin uso en este archivo. La eliminación de Optuna se declara «por petición» en comentario: conservar esa decisión mientras se confirma contexto.
- La promesa impresa de «+25–30% velocidad» por retirar variables no proviene de un benchmark realizado aquí.

## Comprobaciones y continuación

Script: `.diagnostics/audit_training.py`. 21 escenarios comprueban contratos y recorridos con funciones reales, entradas sintéticas y doubles de dependencias, estimadores, split y almacenamiento. Se ejecutó la orquestación con IO simulado, incluidos fallos de creación/guardado; ninguna escritura alcanzó modelos ni exclusiones reales. Las pruebas confirman comportamientos, no precisión predictiva.

SHA-256 de producción antes/después: `21169eeb139796d85c465187ac6971aad292d5a5ef77669bbc2eeffbe89e47a0`.

Pendiente para la fase de mejoras: acordar validación temporal/por encuentro, definición de OOF, contrato del histórico live, tratamiento de faltantes y migración/versionado. Revisar primero documentación del equipo; no reentrenar ni reemplazar artefactos durante la auditoría.

Siguiente sugerido: `core/live_features.py`, para completar el recorrido de ritmo, balance e interacciones antes de revisar la integración del metamodelo. Parar aquí según el acuerdo de un módulo por turno.
