# Módulo 11: metamodel/train_meta_learner_huber.py

Fecha: 16 de octubre de 2026. Archivo completo leído, procedencia inmediata del CSV contrastada y 18 comprobaciones aisladas. No se ejecutó entrenamiento real, no se cargaron joblibs ni se sobrescribieron modelos, coeficientes o datos.

## Responsabilidad y recorrido

Script independiente que lee `data/training_dataset_meta_v1.csv`, selecciona `base_model_prediction` y `pace_poisson_projection`, y utiliza `final_total` como objetivo. Elimina filas con NA en esas tres columnas y hace una división aleatoria de filas 80/20, semilla 42.

Evalúa base y Pacer sobre el test, ajusta `HuberRegressor(epsilon=1.35, max_iter=1000)` con train y evalúa su MAE en ese mismo test. Guarda el estimador en `metamodel/models/meta_learner_v1_final.joblib`, seguido del texto `meta_learner_coefs.txt`. No guarda un bundle con metadatos.

La integración live del módulo 10 carga precisamente esa ruta. No se encontraron llamadas al entrenador desde main, core, analysis u otros scripts Python inspeccionados; el propio bloque `__main__` lo ejecuta al lanzar este archivo. `python main.py --train` es otra ruta y no acredita haber actualizado el Huber.

## Datos presentes y procedencia inmediata

Inspección de sólo lectura del CSV actual:

| Comprobación | Resultado |
| --- | --- |
| Filas / IDs únicos / IDs ausentes | 1556 / 1556 / 0 |
| IDs presentes en el `leagues/NBA.json` actual | 1556 de 1556 |
| Filas duplicadas completas | 0 |
| NA o infinitos en dos entradas y objetivo | 0 |
| Etapa | Todas `is_halftime = 1`; ninguna Q2/Q3/Q4 |
| Minutos restantes | 24 en todas las filas |
| Línea de mercado cero / ausente | 1314 / 0 |
| Campos explícitos de fecha o liga | No presentes |

La cadena encontrada en fuentes es:

`leagues/NBA.json` → `scripts/process_historical_features.py` → `training_dataset_live_v3.csv` → `scripts/generate_base_predictions.py` → `training_dataset_live_v4_stacked.csv` → `scripts/generate_meta_features.py` → `training_dataset_meta_v1.csv` → este entrenador.

El CSV stacked y el CSV meta conservan exactamente los mismos IDs en el mismo orden y los mismos valores de `base_model_prediction`. La fórmula del Pacer reconstruye numéricamente el CSV meta completo. Esto refuerza la relación entre archivos presentes; **no demuestra el historial exacto de ejecuciones ni vincula por sí solo el joblib actual con ese CSV**. No hay manifiesto que una hashes de datos, generadores, parámetros y artefacto.

Los scripts de esta cadena se consultaron para rastrear entradas; no se da por terminada su auditoría completa.

## Hallazgos

### H11-01 — Definición distinta del Pacer entre CSV de entrenamiento y live

`scripts/generate_meta_features.py` utiliza `calculate_pace_poisson` simulando el descanso a 24 minutos. La segunda entrada del CSV es:

`0.6 × (halftime_total × 2) + 0.4 × referencia`

La referencia es `bookie_line` cuando tiene valor distinto de cero y 220 cuando es cero. La reconstrucción coincide en las 1556 filas, con error numérico máximo 2.84e−14.

En cambio, la integración actual usa `calculate_pacer_features`: al descanso NBA, con reloj válido, su proyección es `halftime_total × 2`, sin esa mezcla. **1513/1556 filas difieren**, con diferencia absoluta media **8.206941 puntos** entre las dos definiciones. Esta cifra mide cambio de entrada, no error de predicción ni pérdida de precisión atribuible al Huber.

Así se concreta LF09-11: no es sólo que existan dos funciones con nombre parecido; el CSV objetivo de este entrenador contiene la definición alternativa. Falta acreditar qué datos/código generaron el joblib activo antes de atribuirle esa divergencia. No cambiar unilateralmente el Pacer live: alinear el contrato puede exigir un artefacto distinto y validación previa.

Aunque el entrenador retira `bookie_line` como columna directa, **la información de mercado o su sustituto constante permanece indirectamente dentro de Pacer**. En 1314 registros se usa 220. No afirmar que Huber se entrenó completamente sin mercado porque sólo tenga dos columnas.

### H11-02 — Evaluación aleatoria, sin corte temporal ni cobertura por etapa/liga

La división usa `train_test_split(..., test_size=0.2, random_state=42)`. No se pasan fecha, liga, etapa ni grupos. Con el CSV actual son 1244 filas train y 312 test.

Aquí **no se reproduce la separación de dos muestras del mismo partido de T08-01**: los 1556 IDs son únicos. El problema específico es que la evaluación no establece que entrenamiento preceda temporalmente a test. Si en el futuro se incorporan varias etapas por partido, tampoco existe agrupación que las proteja.

Todo el CSV es del descanso y coincide por ID con NBA. Su evaluación no acredita desempeño en Q1, Q2 en progreso, Q3, inicio de Q4 o WNBA, aunque el mismo artefacto se utilice en el flujo general. Esto es cobertura pendiente de validar, no demostración de que todas esas predicciones sean incorrectas. No ampliar a Q4 tardío ni prórrogas para resolverlo: siguen fuera del producto.

### H11-03 — La base apilada procede de otra ruta y el test externo no está aislado en toda la cadena

El generador identificado para `base_model_prediction` utiliza un XGBoost por fold, 500 árboles, learning rate 0.05, depth 6, con cinco folds aleatorios y variables numéricas del dataset live. Es un intento explícito de OOF: no corresponde llamarlo simplemente predicción dentro de muestra por defecto.

Sin embargo, no es el bundle por liga que usa el predictor principal, generado por `core/training.py` con otras variables/parámetros. Tampoco se guarda aquí una identidad de base compatible. Debe comprobarse cuánto cambia la distribución al alimentar Huber con el bundle live real; no atribuir el MAE del CSV al modelo base principal.

El OOF se prepara antes de dividir train/test del Huber. El script no construye folds internos separados para cada partición externa: los modelos que producen algunas entradas de train del Huber pueden haberse ajustado usando objetivos de filas que luego caen en su test externo. Cada fila OOF puede estar excluida de su propio modelo y aun así no existir un holdout independiente de toda la cadena.

Es una limitación verificable del procedimiento propuesto por esas fuentes, condicionada a su procedencia; no se ha medido el sesgo numérico de la métrica del artefacto actual. Revisar generadores y diseñar evaluación de toda la cadena antes de corregirlo. Los problemas OOF de `core/training.py` son distintos y no se trasladan automáticamente a este CSV.

### H11-04 — Guardado sin condición de aceptación ni identidad de versión

Después de calcular MAE, el script siempre llama a `joblib.dump` con la ruta de producción. No exige mejora sobre base o Pacer, ni compara contra el artefacto existente. Un predictor doble que devolvía cero, claramente peor que la base, llegó igualmente al guardado simulado.

No se escribe un manifiesto con métricas, tamaño/fechas de muestra, columnas, dependencias, identidad de modelos de entrada, liga, hashes o fecha de entrenamiento. El texto de coeficientes no basta para verificar procedencia. La etiqueta `v5.0_huber` de inferencia es un literal de otro módulo, no una versión guardada por este entrenador.

El modelo y sus coeficientes se escriben secuencialmente en rutas fijas, sin temporal/rename ni rollback. Un fallo tras guardar el joblib puede dejar un texto anterior. Es riesgo de persistencia por inspección, no corrupción demostrada de los archivos presentes. No ejecutar el script como simple diagnóstico: sí puede reemplazar el modelo usado en live.

### H11-05 — Limpieza limitada a NA, errores sin diagnóstico controlado

`dropna` elimina filas incompletas en las tres columnas requeridas, pero no informa cuántas se descartaron ni comprueba tipos, finitud, tamaño mínimo o coherencia temporal. El CSV actual no presenta NA/inf en esas columnas, por lo que esos escenarios no explican un fallo presente por sí solos.

Pruebas sobre el cuerpo real con doubles:

- Una fila NA se elimina antes de la división.
- Una columna requerida ausente produce `KeyError`, sin manejo local.
- Una cadena no numérica falla al evaluar la base antes del ajuste.
- `dropna` conserva infinito. El doble de métricas permite observar esa ausencia de limpieza; sklearn real puede rechazarlo ya en la métrica, sin llegar al ajuste.

No hay `try/except` de entrenamiento/guardado ni validación estructurada. Los errores tempranos impiden guardar, lo cual evita reemplazo en esos casos; no confundirlo con una garantía de publicación atómica.

## Decisiones y comportamientos que no se consideran defectos por sí mismos

- Se excluyen explícitamente objetivo, ID y variables Adjuster/RF del DataFrame del Huber. Los artefactos RF antiguos no son un requisito para ejecutar este entrenador.
- Base, Pacer y Huber se comparan en las mismas filas test; esto permite una comparación local coherente, aunque no resuelve los límites temporales y de procedencia descritos.
- Se guarda el estimador ajustado sólo con train. No volver a ajustarlo con test conserva el modelo que se evaluó; no es por sí solo un error. Un eventual reajuste con todo el histórico requeriría preservar la evaluación independiente y versionar el artefacto resultante.
- Huber predice el total completo, igual que espera la integración. No sumar la base de nuevo a su salida.
- No se exige que coeficientes sumen uno ni se restringe su signo. Es una regresión, no una media ponderada obligatoriamente normalizada. Tampoco se propone escalar entradas automáticamente: ambas se expresan en puntos y cualquier transformación debe conservarse en inferencia.
- `numpy` y `LinearRegression` son imports sin uso local. Su retirada queda para la fase de mejoras.

## Comprobaciones y límites

`.diagnostics/audit_train_meta_learner.py` ejecuta la función extraída por AST con DataFrames reales/sintéticos y dobles de split, estimador, métricas y escritura. El split reproduce la permutación NumPy con semilla 42 y redondeo superior del tamaño test; no ejecuta sklearn. Ningún `fit` real ni escritura de producción se realiza.

Pasaron **18 comprobaciones**. En el diagnóstico se corrigió una expectativa: una columna de strings no llega al estimador, pues falla antes en el cálculo de MAE. El informe refleja ese orden. El producto no se modificó.

Como referencia aritmética del CSV y las 312 filas reconstruidas: MAE base 11.3491, Pacer almacenado 11.7244 y extrapolación pura de descanso 14.7276. **No son MAE de Huber**, ni una comparación temporal, ni demuestran causalidad o rentabilidad. No se ha evaluado el joblib real ni se identifican estas cifras con las métricas guardadas de NBA o de modelos Chronos anteriores.

Se comprobaron hashes sin cambios del script, CSV, joblib objetivo y texto de coeficientes. SHA-256 del módulo: `3725406b79fd749b1c39434e8f76512790e6ea779a4e6f90a0cf619bef9bdae3`.

## Continuación

Siguiente módulo sugerido: `scripts/generate_base_predictions.py`, para revisar completamente el OOF, selección de columnas y relación entre base apilada y base live. Después seguir la procedencia hacia los generadores históricos. No se aplican migraciones, reentrenamientos ni cambios de modelos antes de completar el conocimiento y acordar los cambios clave con el usuario.
