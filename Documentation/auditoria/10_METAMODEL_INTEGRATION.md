# Módulo 10: analysis/metamodel_integration.py

Fecha: 16 de octubre de 2026. Lectura completa, contraste de llamadas y 20 comprobaciones aisladas. Sin red, deserialización de modelos, entrenamiento ni cambios funcionales. Se mantiene el acuerdo de comprender el proyecto antes de mejorarlo.

## Responsabilidad y contrato real

`cargar_metamodelo` carga `metamodel/models/meta_learner_v1_final.joblib` mediante joblib y conserva el objeto en `_METAMODEL`. La ruta es relativa al directorio de ejecución. `aplicar_metamodelo_ai` devuelve una pareja: predicción y metadatos.

Flujo comprobado:

1. Cargar o reutilizar el objeto del modelo. Si no se carga, devolver la base con `{'error': 'Model not loaded'}`.
2. Sumar las ocho claves `q1_home` hasta `q4_away` para obtener puntos observados.
3. Calcular Pacer a partir de ese total, `time_info` y línea opcional; calcular Runner a partir de parciales y etapa.
4. Construir un DataFrame de una fila con **dos columnas en este orden**: `base_model_prediction`, `pace_poisson_projection`.
5. Usar `model.predict(...)[0]` como predicción completa, no como offset que se sume otra vez a la base. El ajuste reportado es final menos base.
6. Devolver predicción, ajuste, etiqueta literal `v5.0_huber`, componentes y probabilidades del Pacer. Ante excepción dentro del bloque, devolver la base con el error.

La llamada principal está en `analysis/live_mode.py:1662`, dentro de `get_predictions_with_alerts`. Prepara explícitamente `played_seconds = minutes_assumed * 60`. Después puede modificar el resultado mediante Safe Projection; el monitor puede estabilizarlo de nuevo. Esta función no representa por sí sola la última cifra mostrada ni todas las transformaciones del sistema: ver A04-07/08 y V05.

`core/live_api.py` importa esta función, pero no se encontró una llamada directa a ella en ese módulo. La inferencia principal pasa por el predictor compartido. `apply_metamodel_ai_adjustment` en `analysis/live_mode.py:2087` es otro wrapper; no se encontraron llamadas a él en el código Python inspeccionado. Los comentarios de scripts que dicen que el monitor usa ese wrapper no prueban una conexión actual.

## Aclaraciones que no son averías por sí mismas

| Entrada | Uso efectivo en esta integración |
| --- | --- |
| Predicción base | Primera columna del Huber |
| Parciales Q1–Q4 y reloj | Determinan la segunda columna mediante Pacer |
| `quarter_stage` | Runner diagnóstico; no se añade como columna |
| `bookie_line` | Probabilidad del Pacer; no modifica su proyección ni entra como columna del Huber actual |
| `balance_features`, `live_pace_metrics`, `live_signals`, `team_trends`, `team_stats` | Se aceptan por compatibilidad, pero no se consultan en el cuerpo actual |
| `run_active_flag` y volatilidad Runner | Se guardan en metadatos; no entran en `X_input` |

Esto **no significa que todas esas señales sean ignoradas por el sistema completo**: algunas pueden intervenir antes, en la base, o en otras rutas. Aquí sólo se acredita su ausencia como entradas directas de esta capa. Cambiar cinco argumentos contextuales manteniendo base, parciales y reloj idénticos no cambia la salida de la integración.

El contrato de dos entradas coincide con `metamodel/train_meta_learner_huber.py`, que guarda el mismo nombre de artefacto y excluye explícitamente Adjuster y Bookie Line por las razones anotadas en ese script. Es contexto de diseño, no evidencia de que el fichero actual proceda necesariamente de esa ejecución ni de su calidad de entrenamiento. No restaurar automáticamente CatBoost, LightGBM, RF, reglas o nuevas señales.

El texto `meta_learner_coefs.txt` atribuye aproximadamente 0.247 a base y 0.758 a Pacer, con intercepto 0.100. Es un archivo auxiliar sin identidad verificada con el joblib actual: no usarlo para afirmar pesos efectivos del artefacto cargado. `metadata_chronos_v1.json` enumera nueve variables de otro modelo; no valida el esquema de dos columnas del Huber por compartir carpeta o nombre Chronos.

## Hallazgos y límites

### MI10-01 — Una degradación interna del Pacer puede llegar como entrada normal

El Pacer captura sus propias excepciones y devuelve `pacer_projection = current_total`, probabilidad 0.5. Esta integración no recibe ni comprueba un estado de calidad. Con 100 puntos y `played_seconds = 'bad'`, el cuerpo real del Pacer toma ese respaldo; el predictor recibe 100 y se devuelven metadatos normales, sin clave `error`.

La línea 92 sólo utiliza la base como respaldo si falta la clave `pacer_projection`. No cubre este caso porque la clave existe. En consecuencia, no se debe interpretar toda entrada Pacer como una proyección válida a final del partido. Es un escenario condicionado, no un incidente acreditado en la ejecución del usuario. Definir qué hacer con un especialista degradado requiere una decisión y preservar compatibilidad con entrenamiento.

### MI10-02 — Faltan validaciones de cobertura y salida en esta frontera

Se verificó contra la función real y un predictor doble:

- Un resultado `NaN` se devuelve como predicción y ajuste `NaN`, sin error. Tampoco hay comprobación explícita de finitud de las entradas.
- Un predictor que entrega −20 se acepta aunque el marcador ya sume 100. No hay en esta capa una comprobación de coherencia con puntos observados.
- Un diccionario de parciales vacío equivale a cero puntos; con reloj válido, llega Pacer cero al modelo. No se distingue ausencia de cobertura de un marcador observado 0–0.
- Una clave presente con `None` falla al sumar; se devuelve la base con error. Una clave ausente, en cambio, se convierte en cero.

Estos escenarios demuestran el contrato permisivo, **no que el Huber real haya emitido valores inválidos ni que deba añadirse un clamp arbitrario**. La salvaguarda posterior es otra capa y no sustituye por sí sola una política explícita para `NaN`, cobertura o fallos de especialistas. Antes de cambiarlo, acordar cuándo no recomendar y qué estado conservar en la salida.

### MI10-03 — La carga no valida identidad, esquema o compatibilidad del artefacto

Después de `joblib.load` el objeto se etiqueta como Huber sin comprobar tipo, disponibilidad de `predict`, columnas esperadas ni metadatos/versiones. Un objeto incompatible puede quedar cacheado y fallar en todas las inferencias posteriores. Se reprodujo con un objeto sin `predict`: una carga, dos fallos de inferencia y retorno a base en ambos casos.

Hay una única ruta y una única caché global, sin selección de liga ni duración en esta interfaz. Esto no demuestra que esté mal usar un modelo compartido; **la compatibilidad NBA/WNBA del artefacto no queda acreditada por el módulo**. Debe contrastarse al revisar entrenamiento y datos del metamodelo.

El archivo joblib objetivo existe, pero no se deserializó en esta auditoría. Por tanto, no se ha comprobado su esquema real, clase, coeficientes, procedencia ni rendimiento. La ruta relativa comparte la cuestión pendiente de rutas C02/M01; no mover el modelo aisladamente.

### MI10-04 — Caché sin renovación y reintentos sin estado de fallo persistente

Una carga válida se reutiliza durante toda la vida del módulo aunque el archivo se sustituya en disco o se cambie la fuente del loader. Esto puede ser una decisión razonable de rendimiento; implica que actualizar el artefacto no actualiza una sesión ya abierta.

Si la carga falla, se devuelve `None` sin guardarlo como estado de error: cada inferencia vuelve a intentarlo. Si se carga un objeto incompatible, en cambio, permanece en caché. Ambos comportamientos se reprodujeron con doubles. Registrar como política operativa por decidir; no añadir recarga automática o reintentos de entrenamiento durante la auditoría.

### MI10-05 — El fallback conserva un error, pero el llamador imprime una salida Chronos

Si falla carga o predicción, esta función imprime el error y devuelve la base con una clave `error`. El llamador principal usa `adjustment` por defecto cero y `chronos_prediction` por defecto el resultado devuelto, y continúa imprimiendo `Chronos Prediction` y ajuste cero. No presenta un estado distinto de forma estructurada en ese resumen.

No es un fallo silencioso absoluto: el mensaje de error sí se imprime y el dict se incorpora a `context_info`. El riesgo es atribuir en resúmenes o registros una predicción a Chronos cuando se usó la base. Se relaciona con la trazabilidad A04-07/08; no afirmar que ese fallback ocurrió en el partido del usuario.

### MI10-06 — Reloj exigido por el especialista; helper anterior fuera de la ruta principal

Con 100 puntos, 24 minutos y `played_seconds = 1440`, la entrada Pacer es 200. Si sólo se suministra `total_minutes_played = 24`, Pacer interpreta cero segundos y devuelve 400 por su rama inicial `current_total * 4`.

El wrapper `apply_metamodel_ai_adjustment` sólo entrega etapa, y reproduce esa rama. No se encontró uso activo de ese wrapper; **la llamada principal sí pasa los segundos**. No presentar esto como la causa actual de una predicción live equivocada. Los scripts `verify_metamodel_integration.py` y `reproduce_live_bug.py` también omiten `played_seconds`; sus escenarios no ejercitan correctamente el contrato actual.

La proyección activa sigue fijando 48 minutos dentro del Pacer. Eso ya está registrado en LF09-08 y V05-06: esta integración confirma su consumo, no añade un fallo independiente. No existe un argumento de liga que corrija esa duración en este recorrido. Resolver coherencia de WNBA sin introducir prórrogas, que están fuera del producto.

### MI10-07 — Probabilidades del especialista distintas de la predicción combinada

`pacer_prob_over/under` corresponden al modelo Poisson del Pacer respecto a la línea recibida. No se recalculan desde la predicción del Huber ni desde la salvaguarda posterior. Sin línea, valen 0.5/0.5 por defecto; no representan una estimación calibrada de confianza del metamodelo.

Se comprobó que cambiar línea altera estos metadatos y deja idénticas las dos entradas del predictor. No se encontraron consumidores directos de esas claves fuera del módulo en el código Python inspeccionado; los metadatos completos pueden almacenarse. Por eso se registra una distinción semántica, no un fallo demostrado de las probabilidades mostradas al usuario, que se generan en otra ruta de `live_mode.py`.

La etiqueta «Poisson Projection» no convierte la media en un modelo aprendido de posesiones: el Pacer activo extrapola puntos y reloj; usa Poisson para la probabilidad. No se evaluó su calibración ni se auditó aquí el origen del dataset de entrenamiento.

## Código candidato a revisión posterior

- Imports sin uso local: `json`, `numpy`, `Any`, `Optional`, `datetime`. No se retiraron.
- Argumentos contextuales conservados por compatibilidad y comentarios de Pulse/Chronos más amplios que las entradas efectivas. No retirar parámetros sin comprobar consumidores.
- El flag `METAMODEL_AI_AVAILABLE` del wrapper auxiliar no gobierna la llamada principal, que importa directamente la función. Los errores al importar dependencias ocurren antes del `try` del loader; el fallback de carga no los cubre.
- La construcción de metadatos comparte el bloque de predicción. Una conversión diagnóstica inválida podría descartar una predicción ya calculada. Se probó inyectando Runner con flag `NaN`, pero el Runner actual devuelve enteros controlados: no se clasifica como avería activa.
- Los scripts antiguos de verificación no prueban que Pulse influya en el Huber actual. El script de reproducción cambia estadísticas que esta integración no consulta y omite el reloj requerido; no usar sus etiquetas «bug/fix» como demostración de una reparación actual.

## Pruebas y alcance

`.diagnostics/audit_metamodel_integration.py` extrae por AST las dos funciones reales del módulo, Pacer/Runner reales y el wrapper auxiliar. Usa pandas, un loader y predictor dobles, y una función `poisson.sf` sustituida. Pasaron **20 comprobaciones**, incluida preservación del hash del módulo.

La primera ejecución detectó una expectativa incorrecta del escenario de Runner: Q3 de 40 frente a media 50 no supera el umbral de frío 0.7. Se corrigieron sólo los datos del diagnóstico a Q3 de 20 frente a media 60; el código del producto no se modificó. La ejecución final pasó completa.

Se verificaron esquema, propagación de entradas, suma de marcadores, metadatos, caché, rutas de error y fronteras de cobertura. Las cifras del predictor doble son aritmética de prueba, no predicciones del Huber real. No se ejecutó la suite completa, ni se midió MAE, calidad probabilística o comportamiento end-to-end con navegador.

SHA-256 antes/después: `145d4532de656b5ba875ad359fa61eb26eda9a372291530f9071148dbbdb341b`.

## Decisiones pendientes y continuación

Antes de mejorar: acordar estados de fallo/degradación, identificación de modelo y liga, política de recarga, cobertura mínima y significado de probabilidades. Cualquier cambio de señales o fórmulas debe considerar el artefacto y su entrenamiento; no conectar variables adicionales sólo porque los planes históricos las mencionen.

Siguiente módulo sugerido: `metamodel/train_meta_learner_huber.py`, para revisar el origen de las dos columnas, evaluación, separación de partidos/tiempo y relación con el artefacto utilizado. La lectura puntual del contrato realizada aquí no sustituye esa auditoría. Parar aquí según el acuerdo.
