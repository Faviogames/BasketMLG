# Ejecución controlada de los modelos guardados

16 de octubre de 2026. A petición del usuario se pasó de inspección a inferencia real: se cargaron los joblibs actuales y se ejecutaron las funciones del producto. **No se entrenó, reentrenó ni publicó ningún modelo.**

## Qué se ejecutó

- Huber de `metamodel/models/meta_learner_v1_final.joblib` con `aplicar_metamodelo_ai`.
- Bundles `models/NBA.joblib` y `models/WNBA.joblib` con `build_pre_game_cache` y `get_predictions_with_alerts` reales.
- Tres escenarios de marcadores sintéticos identificados como pruebas, con equipos que existen en cada bundle. No se suministraron estadísticas live de box; el código usó sus defaults para esas entradas.
- No se invocaron navegador, extensión, API de cuotas, tracker, exportador ni monitor estabilizador. El flujo probado termina en el retorno del predictor compartido, después de su posible salvaguarda.

La caché se construyó con el histórico de cada bundle. Estos escenarios no son una reproducción temporal de un partido histórico ni una evaluación contra un resultado final. No se interpretan como recomendaciones.

## Entorno aislado y compatibilidad

No había un Python del proyecto accesible mediante `python`/`py`; se usó el runtime auxiliar Python 3.12.14, con paquetes en `.diagnostics/model-runtime*`, sin modificar una instalación del usuario. Versiones de la ejecución satisfactoria: sklearn 1.7.2, XGBoost 3.1.2, SciPy 1.16.3, pandas 2.3.3, NumPy 2.3.5, joblib 1.5.2.

La primera combinación descargada se bloqueó importando una extensión de SciPy 1.18.1, antes de cargar modelos. Se interrumpió y se identificó el punto mediante faulthandler. Con SciPy 1.16.3 se pudo continuar.

Con pandas 3.0.1 y XGBoost 3.1.2 los tres escenarios fallaban al predecir por nombres de características ausentes. Al repetir con pandas 2.3.3 y los mismos artefactos/código, los tres produjeron resultados. **Es evidencia de compatibilidad del entorno de prueba; no demuestra un fallo actual en la instalación del usuario.** No se corrigió el producto ni se desactivó la validación de nombres para conseguirlo.

XGBoost emitió una advertencia de serialización desde una versión anterior. No se regeneró ni convirtió el modelo. Esta ejecución confirma que carga y predice en la combinación probada, no equivalencia binaria de resultados con cualquier entorno anterior.

## Huber real: contrato corroborado

El artefacto cargado es `HuberRegressor`, con dos entradas en orden:

1. `base_model_prediction`: coeficiente 0.24715510795725623.
2. `pace_poisson_projection`: coeficiente 0.7577792921608889.

Intercepto: 0.09990640085563751. Coinciden con el texto auxiliar de coeficientes. Se resuelve así la falta de verificación del objeto real indicada en MI10-03 para clase/esquema/coeficientes; no su procedencia de entrenamiento ni cobertura por liga/etapa.

En un caso aislado con base 220, total observado 100 y 24 minutos NBA jugados: Pacer 200; Huber **206.0298886**, ajuste −13.9701114. La función real y el modelo guardado responden conforme a ese contrato.

## Resultados del predictor completo

| Escenario sintético | Total observado / minutos | Base real | Pacer | Huber | Retorno tras salvaguarda | Imputadas |
| --- | --- | --- | --- | --- | --- | --- |
| NBA Clippers–Nuggets, descanso | 101 / 24 | 228.3564 | 202.0 | 209.6108 | 209.6108 | 124/188, 66.0% |
| NBA mismo emparejamiento, Q3 en progreso | 133 / 30 | 228.3564 | 212.8 | 217.7948 | 217.7948 | 124/188, 66.0% |
| WNBA Sparks–Aces, descanso | 101 / 20 | 165.5126 | 242.4 | 224.6929 | 238.8586 | 84/153, 54.9% |

Las tres salidas fueron finitas y se devolvieron cinco líneas sintéticas de probabilidades. Estas líneas no son cotizaciones de una casa. No se calculó exactitud, MAE, calibración o rentabilidad a partir de estos ejemplos.

## Qué queda confirmado y qué sigue pendiente

- Los bundles y el Huber actuales son ejecutables en el entorno compatible probado. Esto complementa las pruebas anteriores con estimadores dobles; no certifica todo el sistema live.
- Se reproduce el **mismo recuento 124/188** del log NBA del usuario. No acredita que los valores o las 124 columnas ausentes sean exactamente los de aquel partido. El ensayo sin box real tampoco cuantifica su impacto predictivo.
- El campo de confianza devuelto fue **1.0 en los tres escenarios**, con cero columnas marcadas críticas, pese a la imputación superior al 50%. Confirma en inferencia real la distinción de V05-08 entre ese campo, los mensajes de calidad y la salvaguarda. No interpretarlo como certeza predictiva del 100%.
- WNBA confirma LF09-08/V05-06: con 101 puntos y 20 minutos, Pacer usa `101 / 20 × 48 = 242.4`; una extrapolación a 40 minutos sería 202. Esa comparación identifica la duración, no una predicción correcta del resultado final.
- En ese escenario WNBA la salvaguarda posterior aumenta el resultado de Huber a 238.8586. Se confirma la necesidad de distinguir base, meta y retorno final; no proponer caps ni modificar la fórmula sin acordar la política por liga.
- La diferencia entre definiciones del CSV histórico del entrenador y el Pacer live sigue siendo un asunto de procedencia pendiente. Leer el joblib confirma sus coeficientes, pero no identifica con qué CSV se entrenó.
- No se probó aquí transporte SofaScore, recepción de datos actuales, OddsAPI, guardado de resultados ni estabilización del monitor. El usuario ya confirmó antes una sesión live mediante extensión; esta prueba tiene un alcance diferente.

## Evidencia y preservación

Ejecutor: `.diagnostics/run_saved_models.py`, sin llamadas a `fit` ni métodos de guardado de estimadores. Resultado detallado: `.diagnostics/saved_models_inference.json`; salida capturada: `.diagnostics/saved_models_inference.log`. La ejecución con pandas 3 quedó separada en archivos con sufijo `_pandas3` para conservar el contraste.

Se comprobaron hashes antes/después de todos los joblibs de `models/`, los de `metamodel/models/` y el texto de coeficientes: **sin cambios**. El inventario de archivos de `prediction_records/`, tamaños y tiempos de modificación permaneció igual: **sin exportaciones nuevas ni cambios detectados en los registros**. No se alteraron archivos funcionales del proyecto.

## Dirección de trabajo

La indicación posterior del usuario de ignorar scripts de pruebas se mantiene. El análisis de sus scripts/CSV se conserva como contexto histórico, no como lista de reparaciones del sistema activo. No continuar con `scripts/generate_base_predictions.py` por defecto.

La siguiente revisión operativa sugerida sigue siendo `metamodel/prediction_tracker.py`. Las mejoras de reloj, imputación, confianza y contratos se prepararán después de completar el conocimiento y acordar cambios clave. Esta ejecución no autoriza reentrenar ni modificar fórmulas.
