# Contexto y contraste de Documentation

Revisión: 16 de octubre de 2026. Se leyeron completos los **24 documentos de la raíz de `Documentation/`**, incluidos ejemplos, propuestas, checklists y el análisis extenso Knicks–Nets. Se contrastaron con los informes de `auditoria/` elaborados durante esta conversación y se actualizaron las conclusiones afectadas. La documentación nueva de esta revisión no se cuenta como fuente histórica.

No se ejecutaron comandos, código de ejemplo, instalaciones, redes, despliegues, automatizaciones ni entrenamientos de las guías. Las consultas adicionales al código fueron puntuales para comprobar nombres/conexiones; no constituyen la auditoría completa de tracker, cuotas o metamodelo.

## Lo que aclara y cómo cambia la interpretación

### 1. Hubo también una arquitectura CatBoost

Además de los diseños de reglas/Reality Check, LightGBM y Huber identificados en `unsure`, varios documentos describen `MetaModeloAI` con CatBoost y un archivo `.cbm`. El checkout actual no contiene el fuente `metamodel/meta_model_ai.py` citado; la integración consultada carga Huber. Esto amplía el contexto histórico, **no es una avería ni una orden de restaurar CatBoost**.

Los diagramas que ponen ajustes multiplicativos antes de CatBoost no definen por sí solos el orden correcto del Huber actual. Tampoco que todos los indicadores impresos deban modificar automáticamente la predicción. Antes de cambiar el pipeline, verificar qué entradas y objetivo tuvo el artefacto concreto.

### 2. La separación EMA/directas fue intencionada

`feature_documentation.md` y `NEW_FEATURES_AND_MODELS_UPDATE.md` explican que las métricas que ya agregan tendencias debían ir directamente, reservando EMA para un subconjunto. El número «52% menos features diluidas» aparece como beneficio declarado, sin experimento reproducido aquí.

**Aclarado como diseño, no pendiente de reparación:** no aplicar EMA a todas las variables. Siguen siendo observaciones distintas F06-03 (pérdida de la serie de resultados al repetir un agregado) y F06-04 (variables directas calculadas que no se copian al resultado). La documentación no demuestra que esos contratos se implementaran completamente ni que deban eliminarse las directas.

### 3. El análisis más alarmante mezcla cantidades diferentes

`FALLO_PREDICCION_ANALISIS_DETALLADO.md` resta **139 puntos parciales** a una **predicción final de 269.5** y lo denomina error de +130.5/93%. Esa diferencia no es un error final de predicción. No queda validada la prioridad de emergencia ni las mejoras porcentuales calculadas a partir de esa comparación.

Otros problemas del razonamiento documental:

- Compara puntos/minuto del total de ambos equipos con un ejemplo por equipo (110/48), sin hacer coincidir unidades.
- Llama a 24 minutos tanto 50% como, en otra sección, 75% del partido. Mezcla Q3 iniciado con Q3 finalizado.
- Dice «246.5/2 ≈135», cuando la mitad es 123.25.
- Afirmaciones como «estadísticamente imposible» o «SIEMPRE desacelera Q4» no se derivan de los datos aportados.
- En un punto trata 232 como resultado final de otro caso y, más adelante, como realidad final estimada por ritmo. No hay una reconciliación de resultados fuente que permita evaluar ese ejemplo.
- Los snippets contienen placeholders, código incompleto y un ejemplo de parcial negativo; sus tests propuestos no acreditan una reproducción real ni una mejora medida.

**Retirado como evidencia de avería crítica actual:** el supuesto error +130 puntos y las urgencias deducidas de él. Se conserva el original con una aclaración al inicio para mantener la historia. El posible incidente con resultado final 232 requiere logs/registro final del caso correspondiente; no se declara resuelto ni se mezcla con el marcador parcial 139.

Esto no descarta investigar problemas comprobados de reloj, datos o estabilidad; corrige la justificación y el alcance de la alarma.

### 4. Caché histórica y datos live cumplen funciones diferentes

Las guías describen una caché para antecedentes estáticos y una actualización de marcador/señales durante el partido. **Mantener antecedentes en caché durante Q3 no implica por sí solo que falte información live.** El plan antiguo de poner `pre_game_cache=None` no se adopta como arreglo: V05-03 mostró que el predictor actual no reconstruye todas las estáticas si falta esa caché.

Lo pendiente es integridad, cobertura y actualización correcta de cada familia, no desactivar una optimización por el cuarto. El «70% mejora CPU» continúa como afirmación no medida en esta auditoría.

### 5. FTI y parsing dict/list tienen correcciones anteriores documentadas

`live_signals_troubleshooting.md` documenta el cambio de un baseline FTI 0.5 a aproximadamente 0.17 y la admisión de listas/dicts en `_parse_massive_stats_json`. Ambos patrones están presentes en el fuente actual.

**No reabrirlos como fallos antiguos sin corregir.** Siguen pendientes la calibración/unidades del FTI y los casos de parsing de valores registrados en A04, que son distintos. Un valor cero, una racha inactiva o un valor neutro no prueban por sí solos una caída de SofaScore. Tampoco estadísticas disponibles equivalen a cobertura de las columnas históricas del modelo.

### 6. El almacenamiento híbrido protege más que la consola

Tres guías describen registros diarios, individuales, de temporada e índices mensuales para evaluar y aprender. Existen esos directorios y métodos relacionados en `metamodel/prediction_tracker.py`; `track_prediction` contiene escritura estacional y llamada al índice mensual. Por tanto, **no es correcto tratar todo el almacenamiento híbrido como una mera idea sin implementación**.

El alcance exacto sigue por auditar: el tracker actual no define el `update_prediction_result` ni varios métodos de aprendizaje citados por las guías. Un método o directorio presente no acredita atomicidad, sincronización, deduplicación, actualización de resultados o aprendizaje automático completos. No reorganizar rutas ni borrar registros al limpiar la interfaz manual.

## Inventario de los 24 documentos leídos

Enlaces relativos a este informe. Clasificación de contenido, no juicio global de utilidad ni autorización de borrado.

| Documento | Aporte | Estado frente al código revisado |
| --- | --- | --- |
| [DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md) | Índice y relaciones entre guías | **Navegación histórica.** Incluye enlaces a documentos que no están en esa carpeta y capacidades antiguas. Se añade aviso hacia la auditoría, conservando el índice |
| [system_overview.md](../system_overview.md) | Visión general, señales, RF/CatBoost, transporte Playwright, despliegue | **Mixto/histórico.** No describe la extensión ni el recorrido Huber actual. Sus afirmaciones de rendimiento y producción no se verificaron |
| [api_reference.md](../api_reference.md) | Contratos de cliente, features, predicción, alertas y training | **Referencia desactualizada.** Cita funciones/excepciones/configs que no corresponden a los módulos auditados; varios ejemplos son pseudocódigo |
| [deployment_guide.md](../deployment_guide.md) | Instalación local, contenedores, cloud, métricas y mantenimiento | **Guía/plantillas, no despliegue acreditado.** Flags `--serve`, `--batch`, `--interval` y funciones de entrenamiento citadas no aparecen en las entradas auditadas. No ejecutar comandos como procedimiento vigente |
| [COMPREHENSIVE_FEATURE_GUIDE.md](../COMPREHENSIVE_FEATURE_GUIDE.md) | Intención de variables, fórmulas, interacciones y ablación | **Diccionario de diseño parcial.** Varios nombres/unidades difieren del código; mercado tiene definiciones distintas incluso de `unsure/FEATURES_DICTIONARY.md` |
| [feature_documentation.md](../feature_documentation.md) | Catálogo amplio, directas, EMA, parámetros, etapas | **Contexto valioso, no lista del bundle.** Documenta la separación EMA/directas; repite secciones y mezcla features históricas/live/anteriores |
| [NEW_FEATURES_AND_MODELS_UPDATE.md](../NEW_FEATURES_AND_MODELS_UPDATE.md) | Describe familias psicológicas y por cuarto, RF/CatBoost y tracker | **Actualización de otra etapa con partes presentes.** Los helpers avanzados existen, pero no todas las funciones de orquestación citadas ni sus promesas de consumo/precisión |
| [live_features_analysis.md](../live_features_analysis.md) | Intención O/U, exclusiones y ajustes de señales | **Recorrido anterior.** Su conclusión de que todas las señales ajustan directamente no describe la ruta actual comprobada en V05 |
| [signal_processing_guide.md](../signal_processing_guide.md) | Ejemplos de normalización, thresholds, multiplicadores | **Semántica histórica.** A/TO y rebotes normalizados difieren de las salidas brutas activas; no intercambiar helpers sin migración |
| [signal_learning_explanation.md](../signal_learning_explanation.md) | Entrenamiento sobre residual de predicción ya ajustada y señales simuladas | **Diseño anterior.** `_simulate_historical_signals` permanece, pero no se llama desde el entrenamiento activo auditado. No prueba aprendizaje de señales por Huber actual |
| [live_signals_troubleshooting.md](../live_signals_troubleshooting.md) | Diagnóstico de cobertura, corrección dict/list y baseline FTI | **Mixto, con dos correcciones corroboradas.** Recomendaciones de ajustes, rangos y tests requieren el contrato actual; «resuelve95%» no se comprobó |
| [FEATURE_IMPUTATION_FIX_SUMMARY.md](../FEATURE_IMPUTATION_FIX_SUMMARY.md) | Incidente73/153, calidad, fallbacks y avisos | **Resumen de cambios y objetivos.** El fallback inteligente nombrado ya no tiene implementación actual, aunque queda una llamada alternativa. «<20%» es objetivo, no medición vigente |
| [prediction_misalignment_plan.md](../prediction_misalignment_plan.md) | Evitar uplift tardío, regresión, techo, reloj y telemetría | **Plan**, con checklist sin marcar. Aclara intención conservadora; no es contrato vigente ni evidencia de que deban aplicarse todos sus factores |
| [FALLO_PREDICCION_ANALISIS_DETALLADO.md](../FALLO_PREDICCION_ANALISIS_DETALLADO.md) | Diagnóstico antiguo Knicks–Nets y propuestas extensas | **Análisis que necesita corrección de interpretación.** Se retira la alarma +130 basada en comparar total final con parcial y se añade nota al original |
| [METAMODELO_AI_DOCUMENTATION.md](../METAMODELO_AI_DOCUMENTATION.md) | CatBoost `MetaModeloAI`, features y safety rails | **Arquitectura anterior.** Fuente/modelo específicos no son la integración Huber actual. No se restaura por leerlo |
| [METAMODELO_INTEGRATION_DESIGN.md](../METAMODELO_INTEGRATION_DESIGN.md) | Propone unir tracker y aprendizaje en MetaModeloInteligente | **Diseño de migración.** Incluso prevé borrar `storage/`; esas instrucciones no son autorización ni prueba de migración completa |
| [METAMODELO_UNIFIED_ARCHITECTURE.md](../METAMODELO_UNIFIED_ARCHITECTURE.md) | Declara API unificada de predicción/registro/aprendizaje | **Afirmación histórica con parte de la relocalización presente.** Tracker está en `metamodel/`; no se encontraron los métodos/clase unificada citados como fuentes actuales |
| [HYBRID_STORAGE_SYSTEM_DESIGN.md](../HYBRID_STORAGE_SYSTEM_DESIGN.md) | Diseño diario/individual/estacional/mensual | **Diseño parcialmente corroborado por estructura actual.** Conservación de registros es importante; fórmulas y ejemplos no validan implementación completa |
| [HYBRID_STORAGE_IMPLEMENTATION.md](../HYBRID_STORAGE_IMPLEMENTATION.md) | Declara implementación y actualizaciones de resultados | **Mixto.** Estructura híbrida presente; update_result/aprendizaje y sincronización de todas las copias no se acreditan con las consultas actuales |
| [PREDICTION_TRACKER_DOCUMENTATION.md](../PREDICTION_TRACKER_DOCUMENTATION.md) | Esquemas, IDs, métricas, reportes y métodos | **Referencia parcial con ejemplos divergentes.** Se observan varios esquemas y métodos no definidos actualmente. Auditar el tracker antes de tocar datos |
| [api_odds_integration_summary.md](../api_odds_integration_summary.md) | Búsqueda de línea y exportación automática, respaldo manual | **Flujo parcialmente corroborado por el usuario y live_system.** Fuente activo está en `analysis/odds_api.py`; el documento cita `core/odds_api.py` y un exporter ausente como fuente. Contiene una credencial literal: incluirlo al revisar C02-04, sin copiarla al informe |
| [odds_extraction_system.md](../odds_extraction_system.md) | Colección multimerkado, validación y dataset cuotas | **Subsistema descrito, no cobertura actual probada.** Varias fuentes citadas no están en `core/`; hay algunos rastros `.pyc` y demos. No confundirlo con el cliente de líneas activo |
| [ou_prediction_workflow.md](../ou_prediction_workflow.md) | Workflow histórico+live+ajustes+alertas | **Descripción de otra ruta.** Útil para intención de separación de señales/caché, no para imponer el pipeline actual ni consumo de todas sus features |
| [ou_performance_metrics.md](../ou_performance_metrics.md) | Propuestas de accuracy, Brier, impacto de señales y dashboard | **Especificación/ejemplos de métricas**, no resultados medidos. Algunos snippets omiten entradas como línea/std_dev o simplifican mercados; no interpretar porcentajes de ejemplo como rendimiento real |

## Contradicciones documentales a no trasladar al código

- Las definiciones de mercado no son uniformes: una guía llama `market_total_diff` a línea menos predicción, otra a promedio histórico menos línea; `line_volatility_index` aparece tanto como movimiento temporal de cuotas como Z-score de total. El código actual muestra su fórmula pretendida, aunque F06-01 verifica un error de lookup. No elegir una guía completa como autoridad única.
- Varias guías dicen que el modelo base usa scores live y que se entrenan RF/CatBoost junto a él. El entrenamiento actual selecciona históricas y sólo XGBoost. Esta diferencia no obliga a añadir columnas live ni recuperar modelos antiguos.
- Configuraciones inspiradas en los planes (`REALITY_CHECK_V2`, `CACHE_CONTROL`, `DATA_QUALITY_CONFIG`) existen, pero la auditoría C02 mostró que no se consultan por esos imports. **Su presencia demuestra un rastro de implementación/configuración, no que todas las reglas estén activas.**
- Múltiples guías prometen atomicidad, millones de registros, latencia, CPU/precisión mejoradas o aprendizaje automático completo. No se han reproducido esos benchmarks o garantías; se preservan como afirmaciones históricas, objetivos o requisitos para contrastar.
- El almacenamiento se describe tanto con registros planos como con bloques `metadata/matchup/predictions/result_tracking`, y mezcla estados en español/inglés y precisión bool/texto. Antes de migrar, inventariar formatos reales y consumidores. No modificar registros basándose en un JSON de ejemplo.
- Las instrucciones de cloud, cron, instalación, autenticación y backups son ejemplos. No se crean servicios ni se modifica la máquina por leerlas. El transporte actual por extensión debe preservarse.

## Reconciliación concreta de la auditoría

| Tema | Tratamiento actualizado |
| --- | --- |
| Presencia de RF/Optuna, CatBoost/LightGBM/reglas anteriores | Contexto histórico, no fallo por ausencia ni componente a restaurar automáticamente |
| EMA sólo para un subconjunto | Decisión deliberada; no «completar» todas las familias con EMA |
| Caché prepartido durante Q3 | No se considera error por existir. Pendientes cobertura e integridad, no su eliminación por etapa |
| Baseline antiguo FTI0.5 y parser que sólo aceptaba dict | Correcciones previas corroboradas; no reabrir como causas antiguas sin reparar |
| Supuesto error final +130.5 de Knicks–Nets | Inferencia documental inválida retirada como evidencia de gravedad actual |
| Sistema híbrido de almacenamiento | Parte implementada corroborada por código/estructura; no desechar como plan vacío. Integración de resultados/aprendizaje sigue sin auditar |
| Claves, ceros, parciales, imputación, ventanas y validación final ya reproducidos | Se mantienen con sus límites. Estas guías no prueban que sean intencionados todos sus efectos ni cuantifican su impacto en modelos guardados |

La revisión no cambia las decisiones del usuario: auditoría primero, un módulo por turno, sin limpieza funcional, sin recomendaciones de Q4 tardío ni OT, y retirada posterior de la interfaz manual de predicción. Registrar resultados para evaluación/aprendizaje es una capacidad distinta que debe preservarse o acordarse por separado.

## Continuación

Próximo módulo sigue siendo `analysis/metamodel_integration.py`. Después hay que revisar entrenadores/datasets del metamodelo, tracker/resultados y cliente de cuotas con estos documentos como contexto. No se declaró ninguno de esos módulos completamente auditado mediante esta lectura.

Cambios de esta revisión: este informe, notas en `00_ESTADO.md` y los informes afectados, aviso de vigencia en el índice y aclaración en el análisis Knicks–Nets. El cuerpo histórico de los documentos se conserva. Ningún cambio funcional, ejecución de planes o eliminación.
