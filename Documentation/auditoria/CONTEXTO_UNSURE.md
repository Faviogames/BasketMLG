# Contexto histórico: documentos de unsure

Revisión: 16 de octubre de 2026. Se leyeron los 15 documentos Markdown encontrados en `unsure/`, incluidos sus bloques de propuesta y pruebas. Se hicieron consultas puntuales de nombres, rutas y conexiones con el código actual. No se ejecutó código de esos documentos ni se implementaron sus instrucciones. Los originales quedan intactos.

Esta revisión complementa la auditoría por módulos; no sustituye la revisión pendiente de la integración, entrenadores del metamodelo, tracker y scripts.

## Qué contexto aporta

La intención del proyecto no era una simple predicción histórica: **usar XGBoost como referencia y una segunda capa que la contraste con el partido en marcha**. Esto concuerda con lo explicado por el usuario sobre RF, Optuna y la búsqueda de un metamodelo.

Los documentos muestran varias familias de diseño, que no deben confundirse:

1. **Meta-modelo de reglas/contexto:** `MetaModeloInteligente`, `AnalizadorContexto`, detección de outliers, Reality Check y aprendizaje de parámetros. Los análisis de noviembre de 2025 se centran principalmente en este sistema.
2. **Modelo live aprendido con LightGBM:** visión «Anchor & Navigator», simulación de snapshots y una corrección aprendida sobre una predicción histórica. Hay entrenadores y artefactos LightGBM en el checkout, pero su existencia no acredita uso en la inferencia actual.
3. **Integración actual Huber/Chronos:** la cabecera y ruta de `analysis/metamodel_integration.py` apuntan a `metamodel/models/meta_learner_v1_final.joblib`; consulta Pacer/Runner. La auditoría completa sigue pendiente. No atribuirle automáticamente mecanismos de los otros diseños.

Estas familias explican responsabilidades y nombres conservados. **No constituyen por sí solas una cronología completa de sustituciones**: faltan trazabilidad de versiones, decisiones de retirada y relación exacta entre scripts/datasets/artefactos.

## Inventario y clasificación

Los enlaces de esta tabla son relativos al informe. «Encontrado» se refiere al checkout actual, no prueba que una propuesta nunca se implementara en otra versión.

| Documento original | Tipo y contenido | Contraste con el estado actual |
| --- | --- | --- |
| [PROJECT_VISION.md](../../unsure/PROJECT_VISION.md) | Visión y propuestas: Anchor histórico + Navigator live, intervalos dinámicos, regímenes, mercado | **Contexto de producto valioso**, no especificación literal del algoritmo actual. Identifica Navigator con LightGBM y propone mejoras futuras |
| [AI_CONTEXT.md](../../unsure/AI_CONTEXT.md) | Mapa e instrucciones para desarrolladores de una etapa anterior | **Mixto/desactualizado.** Coinciden bundles dict, `line_odds` e import local contra ciclo; no coinciden motor live, todas las rutas ni el catálogo de variables. No usar su autodeclaración de «fuente de verdad» para desplazar código comprobado o decisiones actuales del usuario |
| [MODULE_MAP.md](../../unsure/MODULE_MAP.md) | Mapa de responsabilidades y workflows | **Guía histórica.** Cita funciones que no existen con esos nombres actuales, como `train_single_model_xgboost`, `calculate_market_features`, `predict_live`; requiere verificar cada referencia |
| [FEATURES_DICTIONARY.md](../../unsure/FEATURES_DICTIONARY.md) | Diccionario de intención y fórmulas, especialmente mercado | **Mixto.** Fórmulas de mercado aclaran el significado pretendido; varios nombres, unidades y descripciones difieren del cálculo actual. No prueba que las variables lleguen al modelo |
| [ANALISIS_PREDICCIONES_SEMANA.md](../../unsure/ANALISIS_PREDICCIONES_SEMANA.md) | Informe histórico de 13 predicciones, 12 resueltas, 10–17 noviembre 2025; recomendaciones | **Diagnóstico histórico pendiente de reconciliar con registros.** Aporta ejemplos de sobrecorrección y confianza, pero no certifica rendimiento actual |
| [ANALISIS_PATRONES_FALLO.md](../../unsure/ANALISIS_PATRONES_FALLO.md) | Reinterpreta esos casos y propone perfiles, umbrales, ensemble RC y aprendizaje | **Análisis + propuestas**, no evidencia de implantación. Se solapa con el informe semanal, no es una muestra independiente |
| [ANALYSIS_REALITY_CHECK_PENALTY.md](../../unsure/ANALYSIS_REALITY_CHECK_PENALTY.md) | Problema de unidades de ritmo y corrección propuesta en `metamodel/outlier_detector.py` | **Diagnóstico/propuesta de otro componente.** Ese fuente no está actualmente en `metamodel/`. Revisar unidades de equipo/partido antes de reutilizar su fórmula de ejemplo |
| [PLAN_MEJORAS_IMPLEMENTACION.md](../../unsure/PLAN_MEJORAS_IMPLEMENTACION.md) | Roadmap fechado 17 noviembre 2025 con código propuesto, rollout y objetivos | **Plan.** Checklists sin marcar, funciones placeholder y rutas propuestas no encontradas como fuentes actuales. No convertir objetivos de acierto o rollout en resultados o autorizaciones |
| [IMPLEMENTATION_CODE_SNIPPETS.md](../../unsure/IMPLEMENTATION_CODE_SNIPPETS.md) | Migrar tres capacidades de `alerts.py`: perfil psicológico, volatilidad y alta presión | **Borrador de implementación/migración**, con checklist sin completar. Sus frases «completa y probada» no prueban que se migrara ni permiten borrar `alerts.py` u otros módulos |
| [COMPREHENSIVE_BUG_ANALYSIS_AND_SOLUTIONS.md](../../unsure/COMPREHENSIVE_BUG_ANALYSIS_AND_SOLUTIONS.md) | Arreglar Q3, Run Detector con contexto y aprendizaje con decay | **Plan parcialmente contrastable.** El arreglo de Q3 está presente; Run Detector actual sigue sin el argumento/contexto descritos. El sistema de aprendizaje fuente citado no está aquí |
| [ADAPTIVE_SPLIT_DETECTOR_FIX_SUMMARY.md](../../unsure/ADAPTIVE_SPLIT_DETECTOR_FIX_SUMMARY.md) | Declara corrección Q3 con tiempo real y umbral diez puntos, más nueve tests | **Implementación corroborada:** coincide con la función actual y las nueve pruebas ejecutadas en la auditoría 09. Beneficios porcentuales esperados siguen sin validación empírica aquí |
| [INTEGRATED_TESTING_VALIDATION_PLAN.md](../../unsure/INTEGRATED_TESTING_VALIDATION_PLAN.md) | Diseña pruebas unitarias, integración, decay, regresión y carga | **Plan de validación, no acta de ejecución.** Varios contratos/nombres no coinciden con el código actual; no se encontraron fuentes de las suites enumeradas con esos nombres |
| [APRENDIZAJE_METAMODELO_GUIDE.md](../../unsure/APRENDIZAJE_METAMODELO_GUIDE.md) | Declara activación de aprendizaje, registro de resultados, optimización tras diez registros | **Afirmación histórica no corroborada como función actual.** Fuentes/clase/métodos citados no se encontraron en código Python actual; hay rastros `.pyc` de módulos antiguos, que no acreditan aprendizaje activo. Conservarla para revisar tracker/resultados y distinguir registro de reentrenamiento |
| [RESUMEN_IMPUTACION_FIX.md](../../unsure/RESUMEN_IMPUTACION_FIX.md) | Corrección de mapeo API→estadísticas, cálculo de calidad y reducción esperada de faltantes | **Fragmentos corroborados:** están `critical_api_to_model`, `combined_stats` y el cálculo de calidad en `core/live_api.py`. La reducción de 50.3% a <10% es esperada, no demostración del 66% actual |
| [PRUEBA_STEALTH_2026-10-16.md](../../unsure/PRUEBA_STEALTH_2026-10-16.md) | Registro de prueba aislada reciente con y sin stealth | **Registro puntual coherente con el contexto de esta conversación.** Ambos controles funcionaron; no demuestra ventaja del plugin. Sigue aparcada por decisión del usuario |

## Decisiones y motivos que ahora se entienden mejor

### Evitar sobrecorrecciones, no reducir predicciones por sistema

Los informes de noviembre describen que una base alta podía ser acertada y que el Reality Check la reducía en exceso. Proponen distinguir un partido de anotación excepcional legítima de un error optimista, especialmente al descanso.

Eso explica la intención detrás de clamps, umbrales de intervención, factores conservadores, perfiles y registro de ajustes. **No demuestra que cada porcentaje literal sea correcto o que todos esos mecanismos sigan conectados.** Mantener las distinciones ya registradas entre ajustes anteriores, metamodelo activo y salvaguarda posterior.

### Fusión de alertas: intención de preservar capacidades

El borrador de migración de `alerts.py` dice expresamente que debían preservarse perfil psicológico, volatilidad dinámica y alta presión antes de retirar el módulo. Sus listas sin marcar y la ausencia actual de los fuentes de destino impiden dar por terminada esa migración.

La retirada del modo manual decidida por el usuario es una decisión distinta. Tampoco autoriza borrar funciones compartidas o dar por presentes las alertas porque un mensaje de consola diga «integradas».

### Aprendizaje: objetivo importante, estado operativo no demostrado

La guía describe registrar resultados finales, comparar error base/ajustada y optimizar parámetros. Es un requerimiento/diseño histórico que conviene preservar como referencia. Un indicador `aprendizaje_pendiente` o una predicción exportada no prueban actualización de Huber, LightGBM ni del modelo base.

No se encontró `metamodel/meta_model.py`, `context_analyzer.py`, `learning_system.py` u `outlier_detector.py` como fuentes en esa carpeta. Sí existen nombres correspondientes en `metamodel/__pycache__`, además de entrenadores y múltiples artefactos de otras versiones. Esto es compatible con un pasado más amplio, pero no explica su retirada ni permite reconstruir conexiones activas. No se ejecutaron ni descompilaron esos binarios.

### Mercado: las fórmulas documentan intención

El diccionario define explícitamente:

- `market_total_diff = promedio puntos local L5 + promedio visitante L5 - línea`.
- `implied_pace_vs_avg = línea / 2.1 - ritmo medio L5`.
- `line_volatility_index = (línea - media de totales de liga) / desviación de liga`.

Esto respalda la interpretación de F06-01: las consultas a claves inexistentes no son la fórmula descrita. Aun así, una futura corrección debe contrastar la versión que entrenó cada modelo; no sustituir variables ni reentrenar en esta fase.

## Precauciones al usar estos documentos como evidencia

1. **Plan no es implementación.** Fechas de inicio, porcentajes esperados, instrucciones «implementar inmediatamente» y checklist de borrado no son autorización actual. Se preserva el acuerdo de auditoría.
2. **Código de ejemplo no equivale a una prueba pasada.** El plan de mejoras incluye `ejecutar_caso_prueba: pass`. El plan integrado contiene tests async bajo `unittest.TestCase` ordinario y mocks/firmas que no corresponden al transporte actual. No se ejecutaron.
3. **Hay contradicciones internas.** El ejemplo del plan de detector espera Miami–NY 270.5 como outlier, mientras sus umbrales hardcoded para esos equipos son 280/285. En los snippets, `analizar` calcula los nuevos análisis después de llamar a los ajustes y no les pasa esos argumentos nuevos. Son razones adicionales para tratarlos como borradores que requieren revisión.
4. **Los informes históricos no permiten calcular rentabilidad actual.** Falta reconciliar línea realmente apostada, cuota, instante y resultado final de cada registro. Hay ejemplos donde `base + RC` no coincide con la final declarada: Boston 222.5 − 3.6 = 218.9, pero se muestra 208.5. Podría haber más ajustes no desglosados o un error documental; revisar logs antes de atribuirlo todo al RC.
5. **Muestras pequeñas y repetidas.** Los dos análisis principales reutilizan los mismos doce casos. Sus porcentajes no constituyen evaluaciones independientes ni calibración de confianza.
6. **Unidades y ligas siguen pendientes.** La visión usa 48 minutos como ejemplo; eso no anula la duración WNBA. Convertir posesiones a puntos de partido exige precisar si se cuenta un equipo o ambos. No copiar la solución de Reality Check sin esa comprobación.
7. **Documentación antigua no amplía alcance.** Cierre de Q4, clutch, países/ligas o aprendizaje manual citados no sustituyen decisiones actuales: Q1–Q3 principal, inicio de Q4 condicionado, sin OT y retirada posterior de la interfaz manual de predicción. Registrar un resultado final para evaluar/aprender es otra función.

## Qué no queda explicado todavía

Estos documentos no justifican de forma específica el umbral de 500 registros, el reemplazo universal de ceros por promedios, la sobrescritura de parciales por un box sin puntos, la escala duplicada de ritmo, la división final aleatoria por filas o el guardado del histórico por etapas para live. Algunas decisiones pudieron tener contexto que aún no tenemos; **motivo desconocido no significa ausencia de motivo**.

Tampoco prueban que las 124/188 imputaciones actuales se deban sólo al parser API. Calidad del box, cobertura de características históricas, ensamblaje y contrato del bundle son niveles distintos.

## Cómo se incorporará a la auditoría

- Mantener separadas intención histórica, implementación actual comprobada y propuestas no verificadas.
- Revisar a continuación `analysis/metamodel_integration.py` usando este contexto, sin tratar LightGBM o el antiguo meta-modelo de reglas como la ruta activa por defecto.
- Después, inventariar entrenadores/datasets/artefactos y revisar tracker/registro de resultados para aclarar qué aprendizaje existe y qué quedó como plan.
- Conservar documentos y artefactos hasta preparar propuestas concretas de migración/limpieza y consultarlas con el usuario.

En la lectura inicial no se borraron hallazgos reproducidos. Posteriormente, a petición del usuario, se actualizaron `00_ESTADO.md` y los informes afectados para cerrar correcciones anteriores, aclarar decisiones históricas y retirar interpretaciones de alarma que la evidencia no sostiene. Se conservan pruebas y observaciones con sus límites. No se modificó código funcional ni se aplicó ninguno de los planes.
