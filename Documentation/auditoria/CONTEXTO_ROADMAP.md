# Contexto histórico: Roadmap

Revisión del 16 de octubre de 2026. Se leyeron completos los 12 archivos de `Roadmap/`, a petición del usuario, y se contrastaron propuestas concretas con código y auditorías anteriores. No se modificaron funciones, modelos ni registros. Los documentos originales se conservan intactos.

Los planes mezclan conversaciones de diseño, pseudocódigo, objetivos y análisis de una arquitectura anterior. Un checkbox vacío no demuestra ausencia de implementación; un nombre de variable presente tampoco demuestra que el bundle actual la utilice correctamente. Las expresiones «aprobado», «eliminar» o «reentrenar» dentro de estos archivos no sustituyen la autorización actual del usuario.

## Inventario y clasificación

| Documento | Situación contrastada | Qué conservar para una fase posterior |
| --- | --- | --- |
| `FASE 1 CONTEXTUAL SPLIT ENHANCEMENT.md` | Implementación parcial: existen `calculate_adaptive_split_detector` y `calculate_quarter_transition_shock`. El arreglo temporal del detector adaptativo ya está corroborado con nueve pruebas. Esto no acredita toda la fase ni resuelve los hallazgos distintos LF09-02/03 | Comparaciones entre periodos normalizadas por tiempo realmente jugado, con nombres y fórmulas definidos. Evitar comparar un Q3 parcial con un Q2 completo sin ajuste |
| `FASE 2 PACE & POSSESSIONS ENRICHMENT.md` | Propuestas de volatilidad de ritmo, aceleración de posesiones y sostenibilidad; existen métricas relacionadas, pero no queda acreditado el paquete propuesto de ventanas temporales | Experimentos acotados tras unificar unidades y reloj. Ventanas de cinco minutos o diez posesiones requieren series observadas con esa cobertura |
| `FASE 3 ADVANCED CONTEXT & META-FEATURE EXPANSION (PRIORIDAD MEDIA).md` | Propuestas y conceptos parcialmente solapados con balance, clutch y garbage actuales; no se demuestra su conexión completa con el Huber | Contexto dinámico y calidad de señales dentro de Q1–Q3/inicio permitido de Q4. Las recomendaciones de Q4 tardío quedan fuera del alcance decidido |
| `FASE 4 VOLATILITY & ACCELERATION MODELING (PRIORIDAD MEDIA).md` | Run score, volatilidad de tiros libres, adaptación y shocks son hipótesis; no dar por implementado un detector temporal por mostrar una señal Run Detector | Series de actualizaciones y deltas observados; evaluar estabilidad sin inventar posesiones ni eventos que la fuente no entrega |
| `Ajustes Finales.md` | Plan de robustez/validación y despliegue por lotes, no evidencia de aplicación completa | Validación por tiempo/temporada/contexto, versiones y reversión. El split final y la cobertura OOF actuales requieren revisión según T08 |
| `OVER_OPTIMISM_CORRECTION_PLAN.md` | Parte concreta implementada: varianza ORTG y diferencias EMA 3/11 de puntos y TS en `core/features.py:1323–1369`. Variables de fricción existen como proxies, no como el cálculo ideal propuesto | Conservar el motivo histórico de enfriamiento/fricción. Comprobar alimentación live y cobertura del bundle antes de añadir otra vez las mismas familias |
| `Over_Under_Historical_Lines_Feature_Plan.md` | Propuesta ampliada de históricos de mercado; no se encontró el módulo propuesto `core/ou_line_features.py`. Sí hay antecedentes de mercado y `performance_vs_closing_line` | Cotizaciones identificadas por partido, casa y momento; cobertura y semántica verificables antes de crear más columnas |
| `Prediction_Tracking_System_Plan.md` | Parcialmente realizado por una estructura diferente: tracker en `metamodel/`, integración live y almacenamiento de registros. No se encontraron los módulos propuestos `core/result_updater.py`, `analysis/prediction_analyzer.py` ni `core/data_exporter.py` | Completar el ciclo predicción–resultado y trazabilidad, después de auditar el tracker existente; preservar datos y consumidores |
| `Prediction_Tracking_TODO.md` | Checklist desactualizado respecto del registro y almacenamiento existentes. No usarlo como inventario ejecutable ni recrear el tracker en otra carpeta | Separar lo existente de resultados pendientes, exportación, métricas y comandos realmente conectados. No asumir automatización operativa |
| `ANALISIS_COMPLETO_PREDICCIONES_NOVIEMBRE_2025.md` | Análisis histórico de doce casos, con inconsistencias internas y relación incompleta entre etapas. No es un benchmark del sistema actual | Investigar trazas originales y medir por separado base, correcciones y salida mostrada; no fijar reglas por equipo a partir de estos ejemplos |
| `REALITY_CHECK_IMPROVEMENT_ROADMAP.md` | Propuesta de noviembre de 2025 para reglas/Ridge y un metamodelo unificado; incluye pseudocódigo incompleto. No describe el Huber actual | Idea de atribuir cada corrección y evitar ajustes duplicados, sujeta a revisar el flujo activo. No ejecutar sus instrucciones de borrar bloques |
| `strategic_plan_beat_bookies.md` | Plan de una etapa CatBoost. La integración actual ya suministra `base_model_prediction` junto con `pace_poisson_projection` al Huber; no procede asumir todavía que el metamodelo ignora la base | Comparación contra referencias simples, calibración y evaluación registrada. Promesas de accuracy/ROI y supuesta ventaja temporal no están demostradas |

## Contexto que aclara observaciones anteriores

### Enfriamiento, volatilidad y fricción

El plan de corrección del optimismo explica la introducción deliberada de estas familias: no son restos sin propósito. `core/features.py` calcula varianza del ORTG de los últimos cinco registros y diferencias EMA 3/11. Su presencia no demuestra que lleguen correctamente al live ni qué columnas usa cada artefacto: siguen vigentes las diferencias de ensamblaje de V05 y F06.

Los comentarios actuales declaran expresamente las aproximaciones: `opponent_pace_ema_5` usa la media simple del ritmo de los cinco partidos recientes propios; `opponent_defensive_efficiency_ema_5` usa la media del ORTG propio en esos encuentros. Pueden representar parte del contexto enfrentado, pero no separan estilo propio y rival ni implementan una EMA del perfil histórico de cada oponente. Se aclara la intención; no se autoriza cambiar fórmula o nombre, porque alteraría el contrato de los modelos.

### Metamodelo que recibe la predicción base

El problema central del plan estratégico era que una versión CatBoost no recibía la base. En `analysis/metamodel_integration.py:100–105` las dos entradas actuales del Huber son precisamente la base y la proyección Pacer. Se cierra esa sospecha documental específica. Esto no acredita entrenamiento sin fugas, calibración, ni aprovechamiento de todas las señales live; la auditoría completa de integración sigue pendiente.

### Tracking y listas de tareas

El registro automático y parte del almacenamiento ya existen; no deben reconstruirse ni borrarse por una ruta antigua del plan. Las rutinas de resultados, garantías de escritura, evaluación y comandos se contrastarán al auditar sus consumidores. Registrar o exportar una predicción no demuestra aprendizaje automático posterior. Una eventual entrada manual de resultado final es distinta del modo manual de predicción que el usuario decidió retirar.

## Análisis históricos que no sustentan cambios actuales

En el análisis de noviembre aparecen estas inconsistencias comprobables sin consultar fuentes externas:

- Con los doce pares base/resultado de su anexo, las medias son **236.5083 y 231.0833**, diferencia **+5.425**; no reproducen 236.8, 230.3 ni +6.5. Esto tampoco demuestra un sesgo del modelo actual.
- El ID `3F0CD5FA` se asigna tanto a Phoenix–Indiana como a Utah–Atlanta. Es necesario resolver identidad y procedencia antes de tratar la tabla como evaluación reproducible.
- Para Boston, 222.5 − 3.6 = 218.9, distinto de 208.5. Para Milwaukee, 214.5 − 4 = 210.5, distinto de 199.5; para Phoenix, 230.6 − 3.7 = 226.9, distinto de 215.6. Puede haber otras capas omitidas; no se puede atribuir toda la caída al RC mostrado.
- Los aciertos OVER/UNDER no se reconstruyen sin las líneas de referencia correspondientes. Un total real por encima de una predicción no determina por sí solo si acertó una recomendación de mercado.
- Comparar 3/7 con 4/5, o dos casos pregame con nueve al descanso, no aísla el efecto del RC ni demuestra superioridad de una etapa. Las poblaciones y contextos difieren.

El roadmap de Reality Check usa otra muestra pequeña y otra arquitectura. Sus conclusiones y las del análisis no deben agregarse como evidencia independiente sin comprobar qué partidos se repiten. Los objetivos 70–75% son aspiraciones, no resultados alcanzados ni mejoras previstas con fiabilidad.

No adoptar los perfiles estáticos de equipos, los pesos manuales del pseudocódigo o sus caps como reparación inmediata. Tampoco considerar el aumento de importancia SHAP un criterio suficiente de éxito: la importancia puede crecer sin mejorar la predicción fuera de muestra. La fórmula llamada `quarter_variance_weighted` debe aclararse antes de usarla, pues ponderar marcadores no equivale a medir su varianza.

## Ideas adaptables, por orden de dependencia

Este orden es una propuesta para la fase posterior a la auditoría, no trabajo aprobado ni promesa de mejora predictiva.

1. **Validación reproducible y contrato de datos.** Separar partidos y tiempo en entrenamiento/evaluación; identificar filas realmente OOF; alinear columnas, faltantes, reloj NBA/WNBA y versiones de base/Pacer/metamodelo. Medir error fuera de muestra y por etapa; mantener una referencia y posibilidad de volver al artefacto anterior.
2. **Seguimiento completo de resultados.** Conservar ID del encuentro, instante y etapa, versión del modelo, base, salida del meta, salvaguarda y valor finalmente mostrado/estabilizado. Registrar línea real, casa, precio y timestamp cuando existan. Evaluar por partido y tratar explícitamente las múltiples actualizaciones correlacionadas. Revisar antes el tracker y A04-07/08.
3. **Cobertura y coherencia de mercado.** Resolver identidad de partidos/cotizaciones y ausencia de líneas; después valorar históricos de mercado. Sólo datos disponibles en el momento de la predicción pueden entrar en sus entradas. Una línea de cierre posterior puede servir para análisis, pero no para simular una entrada que todavía no se conocía.
4. **Evaluar las señales que ya existen.** Primero comprobar enfriamiento, eficiencia, volatilidad y proxies en histórico/live/bundle. Comparar grupos pequeños de variables contra la referencia; no añadir duplicados ni sesgar salidas hacia UNDER para satisfacer una hipótesis antigua.
5. **Señales temporales observadas y calidad de predicción.** Con series suficientes, probar cambios de ritmo o rachas normalizados por minutos/posesiones efectivamente observados. Evaluar calibración y criterios de no recomendar ante datos insuficientes. Los porcentajes fijos propuestos en los planes no son umbrales validados.
6. **Contexto o ensembles adicionales.** Considerarlos sólo si los resultados anteriores lo justifican, con datos suficientes y una prueba aislada por cambio. No implica restaurar RF/Optuna, sustituir Huber por CatBoost ni activar aprendizaje continuo en producción.

## Límites y decisiones que se preservan

- No ampliar recomendaciones al Q4 tardío ni añadir prórrogas. Se puede utilizar Q4 de partidos históricos terminados sin que eso autorice recomendaciones tardías; nunca incorporar periodos futuros del encuentro que se está prediciendo.
- No reintroducir el modo manual de predicción por aparecer en planes de compatibilidad.
- No interpretar «bookmakers usan modelos pregame», «se actualizan con retraso», aumentos porcentuales de accuracy o ROI mensual consistente como hechos acreditados. Ninguna ventaja económica queda demostrada por estos documentos.
- Kelly, gestión de capital y automatización de apuestas no forman parte de esta revisión ni son requisitos aceptados. Tampoco se programan tareas por encontrar frecuencias sugeridas en un documento histórico.
- Al completar el conocimiento del proyecto, presentar cambios clave y sus efectos para decidirlos con el usuario. Hasta entonces se conservan código, modelos y documentos originales.

## Continuación de la auditoría

Sigue pendiente la revisión completa de `analysis/metamodel_integration.py`. Los contrastes puntuales de este informe no la sustituyen, ni dan por auditados el tracker, entrenadores o proveedor de cuotas.
