# Índice documental vigente

Actualizado el18 de octubre de2026 para continuar con Hermes Agent. Sustituye al índice anterior, que mezclaba arquitecturas e incluía enlaces inexistentes. Las fuentes históricas conservan sus rutas y contenido de fondo.

## Lectura inicial mínima

1. [README del proyecto](../README.md): inicio con Hermes, uso y entorno.
2. [AGENTS.md](../AGENTS.md): acuerdos para preservar funcionalidades.
3. [Estado de la auditoría](auditoria/00_ESTADO.md): prioridades, módulos revisados y siguiente paso.
4. Sólo el informe relacionado con la tarea. No cargar todo el archivo documental cada sesión.

El estado central mantiene decisiones y avances. Los informes fechados describen lo observado entonces, incluso si después se reparó el código. Las guías históricas ayudan a entender motivos; una afirmación de implementación requiere comprobar la ruta activa.

## Auditoría por módulos

| Módulo | Informe |
| --- | --- |
| CLI | [01_MAIN](auditoria/01_MAIN.md) |
| Configuración | [02_CONFIG](auditoria/02_CONFIG.md) |
| Orquestación live | [03_LIVE_SYSTEM](auditoria/03_LIVE_SYSTEM.md) |
| Cliente y monitor | [04_LIVE_API](auditoria/04_LIVE_API.md) |
| Predictor compartido | [05_LIVE_MODE](auditoria/05_LIVE_MODE.md) |
| Características históricas | [06_FEATURES](auditoria/06_FEATURES.md) |
| Procesamiento | [07_DATA_PROCESSING](auditoria/07_DATA_PROCESSING.md) |
| Entrenamiento | [08_TRAINING](auditoria/08_TRAINING.md) |
| Características live | [09_LIVE_FEATURES](auditoria/09_LIVE_FEATURES.md) |
| Integración del metamodelo | [10_METAMODEL_INTEGRATION](auditoria/10_METAMODEL_INTEGRATION.md) |
| Entrenador Huber | [11_TRAIN_META_LEARNER_HUBER](auditoria/11_TRAIN_META_LEARNER_HUBER.md) |

Son revisiones locales, no certificaciones del sistema completo. Tracker, puente, sesión y otras rutas conservan pendientes en el estado central. Los scripts de pruebas se excluyeron del recorrido por defecto por decisión del usuario.

## Operación y evidencias recientes

| Tema | Documento y alcance |
| --- | --- |
| Extensión Chrome/Edge | [Instalación y conexión](../browser_extension/README.md): transporte actual |
| Inferencia guardada | [Modelos reales](auditoria/PRUEBA_MODELOS_GUARDADOS.md): entorno y escenarios sintéticos anteriores al arreglo del reloj |
| Cambio funcional autorizado | [Corrección del reloj](auditoria/CORRECCION_RELOJ_2026-10-17.md): NBA/WNBA y pruebas; frescura pendiente |
| Capturas y monitor Chicago | [Prueba live](auditoria/PRUEBA_LIVE_2026-10-17.md): tiempo, marcador, estabilidad y final |
| Reconstrucción retrospectiva | [Descanso Chicago–Minnesota](auditoria/SIMULACION_DESCANSO_16869058.md):250.61 frente a249 |
| Predicción antes del resultado | [Golden State–Portland](auditoria/PRUEBA_GOLDEN_STATE_PORTLAND_2026-10-17.md):247.25 frente a246 comunicado por el usuario |

Pretemporada. No mezclar reconstrucciones con pruebas prospectivas ni deducir rendimiento general de dos casos.

## Archivo histórico: conservar y contrastar

| Ubicación | Inventario y criterio de lectura |
| --- | --- |
| Guías antiguas de Documentation | [Contraste de24 documentos](auditoria/CONTEXTO_DOCUMENTATION.md). Diseños distintos, implementaciones parciales, ejemplos y métricas no verificadas |
| unsure | [Contraste de15 documentos](auditoria/CONTEXTO_UNSURE.md). Visión y fórmulas; AI_CONTEXT/MODULE_MAP no son la arquitectura vigente |
| Roadmap | [Contraste de12 planes](auditoria/CONTEXTO_ROADMAP.md). Ideas posteriores a la auditoría, no cambios autorizados |
| Revisión inicial | [Estado del15/10/2026](../REVISION_ESTADO_2026-10-15.md). Bitácora anterior al transporte por extensión; sus pendientes pueden estar superados |
| Pruebas de cuotas | [Resumen histórico](../tests/ODDS_API_INTEGRATION_SUMMARY.md). No acredita funcionamiento del cliente/tests actuales |

Los avisos de vigencia enlazan estos contrastes desde cada fuente antigua. Los originales se conservan porque explican decisiones; un plan antiguo no autoriza modificar almacenamiento, restaurar modelos o ejecutar despliegues.

## Mantener el relevo

Desde el25/10/2026, por petición del mantenedor, `unsure/` y `tests/` están excluidas de GitHub y se conservan sólo en local. Los enlaces históricos a archivos de esas carpetas requieren esa copia local; los informes de contraste y auditoría permanecen publicados.

Actualizar el estado central con evidencia, límites y siguiente paso. Añadir el informe específico, con fecha, sin duplicar todo el contexto. Evitar copiar credenciales a Markdown.

Preparación del18/10/2026: AGENTS.md y README nuevos, índice sustituido, avisos de vigencia y credenciales documentales redactadas si se encuentran. No se reorganizan módulos. Evidencia de preservación: `.diagnostics/md_handoff_verification.json`. No equivale a una instalación o prueba de Hermes.
