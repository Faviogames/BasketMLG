# Auditoría por módulos

## Acuerdo de trabajo

Preferencia del usuario,17 de octubre de2026: evitar proliferación de módulos en nuevas implementaciones. Añadir por defecto la lógica al módulo existente que la necesita. Crear otro archivo sólo si el módulo receptor ya es muy grande o hay una necesidad clara de separación; justificar esa decisión por su utilidad concreta. No iniciar una reorganización ni fusionar archivos existentes automáticamente por esta preferencia.

Petición del usuario del 16 de octubre de 2026: comprender el proyecto por módulos, detectar errores y código obsoleto, parar tras cada módulo y posponer las mejoras hasta completar la revisión. Consultar los cambios clave para evitar eliminar funcionalidad necesaria.

- Esta fase permite inspección y documentación. No autoriza una limpieza ni modificaciones funcionales.
- Cada turno de revisión cubre un módulo concreto y termina al documentarlo.
- Registrar responsabilidad, entradas/salidas, dependencias, hallazgos, evidencia, límites y decisiones pendientes.
- Distinguir defectos reproducidos, riesgos por inspección, decisiones de producto y supuestos por verificar.
- Una etiqueta «legacy», un nombre «backup» o documentación antigua no prueban que algo sea eliminable. Antes de proponer borrados, comprobar referencias estáticas, usos por CLI/scripts, artefactos y uso real del usuario; las referencias dinámicas pueden requerir confirmación.
- Consultar antes de eliminar o sustituir módulos/flujos, cambiar contratos de datos, rutas de almacenamiento, modelos, fórmulas, entrenamiento o reglas de predicción. Las propuestas se prepararán para revisión después de completar el conocimiento del proyecto; no aplicar reparaciones durante la auditoría.
- No tocar el transporte live que ya funciona ni retomar la prueba stealth salvo instrucción del usuario.
- Aclaración posterior del usuario: algunas fórmulas/exclusiones pudieron introducirse deliberadamente; buscará documentación o chats del equipo. Conservar su motivo como pendiente cuando no esté acreditado. Un comportamiento reproducido no autoriza cambiarlo ni prueba que carezca de una razón histórica.

## Estado

### Presentación pública del repositorio — 25/10/2026

Por petición del usuario se rehace el README para presentar BasketMLG, su uso real por terminal y extensión, arquitectura activa, entorno de referencia y límites de interpretación. Las instrucciones específicas de agentes quedan enlazadas mediante AGENTS.md. Se elimina la ruta personal de instalación de la guía de extensión.

Se retiran `tests/` (dos archivos versionados) y `unsure/` (quince documentos versionados) del seguimiento de Git y se añaden a `.gitignore`, conservando todos sus archivos en local. Los informes históricos permanecen; índice y guía de extensión aclaran qué referencias necesitan la copia local. Esta retirada no purga versiones anteriores del historial Git.

Alcance: presentación y publicación, sin modificar código productivo, dependencias declaradas, modelos, datasets, credenciales ni registros. Verificación: diferencia de Git sin errores de whitespace, enlaces relativos del README existentes, carpetas excluidas del índice y preservación SHA256 de los17 archivos retirados; evidencia local en `.diagnostics/github_presentation_preserved_files.json`. La descripción pública de GitHub también se actualiza. No se pudo ejecutar la ayuda de CLI porque `python` no está en PATH y el lanzador `py` no detecta instalaciones; no se ejecutaron entrenamientos ni consultas live/cuotas. El entorno propuesto se basa en la combinación de diagnóstico previamente documentada, no en una instalación limpia nueva. Siguen pendientes frescura del reloj, validación Q3, probabilidades/trazabilidad y auditoría del tracker.

### Limpieza autorizada y publicación en GitHub — 25/10/2026

El usuario autorizó eliminar archivos que ningún módulo importa (comprobado con el grafo de graphify y un mapa AST de imports): `analysis/live_mode backup.py`, `analysis/legacy_heuristics.py`, `analysis/feature_importance.py`, `new_feature_importance.txt` (idéntico a `full_importance_list.txt`, que se conserva), `old_feature_importance.txt`, `model_features.txt` y `residual_distribution.png` (lo regenera `scripts/analyze_residuals.py`). También se borraron las cachés `__pycache__/` y `.pytest_cache/`. Los informes de auditoría anteriores todavía los mencionan como existentes. No se tocaron modelos, CSV, `prediction_records/` ni los JSON de ligas.

`config.py` lee ahora `ODDS_API_KEY` de la variable de entorno del mismo nombre, guardada como variable de usuario de Windows. Si falta, vale `""`. Sólo hace falta abrir una terminal nueva para que se cargue. El repositorio es público. `.gitignore` excluye los JSON de ligas (raíz y `leagues/`), `.browser_profile/`, `.diagnostics/`, `graphify-out/`, `.env` y las cachés. Por decisión del usuario, la rama `main` de `Faviogames/BasketMLG` se reemplazó por este proyecto. Antes había un bot de playlist M3U y JSON de fútbol; hay una copia en `BasketMLG_remote_backup_2026-10-25.bundle`, junto a la carpeta del proyecto.

### Navegación durante el seguimiento — 22/10/2026

**Corrección posterior por fallo real comunicado por el usuario:** el arranque del monitor fallaba en Windows porque `with create_input()` requería un `__exit__` que `Win32Input` no ofrece. El puente se cerraba por el `finally`, antes de iniciar la predicción. Se sustituye por `contextlib.closing(create_input())`, que usa `close()`, manteniendo `raw_mode()` como gestor del modo de terminal. La prueba previa usaba `MagicMock` y permitía indebidamente el protocolo de contexto; ahora el doble tiene sólo lectura de teclas, `raw_mode()` y `close()`, sin `__enter__/__exit__`. Las21 pruebas aisladas de CLI pasan, incluyendo cierre/restauración al volver, salir, terminar normalmente y fallar el monitor. No hubo consultas live, OddsAPI, entrenamiento ni cambios de modelos/registros. Falta repetir el arranque y los atajos en la terminal Windows del usuario con la extensión conectada; estas pruebas no acreditan lectura física de teclas.

Petición posterior del usuario: volver al listado desde la espera de actualizaciones y seleccionar otro partido sin cambiar el código de la extensión. Implementado en los módulos existentes: `analysis/live_system.py` mantiene un único procesador/puente y permite **V** (volver, sin Enter) y **Q** (salir) en terminal interactiva. Volver cancela y espera la finalización del monitor anterior, consulta otra vez el listado de la misma liga y solicita/consulta la línea del nuevo partido. El cierre final sigue ejecutándose una sola vez. La terminación natural del monitor conserva la salida anterior; lista vacía/error/cancelación de selección terminan el comando.

`core/live_api.py` reinicia al comenzar cada monitor las referencias de predicción, estadísticas, parciales y ficha anterior, contador y tiempo de sesión; reconstruye la caché prepartido como antes. Sin cambios en fórmulas, reloj, disparadores Q3/Q4, modelos, rutas o formato de registros. `browser_extension/panel.js` trata HTTP409 de `/result` como respuesta caducada y continúa consultando: evita que cancelar una petición pendiente desconecte el panel. No cambia token, autenticación ni transporte; hace falta recargar la extensión/panel una vez para cargar el JavaScript actualizado.

Evidencia:20 pruebas aisladas pasan en `tests/test_cli_live.py` (incluidas cuatro nuevas de cambio de partido, teclado/cancelación/restauración del terminal, reinicio de estado y cancelación del puente con reutilización de clave). `tests/test_extension_navigation.js` verifica sin red que una respuesta caducada permite entregar la siguiente con la misma credencial y que HTTP403 sigue rechazándose. Dobles de teclado y AST del monitor evitan dependencias ML, predicciones, OddsAPI y escrituras en registros. No se cargaron modelos ni se entrenó. Los atajos usan `prompt_toolkit`, dependencia ya declarada; el Python auxiliar usado para las pruebas no lo incluye, por lo que la lectura real de teclas no queda acreditada por estas pruebas.

Límites y siguiente paso: comprobar prospectivamente **V** durante la espera en una terminal Windows real y con extensión conectada, seleccionar otro partido y confirmar misma clave y primera actualización independiente. Mantener abiertos panel y pestaña. Una consulta antigua puede tardar en liberar el panel hasta que termine su navegación; los cálculos síncronos bloquean temporalmente la atención del teclado. Esta tarea no cierra los pendientes predictivos ni la auditoría del tracker.

### Cambio autorizado de CLI — 20/10/2026

El usuario autorizó retirar el menú de modos, la predicción manual y la entrada de partidos por ID, y trasladar la vista de modelos a `--models`. Esta petición posterior sustituye las decisiones anteriores de conservar temporalmente esas interfaces; no autoriza retirar cálculos compartidos ni cambiar el transporte.

- `python main.py --live`: carga modelos → selecciona liga → muestra código para extensión → busca/lista partidos → selección → OddsAPI/línea opcional → seguimiento. Sin menú adicional. Lista vacía, acceso fallido o cancelación cierran el cliente; para reintentar se ejecuta de nuevo.
- `python main.py --models`: muestra modelos y termina; no importa/inicia el procesador de navegador. `--train` se conserva. Sin argumentos se muestra ayuda; los tres flags son excluyentes para evitar ejecutar un modo distinto del solicitado.
- Eliminados: ramas manuales de `main.py` y `analysis/live_system.py`, entrada por ID y su fallback, función interactiva `live_mode_with_alerts` e imports exclusivos de esa interfaz. Se conserva `analysis/live_mode.py` con predictor, caché, metamodelo y salvaguardas. Se elimina el antiguo stub de instrucciones de integración del menú, sin llamadas encontradas.
- El seguimiento cierra su procesador en un único `finally` y deja de mantener una referencia para un segundo cierre. Se mantiene la consulta de línea automática y su entrada opcional; no se altera la lógica de predicción ni los pendientes de reloj/Q3/Q4.
- Verificación:16 pruebas aisladas de CLI y ciclo de vida en `tests/test_cli_live.py`, sin red ni entrenamiento. `--help` ejecutado sin dependencias ML. `--models` probado con NBA/WNBA reales en el entorno compatible, sin navegador ni APIs. Persiste el aviso de serialización XGBoost ya conocido; no se convirtieron modelos. La verificación de preservación y comparación AST se registra en `.diagnostics/cli_simplification_verification.json`.

Los informes01/03/05 siguientes conservan la auditoría de la interfaz anterior; sus avisos indican qué partes quedan sustituidas. Auditoría restante y revisión del tracker siguen pendientes. No se ha repetido una sesión live real mediante extensión para este cambio de interfaz.

### Relevo preparado para Hermes Agent — 18/10/2026

Por petición del usuario, se preparó el contexto de Hermes Agent de Nous Research mediante [AGENTS.md](../../AGENTS.md), [README](../../README.md) e [índice vigente](../DOCUMENTATION_INDEX.md). La autorización de esta tarea permite reorganizar/editar Markdown; no modificar ni reorganizar módulos. Los acuerdos funcionales anteriores siguen vigentes.

Lectura inicial: AGENTS → README → este estado → informe específico. Próximo módulo operativo sugerido: `metamodel/prediction_tracker.py`; la prioridad de seguimiento reloj/Q3/Q4 se conserva debajo. No hay que reiniciar toda la lectura ni tratar los planes antiguos como reparaciones aprobadas.

Se sustituyó el índice anterior y se añadieron avisos a52 fuentes históricas/de alcance parcial, preservando su contenido y rutas para no perder motivos de diseño ni enlaces desde auditorías. Se redactó una credencial literal en una guía de cuotas; no se cambió `config.py`. No se borraron documentos con contenido único ni se ejecutaron entrenamientos, predicciones, llamadas de cuotas o cambios de entorno para esta preparación.

Hermes no se encontró en el PATH de la sesión de preparación; no se instaló ni se probó su ejecución. La compatibilidad del archivo de contexto se contrastó con su documentación oficial, enlazada en README. El entorno Python de BasketMLG y los paquetes de diagnóstico no se dan por portables a WSL ni al runtime interno de Hermes. La comprobación de enlaces y hashes queda en `.diagnostics/md_handoff_verification.json`.

### Seguimiento prioritario: reloj y reacción durante Q3/Q4

Petición explícita del usuario,17 de octubre de2026: seguir de cerca este problema antes de dar por fiable la actualización durante el partido. **Pendiente; no cerrado por la corrección inicial del reloj ni por el buen resultado de la simulación al descanso.**

- **Reloj activo por validar:** en Chicago–Minnesota16869058 el marcador avanzó de122–105 a126–107, pero una consulta independiente del evento mantuvo `time.played=758`; Python y la cabecera visible indicaban Q4/11:22 mientras el historial de jugadas ya llegaba a2:02. La pausa explicaba ausencia momentánea de avance, pero no resuelve esta discrepancia. Determinar fuente, frescura y semántica temporal antes de extrapolar timestamps o sustituir el reloj. La sesión sí reconoció correctamente el final y cerró el puente.
- **Actualización de Q4:** `_should_update_prediction` en `core/live_api.py` suma sólo Q1–Q3; los puntos nuevos de Q4 no activan por sí solos una nueva predicción. Confirmado por inspección. La sesión mantuvo283.52 y mostró cambio0; no se guardó el payload bruto de cada ciclo, por lo que no atribuir todos los ciclos a una misma causa sin capturar sus entradas.
- **Q3 pendiente de prueba específica:** aún no se ha validado con varias capturas coherentes durante Q3 después de reparar el reloj. No concluir que el modelo falla intrínsecamente en Q3 ni que ya funciona correctamente. Capturar en cada actualización periodo/reloj bruto y normalizado, parciales, estadísticas, motivo de estabilidad y salidas base/meta/final; contrastar con la ficha y el resultado final.
- **Referencia al descanso:** reconstrucción con sólo Q1/Q2,67–60 (127pts), produjo250.6082 frente al final133–116 (249pts): error+1.6082. El usuario aporta línea de descanso242.5, cuota Over2.04 y Under1.77: la salida queda8.1082 por encima de la línea y el resultado final6.5 por encima. Es un acierto retrospectivo de un partido, sin validación prospectiva, de calibración o rentabilidad; persistieron128/188 entradas imputadas. No usarlo para cerrar los problemas del seguimiento live.
- **Alcance conservado:** descanso como momento principal y seguimiento Q3; Q4 no es el uso habitual deseado, con cualquier uso excepcional limitado al inicio/antes de mitad según el acuerdo previo. No ampliar recomendaciones a Q4 tardío ni prórrogas. Acordar cambios clave antes de modificar filtros, fuentes temporales o fórmulas.

Evidencias: [prueba live](PRUEBA_LIVE_2026-10-17.md), [simulación al descanso](SIMULACION_DESCANSO_16869058.md) y `.diagnostics/halftime_16869058.json`. No se modificó código funcional al registrar este seguimiento.

**Prueba prospectiva al descanso,17 de octubre:** Golden State–Portland67–58, reloj24/48min. Predicción247.2482 registrada antes del resultado, línea238.5 facilitada por el usuario (Over1.87/Under1.83), diferencia+8.7482. Imputación124/188. El usuario comunicó después128–118:total246, error absoluto1.2482 y OVER acertado por7.5 puntos. Partido de pretemporada; resultado final aportado por el usuario, sin verificación independiente. La referencia simple de duplicar el descanso daba250, error4. Conservar esta prueba individual separada de actualizaciones posteriores y de la reconstrucción retrospectiva de Chicago; no inferir precisión general o rentabilidad sostenida. Ver [PRUEBA_GOLDEN_STATE_PORTLAND_2026-10-17.md](PRUEBA_GOLDEN_STATE_PORTLAND_2026-10-17.md). Persistencia sólo de diagnóstico; probabilidades/confianza no validadas como calibración o ventaja de mercado.

**Excepción autorizada el17 de octubre de2026:** el usuario permitió aplicar la corrección del reloj detectada en pruebas reales. Se normalizó periodo/tiempo por liga, se transmitió duración NBA/WNBA a Pacer y salvaguarda, y se bloqueó la predicción con reloj inválido. Ver [CORRECCION_RELOJ_2026-10-17.md](CORRECCION_RELOJ_2026-10-17.md). La autorización no se extiende a limpieza general ni reentrenamiento. Los informes anteriores describen el código anterior a esta reparación; sus hashes y la frase «sin cambios funcionales» corresponden al momento de cada auditoría.

| Módulo | Estado | Informe |
| --- | --- | --- |
| `main.py` | Auditado; CLI simplificada y --models autorizado el20/10 | [01_MAIN.md](01_MAIN.md) |
| `config.py` | Revisión local terminada; sin cambios funcionales | [02_CONFIG.md](02_CONFIG.md) |
| `analysis/live_system.py` | Auditado; búsqueda directa, retirada de menú/ID y cierre único el20/10 | [03_LIVE_SYSTEM.md](03_LIVE_SYSTEM.md) |
| `core/live_api.py` | Auditado; corrección posterior autorizada del reloj/selección de liga el17 de octubre | [04_LIVE_API.md](04_LIVE_API.md), [corrección](CORRECCION_RELOJ_2026-10-17.md) |
| `analysis/live_mode.py` | Auditado; corrección posterior autorizada de duración transmitida y salvaguarda | [05_LIVE_MODE.md](05_LIVE_MODE.md), [corrección](CORRECCION_RELOJ_2026-10-17.md) |
| `core/features.py` | Revisión local terminada; generación histórica y helpers compartidos, sin cambios funcionales | [06_FEATURES.md](06_FEATURES.md) |
| `core/data_processing.py` | Revisión local terminada; cobertura raw, normalización e imputación, sin cambios funcionales | [07_DATA_PROCESSING.md](07_DATA_PROCESSING.md) |
| `core/training.py` | Revisión local terminada; selección, validación, OOF y guardado, sin entrenamiento real ni cambios funcionales | [08_TRAINING.md](08_TRAINING.md) |
| `core/browser_bridge.py` | Pendiente; intervenido antes, aún sin auditoría completa | Pendiente |
| `core/sofascore_session.py` | Pendiente | Pendiente |
| `analysis/metamodel_integration.py` | Revisión local terminada; carga, entradas Huber, metadatos y fallbacks; sin cargar modelos ni cambios funcionales | [10_METAMODEL_INTEGRATION.md](10_METAMODEL_INTEGRATION.md) |
| `metamodel/train_meta_learner_huber.py` | Revisión local terminada; CSV, partición, contrato y guardado, sin entrenar ni cargar modelos | [11_TRAIN_META_LEARNER_HUBER.md](11_TRAIN_META_LEARNER_HUBER.md) |
| `scripts/generate_base_predictions.py` | Fuera de la revisión por indicación posterior del usuario: scripts de pruebas | No continuar por defecto |
| `metamodel/prediction_tracker.py` | Siguiente módulo operativo sugerido; registro y persistencia real de predicciones | Pendiente |
| `core/live_features.py` | Auditado; corrección posterior autorizada de duración de Pacer | [09_LIVE_FEATURES.md](09_LIVE_FEATURES.md), [corrección](CORRECCION_RELOJ_2026-10-17.md) |
| `core/game_clock.py` | Nuevo normalizador del reloj;27 pruebas específicas, modelos reales y9 regresiones Q3 verificadas | [corrección](CORRECCION_RELOJ_2026-10-17.md) |
| Resto del proyecto | Pendiente de inventario progresivo | Pendiente |

El orden posterior se ajustará siguiendo las dependencias. No hay un porcentaje global de cobertura válido todavía: el inventario completo y las relaciones entre módulos aún no están verificados. Consultar únicamente los informes necesarios para retomar, sin releer toda la conversación.

## Contexto documental revisado

Se leyeron los 15 documentos de `unsure/` a petición del usuario. Inventario, clasificación y contraste: [CONTEXTO_UNSURE.md](CONTEXTO_UNSURE.md). Mezclan visión Anchor/Navigator, un meta-modelo anterior de reglas/Reality Check, LightGBM, planes y registros de cambios. No tratarlos como una sola especificación vigente ni ejecutar sus instrucciones de implementación/borrado.

Corroborado: arreglo Q3 con nueve pruebas y fragmentos de mapeo/calidad API. Pendiente: aprendizaje automático continuo, migración completa de alertas, Run Detector con contexto/decay y relación de entrenadores con el Huber activo. Los documentos aportan el motivo de evitar sobrecorrecciones, pero no explican todas las decisiones observadas en la auditoría. No se ha autorizado ningún cambio funcional.

También se leyeron completos los **24 documentos de la raíz de `Documentation/`** y se contrastaron con estos informes. Inventario y conclusiones: [CONTEXTO_DOCUMENTATION.md](CONTEXTO_DOCUMENTATION.md). Aportan una etapa CatBoost adicional, el diseño deliberado EMA/directas y el almacenamiento híbrido parcialmente presente. El análisis Knicks–Nets compara erróneamente predicción final con marcador parcial: se retira su supuesto error +130.5 como evidencia de fallo crítico. No se considera la caché histórica en Q3 un error por sí misma. Se actualizaron los informes afectados y se conservaron los documentos originales con avisos de vigencia donde corresponde.

Se leyeron completos los **12 documentos de `Roadmap/`**. Clasificación de implementación parcial, propuestas y prioridades candidatas: [CONTEXTO_ROADMAP.md](CONTEXTO_ROADMAP.md). Se corroboran cálculos de enfriamiento/volatilidad y la intención de los proxies de fricción. La integración Huber ya recibe la predicción base, por lo que el problema central del plan CatBoost no se presume vigente. Los análisis de noviembre contienen medias, IDs y ajustes inconsistentes; no acreditan un sesgo actual ni justifican cambios urgentes. No se ejecutaron planes ni se modificó código funcional.

## Reconciliación de hallazgos tras leer la documentación histórica

Esta sección actualiza la interpretación de los informes anteriores. Una diferencia de arquitectura entre documentos y código no se considera por sí sola un defecto. La lista de cuestiones pendientes tampoco es una lista de errores graves ni una orden de implementación.

| Tema | Estado actualizado | Consecuencia para la auditoría |
| --- | --- | --- |
| RF, Optuna y nombre «sistema dual» | **Contexto histórico aclarado.** El usuario confirma su uso anterior; el código documenta su retirada | Retirado como sospecha de componente perdido que haya que restaurar. Quedan sólo las etiquetas incorrectas del fallback demostradas en T08-07 y posibles textos antiguos |
| CatBoost, LightGBM, reglas/Reality Check y Huber descritos como «el metamodelo» | **Versiones/diseños distintos**, no prueba de incoherencia funcional actual | La diferencia documental queda como contexto, no como avería. Revisar qué consume la integración actual y conservar artefactos hasta conocer sus referencias |
| Antiguo bug del tiempo fijo de Q3 en el detector adaptativo | **Corrección existente corroborada y cerrada**: función actual y nueve pruebas coinciden con el documento | No queda como reparación pendiente. LF09-03 se refiere a otro punto: tipo del respaldo histórico; LF09-01 al ritmo principal, no al detector corregido |
| Comparación de nombres API frente a nombres del modelo | **Corrección histórica encontrada**, con alcance parcial | No repetir ese error antiguo como explicación demostrada de las 124/188 imputaciones actuales. A04-10 sobre valores parseados y V05-01 sobre ensamblaje siguen siendo cuestiones distintas |
| Clamps y correcciones conservadoras de Reality Check | **Intención aclarada:** evitar sobrecorrecciones y distinguir anotaciones altas legítimas | Se retira cualquier lectura de «reglas arbitrarias porque no entendemos su motivo». Umbrales concretos, conexión actual y unidades siguen pendientes; no restaurar ni cambiar fórmulas por la documentación |
| Ausencia de recomendaciones en Q4 tardío y de soporte OT | **Alcance de producto decidido por el usuario**, no funcionalidad faltante | No queda pendiente añadirlo. Sólo revisar coherencia dentro del inicio permitido de Q4 y reconocimiento de estados fuera de alcance |
| Helpers enhanced, SVI/RWT, ORTG placeholder y otras rutas auxiliares | **Alcance limitado**: pruebas, scripts o recorridos anteriores según cada informe | No clasificarlos como fallo grave de la predicción live actual sin demostrar consumo. Conservar el hallazgo técnico para revisar su ruta específica |
| Aprendizaje continuo, migración de alertas y Run Detector con contexto | **Estado operativo aún por verificar**; los documentos mezclan planes y afirmaciones históricas | No marcar como implementado ni como funcionalidad rota o perdida sin comprobar conexiones. Registro/exportación de predicciones no acredita actualización de modelos |
| Separación EMA/directas | **Diseño deliberado documentado** para evitar aplicar EMA a métricas ya agregadas | No es una carencia que todas las variables no tengan EMA. F06-03/F06-04 sobre repetición de agregados y ensamblaje son observaciones distintas |
| Caché histórica durante Q3 | **Separación de responsabilidades documentada**, no defecto por sí misma | Retirada la recomendación antigua de desactivarla sólo por etapa. V05-03 sobre falta de reconstrucción y cobertura sigue pendiente |
| Supuesto error final +130.5 en análisis Knicks–Nets | **Conclusión documental inválida**, compara total final previsto con parcial139 | No se usa para severidad, urgencias, benchmarks ni para forzar caps conservadores. El incidente con resultado final232 requiere su propia evidencia |
| Almacenamiento híbrido | **Parte implementada corroborada**, no sólo propuesta | Preservar registros diarios/estacionales e índices. API de resultados, garantías y aprendizaje aún por auditar |
| Antiguo baseline FTI0.5 y parser sólo dict | **Cambios anteriores corroborados** en el código actual | No reabrirlos como bugs antiguos pendientes. Calibración/unidades y parsing de valores son cuestiones distintas |

**Se mantienen las observaciones reproducidas**, con su alcance y sin asignarles impacto predictivo no medido: pérdida de parciales D07-01, dependencia del orden D07-02, claves de mercado F06-01, división final por filas T08-01 y diferencias de reloj por liga, entre otras. Los documentos no las justifican específicamente ni invalidan las pruebas. Su motivo histórico puede seguir pendiente, y no se autoriza modificarlas durante esta fase.

**Riesgos condicionados**, no incidentes actuales acreditados: fuga por medias de liga sólo en el camino de listas pequeñas de D07-03; contaminación de métricas y divergencia del entrenamiento del Pacer pendientes de atribuir a los artefactos concretos. No presentar los MAE guardados como falsos ni el proyecto como inutilizable por estos riesgos.

## Decisiones de producto y cuestiones pendientes

### Decisión del usuario: retirar el modo de predicción manual

El16/10/2026 el usuario descartó el modo manual como dirección del producto y aplazó su retirada. **Actualización20/10/2026: retirada autorizada y aplicada**, junto con la entrada por ID. El flujo principal queda live mediante SofaScore o una futura API con cobertura adecuada.

Alcance aplicado: menú de modos, entrada manual predeterminada y función interactiva manual retirados. Sin argumentos se muestra ayuda; `--live` inicia el flujo directo y `--models` consulta modelos. Se conserva el módulo de cálculo compartido `analysis/live_mode.py` y la entrada opcional de línea de apuestas.

No extender esta decisión automáticamente al acceso a un partido por ID ni a introducir una línea de totales cuando no llega de OddsAPI: son funciones distintas y su destino se confirmará si se propone cambiarlas. Tampoco asumir que una API de cuotas sustituye los datos live de marcador, reloj y estadísticas; concretar la cobertura del futuro proveedor cuando se elija.

### Alcance temporal aclarado por el usuario

Tras la revisión de `core/live_api.py`, el usuario aclaró que el producto está orientado a Q1–Q3 y, en ocasiones, al inicio o antes de la mitad de Q4. La segunda mitad de Q4 queda fuera de las recomendaciones previstas y las prórrogas no forman parte del sistema. Se registra como criterio de producto; la auditoría no ha validado empíricamente la rentabilidad de esos tramos.

No proponer soporte de prórrogas ni recomendaciones tardías como una mejora implícita. Separar esa exclusión intencionada de los posibles fallos al reconocer que un partido ya está fuera de alcance. El comportamiento dentro del tramo inicial permitido de Q4 sigue por contrastar con el usuario; ignorar todos los puntos de Q4 no implementa por sí solo un corte a mitad del periodo.

Antes de cambiar reglas, concretar por liga qué significa «en ocasiones», el punto exacto de corte y qué debe hacer el sistema al superarlo: detener seguimiento, seguir mostrando datos sin recomendar o mantener una última proyección identificada como anterior. No se ha aprobado ni implementado ninguna de estas alternativas.

### Otras decisiones pendientes

- M01-D01: prioridad actual de `--live` cuando se pasan ambos flags; decidir si conservarla o rechazar la combinación.
- M01-R02 / C02-R03: política común de rutas, antes de cambiar ubicación de datos, modelos o exclusiones.
- C02-01: conectar configuraciones sin efecto o retirarlas; no modificar reglas de predicción durante la auditoría.
- C02-R01: poda global frente a poda por liga/modelo, preservando las exclusiones actuales hasta decidir su migración.
- C02-04: migración de la credencial de cuotas fuera del código sin perder acceso; pendiente de acuerdo, no aplicada.
- L03-02: definir el significado de la confianza mostrada; no convertir automáticamente compatibilidad en probabilidad predictiva.
- L03-06: sin entrada activa por ID desde20/10/2026; no queda pendiente conservarla o ampliarla. La búsqueda conserva la compatibilidad del listado.
- L03-07: desde20/10/2026 ya no hay menú ni flujos duplicados de ID. Fallo/lista vacía/cancelación termina el comando y cierra el cliente; para reintentar se vuelve a ejecutar.
- A04-01/A04-02/A04-03: respetar el alcance temporal aclarado: sin prórrogas, sin recomendaciones en Q4 tardío. Revisar el tramo inicial permitido de Q4, identificación de estados fuera de alcance y duración por liga antes de cambiar relojes, triggers o fórmulas.
- A04-05: política para equipos desconocidos y normalización exacta de liga/equipos.
- A04-07/A04-08: definir qué etapa de predicción se muestra, se estabiliza y se guarda; preservar interpretación de registros históricos.
- A04-09: uso de la línea de mercado por la inferencia, además de su presentación y exportación.
- A04-10/A04-11: separar transporte y parsing antes de retirar el cliente anterior, que sigue siendo base de la extensión.
- V05-01/V05-02/V05-03: alinear construcción de características con entrenamiento y bundle antes de corregir imputación; no rellenar más columnas con ceros para ocultar el porcentaje.
- V05-06: duración por liga y conservación del reloj suministrado en métricas/metamodelo/salvaguardas; sin incorporar prórrogas.
- V05-08: política de datos insuficientes y significado de confianza; el mensaje de modo conservador y la salvaguarda son mecanismos distintos.
- V05-09: separar líneas sintéticas y probabilidades del modelo de las líneas reales y su interpretación de mercado.
- F06-01/F06-02/F06-03/F06-04: contrato común de variables históricas/live, claves de mercado, escala de ritmo y series para EMA. Revisar modelos guardados y necesidad de reentrenamiento antes de reparar; no aplicar arreglos aislados que cambien distribuciones.
- F06-05/F06-06/F06-08: contrato de helpers, nombres canónicos y representación de cobertura ausente. No sustituir faltantes por ceros para esconder imputación.
- F06: contrastar muestras duplicadas por partido con partición/OOF al auditar entrenamiento; precisar semántica de descanso, clutch y proxies antes de cambiar fórmulas.
- D07-01/D07-02: preservar marcadores de origen y estudiar síntesis de totales/históricos independiente del orden de procesamiento; comprobar consumidores antes de cambiar mutación de objetos raw.
- D07-03/D07-04: política temporal de medias de liga, umbral de 500 y coste. Mantener pendientes las razones históricas; no extrapolar el escenario de fuga en listas pequeñas al bundle NBA actual.
- D07-05/D07-06/D07-08: contrato canónico de estadísticas, identidad de equipos y cobertura de box por cuarto antes de modificar parsing.
- D07-07/D07-09: decidir tratamiento de fechas desconocidas, ceros observados/faltantes, registros incompletos y validación. Conservar buckets de liga y exclusiones hasta contrastar documentación del equipo.
- T08-01/T08-02/T08-03: acordar validación temporal/por encuentro y definición de OOF. OOF agrupa correctamente con IDs, pero rellena el primer bloque dentro de muestra; evaluación final vuelve a dividir filas al azar. No atribuir aún un efecto numérico al MAE guardado.
- T08-04/T08-05: contrato de faltantes e histórico live. Se guarda df por etapas con ambas filas por partido; el consumidor live no deduplica y sus ventanas pueden cubrir menos partidos. Preservar artefactos hasta diseñar migración.
- T08-06/T08-07/T08-08: distinguir muestras de partidos, etiquetar correctamente fallback XGBoost y revisar atomicidad/versionado y exclusiones globales antes de guardar. La retirada de RF/Optuna queda aclarada en la tabla anterior; no es un fallo pendiente.
- LF09-01/LF09-04/LF09-05/LF09-08: contrato común de reloj, etapa y duración por liga; balance coherente dentro del inicio de Q4 autorizado, sin añadir OT ni recomendaciones tardías.
- LF09-02/LF09-03/LF09-06: distinguir cuarto no jugado, datos de tendencias frente a DataFrame histórico y claves necesarias para interacciones. El antiguo bug temporal del detector adaptativo está cerrado; estas observaciones no lo reabren.
- LF09-10/LF09-11/LF09-12: separar proxies, auxiliares de scripts y cálculos activos antes de cambiar fórmulas; comparar las dos implementaciones de Pacer con el entrenamiento real del metamodelo.
- MI10-01/MI10-02/MI10-05: acordar calidad/cobertura mínima y estados explícitos para especialista degradado, modelo no disponible y salidas inválidas; distinguir fallback a base de predicción Huber en trazas y presentación. Los escenarios aislados no demuestran que estos fallos hayan ocurrido en el partido del usuario.
- MI10-03/MI10-04: contrastar identidad/esquema del artefacto, compatibilidad NBA/WNBA, rutas y renovación de caché antes de cambiar carga o modelo. El Huber actual recibe base y Pacer; los argumentos Pulse/Runner no son entradas directas de esta capa por diseño documentado.
- MI10-06/MI10-07: preservar el reloj en segundos de la llamada principal; el wrapper auxiliar que lo omite no tiene llamadas activas encontradas. Separar probabilidades del Pacer de confianza/probabilidades de la salida combinada; no convertir sus metadatos en una recomendación nueva.
- H11-01: diferencia de Pacer confirmada en el CSV objetivo del entrenador Huber: mezcla de extrapolación al descanso y mercado/220 frente a extrapolación pura en live. 1513/1556 filas difieren. Sigue pendiente acreditar procedencia exacta del joblib activo antes de atribuirle esa divergencia; no cambiar un lado del contrato aisladamente.
- H11-02/H11-03: evaluación Huber aleatoria y CSV sólo descanso/NBA, sin fechas/liga explícitas; los IDs actuales son únicos, así que no se demuestra duplicación entre train/test en este CSV. Revisar base OOF preparada antes del split externo y diferencia respecto del bundle live; diseñar evaluación de toda la cadena.
- H11-04/H11-05: acordar publicación/versionado del artefacto, criterios de aceptación y validación de entradas; el entrenador sobrescribe la ruta live aunque no mejore la referencia. No ejecutarlo durante la auditoría. Guardar el estimador entrenado sólo en train no es por sí solo un defecto.

Contexto confirmado por el usuario tras el módulo 08: RF y Optuna existieron en el desarrollo y se buscaba un metamodelo. Mantener pendiente la documentación histórica; no restaurarlos ni retirar el metamodelo actual por esa referencia.

## Funcionalidades existentes: protección durante la auditoría y destino previsto

- Entrenamiento desde `python main.py --train`.
- Modo manual y selección por ID: retirados con autorización20/10/2026. Conservar el predictor compartido y no restaurar esas interfaces por documentación anterior. Sin argumentos se muestra ayuda; vista de modelos mediante `--models`.
- Live desde `python main.py --live`, con extensión como transporte actual.
- Modelos, históricos y registros existentes: no borrar, sobrescribir ni reentrenar durante esta fase.

## Contexto ya comprobado antes de esta auditoría

**Prueba live del17 de octubre:** extensión conectada y listado/stats/modelos ejecutados para Chicago–Minnesota en Q3. Se incluyeron37 puntos de Q3 (total164), pero el extractor informóQ2/12.6min; Huber530.3 y salvaguarda604.55. Se detuvo tras primera actualización y se cerró el puente. Ver [PRUEBA_LIVE_2026-10-17.md](PRUEBA_LIVE_2026-10-17.md). Registro automático3FE7E4DC conservado sin modificar: identificarlo como prueba con reloj incoherente antes de evaluación/aprendizaje. Falta capturar el payload bruto para determinar la semántica temporal exacta; no concluir que Q3 se excluyó ni cambiar fórmulas a ciegas. Priorizar coherencia del reloj antes de reentrenar. Sin cambios funcionales.

**Segunda fase live, sólo lectura:** ya se capturaron seis payloads reales (dos rondas, tres eventos NBA Preseason). Todos indican Q4, mientras el extractor devuelve Q2/Q2/Q3 por interpretar `time.played` como acumulado. Ver ampliación del informe anterior y `.diagnostics/live_clock_probe.json`. Se confirma la incoherencia con datos brutos; no está establecida una fuente exacta de reloj alternativo. La selección de liga sí existe y su fallback distingue12/10min, pero sólo se usa cuando falta el tiempo recibido; el reloj reconstruido para meta y las proyecciones siguientes mantienen48/24. No se ha probado WNBA live. Las capturas no publicaron predicciones ni alteraron registros y el puente quedó cerrado.

**Tercera fase, reloj visible:** el enlace canónico compartido por el usuario permitió leer la ficha normal sin CAPTCHA: Q4(11:22). Desde su listado de Baloncesto se contrastaron también DallasQ4(0:25) y MemphisQ4(3:55). En los tres, el resto de `played` dentro de720s coincide con el reloj visible, mientras el cociente produce un cuarto incorrecto. Es una hipótesis concreta de reparación periodo+reloj, no un cambio aplicado ni prueba de todos los límites/transiciones o de frescura. Ver tabla y límites en el informe live. La lectura DOM es posible sin OCR; no se modificó la extensión.

**Ejecución real posterior:** a petición del usuario se cargaron los bundles NBA/WNBA y el Huber guardados, y se ejecutó inferencia mediante las funciones reales del predictor compartido. Tres escenarios sintéticos respondieron con dependencias compatibles; sin entrenar, cambiar modelos ni exportar registros. Ver [PRUEBA_MODELOS_GUARDADOS.md](PRUEBA_MODELOS_GUARDADOS.md). Se verifican clase, esquema y coeficientes del Huber, se reproduce 124/188 imputadas en NBA y se confirma el uso de 48 minutos por Pacer con reloj WNBA. Esto no evalúa exactitud ni prueba SofaScore/monitor. Los errores iniciales de SciPy/pandas pertenecían al entorno de diagnóstico y no se atribuyen a la instalación del usuario.

**Alcance corregido por el usuario:** ignorar scripts de test/pruebas y volver a módulos operativos. Los informes de scripts/CSV se conservan como contexto, sin asumir su uso en producción ni convertirlos en reparaciones pendientes del artefacto activo. No retomar la siguiente revisión de `scripts/generate_base_predictions.py` sugerida en el módulo 11.

El usuario confirmó listado, selección, estadísticas, predicción, exportación y segunda actualización del modo live mediante extensión. Persisten 124/188 entradas imputadas en una predicción; su causa exacta no se ha determinado. La prueba aislada posterior de Playwright funcionó con y sin stealth, por lo que no acredita una ventaja del plugin. Ver los informes de estado anteriores para los detalles, sin tratar sus hipótesis como conclusiones.

Avance de la auditoría 05: se verificaron causas estructurales de imputación en el ensamblaje histórico (familias no incorporadas, nombres diferentes y diferencias EMA incompletas), además de asimetría de parciales por equipo. No se ha cruzado todavía cada una de las 124 columnas de la ejecución real con su bundle. Preservar esta distinción al retomar.
