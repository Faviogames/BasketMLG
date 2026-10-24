# BasketMLG v2 — contexto del proyecto

## Al retomar

Habla en español. Lee `README.md`, `Documentation/auditoria/00_ESTADO.md` y después sólo el informe necesario para la tarea. Navegación: `Documentation/DOCUMENTATION_INDEX.md`. Evita releer todos los MD cada sesión.

El usuario quiere comprender el proyecto antes de mejorarlo: revisar un módulo, documentar y parar. Auditoría incompleta. Siguiente módulo operativo sugerido: `metamodel/prediction_tracker.py`. Los scripts históricos de pruebas quedaron fuera del recorrido por decisión del usuario.

## Acuerdos

- Consultar cambios clave: eliminar/sustituir funcionalidad, contratos de datos, modelos, fórmulas, entrenamiento o rutas de registros. Preparar la propuesta y su efecto antes de aplicarla. El arreglo de reloj autorizado el17/10/2026 no autoriza limpieza general.
- Preferir ampliar módulos existentes. Consultar antes de crear un módulo productivo salvo necesidad clara ya acordada. No reorganizar módulos al preparar agentes.
- Preservar modelos, JSON/CSV, `prediction_records/` y evidencias. No entrenar ni sobrescribir artefactos para comprobar su carga. Separar pruebas que escriben del historial normal.
- Cambio autorizado el20/10/2026: predicción manual y selección por ID retiradas de las entradas activas. `--live` va directamente a liga → conexión de extensión → listado → partido. `--models` muestra los modelos; sin argumentos se muestra ayuda. La línea de apuestas opcional introducida por el usuario se conserva; no confundirla con el modo manual eliminado.
- La reorganización del18/10/2026 autoriza Markdown, no cambios en código, dependencias, modelos o credenciales. Documentación antigua conserva motivos de diseño, no órdenes de implementar/borrar. RF/Optuna/CatBoost/LightGBM anteriores no deben restaurarse por aparecer allí.
- La petición posterior del20/10/2026 sí autoriza simplificar CLI y retirar esas interfaces, sin cambiar predictor, reloj, modelos, entrenamiento ni transporte. Los informes anteriores describen la interfaz antigua.

## Producto y arquitectura activa

Predicción del TOTAL FINAL usando históricos y datos observados del partido. Momento principal: descanso con Q1/Q2 completos; seguimiento Q3. Q4 sólo excepcional al inicio/antes de mitad. Sin recomendaciones tardías ni OT. NBA48min/cuartos12; WNBA40min/cuartos10. No usar periodos futuros en simulaciones.

`main.py --live` → `analysis/live_system.py` → `LiveProcessor`/`BrowserExtensionClient` en `core/live_api.py` → extensión Chrome/Edge y `core/browser_bridge.py` → `analysis/live_mode.py` → XGBoost guardado, Huber en `analysis/metamodel_integration.py`, salvaguarda y estabilidad del monitor.

Huber usa dos entradas: predicción base y Pacer. No todas las señales impresas son entradas del Huber. La extensión es el transporte probado; no sustituirlo automáticamente por Playwright/stealth. Puente127.0.0.1:8766, panel/pestaña abiertos y código temporal nuevo en cada inicialización. Tras acabar un encuentro, la conexión anterior puede haberse cerrado.

## Prioridades abiertas

1. Frescura/semántica del reloj: hubo marcador nuevo con `time.played` fijo. Comparar datos brutos, periodo, normalización y ficha; no extrapolar timestamps ni usar reloj de timeout sin acreditarlos.
2. `_should_update_prediction` suma sólo Q1–Q3 y no reacciona por sí solo a puntos nuevos de Q4. Acordar política antes de cambiarla. Q3 necesita prueba específica después del arreglo temporal.
3. Tracker/resultados y trazabilidad base/meta/salvaguarda/estabilidad: etiquetas y exportaciones no siempre distinguen capas.
4. Imputación habitual66–68%: distinguir datos futuros no disponibles, históricos recuperables y estadísticas live presentes. No rellenar indiscriminadamente ni confundir confianza impresa con probabilidad calibrada.

Golden State–Portland: predicción prospectiva al descanso247.2482, final246 comunicado por el usuario. Chicago–Minnesota: reconstrucción retrospectiva250.6082, final249. Pretemporada; no acreditan precisión general ni rentabilidad. Detalles y límites en los informes.

## Entorno y verificación

El Python del producto es independiente del que ejecuta Hermes. `README.md` recoge la combinación probada. `requirements.txt` no reproduce ese entorno ni incluye todas las dependencias ML activas. `.diagnostics/` contiene ejecutores y paquetes locales de Codex; no es portable sin comprobar rutas/plataforma.

Verificar lo pertinente al cambio. No ejecutar toda la carpeta tests o scripts automáticamente: algunos consultan red, escriben datos o necesitan interacción. Pruebas live: conexión del usuario, reloj/datos/instante antes del resultado y registros aislados. Conservar modelos y registros normales.

La configuración histórica contiene credenciales: evitar mostrarlas o copiarlas a documentación, prompts, logs o commits. Los códigos temporales del puente sólo sirven para conectar la ejecución actual. No consumir OddsAPI durante una revisión documental.

Cerrar cada revisión actualizando el estado con evidencia, límites y siguiente paso. Distinguir hipótesis, prueba aislada y prueba prospectiva. Los informes fechados pueden describir código anterior a reparaciones posteriores.
