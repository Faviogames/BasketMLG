# Módulo 03: analysis/live_system.py

> Actualización20/10/2026: el usuario autorizó sustituir el menú por búsqueda directa tras seleccionar liga, retirar entrada por ID/manual y mover modelos a `--models`. Ya no existe el fallback de ID cuando la lista está vacía. El cliente se cierra una sola vez y no se importa el procesador para listar modelos. L03-01 y L03-06 quedan sin esas entradas activas; L03-04 se corrige en el nuevo flujo. Los demás hallazgos no se declaran resueltos por esta simplificación. El contenido inferior corresponde a la interfaz anterior.

Fecha: 16 de octubre de 2026. Estado: revisión local terminada; sin cambios funcionales. Se leyó el archivo completo. Las consultas a módulos vecinos comprobaron contratos de llamada, sin completar sus auditorías.

## Responsabilidad y recorridos

Orquesta el sistema interactivo live: carga bundles, selecciona liga y partido, busca la línea de totales, delega el seguimiento y cierra recursos. No calcula predicciones ni estadísticas directamente; tampoco exporta predicciones por sí mismo.

Entradas públicas: `run_basketball_live_system()` desde `main.py --live`, y ejecución directa del archivo. `BasketballLiveSystem` ofrece carga, menú, selectores, vista de modelos y cleanup.

| Opción | Recorrido |
| --- | --- |
| 1 | Selecciona modelo → crea/inicializa `LiveProcessor` → consulta partidos compatibles → selecciona partido o acepta ID si la lista válida está vacía → consulta OddsAPI o pide línea → `monitor_game` → cleanup |
| 2 | Selecciona modelo → pide ID → crea/inicializa procesador → consulta evento → construye `game_info` → OddsAPI/línea manual → seguimiento → cleanup |
| 3 | Intenta abrir el modo manual; actualmente tiene un error de argumentos |
| 4 | Muestra metadatos de modelos cargados |
| 5 | Sale del menú; el punto de entrada limpia recursos y modelos |

## Dependencias y contratos

- `config.MODELS_FOLDER`: ubicación de bundles; `UI_MESSAGES` se importa pero no se usa.
- `joblib.load`: carga objetos `.joblib`; se ignoran explícitamente archivos terminados en `_meta.joblib`.
- `core.live_api.LiveProcessor`: constructor `(model_data, headless=True)`, métodos async `initialize`, `get_available_games`, `monitor_game`, `cleanup`; cliente con `last_discovery_error` y `get_game_event_data`.
- El procesador actual usa `BrowserExtensionClient`, no inicia Playwright por esta ruta. El argumento `headless` se conserva en el contrato existente.
- `analysis.odds_api.OddsApiClient`: import local al flujo; método síncrono `get_odds_for_teams(home, away)`.
- `analysis.live_mode.live_mode_with_alerts`: función síncrona que requiere `trained_data`.
- Entradas por `input`, no por las funciones de `prompt_toolkit` importadas.

Los bundles deben tener al menos `team_names` e `historical_df` para pasar el filtro de claves. El menú de detalles necesita además `features_used`, y la inferencia tiene más requisitos que este cargador no valida. El nombre de archivo sustituye siempre el `league_name` del objeto cargado.

Los eventos por ID se esperan como diccionarios de evento sin envoltorio `event`, con equipos, estado y marcadores opcionales. Se convierten a `game_info`, con ID de entrada, nombres, puntuación y liga tomada del modelo seleccionado.

El intervalo inicial se elige aleatoriamente entre 45 y 60 segundos. La adaptación posterior depende del procesador, no de este módulo. Efectos propios: lectura de modelos, consola, modificación en memoria de su metadata y llamadas a dependencias. No hay escritura directa de predicciones aquí.

## Hallazgos confirmados

### L03-01 — La opción manual del menú está rota

Prioridad alta funcional. Referencia: `analysis/live_system.py:296`; contrato: `analysis/live_mode.py:1837`.

La llamada es `live_mode_with_alerts()` aunque la función requiere `trained_data`. La opción no selecciona liga ni pasa un modelo. Con el modo manual disponible, provoca `TypeError`, capturado por el manejador exterior, y termina el menú entero.

Decisión posterior del usuario: no priorizar reparar esta interfaz; retirar el modo de predicción manual en la fase de mejoras, una vez completada la auditoría. No eliminar ahora la opción ni el módulo de cálculo compartido. Ver `00_ESTADO.md` para el alcance y las funcionalidades que no deben confundirse con ese modo.

Reproducido con el método real del menú y un doble de la función manual con la firma requerida por el código real. No se ejecutó una predicción. El modo manual de `main.py` sí pasa el modelo y no queda invalidado por este hallazgo. No se debe eliminar el modo manual como respuesta al error de esta llamada.

### L03-02 — Confianza y cuartos del listado no usan los campos del productor

Prioridad media. Referencias: `analysis/live_system.py:364`, `analysis/live_system.py:365`; productor: `core/live_api.py:1967`, `core/live_api.py:1970`.

El selector busca `model_confidence` y `estimated_remaining_quarters`. El productor actual añade `compatibility_score` y `remaining_quarters`. Cuando no hay otros campos adicionales, se muestran los valores de respaldo 80 % y dos cuartos.

Reproducido con un partido que contiene compatibilidad 0.93 y un cuarto restante: el selector imprime 80 % y aproximadamente dos cuartos. El 80 % observado por el usuario no acredita una probabilidad calibrada de acierto. Tampoco sería correcto renombrar sin más un score de compatibilidad como «confianza predictiva»: su significado necesita acordarse antes de cambiarlo.

### L03-03 — La validación de bundles deja objetos inutilizables registrados

Prioridad media. Referencias: `analysis/live_system.py:80`, `analysis/live_system.py:91`, `analysis/live_system.py:93`, `analysis/live_system.py:410`.

Hay un filtro útil de tipo y claves mínimas, pero se registra el modelo antes de consultar las longitudes. Si `team_names` es `None`, se captura la excepción del log, pero el objeto ya queda en `self.models`; el cargador puede terminar declarando que encontró modelos válidos. Reproducido.

También se admite un bundle sin `features_used`, que luego rompe la opción 4 con `KeyError`. No se validan aquí los demás campos exigidos para inferencia. No se han deserializado los modelos reales ni se afirma que NBA/WNBA estén dañados. Conecta con M01-01, pero el cargador live no es idéntico al manual de `main.py`.

### L03-04 — Dos referencias de procesador producen cleanup duplicado

Prioridad baja/media de ciclo de vida. Referencias: `analysis/live_system.py:129`, `analysis/live_system.py:221`, `analysis/live_system.py:222`, `analysis/live_system.py:315`, `analysis/live_system.py:422`.

Después de cerrar un seguimiento, se pone `processor = None`, pero `self.current_processor` sigue apuntando al objeto cerrado. Al salir, `system.cleanup()` vuelve a llamar a su cleanup. Se reprodujo también cuando la inicialización devuelve `False`.

No se ha demostrado una fuga de recursos por ello. El transporte actual admite repetición de operaciones de cierre en su camino normal, pero el orquestador queda dependiendo de ese comportamiento. Las excepciones durante limpieza también pueden propagarse desde los `finally`; su efecto real exige la auditoría de los componentes delegados.

### L03-05 — Se aceptan líneas numéricas no válidas para el contrato del dominio

Prioridad media de validación. Referencias: `analysis/live_system.py:205`, `analysis/live_system.py:278`.

Solo se usa `float` y se captura `ValueError`. Valores `nan`, `inf`, negativos y cero llegan a `monitor_game`; se reprodujeron los cuatro casos mediante entrada simulada. Las cadenas no numéricas sí se descartan.

La propagación posterior a probabilidades, comparaciones y exportación no se auditó aquí. Decidir cómo rechazar y volver a pedir una línea debe preservar la opción de continuar sin ella; no se cambia la interacción durante esta revisión.

### L03-06 — El acceso por ID omite el recorrido de compatibilidad y asigna la liga elegida

Prioridad media, contrato entre selección y seguimiento. Referencias: `analysis/live_system.py:161`, `analysis/live_system.py:250`.

La búsqueda automática delega en `get_available_games`, que calcula compatibilidad de liga y equipos. Tanto la opción 2 como el fallback de ID de la opción 1 construyen `game_info` directamente: basta con recibir equipos, y `detected_league` se rellena con el modelo seleccionado.

Se reprodujo un evento etiquetado WNBA con equipos ajenos al bundle NBA: el menú lo entrega al monitor como NBA. Esto demuestra que el orquestador no hace esa validación, no que el procesador consiga predecir correctamente con un modelo incompatible. Sus controles posteriores quedan pendientes de revisión.

Antes de corregirlo, decidir si el ID directo debe permitir partidos fuera del listado, terminados o de pretemporada, y qué compatibilidad mínima debe exigir. No retirar el ID directo, que es una funcionalidad deliberada.

### L03-07 — Fallos en una opción pueden terminar todo el menú

Prioridad media. Referencias: `analysis/live_system.py:107`, `analysis/live_system.py:310`.

El manejador general de excepciones rodea el bucle entero. Una excepción inesperada durante el seguimiento o la vista de modelos se imprime, se limpia y se abandona el menú; no se vuelve a pedir una opción. Se reprodujo con una excepción del doble de monitor y se comprobó que las opciones posteriores no se consumían.

El menú y el punto de entrada también absorben excepciones sin devolver un estado de fallo, relacionado con M01-02. Cancelación por EOF/teclado está atendida en el menú principal y selectores, pero no de manera uniforme en las entradas de ID/línea. La política de recuperar una opción frente a salir debe decidirse con el usuario.

## Otros riesgos y observaciones

- **Errores de OddsAPI silenciosos en opción 2:** `analysis/live_system.py:271` captura y hace `pass`; la opción 1 sí informa del error. Reproducido con una excepción de cuota simulada: el seguimiento continuó sin línea, sin mostrar la causa. Un fallo de OddsAPI no debe confundirse con un fallo del transporte SofaScore.
- **Duplicación de flujos:** las opciones 1 y 2 repiten conversión de evento, consulta de cuotas, lectura de línea y monitorización, con diferencias de mensajes/manejo de errores. Es candidato a refactorización después de fijar el contrato y preservar ambas entradas.
- **Nombre de liga según archivo:** `analysis/live_system.py:87` sobrescribe metadata interna. Es una decisión explícita, no un fallo probado; afecta a rutas con alias o archivos de temporada. Revisar con el sistema de normalización antes de cambiarla.
- **Ruta relativa y recarga:** el cargador usa `./models`, sin protección en `listdir`, como en M01-04/C02-R03. Tampoco vacía el registro antes de recargar; la entrada normal lo llama una vez sobre un objeto nuevo. El riesgo de registros antiguos en futuras recargas no se presenta como fallo actual del flujo normal.
- **Operaciones síncronas en función async:** `input` y consulta OddsAPI ocurren sin delegarlas a un hilo. Pueden bloquear otras tareas del mismo event loop; no se ha demostrado que causen un problema en el recorrido secuencial actual. El puente HTTP de la extensión usa un hilo separado y no debe suponerse bloqueado solo por este hecho.

## Comportamientos que deben conservarse

- Distinción entre lista válida vacía y fallo de acceso: si `last_discovery_error` está presente, vuelve al menú en lugar de ofrecer ID como si no hubiera partidos. Reproducido.
- Inicialización fallida: no se intenta monitorizar y se solicita cleanup. Reproducido.
- Filtro de artefactos `_meta.joblib`: evita tratarlos como bundles principales. Reproducido con nombres simulados.
- Selección de liga, cancelación de selección de partido y salida siguen siendo funcionalidades activas.
- Seguimiento y exportación reales permanecen en el procesador; su funcionamiento observado anteriormente no implica que este módulo exporte por sí mismo.

## Candidatos a código obsoleto, sin borrar

- `add_live_mode_to_main` solo tiene documentación y `pass`. La integración descrita ya existe en `main.py`; no se encontraron llamadas externas en el Python del proyecto inspeccionado. Es candidato a documentación antigua, no a una integración nueva necesaria.
- Imports sin uso interno: `prompt`, `WordCompleter`, `datetime`, `UI_MESSAGES`, `SofaScoreClient`; además `os` se importa dos veces. El import innecesario de `SofaScoreClient` no demuestra que su clase sea eliminable del proyecto.
- Encabezado/docstring describen Playwright como transporte actual, aunque la ruta de producción delega en la extensión. Actualizar texto no equivale a eliminar diagnósticos históricos.
- La modificación global de `sys.path` facilita ejecutar este archivo directamente. No retirarla hasta decidir y verificar qué puntos de entrada deben seguir admitidos.

## Validación y límites

Se ejecutó `.diagnostics/audit_live_system.py`: trece escenarios con aserciones, incluidos varios valores numéricos, usando los cuerpos originales de la clase y funciones compilados desde AST. Se sustituyeron carga de modelos, procesador, función manual, cuotas e inputs por dobles. Los escenarios incluyen rutas normales, errores, registro de modelos, representación y ciclo de vida.

No se cargaron bundles reales, no hubo red, navegador ni puente, y no se entrenó, predijo ni exportó nada. El script no importa los módulos ML ni ejecuta sus cuerpos. Las firmas/campos de dependencias se contrastaron por lectura puntual. Se consultó un test de exportación existente como referencia, sin usarlo como prueba del orquestador ni auditar todavía la suite completa.

No se modificó `analysis/live_system.py` ni ningún otro archivo funcional. No se ha explicado aún la imputación del 66 % de características.

Siguiente módulo sugerido: `core/live_api.py`. Es un archivo extenso con responsabilidades de cliente y procesador; su revisión deberá distinguir claramente ambos bloques y marcar cobertura parcial si se divide en turnos. Esta revisión se detiene aquí.
