> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](Documentation/DOCUMENTATION_INDEX.md) y [estado vigente](Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Revisión del estado — 15 de octubre de 2026

## Resultado

El proyecto conserva código, datos y modelos, pero no queda validado para uso operativo. No se ha modificado la lógica, reentrenado modelos ni alterado registros de predicción.

## Comprobaciones realizadas

- Sintaxis: 97 archivos Python analizados con AST; ninguno presenta errores de sintaxis. Esto no verifica imports ni ejecución.
- Entorno: `py -0p` no encuentra instalaciones; `python`, `python3` y `pytest` no están accesibles por los comandos habituales. El Python auxiliar de Codex es 3.12.14 y carece de sklearn, joblib, XGBoost, LightGBM, Playwright, scipy, requests, pytest y prompt_toolkit. No se han instalado dependencias.
- `requirements.txt` omite scikit-learn, joblib, xgboost, lightgbm y playwright, necesarios en distintas rutas del proyecto. Su instalación por sí sola no reconstruye el entorno.
- Existen bundles NBA/WNBA y `metamodel/models/meta_learner_v1_final.joblib`. No se ha podido comprobar su deserialización o compatibilidad sin las dependencias.
- La API de cuotas responde HTTP 200 con la clave configurada: 84 deportes y cabecera de cuota restante 499. Se consultó únicamente el catálogo de deportes y no se expuso la clave.
- SofaScore devuelve HTTP 403 tanto en `/basketball` como en `/api/v1/sport/basketball/events/live` con peticiones HTTP directas. La prueba no valida el acceso mediante sesión Chrome/Playwright usado por la aplicación; no permite declarar el servicio caído.
- La carpeta no es un repositorio Git: no se dispone de historial o estado de cambios mediante Git.

## Datos y arquitectura

- El entrenamiento toma datos de `./leagues`, donde solo existe NBA: 1.556 partidos del 23/10/2024 al 23/11/2025.
- El NBA.json de la raíz es otra copia: 1.386 partidos hasta el 23/06/2025. WNBA.json contiene 555 hasta el 12/09/2025 y queda fuera del directorio configurado para entrenar.
- El bundle NBA declara creación el 28/11/2025, 2.968 partidos y MAE 3,35449. Ese número de partidos no coincide con el fichero actual de entrenamiento; hace falta reconstruir su procedencia.
- AI_CONTEXT.md y MODULE_MAP.md describen LightGBM como motor live. La integración activa carga un HuberRegressor con dos entradas: predicción base y proyección Pacer. Las señales avanzadas no son entradas de ese modelo.

## Hallazgos que requieren atención

1. **Diferencia entre entrenamiento e inferencia del Pacer.** `scripts/generate_meta_features.py` usa `calculate_pace_poisson`, que mezcla la proyección con mercado o 220 puntos. `analysis/metamodel_integration.py` usa `calculate_pacer_features`, que extrapola sin esa mezcla. Reproducción aislada con las funciones originales: 112 puntos al descanso y sin línea producen 222,4 en entrenamiento y 224 en inferencia. La magnitud depende del marcador y de la línea.
2. **Contrato temporal frágil.** `calculate_pacer_features` exige `played_seconds`: con solo `total_minutes_played=24` devuelve 448 para esos 112 puntos. La ruta principal de `analysis/live_mode.py` sí agrega segundos; el script `scripts/verify_metamodel_integration.py` no lo hace. No se debe interpretar esa prueba como una validación correcta del live.
3. **Script de acceso roto.** `scripts/test_event_access.py` usa `page` sin asignarla tras crear el contexto. Pasar la sintaxis no evita ese NameError al ejecutarlo.
4. **Validación temporal pendiente.** El entrenamiento base y el Huber usan `train_test_split` aleatorio. Sus métricas no demuestran rendimiento en partidos futuros. No se ha confirmado fuga de información; hace falta una evaluación cronológica independiente.
5. **Duplicación de métodos.** SofaScoreClient define dos veces `get_game_statistics`; Python utiliza la última implementación. Complica saber qué comportamiento se está probando.

## Métricas descriptivas, no una evaluación nueva

Los CSV live y meta contienen 1.556 filas. Sobre todo el CSV meta, el error absoluto medio de la columna base es 11,6438 puntos y el de la columna Pacer es 11,3623. Son cálculos descriptivos de columnas ya guardadas: no son resultados de un test independiente ni validan el Huber.

## Orden propuesto para retomarlo

1. Crear un entorno local reproducible e instalar las dependencias mínimas; comprobar imports, CLI y carga de bundles.
2. Verificar SofaScore con sesión de navegador y distinguir bloqueo, ausencia de partidos y fallo de estadísticas.
3. Unificar fórmula Pacer y contrato de tiempo, y revisar la necesidad de regenerar datos y reentrenar el Huber.
4. Unificar la fuente histórica, actualizar datos y evaluar cronológicamente contra un baseline.
5. Recuperar control de versiones y actualizar la documentación a la arquitectura real.

No se ejecutó la batería completa ni un ciclo live extremo a extremo por falta del entorno. El estado de esas rutas sigue pendiente de verificación.

## Seguimiento: sesión preparada por el usuario

Se inspeccionó la pestaña del navegador integrado que el usuario preparó navegando por SofaScore. Las estadísticas de `16685169` mostraban JSON válido. En esa misma sesión, `/api/v1/sport/basketball/events/live` devolvió 11 eventos: cinco de NBA Preseason, incluido Spurs–Suns. `/basketball` cargó los enlaces de partidos; `/event/16685169` redirigió correctamente a la página pública del partido. No se ha observado un cambio de esos endpoints ni de la estructura `events`.

Esto confirma que el 403 depende del contexto de acceso en las pruebas realizadas, pero no determina qué petición falla en la ejecución local del usuario: aún no se ha obtenido su log ni ejecutado el CLI con sus dependencias.

Cambios en `core/live_api.py`: esperar la verificación antes de rechazar el estado inicial; esperar enlaces de partidos en lugar de una pausa fija; recuperar una pestaña SofaScore existente; conservar una pestaña con JSON válido; distinguir fallos de acceso de listas vacías y validar la estructura del listado. El menú muestra ahora el error de acceso. Se reparó la variable `page` no definida del script de acceso y se añadió `scripts/diagnose_sofascore.py`.

Validación: diez pruebas aisladas de regresión pasan. Una reproducción con las identidades, liga y estados de los cinco partidos NBA observados conserva los cinco usando los métodos originales de filtrado y compatibilidad (nombres de equipos conocidos suministrados a la prueba). No es una ejecución del bundle serializado ni del cliente Playwright completo. La sintaxis de los archivos modificados es válida.

La sesión del navegador integrado no es el perfil `.browser_profile` de la aplicación. El resultado real del arranque de `python main.py --live` sigue pendiente de comprobar en ese entorno; no se declara resuelto solo por las pruebas aisladas.

## Restauración del modo oculto y prueba real del cliente

A petición del usuario, `SofaScoreClient` y `LiveProcessor` tienen ahora `headless=True` por defecto; las dos rutas del menú live lo especifican también. Se recuperó el patrón `chromium.launch` + `browser.new_context` observado en la copia compilada `core/__pycache__/live_api.cpython-311.pyc` de noviembre de 2025, en lugar del perfil persistente. Chromium es la primera opción y Chrome instalado la alternativa; ambas se lanzan ocultas. Se conserva la configuración de agente de usuario, viewport y locale encontrada en esa copia, sin incorporar su antiguo argumento para desactivar la seguridad web.

Se instaló Playwright únicamente en `.diagnostics/python`, sin modificar el Python habitual. `.diagnostics/check_discovery.py` ejecuta la clase cliente del archivo actual aislándola de los imports ML; realiza peticiones reales y no usa eventos simulados.

Resultados: el arranque oculto funciona con Chrome instalado (el Chromium correspondiente al Playwright de diagnóstico no está instalado). Con el perfil persistente, las peticiones de la propia página a `live-tournaments`, `scheduled-tournaments`, y la del cliente a `events/live` devolvieron 403. El host `api.sofascore.com` también devolvió 403. Con el contexto nuevo recuperado, `/basketball` redirigió a verificación humana. Por tanto, la apertura de ventanas está corregida, pero la obtención automática del listado en modo oculto **no está resuelta** en la prueba real.

Las 13 pruebas de regresión pasan, incluidas inicialización oculta, alternativa Chrome oculta y fallo explícito ante verificación en segundo plano. Esas pruebas no sustituyen el resultado de red fallido descrito arriba. No se ha completado una ejecución de `main.py` con los modelos.

## Contraste con el log del usuario

El log recibido reproduce la verificación pendiente al buscar y el 403 al consultar un ID manual. Se corrigió la salida prematura que se había introducido para modo oculto: ahora se permiten hasta 20 segundos para que la comprobación automática redirija por sí sola, sin interacción con CAPTCHA. También se retiró la afirmación de sesión establecida antes de probar la API y el menú no invita a introducir un ID manual cuando ya se sabe que el acceso está bloqueado.

Se descargó el Chromium Headless Shell correspondiente a Playwright 1.64.0 en `.diagnostics/browsers` y se repitió el cliente real con ese ejecutable, sin alternativa Chrome. La verificación continuó pendiente tras los 20 segundos. Esto descarta que la ausencia de ese Chromium fuera la única explicación del fallo. El acceso automático sigue pendiente de solución; 14 pruebas de regresión pasan.

## Revisión de backups y separación de módulos

El único backup fuente localizado en la carpeta es `analysis/live_mode backup.py`. No contiene llamadas a Playwright, navegación, SofaScoreClient ni descubrimiento de partidos. No se encontró un archivo fuente `live_api backup`; sí permanece la copia compilada de 2025 utilizada como referencia parcial. La documentación y los tests antiguos sitúan la adquisición de datos en `core/live_api.py`, no en los modelos ni en los cálculos del modo manual.

Se ejecutó una prueba adicional con Chrome instalado 155.0.8059.40, oculto y con configuración estándar de Playwright, sin los argumentos especiales y sin el agente Chrome 120 heredados del código antiguo. La verificación permaneció pendiente. Los ensayos no han demostrado una corrección del acceso; restaurar ajustes antiguos no garantiza recuperar la aceptación del servidor.

Advertencia sobre la evidencia histórica: `tests/test_live_api.py` considera éxito una lista vacía y también imprime éxito final aunque no consiga datos de un partido. Esos mensajes históricos no prueban un ciclo live completo correcto. Las pruebas aisladas añadidas tampoco equivalen a una respuesta real 200.

## Traza por etapas del flujo solicitado

Se añadió `.diagnostics/trace_flow.py`, que inicia la clase cliente real oculta y registra navegación, redirecciones, respuestas API y disponibilidad de datos sin cargar modelos. Resultado completo en `.diagnostics/sofascore_flow_trace.json`:

1. Inicio del navegador: correcto, 0,38 segundos.
2. GET `/basketball`: HTTP 302 a `/captcha.html`; la página de verificación responde HTTP 200. A los 3,17 segundos ya se observa título Sofascore, texto de verificación y cero enlaces de partidos.
3. Tras esperar 20 segundos más: continúa en `/captcha.html`.
4. Listado, selección y estadísticas: no ejecutados porque la sesión nunca llegó a baloncesto. No se atribuye el fallo a esas etapas ni al modelo.

Como comparación, se reprodujo el recorrido completo en la pestaña preparada por el usuario: `/basketball` mostró el encabezado de baloncesto, el endpoint live devolvió nueve eventos sin error, se abrió `/event/16869090` y redirigió a la página Dallas Mavericks–New Orleans Pelicans, y `/api/v1/event/16869090/statistics` devolvió `ALL`, `1Q`, `2Q`, `3Q`, `4Q` con estadísticas. No se ejecutaron cálculos del modelo en esa comparación. La navegación del navegador integrado no expone aquí los estados HTTP; se verificó el contenido DOM/JSON, no se inventan estados de red para esa sesión.

Conclusión acotada: el primer fallo reproducido está en la entrada a `/basketball` de una sesión nueva oculta, antes de solicitar el listado. El mismo HTTP 200 puede pertenecer a la página de verificación y no prueba acceso a partidos. Se agregó al cliente un log `[FLOW]` con el estado y URL final de esa navegación. La causa que determina por qué el servidor acepta la sesión preparada y rechaza la nueva sigue sin resolverse.

## Preparación y reutilización de sesión

La inspección de metadatos del almacén de cookies de `.browser_profile` encontró solo `_gcl_au`, `_ga` y `_ga_HNQ9P9MGZR`; no se leyó ni mostró ningún valor de cookie. No había estado de Playwright exportado. Esto no demuestra que renovar cookies baste, pero el contexto vacío no podía reutilizar una validación entre ejecuciones.

Se añadió `core/sofascore_session.py`: carga estado local de `.browser_profile/sofascore_session.json`, descarta cookies caducadas al cargar y guarda mediante sustitución atómica. El cliente lo carga al iniciar y actualiza solo tras conseguir el JSON válido del listado. Se eliminó el agente fijo Chrome 120; la identidad usa la versión real del navegador, compartiendo configuración entre preparación visible y uso oculto. Se prefiere Chrome instalado en ambos modos y Chromium como alternativa. `.gitignore` excluye estado del navegador y diagnósticos.

Prueba disponible para el usuario:

```
python scripts/prepare_sofascore_session.py
python scripts/prepare_sofascore_session.py --check
python main.py --live
```

El primer comando abre una ventana expresamente y espera a que el usuario complete cualquier verificación y pulse Enter en consola. No automatiza CAPTCHA ni necesita login en una cuenta. Solo guarda si `/sport/basketball/events/live` devuelve HTTP 200 con `events` como lista. El segundo comprueba la reutilización oculto. Una verificación fallida no reemplaza la sesión anterior. Este procedimiento está implementado, pero no se ha ejecutado su paso visible ni validado el resultado oculto con una sesión recién preparada. No se declara solucionado el acceso real.

Validación: 15 pruebas de descubrimiento más cuatro de almacenamiento pasan; CLI `--help` y sintaxis comprobados. Referencias: https://playwright.dev/python/docs/auth y https://developers.cloudflare.com/cloudflare-challenges/troubleshooting/challenge-solve-issues/ .

## Resultado de la preparación visible

Se ejecutó la utilidad de preparación con aprobación para abrir la ventana inicial. El usuario mostró una captura con `Verification failed` y `Access denied`. El navegador fue cerrado antes de validar el listado; no existe `.browser_profile/sofascore_session.json`. Por tanto, no se guardó una sesión válida y la persistencia implementada no ha solucionado el acceso. La utilidad ahora informa de un cierre/interrupción sin mostrar el traceback interno de Playwright.

Una traza oculta adicional no obtuvo un código numérico de Turnstile. Registró un error DNS en un subdominio de `challenges.cloudflare.com`, que no se considera causa demostrada: la documentación de Cloudflare indica que algunos subdominios de prueba DNS fallan de forma esperada y no son fatales. La navegación siguió siendo `/basketball` 302 -> `/captcha.html` 200.

Cloudflare documenta expresamente que los navegadores automatizados, incluidos Playwright, no están admitidos para resolver verificaciones de producción: https://developers.cloudflare.com/cloudflare-challenges/reference/supported-browsers/ . Esto es compatible con el fallo observado, pero no identifica por sí solo la causa exacta de la captura. No se automatizó ni resolvió el CAPTCHA. El acceso real sigue sin recuperarse.

## Transporte mediante extensión del navegador del usuario

Por petición del usuario, se ha implementado `browser_extension/` (Manifest V3 para Chrome/Edge) y `core/browser_bridge.py`. `LiveProcessor` utiliza ahora `BrowserExtensionClient`, que conserva los parsers existentes y sustituye las consultas de Playwright por consultas desde la pestaña de SofaScore seleccionada por el usuario. Python no abre el navegador. El cliente anterior queda disponible para diagnóstico, pero no es el transporte por defecto del modo live.

El panel de la extensión se mantiene abierto y consulta un puente HTTP que escucha exclusivamente en `127.0.0.1:8766`. Cada cliente genera un código aleatorio de conexión. Se verifican token, Host, origen y rutas; no se copian cookies al proceso Python ni se guarda el código. Solo se admiten listado live de baloncesto, evento por ID y sus estadísticas. Las verificaciones de SofaScore siguen siendo manuales en el navegador. Las estadísticas sin ALL o sin categorías requeridas se rechazan en lugar de producir datos para una predicción.

Guía de instalación y uso: `browser_extension/README.md`. Prueba manual sin dependencias ML: `python scripts/test_browser_bridge.py --event ID_ACTUAL`.

Validación inicial local: 16 pruebas Python nuevas (HTTP local real, validación y cliente con respuestas simuladas), cinco pruebas JavaScript con APIs Chrome simuladas y las 19 regresiones anteriores pasaron; sintaxis Python/JavaScript y ayuda del script comprobadas. Estas pruebas no demuestran que SofaScore acepte las consultas de la extensión. El flujo completo con modelos y partido real queda pendiente de esa comprobación.

### Resultados aportados por el usuario y versión 0.2.0

Tras instalar la extensión, el usuario recibió primero `No tab with id`. Se corrigió la resolución de pestañas obsoletas: si solo queda una pestaña válida de SofaScore, se selecciona; si hay varias, se pide elegir. El siguiente intento llegó a SofaScore y recibió HTTP 403 desde la consulta fetch. El posterior `Failed to fetch` correspondía al cierre del puente tras el error, no a la primera causa. Ahora el panel detiene las consultas al recibir un error de acceso y conserva ese mensaje.

La versión 0.2.0 añade un modo predeterminado de navegación: abre el endpoint en la pestaña dedicada y lee su JSON del documento, en lugar de usar fetch. Si el usuario ya abrió el enlace exacto, se lee ese documento una vez; las consultas posteriores vuelven a navegar para renovar los datos. Se validan las claves esperadas y se rechazan HTML, redirecciones a captcha y objetos JSON de acceso denegado. No se interactúa con verificaciones. La consulta fetch queda como modo de comparación.

Validación de esta revisión: 14 pruebas JavaScript simuladas y las 16 pruebas Python del puente/cliente pasan. No se ha validado todavía el modo de navegación con la sesión real del usuario. Que una navegación normal pueda comportarse distinto a fetch es la hipótesis de esta prueba, no un acceso recuperado demostrado.

### Validación real comunicada por el usuario

El usuario confirmó el listado live y la selección de Los Angeles Clippers contra Denver Nuggets (53–48, segundo cuarto). Posteriormente mostró una predicción de 221.4 puntos, exportación automática y una segunda actualización que obtuvo 34 métricas, con avance del reloj de 22.9 a 23.4 minutos jugados. En esa segunda actualización el sistema mantuvo la predicción por su regla de estabilidad ante un cambio de cuatro puntos. Esto confirma el recorrido real de listado, evento, estadísticas, cálculo, exportación y una renovación posterior para ese encuentro; no constituye una validación de precisión del modelo ni de disponibilidad permanente.

Queda un problema distinto: se imputaron 124 de las 188 características del modelo. El código cuenta valores NaN en las columnas de entrada del modelo, que incluyen características históricas y derivadas; ese porcentaje no mide directamente campos ausentes en el endpoint de SofaScore. Hay que identificar las columnas concretas y contrastarlas con la construcción de características y la caché previa al partido. Además, `_show_live_update` usa la predicción final como sustituto de `base_prediction` si esta falta, por lo que la etiqueta de consola `Base Model` no es evidencia suficiente del valor original de XGBoost. No se han modificado modelos, fórmulas ni estos mecanismos durante la recuperación del transporte.
