> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Comparación aislada de Playwright y playwright-stealth

Fecha de la sesión: 16 de octubre de 2026.

Se ejecutó una comparación secuencial real: primero Playwright normal como control, después Playwright con `playwright-stealth` 2.0.3 mediante su API recomendada `Stealth().use_async(...)`. Ambos casos utilizaron Google Chrome 155.0.8059.80 en modo oculto, contextos nuevos, viewport 1280×720, locale en-US y la misma espera pasiva de 20 segundos. No se importaron cookies ni perfiles del usuario, ni se interactuó con verificaciones. El plugin se instaló exclusivamente en `.diagnostics/stealth-python`; la aplicación y la extensión no se modificaron.

| Comprobación | Control sin plugin | Con plugin |
| --- | --- | --- |
| `/basketball` | HTTP 200, página de baloncesto | HTTP 200, página de baloncesto |
| Redirección a captcha detectada | No | No |
| `navigator.webdriver` antes de navegar | true | false |
| Listado live | HTTP 200, 53 eventos | HTTP 200, 53 eventos |
| Evento 16665307 | HTTP 200, objeto `event` | HTTP 200, objeto `event` |
| Estadísticas del evento | HTTP 200, lista `statistics`, periodo ALL | HTTP 200, lista `statistics`, periodo ALL |

El evento seleccionado fue KK Crvena zvezda contra Paris Basketball. No se encontraron eventos cuyo nombre de torneo contuviera NBA en ese listado, por lo que se utilizó el primero disponible. No se ejecutaron modelos ni se validó la precisión de predicciones; la comprobación de estadísticas se limitó a la respuesta y su estructura de periodos.

Conclusión: la prueba con el plugin consiguió obtener las respuestas, pero el control sin plugin también. El rechazo de la sesión anterior no se reprodujo, por lo que este resultado no permite atribuir al plugin la recuperación del acceso ni demostrar que resuelva el bloqueo anterior. Es una comprobación puntual, no una garantía de estabilidad. La versión de Chrome utilizada ahora es distinta de la 155.0.8059.40 registrada en diagnósticos anteriores; no se ha aislado ese factor ni otros cambios entre sesiones.

Reproducción: `.diagnostics/probe_stealth.py`. Informe estructurado: `.diagnostics/stealth_comparison.json`. Las dos ejecuciones finalizaron y sus navegadores de prueba se cerraron.
