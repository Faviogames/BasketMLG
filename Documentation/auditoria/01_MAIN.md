# Módulo 01: main.py

> Actualización20/10/2026: CLI simplificada por petición del usuario. `--live` inicia búsqueda sin menú, `--models` muestra modelos, sin argumentos hay ayuda y los flags son excluyentes. Se retiró la ruta manual y los imports de flujos se realizan bajo su comando. La auditoría de abajo describe el código anterior; no reintroducir el modo manual ni interpretar esos recorridos como interfaz vigente. Ver `00_ESTADO.md` y `tests/test_cli_live.py`.

Fecha: 16 de octubre de 2026. Estado: revisión local terminada; sin modificar código funcional. Esta revisión no certifica sus dependencias ni el proyecto completo.

## Responsabilidad y recorrido

`main.py` es el punto de entrada de consola. No calcula características ni predicciones: selecciona un flujo y delega.

| Invocación | Comportamiento observado | Delegación |
| --- | --- | --- |
| `python main.py --live` | Inicia el sistema live mediante `asyncio.run` | `analysis.live_system.run_basketball_live_system` |
| `python main.py --train` | Llama al orquestador de entrenamiento | `core.training.train_models_by_league` |
| `python main.py` | Busca bundles `.joblib`, muestra ligas, pide selección y abre el modo manual | `analysis.live_mode.live_mode_with_alerts` |
| Ambos flags | Da prioridad a live; no ejecuta entrenamiento | Prioridad explícita en el código |

El modo manual es el comportamiento predeterminado y sigue alcanzable. No es código muerto por estar llamado «clásico» o «legacy».

## Contratos y dependencias

- Imports directos externos: `joblib`, `prompt_toolkit`. Imports estándar: `os`, `argparse`, `asyncio`, `traceback`, `datetime`.
- Imports propios: `MODELS_FOLDER` y `UI_MESSAGES` de `config`, más los tres puntos de delegación anteriores. Se resuelven al importar el archivo, antes de leer argumentos.
- Entrada manual: archivos cuyo nombre termina exactamente en `.joblib` dentro de `MODELS_FOLDER`. El nombre de liga se obtiene del nombre del archivo sustituyendo guiones bajos por espacios y quitando `.joblib`.
- El menú espera que cada objeto cargado tenga `team_names` e `historical_df`, ambos con longitud consultable. No se valida el contrato en este módulo. No se han deserializado bundles reales durante la auditoría.
- Salidas: consola, ejecución de uno de los tres flujos y, en excepciones capturadas, un append a `crash_log.txt` en el directorio de trabajo. El módulo no guarda ni transforma modelos directamente; los efectos de las funciones delegadas quedan para sus auditorías.
- Comprobación puntual de la dependencia de ruta: `config.py:270` define `MODELS_FOLDER = './models'`. Se consultó esa constante y las claves del banner, sin auditar `config.py` completo.

## Hallazgos confirmados

### M01-01 — Bundle cargable pero inválido rompe el listado manual

Prioridad media. Referencias: `main.py:82`, `main.py:95`, `main.py:97`.

`joblib.load` se captura, pero acceder después a los metadatos del objeto queda fuera del bloque protegido. Un diccionario cargado correctamente que no contiene `team_names` causa `KeyError` al mostrar las ligas. Un único bundle incompatible puede impedir llegar a los demás válidos. Reproducido con un doble de carga que devuelve metadatos incompletos; no se afirma que los bundles actuales estén corruptos.

### M01-02 — Una excepción capturada no se propaga como fallo del comando

Prioridad media para ejecución por scripts. Referencias: `main.py:50`, `main.py:60`, `main.py:104`, `main.py:113`.

Tras registrar una excepción, los flujos retornan normalmente. No se retorna un estado de error ni se llama a `sys.exit` con uno. En la entrada normal de este archivo, un fallo capturado y registrado puede terminar con código de salida cero. Se reprodujo que una excepción simulada del entrenamiento se registra y `main()` retorna `None` sin excepción. No se inició entrenamiento real.

### M01-03 — El fallo del propio logger puede sustituir al error original

Prioridad media. Referencias: `main.py:24`, `main.py:26` y llamadas a `log_crash`.

Si `crash_log.txt` no puede abrirse, el logger lanza otra excepción sin mecanismo de respaldo. Puede impedir la continuación tras un fallo de carga de un modelo, además de ocultar el tratamiento previsto del error inicial. Reproducido haciendo que `open` lance `PermissionError`; no se modificaron permisos ni archivos reales.

### M01-04 — Listado de modelos sin protección frente a rutas inaccesibles o incorrectas

Prioridad baja/media, según entorno. Referencias: `main.py:71`, `main.py:78`.

`exists` no comprueba que la ruta sea un directorio; las dos llamadas a `listdir` tampoco capturan errores. Si `models` existe como archivo, el menú termina con `NotADirectoryError`. Reproducido con un doble del sistema de archivos. Un problema de acceso puede seguir el mismo camino de excepción no capturada; no se provocaron errores reales de permisos.

## Riesgos y decisiones, sin clasificarlos como código obsoleto

### M01-R01 — Acoplamiento de dependencias al arrancar

Referencias: `main.py:8`, `main.py:17`, `main.py:18`, `main.py:19`, `main.py:22`.

Los imports de entrenamiento y ambos modos de predicción ocurren antes de `parse_args`. Por inspección, una dependencia que falle en esos imports puede impedir incluso `--help` o un modo distinto. No se han medido tiempos de arranque ni auditado el contenido completo de esos imports. Posible mejora futura: imports por flujo; queda pendiente para la fase de cambios.

### M01-R02 — Rutas relativas al directorio desde el que se ejecuta

Referencias: `main.py:26`, `main.py:71`, `config.py:270`.

Al ejecutar el archivo desde otro directorio, `./models` y `crash_log.txt` se resuelven respecto a ese directorio, no al de `main.py`. Puede impedir localizar modelos o usar otros distintos. Cambiarlo sería una decisión sobre rutas compartidas con otros módulos y requiere revisión conjunta; no se propone corregirlo aisladamente todavía.

### M01-D01 — Prioridad silenciosa cuando se pasan ambos flags

Referencias: `main.py:38`, `main.py:40`, `main.py:48`.

`--train --live` solo ejecuta live. Reproducido con delegaciones simuladas. El orden está comentado explícitamente, así que no se declara un error de implementación. Antes de modificarlo, decidir con el usuario si ambos flags deben ser incompatibles o si esta prioridad es intencionada.

### M01-D02 — Autocompletado sin distinguir mayúsculas, selección exacta

Referencias: `main.py:100`, `main.py:103`.

El autocompletado ignora mayúsculas, pero la selección debe coincidir exactamente con la clave. Escribir `nba` cuando la clave es `NBA` se rechaza si no se acepta la sugerencia. Reproducido. Mejora potencial de interfaz, no un fallo del modelo ni motivo para eliminar el modo manual.

### M01-D03 — Versiones de presentación inconsistentes

El encabezado y descripción de argparse dicen v2.2; el banner de `UI_MESSAGES` dice v3.0. Es documentación/presentación desactualizada. No prueba obsolescencia funcional de los flujos.

## Código obsoleto o eliminable

No se ha demostrado ninguna función, importación o rama eliminable en `main.py`. Las tres rutas tienen entrada explícita. No autorizar borrados a partir de sus comentarios históricos. Los imports adelantados podrían reorganizarse en el futuro, pero eso no equivale a que sus módulos sobren.

## Validación y límites

Se leyó el archivo completo y se ejecutaron seis escenarios aislados con los cuerpos originales de `main` y `log_crash`, sustituyendo dependencias por dobles. Reproducción: `.diagnostics/audit_main.py` con Python estándar. Los seis escenarios verifican comportamientos observados, no son seis garantías de corrección.

No se accedió a SofaScore/OddsAPI, no se abrió navegador, no se cargaron modelos reales, no se entrenó ni se escribió `crash_log.txt`. No se modificó `main.py` ni ningún otro archivo funcional. Se consultaron `MODULE_MAP.md` y `AI_CONTEXT.md` como documentación orientativa; su descripción de otros módulos queda por contrastar y no prevalece sobre las instrucciones del usuario.

## Continuidad

Siguiente módulo sugerido: `config.py`, para entender rutas, selección de características, parámetros y mensajes que comparten el resto de módulos. Esta revisión se detiene aquí. Las mejoras, incluida la política de flags, se decidirán cuando se complete la auditoría y se consulte al usuario sobre cambios clave.
