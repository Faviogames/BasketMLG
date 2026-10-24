# Módulo 02: config.py

Fecha: 16 de octubre de 2026. Estado: revisión local terminada; sin cambios funcionales. Se leyó el archivo completo. Las consultas a consumidores solo comprobaron el uso de su configuración; no equivalen a auditar esos módulos.

## Responsabilidad y dependencias

Centraliza rutas, credencial de OddsAPI, parámetros de señales/contexto, catálogo de características, exclusiones persistentes, flags de ablación, normalización de estadísticas, valores por defecto y textos de interfaz.

No es únicamente declarativo: al importarse lee `useless_features.json`, aplica filtros, construye listas globales y escribe un resumen en consola. Depende de `os`, `json` y `pandas`; `Path` está importado pero no se usa internamente. Pandas se utiliza para fechar el guardado de exclusiones.

Entradas: constantes del archivo y el JSON de exclusiones. Salidas: constantes/listas y funciones públicas. `save_useless_features_to_json` modifica el JSON; ninguna otra función del módulo escribe archivos. Ejecutarlo directamente muestra diagnósticos.

Consumidores encontrados: `main.py`, `core/training.py`, `core/features.py`, `core/live_features.py`, `core/data_processing.py`, `analysis/live_mode.py`, su backup, `analysis/live_system.py`, `analysis/odds_api.py`, `analysis/feature_importance.py` y cuatro scripts/tests adicionales.

## Catálogo comprobado en el estado actual

| Elemento | Cantidad |
| --- | ---: |
| Estadísticas con EMA | 13 |
| Ventanas EMA | 3, 8 y 11 |
| Variantes EMA home/away/diff | 117 |
| Bases de características directas | 33 |
| Variantes directas home/away/diff | 99 |
| Catálogo combinado sin filtrar | 272 |
| Catálogo combinado filtrado | 259 |
| Catálogo histórico sin filtrar | 250 |
| Catálogo histórico filtrado | 239 |

No hay nombres duplicados en esas listas. El JSON actual contiene 11 exclusiones, todas presentes en el catálogo. Solo dos de las 25 exclusiones hardcoded coinciden con nombres del catálogo actual: `is_potential_blowout` y `garbage_time_risk`. Las otras pueden corresponder a versiones anteriores; esto no autoriza borrarlas.

La lista histórica no incluye los nombres declarados en `LIVE_GAME_FEATURES` o `BALANCE_FEATURES`. Esto confirma la separación de listas, no la ausencia de toda fuga de información: verificar el origen y momento de cálculo de cada variable requiere auditar entrenamiento y generación de características.

Los 239 nombres configurados no tienen por qué coincidir con las 188 entradas de un modelo guardado. El consumidor de entrenamiento intersecta la lista con las columnas disponibles (`core/training.py:833`), y los artefactos tienen su propio `features_used`. No se ha atribuido a esta diferencia la imputación del 66 %.

## Hallazgos confirmados

### C02-01 — Parámetros publicados que el live no consulta

Prioridad media. Referencias: `config.py:47`, `config.py:87`, `config.py:95`; imports en `analysis/live_mode.py:16`.

`REALITY_CHECK_V2`, `CACHE_CONTROL` y `DATA_QUALITY_CONFIG` se importan en `analysis/live_mode.py` y su backup, pero no tienen lecturas de sus nombres en sus árboles de código. No se encontraron otros consumidores Python en los directorios inspeccionados.

La consulta puntual del consumidor muestra umbrales literales para caché (50 %) y alertas por imputación (50/30 %), en vez de usar estos diccionarios. Modificar sus claves no ajusta esas ramas. Los controles de Reality Check declarados tampoco se conectan mediante ese import. No se ha auditado todavía la implementación completa de esos ajustes.

Decisión futura: conectar parámetros al comportamiento o retirar configuración engañosa, conservando resultados y consultando al usuario antes de alterar reglas. No borrar estos bloques ahora.

### C02-02 — El JSON se lee sin validar su estructura y puede impedir el arranque

Prioridad media. Referencias: `config.py:563`, `config.py:599`, `config.py:600`.

La función de lectura captura errores de sintaxis JSON, pero no valida que la raíz sea un diccionario ni que `useless_features` sea una lista de cadenas. Se reprodujo:

- Raíz `[]`: `AttributeError` no capturado.
- Lista con elementos no hashables: `TypeError` no capturado.
- Valor de tipo cadena: se convierte silenciosamente en un conjunto de caracteres.
- Archivo inaccesible: `PermissionError` no capturado.

Como esta lectura ocurre durante el import, ciertos fallos del archivo pueden bloquear todos los modos antes de leer argumentos. El JSON real actual es legible y tiene una estructura compatible; no está demostrado que esos errores ocurran hoy.

### C02-03 — Guardado no atómico y sin señal de éxito para el llamador

Prioridad media. Referencia: `config.py:574`.

El guardado abre el archivo final con `w`, escribe directamente y captura todas las excepciones imprimiendo un mensaje. Una interrupción puede dejar JSON parcial y perder el contenido anterior. Se reprodujo con un archivo en memoria que falla durante `json.dump`.

La función retorna `None` tanto si guarda como si falla. El llamador de entrenamiento imprime que se guardaron exclusiones después de invocarla (`core/training.py:948`), sin comprobar un estado. No se simuló un entrenamiento completo ni se alteró el archivo real.

### C02-04 — Credencial de OddsAPI incrustada en el código

Prioridad alta para gestión de credenciales, sin indicios de compromiso demostrados. Referencia: `config.py:16`; consumidor: `analysis/odds_api.py:17`.

La clave es un literal del archivo y el cliente de cuotas la importa. No se reproduce su valor en este informe. Compartir el código también compartiría la credencial. Una futura migración a configuración local/entorno debe preservar el acceso de OddsAPI y acordarse antes de cambiarlo; no se ha movido ni rotado la clave durante la auditoría.

### C02-05 — Resumen de características desactualizado

Prioridad baja, presentación. Referencia: `config.py:643`.

El resumen estima 190 características mientras que el catálogo combinado bruto contiene 272 nombres únicos. Cuenta las 33 bases directas como una variante cuando el catálogo las expande a 99 y no suma todos los grupos. Hay comentarios que siguen hablando de nueve bases EMA, doce exclusiones y una mejora porcentual fija, aunque las listas actuales tienen otros tamaños. No se ha validado la mejora de MAE anunciada por esos textos.

## Riesgos y decisiones que requieren contexto del proyecto

### C02-R01 — Exclusiones globales acumuladas entre ligas/modelos

Referencias: `config.py:271`, `config.py:574`; consumidor: `core/training.py:948`.

El mismo JSON se aplica a todos los catálogos y el guardado une permanentemente las exclusiones anteriores con las nuevas, sin identificar liga, modelo, versión ni motivo. El llamador se encuentra dentro del entrenamiento por liga. Una exclusión detectada para una liga puede afectar al entrenamiento posterior de otra. Se comprobó la acumulación mediante un doble de archivo; no se ha probado que una variable concreta esté dañando un modelo actual.

Cambiar este criterio es un cambio clave de entrenamiento. Antes de hacerlo hay que revisar las importancias por liga y decidir cómo conservar o migrar las exclusiones existentes.

### C02-R02 — Instantáneas al importar y listas mutables

Referencias: `config.py:599`, `config.py:600`, `config.py:635`, `config.py:639`.

Las listas filtradas se calculan una vez. Cambiar el JSON después no las actualiza automáticamente en el proceso actual; una nueva llamada al filtro sí vería el cambio. Reproducido sin modificar el archivo real. La política de refresco durante entrenamientos sucesivos queda por decidir.

Los getters de momentum devuelven las listas globales directamente. Modificar la lista retornada cambia el módulo, pero no reconstruye las listas derivadas creadas anteriormente. No se encontraron consumidores de esos getters; es un riesgo de API, no un fallo de uso demostrado.

### C02-D01 — Desactivar filtrado no desactiva la ablación

Referencia: `config.py:516`.

Con `ENABLE_FEATURE_FILTERING=False`, las exclusiones por flags de ablación siguen aplicándose. Reproducido desactivando `pmci`. El código separa ambas etapas explícitamente: no se clasifica como error hasta decidir si el flag debe desactivar solo la poda o todos los filtros.

### C02-R03 — Rutas según el directorio de trabajo

Referencias: `config.py:269`, `config.py:270`, `config.py:271`.

`./leagues`, `./models` y `./useless_features.json` dependen del directorio de ejecución. Cambiar de directorio puede cambiar tanto los datos/modelos encontrados como las exclusiones aplicadas o el archivo sobrescrito. Conecta con M01-R02; requiere una política común antes de corregirse.

### C02-D02 — No se acredita la calibración de valores por liga

`DEFAULT_STATS` es un bloque general; existen ajustes de contexto específicos WNBA/NBL. Esta revisión explica sus valores y conexiones, pero no certifica que estén calibrados ni que sus unidades y fórmulas sean coherentes con cada consumidor. Eso se comprobará al revisar esos módulos. No se modifican porcentajes, límites ni valores de imputación por intuición.

## Candidatos a obsolescencia, sin autorización de borrado

- `REALITY_CHECK_V2`, `CACHE_CONTROL`, `DATA_QUALITY_CONFIG`: importados sin lectura en los consumidores encontrados; primero decidir si se recupera su conexión.
- `ALERT_TYPES`, `ALERT_THRESHOLDS`, sus variantes WNBA/NBL y `DEAD_FEATURES_AVAILABLE`: no se encontraron referencias consumidoras en la búsqueda Python realizada.
- `QUARTER_SPECIFIC_COLS`: importado por `core/features.py` y `core/live_features.py`, sin lecturas del identificador en esos archivos.
- Getters de momentum: sin consumidores estáticos encontrados. `Path`: sin uso interno ni importación consumidora encontrada.
- Comentario «meta-modelo eliminado completamente»: **contexto histórico aclarado tras leer `unsure`**. Hubo varios diseños de metamodelo; no se considera prueba de una avería ni de que falte restaurar componentes. Su redacción necesita contexto de versión y no autoriza eliminar la integración Huber actual.

No confundir estos candidatos con `CONTEXT_ADJUSTMENT_WNBA` y `CONTEXT_ADJUSTMENT_NBL`: ambos tienen selección explícita en `analysis/live_mode.py`. Tampoco confundir getters no usados con las listas de características, que sí tienen consumidores.

Las búsquedas estáticas no prueban que no haya usos externos o dinámicos. No se ha eliminado nada ni se propone desactivar funcionalidad en esta fase.

## Validación y límites

Se ejecutó `.diagnostics/audit_config.py`: diez escenarios con aserciones, conteos del catálogo y análisis AST de referencias a imports desde `config`. Se recorrieron `main.py` y archivos Python de `core`, `analysis`, `metamodel`, `scripts` y `tests`, sin errores de parseo. Ese recorrido solo cuenta referencias a configuración; no completa la auditoría de los demás módulos.

El código de configuración se evaluó en un espacio aislado con la credencial sustituida por un marcador antes de ejecutarlo. Se leyó el JSON existente, pero todas las pruebas de escritura, errores de acceso y cambios de exclusiones emplearon dobles de memoria. No se modificaron `config.py`, `useless_features.json`, modelos ni datos; no hubo entrenamiento, conexiones de red ni cambios de extensión.

Siguiente módulo sugerido: `analysis/live_system.py`, para revisar la orquestación del menú live, selección de ligas/partidos y ciclo de vida del procesador. La revisión se detiene aquí.
