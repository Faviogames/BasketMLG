# Módulo 07: core/data_processing.py

Fecha: 16 de octubre de 2026. Revisión local terminada; sin cambios funcionales. Archivo completo leído. Se verificaron 20 escenarios aislados con funciones reales y se inspeccionó el JSON histórico disponible, sin sobrescribirlo. No se cargaron modelos ni se ejecutó entrenamiento.

## Criterio de interpretación

El usuario recuerda que algunas fórmulas y exclusiones se introdujeron deliberadamente y buscará su documentación/chats de desarrollo. Separar **comportamiento demostrado**, **motivo todavía desconocido**, **decisión explícita en el código** y **propuesta futura**. Un hallazgo no autoriza una reparación, retirada o reentrenamiento.

Los comentarios del módulo atribuyen parte de la lógica de imputación a una versión con «MAE 5.2». Esto aporta contexto sobre su conservación deliberada, pero no incluye aquí un experimento reproducible ni explica cada regla. No se verificó ese resultado. `Documentation/FEATURE_IMPUTATION_FIX_SUMMARY.md` trata sobre todo la imputación live y no documenta por sí solo el motivo de todas las reglas históricas.

## Responsabilidad y recorrido

1. `load_league_data` lee archivos `.json` y los agrupa por nombre de liga derivado principalmente del archivo.
2. `process_raw_matches` prepara medias de liga sólo si la lista raw tiene menos de 500 entradas.
3. Valida campos básicos, limpia nombres, convierte fecha/marcadores y conserva una referencia `raw_match`.
4. Extrae parciales y totales de mitad desde `quarter_scores`.
5. Aplana `quarter_stats` por equipo/cuarto, normalizando nombres de estadísticas.
6. Si no existe `match_stats`, lo construye sumando los cuartos y **lo inserta en el objeto de entrada**.
7. Filtra y normaliza las estadísticas de partido; imputa mediante histórico del equipo, media de liga o defaults.
8. Añade alias antiguos de tiros de dos/tres y deriva posesiones/`pace_game`.

Salida: `(processed_matches, league_averages)`. Los registros procesados alimentan `core/features.py`; carga e ingeniería están conectadas con entrenamiento. Hay imports compartidos desde `core/live_features.py`, pero no se atribuye a todos ellos un uso activo sin revisar ese módulo.

No hay selección explícita de NBA/WNBA ni duración por liga en este módulo. El normalizador distingue correctamente `WNBA`; no reproduce aquí el problema de substring NBA/WNBA observado en otros módulos.

## Cobertura real disponible en este checkout

En `leagues/` se encontró **un archivo, `NBA.json`, con 1556 registros**. No había un JSON WNBA en esa carpeta al revisar. Esto es un inventario actual, no evidencia de que se hayan borrado datos: los bundles y metadatos previos pueden pertenecer a otros históricos.

| Comprobación en el archivo raw | Resultado |
| --- | --- |
| Registros que pasan la validación mínima actual | 1556/1556 |
| Con `match_stats` no vacío | 0/1556 |
| Con `quarter_stats` no vacío | 1556/1556 |
| Bloques Q1 / Q2 / Q3 / Q4 | 1556 / 1556 / 1556 / 1555 |
| Boxes equipo/cuarto sin `points_scored` | 1934 |
| Partidos con algún box de ese formato | 242 |

Existen dos formatos de estadísticas por cuarto: uno con nombres antiguos de tiros y sin `points_scored`, otro con nombres canónicos y ese campo. Pasar la validación mínima no equivale a tener un box completo ni parciales coherentes.

## Hallazgos comprobados

### D07-01 — El box por cuarto puede borrar un marcador válido

`extract_quarter_data_for_alerts` escribe correctamente `home_q1_points` desde `quarter_scores`. Después, el aplanamiento de `quarter_stats` vuelve a escribir la misma columna desde `points_scored`, usando cero cuando no existe (aproximadamente líneas 350–380). Los totales de mitad se calcularon antes y no se actualizan.

En una **copia de un partido real**: Q1 local original `35` pasa a `0`, pero `home_first_half_points` conserva `65`. En otra muestra del formato con `points_scored`, Q1 `25` se conserva. Los 242 registros identificados tienen el formato que activa este riesgo; se inspeccionaron todos para cobertura, pero sólo se ejecutó el procesamiento sobre una muestra de cada formato, además de los escenarios sintéticos.

La ausencia de estadísticas de puntos en un box es distinta de un cuarto con cero puntos. La contradicción entre parciales y mitades es observable; su reparación deberá preservar los históricos y contrastarse con cómo se crearon los modelos guardados.

### D07-02 — La imputación depende del orden raw y de ejecuciones anteriores

La síntesis de `match_stats` modifica los objetos originales. `calculate_team_league_averages` sólo consulta `match_stats`, sin reconstruir por sí misma los cuartos. Por ello, al buscar antecedentes, puede ver datos sintetizados únicamente de los partidos que ya pasaron por el bucle, aunque existan otros antecedentes cronológicos todavía no procesados.

Con dos partidos idénticos salvo fecha/FTA, el segundo día con FTA cero recibe **40** al procesar la lista ascendente y **20** si llega descendente. Al repetir el procesamiento sobre el mismo objeto descendente, pasa de **20 a 40**, porque ahora el primer pase ya insertó los totales históricos. El corte por fecha funciona, pero la disponibilidad de estadísticas depende del orden de ejecución.

`raw_match` también apunta al objeto modificado. No implica que el JSON en disco se escriba: el módulo no lo guarda. Antes de cambiarlo, confirmar si algún consumidor espera esa mutación y cómo se reconstruirá un historial independiente del recorrido.

### D07-03 — El respaldo de liga puede incluir información futura

`impute_missing_stats` filtra correctamente los antecedentes de equipo a fechas estrictamente anteriores cuando las fechas son válidas. **Ese filtro no se aplica a `league_averages`**, calculadas previamente con toda la entrada raw si hay menos de 500 filas.

Escenario: primer partido FGA 0, partido del día siguiente FGA 100. El primero recibe **50**, la media global que incluye el futuro. Control positivo: con sólo histórico futuro y sin respaldo de liga, obtiene el default 85, demostrando que el filtro del equipo sí opera.

El archivo NBA actual tiene más de 500 registros: en esa ejecución se desactiva el respaldo global, por lo que **este caso específico no prueba contaminación del bundle NBA existente**. Tampoco se ha comprobado la procedencia de su histórico. Sí queda demostrado para entradas pequeñas que contienen `match_stats`.

### D07-04 — El umbral de 500 cambia la política de datos

La decisión usa `len(league_data_raw)` antes de validar los registros. Con los mismos dos partidos válidos y relleno de entradas inválidas: 499 filas producen FGA imputado 50; 500 producen 85 porque las medias de liga quedan `{}`.

Puede ser una decisión de coste o una protección anterior, pero su motivo no está documentado aquí. No retirar el umbral sin consultar. Conviene registrar que el tamaño raw, incluso de registros descartados, afecta a los valores que se entrenan.

### D07-05 — El filtro de estadísticas rechaza los nombres canónicos

`process_match_stats_new_structure` filtra **antes** de normalizar. `USEFUL_MATCH_STATS` admite `3_point_field_goals_made`/`3_point_field_g_attempted`, pero no sus equivalentes canónicos `3point_field_goals_made`/`3point_field_goals_attempted`; ocurre también con dos puntos.

Con un `match_stats` canónico realista de 12 triples sobre 30 intentos, el procesamiento acaba con triples anotados 8 por default y el alias antiguo de intentos 0. Control positivo: el normalizador aislado sí convierte correctamente el alias antiguo. El problema es el contrato/orden del filtro, no que toda normalización esté rota.

En el JSON NBA actual, `match_stats` se deriva de los cuartos usando deliberadamente alias antiguos; ese camino evita este caso concreto. Revisar todos los productores antes de cambiar filtros y nombres.

### D07-06 — El box con FGA agregado pero sin desglose pierde los tiros

En cada cuarto se define FGA = intentos de dos + intentos de tres y no se usa el FGA agregado ya presente. Un box con `field_goals_attempted: 20` y sin desglose produce FGA 0. Esto también afecta a FGM y luego a totales derivados e imputación. Es un escenario de compatibilidad, no una cuantificación de incidencia en el archivo actual.

### D07-07 — Fecha no reconocida se sustituye silenciosamente por el momento actual

`parse_match_date` admite día.mes.año con/sin hora y una fecha incompleta a la que añade el año actual. Una cadena ISO `2025-01-01` devuelve `datetime.now()` en vez de representar esa fecha o informar error.

Esto puede cambiar orden histórico, descanso y selección de antecedentes entre ejecuciones. Fechas que no sean string pueden lanzar errores distintos. Los ejemplos de fecha inspeccionados en NBA sí usan el formato soportado; no se ha afirmado que todos sus registros estén afectados. La política de inferir año puede ser deliberada y requiere confirmación antes de sustituirla.

### D07-08 — Un equipo ajeno se aplana como visitante

Al aplanar estadísticas finales, el prefijo es `home` si el nombre coincide con el local y `away` para cualquier otro. Un tercer equipo `C` dentro de `match_stats` puede sobrescribir estadísticas de `B`; se reprodujo FGA visitante 199. En los boxes por cuarto, en cambio, los equipos desconocidos sí se omiten.

Nombres limpiados con espacios o alias pueden no coincidir con las claves raw. Antes de unificar identidades, definir el contrato y preservar asociaciones existentes.

### D07-09 — Robustez de carga y conversión incompleta

- Un JSON raíz `{"matches": []}` se extiende como claves y se convierte en `{'NBA': ['matches']}`: el loader no exige una lista de objetos.
- `safe_float('nan')` y `safe_float('inf')` conservan no finitos. `safe_int(inf)` lanza `OverflowError`, fuera de sus excepciones previstas. El nombre «safe» no garantiza finitez ni continuidad en todos los casos.
- La validación acepta un resultado final con sólo Q1/Q2 y completa otros parciales a cero; no verifica suma, finalización, tipos numéricos ni coherencia estadística. Los campos requeridos con valor numérico cero se rechazan por falsedad, mientras cadenas `'0'` son verdaderas. Reglas de admisión y cobertura deben acordarse antes de ampliarlas.
- Algunas excepciones por cuarto se silencian; otras descartan el partido completo. La obtención inicial de medias y el listado de directorio quedan fuera de los `try` por partido/archivo. Entradas mal tipadas pueden romper el procesamiento antes del aislamiento previsto.

## Reglas explícitas y cuestiones que requieren contexto, sin calificarlas automáticamente de errores

- **Cero como dato faltante:** `impute_missing_stats` reemplaza todo cero de `DEFAULT_STATS`. Se reprodujo rebote ofensivo 0 → 10. Puede venir de la codificación histórica de ausencia. Distinguir ese convenio de ceros observados antes de proponer cambios.
- **Exclusión de estadísticas:** robos, tapones y faltas no pasan `filter_useful_match_stats`; `DISCARDED_STATS` documenta esa dirección, aunque no se usa directamente para filtrar. Los boxes por cuarto sí se conservan; al sintetizar totales se generan esos campos y después el filtro final los elimina. Contrastar con el propósito de señales posteriores, sin restaurarlos por intuición.
- **Buckets de liga:** `_canonicalize_league_name` agrupa explícitamente EuroLeague/Eurobasket en `EUROBASKET`, prioriza nombre de archivo sobre `stage` y no deduplica los partidos fusionados. Preservar esta política hasta confirmar qué ligas debe cubrir el producto. WNBA está reconocida explícitamente.
- **Medias disponibles:** `calculate_league_averages` agrega boxstats, no resultados `total_score`/`total_score_std`. El bloque de mercado de `features.py` espera esas claves y usa sus constantes 228/20 si no existen. No confundir medias de box con medias de totales ni asumir que esos respaldos están calibrados por liga.
- **Stats parciales:** si falta un equipo de `match_stats`, no se crea automáticamente un registro imputado para él; si existe el nombre con dict vacío, sí se puede imputar. Q1/Q2 de marcador son obligatorios, pero el resto de cobertura no se informa en metadatos.
- **Prórrogas:** las agregaciones recorren Q1–Q4. No añadir OT: el usuario la excluyó del alcance de recomendaciones. En una fase posterior habrá que decidir admisión de partidos históricos con OT y coherencia de sus totales, no ampliar el producto de forma implícita.

## Evidencia, límites y siguiente paso

`.diagnostics/audit_data_processing.py` ejecuta funciones reales del módulo extraídas por AST con constantes `KEY_MAP`/`DEFAULT_STATS` reales. Los 20 escenarios son offline; el loader se comprueba con IO simulado. La inspección de NBA es de lectura y sólo se procesan copias de dos muestras reales representativas. No se procesaron ni imputaron los 1556 registros completos ni se evaluó el efecto predictivo de los hallazgos.

SHA-256 de `core/data_processing.py` antes/después: `6e4c9abc1bde9fc9a239f5b4f8d7dabd15366ea0df3049ae00e9f6ece19c9680`. Sin cambios en el código funcional, JSON originales o modelos.

Los hallazgos aportan causas posibles de diferencias históricas y de calidad de datos. No sustituyen el cruce pendiente del bundle usado en la ejecución que imputó 124/188 características.

Siguiente sugerido: `core/training.py`, para entender selección de variables, particiones por partido/tiempo, OOF, imputaciones guardadas y procedencia de los modelos. Parar al terminar este módulo, según lo acordado.
