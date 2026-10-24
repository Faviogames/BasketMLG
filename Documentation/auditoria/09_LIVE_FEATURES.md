# Módulo 09: core/live_features.py

Fecha: 16 de octubre de 2026. Revisión local terminada, sin cambios funcionales. Se leyó el archivo completo, se verificaron 23 escenarios aislados y se ejecutaron las nueve funciones de prueba existentes del detector adaptativo mediante extracción AST. Sin red, modelos, entrenamiento o modificación de históricos.

## Contexto aportado por el usuario

RF y Optuna existieron durante el desarrollo y se buscaba construir un metamodelo. Esto explica parte de la arquitectura conservada; no autoriza recuperarlos ni retirar la integración actual. Se mantiene pendiente contrastar decisiones y fórmulas con la documentación/chats del equipo.

**Actualización tras leer `unsure`:** el antiguo bug de dividir Q3 parcial por doce minutos en el detector adaptativo queda **cerrado como reparado y corroborado**, no pendiente de mejora. LF09-01 trata de otro cálculo (ritmo principal) y LF09-03 de otro contrato (historial), por lo que no reabren aquel bug. Las observaciones del bloque enhanced y métricas auxiliares LF09-10/LF09-12 quedan como revisión de sus rutas específicas, **sin impacto grave acreditado en la predicción principal**. Dos versiones de Pacer son una cuestión de trazabilidad pendiente, no prueba suficiente de que el metamodelo actual esté mal entrenado.

## Responsabilidad y rutas comprobadas

| Grupo | Uso encontrado |
| --- | --- |
| `calculate_live_pace_metrics` | Ingeniería histórica en `core/features.py` e inferencia en `analysis/live_mode.py` |
| `calculate_real_balance_features` / `calculate_garbage_time_risk` | Construcción histórica; wrapper `calculate_game_balance_features` usado en live |
| `calculate_live_interaction_features` | Predictor live actual y pruebas; muchas salidas no pertenecen a la lista histórica del modelo base |
| `calculate_pacer_features` / `calculate_runner_flags` | Integración activa del metamodelo; consumo exacto a revisar en el siguiente módulo |
| `calculate_enhanced_live_features` y cinco helpers | Pruebas y referencias anteriores; no se encontró llamada desde el predictor live principal actual |
| `calculate_advanced_live_features` / SVI / RWT / métricas detalladas | Script `scripts/process_historical_features.py`; no se encontró llamada activa del wrapper desde el predictor actual |
| `calculate_pace_poisson` | Script `scripts/generate_meta_features.py`; implementación distinta del Pacer usado por la integración live |
| `calculate_ortg_by_quarter`, `calculate_pace_by_quarter`, `calculate_quarter_momentum` | Helpers auxiliares; llamadas del wrapper comentadas para pace/momentum, ORTG es un placeholder |

El nombre «live» de una función no demuestra que contribuya al resultado actual. Debe comprobarse qué entradas consume el modelo concreto; la integración consultada prepara dos variables base/Pacer, mientras Runner se calcula como información adicional. No se auditó aún el metamodelo completo ni su origen de entrenamiento.

## Hallazgos comprobados en cálculos activos o compartidos

### LF09-01 — El ritmo principal sigue usando etapas fijas pese al reloj recibido

`calculate_live_pace_metrics`, líneas 39–46, asigna 24 minutos al descanso, 30 a `q3_progress` y 36 a cualquier otra etapa. `time_info` se pasa al detector adaptativo, pero no se usa para esta proyección principal. El mismo marcador con 5 o 20 minutos jugados devuelve el mismo ritmo.

Además, construye un box estimado desde puntos totales proyectados y aplica un tope de posesiones 140. En los escenarios de descanso con 100 y 120 puntos, ambos ritmos terminan en 140. Se pierde diferenciación; no se ha demostrado que el tope carezca de una justificación histórica ni se ha cuantificado la frecuencia real de saturación.

No afirmar que el módulo ignora todo el reloj: el detector adaptativo sí lo utiliza. Antes de cambiar la escala, contrastar esta estimación sintética con ritmo histórico F06-02 y con las métricas basadas en box real de otros módulos.

### LF09-02 — Se interpreta Q3 sin jugar como shock al descanso

`calculate_quarter_transition_shock` compara Q3 con Q2 en etapa `halftime`. Con Q2 40 y Q3 aún 0, devuelve shock máximo 1. El productor histórico introduce explícitamente ceros para Q3 en muestras del descanso, por lo que activa ese caso. No hay observaciones de minutos iniciales de Q3 que permitan medir una transición real en ese momento.

La rama `q3_end` contiene `pass`; sin historial válido termina en cero. No confundir esto con una implementación completa de ventanas de tres minutos, que la documentación de la función describe pero el código no observa.

### LF09-03 — El detector adaptativo funciona con reloj, pero recibe otro tipo como historial

Se preserva la corrección anterior: normaliza Q3 con `time_into_quarter` y exige al menos diez puntos antes de detectar cambios. Sus **nueve pruebas existentes pasan**, al ejecutarlas contra el cuerpo real de la función sin importar todo el árbol de dependencias.

Su respaldo histórico espera un DataFrame. `calculate_live_pace_metrics` le pasa `team_trends`, que normalmente es un dict de factores home/away. Si no detectó ya un cambio por el umbral directo, la consulta `.empty` sobre el dict falla y el `except` devuelve cero.

Escenario: cambio de 10% frente a un historial estable de variación cero activa el respaldo con DataFrame y devuelve 1; con dict devuelve 0. No sustituir un tipo por otro sin separar explícitamente tendencias y series históricas.

### LF09-04 — Las etiquetas de etapa no son equivalentes para todos los consumidores

Este módulo usa `q3_end`; otras partes de live manejan `q3_just_ended`. Con el mismo Q3 acelerado, Runner devuelve 1 para la primera etiqueta y 0 para la segunda. El detector adaptativo también sólo admite `q3_progress`/`q3_end`, y el blowout binario exige exactamente `q3_end`.

La etapa exacta depende del recorrido del reloj y del llamador. No se ha afirmado que toda ejecución actual use la etiqueta incompatible; sí que el contrato admite nombres distintos con comportamientos diferentes. Acordar vocabulario y migración antes de unificarlo.

### LF09-05 — El balance ignora Q4 incluso dentro del tramo inicial permitido

`calculate_real_balance_features` suma únicamente Q1–Q3. Un local que gana por diez y pasa a perder por diez tras anotar el visitante veinte en Q4 conserva el mismo balance y estabilidad anterior. `calculate_live_pace_metrics` también limita eficiencia/momentum/consistencia a Q1–Q3, aunque su suma inicial de puntos puede incluir Q4: las bases temporales quedan mezcladas.

Esto se refiere a la consistencia del seguimiento durante el inicio de Q4 que el usuario puede permitir. No propone recomendaciones tardías ni soporte de prórrogas. Los cortes exactos por liga y el comportamiento fuera de alcance siguen pendientes de decisión.

### LF09-06 — El wrapper de balance no entrega claves que piden las interacciones

`calculate_game_balance_features` recibe etapa, métricas de ritmo y nombre de liga, pero devuelve únicamente balance + riesgo; no conserva `quarter_stage`, `quarter_variance` ni `quarter_consistency`. Los parámetros de liga y métricas de ritmo no se utilizan.

`calculate_live_interaction_features` busca la etapa en ese dict y usa por defecto `q3_end`, a la que asigna fracción restante 0.1. En un descanso real con clutch 0.5, produce `clutch_time_interaction = 0.05` en lugar de usar su propia rama de descanso 0.25. Si recibiera explícitamente `q3_end`, la fracción literal seguiría siendo 0.1; su calibración necesita contexto.

`quarter_variance` también se suple con constantes. No añadir claves indiscriminadamente: primero fijar unidades, pues un consumidor usa `1 - variance` sin normalizar y otro divide por 50. Un valor absoluto de varianza no es automáticamente una fracción [0,1].

### LF09-07 — Historial opcional None provoca respaldo completo de interacciones

El argumento `team_history_df` es opcional y varios bloques lo comprueban correctamente, pero al final se llama a `calculate_defensive_stops(None)`. Ese helper consulta `.empty` antes de su `try`, lanzando error y provocando que se reemplacen las trece interacciones por sus valores de contingencia.

Se reprodujo la diferencia en `clutch_time_interaction`: la vía normal sin etapa genera 0.05 y el respaldo por error 0.25. El predictor actual suele aportar un DataFrame; no atribuir este caso a todas sus predicciones.

### LF09-08 — Pacer fija 48 minutos y no respeta la duración suministrada

`calculate_pacer_features` calcula restantes = 48*60 - `played_seconds`, ignorando `total_remaining_minutes` y cualquier duración por liga. Con 100 puntos, 20 minutos jugados y 20 restantes, devuelve 240. Una extrapolación constante al total suministrado de 40 minutos sería 200.

La diferencia legítima NBA/WNBA exige respetar el contrato de liga/reloj, no igualar ambas duraciones. Antes de cambiarlo hay que comprobar el entrenamiento del metamodelo que utiliza esta proyección y la salvaguarda posterior. La llamada actual de `analysis/live_mode.py` **sí incluye `played_seconds`**; no se encontró una ausencia de esa clave en ese punto.

Para el cálculo probabilístico, igualdad entre marcador y línea entra en `needed <= 0` y fija 0.99, aunque el over estricto requiera al menos otro punto. Son convenciones de mercado/probabilidad pendientes de revisar; la prueba registra ramas y argumentos, no valida la distribución SciPy.

## Proxies y funciones auxiliares: alcance e intención a confirmar

### LF09-09 — Runner compara un cuarto parcial con cuartos completos

Con Q1/Q2 de cuarenta puntos y Q3 recién iniciado de cuatro, activa señal fría -1. No recibe reloj para distinguir baja anotación de poco tiempo transcurrido. La constante ±30% puede ser deliberada; primero aclarar si pretende detectar ritmo o acumulación por cuarto. Runner se calcula desde la integración, pero no aparece entre las dos columnas de entrada del metamodelo consultadas; su peso efectivo no se ha cuantificado.

### LF09-10 — El bloque enhanced contiene aproximaciones sin observaciones temporales

Estas observaciones **no se presentan como causas comprobadas de la predicción live actual**, porque no se encontró llamada del predictor principal a `calculate_enhanced_live_features`:

- `_calculate_live_pace_efficiency_interaction` busca `live_efficiency_home` en balance, aunque la genera el dict de métricas live. Cambiar esa eficiencia de 0.3 a 0.8 en su origen no cambia el resultado del helper.
- `_calculate_mobile_clutch_factor` calcula `window_minutes` pero no lo usa; consulta partidos históricos, sin ventana de minutos actuales. Su multiplicador crece con la ventaja: caso neutro 0.5 con empate y 1.0 con ventaja veinte, en sentido opuesto al comentario de presión en partidos cerrados. Confirmar significado antes de invertirlo.
- `_calculate_lead_swing_ft_run` genera una racha de tiros libres sólo desde la ventaja. Con diferencia cinco, activa FT run y `ft_clustering_score = 1.1`, sin recibir tiros ni secuencias. Los comentarios reconocen el proxy pendiente de una serie temporal.
- Varianza/consistencia solicitadas por otros helpers no se entregan desde el wrapper de balance; sus valores por defecto no son observaciones del partido.

### LF09-11 — Dos implementaciones diferentes de «Pacer»

`calculate_pace_poisson`, usado por el script de meta-features, lee `current_pace` pero no lo utiliza. Extrapola puntos a 48 minutos y, antes de minuto 36, mezcla con mercado o constante 220. Ritmos 60 y 140 generan la misma salida con igual marcador/reloj.

`calculate_pacer_features`, usado en la integración actual, extrapola linealmente sin esa mezcla y modela puntos restantes cuando calcula la probabilidad. La función alternativa usa la proyección del total completo como media de Poisson. No intercambiarlas por tener nombres similares. Hay que revisar el dataset y el entrenamiento concretos del metamodelo para determinar si existe diferencia entre la variable de entrenamiento y la de inferencia.

**Actualización módulo 11:** el CSV objetivo del entrenador Huber reproduce la fórmula alternativa de mezcla al descanso en todas sus filas; 1513/1556 difieren del Pacer live. Ver H11-01. Queda confirmada la diferencia entre ese dataset y la inferencia; la procedencia del joblib activo todavía no está acreditada. No atribuirle automáticamente un error medido ni modificar fórmulas durante la auditoría.

### LF09-12 — Métricas por cuarto dependen del formato y la cobertura

- **SVI:** al descanso, dos cuartos iguales y Q3/Q4 explícitos en cero se clasifican `CHAOTIC`; si las claves futuras no existen, esos mismos cuartos se clasifican `VERY_STABLE`. No distingue por sí solo cero medido de cuarto no jugado.
- **RWT:** busca `q1`–`q4` en minúscula. Un box `Q1`/`Q2` con faltas devuelve datos insuficientes; en minúscula detecta la tendencia. El wrapper de métricas detalladas sí normaliza mayúsculas, por lo que no comparten el mismo contrato.
- **eFG detallado:** busca `3_point_field_goals_made`; con nombre canónico `3point_field_goals_made` pierde la contribución de triples. Ejemplo: FGM20 + 3PM8 sobre FGA40 da 0.5 en vez de 0.6 según su propia fórmula.
- **TOV/faltas detalladas:** divide conteos sumados de ambos equipos entre posesiones promedio de ambos, no entre la suma. En el escenario TOV4 / posesiones22 = 0.182 frente a 4/44 = 0.091 con denominador combinado. Puede ser un índice por posesión compartida; definir la unidad pretendida antes de cambiarlo por una tasa convencional.
- `calculate_detailed_quarter_metrics` recibe marcadores pero no los utiliza; devuelve `Q*_Pace` como posesiones del cuarto, sin extrapolar por duración. Su parser de strings no admite fracciones tipo `10/20`; un formato inesperado puede interrumpir el wrapper.
- `calculate_ortg_by_quarter` devuelve siempre 110/110 y `STABLE`: es un placeholder. `calculate_pace_by_quarter` ignora el box y usa puntos/12 con conversión fija. No son cálculos de posesiones observadas por el mero nombre.

El wrapper auxiliar tiene uso en scripts: no eliminarlo sólo porque no aparezca en la inferencia principal. No se auditó aquí la transformación completa de esos scripts ni la cobertura real de sus entradas.

## Semántica y configuraciones pendientes

- `live_efficiency_home/away` son fracciones de puntos del marcador, no TS%, ORTG ni puntos por posesión. Su promedio es 0.5 cuando ambas suman uno. El nombre y la interpretación de eficiencia deben contrastarse con consumidores.
- La estabilidad del líder usa comparaciones booleanas `>` y puede interpretar empates como un bando; `comeback_indicator` no exige una desventaja previa real. No se validó empíricamente su utilidad.
- `calculate_garbage_time_risk` combina proxies con pesos fijos. El argumento `quarter_variance` no se usa; recalcula un CV. El nombre `intensity_drop_factor` guarda consistencia, que vale uno con anotación uniforme, no una caída medida de intensidad. No invertir valores sin revisar quién los usa.
- Hay divisores 12/48 y benchmarks NBA en helpers que no reciben liga. Registrar por función si se pretende normalización estándar o duración efectiva: WNBA de 40 minutos no es una excepción accidental.
- Imports de config se mantienen pero ninguno de sus nombres se consulta en los cuerpos de este módulo. Muchos imports de procesamiento/features tampoco tienen uso. Su retirada es candidata posterior; no modifica por sí sola las reglas literales.
- SciPy se importa al cargar el módulo, por lo que hasta helpers sin probabilidad dependen de su disponibilidad. El ciclo con `core/features.py` se evita allí mediante import local; preservar esa separación al refactorizar.

## Pruebas y límites

`.diagnostics/audit_live_features.py` extrae funciones reales por AST y usa pandas/numpy y cinco helpers reales de `core/features.py`, además de conversiones reales. SciPy no está instalado en el runtime auxiliar: Poisson se sustituye por un doble que registra llamadas. No se han validado valores CDF/SF ni calibración probabilística.

Los 23 escenarios verifican aritmética, formatos, cobertura y contratos. Las nueve funciones de `tests/test_adaptive_split_detector_fix.py` se ejecutaron contra el cuerpo real del detector, sin importar el proyecto completo. Esto no equivale a pasar toda la suite ni a una prueba end-to-end del modelo.

SHA-256 del módulo antes/después: `59af2eb6753d4e3ecfcbd54ccfa477801e1cc54c678a297b9b966ae53edb0e69`. Sin cambios funcionales.

Siguiente sugerido: `analysis/metamodel_integration.py`, para comprobar qué señales y proyecciones consume realmente Chronos/Huber, cómo se carga y qué devuelve ante errores. Parar aquí según el acuerdo.
