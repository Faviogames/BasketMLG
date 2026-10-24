# BasketMLG v2

Predicción del total final de baloncesto con históricos y datos en vivo. Entrada principal al descanso con Q1/Q2 completos; seguimiento Q3 pendiente de validación. NBA y WNBA conservan duraciones distintas. Proyecto en auditoría, sin precisión general certificada.

## Retomar con Hermes Agent

Desde PowerShell, con Hermes ya instalado:

```powershell
Set-Location -LiteralPath 'C:\Users\machi\Documents\BasketMLG v2'
hermes
```

Desde WSL, si se utiliza el montaje habitual de esta carpeta:

```bash
cd '/mnt/c/Users/machi/Documents/BasketMLG v2'
hermes
```

[AGENTS.md](AGENTS.md) es el contexto del proyecto. Hermes lo descubre desde la carpeta de trabajo, según su [documentación oficial](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files). No se duplica en `.hermes.md`; si existe un override local o contexto de mayor prioridad, comprobar qué archivo carga la instalación.

Esta preparación no instala/configura Hermes ni cambia su memoria o personalidad global. `hermes` no se encontró en el PATH de la sesión de preparación; eso no comprueba otras terminales, perfiles o WSL. Si hace falta instalarlo, consultar la [guía oficial](https://hermes-agent.nousresearch.com/docs/getting-started/installation).

Prompt sugerido:

> Lee AGENTS.md, README.md y Documentation/auditoria/00_ESTADO.md. Resume el estado y las restricciones. Continúa la auditoría de metamodel/prediction_tracker.py, documenta sus hallazgos y detente antes de implementar cambios. No entrenes ni modifiques módulos, modelos o registros.

## Documentación de entrada

| Necesidad | Documento |
| --- | --- |
| Acuerdos para agentes | [AGENTS.md](AGENTS.md) |
| Prioridades y auditoría por módulos | [Estado central](Documentation/auditoria/00_ESTADO.md) |
| Índice de módulos, pruebas e históricos | [Documentación](Documentation/DOCUMENTATION_INDEX.md) |
| Conectar el transporte live | [Extensión Chrome/Edge](browser_extension/README.md) |
| Entorno e inferencia comprobados | [Modelos guardados](Documentation/auditoria/PRUEBA_MODELOS_GUARDADOS.md) |

## Uso

Desde la raíz, con el Python del producto preparado:

```powershell
python main.py --live
```

Selecciona NBA/WNBA: aparece directamente el código de conexión de la extensión y se inicia la búsqueda, sin menú de modos. Conecta el código temporal en la extensión, seleccionando una pestaña de SofaScore del navegador habitual; mantén panel y pestaña abiertos. Elige el partido del listado. Puente local127.0.0.1:8766, sin exponerlo a la red. Cada nueva inicialización requiere código nuevo. Una sesión rechazada por SofaScore puede seguir fallando con extensión.

Durante el seguimiento, con el foco en la terminal, pulsa **V** para volver a consultar el listado y seleccionar otro partido de la misma liga, o **Q** para salir; no necesitas Enter. Volver conserva la conexión y el código de la extensión dentro de esa ejecución. Cada partido inicia su propio estado de seguimiento y consulta/pide su línea opcional. Los atajos requieren una terminal interactiva y se atienden durante las esperas; un cálculo síncrono en curso debe terminar primero. Tras actualizar esta versión, recarga la extensión en Chrome/Edge y vuelve a abrir su panel una vez para cargar el tratamiento de respuestas canceladas.

```powershell
python main.py --models
```

Muestra los modelos guardados y termina, sin iniciar el puente ni pedir conexión al navegador. La predicción manual y la selección de partidos por ID se retiraron el20/10/2026. Si no hay partidos o falla la búsqueda, se informa y se cierra el cliente; se puede volver a ejecutar `--live`. La entrada opcional de línea de apuestas se conserva.

Sin argumentos se muestra la ayuda. `--live`, `--models` y `--train` son excluyentes. `python main.py --train` conserva el entrenamiento, que modifica artefactos; no usarlo para preparar el agente o probar imports.

## Python y compatibilidad

Combinación de diagnóstico probada: Python3.12.14, NumPy2.3.5, pandas2.3.3, SciPy1.16.3, scikit-learn1.7.2, XGBoost3.1.2, joblib1.5.2 y prompt_toolkit. Requests se necesita para OddsAPI. Es una referencia de compatibilidad, no un lockfile completo ni garantía multiplataforma.

`requirements.txt` tiene mínimos amplios y omite dependencias ML activas. Instalarlo tal cual no reproduce las pruebas. En diagnóstico pandas3.0.1 provocó errores de columnas con XGBoost, y otra versión de SciPy bloqueó imports. No actualizar paquetes ni convertir modelos para silenciar advertencias sin revisar consecuencias.

`.diagnostics/model-runtime*` y sus ejecutores dependen del entorno Windows de las pruebas. No reutilizar esos binarios como entorno Linux/WSL. Hermes puede usar otro Python: mantener las dependencias de BasketMLG fuera del entorno interno de Hermes.

## Mapa funcional

| Ruta | Responsabilidad |
| --- | --- |
| `main.py`, `analysis/live_system.py` | CLI y selección |
| `core/live_api.py`, `core/browser_bridge.py`, `browser_extension/` | Eventos/estadísticas mediante navegador del usuario |
| `core/game_clock.py` | Periodo y tiempo por liga |
| `analysis/live_mode.py`, `core/live_features.py` | Ensamblaje y predicción live |
| `core/features.py`, `core/data_processing.py`, `core/training.py` | Históricos y entrenamiento |
| `analysis/metamodel_integration.py` | Huber guardado, base y Pacer |
| `metamodel/prediction_tracker.py`, `prediction_records/` | Persistencia; auditoría completa pendiente |
| `models/`, `metamodel/models/` | Artefactos que se preservan |

## Prioridades y documentación histórica

Validar frescura del reloj durante juego, reacción a puntos de Q4, seguimiento Q3 y trazabilidad entre base/meta/salvaguarda/estabilidad. Analizar imputación66–68% por familias, sin inventar datos futuros. Q4 tardío y OT quedan fuera de recomendaciones. Los casos favorables al descanso no certifican probabilidades ni rentabilidad.

Los documentos anteriores conservan razones de diseño; tienen avisos de vigencia y contrastes en el índice. Sus snippets, planes y porcentajes no son instrucciones actuales ni pruebas de implementación. El estado central distingue hechos, hipótesis y decisiones del usuario.
