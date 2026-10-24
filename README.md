# BasketMLG v2

### Predicción del total final de baloncesto con históricos y datos en vivo

BasketMLG es una aplicación de terminal para analizar partidos NBA y WNBA y estimar cuántos puntos sumarán ambos equipos al finalizar el encuentro. Combina modelos XGBoost guardados con una segunda capa Huber que incorpora una proyección del ritmo observado durante el partido.

Su momento principal de uso es **el descanso, con Q1 y Q2 completos**, con seguimiento durante Q3. Los datos llegan desde SofaScore mediante una extensión local de Chrome o Edge conectada a Python.

**Estado:** proyecto en desarrollo y auditoría. El flujo live está operativo; la calibración de probabilidades y la fiabilidad del seguimiento durante Q3 siguen pendientes de validación.

## Qué permite hacer

- Seleccionar NBA o WNBA y elegir un encuentro del listado disponible.
- Consultar marcador, parciales, reloj y estadísticas durante el seguimiento.
- Obtener una estimación del total final con modelos guardados, sin entrenar para cada partido.
- Comparar la proyección con una línea de totales obtenida de OddsAPI o introducida por el usuario.
- Registrar las predicciones para su revisión posterior.
- Volver al listado y cambiar de partido conservando la conexión de la extensión.

## Preparación

### 1. Descargar el proyecto

```powershell
git clone https://github.com/Faviogames/BasketMLG.git
cd BasketMLG
```

### 2. Preparar Python

La inferencia con los artefactos guardados se comprobó en **Python 3.12.14**, con NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2, XGBoost 3.1.2 y joblib 1.5.2. El flujo de terminal también necesita `prompt_toolkit`; la consulta de cuotas utiliza `requests`.

Ejemplo de entorno independiente en Windows, con Python 3.12 instalado:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy==2.3.5 pandas==2.3.3 scipy==1.16.3 scikit-learn==1.7.2 xgboost==3.1.2 joblib==1.5.2 prompt_toolkit requests
```

Esta combinación es una referencia de compatibilidad para inferencia, no un entorno completo validado desde una instalación limpia en todas las plataformas. `requirements.txt` contiene dependencias históricas y no reproduce por sí solo el entorno comprobado. Los detalles y límites están en la [verificación de modelos guardados](Documentation/auditoria/PRUEBA_MODELOS_GUARDADOS.md).

Los bundles de `models/` y el Huber de `metamodel/models/` se utilizan tal como están guardados. Los históricos completos de las ligas no se distribuyen en este repositorio; el entrenamiento requiere disponer de esos datos localmente.

### 3. Instalar la extensión

1. Abre `chrome://extensions` o `edge://extensions`.
2. Activa **Modo de desarrollador** y pulsa **Cargar descomprimida**.
3. Selecciona la carpeta `browser_extension/` del proyecto clonado.
4. Abre una pestaña dedicada a [SofaScore baloncesto](https://www.sofascore.com/basketball) y comprueba que carga.
5. Abre el panel de **BasketMLG · SofaScore local** desde su icono.

La extensión comunica esa pestaña con el puente local `127.0.0.1:8766`. Mantén abiertos el panel y la pestaña durante el seguimiento. Consulta la [guía de la extensión](browser_extension/README.md) para permisos y resolución de problemas.

## Uso

```powershell
python main.py --live
```

1. Selecciona la liga en la terminal.
2. Copia el código temporal que muestra Python en el panel de la extensión.
3. En el panel, selecciona la pestaña de SofaScore y pulsa **Conectar**. El modo predeterminado es **Abrir el enlace y leer el JSON de la página**.
4. Elige el partido en el listado de la terminal.
5. Revisa la línea de totales obtenida automáticamente o introduce una si la aplicación la solicita.
6. Sigue las actualizaciones del marcador, reloj y predicción.

La consulta automática de cuotas necesita una credencial válida de OddsAPI en la configuración local. SofaScore proporciona los datos del partido; OddsAPI proporciona la referencia de mercado.

Durante el seguimiento, con el foco en una terminal interactiva:

| Tecla | Acción |
| --- | --- |
| **V** | Volver al listado y elegir otro partido de la misma liga, con la misma conexión |
| **Q** | Salir y cerrar la conexión |

No hace falta pulsar Enter. Las teclas se atienden durante las esperas; un cálculo en curso debe terminar primero. Cada nueva ejecución de Python genera un código de conexión nuevo. Si la búsqueda falla o no devuelve partidos, vuelve a ejecutar el comando.

### Otros comandos

| Comando | Uso |
| --- | --- |
| `python main.py --models` | Mostrar los modelos disponibles sin conectar el navegador |
| `python main.py --help` | Consultar la ayuda |
| `python main.py --train` | Entrenar con los históricos locales; modifica artefactos |

Sin argumentos se muestra la ayuda. Los modos son excluyentes. La predicción manual y la selección por ID ya no forman parte de la interfaz activa; introducir una línea de totales sigue siendo posible.

## Cómo se obtiene la predicción

```text
SofaScore → Extensión Chrome/Edge → Puente local → Datos del partido
                                                       ↓
Históricos + estadísticas observadas → XGBoost → Huber (base + Pacer)
                                                       ↓
                                Salvaguarda de ritmo → Suavizado del monitor
                                                       ↓
                                         Predicción visible en terminal
```

El Huber recibe dos entradas: la predicción base y la proyección del Pacer. Las demás señales impresas no son todas entradas directas de esa capa. La salvaguarda puede acercar una salida extrema a una referencia de ritmo y el monitor suaviza los cambios entre actualizaciones.

Durante un cuarto detectado como activo, el monitor consulta aproximadamente cada 60 segundos y, después de la primera predicción, exige normalmente al menos cinco puntos de cambio acumulado entre ambos equipos para recalcular. El paso del tiempo por sí solo no activa una nueva predicción.

## Interpretar los resultados

- La predicción expresa el **total final esperado de ambos equipos**, no los puntos restantes ni el ganador.
- Una diferencia positiva respecto a la línea indica que la proyección queda por encima de ella. No demuestra por sí sola una ventaja rentable.
- Los porcentajes Over/Under se calculan con una distribución normal y una desviación guardada. **No están validados como probabilidades calibradas de acierto.**
- En ejecuciones revisadas hubo alrededor de un 66–68 % de características imputadas. Eso describe entradas sustituidas, no una tasa de error, y limita la confianza que puede atribuirse a la salida.
- Las etiquetas de base/metamodelo y los registros todavía requieren revisión: el registro automático precede al suavizado y puede diferir de la predicción visible.

La frescura del reloj y las actualizaciones sucesivas en Q3 siguen pendientes de comprobación específica. NBA y WNBA tienen duraciones distintas; no se da por validada toda la cadena para ambas ligas. El uso previsto excluye prórrogas y recomendaciones en Q4 tardío; cualquier uso excepcional de Q4 se limita a su tramo inicial y su política operativa sigue pendiente.

## Estructura y documentación

| Ruta | Responsabilidad |
| --- | --- |
| `main.py`, `analysis/live_system.py` | Entrada de terminal y selección de partidos |
| `browser_extension/`, `core/browser_bridge.py`, `core/live_api.py` | Transporte de datos y monitor |
| `core/game_clock.py` | Normalización del reloj por liga |
| `analysis/live_mode.py`, `core/live_features.py` | Características y predicción live |
| `analysis/metamodel_integration.py` | Integración Huber con base y Pacer |
| `core/features.py`, `core/data_processing.py`, `core/training.py` | Preparación histórica y entrenamiento |
| `metamodel/prediction_tracker.py`, `prediction_records/` | Persistencia de predicciones |
| `models/`, `metamodel/models/` | Modelos y artefactos guardados |

Consulta el [índice documental](Documentation/DOCUMENTATION_INDEX.md) y el [estado de auditoría](Documentation/auditoria/00_ESTADO.md) para distinguir funcionamiento comprobado, limitaciones y propuestas históricas. Algunos documentos describen versiones anteriores.

Para continuar el desarrollo con un agente, las instrucciones están en [AGENTS.md](AGENTS.md). Las carpetas `tests/` y `unsure/` se conservan únicamente en el entorno local del mantenedor y están excluidas del repositorio publicado.
