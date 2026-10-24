# Extensión local de BasketMLG

Esta versión cambia el transporte del modo live: Python pide datos a una pestaña de SofaScore de tu navegador habitual. La extensión usa esa sesión y devuelve únicamente JSON a Python. No necesita Playwright, no importa cookies y no resuelve verificaciones de SofaScore.

Desde la versión 0.2.0, el modo predeterminado **Abrir el enlace y leer el JSON de la página** navega en la pestaña seleccionada hasta el endpoint solicitado y lee el documento resultante. No hace una petición `fetch` a SofaScore en ese modo. El modo **Consultar con fetch (comparación)** conserva la consulta anterior para diagnóstico. La navegación normal puede ser rechazada igualmente; no garantiza superar el 403.

## Instalar una vez

1. Abre Chrome y entra en `chrome://extensions` (Edge: `edge://extensions`).
2. Activa **Modo de desarrollador**.
3. Pulsa **Cargar descomprimida** y selecciona esta carpeta:
   `C:\Users\machi\Documents\BasketMLG v2\browser_extension`
4. Abre [SofaScore baloncesto](https://www.sofascore.com/basketball) en ese mismo navegador. Comprueba que carga normalmente. Si pide una verificación, complétala tú.
5. Pulsa el icono de la extensión **BasketMLG · SofaScore local** para abrir su panel.

Los permisos se limitan a `www.sofascore.com`, el puente local `127.0.0.1:8766` y ejecutar la consulta en la pestaña seleccionada. No solicita acceso a cookies, historial ni otras webs. El código de conexión queda en memoria del panel, sin almacenamiento persistente.

## Primera prueba, sin modelos

Desde la carpeta del proyecto:

```powershell
python scripts/test_browser_bridge.py --event 16869090
```

Usa preferentemente un ID de partido actual con estadísticas disponibles; el de arriba es solo un ejemplo. Puedes omitir `--event` para comprobar únicamente el listado.

En el panel de la extensión:

1. Copia el código que acaba de mostrar la consola.
2. Pulsa **Actualizar pestañas**, selecciona la pestaña de SofaScore, deja el modo **Abrir el enlace y leer el JSON de la página** y pulsa **Conectar**.
3. Mantén el panel abierto: la consola mostrará los partidos, el ID del evento y los periodos de estadísticas recibidos.

Usa una pestaña dedicada a la conexión: el modo de navegación cambia su dirección al listado, evento o estadísticas solicitados. Si el enlace exacto ya está abierto, se lee el documento existente la primera vez; en consultas posteriores se vuelve a navegar para renovar los datos. No se selecciona ni resuelve ningún CAPTCHA automáticamente.

Esta prueba realiza consultas reales. Un listado de cero partidos indica que el endpoint no devuelve eventos en ese momento; no demuestra por sí solo que existan estadísticas de un evento concreto.

## Uso normal

```powershell
python main.py --live
```

Selecciona la liga directamente, sin menú de modos. Cuando aparezca el código, conéctalo en el panel. El listado y la selección de partidos siguen en la consola; después se consultan el evento y sus estadísticas para alimentar los cálculos existentes. Desde el20/10/2026 se retiró la entrada por ID y el modo manual. Para consultar los modelos usa `python main.py --models`, que no inicia conexión al navegador.

Cada nueva inicialización del cliente muestra un código nuevo. Si termina el seguimiento, no hay partidos o falla la búsqueda, el comando cierra su conexión. Para buscar de nuevo ejecuta `python main.py --live`, pulsa **Desconectar** en el panel y usa el nuevo código. Python no abre ni cierra el navegador. Sí debes mantener abiertos tanto el panel como la pestaña seleccionada. El botón **Abrir SofaScore** abre una pestaña por petición tuya.

Desde el22/10/2026, mientras el seguimiento está activo puedes pulsar **V** en la terminal para actualizar el listado y elegir otro partido de la misma liga conservando conexión y código; **Q** sale. Ambos atajos funcionan sin Enter en una terminal interactiva. Una respuesta tardía del partido anterior se descarta sin desconectar el panel. Para cargar este cambio del panel, recarga la extensión y vuelve a abrir el panel una vez; los cambios posteriores de partido dentro de esa sesión no necesitan otro código. Reiniciar Python sigue creando una sesión nueva.

## Si falla

- **Código rechazado:** pega el código de la ejecución actual, sin espacios.
- **No responde / conexión detenida:** verifica que Python sigue esperando, que no hay otra instancia usando el puerto 8766 y que el panel está conectado. El modo live espera hasta 90 segundos por consulta.
- **HTTP 403 / verificación / respuesta no JSON:** revisa la pestaña seleccionada. Navega normalmente por SofaScore, resuelve tú cualquier comprobación y reintenta desde la consola. La extensión no garantiza acceso si SofaScore rechaza esa sesión o las consultas.
- **403 en modo fetch:** prueba el modo de navegación. Puedes abrir manualmente `https://www.sofascore.com/api/v1/sport/basketball/events/live` en la pestaña seleccionada antes de conectar. Si ves JSON con `events`, la extensión lo leerá; si el navegador muestra acceso denegado, no hay datos válidos que leer.
- **Pestaña cerrada o cambiada a otra web:** desconecta, actualiza la lista y selecciona de nuevo SofaScore.
- **Identificador de pestaña obsoleto (`No tab with id`):** la versión 0.1.1 busca la pestaña actual y se recupera automáticamente si solo hay una de SofaScore. Si hay varias, pide seleccionarla. Tras actualizar la extensión en `chrome://extensions`, cierra los paneles antiguos y abre uno nuevo desde su icono para cargar el código nuevo.
- **Sin estadísticas ALL:** el endpoint no ofrece aún los acumulados necesarios. No se sustituye un único cuarto por los totales del partido.
- **Estadísticas incompletas:** faltan categorías de datos que el modelo necesita. La actualización se omite y se indica qué falta.

La caché existente de 30 segundos se conserva. Los modelos y sus fórmulas no se han reentrenado ni revisado con este cambio. Antes de confiar en predicciones, hay que validar el flujo real completo con un partido en curso.

## Comprobaciones locales

```powershell
python -m unittest discover -s tests -p "test_browser_bridge.py"
python -m unittest discover -s tests -p "test_browser_extension_client.py"
```

Las pruebas locales verifican el transporte y la integración con respuestas simuladas; no acreditan acceso real a SofaScore. El puente escucha solo en IPv4 local, exige el código aleatorio y acepta exclusivamente las tres rutas de datos previstas. No publica una API general de navegación o ejecución de código.
