# Prueba prospectiva al descanso: Golden State–Portland

17 de octubre de2026. Misma sesión de Python de la prueba anterior, transporte extensión reconectado. Primer intento de listado sin `events`; segundo intento correcto:6 eventos recibidos y3 disponibles en la selección. Se seleccionó Golden State Warriors–Portland Trail Blazers al descanso,67–58.

El usuario proporcionó captura del mercado Total Points (incl. overtime), con líneas237.5/238.5/239.5. Se eligió la central238.5, Over1.87 y Under1.83, y se introdujo238.5 en el menú. Las cuotas se documentan, pero no se pasan como entradas al predictor. OddsAPI no se consultó correctamente por ausencia de requests en el entorno de diagnóstico; no se requirió para usar la línea facilitada.

## Referencia registrada antes del resultado

- Captura del tracker de diagnóstico:2026-10-17T01:21:25.085071+00:00 (03:21:25 en Europe/Madrid).
- Q1:35–26. Q2:32–32. Q3/Q4:0. Total al descanso125.
- Reloj normalizado: Q2 terminado,24min jugados,24 restantes,50% completado.
- Estadísticas obtenidas:34 métricas. Imputación124/188 (66%).
- Predicción final devuelta:247.24819119164312 puntos.
- Línea de mercado facilitada:238.5. Diferencia prevista:+8.748191191643116; lado que muestra el sistema:OVER.
- Proyección por ritmo constante:250.0. Huber247.2482, ajuste aproximadamente+13.8 respecto de la base. La etiqueta Base Model del monitor muestra la salida ya combinada; no confundir247.2482 con la salida XGBoost previa al Huber.
- Resultado final comunicado después por el usuario:128–118, total246. La referencia inicial se conserva sin recalcular.

## Resultado de la referencia al descanso

La predicción247.2481912 supera el total final246 en1.2481912 puntos (error absoluto1.25). La línea238.5 queda7.5 puntos por debajo del resultado: el lado OVER señalado al descanso resulta acertado a la cuota1.87 facilitada. No se ejecutó ninguna apuesta. El marcador final procede del usuario; no se verificó independientemente ni se confirmó aquí si hubo prórroga.

La referencia simple de duplicar125 puntos del descanso daba250, error absoluto4. En este encuentro la salida combinada se acercó más al resultado. Es una prueba prospectiva individual al descanso de pretemporada: favorable para continuar evaluando ese momento, sin demostrar un MAE general, calibración de probabilidades o rentabilidad sostenida. No mezclarla con la reconstrucción retrospectiva de Chicago como si ambas fueran capturas prospectivas.

Se mantienen modelos y código funcional. Persistencia redirigida sólo a `.diagnostics/live_test_predictions.json`; esta ejecución no publica recomendaciones ni exporta al histórico normal. Tras registrar la primera predicción se dejó el proceso consultando. Más tarde se intentó cerrar el seguimiento al despedirse el usuario, pero la herramienta perdió su canal de salida; no quedó confirmado el cierre. Cualquier salida posterior debe distinguirse de la referencia al descanso.

La consola mostró85.6% Over y un supuesto edge35.6% para238.5, pero no se ha acreditado calibración de esas probabilidades. El JSON del tracker conserva otra parrilla de líneas alrededor de247, y confidence0.75 es el valor fijo del monitor. Documentar estas diferencias, sin convertir sus etiquetas en probabilidad validada ni ventaja de mercado demostrada.

El mercado de la captura incluye prórroga; el sistema no modela OT. Si hubiese prórroga, separar resultado reglamentario y total del mercado para interpretar el error y el resultado de la línea.
