# Reconstrucción del descanso: Chicago–Minnesota

Fecha de prueba:17 de octubre de2026. Evento16869058, NBA Preseason. Solicitud del usuario: simular la predicción al final de la primera mitad después de conocer el resultado.

## Datos y procedimiento

SofaScore conserva los periodos1Q y2Q en `/api/v1/event/16869058/statistics`. Se leyeron sus valores del JSON visible en el navegador integrado. Se sumaron únicamente contadores de ambas partes: tiros anotados e intentados, rebotes, asistencias, pérdidas, robos, tapones y faltas. Los porcentajes se derivan de contadores; no se promedian porcentajes entre cuartos. No se incorporaron estadísticas ALL del partido completo ni Q3/Q4. No se reconstruyeron rachas minuto a minuto ni máximos de ventaja.

Parciales: Q1 Chicago33–Minnesota28; Q2 Chicago34–Minnesota32. Descanso67–60:127 puntos. Estado reconstruido Halftime,24min jugados de48 y Q3/Q4 a cero. El parser real recibió un bloque ALL que representa expresamente el acumulado reconstruido de la primera mitad. Se comprobaron presencia de métricas críticas y concordancia de puntos con2P/3P/FT por equipo.

Se ejecutaron el parser, normalizador de reloj, señales live, caché histórica y predictor compartido reales con los bundles guardados. No se entrenó, no se cambió configuración y no se invocó el tracker. Los hashes de artefactos y el inventario/tamaños/mtime de prediction_records se mantuvieron iguales durante esta ejecución.

## Resultado

| Salida | Total previsto | Diferencia respecto de249 |
| --- | ---: | ---: |
| XGBoost base |234.8015|−14.1985|
| Extrapolación constante del ritmo al descanso |254.0000|+5.0000|
| Huber / predicción final devuelta |250.6082|+1.6082|

Resultado final Chicago133–Minnesota116:249 puntos, comunicado por el usuario y corroborado en la ficha terminada. El resultado final se utiliza después de la inferencia para calcular el error; no se pasa al predictor.128/188 entradas imputadas (68.1%). No se proporcionó línea de mercado ni se evaluó rentabilidad/acierto de apuesta.

## Alcance y límites

Reconstrucción retrospectiva con estadísticas por cuarto disponibles después del partido, que pueden incluir correcciones posteriores: no es una captura prospectiva del descanso. Se usan históricos del bundle actual, cuya fecha máxima no quedó acreditada en esta ejecución; no afirmar ausencia de fuga temporal en el entrenamiento ni considerarla evaluación fuera de muestra validada. Es un único partido de pretemporada y no demuestra un MAE general de1.61 puntos.

La salida base234.8015 coincide con la registrada en la prueba de Q4 previa, aunque sus datos live eran distintos. Dejar constancia para revisar qué entradas afectan realmente a la base; esta coincidencia aislada no acredita por sí sola una causa concreta. El Huber sí cambió al recibir el Pacer calculado con el reloj correcto del descanso.

Evidencias: `.diagnostics/replay_halftime_16869058.py`, `.diagnostics/halftime_16869058.json` (entrada y resultados) y `.diagnostics/halftime_16869058.log` (trazas).
