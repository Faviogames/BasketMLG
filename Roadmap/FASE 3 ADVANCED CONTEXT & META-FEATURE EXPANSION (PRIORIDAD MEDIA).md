> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_ROADMAP.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# y ahora esta FASE 3: ADVANCED CONTEXT \& META-FEATURE EXPANSION (PRIORIDAD MEDIA)

Objetivo: Profundizar en las meta-features que muestran crecimiento en importancia
3.1 Clutch Performance Multi-Horizon
clutch_short_term: rendimiento en últimos 6 minutos
clutch_mid_term: rendimiento Q4 completo
clutch_long_term: rendimiento segunda mitad
clutch_adaptive: horizonte dinámico basado en tiempo restante
3.2 Garbage Time Risk Enhancement
garbage_time_risk_temporal: riesgo que evoluciona con tiempo (no binario)
garbage_time_intensity: magnitud del garbage time basado en lead + tiempo
garbage_time_recovery: capacidad de equipos para "volver" de garbage time
3.3 Game Flow \& Efficiency Integration
game_flow_volatility: variabilidad del flow del partido
scoring_efficiency_trend: tendencia de eficiencia anotadora
defensive_stops_rate: tasa de paradas defensivas por cuarto
Stacking Enhancement: Crear meta-features que combinen outputs de diferentes horizontes de clutch performance.

¡La FASE 3 está muy bien formulada y es justo lo que desmarca a un sistema predictivo “pro” respecto a cualquier modelo puramente estadístico! Aquí tienes una evaluación técnica, sugerencias adicionales y trucos para maximizar el valor de cada bloque:

***

### **3.1 Clutch Performance Multi-Horizon**

- **clutch_short_term:** rendimiento últimos 6 minutos (ideales para ventanas móviles, ej. puntaje neto, stops, eficiencia).
- **clutch_mid_term:** rendimiento Q4 completo (típicamente suma/ratio de scoring, stops, turnovers solo cuartos finales).
- **clutch_long_term:** rendimiento segunda mitad (robustez sobre sample size, ideal para equipos con tendencia a ajustar en 2H).
- **clutch_adaptive:** calcula el clutch sobre la ventana que queda (ej. “lastX” donde X = minutos_restantes o X = posesiones_restantes).
    - **Clave:** Usarlo en modo “live”, para adaptar el horizonte según el contexto de partido real.
- **Stacking enhancement:** meta-feature con combinación no lineal:
    - Ej: max(clutch_short, clutch_mid, clutch_long), avg, o incluso un modelo que elija el más relevante según contexto (partido cerrado/abierto).

***

### **3.2 Garbage Time Risk Enhancement**

- **garbage_time_risk_temporal:** mide dinámica, usando lead, tiempo y variabilidad reciente.
    - Sugerido: modelo logístico o scoring escalado, tipo $\text{logit}(lead / minutos\_restantes)$.
- **garbage_time_intensity:** magnitud real, ej: si lead >15 y faltan <8min, intensidad = (lead-15)/(tiempo*escala_league).
    - Opcional: pondera con el scoring pace live. Cuando ambos se disparan → high risk.
- **garbage_time_recovery:** cuántas veces cada equipo logró remontar diferencias grandes en última mitad (histórico H2H y rolling).
    - **Meta:** taguear partidos de alta volatilidad como “no confiables” para overs/unders conservadores.

***

### **3.3 Game Flow \& Efficiency Integration**

- **game_flow_volatility:** std o coeficiente de variabilidad del “game flow indicator” en bloques móviles.
- **scoring_efficiency_trend:** calcula rolling o slope de eficiencia en cada cuarto (o por intervalos cortos en Q4).
- **defensive_stops_rate:** stops/Q o stops/min por fase; ideal para equipos que suben la intensidad defensiva tarde.
- **Meta:** crea features del tipo “flow reverter”, es decir, si la tendencia de flow cambia radicalmente en el último Q.

***

### **Stacking Enhancement: multi-horizonte**

- Usa como input al meta-modelo las distintas mediciones de clutch y garbage time combinadas:
    - Por ejemplo, crea features tipo “clutch differential”: clutch_short_term - clutch_long_term.
    - O features compuestas: “clutch_during_garbage_time” que mida si el equipo produce mucho justo en los minutos donde ya hay riesgo de garbage time.

***

## **Tips extra de implementación**

- Todas estas features se benefician de estar normalizadas y bucketizadas, ya que su interpretación depende del contexto (NBA no es igual que NBL).
- Añade “early warning flags” en live cuando garbage_time_intensity + clutch_short_term se combinan (ideal para O/U en partidos resueltos).
- Para reporting/análisis: guarda en tus logs ejemplos de partidos con clutch_adaptive extremo para estudio de edge y poder ajustar reglas auto.
- Puedes usar PCA solo de las meta-features para crear un “meta-context index” y tener un solo score resumen de contexto extremo.

***

**Conclusión:**
Esta fase va a darte la capacidad de:

- Detectar partidos trampa,
- Ajustar edges según contexto,
- Mejorar la robustez de las picks especialmente en extremos (blowouts/partidos igualados con finales explosivos).

Si quieres, te armo un ejemplo Python para calcular alguna de estas meta-features o un plan de validación experimental para medir cuál añade más edge en train/live. ¿Algún bloque concreto del que quieras el código listo para integrar?

