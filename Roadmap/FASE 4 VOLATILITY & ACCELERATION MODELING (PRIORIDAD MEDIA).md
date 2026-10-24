> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_ROADMAP.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# FASE 4: VOLATILITY \& ACCELERATION MODELING (PRIORIDAD MEDIA)

Objetivo: Modelar volatilidad y aceleración extrema como sugiere el SHAP
4.1 Acceleration Detection System
score_acceleration_detector: algoritmo para detectar runs de anotación
pace_acceleration_detector: algoritmo para detectar runs de ritmo
combined_acceleration_index: índice que combina ambos
4.2 Volatility Context Features
q4_volatility_outlier: mide si Q4 es atípicamente volátil vs histórico
ft_temporal_volatility: volatilidad de tiros libres por tiempo
score_volatility_decay: cómo cambia la volatilidad conforme avanza el partido
4.3 Extreme Event Modeling
outlier_delta_detection: detecta cambios extremos (>2σ) en cualquier métrica
shock_absorption_capacity: capacidad del equipo para absorber shocks
volatility_adaptation: cómo se adapta el modelo a volatilidad creciente

Brillante la FASE 4, ¡esto es puro +EV y llevar los modelos de sports analytics a un nivel quasi-market maker! Aquí tienes evaluación y recomendaciones para pulir cada punto—esto, de hecho, ataca los “outlier risks” que probablemente los libros y rivales no modelan bien.

***

### **4.1 Acceleration Detection System**

- **score_acceleration_detector:** Algoritmo “rolling window” (por puntos/min o puntos/posesión) que marca el inicio/fin de runs (ej. una ventana de 3 minutos con incremento >1.5-2σ respecto a media).
    - Crucial para flaggear partidos que pueden “romper el script”, por lesiones, expulsiones, runs tras timeout, etc.
- **pace_acceleration_detector:** Idéntico, pero midiendo pace (posesiones/minuto o cambios de pace respecto a baseline histórico).
    - Recomendable: Normalizar por liga y por score/time context para hacerlo robusto a cualquier competición.
- **combined_acceleration_index:** Suma ponderada o producto de ambos → si los dos índices se disparan, alertar máximo riesgo/postura LIVE adaptativa.

***

### **4.2 Volatility Context Features**

- **q4_volatility_outlier:** Compara el std de puntos/min en Q4 contra el histórico de Q4 para esa liga o matchup.
    - Si excede, flag para extremos (ideal para unders en partidos ya resueltos o overs en partidos no resueltos).
- **ft_temporal_volatility:** Rolling std de tiros libres por minuto.
    - Interesante porque a veces un run de FT anticipa garbage o estrategia de fouling masivo.
- **score_volatility_decay:** Calcula cómo cae la varianza de score/pace al avanzar el partido (slope negativo esperable, pero si sube es “caos”).
    - Meta: Buscar slope “contrarian” y alertar como warning para el meta-modelo.

***

### **4.3 Extreme Event Modeling**

- **outlier_delta_detection:** Detecta cualquier variable que cambie >2σ en una ventana temporal—marca evento extremo (run, shock, foul-fest…).
    - Usa esto como trigger para cambiar pesos de predicción o confianza en vivo.
- **shock_absorption_capacity:** Cuantifica la capacidad del equipo para “no perder la compostura” tras un outlier, p.ej., scoring neto tras recibir un run.
    - Analízalo en H2H e histórico general; dale más peso a equipos consistentes bajo presión.
- **volatility_adaptation:** Feature que mide si el modelo ajusta rápido ante volatilidad. Ej: tracking del spread real vs predicho tras un shock, y cuánto tarda el modelo en “adaptarse”.

***

## **Tips de implementación**

- Usa rolling windows para todas las métricas de aceleración y volatilidad (2–5 minutos típicamente, pero testea; el Q4 suele ser más corto e intenso).
- Bucketiza (flag “alta”, “media”, “baja”) para logging y reglas de trading/modelo adaptativo.
- Usa logs para estudiar cómo el edge del modelo cambia tras eventos extremos; si pierdes edge justo después de runs, conviene entrenar el meta-model en partidas con muchos extremos.
- Considera hacer “early-warnings” a tu sistema de advice/picks si combined_acceleration_index + q4_volatility_outlier disparan juntos.

***

**Conclusión:**
La Fase 4 es la llave para:

- Blindarte frente a partidos trampa,
- Hacer tu sistema “reactivo” a extremos,
- Aparecer ante el mercado con edge justo cuando todos los demás están perdidos.

¿Quieres functions prototipo para algún detector (aceleración, shock detection, volatility slope) en Python? ¿O el bloque entero para “features_volatility.py” directamente listo para ti?

