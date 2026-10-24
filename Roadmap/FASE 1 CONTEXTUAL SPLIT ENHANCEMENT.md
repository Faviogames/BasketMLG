> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_ROADMAP.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

FASE 1: CONTEXTUAL SPLIT ENHANCEMENT (PRIORIDAD ALTA)   


1.1 Advanced Quarter Transitions
quarter_acceleration_q2_q3: (q3total - q2total) / q2total - mide aceleración entre mitades
quarter_momentum_shift: sign(q3total - q2total) * abs(q3total - q2total) / halftime_total
split_efficiency_ratio: q3endtotal / halftime_total - ratio de eficiencia entre mitades
1.2 Contextual Volatility Metrics
quarter_variance_weighted: varianza ponderada por importancia de cuarto (Q4 > Q3 > Q2 > Q1)
intensity_drop_acceleration: intensity_drop_factor * quarter_variance - combina ambos insights SHAP
phase_stability_score: 1 - (quarter_variance / halftime_total) - estabilidad relativa
1.3 Dynamic Split Boundaries
adaptive_split_detector: algoritmo que detecta cambios de momentum reales vs splits fijos
quarter_transition_shock: mide "shocks" en transiciones (puntos/minuto cambios bruscos)
contextual_split_weight: pesos dinámicos para splits basados en volatilidad histórica


### **1.1 Advanced Quarter Transitions**

- `quarter_acceleration_q2_q3: (q3total - q2total) / q2total`
    - Perfecta para capturar el “turno” o aceleración al inicio del 2T (muy útil para detectar cambios de ritmo después del descanso).
    - **Extra:** Calcula también como feature “binaria” si supera cierto umbral histórico (ej: top 10% de aceleraciones → flag de potencial run).
- `quarter_momentum_shift: sign(q3total - q2total) * abs(q3total - q2total) / halftime_total`
    - Me encanta como señal de momentum, pero testea también el signo: en algunas ligas, cambios negativos pueden anticipar “over-rebound” si hay reacción fuerte en el Q4.
- `split_efficiency_ratio: q3endtotal / halftime_total`
    - La más “sólida” en test: sugiere si el partido mantiene, baja o sube el nivel ofensivo tras el entretiempo.


### **1.2 Contextual Volatility Metrics**

- `quarter_variance_weighted`: suma ponderada tipo $\alpha Q4 + \beta Q3 + \gamma Q2 + \delta Q1$, donde $\alpha > \beta > \gamma > \delta$.
    - **Tip:** Usa histórico de la liga para definir los pesos óptimos.
- `intensity_drop_acceleration: intensity_drop_factor * quarter_variance`
    - Spot-on: multiplica dos top SHAP features, probablemente muy relevante en Q3/Q4 si “baja la guardia”.
- `phase_stability_score: 1 - (quarter_variance / halftime_total)`
    - Esta está buenísima como normalizador: permite detectar “falsos” partidos estables (bajo variance pero halftime muy bajo) o lo contrario.


### **1.3 Dynamic Split Boundaries**

- `adaptive_split_detector`: clave, aquí puedes experimentar con métodos como _change point detection_ (rupturas tipo PELT, Ruptures) o simple Z-score sobre los puntos/minuto. Dale un umbral autoajustado por liga y testea los “falsos positivos”.
- `quarter_transition_shock`: podrías calcular el máximo cambio de puntos/min dentro de ventanas de 3 minutos al final/inicio de cuarto para encontrar los verdaderos shock-points.
- `contextual_split_weight`: recomiendo que este “peso” afecte los features de transición y que se retroalimente de la volatilidad intra-partido de la liga.

***

### **Extra-Tips para potenciar este bloque**

- Normaliza todas las features nuevas con stats de la liga/jornada ($z$-score contextual), para hacerlos transferibles entre NBA/NBL/WNBA.
- Añade interacción con “run detector” o flags de garbage time (ej: si quarter_acceleration y run están activos en Q3, potencia la predicción de OVER).
- Guarda los valores en train/test para auditar cómo mejoran la feature importance SHAP tras cada batch.
- Hazte un “feature leaderboard” por periodo: en Q1/Q2, predicen mejor los splits; en Q3/Q4, la aceleración y momentum/context.


