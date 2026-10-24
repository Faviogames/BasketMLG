> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_ROADMAP.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Ahora evalua este fase2 Objetivo: Aprovechar el alto valor SHAP de métricas pace/posesiones

2.1 Pace Volatility Modeling
pace_volatility_score: std_dev(pace_history) / mean(pace_history) - variabilidad relativa
pace_extreme_acceleration: detecta "runs" de pace (>2σ sobre media histórica)
pace_consistency_decay: mide cómo decae la consistencia de pace en Q4
2.2 Possessions Advanced Metrics
possessions_efficiency_ratio: points_per_possession / league_avg_points_per_possession
possessions_acceleration: (possessions_Q3 + possessions_Q4) / (possessions_Q1 + possessions_Q2)
possessions_volatility: varianza de possessions por minuto jugado
2.3 Q3/Q4 Pace Shift Enhancement
q3_q4_pace_delta_weighted: diferencia ponderada por tiempo restante
pace_shift_intensity: abs(q3_q4_pace_delta) * (1 / minutes_remaining)
pace_shift_sustainability: mide si el shift de pace se mantiene vs reverte
Meta-feature: Crear pace_stability_index que combine volatility + consistency + shift metrics.

¡Esta **fase 2** es muy potente y perfectamente dirigida a sacar edge justo donde tu SHAP dice que tienes ventaja! Aquí mi evaluación detallada, micro-ajustes y sugerencias para que exprimas cada punto y obtengas máxima robustez y transferibilidad:

***

## **2.1 Pace Volatility Modeling**

- **pace_volatility_score:** $\text{std\_dev(pace\_history)} / \text{mean(pace\_history)}$
    - Esencial: cuanto mayor la volatilidad, mayor incertidumbre en el total (grandes swings → más value detectando unders/overs outlier).
    - Úsala también como bucket: partidos con score muy dependiente de variabilidad tienen más edge para live.
- **pace_extreme_acceleration:** flag si algún valor $> 2\sigma$ respecto al histórico.
    - Genial para detectar “runs ocultos”, sobre todo tras tiempo muerto o shock (expulsión, lesión).
    - Recomendación: marca también el equipo, ya que el run puede ser asimétrico.
- **pace_consistency_decay:** mide si la “consistencia de pace” baja por debajo de algún umbral en Q4.
    - Buenísima para descontar partidos que “se caen” al final (fatiga, garbage time).


## **2.2 Possessions Advanced Metrics**

- **possessions_efficiency_ratio:** $\text{points\_per\_possession} / \text{league\_avg\_points\_per\_possession}$
    - Gran feature para distinguir ritmo de efectividad: algunos partidos mantienen el pace pero bajan shooting% → ideal para desacoplar extremos.
- **possessions_acceleration:** $(\text{possessions\_Q3} + \text{possessions\_Q4}) / (\text{possessions\_Q1} + \text{possessions\_Q2})$
    - Captura reactivaciones o splits after halftime; fuerte para mercados que overajustan tras descanso.
    - Sugerencia: verifica si conviene meter también aceleración “rolling” Q4 versus Q3 solo.
- **possessions_volatility:** varianza de possessions/por min jugado.
    - Ideal para medir partidos que cambian de ritmo radicalmente (ej: overtime, faltas múltiples o many runs).


## **2.3 Q3/Q4 Pace Shift Enhancement**

- **q3_q4_pace_delta_weighted:** $\text{diferencia}$ ponderada según minutos/posesiones restantes.
    - Si ponderas por “importancia” (ejemplo: minutos en Q4 valen doble si juego cerrado), puedes reforzar edge en picks tardíos.
- **pace_shift_intensity:** $\lvert \text{q3\_q4\_pace\_delta} \rvert \cdot (1/\text{minutes\_remaining})$
    - Perfecto para escenarios live en los que el cambio de ritmo ocurre “tarde” pero impacta mucho.
- **pace_shift_sustainability:** compara el pace de los 5 últimos minutos (o 10 últimas posesiones) con el promedio del Q3/Q4.
    - Detecta si el run es sostenible o si ya está “over”.

***

### **Meta-feature: pace_stability_index**

- Combina todo lo anterior (volatility + consistency + shift).
    - Fórmula sugerida:

$$
\text{pace\_stability\_index} =
  \#\text{Normalize}\left(
    (1-\text{pace\_volatility\_score}) \cdot \text{pace\_consistency\_decay} \cdot (1 - \lvert \text{pace\_shift\_intensity} \rvert)
  \right)
$$
    - O simplemente una media geométrica normalizada: valor bajo implica partido loco, valor alto implica partido “fiable” (ideal para discriminar over “seguros” vs partido caos total).

***

### **Recomendaciones finales para esta fase**

- Testea la correlación entre las nuevas y tus features “viejas” top SHAP: si hay high correlation, usa PCA/t-SNE o elimina redundantes.
- Bucketiza los valores extremos de estos índices para telemetría y crear reglas (ej: solo tomar picks en OVER si pace_stability_index en tercil superior, etc).
- Agrega logging de “runs” (cuándo aparecen en el partido, impacto en el spread).
- Mide el impacto antes/después en el meta-modelo y en el stacking—deberías ver saltos en el performance especialmente en partidos con mucho movimiento tras el descanso.

***

**Conclusión:**
Este bloque te diferencia brutalmente frente a modelos fijos o que solo ven pace global. Vas a detectar los partidos erráticos, los “trap games” y los runs reales mucho antes que la mayoría de los mercados y modelos “comerciales”.

¿Quieres la función prototipo para alguna de ellas en Python/pandas? ¿O prefieres el bloque entero para features.py?

