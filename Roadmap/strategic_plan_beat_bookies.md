> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_ROADMAP.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Plan Estratégico: Sistema Dual XGBoost + Metamodel Ajustador

## 🎯 Visión

Combinar datos históricos (XGBoost) con análisis live (Metamodel) para superar a las casas de apuestas.

```
XGBoost (históricos) → Predicción base
         ↓
Metamodel (live context) → Ajusta la base
         ↓
    Predicción final mejorada
```

---

## ❌ Problema Actual

El Metamodel NO ajusta la base, genera predicción independiente. Necesita recibir `base_prediction` como feature.

---

## ✅ Plan de Implementación: 3 Niveles

### **Nivel 1: Quick Win (1 Semana)**

**Objetivo:** Metamodel que ajusta (no reemplace) predicción base

1. **Modificar training data** (`scripts/simulate_live_data.py`):
   - Agregar column `base_model_prediction`
   - Calcular predicción XGBoost para cada partido histórico en halftime
   
2. **Re-entrenar Metamodel** (`metamodel/train_model_catboost.py`):
   - Features: 11 actuales + `base_model_prediction` = 12 total
   - El modelo aprende a ajustar según contexto live

3. **Actualizar predicción live** (`metamodel/meta_model_ai.py`):
   - Incluir `base_prediction` en features
   - Metamodel genera ajuste informado

**Impacto:** +2-3% accuracy

---

### **Nivel 2: Edge Serio (2 Semanas)**

**Objetivo:** Incorporar información de mercado y features avanzadas

1. **Bookie line como feature**:
   - Agregar `bookie_line` al training data
   - Feature derivada: `model_vs_bookie_diff = base - bookie`
   - Metamodel aprende cuándo la casa se equivoca

2. **Features avanzadas**:
   ```python
   - q2_surge: momentum reciente
   - foul_trouble: penalty situation
   - fg_variance: shooting consistency
   - lead_changes: game intensity
   ```

3. **Sistema de tracking**:
   - Predicción vs resultado real
   - Edge vs línea de cierre
   - ROI por liga/timing

**Impacto:** +5-7% accuracy, identificar value bets

---

### **Nivel 3: Sistema Profesional (1-2 Meses)**

**Objetivo:** Sistema de betting profesional

1. **Ensemble de metamodels**:
   - Conservador (bajo riesgo)
   - Agresivo (high variance)
   - Situacional (close games, blowouts)

2. **Bankroll management**:
   - Kelly Criterion
   - Position sizing dinámico

3. **Sistema de alertas**:
   - Solo notificar edge > 5%
   - Filtrar por confidence > 75%

4. **Dashboard analytics**:
   - ROI por liga
   - Win rate por timing (Q2, Q3, Q4)
   - Edge vs cierre

**Impacto:** ROI consistente 5-10%

---

## 💎 Edge Sobre Bookmakers

### 1. Timing Advantage
- Sistema actualiza cada 60s
- Casas ajustan con delay
- **Edge:** Detectar momentum ANTES

### 2. Live Context
Casas usan pre-game models, no ven:
- Foul trouble real-time
- Hot/cold shooting actual
- Fatigue de back-to-back

### 3. Triangulación
```
XGBoost + Metamodel + Bookie Line = 3 perspectivas
```
Cuando coinciden → HIGH confidence
Cuando divergen → Potential value

### 4. Selective Betting
NO apostar todos los juegos. Solo cuando:
- Edge > 5%
- Confidence > 75%
- Data quality > 80%

**Calidad > Cantidad**

---

## 🎯 Expectativas Realistas

| Métrica | Bookmakers | Nuestro Objetivo |
|---------|-----------|------------------|
| Accuracy | 52-53% | 55-57% |
| ROI | Break-even | 5-10% |
| Win rate | ~50% | ~55% (selective) |

**Ejemplo:** $1000 bankroll → $50-100 profit/mes (conservador)

---

## 📋 Roadmap Detallado

### Mes 1: Fundación

**Semana 1-2:**
- [ ] Modificar `simulate_live_data.py` con `base_prediction`
- [ ] Re-entrenar metamodel (12 features)
- [ ] Testing: Validar mejora vs actual

**Semana 3-4:**
- [ ] Limpiar `live_mode.py`: Mover ajustes al metamodel
- [ ] Implementar `bookie_line` como feature
- [ ] Sistema básico de tracking

### Mes 2: Optimización

**Semana 1-2:**
- [ ] Features avanzadas (foul trouble, variance, momentum)
- [ ] Análisis de edge por situación
- [ ] Calibración de confidence scores

**Semana 3-4:**
- [ ] Backtesting riguroso (100+ juegos)
- [ ] Identificar nichos de value
- [ ] Documentar patrones de éxito

### Mes 3: Profesionalización

**Semana 1-2:**
- [ ] Ensemble models
- [ ] Kelly Criterion implementation
- [ ] Sistema de alertas inteligente

**Semana 3-4:**
- [ ] Dashboard analytics
- [ ] Paper trading (sin dinero real)
- [ ] Refinamiento basado en resultados

---

## ⚠️ Consideraciones Importantes

### Gestión de Riesgo
1. Nunca apostar > 5% del bankroll en un juego
2. Diversificar (múltiples ligas, timings)
3. Stop-loss rules

### Data Quality
1. Monitorear imputation levels
2. Rechazar predicciones con confidence < 70%
3. Validar data de APIs regularmente

### Adaptación
1. Re-entrenar metamodel mensualmente
2. Ajustar features según performance
3. Aprender de errores (post-mortem analysis)

---

## 🚀 Próximos Pasos Inmediatos

1. **Analizar ajustes actuales** en `live_mode.py`:
   - Reality check
   - Context adjustments
   - Determinar cuáles mover al metamodel

2. **Limpiar `live_mode.py`**:
   - Eliminar ajustes redundantes
   - Simplificar flujo

3. **Implementar Nivel 1**:
   - Metamodel que ajusta base prediction

---

## 📊 KPIs para Medir Éxito

### Accuracy Metrics
- Overall accuracy: 55%+ target
- ROI: 5%+ mensual
- Sharpe ratio: > 1.0

### Operational Metrics
- Data quality score: > 85%
- Prediction confidence avg: > 75%
- Edge identification rate: 20-30% de juegos

### Business Metrics
- Profitable months: 80%+
- Max drawdown: < 20%
- Bankroll growth: 5-10% mensual

---

**Última actualización:** 2025-11-24
**Status:** En diseño - Pendiente implementación Nivel 1
