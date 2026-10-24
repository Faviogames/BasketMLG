> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# 👁️ Project Vision: The "Anchor & Navigator" Architecture

> **Philosophy**: BasketMLG v2 is not just a prediction model; it is a **self-correcting system**. It acknowledges that pre-game analysis (History) and in-game reality (Live) are two different sources of truth that must be reconciled.

---

## 🏛️ The Core Concept

We treat the prediction problem as a dialogue between two agents:

### 1. The Anchor ⚓ (Pre-Game / XGBoost)
*   **Role**: The Strategist.
*   **Perspective**: "Based on the last 11 games, the market odds, and team fatigue, this game *should* end with **225 points**."
*   **Strength**: Deep context, historical trends, market wisdom.
*   **Weakness**: Blind to the actual game flow once it starts. cannot see if a star player is injured in Q1 or if the refs are calling everything.

### 2. The Navigator 🧭 (In-Game / LightGBM)
*   **Role**: The Tactician.
*   **Perspective**: "I see the Anchor predicted 225, but they just scored 70 in Q1 and the pace is 105 possessions. I am correcting the course to **238 points**."
*   **Strength**: Real-time adaptability, immediate feedback loop.
*   **Weakness**: Can overreact to short-term noise (e.g., a hot streak) without the Anchor's grounding.

### 🤝 The Synergy
The magic happens in the **Integration**. The Navigator doesn't discard the Anchor; it uses it as a baseline.
*   **Equation**: `Final_Pred = Anchor_Pred + f(Live_Deviation, Time_Remaining)`
*   At `Time=0` (Start), the prediction is 100% Anchor.
*   At `Time=48` (End), the prediction is 100% Reality.
*   In between, the Navigator gradually takes the wheel.

---

## 🚀 Potential Enhancements (The Future)

To maximize this synergy, we can implement the following improvements:

### 1. 🧠 Context-Aware Navigator (The "Why")
Currently, the Navigator sees *that* the score is high, but not *why*.
*   **Idea**: Feed specific pre-game context to the Live Model.
*   **Feature**: `is_high_variance_team` (from XGBoost).
*   **Effect**: If a team is known to be volatile (high variance), the Navigator should be *more aggressive* in adjusting the prediction. If they are stable, it should be more conservative.

### 2. 📉 Dynamic Confidence Intervals
Instead of just a point prediction, the system should output a "Confidence Zone".
*   **Idea**: The Anchor provides the initial width of the zone. As the game progresses, the Navigator narrows it.
*   **Benefit**: Helps identifying "Live Value" bets. If the live line drifts outside our confidence zone, it's a bet.

### 3. 🎭 Regime Detection (The "Blowout" Switch)
Games behave differently in "Garbage Time" (blowouts).
*   **Idea**: A dedicated classifier to detect `is_garbage_time`.
*   **Effect**: If `garbage_time = True`, the Navigator switches to a different logic (e.g., expecting lower scoring efficiency due to bench players).

### 4. 📡 Market Feedback Loop
The market (odds) also adjusts in real-time.
*   **Idea**: Feed the *Live Market Line* into the Navigator.
*   **Feature**: `our_pred_vs_live_market`.
*   **Effect**: If our model and the market *both* move in the same direction, confidence increases. If we diverge, it's a signal to re-evaluate or find an edge.

---

## 📝 Summary
BasketMLG v2 is a **Hybrid Correction Engine**. Its success depends not just on how good the XGBoost or LightGBM models are individually, but on **how well they talk to each other**. The improvements above focus on enriching that conversation.
