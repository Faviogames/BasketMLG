> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# 📚 Feature Dictionary - BasketMLG v2

This document details the features used in both the Historical (Pre-Game) and Live (In-Game) models.

---

## 🏛️ 1. Historical Model Features (XGBoost)
**Source**: `core/features.py`
**Target**: `total_score` (Game Total)

### 📈 Momentum & Trends (EMA)
These features track team performance over time using Exponential Moving Averages (EMA).
*   **`home_ema_scoring_efficiency_long_term_11`**: Home team's scoring efficiency (Points/Possession) over last 11 games.
*   **`away_ema_scoring_efficiency_long_term_11`**: Away team's scoring efficiency over last 11 games.
*   **`home_ema_pace_long_term_11`**: Home team's pace (Possessions/48min) over last 11 games.
*   **`away_ema_pace_long_term_11`**: Away team's pace over last 11 games.
*   **`quarter_variance`**: Variance in scoring across quarters (Consistency metric).

### 📊 Market & Odds (New!)
Features derived from bookmaker lines (`match['line_odds']`).
*   **`market_total_diff`**: Difference between teams' recent scoring average (L5) and the bookmaker's total line.
    *   *Formula*: `(Home_Pts_L5 + Away_Pts_L5) - Market_Total`
    *   *Interpretation*: Positive = Teams scoring more than market expects (Potential Over).
*   **`implied_pace_vs_avg`**: Difference between the pace implied by the total line and the teams' actual average pace.
    *   *Formula*: `(Market_Total / 2.1) - Avg_Pace_L5`
*   **`line_volatility_index`**: How "extreme" the line is compared to league average.
    *   *Formula*: `(Market_Total - League_Avg_Total) / League_Std_Dev`

### 🧠 Advanced Metrics (Ablation Controlled)
*   **`pmci` (Psychological Momentum Cascade Index)**: Measures momentum shifts.
*   **`cprc` (Competitive Pressure Response)**: How teams perform in close games.
*   **`tav` (Tactical Adaptation Velocity)**: Speed of adjustment to opponent strategies.

### 🔢 Basic Stats (Rolling L5)
*   `home_points_scored`, `away_points_scored`
*   `home_pace`, `away_pace`
*   `home_offensive_rating`, `away_offensive_rating`

---

## ⚡ 2. Live Metamodel Features (LightGBM)
**Source**: `core/live_features.py` / `metamodel/train_final_model_lightgbm.py`
**Target**: `final_total` (End of Game Score)

### ⏱️ Game State
*   **`current_total`**: Sum of points scored so far.
*   **`time_remaining`**: Seconds left in the game.
*   **`quarter`**: Current quarter (1-4).

### 🏃‍♂️ Live Dynamics
*   **`live_pace`**: Current pace of the game (Possessions/48min) calculated from elapsed time.
*   **`projected_total`**: Simple projection: `current_total + (current_rate * time_remaining)`.
*   **`scoring_surge`**: Recent scoring rate (last 3 mins) vs game average.

### 🤖 Model Integration
*   **`pre_game_prediction`**: The output from the XGBoost model. This is the **anchor** for the live model.
*   **`model_error`**: Difference between `projected_total` and `pre_game_prediction`.

---

## 🛠️ Feature Engineering Rules
1.  **Imputation**: Missing values are imputed using League Averages (`core/data_processing.py`).
2.  **Normalization**: Most features are raw values, but some (like `line_volatility_index`) are Z-scored.
3.  **Safety**: All divisions are protected by `safe_divide()` to prevent DivisionByZero errors.
