> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# 🧠 AI Context & Project Map - BasketMLG v2

> **STOP & READ**: This file is the primary source of truth for understanding the BasketMLG v2 architecture. Read this before making changes to understand the "Hybrid" nature of the system.

## 🎯 Project Objective
**BasketMLG v2** is an advanced NBA prediction engine focused on **Total Score (Over/Under)** markets.
It uses a **Hybrid Architecture** combining:
1.  **Pre-Game Historical Model (XGBoost)**: Predicts the total score based on historical trends, momentum, and matchups.
2.  **In-Game Metamodel (LightGBM)**: Refines the prediction in real-time using live game state (score, pace, time) and the pre-game prediction as a baseline.

## 🏗️ System Architecture

### 1. The Core (Historical / Pre-Game)
*   **Goal**: Generate a baseline prediction before tip-off.
*   **Engine**: XGBoost (`core/training.py`).
*   **Data**: Historical match JSONs in `data/` (e.g., `data/NBA_2024.json`).
*   **Key Logic**:
    *   **ETL**: `core/data_processing.py` loads and cleans raw JSONs.
    *   **Feature Engineering**: `core/features.py` calculates ~150 features (EMA, Momentum, Market).
    *   **Training**: `core/training.py` trains the model and saves it to `models/NBA.joblib`.

### 2. The Metamodel (Live / In-Game)
*   **Goal**: Adjust prediction based on what is actually happening.
*   **Engine**: LightGBM (`metamodel/train_final_model_lightgbm.py`).
*   **Data**: Live snapshots stored in `data/training_dataset_live_v4_stacked.csv`.
*   **Key Logic**:
    *   **Live Features**: `core/live_features.py` calculates real-time metrics (e.g., `current_pace`, `scoring_surge`).
    *   **Integration**: Takes `pre_game_prediction` + `live_features` -> `final_prediction`.

## 📂 Directory Map

| Directory | Purpose | Key Files |
| :--- | :--- | :--- |
| **`core/`** | **Historical Engine** | `training.py` (Main Trainer), `features.py` (Feature Calc), `data_processing.py` (ETL) |
| **`metamodel/`** | **Live Engine** | `train_final_model_lightgbm.py` (Live Trainer), `prediction_tracker.py` (Live Inference) |
| **`analysis/`** | **Tools & Sims** | `live_mode.py` (Live Simulator), `odds_api.py` (Market Data Fetcher) |
| **`data/`** | **Raw Data** | `*.json` (Historical Matches), `*.csv` (Live Training Data) |
| **`models/`** | **Artifacts** | `*.joblib` (XGBoost Models), `*.txt` (LightGBM Models), `*.json` (Metadata) |
| **`scripts/`** | **Utilities** | `analyze_features.py` (Feature Importance), `simulate_live_data.py` (Data Gen) |

## ⚠️ Critical Development Rules (DO NOT IGNORE)

1.  **NO Emojis in Logs**: Windows consoles crash with Unicode characters. Use `[INFO]`, `[WARN]`, `[ERROR]` instead of 🚀, ⚠️, ❌.
2.  **Circular Imports**: `core/features.py` and `core/live_features.py` are prone to circular imports.
    *   **Rule**: If `features.py` needs `live_features`, import it **inside the function**, not at the top level.
3.  **Safe Casting**: Always use `safe_float()` and `safe_int()` from `core/data_processing.py` when dealing with raw JSON data. Never assume data exists or is clean.
4.  **Market Features**: Market data (odds) is found in `match['line_odds']`, NOT `match['raw_match']['odds']`.
5.  **Model Bundles**: `models/*.joblib` files are **Dictionaries** (Bundles), not raw models. They contain `{'model': xgb_model, 'mae': ..., 'features_used': ...}`. Always unpack them.

## 🔄 Common Workflows

### A. Adding a New Historical Feature
1.  Edit `core/features.py`: Add calculation logic.
2.  Edit `config.py`: Add feature name to `FILTERED_FEATURES_TO_USE` (if filtering is active).
3.  Run `python -m core.training`: Retrain XGBoost.
4.  Run `python scripts/analyze_features.py`: Verify importance.

### B. Updating the Live Model
1.  Edit `core/live_features.py`: Add new live metric.
2.  Run `scripts/simulate_live_data.py`: Regenerate `training_dataset_live_v4_stacked.csv`.
3.  Run `metamodel/train_final_model_lightgbm.py`: Retrain LightGBM.
