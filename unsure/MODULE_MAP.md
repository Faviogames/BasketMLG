> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# 🗺️ Module Map - BasketMLG v2

This document provides a technical breakdown of the project's modules.

## 📦 Core Modules (`core/`)

### `core/training.py`
*   **Role**: Orchestrator for the Historical Model.
*   **Key Functions**:
    *   `train_models_by_league()`: Main loop. Loads data -> Calc Features -> Trains XGBoost -> Saves Model.
    *   `train_single_model_xgboost()`: Configures and trains the XGBoost regressor.
*   **Dependencies**: `core.features`, `core.data_processing`, `config`.

### `core/features.py`
*   **Role**: Feature Engineering for Historical Data.
*   **Key Functions**:
    *   `calculate_features(league_data)`: Main entry point. Returns DataFrame.
    *   `calculate_momentum_metrics()`: Calculates EMA and trend features.
    *   `calculate_market_features()`: Extracts odds and calculates market diffs.
*   **Dependencies**: `core.live_features` (Local import only!), `config`.

### `core/data_processing.py`
*   **Role**: ETL (Extract, Transform, Load).
*   **Key Functions**:
    *   `load_league_data()`: Reads JSONs from `data/`.
    *   `process_raw_matches()`: Cleans raw JSON data, fixes types.
    *   `safe_float()`, `safe_int()`: Utility functions for safe casting.

### `core/live_features.py`
*   **Role**: Feature Engineering for Live Data.
*   **Key Functions**:
    *   `calculate_live_pace_metrics()`: Estimates current game pace.
    *   `calculate_live_features()`: Generates the feature vector for the Metamodel.

---

## 🚀 Metamodel Modules (`metamodel/`)

### `metamodel/train_final_model_lightgbm.py`
*   **Role**: Trainer for the Live Metamodel.
*   **Key Functions**:
    *   `train_final_metamodel_lightgbm()`: Loads `training_dataset_live_v4_stacked.csv`, trains LightGBM, saves to `metamodel/models/`.

### `metamodel/prediction_tracker.py`
*   **Role**: Inference Engine for Live Mode.
*   **Key Functions**:
    *   `predict_live()`: Takes live game state -> Returns final score prediction.

---

## 🔬 Analysis & Tools (`analysis/`, `scripts/`)

### `scripts/analyze_features.py`
*   **Role**: Feature Importance Analyzer.
*   **Usage**: `python scripts/analyze_features.py --league NBA`
*   **Output**: Prints top features and checks for missing/zero-importance features.

### `scripts/simulate_live_data.py`
*   **Role**: Data Generator.
*   **Usage**: Simulates live games from historical data to create the training dataset for the Metamodel.

### `analysis/live_mode.py`
*   **Role**: Real-time Simulator.
*   **Usage**: Simulates a "live" environment using historical games to test the full pipeline (Historical + Live).

---

## ⚙️ Configuration (`config.py`)
*   **Role**: Global Settings.
*   **Key Variables**:
    *   `FILTERED_FEATURES_TO_USE`: List of active features.
    *   `DATA_FOLDER`: Path to raw data.
    *   `MODELS_FOLDER`: Path to saved models.
