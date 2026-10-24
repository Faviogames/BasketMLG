> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](auditoria/CONTEXTO_DOCUMENTATION.md) y [estado vigente](auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# API Reference

## Overview
This document provides a comprehensive reference for all APIs and interfaces used in the Basketball O/U Prediction System.

## Core APIs

### Live API (`core/live_api.py`)

#### SofaScoreClient Class

##### Constructor
```python
client = SofaScoreClient(cache_duration=30, rate_limit=2.0)
```

**Parameters:**
- `cache_duration` (int): Cache duration in seconds (default: 30)
- `rate_limit` (float): Minimum seconds between API calls (default: 2.0)

##### Methods

###### `get_live_basketball_games()`
Retrieves list of currently live basketball games.

**Returns:**
- `List[Dict]`: List of live game dictionaries with keys:
  - `id`: Game ID
  - `home_team`: Home team name
  - `away_team`: Away team name
  - `home_score`: Current home score
  - `away_score`: Current away score
  - `quarter`: Current quarter
  - `time_remaining`: Time remaining in quarter

**Raises:**
- `APIError`: When SofaScore API is unavailable
- `RateLimitError`: When rate limit is exceeded

###### `get_game_statistics(game_id)`
Retrieves detailed statistics for a specific game.

**Parameters:**
- `game_id` (str): SofaScore game identifier

**Returns:**
- `Dict`: Game statistics with keys:
  - `home_assists`: Home team assists
  - `away_assists`: Away team assists
  - `home_turnovers`: Home team turnovers
  - `away_turnovers`: Away team turnovers
  - `home_total_rebounds`: Home team total rebounds
  - `away_total_rebounds`: Away team total rebounds
  - `home_personal_fouls`: Home team personal fouls
  - `away_personal_fouls`: Away team personal fouls
  - Additional shooting and possession stats

**Raises:**
- `GameNotFoundError`: When game ID is invalid
- `APIError`: When SofaScore API fails

###### `health_check()`
Performs health check on SofaScore API connectivity.

**Returns:**
- `bool`: True if API is accessible, False otherwise

### Feature Engineering API (`core/features.py`)

#### Feature Generation Functions

##### `generate_features(df, league)`
Generates all features for a dataset.

**Parameters:**
- `df` (pd.DataFrame): Input game data
- `league` (str): League identifier ('NBA', 'WNBA', 'NBL')

**Returns:**
- `pd.DataFrame`: Feature matrix with 120+ columns

##### `compute_live_signals(stats, q_scores, game_info)`
Computes O/U-focused live signals.

**Parameters:**
- `stats` (Dict): Raw SofaScore statistics
- `q_scores` (Dict): Quarter-by-quarter scores
- `game_info` (Dict): Game metadata

**Returns:**
- `Dict`: Normalized signals with keys:
  - `home_ato_ratio`: Assist-to-turnover ratio (0-1)
  - `away_ato_ratio`: Assist-to-turnover ratio (0-1)
  - `diff_ato_ratio`: A/TO differential (-1 to 1)
  - `home_treb_diff`: Total rebound differential (-1 to 1)
  - `home_fti`: Foul trouble index (0-1)
  - `away_fti`: Foul trouble index (0-1)

### Live Mode API (`analysis/live_mode.py`)

#### Prediction Functions

##### `get_live_prediction(home_team, away_team, league, live_stats, q_scores)`
Generates live prediction with signal adjustments.

**Parameters:**
- `home_team` (str): Home team name
- `away_team` (str): Away team name
- `league` (str): League identifier
- `live_stats` (Dict): Current game statistics
- `q_scores` (Dict): Quarter scores

**Returns:**
- `Dict`: Prediction results with keys:
  - `base_prediction`: ML model prediction
  - `final_prediction`: Signal-adjusted prediction
  - `adjustment_factor`: Applied adjustment multiplier
  - `applied_adjustments`: List of triggered adjustments
  - `live_signals`: Computed signal values
  - `confidence_intervals`: Prediction confidence ranges

##### `apply_context_adjustment(prediction, balance_features, live_pace_metrics, quarter_stage, live_signals=None)`
Applies context adjustments to base prediction.

**Parameters:**
- `prediction` (float): Base model prediction
- `balance_features` (Dict): Game balance metrics
- `live_pace_metrics` (Dict): Current pace estimates
- `quarter_stage` (str): Current quarter ('Q1', 'Q2', 'Q3', 'Q4')
- `live_signals` (Dict, optional): Live signal values

**Returns:**
- `Tuple[float, Dict]`: (adjusted_prediction, context_info)

### Alert System API (`analysis/alerts.py`)

#### Alert Generation Functions

##### `generate_live_alerts(signals, stats, game_info)`
Generates alerts based on signal thresholds.

**Parameters:**
- `signals` (Dict): Computed live signals
- `stats` (Dict): Raw game statistics
- `game_info` (Dict): Game metadata

**Returns:**
- `List[str]`: List of alert messages in Spanish

##### `check_signal_thresholds(signals)`
Checks which signals exceed alert thresholds.

**Parameters:**
- `signals` (Dict): Signal values

**Returns:**
- `Dict`: Threshold check results with boolean flags

### Training API (`core/training.py`)

#### Model Training Functions

##### `train_league_model(league)`
Trains complete model pipeline for a league.

**Parameters:**
- `league` (str): League identifier

**Returns:**
- `Dict`: Training results with keys:
  - `model`: Trained XGBoost model
  - `meta_model`: Trained meta-model
  - `feature_names`: Feature column names
  - `performance_metrics`: CV scores and metrics

##### `load_model(league)`
Loads trained model artifacts.

**Parameters:**
- `league` (str): League identifier

**Returns:**
- `Dict`: Model artifacts or None if not found

## Data Structures

### Game Data Format
```python
{
    "id": "game_12345",
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "home_score": 28,
    "away_score": 25,
    "quarter": 2,
    "time_remaining": "8:32",
    "league": "NBA",
    "season": "2024-25"
}
```

### Signal Data Format
```python
{
    "home_ato_ratio": 0.79,      # 0-1 normalized
    "away_ato_ratio": 0.50,      # 0-1 normalized
    "diff_ato_ratio": 0.29,      # -1 to 1 differential
    "home_treb_diff": 0.25,      # -1 to 1 differential
    "home_fti": 0.0,             # 0-1 index
    "away_fti": 0.0,             # 0-1 index
    "run_active": False,         # Boolean flag
    "run_strength": 0.0          # 0-1 strength
}
```

### Prediction Output Format
```python
{
    "base_prediction": 210.5,
    "final_prediction": 219.7,
    "adjustment_factor": 1.0435,
    "adjustment_percentage": 4.35,
    "applied_adjustments": [
        "Control de balón mejor: Más posesiones eficientes"
    ],
    "live_signals": {...},  # Signal values
    "confidence_intervals": {
        "line_-2": 196.3,
        "line_-1": 203.4,
        "line_center": 210.5,
        "line_+1": 217.6,
        "line_+2": 224.7
    }
}
```

## Configuration API (`config.py`)

### Constants

#### League Parameters
```python
LEAGUE_PARAMS = {
    'NBA': {
        'scoring_baseline': 225.0,
        'pace_multiplier': 1.2,
        'fouls_per_minute_baseline': 0.5,
        'ato_baseline': 1.2,
        'ato_range': 1.8
    },
    'WNBA': {
        'scoring_baseline': 168.0,
        'pace_multiplier': 0.8,
        'fouls_per_minute_baseline': 0.45,
        'ato_baseline': 1.1,
        'ato_range': 1.6
    },
    'NBL': {
        'scoring_baseline': 178.0,
        'pace_multiplier': 0.98,
        'fouls_per_minute_baseline': 0.48,
        'ato_baseline': 1.3,
        'ato_range': 2.0
    }
}
```

#### Signal Thresholds
```python
SIGNAL_THRESHOLDS = {
    'ato_ratio': 0.25,      # A/TO differential threshold
    'treb_diff': 0.4,       # Rebound differential threshold
    'fti': 0.7,             # Foul trouble index threshold
    'run_strength': 0.3     # Run detector strength threshold
}
```

#### Adjustment Factors
```python
ADJUSTMENT_FACTORS = {
    'ato_boost': 0.15,      # Max 15% adjustment for A/TO
    'treb_boost': 0.12,     # Max 12% adjustment for rebounds
    'fti_boost': 0.30,      # Max 30% adjustment for foul trouble
    'run_boost': 0.20       # Max 20% adjustment for runs
}
```

## Error Handling

### Custom Exceptions

#### `APIError`
Raised when SofaScore API operations fail.

#### `GameNotFoundError`
Raised when requested game cannot be found.

#### `RateLimitError`
Raised when API rate limits are exceeded.

#### `ModelLoadError`
Raised when model artifacts cannot be loaded.

### Error Recovery

#### Automatic Retry Logic
```python
@retry(max_attempts=3, delay=2, exceptions=(APIError, RateLimitError))
def api_call_with_retry():
    # API operation with automatic retry
    pass
```

#### Fallback Values
- **Signal Computation**: Returns neutral values (0.5) on failure
- **Prediction**: Uses base model prediction without adjustments
- **API Calls**: Returns cached data when available

## Performance Characteristics

### Latency Targets
- **Signal Computation**: <100ms
- **Model Prediction**: <500ms
- **API Calls**: <2 seconds (with caching)
- **Full Prediction Pipeline**: <3 seconds

### Rate Limits
- **SofaScore API**: 30 requests/minute
- **Cache Duration**: 30 seconds for live data
- **Concurrent Games**: Up to 10 simultaneous monitoring sessions

### Memory Usage
- **Model Loading**: ~200MB per league model
- **Live Monitoring**: ~50MB per concurrent game
- **Cache**: ~10MB for API response caching

## Testing API

### Test Functions

#### `test_live_system()`
Runs comprehensive system test.

**Returns:**
- `bool`: True if all tests pass

#### `test_signal_computation()`
Tests signal computation accuracy.

**Returns:**
- `Dict`: Test results with accuracy metrics

#### `test_api_connectivity()`
Tests SofaScore API connectivity.

**Returns:**
- `bool`: True if API is accessible

## Usage Examples

### Basic Prediction
```python
from analysis.live_mode import get_live_prediction

# Get prediction for live game
result = get_live_prediction(
    home_team="Lakers",
    away_team="Celtics",
    league="NBA",
    live_stats=current_stats,
    q_scores=quarter_scores
)

print(f"Final prediction: {result['final_prediction']:.1f} points")
print(f"Adjustment: {result['adjustment_percentage']:+.1f}%")
```

### Signal Analysis
```python
from core.features import compute_live_signals

# Compute signals from stats
signals = compute_live_signals(stats, q_scores, game_info)

# Check if A/TO signal triggers
if abs(signals['diff_ato_ratio']) > 0.25:
    print("A/TO signal triggered - better ball control detected")
```

### Alert Generation
```python
from analysis.alerts import generate_live_alerts

# Generate alerts
alerts = generate_live_alerts(signals, stats, game_info)

for alert in alerts:
    print(f"🚨 {alert}")
```

This API reference provides complete documentation for integrating with and extending the Basketball O/U Prediction System.