# Over/Under Prediction Workflow

## Complete End-to-End Process

This document outlines the complete workflow for generating Over/Under predictions, from data acquisition to final output.

## Phase 1: Pre-Game Analysis

### 1.1 Historical Data Processing
```python
# Load trained model and historical data
trained_data = load_model('NBA')  # or 'WNBA', 'NBL'

# Contains:
# - trained_data['model']: ML model for predictions
# - trained_data['historical_df']: Historical game data
# - trained_data['features_used']: Feature names for model
# - trained_data['std_dev']: Prediction uncertainty
```

### 1.2 Team Data Preparation
```python
# Extract team histories
home_history = historical_df[
    (historical_df['home_team'] == home_team) |
    (historical_df['away_team'] == home_team)
]

away_history = historical_df[
    (historical_df['home_team'] == away_team) |
    (historical_df['away_team'] == away_team)
]
```

### 1.3 Feature Engineering
**120+ Pre-Game Features Generated:**
- **Momentum Features**: EMA calculations for win rate, plus/minus, scoring efficiency
- **Advanced Stats**: Possessions, pace, offensive/defensive ratings
- **H2H Features**: Head-to-head performance metrics
- **Context Features**: Home advantage, consistency, schedule analysis

## Phase 2: Live Game Processing

### 2.1 Live Data Acquisition
```python
# Initialize SofaScore client
client = SofaScoreClient()

# Get live games
live_games = await client.get_live_basketball_games()

# Find target game
target_game = find_game_by_teams(live_games, home_team, away_team)
```

### 2.2 Quarter-by-Quarter Updates
```python
# Current game state
q_scores = {
    'q1_home': 28, 'q1_away': 25,
    'q2_home': 27, 'q2_away': 30,
    'q3_home': 26, 'q3_away': 22
}

# Calculate live totals
halftime_total = q_scores['q1_home'] + q_scores['q1_away'] + q_scores['q2_home'] + q_scores['q2_away']
current_total = halftime_total + q_scores['q3_home'] + q_scores['q3_away']
```

### 2.3 Live Statistics Fetching
```python
# Get detailed live stats from SofaScore
live_stats = await client.get_game_statistics(game_id)

# Key O/U-relevant stats:
stats = {
    'home_assists': live_stats['home_assists'],
    'away_assists': live_stats['away_assists'],
    'home_turnovers': live_stats['home_turnovers'],
    'away_turnovers': live_stats['away_turnovers'],
    'home_total_rebounds': live_stats['home_total_rebounds'],
    'away_total_rebounds': live_stats['away_total_rebounds'],
    'home_personal_fouls': live_stats['home_personal_fouls'],
    'away_personal_fouls': live_stats['away_personal_fouls'],
    # ... shooting stats, free throws, etc.
}
```

## Phase 3: Signal Processing

### 3.1 O/U Signal Computation
```python
# Compute live signals
live_signals = compute_live_signals(live_stats, q_scores, game_info)

# Generated signals:
signals = {
    'home_ato_ratio': 0.79,      # Assist-to-turnover ratio
    'away_ato_ratio': 0.50,
    'diff_ato_ratio': 0.29,
    'home_treb_diff': 0.25,      # Total rebound differential
    'home_fti': 0.0,             # Foul trouble index
    'away_fti': 0.0,
    'home_ts_live': 0.55,        # True shooting %
    'away_ts_live': 0.52,
    'run_active': False,         # Run detector
    'run_strength': 0.0
}
```

### 3.2 Signal Threshold Analysis
```python
# Check which signals exceed thresholds
signal_analysis = {
    'ato_triggered': abs(signals['diff_ato_ratio']) > 0.25,  # True (0.29 > 0.25)
    'treb_triggered': abs(signals['home_treb_diff']) > 0.4,  # False (0.25 < 0.4)
    'fti_triggered': max(signals['home_fti'], signals['away_fti']) > 0.7,  # False
    'run_triggered': signals['run_active'] and signals['run_strength'] > 0.3,  # False
}
```

## Phase 4: Model Prediction

### 4.1 Feature Assembly
```python
# Combine pre-game + live features
X_features = pd.DataFrame([{
    # 120+ pre-game features
    'home_avg_points_scored_last_5': home_stats_5['avg_points_scored'],
    'away_avg_points_scored_last_5': away_stats_5['avg_points_scored'],
    # ... other pre-game features

    # Live features
    'q1_total': q_scores['q1_home'] + q_scores['q1_away'],
    'q2_total': q_scores['q2_home'] + q_scores['q2_away'],
    'q3_total': q_scores['q3_home'] + q_scores['q3_away'],
    'halftime_total': halftime_total,
    'live_pace_estimate': live_pace_metrics['live_pace_estimate'],
    # ... other live features
}], columns=features_used_in_model)
```

### 4.2 Base Model Prediction
```python
# Generate base prediction
base_prediction = model.predict(X_features)[0]  # e.g., 210.5 points

# Calculate prediction uncertainty
std_dev = trained_data['std_dev']  # e.g., 12.5 points
```

## Phase 5: Context Adjustments

### 5.1 Apply Signal-Based Adjustments
```python
# Initialize adjustment factor
adjustment_factor = 1.0
applied_adjustments = []

# A/TO Ratio Adjustment (triggered)
if abs(signals['diff_ato_ratio']) > 0.25:
    ato_boost = 1 + (abs(signals['diff_ato_ratio']) * 0.15)  # 1 + (0.29 * 0.15) = 1.0435
    adjustment_factor *= ato_boost  # +4.35%
    applied_adjustments.append("Control de balón mejor: Más posesiones eficientes")

# Other signal checks...
# (Rebound, FTI, Run detector, etc.)
```

### 5.2 Additional Context Adjustments
```python
# Blowout adjustment (if applicable)
if score_diff > 15 and time_remaining_pct <= 0.25:
    blowout_reduction = min(0.04, (score_diff - 20) / 200.0)
    adjustment_factor *= (1 - blowout_reduction)

# Garbage time adjustment (if applicable)
if garbage_time_risk > 0.6:
    gt_reduction = min(0.12, garbage_time_risk * 0.12)
    adjustment_factor *= (1 - gt_reduction)
```

### 5.3 Final Prediction Calculation
```python
# Apply all adjustments
final_prediction = base_prediction * adjustment_factor

# Example: 210.5 * 1.0435 ≈ 219.7 points
```

## Phase 6: Output Generation

### 6.1 Prediction Summary
```python
prediction_output = {
    'base_prediction': base_prediction,
    'final_prediction': final_prediction,
    'adjustment_factor': adjustment_factor,
    'adjustment_percentage': (adjustment_factor - 1) * 100,
    'applied_adjustments': applied_adjustments,
    'confidence_intervals': {
        'line_-2': final_prediction - 2 * std_dev,
        'line_-1': final_prediction - std_dev,
        'line_center': final_prediction,
        'line_+1': final_prediction + std_dev,
        'line_+2': final_prediction + 2 * std_dev
    }
}
```

### 6.2 Alert Generation
```python
# Generate live alerts based on signals
live_alerts = generate_live_alerts(signals, live_stats, game_info)

# Example alerts:
alerts = [
    "Control de balón mejor: Más posesiones eficientes",
    "Partido cerrado + pace alto: +2% (pace: 110.5, TS boost: 1.05)"
]
```

### 6.3 User Display
```
🏀 PREDICCIÓN FINAL: 219.7 pts (+4.4% vs base)

📊 LÍNEAS SUGERIDAS:
  🔥 Línea 207.0: Over 65.1% | Under 34.9%
  📈 Línea 219.7: Over 50.0% | Under 50.0%
  📈 Línea 232.4: Over 34.9% | Under 65.1%

🚨 ALERTAS:
   • Control de balón mejor: Más posesiones eficientes
   • Partido cerrado + pace alto: +2%
```

## Phase 7: Performance Tracking

### 7.1 Accuracy Metrics
```python
# Track prediction accuracy
accuracy_metrics = {
    'predictions_made': total_predictions,
    'correct_over_calls': correct_overs,
    'correct_under_calls': correct_unders,
    'avg_adjustment_impact': mean(abs(adjustment_factor - 1)),
    'signal_trigger_rate': signals_triggered / total_games
}
```

### 7.2 Signal Performance
```python
# Monitor signal effectiveness
signal_performance = {
    'ato_signal_accuracy': calculate_signal_accuracy('ato_ratio'),
    'treb_signal_accuracy': calculate_signal_accuracy('treb_diff'),
    'fti_signal_accuracy': calculate_signal_accuracy('fti'),
    'run_signal_accuracy': calculate_signal_accuracy('run_detector')
}
```

## Key Integration Points

### ML Model ↔ Live Signals
- **Model**: Provides base prediction using historical patterns
- **Signals**: Modify prediction based on live game dynamics
- **Result**: Context-aware prediction that adapts to current game flow

### Pre-Game ↔ Live Features
- **Pre-Game**: 120+ features capture historical team performance
- **Live**: 18 features capture current game state + signals
- **Integration**: Seamless combination for comprehensive prediction

### Signals ↔ Alerts
- **Signals**: Mathematical calculations for prediction adjustment
- **Alerts**: Human-readable explanations of signal impacts
- **Purpose**: Provide actionable insights for betting decisions

## Workflow Optimization

### Performance Improvements
- **Caching**: Pre-game features computed once per game
- **Async Processing**: Non-blocking API calls for live data
- **Signal Filtering**: Only O/U-relevant signals processed
- **Efficient Updates**: Minimal computation per live update

### Error Handling
- **Fallback Values**: Safe defaults for missing data
- **Validation**: Input validation at each step
- **Logging**: Comprehensive error tracking and debugging
- **Recovery**: Graceful degradation when components fail

This workflow ensures accurate, real-time O/U predictions that combine historical ML model insights with live game dynamics through sophisticated signal processing.