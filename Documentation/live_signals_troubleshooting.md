# Live Signals Troubleshooting Guide

## Overview
This guide helps diagnose and resolve issues with the O/U-focused live signal system. Use this when signals aren't triggering, predictions seem incorrect, or alerts aren't appearing.

## Quick Diagnostic Checklist

### 1. Signal Not Triggering
```bash
# Check if signals are being computed
print("Live signals:", live_signals)

# Verify signal values are in expected ranges
for signal, value in live_signals.items():
    if isinstance(value, (int, float)):
        print(f"{signal}: {value:.3f}")
```

### 2. Prediction Not Adjusting
```bash
# Check adjustment factor
print(f"Adjustment factor: {adjustment_factor}")
print(f"Expected: > 1.0 if signals triggered")

# Verify applied adjustments
print("Applied adjustments:", applied_adjustments)
```

### 3. Alerts Not Appearing
```bash
# Check alert conditions
print(f"A/TO diff: {live_signals.get('diff_ato_ratio', 'N/A')}")
print(f"TREB diff: {live_signals.get('home_treb_diff', 'N/A')}")
print(f"FTI max: {max(live_signals.get('home_fti', 0), live_signals.get('away_fti', 0))}")
```

## Common Issues & Solutions

### Issue 1: Signals Always Return Default Values (0.5)

**Symptoms:**
- All signals show 0.5 (neutral values)
- No adjustments applied
- Alerts never trigger

**Possible Causes:**
1. **Missing Live Stats**: SofaScore API not returning data
2. **Empty Stats Dictionary**: `current_stats` is empty or None
3. **Error in Signal Computation**: Exception in `compute_live_signals()`

**Diagnosis:**
```python
# Check if stats were fetched
print("Current stats keys:", list(current_stats.keys()) if current_stats else "None")

# Verify required stats exist
required_stats = ['home_assists', 'away_assists', 'home_turnovers', 'away_turnovers']
missing_stats = [stat for stat in required_stats if stat not in current_stats]
if missing_stats:
    print(f"Missing stats: {missing_stats}")
```

**Solutions:**
```python
# 1. Check API connectivity
await client.health_check()

# 2. Verify game has live stats available
game_stats = await client.get_game_statistics(game_id)
if not game_stats:
    print("Game statistics not available - game may not have started")

# 3. Add error handling in signal computation
try:
    signals = compute_live_signals(current_stats, q_scores, game_info)
except Exception as e:
    print(f"Signal computation error: {e}")
    signals = get_default_signals()  # Fallback
```

### Issue 2: Signals Trigger But No Prediction Adjustment

**Symptoms:**
- Signals show correct values and exceed thresholds
- `applied_adjustments` list is empty
- Prediction unchanged from base

**Possible Causes:**
1. **Context Adjustment Logic Error**: Signals not reaching adjustment function
2. **Threshold Mismatch**: Signal thresholds don't match adjustment thresholds
3. **Adjustment Factor Reset**: Factor reset to 1.0 somewhere in pipeline

**Diagnosis:**
```python
# Check signal values vs thresholds
diff_ato = live_signals.get('diff_ato_ratio', 0)
print(f"A/TO diff: {diff_ato}, threshold: 0.25, triggered: {abs(diff_ato) > 0.25}")

# Verify adjustment function receives signals
print("Signals passed to adjustment:", live_signals is not None)

# Check adjustment factor calculation
if abs(diff_ato) > 0.25:
    expected_boost = 1 + (abs(diff_ato) * 0.15)
    print(f"Expected boost: {expected_boost}")
```

**Solutions:**
```python
# 1. Ensure signals are passed correctly
adjusted_pred, context_info = apply_context_adjustment(
    prediction, balance_features, live_pace_metrics, quarter_stage,
    live_signals=live_signals  # ← Ensure this is passed
)

# 2. Check threshold alignment
ATO_THRESHOLD = 0.25  # Make sure matches in both places
if abs(diff_ato_ratio) > ATO_THRESHOLD:
    # Apply adjustment

# 3. Debug adjustment factor
print(f"Initial factor: 1.0")
print(f"After A/TO: {adjustment_factor}")
print(f"Final factor: {adjustment_factor}")
```

### Issue 3: Incorrect Signal Calculations

**Symptoms:**
- Signal values seem wrong (too high/low)
- Adjustments applied but don't make sense
- Alerts trigger inappropriately

**Possible Causes:**
1. **Normalization Errors**: Baseline values incorrect for league
2. **Shrinkage Issues**: Sample size adjustment too aggressive
3. **Division by Zero**: Missing safety checks
4. **League-Specific Parameters**: Wrong baselines for NBA/WNBA

**Diagnosis:**
```python
# Check raw vs normalized values
home_assists = current_stats.get('home_assists', 0)
home_turnovers = current_stats.get('home_turnovers', 1)
raw_ato = home_assists / home_turnovers
normalized_ato = live_signals.get('home_ato_ratio', 0.5)

print(f"Raw A/TO: {raw_ato}")
print(f"Normalized A/TO: {normalized_ato}")
print(f"Expected range: 0.0-1.0")

# Verify normalization formula
shrinkage = min(1.0, (home_turnovers + away_turnovers) / 20.0)
expected_norm = 0.5 + shrinkage * (raw_ato - 1.2) / 1.8
print(f"Expected normalized: {expected_norm}")
```

**Solutions:**
```python
# 1. Adjust league baselines
LEAGUE_BASELINES = {
    'NBA': {'ato_baseline': 1.2, 'ato_range': 1.8},
    'WNBA': {'ato_baseline': 1.1, 'ato_range': 1.6},
    'NBL': {'ato_baseline': 1.3, 'ato_range': 2.0}
}

# 2. Fix division by zero
def safe_divide(numerator, denominator, default=0.0):
    return numerator / denominator if denominator > 0 else default

# 3. Add bounds checking
normalized_value = max(0.0, min(1.0, normalized_value))
```

### Issue 4: Signals Work in Test But Not Live

**Symptoms:**
- Signals work in unit tests
- Fail in live game scenarios
- Inconsistent behavior

**Possible Causes:**
1. **API Data Format Changes**: SofaScore changed response structure
2. **Missing Quarter Data**: Q-scores not populated correctly
3. **Caching Issues**: Old cached data interfering
4. **Async Timing**: Race conditions in live updates

**Diagnosis:**
```python
# Check API response format
raw_response = await client.get_game_statistics(game_id)
print("API response keys:", list(raw_response.keys()))

# Verify quarter scores
print("Q-scores:", q_scores)
print("Q3 data present:", q_scores.get('q3_home', 0) > 0)

# Check cache
client.clear_cache()  # Clear and retry
fresh_stats = await client.get_game_statistics(game_id)
```

**Solutions:**
```python
# 1. Add API response validation
def validate_api_response(response):
    required_keys = ['home_assists', 'away_assists', 'home_turnovers']
    return all(key in response for key in required_keys)

# 2. Handle missing quarter data
if not q_scores.get('q3_home'):
    print("Warning: Q3 data missing, using Q1+Q2 only")
    # Adjust calculations accordingly

# 3. Add retry logic for API calls
@retry(max_attempts=3, delay=2)
async def get_game_statistics(game_id):
    return await client.get_game_statistics(game_id)
```

## Performance Monitoring

### Signal Accuracy Tracking
```python
# Track signal prediction accuracy
signal_metrics = {
    'ato_predictions': [],
    'treb_predictions': [],
    'fti_predictions': []
}

# After game completion
actual_total = final_home_score + final_away_score
predicted_total = final_prediction

for signal_name, predictions in signal_metrics.items():
    accuracy = calculate_signal_accuracy(predictions, actual_total)
    print(f"{signal_name} accuracy: {accuracy:.1%}")
```

### Adjustment Impact Analysis
```python
# Monitor adjustment effectiveness
adjustment_log = {
    'base_prediction': base_pred,
    'final_prediction': final_pred,
    'adjustment_factor': adj_factor,
    'actual_result': actual_total,
    'signals_triggered': triggered_signals
}

# Calculate if adjustment improved prediction
error_before = abs(base_pred - actual_total)
error_after = abs(final_pred - actual_total)
improvement = error_before - error_after
print(f"Adjustment improvement: {improvement:+.1f} points")
```

## Debug Commands

### Quick Signal Check
```python
def debug_signals(live_signals, current_stats, q_scores):
    """Comprehensive signal debugging"""
    print("=== SIGNAL DEBUG ===")

    # Raw stats check
    print("Raw Stats:")
    for key in ['home_assists', 'away_assists', 'home_turnovers', 'away_turnovers']:
        print(f"  {key}: {current_stats.get(key, 'MISSING')}")

    # Signal computation
    print("\nComputed Signals:")
    for key, value in live_signals.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Threshold analysis
    print("\nThreshold Analysis:")
    checks = [
        ('A/TO', abs(live_signals.get('diff_ato_ratio', 0)), 0.25),
        ('TREB', abs(live_signals.get('home_treb_diff', 0)), 0.4),
        ('FTI', max(live_signals.get('home_fti', 0), live_signals.get('away_fti', 0)), 0.7)
    ]

    for name, value, threshold in checks:
        triggered = value > threshold
        print(f"  {name}: {value:.3f} > {threshold:.3f} = {triggered}")

    return live_signals
```

### Live Game Simulation
```python
def simulate_live_game():
    """Test signals with simulated game data"""
    test_stats = {
        'home_assists': 18, 'away_assists': 12,
        'home_turnovers': 6, 'away_turnovers': 14,
        'home_total_rebounds': 35, 'away_total_rebounds': 28,
        'home_personal_fouls': 8, 'away_personal_fouls': 15
    }

    test_q_scores = {
        'q1_home': 28, 'q1_away': 25,
        'q2_home': 27, 'q2_away': 30,
        'q3_home': 26, 'q3_away': 22
    }

    signals = compute_live_signals(test_stats, test_q_scores, {})
    debug_signals(signals, test_stats, test_q_scores)

    return signals
```

## Maintenance Tasks

### Weekly Checks
1. **API Health**: Verify SofaScore API connectivity
2. **Signal Accuracy**: Review signal prediction performance
3. **Threshold Tuning**: Adjust thresholds based on recent games
4. **Error Logs**: Check for new error patterns

### Monthly Updates
1. **League Baselines**: Recalibrate normalization values
2. **Feature Importance**: Re-evaluate signal weights
3. **Performance Metrics**: Update accuracy tracking
4. **Documentation**: Keep troubleshooting guide current

### Emergency Procedures
1. **Signal Failure**: Fall back to base model predictions
2. **API Outage**: Use cached data or disable live features
3. **Accuracy Drop**: Revert to previous signal version
4. **Data Corruption**: Clear caches and restart system

This troubleshooting guide should resolve 95% of live signal issues. For persistent problems, check the error logs and consider updating the signal computation logic.