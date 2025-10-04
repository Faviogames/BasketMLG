# O/U Signal Processing Guide

## Overview
This guide explains how live signals are processed and integrated into Over/Under predictions. The system transforms raw SofaScore statistics into actionable prediction adjustments.

## Signal Processing Pipeline

### 1. Data Acquisition
```python
# Raw stats fetched from SofaScore API
live_stats = {
    'home_assists': 15,
    'away_assists': 12,
    'home_turnovers': 8,
    'away_turnovers': 10,
    'home_total_rebounds': 30,
    'away_total_rebounds': 25,
    'home_personal_fouls': 12,
    'away_personal_fouls': 5,
    # ... other stats
}
```

### 2. Signal Computation
Raw stats are transformed into normalized signals (0-1 scale):

#### Assist-to-Turnover Ratio (A/TO)
```python
# Raw calculation
home_ato = home_assists / max(home_turnovers, 1)  # 15/8 = 1.875
away_ato = away_assists / max(away_turnovers, 1)  # 12/10 = 1.2

# Shrinkage for stability (accounts for sample size)
total_turnovers = home_turnovers + away_turnovers  # 18
shrinkage = min(1.0, total_turnovers / 20.0)  # 0.9

# Normalization vs league baseline (NBA/WNBA avg A/TO ~1.2)
home_ato_norm = 0.5 + shrinkage * (home_ato - 1.2) / 1.8  # 0.5 + 0.9*(1.875-1.2)/1.8 = 0.79
away_ato_norm = 0.5 + shrinkage * (away_ato - 1.2) / 1.8  # 0.5 + 0.9*(1.2-1.2)/1.8 = 0.5

# Differential for prediction adjustment
diff_ato_ratio = home_ato_norm - away_ato_norm  # 0.79 - 0.5 = 0.29
```

#### Rebound Efficiency
```python
# Offensive rebound percentage
total_oreb_opportunities = home_oreb + away_oreb
home_oreb_pct = home_oreb / total_oreb_opportunities if total_oreb_opportunities > 0 else 0.25

# Total rebound differential (normalized)
total_reb_diff = home_total_rebounds - away_total_rebounds  # 30 - 25 = 5
home_treb_diff_norm = max(-1.0, min(1.0, total_reb_diff / 20.0))  # 5/20 = 0.25
```

#### Foul Trouble Index (FTI)
```python
# Estimate minutes played
quarters_played = 3  # Based on available quarter data
minutes_played = quarters_played * 12  # 36 minutes

# Fouls per minute
home_pf_rate = home_personal_fouls / minutes_played  # 12/36 = 0.333
away_pf_rate = away_personal_fouls / minutes_played  # 5/36 = 0.139

# Normalize vs league baseline (NBA avg ~0.5 fouls/minute)
home_fti = min(1.0, max(0.0, (home_pf_rate - 0.5) / 1.5))  # max(0, (0.333-0.5)/1.5) = 0.0
away_fti = min(1.0, max(0.0, (away_pf_rate - 0.5) / 1.5))  # max(0, (0.139-0.5)/1.5) = 0.0
```

### 3. Context Adjustment Application
Signals trigger prediction modifications when they exceed thresholds:

#### A/TO Ratio Adjustment
```python
if abs(diff_ato_ratio) > 0.25:  # Threshold exceeded (0.29 > 0.25)
    ato_boost = 1 + (abs(diff_ato_ratio) * 0.15)  # 1 + (0.29 * 0.15) = 1.0435
    adjustment_factor *= ato_boost  # +4.35% prediction increase
    alerts.append("Control de balón mejor: Más posesiones eficientes")
```

#### Rebound Efficiency Adjustment
```python
if abs(home_treb_diff_norm) > 0.4:  # Threshold check (0.25 < 0.4)
    # No adjustment triggered
    pass
else:
    # Signal within normal range
    pass
```

## Signal Thresholds & Impacts

| Signal | Threshold | Max Adjustment | O/U Logic |
|--------|-----------|----------------|-----------|
| A/TO Differential | >0.25 | ±15% | Better control = More efficient scoring |
| TREB Differential | >0.4 | ±12% | More rebounds = More possessions |
| FTI (Foul Trouble) | >0.7 | +30% | More fouls = More free throws |
| TS% Differential | >0.15 | ±15% | Better shooting = More points |
| Run Strength | >0.3 | +20% | Scoring bursts = Higher totals |

## Real-World Example

**Game Scenario**: Team A has excellent ball control, Team B struggles with turnovers

**Raw Stats**:
- Team A: 18 assists, 6 turnovers
- Team B: 10 assists, 14 turnovers

**Signal Processing**:
1. A/TO calculation: Team A (3.0), Team B (0.71)
2. Normalization: Team A (0.85), Team B (0.35)
3. Differential: 0.85 - 0.35 = 0.5
4. Threshold check: 0.5 > 0.25 ✓
5. Adjustment: 1 + (0.5 × 0.15) = 1.075 (+7.5% to prediction)

**Result**: Prediction increased by 7.5% due to superior ball control

## Signal Validation

### Testing Framework
```python
def test_ou_signals():
    # Test cases with expected outcomes
    test_cases = [
        {'name': 'High A/TO Differential', 'expected': 'adjustment_triggered'},
        {'name': 'Normal Range Signals', 'expected': 'no_adjustment'},
        {'name': 'Extreme Foul Trouble', 'expected': 'max_adjustment'}
    ]
```

### Performance Metrics
- **Signal Accuracy**: % of signals that correctly predict O/U outcomes
- **Adjustment Precision**: How well adjustments match actual game flow
- **Alert Relevance**: % of alerts that provide actionable O/U insights

## Troubleshooting

### Common Issues

#### Signal Not Triggering
```python
# Check signal calculation
print(f"A/TO Diff: {diff_ato_ratio}")  # Should be > 0.25 for adjustment
print(f"Threshold: 0.25")
print(f"Adjustment triggered: {abs(diff_ato_ratio) > 0.25}")
```

#### Unexpected Adjustment
```python
# Verify signal values
print(f"Signal values: {live_signals}")
print(f"Applied adjustments: {context_info['applied_adjustments']}")
```

#### Alert Not Showing
```python
# Check alert conditions
if abs(diff_ato_ratio) > 0.3:  # Higher threshold for alerts
    print("Alert should trigger")
```

## Integration with ML Model

### Feature Flow
```
Raw SofaScore Stats → Signal Processing → Context Adjustments → Modified Prediction
       ↓                    ↓                    ↓                    ↓
   API Response     Normalization     Multipliers     Final O/U Total
   (JSON)          (0-1 Scale)       (±15% max)      (Points)
```

### Prediction Modification
```python
# Base model prediction
base_prediction = model.predict(X_features)[0]  # e.g., 210.5

# Apply signal adjustments
final_prediction = base_prediction * adjustment_factor  # e.g., 210.5 * 1.075 = 226.29

# Result: Prediction increased 7.5% due to A/TO signal
```

## Best Practices

1. **Monitor Signal Thresholds**: Adjust based on league-specific patterns
2. **Validate Adjustments**: Compare predicted vs actual totals
3. **Log Signal Performance**: Track which signals provide best O/U accuracy
4. **Update Baselines**: Recalibrate normalization values as seasons progress

## Future Enhancements

- **Dynamic Thresholds**: Adjust based on game context (close game vs blowout)
- **Signal Combinations**: Multi-signal interactions for complex scenarios
- **League-Specific Tuning**: Different parameters for NBA vs WNBA vs NBL
- **Real-Time Calibration**: Automatic threshold adjustment based on performance