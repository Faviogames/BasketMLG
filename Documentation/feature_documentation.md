# Basketball Analytics Feature Documentation

## Overview
This document provides comprehensive documentation of all features used in the basketball Over/Under prediction model. Features are categorized by type and include calculation methods, parameters, and predictive purpose.

## 1. Momentum Features (EMA-Based)

### Win Rate Features
- **Purpose**: Track team winning consistency and momentum
- **Calculation**: Exponential Moving Average (EMA) of win percentage
- **Ranges**: Short-term (3 games), Medium-term (8 games), Long-term (11 games)
- **Features**:
  - `ema_win_short_term_3`
  - `ema_win_medium_term_8`
  - `ema_win_long_term_11`

### Plus/Minus Features
- **Purpose**: Measure offensive/defensive efficiency trends
- **Calculation**: EMA of point differential per game
- **Features**:
  - `ema_plus_minus_short_term_3`
  - `ema_plus_minus_medium_term_8`
  - `ema_plus_minus_long_term_11`

### Scoring Efficiency Features
- **Purpose**: Track offensive scoring trends
- **Calculation**: EMA of points scored per game
- **Features**:
  - `ema_scoring_efficiency_short_term_3`
  - `ema_scoring_efficiency_medium_term_8`
  - `ema_scoring_efficiency_long_term_11`

### Defensive Stops Features
- **Purpose**: Track defensive performance trends
- **Calculation**: EMA of points allowed per game
- **Features**:
  - `ema_defensive_stops_short_term_3`
  - `ema_defensive_stops_medium_term_8`
  - `ema_defensive_stops_long_term_11`

### Clutch Performance Features
- **Purpose**: Measure performance in close games
- **Calculation**: EMA of win rate in games decided by ≤10 points
- **Features**:
  - `ema_clutch_performance_short_term_3`
  - `ema_clutch_performance_medium_term_8`
  - `ema_clutch_performance_long_term_11`

### Turnover Momentum Features
- **Purpose**: Track ball control and pace trends
- **Calculation**: EMA of turnover rate trends
- **Features**:
  - `ema_turnover_momentum_short_term_3`
  - `ema_turnover_momentum_medium_term_8`
  - `ema_turnover_momentum_long_term_11`

## 2. Advanced Statistics Features

### Possessions Features
- **Purpose**: Track pace and tempo
- **Calculation**: Rolling average of estimated possessions per game
- **Formula**: FGA - ORB + TOV + (0.44 × FTA)
- **Features**:
  - `home_avg_possessions_last_5`
  - `away_avg_possessions_last_5`
  - `diff_avg_possessions_last_5`

### Pace Features
- **Purpose**: Measure game tempo
- **Calculation**: Possessions normalized to 48 minutes
- **Features**:
  - `home_avg_pace_last_5`
  - `away_avg_pace_last_5`
  - `diff_avg_pace_last_5`

### Offensive Rating Features
- **Purpose**: Track offensive efficiency
- **Calculation**: Points per 100 possessions
- **Formula**: (PTS / POSS) × 100
- **Features**:
  - `home_avg_ortg_last_5`
  - `away_avg_ortg_last_5`
  - `diff_avg_ortg_last_5`

### True Shooting Percentage Features
- **Purpose**: Advanced shooting efficiency metric
- **Calculation**: Rolling average of TS%
- **Formula**: PTS / (2 × (FGA + 0.44 × FTA))
- **Features**:
  - `home_true_shooting_percentage`
  - `away_true_shooting_percentage`
  - `diff_true_shooting_percentage`

## 3. Performance Context Features

### Home Advantage Factor
- **Purpose**: Measure home court advantage
- **Calculation**: Recent performance adjustment (1.0 + PM/100)
- **Features**:
  - `home_home_advantage_factor`
  - `away_home_advantage_factor`
  - `diff_home_advantage_factor`

### Comeback Ability
- **Purpose**: Track ability to overcome deficits
- **Calculation**: Frequency of wins after trailing
- **Features**:
  - `home_comeback_ability`
  - `away_comeback_ability`
  - `diff_comeback_ability`

### Consistency Index
- **Purpose**: Measure scoring consistency
- **Calculation**: 1 / (1 + CV) where CV is coefficient of variation
- **Features**:
  - `home_consistency_index`
  - `away_consistency_index`
  - `diff_consistency_index`

## 4. Live Game Features

### Quarter Totals
- **Purpose**: Current game scoring by period
- **Calculation**: Sum of points scored in each quarter
- **Features**:
  - `q1_total`, `q2_total`, `q3_total`
  - `halftime_total`, `q3_end_total`

### Live Efficiency Metrics
- **Purpose**: Real-time offensive/defensive efficiency
- **Calculation**: Points per possession in current game
- **Features**:
  - `live_efficiency_home`
  - `live_efficiency_away`

### Live Pace Metrics
- **Purpose**: Current game tempo
- **Calculation**: Estimated possessions per 48 minutes
- **Features**:
  - `live_pace_estimate`
  - `enhanced_pace_estimate`

### Momentum Indicators
- **Purpose**: Track scoring momentum shifts
- **Calculation**: Quarter-to-quarter scoring changes
- **Features**:
  - `q2_trend`, `q3_trend`
  - `live_momentum_shift`

## 4.5. O/U-Focused Live Signals (PHASE 2)

### Overview
Advanced live signals specifically designed for Over/Under predictions. These signals are computed from SofaScore live stats and directly impact model predictions through context adjustments.

### Foul Trouble Index (FTI)
- **Purpose**: Detect foul trouble situations that lead to more free throws
- **Input**: `personal_fouls` from SofaScore API
- **Calculation**: FTI = (pf_rate - 0.5) / 1.5 (normalized 0-1)
- **Thresholds**:
  - >0.7: High foul trouble (+25% max prediction adjustment)
  - >0.8: Extreme foul trouble (+30% max prediction adjustment)
- **O/U Impact**: More free throws = More points
- **Features**:
  - `home_fti`, `away_fti`, `diff_fti`

### True Shooting % (TS%) Live
- **Purpose**: Real-time shooting efficiency measurement
- **Input**: `field_goals_made/attempted` + `free_throws_made/attempted`
- **Calculation**: TS% = Points / (2 × (FGA + 0.44 × FTA)) with shrinkage
- **Thresholds**: Differential >0.15 triggers shooting efficiency adjustments
- **O/U Impact**: Better shooting = More points per possession
- **Features**:
  - `home_ts_live`, `away_ts_live`, `diff_ts_live`

### Effective Field Goal % (eFG%) Live
- **Purpose**: Advanced shooting efficiency with 3-point bonus
- **Input**: Field goal stats with 3-point weighting
- **Calculation**: eFG% = (FGM + 0.5 × 3PM) / FGA with shrinkage
- **O/U Impact**: Accounts for extra value of 3-point shots
- **Features**:
  - `home_efg_live`, `away_efg_live`, `diff_efg_live`

### Run Detector
- **Purpose**: Identify scoring runs and momentum shifts
- **Input**: Scoring patterns and time-based deltas
- **Logic**: Detects 6-8 point runs in 3-5 minute windows
- **Thresholds**: Strength >0.3 triggers +20% prediction adjustment
- **O/U Impact**: Immediate point bursts affect totals
- **Features**:
  - `run_active`, `run_side`, `run_strength`

### Assist-to-Turnover Ratio (A/TO)
- **Purpose**: Measure ball control and offensive efficiency
- **Input**: `assists` and `turnovers` from SofaScore
- **Calculation**: A/TO ratio with shrinkage for stability
- **Thresholds**: Differential >0.25 triggers ±15% prediction adjustment
- **O/U Impact**: Better ball control = More efficient possessions = More points
- **Features**:
  - `home_ato_ratio`, `away_ato_ratio`, `diff_ato_ratio`

### Rebound Efficiency Metrics
- **Purpose**: Track possession battle through rebounding
- **Input**: `offensive_rebounds`, `defensive_rebounds`, `total_rebounds`
- **Calculation**: OREB%, DREB%, TREB differential
- **Thresholds**: TREB differential >0.4 triggers ±12% prediction adjustment
- **O/U Impact**: More rebounds = More possessions = More points
- **Features**:
  - `home_oreb_pct`, `away_oreb_pct`, `diff_oreb_pct`
  - `home_dreb_pct`, `away_dreb_pct`, `diff_dreb_pct`
  - `home_treb_diff`, `away_treb_diff`

### Signal Integration
- **Processing**: Raw SofaScore stats → Advanced signals → Context adjustments
- **Impact**: Signals modify predictions through multiplicative factors
- **Alerts**: Automatic alerts when signals exceed thresholds
- **Logging**: Full telemetry of signal calculations and adjustments

## 5. Balance Features

### Game Balance Score
- **Purpose**: Measure game competitiveness
- **Calculation**: 1 / (1 + lead/10) - higher = more balanced
- **Features**:
  - `game_balance_score`

### Potential Blowout Indicator
- **Purpose**: Detect games likely to become blowouts
- **Calculation**: Binary flag when Q3 lead ≥15 points
- **Features**:
  - `is_potential_blowout`

### Garbage Time Risk
- **Purpose**: Predict likelihood of meaningless final minutes
- **Calculation**: Logistic function based on lead and time remaining
- **Features**:
  - `garbage_time_risk`

### Intensity Drop Factor
- **Purpose**: Measure decline in game intensity
- **Calculation**: Quarter-to-quarter scoring variance
- **Features**:
  - `intensity_drop_factor`

### Lead Stability
- **Purpose**: Track lead changes throughout game
- **Calculation**: Whether leader changes between quarters
- **Features**:
  - `lead_stability`

## 6. Advanced Psychological Features

### Psychological Momentum Cascade Index (PMCI)
- **Purpose**: Detect psychological momentum spirals
- **Calculation**: Weighted cascade of negative events
- **Formula**: Negative events (deficit ≥10, FG%<40%, TOV>18) create exponential cascades
- **Parameters**:
  - `pmci_neg_pm_threshold`: -10 (point deficit threshold)
  - `pmci_low_fg_pct`: 0.40 (field goal % threshold)
  - `pmci_high_turnovers`: 18 (turnover threshold)
  - `pmci_streak_exponent`: 1.5 (cascade amplification)
- **Features**:
  - `home_pmci`, `away_pmci`, `diff_pmci`

### Competitive Pressure Response Coefficient (CPRC)
- **Purpose**: Measure response to opponent strength
- **Calculation**: Performance vs expected based on opponent rating
- **Formula**: (Actual - Expected) / Expected, adjusted by opponent strength
- **Parameters**:
  - `cprc_expected_slope`: 0.3 (adjustment per strength point)
  - `cprc_response_cap_weak`: 1.2 (cap for weak opponents)
- **Features**:
  - `home_cprc`, `away_cprc`, `diff_cprc`

### Tactical Adaptation Velocity (TAV)
- **Purpose**: Track speed of strategic adjustments
- **Calculation**: Recovery rate after poor quarters
- **Formula**: Count improvements after bad quarters, with sustainability bonus
- **Parameters**:
  - `tav_window`: 8 (games to analyze)
  - `tav_eff_threshold`: 1.5 (efficiency recovery threshold)
  - `tav_recovery_multiplier`: 1.3 (improvement multiplier)
- **Features**:
  - `home_tav`, `away_tav`, `diff_tav`

### Energy Distribution Intelligence (EDI)
- **Purpose**: Measure strategic energy management
- **Calculation**: Pacing pattern analysis across quarters
- **Formula**: Points per possession by quarter, with Q4 FT bonus
- **Parameters**:
  - `edi_window`: 6 (games to analyze)
  - `edi_peak_q4_bonus`: 1.5 (Q4 intensity bonus)
  - `edi_close_game_margin`: 5 (close game threshold)
- **Features**:
  - `home_edi`, `away_edi`, `diff_edi`

### Power Law Scoring Volatility Index (PLSVI)
- **Purpose**: Detect explosive scoring events
- **Calculation**: Frequency of power-law scoring bursts in Q4
- **Formula**: Events where Q4 points exceed expected by 30%
- **Parameters**:
  - `plsvi_window`: 12 (games to analyze)
  - `plsvi_q4_positive_threshold`: 28 (burst threshold)
  - `plsvi_q4_negative_threshold`: 15 (collapse threshold)
- **Features**:
  - `home_plsvi`, `away_plsvi`, `diff_plsvi`

## 7. Quarter-Based Features

### Q4 Free Throw Rate
- **Purpose**: Predict Q4 foul trouble and free throws
- **Calculation**: Q4 FTA / (Q1+Q2+Q3 FTA average)
- **Features**:
  - `home_q4_ft_rate`, `away_q4_ft_rate`, `diff_q4_ft_rate`

### Q3→Q4 Pace Shift
- **Purpose**: Track pace acceleration/deceleration into Q4
- **Calculation**: Q4 possessions - Q3 possessions
- **Formula**: Uses FGA - ORB + TOV + 0.44×FTA per quarter
- **Features**:
  - `home_q3_q4_pace_shift`, `away_q3_q4_pace_shift`, `diff_q3_q4_pace_shift`

### Q4 Turnover Rate
- **Purpose**: Measure ball control in clutch periods
- **Calculation**: Q4 TOV / Q4 estimated possessions
- **Features**:
  - `home_q4_to_rate`, `away_q4_to_rate`, `diff_q4_to_rate`

## 8. Direct Momentum Features

### Basic Direct Features
- **win_rate**: Simple win percentage (redundant with EMA)
- **avg_plus_minus**: Simple point differential average
- **scoring_consistency**: 1 / (1 + CV) of points scored
- **pace_stability**: 1 / (1 + CV) of total score

### Advanced Direct Features
- **offensive_efficiency_trend**: Recent vs historical scoring ratio
- **quarter_scoring_pattern**: Q4 vs Q1 scoring differential
- **defensive_fatigue**: Recent vs historical points allowed
- **shooting_rhythm**: Enhanced shooting consistency metric
- **second_half_efficiency**: Q3+Q4 vs Q1+Q2 scoring ratio
- **efficiency_differential**: Offensive vs defensive efficiency gap

### Schedule-Based Features
- **schedule_analysis_score**: Rest, back-to-back, and density analysis
- **back_to_back_fatigue**: Binary fatigue indicator
- **defensive_intensity_drop**: Defensive performance trend
- **rolling_volatility_3_games**: 3-game scoring variance
- **momentum_acceleration**: Scoring trend acceleration
- **pace_differential_trend**: Pace change over time

## 9. Shot Selection Features

### Shot Selection Intelligence
- **Purpose**: Evaluate shot quality and distribution
- **Calculation**: 3-point rate, efficiency, and 2-point efficiency
- **Formula**: Weighted combination of rates and efficiencies
- **Features**:
  - `home_shot_selection_score`
  - `away_shot_selection_score`
  - `diff_shot_selection_score`

## 10. Head-to-Head (H2H) Features

### H2H Average Total Score
- **Purpose**: Historical scoring in matchups
- **Calculation**: Average total points in previous meetings
- **Features**:
  - `h2h_avg_total_score`

### H2H Pace Average
- **Purpose**: Typical tempo in matchups
- **Calculation**: Average pace in previous meetings
- **Features**:
  - `h2h_pace_avg`

### H2H First Half Average
- **Purpose**: Early game scoring patterns
- **Calculation**: Average first half points in matchups
- **Features**:
  - `h2h_first_half_avg_total`

### H2H Second Half Surge
- **Purpose**: Late game scoring acceleration
- **Calculation**: Second half - first half differential
- **Features**:
  - `h2h_second_half_surge_avg`

### H2H Comeback Frequency
- **Purpose**: Comeback likelihood in matchups
- **Calculation**: Frequency of comebacks in previous meetings
- **Features**:
  - `h2h_comeback_freq`

## Configuration and Parameters

### EMA Ranges
```python
EMA_RANGES = {
    'short_term': 3,    # Recent performance
    'medium_term': 8,   # Mid-term trends
    'long_term': 11     # Season-long patterns
}
```

### Advanced Feature Parameters
Located in `config.py` under `ADVANCED_FEATURE_PARAMS` with calibration values for:
- PMCI cascade thresholds
- CPRC response coefficients
- TAV adaptation windows
- EDI energy patterns
- PLSVI volatility detection

### Ablation Controls
- `ADVANCED_FEATURE_ABLATION`: Enable/disable advanced features
- `QUARTER_FEATURES_ABLATION`: Enable/disable quarter-based features
- `ENABLE_FEATURE_FILTERING`: Remove useless features

## Feature Engineering Notes

1. **Anti-Leakage**: All pre-game features use historical data only (df.iloc[:index])
2. **Normalization**: Features scaled to [0,1] where applicable
3. **Robustness**: Safe division and NaN handling throughout
4. **Live Compatibility**: Features work with partial game data
5. **Multi-League**: Parameters tuned for NBA/WNBA differences

## Predictive Impact

Top features by importance typically include:
1. Live game state (halftime_total, q1_total)
2. Recent performance (EMA short-term features)
3. Pace metrics (live_pace_estimate, possessions)
4. Advanced psychological features (PMCI, CPRC)
5. Schedule factors (schedule_analysis_score)

This comprehensive feature set captures traditional box score metrics, advanced analytics, psychological factors, and live game dynamics for robust Over/Under predictions.