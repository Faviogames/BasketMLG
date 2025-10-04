# O/U Performance Metrics & Analytics

## Overview
This document defines key performance indicators (KPIs) for measuring the effectiveness of the O/U-focused live signal system and provides analytics for continuous improvement.

## Core Performance Metrics

### 1. Prediction Accuracy

#### Overall Accuracy
```python
def calculate_overall_accuracy(predictions, actuals):
    """Calculate overall prediction accuracy"""
    correct = 0
    total = len(predictions)

    for pred, actual in zip(predictions, actuals):
        if (pred > line and actual > line) or (pred < line and actual < line):
            correct += 1

    return correct / total if total > 0 else 0.0
```

#### Brier Score (Prediction Calibration)
```python
def calculate_brier_score(predictions, actuals, lines):
    """Measure prediction calibration quality"""
    brier_sum = 0
    n = len(predictions)

    for pred, actual, line in zip(predictions, actuals, lines):
        # Convert to probability (assuming normal distribution)
        z_score = (line - pred) / std_dev
        over_prob = 1 - stats.norm.cdf(z_score)

        # Actual outcome (1 if over, 0 if under)
        outcome = 1 if actual > line else 0

        # Brier score component
        brier_sum += (over_prob - outcome) ** 2

    return brier_sum / n if n > 0 else 1.0
```

### 2. Signal Effectiveness

#### Signal Trigger Rate
```python
def calculate_signal_trigger_rate(signal_history):
    """Percentage of games where signals trigger adjustments"""
    triggered_games = 0
    total_games = len(signal_history)

    for game_signals in signal_history:
        if any(abs(value) > threshold for signal, value in game_signals.items()
               if signal in ['diff_ato_ratio', 'home_treb_diff', 'home_fti', 'away_fti']):
            triggered_games += 1

    return triggered_games / total_games if total_games > 0 else 0.0
```

#### Signal Impact Analysis
```python
def analyze_signal_impact(predictions_with_signals, predictions_without_signals, actuals):
    """Measure improvement from signal adjustments"""
    errors_with_signals = [abs(pred - actual) for pred, actual in zip(predictions_with_signals, actuals)]
    errors_without_signals = [abs(pred - actual) for pred, actual in zip(predictions_without_signals, actuals)]

    avg_error_with = sum(errors_with_signals) / len(errors_with_signals)
    avg_error_without = sum(errors_without_signals) / len(errors_without_signals)

    improvement = avg_error_without - avg_error_with
    improvement_pct = (improvement / avg_error_without) * 100

    return {
        'avg_error_with_signals': avg_error_with,
        'avg_error_without_signals': avg_error_without,
        'absolute_improvement': improvement,
        'percentage_improvement': improvement_pct
    }
```

### 3. Adjustment Performance

#### Adjustment Distribution
```python
def analyze_adjustment_distribution(adjustment_factors):
    """Analyze how often and by how much predictions are adjusted"""
    adjustments = [abs(f - 1.0) for f in adjustment_factors]

    return {
        'mean_adjustment': sum(adjustments) / len(adjustments),
        'max_adjustment': max(adjustments),
        'adjustment_std': statistics.stdev(adjustments),
        'no_adjustment_rate': sum(1 for a in adjustments if a < 0.01) / len(adjustments),
        'significant_adjustment_rate': sum(1 for a in adjustments if a > 0.05) / len(adjustments)
    }
```

#### Adjustment Accuracy by Signal
```python
def calculate_adjustment_accuracy_by_signal(signal_history, adjustment_factors, actuals, baselines):
    """Calculate accuracy improvement for each signal type"""
    signal_accuracies = {}

    for signal_name in ['ato_ratio', 'treb_diff', 'fti', 'run_detector']:
        # Find games where this signal triggered
        triggered_games = []
        triggered_adjustments = []
        triggered_actuals = []

        for i, (signals, adj_factor, actual) in enumerate(zip(signal_history, adjustment_factors, actuals)):
            signal_key = f'diff_{signal_name}' if signal_name != 'fti' else 'home_fti'
            if signal_key in signals and abs(signals[signal_key]) > get_threshold(signal_name):
                triggered_games.append(i)
                triggered_adjustments.append(adj_factor)
                triggered_actuals.append(actual)

        if triggered_games:
            # Calculate baseline error (no adjustment)
            baseline_errors = [abs(baselines[i] - actual) for i, actual in zip(triggered_games, triggered_actuals)]

            # Calculate adjusted error
            adjusted_errors = [abs(baselines[i] * adj - actual) for i, adj, actual in zip(triggered_games, triggered_adjustments, triggered_actuals)]

            improvement = sum(baseline_errors) / len(baseline_errors) - sum(adjusted_errors) / len(adjusted_errors)

            signal_accuracies[signal_name] = {
                'triggered_games': len(triggered_games),
                'avg_improvement': improvement,
                'improvement_rate': improvement / (sum(baseline_errors) / len(baseline_errors)) * 100
            }

    return signal_accuracies
```

## Advanced Analytics

### 1. Signal Correlation Analysis

#### Signal Intercorrelation
```python
def analyze_signal_correlations(signal_history):
    """Analyze correlations between different signals"""
    signal_names = ['diff_ato_ratio', 'home_treb_diff', 'home_fti', 'away_fti', 'run_strength']
    correlation_matrix = {}

    for i, signal1 in enumerate(signal_names):
        correlation_matrix[signal1] = {}
        for j, signal2 in enumerate(signal_names):
            if i != j:
                values1 = [game.get(signal1, 0) for game in signal_history]
                values2 = [game.get(signal2, 0) for game in signal_history]

                if len(values1) > 1 and len(values2) > 1:
                    corr = statistics.correlation(values1, values2)
                    correlation_matrix[signal1][signal2] = corr
                else:
                    correlation_matrix[signal1][signal2] = 0.0

    return correlation_matrix
```

#### Signal-to-Outcome Correlation
```python
def analyze_signal_outcome_correlation(signal_history, actuals, lines):
    """Analyze how signals correlate with actual O/U outcomes"""
    correlations = {}

    for signal_name in ['diff_ato_ratio', 'home_treb_diff', 'home_fti', 'away_fti']:
        signal_values = [game.get(signal_name, 0) for game in signal_history]
        outcomes = [1 if actual > line else 0 for actual, line in zip(actuals, lines)]

        if len(signal_values) > 1:
            corr = statistics.correlation(signal_values, outcomes)
            correlations[signal_name] = corr

    return correlations
```

### 2. Contextual Performance Analysis

#### Performance by Game Context
```python
def analyze_context_performance(predictions, actuals, contexts):
    """Analyze performance across different game contexts"""
    context_performance = {}

    for context in ['close_game', 'high_pace', 'late_game', 'blowout']:
        context_games = [i for i, ctx in enumerate(contexts) if ctx.get(context, False)]

        if context_games:
            context_preds = [predictions[i] for i in context_games]
            context_actuals = [actuals[i] for i in context_games]

            accuracy = calculate_overall_accuracy(context_preds, context_actuals)
            avg_error = sum(abs(p - a) for p, a in zip(context_preds, context_actuals)) / len(context_games)

            context_performance[context] = {
                'games': len(context_games),
                'accuracy': accuracy,
                'avg_error': avg_error
            }

    return context_performance
```

#### League-Specific Performance
```python
def analyze_league_performance(predictions, actuals, leagues):
    """Analyze performance by league"""
    league_performance = {}

    for league in set(leagues):
        league_games = [i for i, l in enumerate(leagues) if l == league]

        if league_games:
            league_preds = [predictions[i] for i in league_games]
            league_actuals = [actuals[i] for i in league_games]

            accuracy = calculate_overall_accuracy(league_preds, league_actuals)
            brier = calculate_brier_score(league_preds, league_actuals, [0] * len(league_games))  # Simplified

            league_performance[league] = {
                'games': len(league_games),
                'accuracy': accuracy,
                'brier_score': brier
            }

    return league_performance
```

## Monitoring Dashboard

### Real-Time Metrics
```python
class OUMetricsDashboard:
    def __init__(self):
        self.metrics_history = []
        self.signal_performance = {}
        self.adjustment_analytics = {}

    def update_metrics(self, prediction_data, actual_result):
        """Update dashboard with new game data"""
        metrics = {
            'timestamp': datetime.now(),
            'prediction': prediction_data['final_prediction'],
            'actual': actual_result,
            'adjustment_factor': prediction_data['adjustment_factor'],
            'signals_triggered': prediction_data['live_signals_used'],
            'alerts_generated': len(prediction_data.get('live_alerts', []))
        }

        self.metrics_history.append(metrics)
        self._update_signal_performance(metrics)
        self._update_adjustment_analytics(metrics)

    def _update_signal_performance(self, metrics):
        """Track individual signal performance"""
        signals = metrics['signals_triggered']

        for signal_name, signal_value in signals.items():
            if signal_name not in self.signal_performance:
                self.signal_performance[signal_name] = []

            self.signal_performance[signal_name].append({
                'value': signal_value,
                'adjustment': metrics['adjustment_factor'],
                'accuracy': 1 if abs(metrics['prediction'] - metrics['actual']) < 10 else 0  # Simplified
            })

    def _update_adjustment_analytics(self, metrics):
        """Track adjustment effectiveness"""
        adj_factor = metrics['adjustment_factor']
        error = abs(metrics['prediction'] - metrics['actual'])

        adj_range = self._categorize_adjustment(adj_factor)

        if adj_range not in self.adjustment_analytics:
            self.adjustment_analytics[adj_range] = []

        self.adjustment_analytics[adj_range].append(error)

    def _categorize_adjustment(self, factor):
        """Categorize adjustment magnitude"""
        diff = abs(factor - 1.0)
        if diff < 0.02:
            return 'none'
        elif diff < 0.05:
            return 'small'
        elif diff < 0.10:
            return 'medium'
        else:
            return 'large'

    def get_dashboard_summary(self):
        """Generate comprehensive dashboard summary"""
        return {
            'overall_accuracy': self._calculate_overall_accuracy(),
            'signal_effectiveness': self._calculate_signal_effectiveness(),
            'adjustment_performance': self._calculate_adjustment_performance(),
            'recent_trends': self._calculate_recent_trends()
        }
```

## Alert System Analytics

### Alert Relevance Scoring
```python
def calculate_alert_relevance(alerts_history, outcomes):
    """Score how relevant alerts are to actual outcomes"""
    alert_relevance = {}

    for alert_type in ['ato_ratio', 'treb_diff', 'fti', 'run_detector']:
        relevant_alerts = 0
        total_alerts = 0

        for alerts, outcome in zip(alerts_history, outcomes):
            type_alerts = [a for a in alerts if alert_type in a.lower()]
            total_alerts += len(type_alerts)

            # Check if alert correctly predicted outcome
            for alert in type_alerts:
                if self._alert_matches_outcome(alert, outcome, alert_type):
                    relevant_alerts += 1

        relevance_score = relevant_alerts / total_alerts if total_alerts > 0 else 0.0
        alert_relevance[alert_type] = relevance_score

    return alert_relevance
```

## Continuous Improvement Framework

### 1. Weekly Review Process
- **Accuracy Analysis**: Review prediction accuracy by signal type
- **Threshold Tuning**: Adjust signal thresholds based on performance
- **Feature Importance**: Re-evaluate signal weights in adjustment formula

### 2. Monthly Optimization
- **Baseline Recalibration**: Update league-specific baselines
- **Signal Enhancement**: Add new O/U-relevant signals
- **Model Retraining**: Incorporate signal performance data

### 3. Quarterly Strategic Review
- **System Architecture**: Evaluate overall signal processing pipeline
- **New Signal Development**: Research additional O/U predictors
- **Competitive Analysis**: Benchmark against other O/U systems

## Implementation Example

```python
# Initialize metrics tracking
metrics_dashboard = OUMetricsDashboard()

# After each game
metrics_dashboard.update_metrics(prediction_data, actual_total)

# Generate weekly report
weekly_report = metrics_dashboard.get_dashboard_summary()
print(f"Weekly Accuracy: {weekly_report['overall_accuracy']:.1%}")
print(f"Signal Effectiveness: {weekly_report['signal_effectiveness']}")

# Identify improvement opportunities
if weekly_report['overall_accuracy'] < 0.55:
    print("⚠️  Accuracy below threshold - review signal thresholds")
    # Trigger threshold adjustment process
```

This comprehensive metrics framework ensures continuous monitoring and improvement of the O/U prediction system's performance.