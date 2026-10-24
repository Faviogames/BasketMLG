> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](auditoria/CONTEXTO_DOCUMENTATION.md) y [estado vigente](auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# NBA Odds Data Extraction System for Machine Learning

## Overview

This system provides a comprehensive solution for collecting NBA odds data from the-odds-api.com and preparing it for machine learning applications. The system extracts data from three market types (h2h/moneyline, spreads, and totals), validates the data quality, and exports it in multiple formats optimized for ML workflows.

## 🏀 Key Features

- **Multi-Market Collection**: Collects h2h (moneyline), spreads, and totals data simultaneously
- **Automated Pipeline**: Scheduled data collection with caching and deduplication
- **Data Validation**: Comprehensive quality checks and data cleaning
- **Multiple Export Formats**: CSV, JSON, and Parquet formats for different ML workflows
- **ML-Ready Features**: Pre-engineered features for machine learning
- **Quality Scoring**: Automatic data quality assessment and recommendations

## 📁 System Architecture

```
BasketMLG v2/
├── core/
│   ├── comprehensive_odds_extractor.py    # Main data collection engine
│   ├── odds_data_pipeline.py             # Automated collection pipeline
│   ├── odds_data_validator.py            # Data quality and validation
│   └── odds_api.py                       # Original odds API (existing)
├── data/
│   └── odds_collection/                  # Output directory
├── odds_extraction_demo.py              # Complete usage demonstration
└── Documentation/
    └── odds_extraction_system.md        # This documentation
```

## 🚀 Quick Start

### 1. Basic Data Collection

```python
from core.comprehensive_odds_extractor import ComprehensiveOddsExtractor

# Initialize the extractor
extractor = ComprehensiveOddsExtractor()

# Collect all market data
data = extractor.get_all_markets_odds()

# Export in multiple formats
from core.comprehensive_odds_extractor import export_all_formats
files = export_all_formats(data, "my_nba_odds")
```

### 2. Data Validation and Cleaning

```python
from core.odds_data_validator import validate_and_clean_odds_data

# Validate and clean data
cleaned_data, validation_result = validate_and_clean_odds_data(data)

print(f"Quality Score: {validation_result.quality_score:.2f}")
print(f"Valid Records: {validation_result.valid_records}")
```

### 3. Automated Collection Pipeline

```python
from core.odds_data_pipeline import OddsDataPipeline

# Initialize pipeline
pipeline = OddsDataPipeline()

# Run collection cycle
result = pipeline.run_collection_cycle()

# Get dataset statistics
stats = pipeline.get_dataset_statistics()
```

## 📊 Data Structure

### MLGameData Object

Each game record contains the following fields:

#### Basic Information
- `game_id`: Unique identifier for the game
- `game_date`: Date when data was collected
- `home_team`: Normalized home team name
- `away_team`: Normalized away team name
- `commence_time`: Scheduled game time

#### Head-to-Head (Moneyline) Odds
- `h2h_home_odds`: Decimal odds for home team win
- `h2h_away_odds`: Decimal odds for away team win
- `h2h_bookmaker`: Bookmaker providing these odds

#### Spreads Odds
- `spread_home_line`: Point spread for home team
- `spread_home_odds`: Decimal odds for home team covering spread
- `spread_away_line`: Point spread for away team (negative of home)
- `spread_away_odds`: Decimal odds for away team covering spread
- `spread_bookmaker`: Bookmaker providing these odds

#### Totals (Over/Under) Odds
- `totals_line`: Total points line
- `over_odds`: Decimal odds for Over
- `under_odds`: Decimal odds for Under
- `totals_bookmaker`: Bookmaker providing these odds

#### Metadata
- `api_response_time`: When the data was collected
- `raw_data`: Complete original API response

## 🔧 Core Components

### 1. ComprehensiveOddsExtractor

The main data collection engine that:
- Fetches data from all three market types (h2h, spreads, totals)
- Normalizes team names for consistency
- Combines data from multiple markets into unified records
- Provides multiple export formats (CSV, JSON, Parquet)
- Includes caching to avoid redundant API calls

**Key Methods:**
- `get_all_markets_odds()`: Main collection method
- `export_to_csv()`: Export to CSV format
- `export_to_json()`: Export to JSON format
- `export_to_parquet()`: Export to Parquet format (optimized for ML)
- `get_data_summary()`: Get collection statistics

### 2. OddsDataPipeline

Automated pipeline for continuous data collection:
- Scheduled data collection every 15 minutes
- Automatic data processing and feature engineering
- Historical data storage and management
- ML dataset preparation
- Data retention and cleanup

**Key Methods:**
- `run_collection_cycle()`: Execute one collection cycle
- `get_dataset_statistics()`: Get comprehensive dataset statistics
- `start_automated_collection()`: Start continuous collection
- `export_ml_dataset()`: Export ML-ready dataset

### 3. OddsDataValidator

Data quality assurance and validation:
- Comprehensive data validation rules
- Quality scoring (0-1 scale)
- Data cleaning and deduplication
- Data profiling and statistics
- Actionable recommendations

**Key Methods:**
- `validate_data()`: Complete validation with scoring
- `get_data_profile()`: Generate data profile
- `clean_data()`: Remove invalid records
- `calculate_quality_score()`: Compute quality metrics

## 📈 ML-Optimized Features

The system automatically adds several ML-friendly features:

### Time-Based Features
- `days_to_game`: Days until game starts
- `hours_to_game`: Hours until game starts

### Odds-Derived Features
- `h2h_implied_prob_home`: Implied probability for home team
- `h2h_implied_prob_away`: Implied probability for away team
- `h2h_market_margin`: Bookmaker margin
- `over_under_spread`: Difference between Over/Under odds
- `totals_market_efficiency`: Market efficiency measure

### Team Strength Indicators
- `home_strength`: Home team strength based on odds
- `away_strength`: Away team strength based on odds
- `game_competitiveness`: How competitive the game is expected to be

### Spread Features
- `spread_value`: Point spread value
- `spread_favorite`: Which team is favored

## 💾 Data Export Formats

### CSV Format
- **Use case**: General analysis, Excel, basic ML
- **Pros**: Universal compatibility, human-readable
- **Cons**: Larger file size, no data type preservation

### JSON Format
- **Use case**: Web applications, complex nested data
- **Pros**: Preserves all data including raw responses
- **Cons**: Larger file size, slower for large datasets

### Parquet Format
- **Use case**: Large-scale ML training, cloud storage
- **Pros**: Optimized for ML, fast reading, compressed
- **Cons**: Requires specific libraries (pandas, pyarrow)

## 🔄 Automated Workflow

### 1. Data Collection
```
API Request → Data Extraction → Normalization → Caching
```

### 2. Data Processing
```
Raw Data → Validation → Feature Engineering → Storage
```

### 3. ML Preparation
```
Processed Data → Quality Checks → Target Variables → ML Dataset
```

## 📊 Data Quality Metrics

The system provides comprehensive quality metrics:

### Completeness Score
- Percentage of non-null values across all fields
- Market coverage statistics
- Data availability per bookmaker

### Accuracy Score
- Odds value validation (reasonable ranges)
- Spread and totals range validation
- Team name consistency checks

### Overall Quality Score (0-1)
- Weighted combination of completeness and accuracy
- Penalty for critical issues
- Bonus for high data completeness

## 🎯 ML Use Cases

### 1. Odds Prediction
Use historical odds to predict future odds movements

### 2. Game Outcome Prediction
Combine odds data with team statistics to predict game winners

### 3. Betting Strategy Development
Analyze market efficiency and identify value bets

### 4. Live Betting Models
Use odds changes during games to predict outcomes

### 5. Arbitrage Detection
Monitor multiple bookmakers for arbitrage opportunities

## 🔍 Validation Rules

### Critical Issues
- Missing required fields (teams, times)
- Invalid odds values (≤0 or extremely high)
- Inconsistent spread lines
- Games scheduled too far in future

### Warnings
- Missing optional odds data
- Unusually low odds
- Inconsistent team name normalization
- Limited bookmaker diversity

### Quality Recommendations
- Data collection improvements
- Additional data sources
- Validation rule adjustments
- ML model preparation suggestions

## 🚀 Advanced Usage

### Custom Validation Rules

```python
# Add custom validation rules
validator = OddsDataValidator()
validator.validation_rules['odds_range'] = (1.01, 15.0)  # Wider odds range
validator.validation_rules['totals_range'] = (170.0, 300.0)  # Extended totals range

# Custom validation
result = validator.validate_data(data)
```

### Custom Feature Engineering

```python
# Extend the pipeline with custom features
def add_custom_features(df):
    # Add your custom features here
    df['home_advantage'] = df['h2h_home_odds'] < df['h2h_away_odds']
    df['high_total_game'] = df['totals_line'] > 220
    return df

# Use in pipeline
pipeline = OddsDataPipeline()
# Override the _add_ml_features method or extend the class
```

### Multi-Sport Extension

```python
# Extend to other sports
extractor = ComprehensiveOddsExtractor()
# Modify base_url and sport parameters in _fetch_market_data method
```

## 🛠️ Configuration

### Environment Variables
- `ODDS_API_KEY`: Your the-odds-api.com API key
- `DATA_RETENTION_DAYS`: How long to keep historical data (default: 30)
- `COLLECTION_INTERVAL_MINUTES`: How often to collect data (default: 15)

### File Structure
```
data/odds_collection/
├── raw_odds_history.json        # Raw collected data
├── processed_odds_history.parquet # Processed data with features
├── ml_training_dataset.parquet   # ML-ready dataset
├── nba_odds_YYYYMMDD_HHMMSS.csv  # Export files
├── nba_odds_YYYYMMDD_HHMMSS.json
├── nba_odds_YYYYMMDD_HHMMSS.parquet
└── ml_dataset_report.json       # Collection summary
```

## 🔧 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - The system includes caching to minimize API calls
   - Default cache duration: 5 minutes
   - Monitor API usage in the-odds-api.com dashboard

2. **Data Quality Issues**
   - Run validation to identify specific problems
   - Check API response format changes
   - Verify team name mappings

3. **Missing Markets**
   - Not all bookmakers offer all markets
   - Some games may lack spreads or totals data
   - This is normal and handled by the system

4. **Collection Failures**
   - Check internet connection
   - Verify API key validity
   - Monitor API status page

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
extractor = ComprehensiveOddsExtractor()
data = extractor.get_all_markets_odds()
```

## 📋 Example Output

### Console Output
```
2025-11-10 14:25:30 - INFO - Collecting odds data from all markets...
2025-11-10 14:25:32 - INFO - Fetching h2h market data...
2025-11-10 14:25:33 - INFO - Fetching spreads market data...
2025-11-10 14:25:34 - INFO - Fetching totals market data...
2025-11-10 14:25:35 - INFO - Collected odds for 8 games
2025-11-10 14:25:35 - INFO - Data exported to CSV: data/odds_collection/nba_odds_20251110_142535.csv
```

### Sample Data
```json
{
  "game_id": "LA Lakers_vs_GS Warriors",
  "home_team": "LA Lakers",
  "away_team": "GS Warriors", 
  "commence_time": "2025-11-10T20:30:00Z",
  "h2h_home_odds": 2.10,
  "h2h_away_odds": 1.75,
  "spread_home_line": -3.5,
  "spread_home_odds": 1.91,
  "spread_away_line": 3.5,
  "spread_away_odds": 1.94,
  "totals_line": 225.5,
  "over_odds": 1.87,
  "under_odds": 1.98,
  "collected_at": "2025-11-10T14:25:35Z"
}
```

## 🎯 Next Steps for ML Implementation

1. **Add Target Variables**: Integrate with game results API to add actual outcomes
2. **Feature Engineering**: Create additional features based on your specific ML needs
3. **Model Training**: Use the exported datasets for training your ML models
4. **Continuous Learning**: Set up automated collection for model retraining
5. **Model Evaluation**: Implement prediction accuracy tracking

## 📞 Support

For issues or questions:
1. Check the validation output for specific error messages
2. Review API documentation at the-odds-api.com
3. Examine the demo script for usage examples
4. Check the existing project structure for integration examples

## 🏆 Success Metrics

Track these metrics to ensure system success:
- **Data Quality Score**: Target >0.8 for ML readiness
- **Collection Rate**: >95% successful collections
- **Market Coverage**: >80% of games with complete data
- **Data Freshness**: <1 hour old for live ML applications
- **Storage Efficiency**: Compressed formats for large datasets

---

This system provides a complete solution for NBA odds data extraction and preparation for machine learning. The modular design allows for easy customization and extension for specific ML requirements.