# Winning Techniques - Store Sales Time Series Forecasting

## Competition Performance Benchmarks

### Leaderboard Statistics (Educational Competition)
- **Metric**: RMSLE (Root Mean Squared Logarithmic Error)
- **Baseline**: Simple average historical sales typically scores ~0.7-0.9 RMSLE
- **Competitive**: Good solutions achieve ~0.4-0.5 RMSLE
- **Top Performance**: Best approaches reach ~0.3-0.4 RMSLE

## Common Successful Approaches

### 1. Time Series Decomposition
- **Trend**: Overall growth/decline patterns
- **Seasonality**: Weekly (day of week), monthly, yearly patterns
- **Holiday Effects**: Special handling for holidays and events
- **Promotions**: Significant impact on sales when items on promotion

### 2. Feature Engineering Patterns
**Temporal Features**:
- Day of week, month, year
- Days to/from holidays
- Weekend indicators
- Pay day indicators (15th and last day of month in Ecuador)

**Lag Features**:
- Previous day/week/month sales
- Moving averages (7-day, 30-day)
- Exponentially weighted means

**External Data Integration**:
- Oil prices (Ecuador's economy is oil-dependent)
- Holiday types and transferred holiday handling
- Store metadata (location, type, cluster)

### 3. Model Approaches (Typically Used)

**Traditional Time Series**:
- ARIMA/SARIMA for individual series
- Exponential Smoothing (Holt-Winters)
- Prophet for handling multiple seasonalities

**Machine Learning**:
- LightGBM/XGBoost with lag features
- Random Forest for robustness
- Linear models with proper feature engineering

**Ensemble Methods**:
- Weighted average of multiple models
- Stacking different model types
- Store-specific vs global models

## Performance Expectations

### For MLE-bench Conversion
- **Baseline Target**: RMSLE < 0.8 (simple historical average)
- **Bronze Medal**: RMSLE < 0.6 (basic ML with features)
- **Silver Medal**: RMSLE < 0.5 (good feature engineering + ensemble)
- **Gold Medal**: RMSLE < 0.4 (sophisticated approach)

This falls into MLE-bench "Medium Complexity" category (2-10 hours for experienced engineer).
