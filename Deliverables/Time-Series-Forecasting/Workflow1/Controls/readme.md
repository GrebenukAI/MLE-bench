# Store Sales - Time Series Forecasting Controls

## Competition Control Overview
**Competition:** Store Sales - Time Series Forecasting
**Platform:** Kaggle Getting Started Competition
**URL:** https://www.kaggle.com/competitions/store-sales-time-series-forecasting
**Status:** Finished/Educational (Always Open)
**Participants:** Educational competition with continuous participation

## Why This Competition Works as Control:
✅ **Known Outcomes** - Established leaderboard with documented solutions
✅ **Proven Difficulty** - Medium complexity time series problem
✅ **Clear Evaluation** - RMSLE (Root Mean Squared Logarithmic Error)
✅ **Educational Value** - Real-world retail forecasting with multiple data sources

## What We Know:

### Competition Characteristics
- **Task:** Forecast store sales for Corporación Favorita (Ecuador retail chain)
- **Timeline:** Predict 16 days of sales (2017-08-16 to 2017-08-31)
- **Scale:** 54 stores × 33 product families = 1,782 time series
- **Training Data:** 4.5 years of historical sales (2013-01-01 to 2017-08-15)

### Data Structure
- **Main Files:**
  - train.csv: 3,000,888 rows (historical sales)
  - test.csv: 28,512 rows (prediction period)
  - sample_submission.csv: Expected format

- **Supplementary Data:**
  - stores.csv: Store metadata (city, state, type, cluster)
  - oil.csv: Daily oil price (Ecuador is oil-dependent)
  - holidays_events.csv: Holiday and events calendar
  - transactions.csv: Daily transactions per store

### Evaluation Metric
- **RMSLE:** Root Mean Squared Logarithmic Error
- Logarithmic scale reduces impact of large sales outliers
- Symmetric penalty for over/under prediction
- Formula: sqrt(mean((log(predicted + 1) - log(actual + 1))^2))

### Key Challenges
1. **Multiple Seasonalities:** Daily, weekly, monthly, yearly patterns
2. **Holiday Effects:** National, regional, and local holidays impact
3. **Promotions:** On-promotion flag affects sales significantly
4. **Zero Sales:** Many legitimate zero sales days (closed stores, out of stock)
5. **New Products:** Some product families have limited history

## MLE-bench Conversion Suitability

### Meets All 9 Criteria:
1. ✅ **ML Engineering Focus** - Pure time series forecasting problem
2. ✅ **Well-Specified** - Clear prediction task with defined target
3. ✅ **Local Evaluation** - RMSLE computable without external APIs
4. ✅ **Finished Competition** - Educational/always-open status
5. ✅ **Unique Dataset** - Ecuadorian retail data (not overused)
6. ✅ **Same Distribution** - Train/test from same time period continuum
7. ✅ **CSV Submission** - Standard CSV format required
8. ✅ **Self-Contained** - All necessary data provided
9. ✅ **Permissive License** - Public educational competition

### Complexity Assessment
- **Level:** Medium (2-10 hours for experienced ML engineer)
- **Why:** Requires feature engineering, handling multiple seasonalities, and careful validation strategy

## Control Verification
- Competition accessible via API without rules acceptance
- All data files successfully downloaded and verified
- Evaluation metric implementation straightforward
- No external dependencies or API requirements