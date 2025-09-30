# Starting Materials for Store Sales - Time Series Forecasting

## Downloaded Competition Files

### Primary Data Files (Verified)
- **train.csv**: 3,000,888 rows × 6 columns
  - id: unique identifier
  - date: from 2013-01-01 to 2017-08-15
  - store_nbr: store identifier (1-54)
  - family: product family (33 categories)
  - sales: target variable (unit sales)
  - onpromotion: number of items on promotion

- **test.csv**: 28,512 rows × 5 columns
  - Same structure as train minus 'sales' column
  - Date range: 2017-08-16 to 2017-08-31

- **sample_submission.csv**: 28,512 rows × 2 columns
  - id: matches test.csv id
  - sales: predicted sales (to be filled)

### Supplementary Data Files (Verified)
- **stores.csv**: 54 rows × 5 columns
  - store_nbr, city, state, type, cluster

- **oil.csv**: 1,218 rows × 2 columns
  - date, dcoilwtico (daily oil price)

- **holidays_events.csv**: 350 rows × 6 columns
  - date, type, locale, locale_name, description, transferred

- **transactions.csv**: 83,488 rows × 3 columns
  - date, store_nbr, transactions

## File Sizes (Actual from Extraction)
```
train.csv: ~180 MB
test.csv: ~1.5 MB
sample_submission.csv: ~500 KB
stores.csv: ~3 KB
oil.csv: ~20 KB
holidays_events.csv: ~20 KB
transactions.csv: ~2 MB
```

## Data Availability
-  All files successfully downloaded from Kaggle API
-  No authentication issues or rules acceptance required
-  Data extracted from: store-sales-time-series-forecasting.zip
-  Location: ~/workspace/kaggle_data/store-sales-time-series-forecasting/

## Known Data Characteristics (From Analysis)

### Date Coverage
- **Training period**: 1,689 unique dates (2013-01-01 to 2017-08-15)
- **Test period**: 16 days (2017-08-16 to 2017-08-31)
- **Gap**: No gap between train and test (continuous)

### Product Families (33 total)
Categories include: AUTOMOTIVE, BABY CARE, BEAUTY, BEVERAGES, BOOKS, BREAD/BAKERY, CELEBRATION, CLEANING, DAIRY, DELI, EGGS, FROZEN FOODS, GROCERY I, GROCERY II, HARDWARE, HOME AND KITCHEN I, HOME AND KITCHEN II, HOME APPLIANCES, HOME CARE, LADIESWEAR, LAWN AND GARDEN, LINGERIE, LIQUOR/WINE/BEER, MAGAZINES, MEATS, PERSONAL CARE, PET SUPPLIES, PLAYERS AND ELECTRONICS, POULTRY, PREPARED FOODS, PRODUCE, SCHOOL AND OFFICE SUPPLIES, SEAFOOD

### Store Distribution
- 54 stores across Ecuador
- Multiple cities and states
- Different store types and clusters

### Data Quality Notes
- Contains legitimate zeros (store closures, out of stock)
- Oil price data has some missing values (weekends/holidays)
- Promotion data is sparse but important
- Transaction data not available for all store-date combinations

## Evaluation Setup

### Metric Implementation
```python
# RMSLE (Root Mean Squared Logarithmic Error)
def rmsle(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))
```

### Submission Format
- CSV file with columns: ['id', 'sales']
- 28,512 predictions required
- Sales values must be non-negative
- No missing values allowed

## Competition Access Status
- **API Access**:  Full access confirmed
- **Data Download**:  Completed successfully
- **Rules Acceptance**: Not required (Getting Started competition)
- **Last Verified**: 2025-09-28 00:42 UTC