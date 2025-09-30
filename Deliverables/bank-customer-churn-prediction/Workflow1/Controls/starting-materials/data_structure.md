# Data Structure Analysis - Bank Customer Churn

## Dataset Overview
- **Training Set:** 15,000 rows × 13 columns
- **Test Set:** 10,000 rows × 12 columns (no target)
- **Target Variable:** Exited (binary: 0/1)
- **Class Distribution:** 79.6% retained (0), 20.4% churned (1)

## Column Descriptions

### Identifiers
| Column | Type | Description | Unique Values | Missing |
|--------|------|-------------|---------------|---------|
| id | int | Row identifier | 15,000 | 0% |
| CustomerId | int | Customer unique ID | 15,000 | 0% |

### Demographic Features
| Column | Type | Description | Range/Values | Missing |
|--------|------|-------------|--------------|---------|
| Surname | string | Customer surname | ~11,000 unique | 0% |
| Geography | string | Country/Region | 3 unique | 0% |
| Gender | string | Customer gender | 2 unique (Male/Female) | 0% |
| Age | int | Customer age | 18-92 years | 0% |

### Banking Relationship
| Column | Type | Description | Range/Values | Missing |
|--------|------|-------------|--------------|---------|
| Tenure | int | Years with bank | 0-10 years | 0% |
| NumOfProducts | int | Number of products | 1-4 products | 0% |
| HasCrCard | float | Has credit card | 0/1 binary | 0% |
| IsActiveMember | float | Active status | 0/1 binary | 0% |

### Financial Features
| Column | Type | Description | Range/Values | Missing |
|--------|------|-------------|--------------|---------|
| CreditScore | int | Credit score | 350-850 | 0% |
| Balance | float | Account balance | 0-250,898 | 0% |
| EstimatedSalary | float | Annual salary | 11-199,992 | 0% |

### Target Variable
| Column | Type | Description | Distribution | Missing |
|--------|------|-------------|--------------|---------|
| Exited | float | Churn indicator | 0: 79.6%, 1: 20.4% | 0% |

## Data Quality Assessment

### Numerical Features Statistics
```python
# CreditScore
Mean: 650.5, Std: 96.7
Min: 350, Q1: 584, Median: 652, Q3: 718, Max: 850

# Age
Mean: 38.9, Std: 10.5
Min: 18, Q1: 32, Median: 37, Q3: 44, Max: 92

# Balance
Mean: 76,486, Std: 62,397
Min: 0, Q1: 0, Median: 97,198, Q3: 127,644, Max: 250,898
Note: ~36% have zero balance

# EstimatedSalary
Mean: 100,090, Std: 57,510
Min: 11.58, Q1: 51,002, Median: 100,194, Q3: 149,388, Max: 199,992
```

### Categorical Features Distribution
```python
# Geography
France: ~50%, Spain: ~25%, Germany: ~25%

# Gender
Male: ~55%, Female: ~45%

# NumOfProducts
1 product: ~51%, 2 products: ~46%, 3 products: ~3%, 4 products: <1%

# HasCrCard
Yes: ~71%, No: ~29%

# IsActiveMember
Yes: ~52%, No: ~48%
```

## Data Type Considerations

### Features to Drop for ML
- `id`: Just row identifier
- `CustomerId`: Unique per customer, no predictive value
- `Surname`: High cardinality text, minimal predictive value

### Encoding Requirements
- **One-Hot Encoding:** Geography (3 categories)
- **Binary Encoding:** Gender (Male=1, Female=0)
- **Already Encoded:** HasCrCard, IsActiveMember

### Scaling Considerations
- **StandardScaler:** CreditScore, Age, EstimatedSalary
- **RobustScaler:** Balance (due to many zeros)
- **No Scaling:** Binary features, NumOfProducts

## Potential Data Issues

### Class Imbalance
- 20.4% positive class (churned)
- Requires balanced sampling or class weights
- Stratified splitting essential

### Zero Inflation
- Balance has 36% zeros (customers with no balance)
- Consider zero-balance flag feature

### Outliers
- Age has some outliers (>80 years)
- Balance has extreme values (top 1% > 200K)

### Correlations
- Balance and NumOfProducts likely correlated
- Age and Tenure may have relationship
- Geography and Balance might interact

## Recommended Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define column groups
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender']
binary_features = ['NumOfProducts', 'HasCrCard', 'IsActiveMember']
drop_features = ['id', 'CustomerId', 'Surname']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('bin', 'passthrough', binary_features)
    ],
    remainder='drop'
)

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier())
])
```

## Data Splitting Strategy
- Use stratified split to maintain 20.4% positive rate
- 90/10 train/test split for MLE-bench
- Set random_state=42 for reproducibility
- No temporal aspect (shuffle is safe)