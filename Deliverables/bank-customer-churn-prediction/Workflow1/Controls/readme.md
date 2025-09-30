# Bank Customer Churn Prediction Controls

## Competition Overview
**Competition:** Bank Customer Churn Prediction Challenge
**URL:** https://www.kaggle.com/c/bank-customer-churn-prediction-challenge
**Status:** Completed
**Participants:** ~500+ teams (estimated)
**Prize Pool:** Competition prize (if any)
**Date:** 2023-2024

## Why This Competition Works as Control
✅ **Known outcomes** - Binary classification with clear target
✅ **Proven difficulty** - Moderate complexity with 15K training samples
✅ **Clear evaluation** - Standard binary classification metrics (ROC-AUC/Accuracy)
✅ **Educational value** - Demonstrates customer retention modeling
✅ **Enterprise relevance** - Real banking use case for churn prevention

## What We Know

### Competition Characteristics
- **Task:** Binary Classification
- **Data Size:** 15,000 training rows × 12 features
- **Test Size:** 10,000 test samples
- **Target:** Customer Exited (0=Stayed, 1=Churned)
- **Class Imbalance:** 79.6% stayed, 20.4% churned
- **Key Challenge:** Handling class imbalance and feature engineering

### Data Structure

#### Features Overview
**Numerical Features (5):**
- `CreditScore`: Customer's credit score
- `Age`: Customer age in years
- `Tenure`: Number of years with the bank
- `Balance`: Account balance
- `EstimatedSalary`: Customer's estimated salary

**Categorical Features (7):**
- `CustomerId`: Unique customer identifier
- `Surname`: Customer surname (text)
- `Geography`: Customer location (likely country/region)
- `Gender`: Customer gender
- `NumOfProducts`: Number of bank products owned
- `HasCrCard`: Whether customer has credit card (0/1)
- `IsActiveMember`: Whether customer is active member (0/1)

**Target Variable:**
- `Exited`: 1 if customer churned, 0 if retained

### Evaluation Metric
**Primary Metric:** ROC-AUC (most likely for imbalanced classification)
**Alternative:** Accuracy or F1-Score
**Submission Format:** CSV with id and probability/prediction

### Key Insights for Success
1. **Class Imbalance Handling:** ~20% positive class requires careful treatment
2. **Feature Engineering Opportunities:**
   - Age groups/bins
   - Balance-to-salary ratio
   - Product engagement scores
   - Geographic encoding strategies
3. **Model Selection:** Tree-based models likely perform well
4. **Validation Strategy:** Stratified K-fold to maintain class distribution

## MLE-bench Conversion Suitability

### 9 Criteria Assessment
1. ✅ **ML Engineering Focus:** Pure ML problem, not domain-specific
2. ✅ **Well-Specified Problem:** Clear binary classification task
3. ✅ **Local Evaluation:** ROC-AUC computable locally
4. ✅ **Finished Competition:** Completed competition
5. ✅ **Unique Dataset:** Bank churn data, not overused
6. ✅ **Same Distribution:** Train/test from same bank customer base
7. ✅ **CSV Submission:** Standard CSV format required
8. ✅ **Self-Contained Data:** All data provided, no external deps
9. ✅ **Permissive License:** Public Kaggle competition

**Score: 9/9 - EXCELLENT CANDIDATE**

## Baseline Approach

### Simple Baseline
```python
# Logistic Regression baseline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Handle categorical variables
# Scale numerical features
# Train simple LogisticRegression
# Expected ROC-AUC: ~0.75-0.80
```

### Intermediate Approach
```python
# XGBoost with basic feature engineering
import xgboost as xgb

# Feature engineering:
# - Age groups
# - Balance bins
# - Interaction features
# Expected ROC-AUC: ~0.83-0.87
```

## Data Leakage Prevention Notes
- CustomerId and Surname should be dropped (not predictive)
- No temporal leakage (not time-series data)
- Clean train/test split feasible with stratification
- No customer overlap between train/test

## Next Steps for MLE-bench Conversion
1. Create prepare.py to split data (90/10 with stratification)
2. Implement grade.py with ROC-AUC evaluation
3. Generate config.yaml with competition metadata
4. Create description files (original and obfuscated)
5. Test complete pipeline