# Baseline Approach - Bank Customer Churn Prediction

## Simple Baseline: Logistic Regression

### Expected Performance
- **ROC-AUC:** 0.75-0.80
- **Time to train:** < 1 minute
- **Complexity:** Low

### Implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Prepare features
def prepare_features(df):
    """Basic feature preparation."""
    df = df.copy()

    # Drop non-predictive columns
    drop_cols = ['id', 'CustomerId', 'Surname']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categorical variables
    if 'Geography' in df.columns:
        df = pd.get_dummies(df, columns=['Geography'], prefix='geo')

    if 'Gender' in df.columns:
        df['Gender'] = (df['Gender'] == 'Male').astype(int)

    return df

# Prepare training data
X = prepare_features(train_df.drop('Exited', axis=1))
y = train_df['Exited']

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train logistic regression
lr = LogisticRegression(
    class_weight='balanced',  # Handle imbalance
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_scaled, y_train)

# Validate
val_pred = lr.predict_proba(X_val_scaled)[:, 1]
val_score = roc_auc_score(y_val, val_pred)
print(f"Validation ROC-AUC: {val_score:.4f}")

# Prepare submission
X_test = prepare_features(test_df)
X_test_scaled = scaler.transform(X_test)
test_pred = lr.predict_proba(X_test_scaled)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Exited': test_pred
})
submission.to_csv('submission_baseline.csv', index=False)
```

## Intermediate Baseline: Random Forest

### Expected Performance
- **ROC-AUC:** 0.82-0.85
- **Time to train:** 2-5 minutes
- **Complexity:** Medium

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier

# Using same preprocessing as above
X_train_scaled, X_val_scaled, y_train, y_val = ... # from above

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# Validate
val_pred = rf.predict_proba(X_val_scaled)[:, 1]
val_score = roc_auc_score(y_val, val_pred)
print(f"Validation ROC-AUC: {val_score:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10))
```

## Advanced Baseline: XGBoost

### Expected Performance
- **ROC-AUC:** 0.85-0.88
- **Time to train:** 5-10 minutes
- **Complexity:** High

### Implementation

```python
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

# Feature engineering
def engineer_features(df):
    """Add engineered features."""
    df = df.copy()

    # Balance-based features
    df['has_zero_balance'] = (df['Balance'] == 0).astype(int)
    df['balance_salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

    # Age groups
    df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=False)

    # Activity score
    if 'NumOfProducts' in df.columns and 'IsActiveMember' in df.columns:
        df['activity_score'] = df['NumOfProducts'] * df['IsActiveMember']

    # Tenure groups
    df['tenure_group'] = pd.cut(df['Tenure'], bins=[-1, 2, 5, 8, 11], labels=False)

    return df

# Prepare data with engineering
train_df_eng = engineer_features(train_df)
test_df_eng = engineer_features(test_df)

X = prepare_features(train_df_eng.drop('Exited', axis=1))
y = train_df_eng['Exited']

# Cross-validation with XGBoost
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)

    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=len(y_train_fold) / sum(y_train_fold) - 1,
        random_state=42
    )

    xgb_model.fit(
        X_train_scaled, y_train_fold,
        eval_set=[(X_val_scaled, y_val_fold)],
        early_stopping_rounds=50,
        verbose=False
    )

    # Validate
    val_pred = xgb_model.predict_proba(X_val_scaled)[:, 1]
    val_score = roc_auc_score(y_val_fold, val_pred)
    val_scores.append(val_score)
    print(f"Fold {fold+1} ROC-AUC: {val_score:.4f}")

print(f"\nMean CV ROC-AUC: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")

# Final model on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=len(y) / sum(y) - 1,
    random_state=42
)
final_model.fit(X_scaled, y)

# Prepare submission
X_test = prepare_features(test_df_eng)
X_test_scaled = scaler.transform(X_test)
test_pred = final_model.predict_proba(X_test_scaled)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Exited': test_pred
})
submission.to_csv('submission_xgboost.csv', index=False)
```

## Quick Start Checklist

### 1. Data Loading
```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')
```

### 2. Basic EDA
```python
print(f"Train shape: {train.shape}")
print(f"Class distribution: {train['Exited'].value_counts(normalize=True)}")
print(f"Missing values: {train.isnull().sum().sum()}")
```

### 3. Feature Preparation
- Drop: id, CustomerId, Surname
- Encode: Geography (one-hot), Gender (binary)
- Scale: All numerical features

### 4. Handle Class Imbalance
- Use `class_weight='balanced'`
- Or SMOTE/SMOTEENN for sampling
- Or adjust `scale_pos_weight` in XGBoost

### 5. Validation Strategy
- Stratified K-Fold (k=5)
- Maintain class distribution
- Use ROC-AUC for evaluation

### 6. Submission Format
```python
submission = pd.DataFrame({
    'id': test['id'],
    'Exited': predictions  # Probabilities [0, 1]
})
```

## Common Pitfalls to Avoid

1. **Not handling class imbalance** - 20% positive class needs attention
2. **Using accuracy as metric** - Misleading with imbalanced data
3. **Forgetting to scale features** - Important for linear models
4. **Not dropping CustomerId** - It's unique per row, causes overfitting
5. **Including Surname in features** - High cardinality, no signal
6. **Not using stratified splitting** - Maintains class distribution

## Expected Leaderboard Position

With these baselines:
- **Logistic Regression:** Bottom 50%
- **Random Forest:** Top 50-30%
- **XGBoost (tuned):** Top 30-20%
- **Advanced XGBoost + SMOTEENN:** Top 20-10%
- **GA-optimized XGBoost:** Top 10-5%