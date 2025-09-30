# Winning Techniques - Bank Customer Churn Prediction

## Top Solution: GA-XGBoost (90% F1, 99% AUC)
**Approach:** Genetic Algorithm optimized XGBoost
**Key Innovation:** SMOTEENN for imbalanced data + GA hyperparameter tuning

### Approach Overview
The winning approach combined XGBoost with genetic algorithm optimization, achieving exceptional results through systematic handling of class imbalance and interpretable modeling.

### Key Techniques

1. **Data Balancing with SMOTEENN**
   - Superior to SMOTE and ADASYN for banking data
   - Combines oversampling minority class with cleaning
   - Removes noisy samples from boundaries
   ```python
   from imblearn.combine import SMOTEENN
   smote_enn = SMOTEENN(random_state=42)
   X_balanced, y_balanced = smote_enn.fit_resample(X_train, y_train)
   ```

2. **Genetic Algorithm Hyperparameter Optimization**
   - Automated search for optimal XGBoost parameters
   - Focused on F1 score maximization
   - Key optimized parameters:
     - max_depth: 5-7
     - learning_rate: 0.05
     - n_estimators: 200-300
     - subsample: 0.8
     - colsample_bytree: 0.8

3. **SHAP Framework for Interpretability**
   - Global feature importance analysis
   - Local prediction explanations
   - Customer-level churn drivers
   ```python
   import shap
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_test)
   ```

### Why It Worked
- SMOTEENN handled class imbalance without creating noise
- GA found optimal hyperparameters beyond grid search
- SHAP provided business-interpretable results

## 2nd Best: Standard XGBoost with Grid Search (85% Accuracy)

### Approach
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False
)

grid_search = GridSearchCV(
    xgb, param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
```

### Best Parameters Found:
- max_depth: 5
- learning_rate: 0.05
- n_estimators: 200
- subsample: 0.8
- colsample_bytree: 0.8

## 3rd Approach: LightGBM (85.3% Accuracy)

### Key Advantages
- Faster training than XGBoost
- Better handling of categorical features
- Lower memory usage

```python
import lightgbm as lgb

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

lgb_train = lgb.Dataset(X_train, y_train)
model = lgb.train(lgb_params, lgb_train, num_boost_round=200)
```

## Common Patterns Among Winners

### Feature Engineering
1. **Most Important Features:**
   - EstimatedSalary (highest importance)
   - CreditScore
   - Balance
   - Age
   - Number of products

2. **Engineered Features:**
   ```python
   # Balance to salary ratio
   df['balance_salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

   # Age groups
   df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100])

   # Zero balance flag
   df['has_zero_balance'] = (df['Balance'] == 0).astype(int)

   # Activity score
   df['activity_score'] = df['NumOfProducts'] * df['IsActiveMember']
   ```

### Validation Strategy
- 5-fold stratified cross-validation
- Maintaining class distribution in each fold
- Early stopping based on validation AUC

### Handling Class Imbalance
1. **Sampling Techniques:**
   - SMOTEENN (best)
   - SMOTE
   - Random oversampling

2. **Class Weight Adjustment:**
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   classes = np.unique(y_train)
   weights = compute_class_weight('balanced', classes=classes, y=y_train)
   class_weight = dict(zip(classes, weights))
   ```

## Code References
- GitHub: KindlyGentleman/DS-DA-Bank-Churn-Prediction
- Kaggle Notebooks:
  - faizanyousafonly/bank-churn-prediction-xgboost-85-6
  - kmalit/bank-customer-churn-prediction
  - anityagangurde/bank-customer-churn-prediction-xgboost-gpu

## Lessons for MLE-bench

### What AI Should Learn
1. **Always handle class imbalance** - 80/20 split requires attention
2. **Feature importance varies** - Salary and credit score dominate
3. **Tree-based models excel** - XGBoost/LightGBM consistently win
4. **Simple features matter** - Zero balance flag is highly predictive
5. **Ensemble rarely needed** - Single well-tuned model sufficient

### Expected Performance Benchmarks
- **Baseline (Logistic Regression):** 75-80% AUC
- **Good (Basic XGBoost):** 83-87% AUC
- **Excellent (Optimized XGBoost):** 90%+ AUC
- **Best (GA-XGBoost + SMOTEENN):** 95-99% AUC

### Key Insights
1. SMOTEENN > SMOTE for banking data
2. Genetic algorithms find better parameters than grid search
3. EstimatedSalary is the strongest predictor
4. Customers with zero balance are high churn risk
5. SHAP values provide actionable business insights