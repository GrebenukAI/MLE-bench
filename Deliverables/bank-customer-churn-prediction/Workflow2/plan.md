# Bank Customer Churn Prediction - Solution Plan

## Competition Overview
Binary classification task to predict customer churn in a banking dataset with imbalanced classes (79.6% retained, 20.4% churned). Evaluation metric is ROC-AUC.

---

## Step 1: Environment Setup and Data Loading

### Actions:
1. Import required libraries: pandas, numpy, sklearn modules (including CalibratedClassifierCV), warnings, xgboost
2. Set random seeds for reproducibility: numpy.random.seed(42) and random.seed(42)
3. Load training data from '../Workflow1/public/train.csv'
4. Load test data from '../Workflow1/public/test.csv'
5. Suppress warnings with warnings.filterwarnings('ignore')

### Expected Output:
- Training data shape: (13572, 14)
- Test data shape: (1428, 13)
- Random seeds set to 42

### Reasoning:
Setting seeds to 42 ensures reproducibility across runs. Loading data from Workflow1 public folder maintains consistency with the MLE-bench converted dataset.

---

## Step 2: Exploratory Data Analysis

### Actions:
1. Check data shapes and column names
2. Identify target column 'Exited' and calculate class distribution
3. Identify categorical features: Geography (3 unique values), Gender (2 unique values)
4. Identify numerical features: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
5. Identify identifier columns: CustomerId, Surname (to be dropped)
6. Check for missing values (none expected)

### Expected Output:
- Target distribution: Class 0 (retained) = 79.6%, Class 1 (churned) = 20.4%
- 2 categorical features identified
- 8 numerical features identified
- 2 identifier columns to drop
- No missing values

### Reasoning:
Understanding the severe class imbalance (80/20) is critical for model selection. Identifying feature types ensures proper preprocessing.

---

## Step 3: Data Preprocessing

### Actions:
1. Drop non-predictive columns: CustomerId and Surname
2. Separate features (X) and target (y) for training data, also drop 'id' column from X_train
3. Apply OneHotEncoder to Geography with parameters: drop='first', sparse_output=False
4. Apply LabelEncoder to Gender (Female=0, Male=1)
5. Apply StandardScaler to numerical features: ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
6. Keep binary features unchanged: HasCrCard, IsActiveMember, NumOfProducts

### Expected Output:
- X_train shape: (13572, 13) after dropping 2 columns and adding 1 from one-hot encoding
- All features scaled/encoded with StandardScaler, OneHotEncoder, and LabelEncoder
- No missing values introduced

### Reasoning:
OneHotEncoder with drop='first' prevents multicollinearity. StandardScaler normalizes features with different scales (e.g., Age ranges 18-92, Balance 0-250000). Binary features don't need scaling.

---

## Step 4: Feature Engineering and Selection

### Actions:
1. Create balance_salary_ratio = Balance / (EstimatedSalary + 1)
2. Create age_tenure_ratio = Age / (Tenure + 1)
3. Create products_active = NumOfProducts * IsActiveMember
4. Create zero_balance_flag = (Balance == 0).astype(int)
5. Create balance_volatility = np.abs(Balance - Balance.mean()) as proxy for account stability
6. Create tenure_age_interaction = Tenure * Age / 1000 for lifecycle modeling
7. Add these 6 new features to the dataset
8. No feature selection - use all 17 features total

### Expected Output:
- 6 new engineered features added
- Final feature count: 17 features
- All ratios handle division by zero with +1 in denominator

### Reasoning:
Zero balance is highly predictive of churn based on winning techniques analysis. Ratio features capture relationships between variables. The +1 prevents division by zero errors.

---

## Step 5: Model Selection and Training

### Actions:
1. Initialize XGBClassifier with parameters:
   - n_estimators=200, max_depth=5, learning_rate=0.05
   - subsample=0.8, colsample_bytree=0.8, random_state=42
   - eval_metric='logloss', use_label_encoder=False

2. Initialize RandomForestClassifier with parameters:
   - n_estimators=150, max_depth=10, min_samples_split=5
   - min_samples_leaf=2, random_state=42, class_weight='balanced'

3. Initialize LogisticRegression with parameters:
   - C=0.1, max_iter=1000, random_state=42, class_weight='balanced'

### Expected Output:
- Three model objects initialized with specified parameters
- No training yet, just initialization

### Reasoning:
XGBoost parameters based on winning solution insights. Class_weight='balanced' helps handle the 80/20 imbalance for RF and LR. Multiple models provide comparison baseline.

---

## Step 6: Hyperparameter Optimization

### Actions:
1. Define parameter grid for XGBoost only:
   - n_estimators: [150, 200]
   - max_depth: [4, 5, 6]
   - learning_rate: [0.05, 0.1]
2. Create GridSearchCV with cv=3, scoring='roc_auc', n_jobs=-1
3. Fit GridSearchCV on training data
4. Extract best parameters
5. Update XGBoost model with best parameters

### Expected Output:
- 12 parameter combinations tested (2×3×2)
- Best parameters identified
- XGBoost model updated with optimal parameters

### Reasoning:
Grid search on XGBoost only because it's the most promising model based on winning techniques. Limited parameter ranges to avoid overfitting with cv=3 for faster execution.

---

## Step 7: Model Training and Validation

### Actions:
1. Split training data using train_test_split:
   - test_size=0.2, random_state=42, stratify=y
2. Train all three models on the training portion
3. Generate predictions on validation set using predict_proba
4. Calculate ROC-AUC score for each model
5. Select the model with highest validation AUC
6. Apply probability calibration using CalibratedClassifierCV with method='isotonic' to best model

### Expected Output:
- Training set: ~10,858 samples
- Validation set: ~2,714 samples
- Three AUC scores calculated
- Best model identified

### Reasoning:
Stratified split maintains class distribution. 20% validation provides sufficient samples for reliable evaluation. ROC-AUC is the competition metric.

---

## Step 8: Prediction and Submission

### Actions:
1. Retrain best model on full training data (all 13,572 samples)
2. Apply identical preprocessing to test data
3. Apply identical feature engineering to test data
4. Generate probability predictions using predict_proba(X_test)[:, 1]
5. Create submission DataFrame with columns ['id', 'Exited']
6. Save to 'submission.csv' with index=False

### Expected Output:
- submission.csv with 1,428 rows
- Probability values between 0 and 1
- Correct column names: 'id' and 'Exited'

### Reasoning:
Retraining on full data maximizes learning. Using [:, 1] extracts positive class probabilities as required for ROC-AUC evaluation.

---

## Step 9: Results Analysis and Documentation

### Actions:
1. Print validation AUC score for each model
2. Print name of best performing model
3. If tree-based model wins, extract feature_importances_
4. Print top 5 most important features
5. Print confirmation: "Submission saved to submission.csv"
6. Print final best model validation AUC

### Expected Output:
- Three AUC scores displayed
- Best model name printed
- Top 5 features listed (if applicable)
- Confirmation message displayed
- Expected AUC range: 0.83-0.87

### Reasoning:
Documentation helps understand model performance and key drivers. Feature importance provides insights into what drives churn predictions.

---

## Summary of Key Parameters (27 total)

1. numpy.random.seed(42)
2. random.seed(42)
3. test_size=0.2
4. stratify=y
5. drop='first'
6. sparse_output=False
7. n_estimators=200 (XGB initial)
8. max_depth=5 (XGB initial)
9. learning_rate=0.05 (XGB initial)
10. subsample=0.8
11. colsample_bytree=0.8
12. eval_metric='logloss'
13. use_label_encoder=False
14. n_estimators=150 (RF)
15. max_depth=10 (RF)
16. min_samples_split=5
17. min_samples_leaf=2
18. class_weight='balanced' (RF)
19. C=0.1
20. max_iter=1000
21. class_weight='balanced' (LR)
22. cv=3
23. scoring='roc_auc'
24. n_jobs=-1
25. index=False (for CSV saving)
26. method='isotonic' (for probability calibration)
27. cv='prefit' (for CalibratedClassifierCV)

All parameters are specified exactly to ensure reproducible implementation without ambiguity.