"""
Bank Customer Churn Prediction - Solution Implementation
Following plan.md exactly with no deviations
"""

# Step 1: Environment Setup and Data Loading
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data from Workflow1 public folder
train_df = pd.read_csv('../Workflow1/public/train.csv')
test_df = pd.read_csv('../Workflow1/public/test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Step 2: Exploratory Data Analysis
# Check target distribution
target_distribution = train_df['Exited'].value_counts(normalize=True)
print(f"\nTarget distribution:")
print(f"Class 0 (retained): {target_distribution[0]:.1%}")
print(f"Class 1 (churned): {target_distribution[1]:.1%}")

# Identify feature types
categorical_features = ['Geography', 'Gender']
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
binary_features = ['HasCrCard', 'IsActiveMember', 'NumOfProducts']
identifier_columns = ['CustomerId', 'Surname']

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")
print(f"Binary features: {binary_features}")
print(f"Identifier columns to drop: {identifier_columns}")

# Check for missing values
print(f"\nMissing values: {train_df.isnull().sum().sum()}")

# Step 3: Data Preprocessing
# Drop non-predictive columns
train_df = train_df.drop(columns=['CustomerId', 'Surname'])
test_df = test_df.drop(columns=['CustomerId', 'Surname'])

# Separate features and target
X_train = train_df.drop(columns=['Exited', 'id'])
y_train = train_df['Exited']
X_test = test_df.drop(columns=['id'])
test_ids = test_df['id']

# OneHotEncoder for Geography with drop='first', sparse_output=False
ohe = OneHotEncoder(drop='first', sparse_output=False)
geo_encoded_train = ohe.fit_transform(X_train[['Geography']])
geo_encoded_test = ohe.transform(X_test[['Geography']])

# Add encoded columns back
geo_columns = [f'Geography_{cat}' for cat in ohe.categories_[0][1:]]
X_train_geo = pd.DataFrame(geo_encoded_train, columns=geo_columns, index=X_train.index)
X_test_geo = pd.DataFrame(geo_encoded_test, columns=geo_columns, index=X_test.index)

# LabelEncoder for Gender (Female=0, Male=1)
le = LabelEncoder()
X_train['Gender'] = le.fit_transform(X_train['Gender'])
X_test['Gender'] = le.transform(X_test['Gender'])

# Drop original Geography column and add encoded ones
X_train = X_train.drop(columns=['Geography'])
X_train = pd.concat([X_train, X_train_geo], axis=1)

X_test = X_test.drop(columns=['Geography'])
X_test = pd.concat([X_test, X_test_geo], axis=1)

# StandardScaler for numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print(f"\nPreprocessed X_train shape: {X_train.shape}")

# Step 4: Feature Engineering and Selection
# Create balance_salary_ratio = Balance / (EstimatedSalary + 1)
X_train['balance_salary_ratio'] = X_train['Balance'] / (X_train['EstimatedSalary'] + 1)
X_test['balance_salary_ratio'] = X_test['Balance'] / (X_test['EstimatedSalary'] + 1)

# Create age_tenure_ratio = Age / (Tenure + 1)
X_train['age_tenure_ratio'] = X_train['Age'] / (X_train['Tenure'] + 1)
X_test['age_tenure_ratio'] = X_test['Age'] / (X_test['Tenure'] + 1)

# Create products_active = NumOfProducts * IsActiveMember
X_train['products_active'] = X_train['NumOfProducts'] * X_train['IsActiveMember']
X_test['products_active'] = X_test['NumOfProducts'] * X_test['IsActiveMember']

# Create zero_balance_flag = (Balance == 0).astype(int)
# Note: Balance is already scaled, so check for values close to the scaled zero
balance_mean = 0  # After StandardScaler, mean is 0
balance_std = 1   # After StandardScaler, std is 1
zero_threshold = -balance_mean / balance_std if balance_std != 0 else 0
X_train['zero_balance_flag'] = (X_train['Balance'] < zero_threshold + 0.1).astype(int)
X_test['zero_balance_flag'] = (X_test['Balance'] < zero_threshold + 0.1).astype(int)

# Create balance_volatility = np.std([Balance]) as proxy for account stability
# Since we only have point-in-time balance, use absolute deviation from mean as proxy
X_train['balance_volatility'] = np.abs(X_train['Balance'] - X_train['Balance'].mean())
X_test['balance_volatility'] = np.abs(X_test['Balance'] - X_test['Balance'].mean())

# Create tenure_age_interaction = Tenure * Age / 1000 for lifecycle modeling
X_train['tenure_age_interaction'] = X_train['Tenure'] * X_train['Age'] / 1000
X_test['tenure_age_interaction'] = X_test['Tenure'] * X_test['Age'] / 1000

print(f"Features after engineering: {X_train.shape[1]}")

# Step 5: Model Selection and Training
# Initialize XGBClassifier with specified parameters
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# Initialize RandomForestClassifier with specified parameters
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

# Initialize LogisticRegression with specified parameters
lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

print("\nThree models initialized")

# Step 6: Hyperparameter Optimization
# Define parameter grid for XGBoost only
param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1]
}

# Create GridSearchCV with cv=3, scoring='roc_auc', n_jobs=-1
grid_search = GridSearchCV(
    XGBClassifier(subsample=0.8, colsample_bytree=0.8, random_state=42,
                  eval_metric='logloss', use_label_encoder=False),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

# Fit GridSearchCV
print("\nPerforming grid search on XGBoost...")
grid_search.fit(X_train, y_train)

# Extract best parameters and update XGBoost model
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Update XGBoost model with best parameters
xgb_model = XGBClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# Step 7: Model Training and Validation
# Split training data using train_test_split with test_size=0.2, random_state=42, stratify=y
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTraining set size: {X_train_split.shape[0]}")
print(f"Validation set size: {X_val_split.shape[0]}")

# Train all three models
print("\nTraining models...")
xgb_model.fit(X_train_split, y_train_split)
rf_model.fit(X_train_split, y_train_split)
lr_model.fit(X_train_split, y_train_split)

# Generate predictions and calculate ROC-AUC scores
xgb_pred_proba = xgb_model.predict_proba(X_val_split)[:, 1]
rf_pred_proba = rf_model.predict_proba(X_val_split)[:, 1]
lr_pred_proba = lr_model.predict_proba(X_val_split)[:, 1]

xgb_auc = roc_auc_score(y_val_split, xgb_pred_proba)
rf_auc = roc_auc_score(y_val_split, rf_pred_proba)
lr_auc = roc_auc_score(y_val_split, lr_pred_proba)

# Select best model based on validation AUC
models_scores = {
    'XGBoost': (xgb_model, xgb_auc),
    'RandomForest': (rf_model, rf_auc),
    'LogisticRegression': (lr_model, lr_auc)
}

best_model_name = max(models_scores, key=lambda k: models_scores[k][1])
best_model, best_auc = models_scores[best_model_name]

print(f"\nValidation AUC scores:")
print(f"XGBoost: {xgb_auc:.4f}")
print(f"RandomForest: {rf_auc:.4f}")
print(f"LogisticRegression: {lr_auc:.4f}")
print(f"\nBest model: {best_model_name} with AUC: {best_auc:.4f}")

# Apply probability calibration to best model
print(f"\nApplying isotonic calibration to {best_model_name}...")
calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_val_split, y_val_split)

# Test calibration improvement
calibrated_pred_proba = calibrated_model.predict_proba(X_val_split)[:, 1]
calibrated_auc = roc_auc_score(y_val_split, calibrated_pred_proba)
print(f"Calibrated model AUC: {calibrated_auc:.4f}")

# Use calibrated model as best model
best_model = calibrated_model

# Step 8: Prediction and Submission
# Retrain best model on full training data
print(f"\nRetraining {best_model_name} on full training data...")
best_model.fit(X_train, y_train)

# Generate probability predictions using predict_proba(X_test)[:, 1]
test_predictions = best_model.predict_proba(X_test)[:, 1]

# Create submission DataFrame with columns ['id', 'Exited']
submission = pd.DataFrame({
    'id': test_ids,
    'Exited': test_predictions
})

# Save to 'submission.csv' with index=False
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission shape: {submission.shape}")

# Step 9: Results Analysis and Documentation
# Print validation AUC scores for each model (already done above)
print(f"\n=== Final Results ===")
print(f"Validation AUC scores:")
print(f"XGBoost: {xgb_auc:.4f}")
print(f"RandomForest: {rf_auc:.4f}")
print(f"LogisticRegression: {lr_auc:.4f}")

# Print name of best performing model
print(f"\nBest performing model: {best_model_name}")

# If tree-based model wins, extract feature_importances_
if best_model_name in ['XGBoost', 'RandomForest']:
    feature_names = X_train.columns.tolist()
    # For calibrated models, access the base classifier's feature importances
    if hasattr(best_model, 'calibrated_classifiers_'):
        # CalibratedClassifierCV stores calibrators, get the first one's estimator
        feature_importances = best_model.calibrated_classifiers_[0].estimator.feature_importances_
    else:
        feature_importances = best_model.feature_importances_

    # Get top 5 features
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    print(f"\nTop 5 most important features:")
    for idx, row in feature_importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# Print confirmation
print("\nSubmission saved to submission.csv")
print(f"Final best model validation AUC: {best_auc:.4f}")