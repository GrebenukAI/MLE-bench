# Evaluation Metric - Bank Customer Churn Prediction

## Primary Metric: ROC-AUC

### Why ROC-AUC?
- **Class Imbalance:** With 20.4% positive class, accuracy alone is misleading
- **Probability Output:** Churn prediction benefits from probability scores
- **Business Value:** Banks need to rank customers by churn likelihood
- **Standard Practice:** ROC-AUC is standard for binary classification in banking

### Metric Definition
ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures the ability of the model to distinguish between churned and retained customers across all probability thresholds.

### Implementation

```python
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def calculate_roc_auc(y_true, y_pred_proba):
    """
    Calculate ROC-AUC score for binary classification.

    Args:
        y_true: Ground truth labels (0/1)
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        float: ROC-AUC score (0.5 = random, 1.0 = perfect)
    """
    try:
        score = roc_auc_score(y_true, y_pred_proba)
        return score
    except ValueError as e:
        # Handle edge cases (all one class)
        if len(np.unique(y_true)) == 1:
            return 0.5  # Random performance if only one class
        raise e

def grade_submission(submission_df, answers_df):
    """
    Grade competition submission using ROC-AUC.

    Args:
        submission_df: DataFrame with columns ['id', 'Exited']
        answers_df: DataFrame with columns ['id', 'Exited']

    Returns:
        float: ROC-AUC score
    """
    # Merge on id to align predictions with answers
    merged = submission_df.merge(answers_df, on='id', suffixes=('_pred', '_true'))

    # Calculate ROC-AUC
    score = calculate_roc_auc(
        merged['Exited_true'],
        merged['Exited_pred']
    )

    return score
```

### Interpretation
- **0.50:** Random guessing (no predictive power)
- **0.70-0.75:** Poor model
- **0.75-0.80:** Fair model
- **0.80-0.85:** Good model
- **0.85-0.90:** Very good model
- **0.90+:** Excellent model

### Expected Performance Ranges
Based on winning solutions:
- **Baseline (Logistic Regression):** 0.75-0.80
- **Tree-based models (untuned):** 0.80-0.83
- **Tuned XGBoost/LightGBM:** 0.85-0.90
- **Best solutions (GA-XGBoost):** 0.95-0.99

## Alternative Metrics

### F1 Score (for discrete predictions)
```python
from sklearn.metrics import f1_score

def calculate_f1(y_true, y_pred, threshold=0.5):
    """
    Calculate F1 score using threshold on probabilities.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_pred_binary)
```

### Balanced Accuracy
```python
from sklearn.metrics import balanced_accuracy_score

def calculate_balanced_accuracy(y_true, y_pred, threshold=0.5):
    """
    Balanced accuracy accounts for class imbalance.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return balanced_accuracy_score(y_true, y_pred_binary)
```

### Average Precision (PR-AUC)
```python
from sklearn.metrics import average_precision_score

def calculate_average_precision(y_true, y_pred_proba):
    """
    Average precision summarizes precision-recall curve.
    Better than ROC-AUC for highly imbalanced datasets.
    """
    return average_precision_score(y_true, y_pred_proba)
```

## Submission Format Validation

```python
def validate_submission(submission_df, test_df):
    """
    Validate submission format before grading.

    Checks:
    1. Required columns exist
    2. All test IDs present
    3. Predictions in valid range
    4. No missing values
    """
    errors = []

    # Check columns
    required_cols = ['id', 'Exited']
    missing_cols = [c for c in required_cols if c not in submission_df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check IDs match
    test_ids = set(test_df['id'])
    submission_ids = set(submission_df['id'])
    if test_ids != submission_ids:
        missing = test_ids - submission_ids
        extra = submission_ids - test_ids
        if missing:
            errors.append(f"Missing {len(missing)} IDs")
        if extra:
            errors.append(f"Extra {len(extra)} IDs")

    # Check prediction range
    if 'Exited' in submission_df.columns:
        pred_min = submission_df['Exited'].min()
        pred_max = submission_df['Exited'].max()
        if pred_min < 0 or pred_max > 1:
            errors.append(f"Predictions outside [0, 1]: [{pred_min}, {pred_max}]")

        # Check for NaN
        if submission_df['Exited'].isna().any():
            errors.append("Predictions contain NaN values")

    if errors:
        raise ValueError("Submission validation failed:\n" + "\n".join(errors))

    return True
```

## MLE-bench Implementation

For the MLE-bench grade.py file:

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade Bank Customer Churn Prediction using ROC-AUC.

    Args:
        submission: AI's predictions with columns ['id', 'Exited']
        answers: Ground truth with columns ['id', 'Exited']

    Returns:
        float: ROC-AUC score (higher is better)
    """
    # Validate submission
    if 'id' not in submission.columns or 'Exited' not in submission.columns:
        raise ValueError("Submission must have 'id' and 'Exited' columns")

    # Merge and align
    merged = submission.merge(answers, on='id', suffixes=('_pred', '_true'))

    if len(merged) != len(answers):
        raise ValueError(f"Submission has {len(submission)} rows, expected {len(answers)}")

    # Calculate ROC-AUC
    score = roc_auc_score(merged['Exited_true'], merged['Exited_pred'])

    return score
```