"""
MLE-bench Grading for Store Sales - Time Series Forecasting
Implements RMSLE (Root Mean Squared Logarithmic Error) evaluation
"""

import pandas as pd
import numpy as np
from typing import Union

# MLE-bench standard error class
class InvalidSubmissionError(Exception):
    """Raised when submission format is invalid"""
    pass


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Logarithmic Error.

    RMSLE = sqrt(mean((log(predicted + 1) - log(actual + 1))^2))

    This metric is less sensitive to large outliers and penalizes
    under-prediction more than over-prediction.

    Args:
        y_true: Ground truth sales values
        y_pred: Predicted sales values

    Returns:
        float: RMSLE score (lower is better)
    """
    # Ensure non-negative predictions (sales cannot be negative)
    y_pred = np.maximum(y_pred, 0)

    # Calculate RMSLE
    log_true = np.log1p(y_true)  # log1p = log(1 + x) for numerical stability
    log_pred = np.log1p(y_pred)

    squared_error = (log_true - log_pred) ** 2
    mean_squared_error = np.mean(squared_error)

    return np.sqrt(mean_squared_error)


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade submission against ground truth answers using RMSLE.

    Args:
        submission: AI's predictions with columns ['id', 'sales']
        answers: Ground truth with columns ['id', 'sales']

    Returns:
        float: RMSLE score (lower is better)

    Raises:
        InvalidSubmissionError: If submission format is invalid
    """

    # 1. Validate submission has required columns
    required_cols = ['id', 'sales']
    if not all(col in submission.columns for col in required_cols):
        missing = [col for col in required_cols if col not in submission.columns]
        raise InvalidSubmissionError(f"Missing required columns: {missing}")

    # 2. Validate answers has required columns
    if not all(col in answers.columns for col in required_cols):
        raise InvalidSubmissionError(f"Answers missing required columns")

    # 3. Check submission has correct number of rows
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission has {len(submission)} rows, expected {len(answers)}"
        )

    # 4. Align submission and answers by ID
    submission = submission.sort_values('id').reset_index(drop=True)
    answers = answers.sort_values('id').reset_index(drop=True)

    # 5. Verify IDs match exactly
    if not submission['id'].equals(answers['id']):
        raise InvalidSubmissionError(
            "Submission IDs do not match expected IDs"
        )

    # 6. Extract predictions and ground truth
    y_pred = submission['sales'].values
    y_true = answers['sales'].values

    # 7. Validate predictions are numeric
    if not np.issubdtype(y_pred.dtype, np.number):
        raise InvalidSubmissionError(
            f"Predictions must be numeric, got dtype: {y_pred.dtype}"
        )

    # 8. Check for NaN values
    if np.any(np.isnan(y_pred)):
        n_nan = np.sum(np.isnan(y_pred))
        raise InvalidSubmissionError(
            f"Submission contains {n_nan} NaN values"
        )

    # 9. Check for negative sales (warning, but allow with clipping)
    if np.any(y_pred < 0):
        n_negative = np.sum(y_pred < 0)
        print(f"Warning: {n_negative} negative predictions will be clipped to 0")

    # 10. Calculate and return RMSLE score
    score = rmsle(y_true, y_pred)

    # Provide feedback based on score
    if score < 0.4:
        print(f"Excellent! RMSLE = {score:.6f} (top-tier performance)")
    elif score < 0.5:
        print(f"Good! RMSLE = {score:.6f} (competitive performance)")
    elif score < 0.8:
        print(f"Decent. RMSLE = {score:.6f} (room for improvement)")
    else:
        print(f"Baseline. RMSLE = {score:.6f} (significant improvement needed)")

    return score


if __name__ == "__main__":
    # Testing code
    import sys

    if len(sys.argv) != 3:
        print("Usage: python grade.py <submission.csv> <answers.csv>")
        sys.exit(1)

    # Load files
    submission_path = sys.argv[1]
    answers_path = sys.argv[2]

    try:
        submission_df = pd.read_csv(submission_path)
        answers_df = pd.read_csv(answers_path)

        # Calculate score
        score = grade(submission_df, answers_df)
        print(f"\nFinal RMSLE Score: {score:.6f}")

    except InvalidSubmissionError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)