"""
Evaluation script for Bank Customer Churn Prediction.
Implements ROC-AUC evaluation metric for binary classification.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Union


class InvalidSubmissionError(Exception):
    """Raised when submission format is invalid."""
    pass


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade Bank Customer Churn Prediction submission using ROC-AUC.

    Args:
        submission: AI's predictions with columns ['id', 'Exited']
                   'Exited' should contain probability values [0, 1]
        answers: Ground truth with columns ['id', 'Exited']
                'Exited' contains actual binary labels (0 or 1)

    Returns:
        float: ROC-AUC score (0.5 = random, 1.0 = perfect)

    Raises:
        InvalidSubmissionError: If submission format is invalid
    """
    # Validate submission format
    if submission is None or answers is None:
        raise InvalidSubmissionError("Submission or answers is None")

    # Check required columns
    required_columns = ['id', 'Exited']
    for df, name in [(submission, "submission"), (answers, "answers")]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise InvalidSubmissionError(
                f"{name} missing required columns: {missing_columns}"
            )

    # Check submission size
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission has {len(submission)} rows but expected {len(answers)}"
        )

    # Check for duplicate IDs in submission
    if submission['id'].duplicated().any():
        duplicated_ids = submission[submission['id'].duplicated()]['id'].tolist()
        raise InvalidSubmissionError(
            f"Submission contains duplicate IDs: {duplicated_ids[:5]}..."
        )

    # Merge submission with answers
    merged = submission.merge(
        answers,
        on='id',
        how='inner',
        suffixes=('_pred', '_true')
    )

    # Check if all IDs matched
    if len(merged) != len(answers):
        missing_ids = set(answers['id']) - set(submission['id'])
        extra_ids = set(submission['id']) - set(answers['id'])
        error_msg = []
        if missing_ids:
            error_msg.append(f"Missing {len(missing_ids)} IDs from submission")
        if extra_ids:
            error_msg.append(f"Submission contains {len(extra_ids)} extra IDs")
        raise InvalidSubmissionError("; ".join(error_msg))

    # Extract predictions and true labels
    y_true = merged['Exited_true'].values
    y_pred = merged['Exited_pred'].values

    # Validate prediction values
    if np.any(np.isnan(y_pred)):
        nan_count = np.sum(np.isnan(y_pred))
        raise InvalidSubmissionError(
            f"Submission contains {nan_count} NaN values in predictions"
        )

    if np.any(np.isinf(y_pred)):
        inf_count = np.sum(np.isinf(y_pred))
        raise InvalidSubmissionError(
            f"Submission contains {inf_count} infinite values in predictions"
        )

    # Check if predictions are in valid range [0, 1]
    # Allow small numerical errors
    epsilon = 1e-6
    if np.any(y_pred < -epsilon) or np.any(y_pred > 1 + epsilon):
        min_val, max_val = np.min(y_pred), np.max(y_pred)
        raise InvalidSubmissionError(
            f"Predictions must be in range [0, 1], got [{min_val:.4f}, {max_val:.4f}]"
        )

    # Clip predictions to [0, 1] to handle numerical errors
    y_pred = np.clip(y_pred, 0, 1)

    # Check if all predictions are the same (no discrimination)
    if len(np.unique(y_pred)) == 1:
        # Return 0.5 (random performance) if all predictions are identical
        return 0.5

    # Check if true labels have both classes
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 1:
        # Edge case: only one class in test set
        # ROC-AUC is undefined, return 0.5
        return 0.5

    # Calculate ROC-AUC score
    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        # Handle any other edge cases
        raise InvalidSubmissionError(f"Error calculating ROC-AUC: {str(e)}")

    # Ensure score is in valid range
    if not (0 <= score <= 1):
        raise InvalidSubmissionError(
            f"ROC-AUC score {score} is outside valid range [0, 1]"
        )

    return float(score)


def validate_submission_format(submission_path: str, test_path: str) -> bool:
    """
    Validate submission file format before grading.

    Args:
        submission_path: Path to submission CSV file
        test_path: Path to test data CSV file

    Returns:
        bool: True if format is valid

    Raises:
        InvalidSubmissionError: If format is invalid
    """
    try:
        # Load files
        submission = pd.read_csv(submission_path)
        test = pd.read_csv(test_path)

        # Check required columns
        if 'id' not in submission.columns or 'Exited' not in submission.columns:
            raise InvalidSubmissionError(
                "Submission must have 'id' and 'Exited' columns"
            )

        # Check row count
        if len(submission) != len(test):
            raise InvalidSubmissionError(
                f"Submission has {len(submission)} rows, expected {len(test)}"
            )

        # Check ID alignment
        test_ids = set(test['id'])
        submission_ids = set(submission['id'])
        if test_ids != submission_ids:
            raise InvalidSubmissionError("Submission IDs don't match test IDs")

        # Check prediction values are numeric
        if not pd.api.types.is_numeric_dtype(submission['Exited']):
            raise InvalidSubmissionError("Predictions must be numeric")

        return True

    except Exception as e:
        raise InvalidSubmissionError(f"Error validating submission: {str(e)}")


if __name__ == "__main__":
    # Test grading function
    import sys

    if len(sys.argv) != 3:
        print("Usage: python grade.py <submission_path> <answers_path>")
        sys.exit(1)

    submission_path = sys.argv[1]
    answers_path = sys.argv[2]

    # Load data
    submission_df = pd.read_csv(submission_path)
    answers_df = pd.read_csv(answers_path)

    # Calculate score
    try:
        score = grade(submission_df, answers_df)
        print(f"ROC-AUC Score: {score:.6f}")
    except InvalidSubmissionError as e:
        print(f"Invalid submission: {e}")
        sys.exit(1)