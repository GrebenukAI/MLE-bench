#!/usr/bin/env python3
"""
Grade Santander Customer Transaction Prediction submissions using AUC-ROC.

Evaluates binary classification predictions against REAL ground truth labels
from the train.csv split (not fabricated data).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


class InvalidSubmissionError(Exception):
    """Raised when submission format is invalid."""
    pass


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade Santander submission using AUC-ROC metric.

    Args:
        submission: AI's predictions with columns ['ID_code', 'target']
                   where target is probability of class 1
        answers: Ground truth with columns ['ID_code', 'target']
                where target is actual binary label (0 or 1)

    Returns:
        float: AUC-ROC score between 0 and 1

    Raises:
        InvalidSubmissionError: If submission format is invalid
    """

    # Validate submission has required columns
    required_columns = ['ID_code', 'target']
    if not all(col in submission.columns for col in required_columns):
        missing = [col for col in required_columns if col not in submission.columns]
        raise InvalidSubmissionError(f"Missing required columns: {missing}")

    # Validate answers has required columns
    if not all(col in answers.columns for col in required_columns):
        raise InvalidSubmissionError(f"Answers file missing required columns")

    # Check submission and answers have same length
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission has {len(submission)} rows but answers has {len(answers)} rows"
        )

    # Sort both by ID to ensure alignment
    submission = submission.sort_values('ID_code').reset_index(drop=True)
    answers = answers.sort_values('ID_code').reset_index(drop=True)

    # Verify IDs match exactly
    if not submission['ID_code'].equals(answers['ID_code']):
        raise InvalidSubmissionError("Submission and answers have mismatched ID_codes")

    # Extract predictions and ground truth
    y_pred = submission['target'].values
    y_true = answers['target'].values

    # Validate prediction values
    if not np.all(np.isfinite(y_pred)):
        raise InvalidSubmissionError("Predictions contain NaN or infinite values")

    # For binary classification, predictions should be probabilities [0, 1]
    # But we'll clip to be safe rather than reject
    y_pred = np.clip(y_pred, 0, 1)

    # Validate ground truth is binary
    unique_labels = np.unique(y_true)
    if not set(unique_labels).issubset({0, 1}):
        raise InvalidSubmissionError(
            f"Ground truth contains non-binary values: {unique_labels}"
        )

    # Check we have both classes in ground truth
    if len(unique_labels) < 2:
        raise InvalidSubmissionError(
            f"Ground truth contains only one class: {unique_labels}"
        )

    # Calculate AUC-ROC
    try:
        auc_score = roc_auc_score(y_true, y_pred)
    except Exception as e:
        raise InvalidSubmissionError(f"Error calculating AUC-ROC: {str(e)}")

    # Validate score is reasonable
    if not 0 <= auc_score <= 1:
        raise InvalidSubmissionError(f"Invalid AUC-ROC score: {auc_score}")

    return float(auc_score)


def test_grader():
    """Test the grading function with sample data."""

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    # Ground truth with ~10% positive class
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

    # Create answers dataframe
    answers = pd.DataFrame({
        'ID_code': [f'test_{i}' for i in range(n_samples)],
        'target': y_true
    })

    # Test 1: Random predictions (should give ~0.5 AUC)
    random_pred = np.random.random(n_samples)
    submission_random = pd.DataFrame({
        'ID_code': [f'test_{i}' for i in range(n_samples)],
        'target': random_pred
    })
    score_random = grade(submission_random, answers)
    print(f"Random predictions AUC: {score_random:.4f} (expected ~0.5)")

    # Test 2: Perfect predictions (should give 1.0 AUC)
    submission_perfect = pd.DataFrame({
        'ID_code': [f'test_{i}' for i in range(n_samples)],
        'target': y_true.astype(float)
    })
    score_perfect = grade(submission_perfect, answers)
    print(f"Perfect predictions AUC: {score_perfect:.4f} (expected 1.0)")

    # Test 3: Inverted predictions (should give ~0.0 AUC)
    submission_inverted = pd.DataFrame({
        'ID_code': [f'test_{i}' for i in range(n_samples)],
        'target': 1 - y_true.astype(float)
    })
    score_inverted = grade(submission_inverted, answers)
    print(f"Inverted predictions AUC: {score_inverted:.4f} (expected ~0.0)")

    print("\nGrader tests completed successfully!")


if __name__ == "__main__":
    test_grader()