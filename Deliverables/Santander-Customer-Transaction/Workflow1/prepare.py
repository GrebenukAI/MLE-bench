#!/usr/bin/env python3
"""
Prepare Santander Customer Transaction Prediction data for MLE-bench.

CORRECT APPROACH:
1. Use ONLY train.csv (200,000 rows WITH labels)
2. Split it 90/10 for train/test sets
3. NEVER touch Kaggle's test.csv (has no labels)

This script:
- Loads Kaggle's train.csv which has the 'target' column
- Creates a stratified 90/10 split maintaining class balance
- Saves public train/test and private answers for MLE-bench evaluation
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import hashlib


def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare Santander Customer Transaction data for MLE-bench format.

    CRITICAL: Uses ONLY train.csv, ignores test.csv completely.

    Args:
        raw: Directory containing original competition files
        public: Output directory for public data (train/test without answers)
        private: Output directory for private data (test answers for grading)
    """
    # Create directories
    Path(public).mkdir(parents=True, exist_ok=True)
    Path(private).mkdir(parents=True, exist_ok=True)

    # CRITICAL: Load ONLY train.csv - it has the labels we need
    print("Loading Kaggle train.csv (the ONLY file with labels)...")
    train_path = os.path.join(raw, 'train.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")

    train_data = pd.read_csv(train_path)
    print(f"Loaded train.csv: {train_data.shape}")

    # Verify target column exists
    if 'target' not in train_data.columns:
        raise ValueError("ABORT: No 'target' column in train.csv - cannot create MLE-bench task")

    # Check class distribution
    target_dist = train_data['target'].value_counts()
    print(f"Target distribution in original data:")
    print(f"  Class 0: {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(train_data)*100:.2f}%)")
    print(f"  Class 1: {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(train_data)*100:.2f}%)")

    # Separate features and target
    X = train_data.drop('target', axis=1)
    y = train_data['target']

    # Create stratified train/test split (90/10)
    print("\nCreating 90/10 stratified split from train.csv...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,  # 10% for testing
        random_state=42,
        stratify=y  # Maintain class balance
    )

    print(f"Split sizes:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")

    # Verify stratification worked
    test_dist = y_test.value_counts()
    print(f"\nTarget distribution in test set:")
    print(f"  Class 0: {test_dist.get(0, 0)} ({test_dist.get(0, 0)/len(y_test)*100:.2f}%)")
    print(f"  Class 1: {test_dist.get(1, 0)} ({test_dist.get(1, 0)/len(y_test)*100:.2f}%)")

    # Save public training data (WITH labels for training)
    print("\nSaving public data...")
    train_with_labels = pd.concat([X_train, y_train], axis=1)
    train_with_labels.to_csv(os.path.join(public, 'train.csv'), index=False)
    print(f"  Saved public/train.csv: {train_with_labels.shape}")

    # Save public test data (WITHOUT labels for prediction)
    X_test.to_csv(os.path.join(public, 'test.csv'), index=False)
    print(f"  Saved public/test.csv: {X_test.shape}")

    # Create sample submission format
    sample_submission = pd.DataFrame({
        'ID_code': X_test['ID_code'],
        'target': 0  # Default predictions (all zeros)
    })
    sample_submission.to_csv(os.path.join(public, 'sample_submission.csv'), index=False)
    print(f"  Saved public/sample_submission.csv: {sample_submission.shape}")

    # Save private test answers (for grading)
    print("\nSaving private data...")
    test_answers = pd.DataFrame({
        'ID_code': X_test['ID_code'].values,
        'target': y_test.values
    })
    test_answers.to_csv(os.path.join(private, 'test.csv'), index=False)
    print(f"  Saved private/test.csv: {test_answers.shape}")

    print("\nData preparation complete!")
    print("=" * 50)
    print("VERIFICATION: Using REAL labels from train.csv")
    print("NO FABRICATED DATA - all labels are genuine")
    print("=" * 50)


if __name__ == "__main__":
    # Test with actual paths
    raw_path = Path("/home/runner/workspace/kaggle_data/santander-customer-transaction-prediction")
    public_path = Path("/home/runner/workspace/Deliverables/Santander-Customer-Transaction/Workflow1/public")
    private_path = Path("/home/runner/workspace/Deliverables/Santander-Customer-Transaction/Workflow1/private")

    prepare(raw_path, public_path, private_path)