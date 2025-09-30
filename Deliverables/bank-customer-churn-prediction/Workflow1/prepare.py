"""
Data preparation script for Bank Customer Churn Prediction.
CRITICAL: Splits by CUSTOMER ID to prevent data leakage when customers have multiple rows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare Bank Customer Churn data for MLE-bench format.

    CRITICAL DATA LEAKAGE PREVENTION:
    - Check if CustomerId has duplicates (same customer multiple rows)
    - If duplicates exist: Split by CustomerId NOT by row
    - Ensure NO customer appears in both train and test

    Args:
        raw: Path to original competition data (Controls folder)
        public: Path for public files (visible during training)
        private: Path for private answers (used for grading)
    """
    logger.info("Starting Bank Customer Churn data preparation...")

    # CRITICAL: Load ONLY Kaggle's train.csv (which has labels)
    # IGNORE Kaggle's test.csv (no labels - useless for MLE-bench)
    train_path = raw / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    logger.info(f"Loading training data from {train_path}")
    full_train = pd.read_csv(train_path)

    # Validate data
    logger.info(f"Loaded {len(full_train)} samples with {len(full_train.columns)} columns")

    # Check for target column
    if 'Exited' not in full_train.columns:
        raise ValueError("Target column 'Exited' not found in training data")

    # Check class distribution
    class_dist = full_train['Exited'].value_counts(normalize=True)
    logger.info(f"Class distribution - Retained: {class_dist[0]:.1%}, Churned: {class_dist[1]:.1%}")

    # CRITICAL LEAKAGE CHECK: Multiple rows per customer?
    if 'CustomerId' in full_train.columns:
        unique_customers = full_train['CustomerId'].nunique()
        total_rows = len(full_train)

        if unique_customers < total_rows:
            # LEAKAGE RISK: Same customer appears multiple times
            logger.warning("=" * 60)
            logger.warning(f"DATA LEAKAGE RISK DETECTED!")
            logger.warning(f"  - {total_rows} total rows")
            logger.warning(f"  - {unique_customers} unique customers")
            logger.warning(f"  - Average {total_rows/unique_customers:.1f} rows per customer")
            logger.warning("USING CUSTOMER-LEVEL SPLITTING TO PREVENT LEAKAGE")
            logger.warning("=" * 60)

            # Get unique customer IDs
            customer_ids = full_train['CustomerId'].unique()

            # Split CUSTOMERS (not rows) to prevent leakage
            train_customers, test_customers = train_test_split(
                customer_ids,
                test_size=0.1,  # 10% of CUSTOMERS for testing
                random_state=42
            )

            # Select ALL rows for each customer group
            train_mask = full_train['CustomerId'].isin(train_customers)
            test_mask = full_train['CustomerId'].isin(test_customers)

            new_train = full_train[train_mask].copy()
            new_test = full_train[test_mask].copy()

            # VERIFY no customer overlap
            train_customer_set = set(new_train['CustomerId'].unique())
            test_customer_set = set(new_test['CustomerId'].unique())
            overlap = train_customer_set.intersection(test_customer_set)

            if overlap:
                raise ValueError(f"CRITICAL DATA LEAKAGE: {len(overlap)} customers appear in both train and test!")

            logger.info(f"✓ NO CUSTOMER OVERLAP VERIFIED:")
            logger.info(f"  - {len(train_customers)} unique customers in train")
            logger.info(f"  - {len(test_customers)} unique customers in test")
            logger.info(f"  - {len(new_train)} total rows in train")
            logger.info(f"  - {len(new_test)} total rows in test")

            # Separate features and target for saving
            X_train = new_train.drop('Exited', axis=1)
            y_train = new_train['Exited']
            X_test = new_test.drop('Exited', axis=1)
            y_test = new_test['Exited']

        else:
            # No duplicate customers - standard stratified split is safe
            logger.info("No duplicate customers detected - using standard stratified split...")
            X = full_train.drop('Exited', axis=1)
            y = full_train['Exited']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.1,
                random_state=42,
                stratify=y  # Maintain class balance
            )

            # Recombine for statistics
            new_train = pd.concat([X_train, y_train], axis=1)
            new_test = pd.concat([X_test, y_test], axis=1)

    else:
        # No CustomerId column - use standard split
        logger.info("No CustomerId column found - using standard stratified split...")
        X = full_train.drop('Exited', axis=1)
        y = full_train['Exited']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.1,
            random_state=42,
            stratify=y  # Maintain class balance
        )

        # Recombine for statistics
        new_train = pd.concat([X_train, y_train], axis=1)
        new_test = pd.concat([X_test, y_test], axis=1)

    # Verify split maintains class distribution
    train_dist = new_train['Exited'].value_counts(normalize=True)
    test_dist = new_test['Exited'].value_counts(normalize=True)
    logger.info(f"Train class distribution - Retained: {train_dist[0]:.1%}, Churned: {train_dist[1]:.1%}")
    logger.info(f"Test class distribution - Retained: {test_dist[0]:.1%}, Churned: {test_dist[1]:.1%}")

    # Create directories if they don't exist
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    # Save public files
    logger.info("Saving public files...")

    # 1. Training data (with labels)
    train_file = public / "train.csv"
    new_train.to_csv(train_file, index=False)
    logger.info(f"  - Saved training data: {len(new_train)} samples to {train_file}")

    # 2. Test data (WITHOUT labels - for prediction)
    test_file = public / "test.csv"
    X_test.to_csv(test_file, index=False)
    logger.info(f"  - Saved test data (no labels): {len(X_test)} samples to {test_file}")

    # 3. Sample submission format
    sample_submission = pd.DataFrame({
        'id': X_test['id'],
        'Exited': 0.5  # Baseline prediction (50% probability)
    })
    submission_file = public / "sample_submission.csv"
    sample_submission.to_csv(submission_file, index=False)
    logger.info(f"  - Saved sample submission: {len(sample_submission)} rows to {submission_file}")

    # Save private answers (for grading)
    logger.info("Saving private answers...")
    answers = pd.DataFrame({
        'id': X_test['id'],
        'Exited': y_test
    })
    answers_file = private / "test.csv"
    answers.to_csv(answers_file, index=False)
    logger.info(f"  - Saved private answers: {len(answers)} rows to {answers_file}")

    # Generate data statistics
    logger.info("\nData preparation complete!")
    logger.info("=" * 50)
    logger.info(f"Training samples: {len(new_train)}")
    logger.info(f"Test samples: {len(new_test)}")
    logger.info(f"Features: {len(X_train.columns)}")
    logger.info(f"Target variable: Exited (binary)")
    logger.info("=" * 50)

    # COMPREHENSIVE LEAKAGE VERIFICATION
    logger.info("\nDATA LEAKAGE VERIFICATION:")

    # 1. Check ID overlap
    train_ids = set(new_train['id'])
    test_ids = set(new_test['id'])
    id_overlap = train_ids.intersection(test_ids)
    if id_overlap:
        raise ValueError(f"ID LEAKAGE: {len(id_overlap)} IDs appear in both train and test")
    logger.info("✓ No ID overlap")

    # 2. Check CustomerID overlap (if exists)
    if 'CustomerId' in new_train.columns and 'CustomerId' in new_test.columns:
        train_customers = set(new_train['CustomerId'].unique())
        test_customers = set(new_test['CustomerId'].unique())
        customer_overlap = train_customers.intersection(test_customers)
        if customer_overlap:
            raise ValueError(f"CUSTOMER LEAKAGE: {len(customer_overlap)} customers appear in both train and test!")
        logger.info(f"✓ No CustomerID overlap ({len(train_customers)} train, {len(test_customers)} test)")

    logger.info("✓ ALL LEAKAGE CHECKS PASSED")

    # Calculate file checksums for verification
    def calculate_checksum(file_path):
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    logger.info("\nFile checksums:")
    for file_path in [train_file, test_file, submission_file, answers_file]:
        checksum = calculate_checksum(file_path)
        logger.info(f"  {file_path.name}: {checksum[:16]}...")


if __name__ == "__main__":
    # Test with actual paths
    import sys

    if len(sys.argv) != 4:
        print("Usage: python prepare.py <raw_dir> <public_dir> <private_dir>")
        sys.exit(1)

    raw_dir = Path(sys.argv[1])
    public_dir = Path(sys.argv[2])
    private_dir = Path(sys.argv[3])

    prepare(raw_dir, public_dir, private_dir)