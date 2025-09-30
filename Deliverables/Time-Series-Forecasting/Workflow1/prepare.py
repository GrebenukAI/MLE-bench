"""
MLE-bench Data Preparation for Store Sales - Time Series Forecasting
Implements temporal splitting to prevent data leakage in time series
"""

from pathlib import Path
import pandas as pd
import shutil
from typing import Tuple

def prepare(raw: Path, public: Path, private: Path):
    """
    Prepare Store Sales time series data with proper temporal splitting.

    CRITICAL: Time series must use temporal split (not random) to prevent leakage
    - Training: Earlier 90% of dates
    - Test: Later 10% of dates (no future data in training)

    Args:
        raw: Path to original Kaggle competition data
        public: Path for public training/test data (visible to AI)
        private: Path for private answers (hidden from AI)
    """

    # 1. Load main training data
    print("Loading training data...")
    train_df = pd.read_csv(raw / "train.csv")
    print(f"Loaded {len(train_df):,} rows from train.csv")

    # 2. Get unique dates for temporal splitting
    print("\nAnalyzing temporal structure...")
    unique_dates = sorted(train_df['date'].unique())
    n_dates = len(unique_dates)
    print(f"Found {n_dates} unique dates from {unique_dates[0]} to {unique_dates[-1]}")

    # 3. Calculate 90/10 temporal split point
    split_idx = int(n_dates * 0.9)
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]

    print(f"\nTemporal split:")
    print(f"  Train: {len(train_dates)} dates ({train_dates[0]} to {train_dates[-1]})")
    print(f"  Test:  {len(test_dates)} dates ({test_dates[0]} to {test_dates[-1]})")

    # 4. Apply temporal split (no data leakage - test dates strictly after train)
    train_mask = train_df['date'].isin(train_dates)
    new_train = train_df[train_mask].copy()
    new_test = train_df[~train_mask].copy()

    print(f"\nData split results:")
    print(f"  Train: {len(new_train):,} rows ({len(new_train)/len(train_df)*100:.1f}%)")
    print(f"  Test:  {len(new_test):,} rows ({len(new_test)/len(train_df)*100:.1f}%)")

    # 5. Verify no temporal leakage
    train_max_date = new_train['date'].max()
    test_min_date = new_test['date'].min()
    assert train_max_date < test_min_date, f"Temporal leakage detected! Train ends {train_max_date}, test starts {test_min_date}"
    print(f"\n✓ No temporal leakage: train ends {train_max_date}, test starts {test_min_date}")

    # 6. Create output directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    # 7. Generate public training data (includes all columns)
    print("\nGenerating output files...")
    train_output = new_train[['id', 'date', 'store_nbr', 'family', 'sales', 'onpromotion']]
    train_output.to_csv(public / "train.csv", index=False)
    print(f"  ✓ public/train.csv ({len(train_output):,} rows)")

    # 8. Generate public test data (excludes target 'sales' column)
    test_output = new_test[['id', 'date', 'store_nbr', 'family', 'onpromotion']]
    test_output.to_csv(public / "test.csv", index=False)
    print(f"  ✓ public/test.csv ({len(test_output):,} rows)")

    # 9. Generate sample submission template
    sample_submission = pd.DataFrame({
        'id': new_test['id'].values,
        'sales': 0.0  # Initialize with zeros
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    print(f"  ✓ public/sample_submission.csv ({len(sample_submission):,} rows)")

    # 10. Generate private test answers (ground truth)
    answers = pd.DataFrame({
        'id': new_test['id'].values,
        'sales': new_test['sales'].values
    })
    answers.to_csv(private / "test.csv", index=False)
    print(f"  ✓ private/test.csv ({len(answers):,} rows)")

    # 11. Copy supplementary files to public directory
    print("\nCopying supplementary data files...")
    supplementary_files = ['stores.csv', 'oil.csv', 'holidays_events.csv', 'transactions.csv']

    for filename in supplementary_files:
        src_path = raw / filename
        if src_path.exists():
            shutil.copy2(src_path, public / filename)
            print(f"  ✓ Copied {filename}")
        else:
            print(f"  ⚠ Warning: {filename} not found in raw data")

    # 12. Final validation statistics
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETE")
    print("="*50)
    print(f"Total rows processed: {len(train_df):,}")
    print(f"Training set: {len(new_train):,} rows ({len(new_train)/len(train_df)*100:.1f}%)")
    print(f"Test set: {len(new_test):,} rows ({len(new_test)/len(train_df)*100:.1f}%)")
    print(f"Unique stores: {train_df['store_nbr'].nunique()}")
    print(f"Unique families: {train_df['family'].nunique()}")
    print(f"Date range preserved: {unique_dates[0]} to {unique_dates[-1]}")
    print("\n✓ All files generated successfully with no data leakage!")


if __name__ == "__main__":
    # For testing purposes
    import sys
    if len(sys.argv) != 4:
        print("Usage: python prepare.py <raw_dir> <public_dir> <private_dir>")
        sys.exit(1)

    raw_path = Path(sys.argv[1])
    public_path = Path(sys.argv[2])
    private_path = Path(sys.argv[3])

    prepare(raw_path, public_path, private_path)