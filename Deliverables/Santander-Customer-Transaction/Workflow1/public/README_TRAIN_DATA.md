# Training Data Note

The `train.csv` file (260MB) has been excluded from this repository due to GitHub's file size limitations.

## To Regenerate:
Run `prepare.py` with the original Kaggle data to recreate all public and private files:
```bash
python prepare.py
```

This will split the original Santander train.csv (200,000 rows) into:
- public/train.csv: 180,000 rows with labels
- public/test.csv: 20,000 rows without labels
- private/test.csv: 20,000 rows with answers

The data split uses random_seed=42 for reproducibility.