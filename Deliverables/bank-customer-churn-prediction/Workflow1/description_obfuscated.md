# Binary Classification Task

## Task Description

You are provided with a dataset containing various features about entities in a system. Your task is to predict a binary outcome for each entity based on the provided features.

## Data Description

The dataset contains multiple features that describe different characteristics of each entity.

### Features

The training data includes:
- Numerical features of varying scales
- Categorical features with different cardinalities
- Binary indicator variables
- An ID column for each entity
- The target variable (binary: 0 or 1)

The test data contains the same features except for the target variable, which you need to predict.

### Files

- `train.csv`: Training data with features and target variable
- `test.csv`: Test data with features only (no target variable)
- `sample_submission.csv`: Example submission file showing the required format

## Evaluation

Submissions are evaluated using the **ROC-AUC score**. This metric measures your model's ability to distinguish between the two classes.

Your submission should contain probability values between 0 and 1, representing the likelihood that each entity belongs to the positive class.

## Submission Format

Your submission file should be a CSV with the following format:
```
id,target
1,0.75
2,0.23
3,0.91
...
```

Where:
- `id`: Entity identifier from the test set
- `target`: Predicted probability of positive class (between 0 and 1)

## Dataset Characteristics

- Training samples: 13,500
- Test samples: 1,500
- Number of features: 12 (excluding ID and target)
- Class distribution: Imbalanced (approximately 80/20 split)
- Missing values: None

## Important Notes

1. The dataset exhibits class imbalance. Consider appropriate techniques to handle this.

2. The dataset contains both numerical and categorical features requiring appropriate preprocessing.

3. The evaluation metric requires probability predictions, not binary classifications.

4. Some features may be more predictive than others; feature selection or engineering may improve performance.

## Performance Benchmarks

- Basic models: ROC-AUC ~0.75-0.80
- Advanced models: ROC-AUC >0.90