# Binary Classification Task

## Task Description

Given a dataset with anonymized numeric features, predict a binary target variable for each instance in the test set.

## Data Structure

### Input Features
- 200 continuous numeric features (normalized)
- All features are anonymized with generic names
- No missing values in the dataset

### Target Variable
- Binary classification target (0 or 1)
- Imbalanced classes with minority class ~10%

### Data Format
- Training data: Features with corresponding binary labels
- Test data: Features only (predict the labels)
- Each instance has a unique identifier

## Evaluation Metric

Predictions are evaluated using the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

The AUC-ROC score measures the model's ability to distinguish between the two classes across all possible classification thresholds. A score of 0.5 indicates random guessing, while 1.0 indicates perfect classification.

## Submission Format

Submit predictions as probabilities (between 0 and 1) for the positive class. The submission file should contain:
- Column 1: Instance identifier
- Column 2: Predicted probability

## Task Characteristics

- High-dimensional feature space
- Severe class imbalance
- No domain knowledge applicable (anonymized features)
- Feature interactions likely important
- Robust validation strategy required

## Objective

Develop a machine learning model that can accurately predict the binary target variable based on the provided anonymized features, with performance measured by AUC-ROC score.