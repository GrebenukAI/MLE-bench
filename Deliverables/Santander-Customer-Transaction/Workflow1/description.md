# Santander Customer Transaction Prediction

## Competition Description

At Santander our mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.

Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?

In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.

## Data Description

You are provided with an anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column.

The task is to predict the value of target column in the test set.

### File Descriptions

- **train.csv** - the training set containing 200,000 rows
- **test.csv** - the test set containing 200,000 rows (Note: This file is not used in MLE-bench conversion)
- **sample_submission.csv** - a sample submission file in the correct format

### Data Fields

- **ID_code** - unique identifier for each row
- **target** - the binary target variable (1 indicates the customer made the transaction)
- **var_0, var_1, ..., var_199** - 200 numeric feature variables

## Evaluation

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

### Submission File

For each ID_code in the test set, you must predict a probability for the target variable. The file should contain a header and have the following format:

```
ID_code,target
test_0,0.5
test_1,0.5
test_2,0.5
etc.
```

## Important Notes

The dataset is highly imbalanced with approximately 10% positive samples. All features have been anonymized and scaled, making domain knowledge inapplicable. Success in this competition relies heavily on feature engineering and handling class imbalance effectively.

## Prize

This was a featured competition with $65,000 in prizes.

## Competition Timeline

- Start Date: February 13, 2019
- Final Submission Deadline: May 14, 2019

---

*Note for MLE-bench: This conversion uses only the train.csv file with real labels. The original competition's test.csv is not used as it contains no labels. A 90/10 stratified split of train.csv creates the evaluation framework.*