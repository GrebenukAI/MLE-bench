# Bank Customer Churn Prediction Challenge

## Competition Description

Customer churn is a critical challenge in the banking industry. When customers leave, banks lose not only revenue but also the investment made in acquiring and maintaining those relationships. This competition challenges you to predict which bank customers are likely to churn (exit) based on their demographic and account information.

Your task is to build a machine learning model that can accurately identify customers at risk of churning, enabling the bank to take proactive retention measures.

## Business Context

Banks spend considerable resources on customer acquisition. Studies show that acquiring a new customer can cost five times more than retaining an existing one. Additionally, increasing customer retention by just 5% can increase profits by 25% to 95%.

By accurately predicting customer churn, banks can:
- Implement targeted retention campaigns
- Offer personalized incentives to at-risk customers
- Improve customer service for high-value customers at risk
- Optimize resource allocation for retention efforts

## Data Description

The dataset contains information about bank customers, including demographic details, account characteristics, and their churn status.

### Features

**Customer Demographics:**
- `CustomerId`: Unique identifier for each customer
- `Surname`: Customer's surname
- `Geography`: Customer's location (country/region)
- `Gender`: Customer's gender
- `Age`: Customer's age in years

**Banking Relationship:**
- `Tenure`: Number of years the customer has been with the bank
- `NumOfProducts`: Number of bank products the customer uses
- `HasCrCard`: Whether the customer has a credit card (1 = Yes, 0 = No)
- `IsActiveMember`: Whether the customer is an active member (1 = Yes, 0 = No)

**Financial Information:**
- `CreditScore`: Customer's credit score
- `Balance`: Customer's account balance
- `EstimatedSalary`: Customer's estimated annual salary

**Target Variable:**
- `Exited`: Whether the customer churned (1 = Yes, 0 = No)

### Files

- `train.csv`: Training data with features and target variable
- `test.csv`: Test data with features only (no target variable)
- `sample_submission.csv`: Sample submission file showing the required format

## Evaluation

Submissions are evaluated using the **ROC-AUC score**. This metric measures the ability of your model to distinguish between customers who will churn and those who won't.

The ROC-AUC score represents the area under the Receiver Operating Characteristic curve:
- A score of 0.5 indicates random predictions (no predictive power)
- A score of 1.0 indicates perfect predictions
- Higher scores indicate better model performance

Your submission should contain probability values between 0 and 1, representing the likelihood that each customer will churn.

## Submission Format

Your submission file should be a CSV with the following format:
```
id,Exited
1,0.75
2,0.23
3,0.91
...
```

Where:
- `id`: Customer identifier from the test set
- `Exited`: Predicted probability of churn (between 0 and 1)

## Dataset Characteristics

- **Training samples:** 13,500
- **Test samples:** 1,500
- **Number of features:** 12 (excluding ID and target)
- **Class distribution:** Approximately 80% retained, 20% churned
- **Missing values:** None

## Important Notes

1. **Class Imbalance:** The dataset is imbalanced with approximately 20% positive class (churned customers). Consider using appropriate techniques to handle this imbalance.

2. **Feature Types:** The dataset contains both numerical and categorical features. Appropriate preprocessing may be required.

3. **Privacy:** Customer IDs and surnames are anonymized and should not be used as predictive features.

4. **Probability Output:** The evaluation metric (ROC-AUC) requires probability predictions, not binary classifications.

## Tips for Success

1. **Feature Engineering:** Consider creating interaction features or deriving new features from existing ones (e.g., balance-to-salary ratio)

2. **Handle Imbalance:** Use techniques like class weighting, SMOTE, or other sampling methods to address class imbalance

3. **Model Selection:** Tree-based models (XGBoost, LightGBM, Random Forest) often perform well on this type of tabular data

4. **Cross-Validation:** Use stratified k-fold cross-validation to ensure robust model evaluation

5. **Feature Importance:** Analyze which features are most predictive of churn to gain business insights

## Baseline Performance

A simple logistic regression model with basic preprocessing typically achieves:
- ROC-AUC: ~0.75-0.80

Top performing models can achieve:
- ROC-AUC: >0.90

Good luck with your predictions!