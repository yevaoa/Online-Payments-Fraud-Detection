# Online Payments Fraud Detection

A machine learning project that detects fraudulent online transactions using a Decision Tree classifier.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) is a synthetic dataset of mobile money transactions with about 6.36 million rows. Only about 0.13% of the transactions are fraud, so the data is highly imbalanced.

| Feature | Description |
|---|---|
| step | Unit of time (1 step = 1 hour) |
| type | Type of transaction |
| amount | Amount of the transaction |
| nameOrig | Customer starting the transaction |
| oldbalanceOrg / newbalanceOrig | Sender balance before / after |
| nameDest | Recipient of the transaction |
| oldbalanceDest / newbalanceDest | Recipient balance before / after |
| isFraud | Target: 1 = fraud, 0 = not fraud |

## Project Steps

1. Downloaded the dataset with `kagglehub` and loaded it with Pandas.
2. Explored the data (shape, data types, missing values, fraud distribution).
3. Visualized the fraud vs. non-fraud split with a pie chart (`plotly.express.pie`).
4. Preprocessed the data (see below).
5. Split the data into 70% training and 30% testing (`train_test_split`, stratified so both sets keep the same fraud ratio).
6. Trained a `DecisionTreeClassifier`.
7. Evaluated the model with `model.score()`, ROC-AUC, and a classification report.
8. Tested the model on a single example with `model.predict()` and saved the result.

## Features Used by the Model

- **Original numeric features:** step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud
- **Transaction type:** one-hot encoded with `pd.get_dummies` (`drop_first=True`)
- **Engineered features:**
  - `transaction_diff` = oldbalanceOrg - newbalanceOrig (how much left the sender's account)
  - `recipient_diff` = oldbalanceDest - newbalanceDest (how much arrived at the recipient)
  - `amount_log` = log(1 + amount) (reduces the effect of very large amounts)
- **Dropped:** `nameOrig` and `nameDest`, because they are unique customer IDs and do not help predict fraud.

## Model

```python
DecisionTreeClassifier(max_depth=5, min_samples_split=10, class_weight='balanced', random_state=42)
```

### Why this training method?

- **`class_weight='balanced'`:** only about 0.13% of transactions are fraud, so a normal model could reach very high accuracy by always predicting "not fraud" while catching no fraud at all. Balanced class weights make the model treat a missed fraud as much more costly, which pushes it to actually find fraud cases.
- **`max_depth=5` and `min_samples_split=10`:** these keep the tree small and simple, which makes it easier to understand and reduces overfitting.
- **Stratified split:** keeps the same small share of fraud in both the training and test sets, so the evaluation is fair.

## Evaluation

Evaluated on the test set of 1,908,786 transactions (2,464 of them fraud).

```
Model Accuracy: 0.9864
ROC-AUC Score: 0.9941

              precision    recall  f1-score   support

           0       1.00      0.99      0.99   1906322
           1       0.08      0.96      0.15      2464

    accuracy                           0.99   1908786
   macro avg       0.54      0.97      0.57   1908786
weighted avg       1.00      0.99      0.99   1908786
```

### Understanding the results

- **Accuracy (0.9864):** about 98.6% of all transactions were classified correctly. Accuracy alone can be misleading on imbalanced data, so the other metrics matter more here.
- **ROC-AUC (0.9941):** the model is very good at ranking fraudulent transactions above normal ones.
- **Fraud recall (0.96):** the model catches about 96% of all real fraud cases. This is the main strength of the model.
- **Fraud precision (0.08):** only about 8% of the transactions flagged as fraud are actually fraud, so the model produces many false alarms.

### Trade-off

The model was set up to prioritize **catching fraud (high recall)** over avoiding false alarms (precision). In fraud detection, a missed fraud is usually more costly than a normal transaction that gets flagged for review, so this trade-off is reasonable when flagged transactions are checked afterwards (for example, by an automated second check or a human reviewer). The downside is the large number of false alarms, which would need to be handled in a real system.

## Example Prediction

The model was tested on one transaction from the test set using `model.predict()`, and the prediction was saved to `prediction_result.txt`.

## Possible Future Improvements

- Try other models such as Random Forest or Gradient Boosting.
- Tune the hyperparameters (tree depth, class weights) to improve precision without losing too much recall.
- Use cross-validation for a more reliable evaluation.

## Libraries

Pandas, NumPy, scikit-learn, Plotly, kagglehub
