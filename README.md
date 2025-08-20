# Mail-classification by NLP
- Model that classify weather the e-mail is a spam or not
- This model aims to build a machine learning model that can classify SMS or email messages as Spam or Non-Spam using Natural Language Processing (NLP) techniques. 

## The dataset contains
- Message_body: The raw text message.
- Label: The classification label — 0 for Non-Spam and 1 for Spam.

## preprocessing Steps:
- Lowercasing all text
- Removing punctuation and special characters
- Removing stopwords
- Lemmatization (with SpaCy)
- Removing messages with excessive weird/unreadable characters
- TF-IDF vectorization with: max_features=5000,ngram_range=(1, 2),min_df=5, max_df=0.7

##  Model Metrics

| Metric                        | Value  |
|--------------------------------|-------|
| Best C (Regularization)        | 100    |
| Best Cross-Validation Accuracy | 0.964  |
| Test Accuracy (Final Model)    | 0.864  |

---

## Classification Report

| Class / Metric   | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| 0               | 0.58      | 1.00   | 0.73     | 49      |
| 1               | 1.00      | 0.53   | 0.69     | 76      |
| **Accuracy**    | -         | -      | 0.71     | 125     |
| **Macro Avg**   | 0.79      | 0.76   | 0.71     | -       |
| **Weighted Avg**| 0.83      | 0.71   | 0.71     | -       |


## Observation:
Cross-validation showed high accuracy (~96%), indicating good model generalization.
However, test set accuracy was slightly lower (~86% with tuned hyperparameter), possibly due to class imbalance or limited data.
Model struggles a bit with identifying non-spam (some false positives).

## Future Improvements:
- Handle class imbalance using SMOTE or undersampling
- Try other models: Naive Bayes, SVM, XGBoost
- Deploy as a web app using Streamlit or Flask

