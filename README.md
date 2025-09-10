#  Mail Classification by NLP  

This project builds a **machine learning model** that classifies SMS or email messages as **Spam** or **Non-Spam** using **Natural Language Processing (NLP)** techniques.  

---

##  Dataset  

The dataset contains the following columns:  
- **Message_body**: The raw text message  
- **Label**: Classification label — `0` for Non-Spam, `1` for Spam  

---

##  Preprocessing Steps  

1. Lowercasing all text  
2. Removing punctuation and special characters  
3. Removing stopwords  
4. Lemmatization using **SpaCy**  
5. Removing messages with excessive weird/unreadable characters  
6. TF-IDF vectorization with:  
   - `max_features=5000`  
   - `ngram_range=(1,2)`  
   - `min_df=5`  
   - `max_df=0.7`  

---

##  Model Metrics  

| Metric                        | Value  |
|--------------------------------|-------|
| Best C (Regularization)        | 100    |
| Best Cross-Validation Accuracy | 0.964  |
| Test Accuracy (Final Model)    | 0.864  |

---

##  Classification Report  

| Class / Metric   | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| 0 (Non-Spam)    | 0.58      | 1.00   | 0.73     | 49      |
| 1 (Spam)        | 1.00      | 0.53   | 0.69     | 76      |
| **Accuracy**    | -         | -      | 0.71     | 125     |
| **Macro Avg**   | 0.79      | 0.76   | 0.71     | -       |
| **Weighted Avg**| 0.83      | 0.71   | 0.71     | -       |

---

##  Observations  

- Cross-validation showed high accuracy (~96%), indicating good model generalization.  
- Test set accuracy was slightly lower (~86%) with tuned hyperparameters, possibly due to **class imbalance** or limited data.  
- The model struggles a bit with identifying Non-Spam messages, resulting in some false positives.  

---

##  Future Improvements  

- Handle class imbalance using **SMOTE** or undersampling  
- Experiment with other models: **Naive Bayes, SVM, XGBoost**  
- Deploy as a web app using **Streamlit** or **Flask**  

---

##  Project Structure  

```bash
email-classification-nlp/
├── SMS_train.csv           # Training dataset
├── SMS_test.csv            # Test dataset
├── NLP-mail-classification.py # Notebook with full analysis
└── README.md               # Project documentation
