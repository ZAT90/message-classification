# Multi-Class Text Classification with Word2Vec

This project uses Word2Vec embeddings and Logistic Regression to classify text messages into two categories: Spam and Ham. The model is capable of handling multi-class classification, meaning it can scale if additional categories are added in the future.


[Overview](#overview)

[Techniques Covered](#techniques-covered)

[Features](#features)

[Usage](#usage)

[Dependencies](#dependencies)

[Results](#results)


## OverView
The goal of this project is to classify text messages into two categories: Spam and Ham using Word2Vec embeddings and Logistic Regression. The model uses the One-vs-Rest strategy for multi-class classification, which allows it to scale to more categories if needed.

## Techniques Covered
- Word2Vec for converting text into document vectors.
- Text Preprocessing: Tokenization, lemmatization, and stopword removal.
- Label Encoding for multi-class classification.
- Logistic Regression for multi-class classification using the One-vs-Rest strategy.
- Model Evaluation using accuracy and classification report.

## Features
- Text Preprocessing: Tokenization, lemmatization, and stopword removal.
- Word2Vec Embeddings: Converts text into document vectors.
- Logistic Regression: Multi-class classifier using the One-vs-Rest strategy.
- Model Evaluation: Accuracy and classification report.

## Usage
- Load and Preprocess Data: The dataset is loaded, cleaned, and preprocessed by tokenizing, lemmatizing, and removing stopwords.
- Generate Document Vectors: The processed text data is transformed into document vectors using Word2Vec embeddings.
- Train the Model: The Logistic Regression model is trained using the document vectors and labels (Spam, Ham).
- Evaluate the Model: After training, the model’s performance is evaluated using accuracy and classification reports.

## Dependencies
```
gensim        # Word2Vec embeddings
nltk          # Text preprocessing
scikit-learn  # Logistic Regression, classification report
pandas        # Data manipulation

```
## Results
- Test Accuracy: The accuracy of the model after training, evaluated on the test dataset.
- Classification Report: Provides insights into precision, recall, F1 score, and model performance on the test data.

### Sample Output

#### Test accuracy
```
Accuracy: 91.39%
```
#### Classification Report
```
Classification Report:
              precision    recall  f1-score   support

        spam       0.91      1.00      0.95       965
         ham       0.98      0.37      0.53       150

    accuracy                           0.91      1115
   macro avg       0.95      0.68      0.74      1115
weighted avg       0.92      0.91      0.90      1115
```

