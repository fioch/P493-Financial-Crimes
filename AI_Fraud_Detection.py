###---Machine Learning Example---###
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read in data file
data = pd.read_csv('synthetic_fraud_data.csv')

## Data Preparation (cleaning, formatting)
# Preprocessing steps
numeric_features = ['Amount', 'Time', 'PreviousTransactions', 'AccountAge', 'TransactionFrequency']
categorical_features = ['Location', 'DeviceType', 'TransactionType']

# Column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

## Training the algorithm to learn the patterns in the data
# Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])

# Train-test split
X = data.drop(columns=['Fraud'])
y = data['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Remove 'AccountNumber' from training features
X_train = X_train.drop(columns=['AccountNumber'])
X_test_with_acc = X_test.copy()  # Keep a copy with 'AccountNumber' for reference
X_test = X_test.drop(columns=['AccountNumber'])

# Training the model
pipeline.fit(X_train, y_train)

## Test how well the algorithm does on predicting fraud 
# Predictions
y_pred = pipeline.predict(X_test)

# Retrieve account numbers from the test set
account_numbers_test = X_test_with_acc['AccountNumber'].values

# Print out each prediction with the account number
for acc_num, prediction in zip(account_numbers_test, y_pred):
    print(f"Account Number: {acc_num} - Prediction: {'Fraud' if prediction == 1 else 'Non-Fraud'}")

# You might also want to print out the actual label to compare the predictions with the truth
y_test_values = y_test.values
for acc_num, prediction, actual in zip(account_numbers_test, y_pred, y_test_values):
    print(f"Account Number: {acc_num} - Prediction: {'Fraud' if prediction == 1 else 'Non-Fraud'} - Actual: {'Fraud' if actual == 1 else 'Non-Fraud'}")

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Output metrics results
print(f"Accuracy of the logistic regression classifier: {accuracy * 100:.2f}%")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")

