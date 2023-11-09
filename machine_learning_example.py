###---Machine Learning Example---###

## Dataset Creation ##
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Seed for reproducibility
np.random.seed(42)

# Parameters
num_records = 2000
fraud_rate = 0.10
num_fraud = int(num_records * fraud_rate)
num_non_fraud = num_records - num_fraud

# Generate features for non-fraudulent transactions
non_fraud_data = {
    'Amount': np.round(np.random.uniform(10, 1000, num_non_fraud), 2),
    'Time': np.random.normal(loc=50, scale=10, size=num_non_fraud),
    'Location': np.random.choice(['Online', 'Branch', 'ATM'], num_non_fraud),
    'DeviceType': np.random.choice(['Mobile', 'Desktop', 'Tablet'], num_non_fraud),
    'TransactionType': np.random.choice(['Purchase', 'Withdrawal', 'Transfer', 'Deposit'], num_non_fraud),
    'PreviousTransactions': np.random.poisson(5, num_non_fraud),
    'AccountAge': np.random.randint(60, 3650, num_non_fraud),  # In days
    'TransactionFrequency': np.random.randint(1, 50, num_non_fraud)  # Transactions per month
}

# Generate features for fraudulent transactions
fraud_data = {
    'Amount': np.round(np.random.uniform(50, 2000, num_fraud), 2), #Higher amounts on average
    'Time': np.random.normal(loc=70, scale=20, size=num_fraud), # Different distribution
    'Location': np.random.choice(['Online', 'Branch', 'ATM'], num_fraud),
    'DeviceType': np.random.choice(['Mobile', 'Desktop', 'Tablet'], num_fraud),
    'TransactionType': np.random.choice(['Purchase', 'Withdrawal', 'Transfer', 'Desposit'], num_fraud),
    'PreviousTransactions': np.random.poisson(4, num_fraud), # Fewer previous transactions
    'AccountAge': np.random.randint(1, 730, num_fraud),  # In days, newer accounts
    'TransactionFrequency': np.random.randint(5, 100, num_fraud)  # Higher frequency of transactions
}

# Combine the data into a DataFrame
data = pd.DataFrame(non_fraud_data)
data = data.append(pd.DataFrame(fraud_data), ignore_index=True)
data['Fraud'] = np.concatenate([np.zeros(num_non_fraud), np.ones(num_fraud)], axis=0)

# Function to generate unique account numbers
def generate_unique_account_numbers(n):
    generated = set()
    while len(generated) < n:
        # Generate a random number and format it as a 9-digit account number
        account_number = f"{np.random.randint(0, 1_000_000_000):09d}"
        generated.add(account_number)
    return list(generated)

# Generate unique account numbers for all records
account_numbers = generate_unique_account_numbers(num_records)

# Create a DataFrame with the account numbers
account_numbers_df = pd.DataFrame({'AccountNumber': account_numbers})

# Concatenate the account number DataFrame with the original data DataFrame
# Make sure the index of both DataFrames aligns for a correct merge
data = pd.concat([account_numbers_df, data.reset_index(drop=True)], axis=1)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Show the first few rows of the DataFrame
print(data.head())

# Save as CSV file
data.to_csv('synthetic_fraud_data.csv', index=False)

###----------------------------------------------###

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


