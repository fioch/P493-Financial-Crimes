###---Benford's Law Example---###
## Detecting Fraudulent Credit Card Transactions ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Load the dataset
df = pd.read_csv('credit_card_transactions.csv')

# Function to extract the first digit
def first_digit(number):
    return int(str(number)[0])

# Apply the function to the 'Amount' column
df['First_Digit'] = df['Amount'].apply(first_digit)

# Count the occurrences of each first digit
first_digit_counts = df['First_Digit'].value_counts().sort_index()

# Calculate the expected distribution according to Benford's Law
benford_distribution = [np.log10(1 + 1/d) for d in range(1, 10)]
expected_proportions = np.array(benford_distribution)
expected_counts = expected_proportions * df.shape[0]  # Scale to the total number of transactions

# Because chi-square requires that the sum of expected frequencies must be equal to the sum of observed frequencies,
# we scale the expected counts accordingly
scale_factor = first_digit_counts.sum() / expected_counts.sum()
expected_counts *= scale_factor

# Perform the chi-square test
chi_square_stat, p_value = chisquare(first_digit_counts, f_exp=expected_counts)

# Check if the p-value is less than the significance level
significance_level = 0.05
if p_value < significance_level:
    print(f"Potential fraud detected (p-value = {p_value})")
else:
    print(f"No significant deviation from Benford's Law (p-value = {p_value})")

# Plot the distribution of first digits
plt.bar(first_digit_counts.index, first_digit_counts.values, label='Observed', alpha=0.7)
plt.plot(range(1, 10), expected_counts, marker='o', color='red', linestyle='--', label='Expected (Benford\'s Law)')
plt.xlabel('First Digit')
plt.ylabel('Frequency')
plt.title('First Digit Distribution')
plt.legend()
plt.show()

# If the p-value is less than the significance level, we detect potential fraud
if p_value < significance_level:
    print(f"Potential fraud detected (p-value = {p_value})")
    # Here we flag the transactions that contribute most to the deviation
    # For simplicity, we flag transactions that don't follow the expected leading digits in Benford's Law
    benford_first_digits = [1, 2, 3]  # More common leading digits according to Benford's Law
    df['Flagged'] = df['First_Digit'].apply(lambda x: 0 if x in benford_first_digits else 1)
else:
    print(f"No significant deviation from Benford's Law (p-value = {p_value})")
    df['Flagged'] = 0  # No transactions are flagged

print(df.head())

#-----------------------------------------------------------#

## Evaluating Benford's Law performance ##
from sklearn.metrics import confusion_matrix, classification_report

# Assuming df['Label'] is 'Fraudulent' or 'Non-Fraudulent'
# and df['Flagged'] is 1 for flagged transactions and 0 for non-flagged
df['Label_Binary'] = df['Label'].apply(lambda x: 1 if x == 'Fraudulent' else 0)

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(df['Label_Binary'], df['Flagged']).ravel()

# Output the results
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")

# Additionally, print out the classification report for precision, recall, f1-score
print(classification_report(df['Label_Binary'], df['Flagged'], target_names=['Non-Fraudulent', 'Fraudulent']))