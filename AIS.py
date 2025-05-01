# AIS-Based Spam Detection (with Oversampling, using local synthetic dataset)

# Step 1: Install and import libraries
# (Run this in your terminal/Notebook once; remove the '!' if running as a script)


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)

from imblearn.over_sampling import RandomOverSampler
from aisp.nsa import RNSA

# Step 2: Load and inspect YOUR local synthetic dataset
df = pd.read_csv('spam_ham_dataset.csv')
# It already has columns ['label','message'], where label is 'spam' or 'ham'
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("Total messages:", len(df))
print(df['label'].value_counts(), "\n")

# Step 3: Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label_num'].values

# Step 4: Split the data
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set label counts:\n", pd.Series(train_y).value_counts())
print("Test set label counts:\n", pd.Series(test_y).value_counts(), "\n")

# Step 5: Oversample the minority class in training set
ros = RandomOverSampler(random_state=42)
train_x_resampled, train_y_resampled = ros.fit_resample(train_x, train_y)

print("Resampled training set label counts:\n", pd.Series(train_y_resampled).value_counts(), "\n")

# Step 6: Train the AIS Model (RNSA)
rnsa = RNSA(num_detectors=500, radius=0.2, verbose=True)
rnsa.fit(train_x_resampled, train_y_resampled)

# Step 7: Make predictions
pred_y = rnsa.predict(test_x)

# Step 8: Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(test_y, pred_y))
print("\nClassification Report:\n", classification_report(test_y, pred_y))
print("Accuracy: ", accuracy_score(test_y, pred_y))
print("Precision:", precision_score(test_y, pred_y, zero_division=0))
print("Recall:   ", recall_score(test_y, pred_y, zero_division=0))
print("F1 Score: ", f1_score(test_y, pred_y, zero_division=0))
