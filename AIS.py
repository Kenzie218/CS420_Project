import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from aisp.nsa import RNSA
from tqdm import tqdm

# Load dataset
df = pd.read_csv('spam_ham_dataset.csv')
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize messages
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label_num'].values

# Split data
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Oversample
ros = RandomOverSampler(random_state=42)
train_x_resampled, train_y_resampled = ros.fit_resample(train_x, train_y)

# Prepare parameter grid
detector_range = range(100, 10001, 100)
#radius_range = np.round(np.arange(0.05, 0.250, 0.05), 2)

# Store results
results = []

# Grid search
for num_detectors in tqdm(detector_range, desc="Detector Loop"):
    
        rnsa = RNSA(N=num_detectors, r=0.25, seed=25, verbose=False)
        rnsa.fit(train_x_resampled, train_y_resampled)
        pred_y = rnsa.predict(test_x)
        cm = confusion_matrix(test_y, pred_y)
        
        tn, fp = cm[0]
        fn, tp = cm[1]

        ham_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0
        spam_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0

        results.append({
            'num_detectors': num_detectors,
            'radius': 0.25,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'ham_accuracy': round(ham_accuracy, 4),
            'spam_accuracy': round(spam_accuracy, 4)
        })

# Output results
results_df = pd.DataFrame(results)
results_df.to_csv("ais_grid_search_results8.csv", index=False)

# Show sample of results
import ace_tools as tools; tools.display_dataframe_to_user(name="AIS Grid Search Results", dataframe=results_df)
