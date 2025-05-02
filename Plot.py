import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv("Results/Ais_Results_100.csv")  # Update path if needed

# --- Line Plot for Spam vs Ham Accuracy ---
plt.figure(figsize=(14, 6))

# Group and plot for each radius
for radius in sorted(df['radius'].unique()):
    subset = df[df['radius'] == radius]
    plt.plot(subset['num_detectors'], subset['spam_accuracy'], label=f"Spam r={radius}", linestyle='--')
    plt.plot(subset['num_detectors'], subset['ham_accuracy'], label=f"Ham r={radius}", linestyle='-')

plt.title("Spam vs Ham Accuracy by Number of Detectors (Grouped by Radius)")
plt.xlabel("Number of Detectors")
plt.ylabel("Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
