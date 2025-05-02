import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV
file_path = "Results/Ais_Results_500.csv"
df = pd.read_csv(file_path)

# Pivot tables for heatmaps
heatmap_data_spam = df.pivot_table(index="radius", columns="num_detectors", values="spam_accuracy")
heatmap_data_ham = df.pivot_table(index="radius", columns="num_detectors", values="ham_accuracy")

# Plotting heatmaps
plt.figure(figsize=(14, 6))
plt.title("Spam Accuracy Heatmap")
sns.heatmap(heatmap_data_spam, annot=True, fmt=".2f", cmap="YlOrRd")
plt.xlabel("Number of Detectors")
plt.ylabel("Radius")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.title("Ham Accuracy Heatmap")
sns.heatmap(heatmap_data_ham, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Number of Detectors")
plt.ylabel("Radius")
plt.tight_layout()
plt.show()
