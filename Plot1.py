import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
#file_path = "irrelevant/ais_grid_search_results6.csv"
file_path = "Results/Radius_doesnt_matter.csv"
df = pd.read_csv(file_path)

# Create the line plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x="radius", y="spam_accuracy", hue="num_detectors", marker='o', palette="tab10")

# Customize the plot
plt.title("Spam Accuracy vs Radius for Different Number of Detectors")
plt.xlabel("Radius")
plt.ylabel("Spam Accuracy")
plt.legend(title="Num Detectors", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
