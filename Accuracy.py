import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "Results/Ais_Results_100.csv"
df = pd.read_csv(file_path)

# Calculate total accuracy
df["total_accuracy"] = (df["TP"] + df["TN"]) / (df["TP"] + df["TN"] + df["FP"] + df["FN"])

# Plot total accuracy vs number of detectors
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="num_detectors", y="total_accuracy", marker='o')

# Customize the plot
plt.title("Total Accuracy vs Number of Detectors")
plt.xlabel("Number of Detectors")
plt.ylabel("Total Accuracy")
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
