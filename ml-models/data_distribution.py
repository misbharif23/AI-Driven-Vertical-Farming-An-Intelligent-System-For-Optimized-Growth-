import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the results
df = pd.read_csv("cluster_assignments.csv")

# 2. Count images per cluster
cluster_counts = df['cluster_id'].value_counts().sort_index()

# 3. Create a clean DataFrame for your report
dist_table = pd.DataFrame({
    'Cluster_ID': cluster_counts.index,
    'Image_Count': cluster_counts.values,
    'Percentage': (cluster_counts / len(df) * 100).round(2)
})

print("\n=== DATA DISTRIBUTION SUMMARY ===")
print(dist_table.to_string(index=False))

# 4. Plot the Bar Chart
plt.figure(figsize=(8, 5))
cluster_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Images across Clusters')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Images')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("cluster_distribution.png")
plt.show()