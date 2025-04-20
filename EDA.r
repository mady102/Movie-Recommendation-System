import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("tmdb_with_moods.csv")

# Set style
sns.set(style="whitegrid")

# 1. Summary Statistics
print("Summary Statistics:\n", df.describe())

# 2. Distribution Plots for Numerical Features
numerical_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 3. Boxplots to Spot Outliers
for col in numerical_cols:
    plt.figure(figsize=(8, 2))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 6))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 5. Count Plots for Categorical Data (e.g., moods)
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='mood', order=df['mood'].value_counts().index)
plt.title('Mood Distribution')
plt.xlabel('Mood')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
