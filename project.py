import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the raw dataset
df = pd.read_csv('raw_dataset.csv')

# 2. Clean data fields (Removes exact row duplicates)
df = df.drop_duplicates()

# 3. Clean metric columns (Filter out impossible outlier rows)
df = df[(df['Age'] > 0) & (df['Age'] < 100)]
df = df[(df['Exam_Score'] >= 0) & (df['Exam_Score'] <= 100)]

# 4. Handle remaining missing value entries using column medians
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Study_Hours'] = df['Study_Hours'].fillna(df['Study_Hours'].median())
df['Exam_Score'] = df['Exam_Score'].fillna(df['Exam_Score'].median())

# 5. Export clean database file
df.to_csv('cleaned_dataset.csv', index=False)

# ==========================================
# 6. GENERATE THE VISUAL DASHBOARD IMAGE
# ==========================================
# Set up a clean grid style for the graphs
sns.set_theme(style="whitegrid")

# Create a window layout containing two plots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Student Performance Dashboard', fontsize=16, fontweight='bold')

# Plot A: Assigned to the first window slot (axes[0])
sns.scatterplot(data=df, x='Study_Hours', y='Exam_Score', hue='Age', palette='magma', ax=axes[0])
axes[0].set_title('Study Hours vs. Exam Scores', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Hours Studied')
axes[0].set_ylabel('Exam Score (%)')

# Plot B: Assigned to the second window slot (axes[1])
sns.histplot(data=df, x='Exam_Score', kde=True, color='crimson', ax=axes[1])
axes[1].set_title('Overall Score Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Exam Score (%)')
axes[1].set_ylabel('Number of Students')

plt.tight_layout()

# Force save the visual image file directly onto your Desktop workspace path
plt.savefig(r'C:\Users\mahim\OneDrive\Desktop\data_insights_dashboard.png', dpi=300, bbox_inches='tight')
print("Dashboard image created successfully on your Desktop!")
