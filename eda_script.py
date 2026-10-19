import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the clean dataset
df = pd.read_csv(r'C:\Users\mahim\OneDrive\Desktop\cleaned_dataset.csv')

# 2. Generate and print Statistical Summary reports in the terminal
print("=== GLOBAL STATISTICAL SUMMARY ===")
print(df.describe())
print("\n=== CORRELATION MATRIX ===")
correlation_matrix = df[['Age', 'Study_Hours', 'Exam_Score']].corr()
print(correlation_matrix)

# ==========================================
# 3. GENERATE THE EDA VISUALIZATION DASHBOARD
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Exploratory Data Analysis (EDA) Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Correlation Heatmap (Identifies key influencing factors)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('Variable Correlation Heatmap', fontsize=12, fontweight='bold')

# Plot 2: Boxplot of Exam Scores by Study Hours category
# Groups study hours to show trends clearly
df['Study_Category'] = pd.cut(df['Study_Hours'], bins=[0, 4, 8, 12, 20], labels=['Low (0-4)', 'Medium (4-8)', 'High (8-12)', 'Intense (12+)'])
sns.boxplot(data=df, x='Study_Category', y='Exam_Score', palette='Set2', ax=axes[1])
axes[1].set_title('Exam Scores by Study Intensity Category', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Study Hours Category')
axes[1].set_ylabel('Exam Score (%)')

plt.tight_layout()

# Save the EDA dashboard image to your Desktop workspace
plt.savefig(r'C:\Users\mahim\OneDrive\Desktop\eda_dashboard.png', dpi=300, bbox_inches='tight')
print("\nEDA dashboard image created successfully on your Desktop!")
