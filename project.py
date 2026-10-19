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
