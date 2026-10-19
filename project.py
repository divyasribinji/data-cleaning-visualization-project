import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


# 1. Load your clean dataset
df = pd.read_csv(r'C:\Users\mahim\OneDrive\Desktop\cleaned_dataset.csv')

# 2. Create a Classification Target Variable (Pass = 1, Fail = 0)
# Students scoring 70% or higher are marked as passed
df['Pass'] = (df['Exam_Score'] >= 70).astype(int)

# 3. Separate Features (X) and Target (y)
X = df[['Age', 'Study_Hours']]
y = df['Pass']

# 4. Train/Test Split (80% training data, 20% testing data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Make Predictions and Calculate Accuracy
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Complete! Test Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# 7. VISUALIZE PERFORMANCE (Confusion Matrix & ROC Curve)
# ==========================================
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Machine Learning Model Evaluation Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Predicted Fail', 'Predicted Pass'],
            yticklabels=['Actual Fail', 'Actual Pass'])
axes[0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
axes[1].legend(loc="lower right")

plt.tight_layout()

# Save dashboard evaluation directly to your Desktop screen
plt.savefig(r'C:\Users\mahim\OneDrive\Desktop\ml_performance_dashboard.png', dpi=300, bbox_inches='tight')
print("ML Evaluation dashboard image created successfully on your Desktop!")
