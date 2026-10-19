import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. GENERATE A REAL-WORLD FINANICIAL DATASET
# ==========================================
# Creating 100 days of stock prices data for an AI Tech company
np.random.seed(42)
days = np.arange(1, 101)
base_price = 150
trends = np.cumsum(np.random.normal(0.5, 2, 100)) # Upward moving trend
stock_prices = base_price + trends
trading_volume = np.random.randint(50000, 150000, 100)

finance_df = pd.DataFrame({
    'Day': days,
    'Stock_Price': stock_prices,
    'Volume_Traded': trading_volume
})

# Save the dataset locally
finance_df.to_csv('finance_dataset.csv', index=False)
print("--- FINANCIAL DATASET CREATED SUCCESSFULLY ---")

# ==========================================
# 2. END-TO-END DATA ANALYSIS (EDA)
# ==========================================
print("\n--- STATISTICAL SUMMARY OF STOCK DATA ---")
print(finance_df.describe())

# Calculate moving average trend factor
finance_df['7_Day_Moving_Avg'] = finance_df['Stock_Price'].rolling(window=7).mean()

# ==========================================
# 3. MACHINE LEARNING: PREDICT FUTURE PRICE
# ==========================================
# Use the current Day number to predict the Stock Price trend
X = finance_df[['Day']]
y = finance_df['Stock_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make future predictions
finance_df['Predicted_Price'] = model.predict(X)
print("\n--- MACHINE LEARNING MODEL TRAINED SUCCESSFULLY ---")

# ==========================================
# 4. PRESENT FINDINGS WITH VISUALIZATIONS
# ==========================================
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Real-World Financial Analytics Dashboard (Stock Prediction)', fontsize=16, fontweight='bold')

# Plot 1: Historical Prices and Moving Average Trend
axes[0].plot(finance_df['Day'], finance_df['Stock_Price'], label='Actual Stock Price', color='royalblue', lw=2)
axes[0].plot(finance_df['Day'], finance_df['7_Day_Moving_Avg'], label='7-Day Moving Average', color='orange', linestyle='--', lw=2)
axes[0].set_title('Stock Price Trends Over 100 Days', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Price ($)')
axes[0].legend()

# Plot 2: Machine Learning Prediction Model (Regression Line)
axes[1].scatter(X_test, y_test, color='crimson', label='Actual Test Data', alpha=0.7)
axes[1].plot(finance_df['Day'], finance_df['Predicted_Price'], color='emerald' if 'emerald' in globals() else 'green', lw=2, label='ML Predicted Trend Line')
axes[1].set_title('Machine Learning Price Prediction', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Price ($)')
axes[1].legend()

plt.tight_layout()

# Save the final real-world project dashboard
plt.savefig('finance_insights_dashboard.png', dpi=300, bbox_inches='tight')
print("Dashboard visualization saved successfully as 'finance_insights_dashboard.png'")
