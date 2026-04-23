# ============================================================
#   Smart Environmental Monitoring System
#   Using Machine Learning to Predict Air Quality Index (AQI)
# ============================================================

# ── 1. Import Libraries ──────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ── 2. Load Dataset ──────────────────────────────────────────
data = {
    "PM2.5": [85, 60, 120, 30, 150, 90, 110, 70, 140, 100],
    "PM10":  [120, 90, 180, 50, 220, 140, 160, 100, 200, 150],
    "NO2":   [40, 30, 60, 20, 80, 50, 55, 35, 75, 45],
    "SO2":   [10, 8, 15, 5, 20, 12, 14, 9, 18, 11],
    "CO":    [1.2, 0.8, 1.5, 0.5, 2.0, 1.1, 1.3, 0.9, 1.8, 1.0],
    "O3":    [30, 25, 40, 20, 50, 35, 38, 28, 45, 32],
    "AQI":   [150, 110, 200, 70, 300, 160, 180, 120, 250, 140]
}

df = pd.DataFrame(data)
print("── Dataset Preview ──")
print(df.head())

# ── 3. Data Understanding ────────────────────────────────────
print("\n── Shape ──")
print(df.shape)

print("\n── Info ──")
df.info()

print("\n── Statistical Summary ──")
print(df.describe())

# ── 4. Data Cleaning ─────────────────────────────────────────
print("\n── Missing Values ──")
print(df.isnull().sum())
# No missing values in this dataset

# ── 5. Feature Engineering ───────────────────────────────────
# Create AQI Category: 0 = Good, 1 = Moderate, 2 = Poor
df['Category'] = df['AQI'].apply(
    lambda x: 0 if x <= 100 else 1 if x <= 200 else 2
)
print("\n── AQI Categories ──")
print(df[['AQI', 'Category']])

# ── 6. Exploratory Data Analysis ─────────────────────────────
plt.figure(figsize=(6, 4))
plt.scatter(df['PM2.5'], df['AQI'], color='steelblue', edgecolors='black')
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.title("PM2.5 vs AQI")
plt.tight_layout()
plt.savefig("pm25_vs_aqi.png")
plt.show()
print("Plot saved: pm25_vs_aqi.png")

# ── 7. Train-Test Split ──────────────────────────────────────
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 8. Model Training ────────────────────────────────────────

# 8a. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 8b. Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 8c. Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ── 9. Model Evaluation ──────────────────────────────────────
print("\n── Model R² Scores ──")
print("Linear Regression R2 :", round(r2_score(y_test, lr_pred), 4))
print("Decision Tree R2     :", round(r2_score(y_test, dt_pred), 4))
print("Random Forest R2     :", round(r2_score(y_test, rf_pred), 4))

print("\n── Model MSE Scores ──")
print("Linear Regression MSE:", round(mean_squared_error(y_test, lr_pred), 4))
print("Decision Tree MSE    :", round(mean_squared_error(y_test, dt_pred), 4))
print("Random Forest MSE    :", round(mean_squared_error(y_test, rf_pred), 4))

# ── 10. Visualization – Model Comparison ─────────────────────
models = ["Linear Regression", "Decision Tree", "Random Forest"]
scores = [
    r2_score(y_test, lr_pred),
    r2_score(y_test, dt_pred),
    r2_score(y_test, rf_pred)
]

plt.figure(figsize=(7, 4))
bars = plt.bar(models, scores, color=['#4C72B0', '#DD8452', '#55A868'], edgecolor='black')
plt.ylabel("R² Score")
plt.title("Model Comparison – R² Score")
plt.ylim(0, 1.1)
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02,
             f"{score:.3f}", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
print("Plot saved: model_comparison.png")

# ── 11. Prediction Example ───────────────────────────────────
sample = np.array([[100, 150, 50, 12, 1.0, 35]])
prediction = rf.predict(sample)
print(f"\n── Prediction Example ──")
print(f"Input  : PM2.5=100, PM10=150, NO2=50, SO2=12, CO=1.0, O3=35")
print(f"Predicted AQI : {prediction[0]:.2f}")

# ── 12. AQI Category Distribution (Optional Classification) ──
df['Predicted_Category'] = df['AQI'].apply(
    lambda x: 0 if x <= 100 else 1 if x <= 200 else 2
)

plt.figure(figsize=(5, 4))
sns.countplot(x='Predicted_Category', data=df,
              palette=['#2ecc71', '#f39c12', '#e74c3c'])
plt.xticks([0, 1, 2], ['Good', 'Moderate', 'Poor'])
plt.title("AQI Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("aqi_category_distribution.png")
plt.show()
print("Plot saved: aqi_category_distribution.png")

print("\n✅ Smart Environmental Monitoring System – Complete!")
