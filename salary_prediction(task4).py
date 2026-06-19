import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Download dataset
path = kagglehub.dataset_download("rsadiq/salary")

# Load dataset
df = pd.read_csv(os.path.join(path, "Salary.csv"))

print("First 5 Rows:")
print(df.head())

# Features and Target
X = df[['YearsExperience']]
y = df['Salary']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Salary
y_pred = model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)

print("\nSALARY PREDICTION MODEL")
print("=" * 40)
print(f"Total Records       : {len(df)}")
print(f"Mean Squared Error  : {mse:.2f}")
print("=" * 40)

# Graph
plt.figure(figsize=(8,5))

plt.scatter(
    X_test,
    y_test,
    label="Actual Salary"
)

plt.plot(
    X_test,
    y_pred,
    label="Predicted Salary"
)

plt.title("Salary Prediction using Linear Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nSalary Prediction Completed Successfully!")