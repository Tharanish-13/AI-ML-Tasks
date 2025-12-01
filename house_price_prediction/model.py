import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Predict California House Price from User Input ---")
print("Enter the following details:")

try:
    MedInc = float(input("1. Median Income (in tens of thousands): "))
    HouseAge = float(input("2. House Age (years): "))
    AveRooms = float(input("3. Average Rooms per Household: "))
    AveBedrms = float(input("4. Average Bedrooms per Household: "))
    Population = float(input("5. Population: "))
    AveOccup = float(input("6. Average Occupancy: "))
    Latitude = float(input("7. Latitude: "))
    Longitude = float(input("8. Longitude: "))

    user_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    user_prediction = model.predict(user_data)[0]

    print("\nPredicted House Price (in $100,000s):", round(user_prediction, 3))
    print("Approx Estimated Value in USD: $", round(user_prediction * 100000, 2))

except ValueError:
    print("❌ Invalid input! Please enter numeric values.")

y_pred = model.predict(X_test)

print("\n📌 Model Performance")
print("---------------------")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.show()
