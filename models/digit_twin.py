
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load your dataset
csv_file = input("Enter your CSV file path: ").strip().strip('"')
df = pd.read_csv(csv_file)

# Features used to predict motor temperature
features = ["speed_rpm", "Current", "Voltage", "temp_amb"]
target = "temp_mot"

# Keep only rows with valid values
data = df[features + [target]].dropna()

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Digital Twin model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("\n=== Digital Twin Results ===")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 3))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# Save the trained model
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": features
    },
    "models/digital_twin.joblib"
)

print("\nDigital Twin saved to models/digital_twin.joblib")