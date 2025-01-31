import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load preprocessed data
data_file = '../data/processed_data.csv'
print("Loading preprocessed data...")
df = pd.read_csv(data_file)

# Prepare features and target
X = df[['Gender', 'Age', 'Dur']]
y = df['PPV']

# Split data into training and testing sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
print("Training XGBoost model...")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# Make predictions
print("Evaluating model...")
y_pred = model.predict(X_test)

# Evaluate with RMSE and MAE
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Save the trained model
model.save_model('../models/saved_model_xgboost.json')
print("Model saved.")
