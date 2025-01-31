import os
import subprocess

# Run preprocessing
print("Running preprocessing...")
subprocess.run(["python", "scripts/preprocess.py"])

# Train the XGBoost model
print("Training the XGBoost model...")
subprocess.run(["python", "scripts/train_xgboost.py"])
