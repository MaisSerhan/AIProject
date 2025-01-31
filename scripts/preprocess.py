import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# File paths
input_file = '../data/dataset.xlsx'  # Path to the original dataset
output_file = '../data/processed_data.csv'  # Path to save processed data

# Load the dataset
print("Loading dataset...")
df = pd.read_excel(input_file)

# Handling missing values
print("Handling missing values...")
df['Gender'].fillna('Unknown', inplace=True)  # Fill categorical nulls
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill numeric nulls
df['Dur'].fillna(df['Dur'].median(), inplace=True)
df['PPV'].dropna(inplace=True)  # Ensure the target variable has no missing values

# Encode categorical data
print("Encoding categorical features...")
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Scale numeric features
print("Scaling numeric features...")
scaler = StandardScaler()
df[['Age', 'Dur']] = scaler.fit_transform(df[['Age', 'Dur']])

# Save the preprocessed dataset
print(f"Saving preprocessed data to {output_file}...")
df.to_csv(output_file, index=False)
print("Processing complete.")
