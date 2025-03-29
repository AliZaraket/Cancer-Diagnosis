# Import required libraries
import numpy as np
import pandas as pd
import joblib  # Load saved model
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load Dataset 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
df = pd.read_csv(url, names=column_names)

# Drop ID column
df.drop(columns=['ID'], inplace=True)

# Convert Diagnosis to binary (M=1, B=0)
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

# Step 2: Data Preprocessing
X = df.drop(columns=['Diagnosis'])  # Features
y = df['Diagnosis']  # Target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Load the Trained Model
try:
    ensemble_model = joblib.load("ensemble_model.pkl")
    print("✅ Loaded trained model from 'ensemble_model.pkl'")
except FileNotFoundError:
    print("❌ Model file not found. Run 'cancer_diagnosis.py' first to train and save the model.")
    exit()

# Step 4: Select a Test Sample (Unseen Data)
i = random.randint(0, len(X_test) - 1)  # Select a random test sample
actual_label = "Malignant" if y_test.iloc[i] == 1 else "Benign"

# Get the test instance's original feature values
instance_df = pd.DataFrame(scaler.inverse_transform(X_test[i].reshape(1, -1)), columns=df.columns[1:])

# Save the test sample for review
instance_df.to_csv("predicted_test_sample.csv", index=False)

# Step 5: Make Prediction on the Test Sample
predicted_label = "Malignant" if ensemble_model.predict(X_test[i].reshape(1, -1))[0] == 1 else "Benign"

# Get probability predictions from the model
probabilities = ensemble_model.predict_proba(X_test[i].reshape(1, -1))[0]

# Step 6: Display the Results
print("\n🔹 **Selected Test Sample for Prediction** 🔹")
print(f"Instance Index: {i}")
print(f"Actual Diagnosis: {actual_label}\n")
print(instance_df)

print("\n🔹 **Model Prediction Results** 🔹")
print(f"Predicted Diagnosis: {predicted_label}")
print(f"Model Probabilities: Benign={probabilities[0]:.4f}, Malignant={probabilities[1]:.4f}")

# Step 7: Check if the Prediction is Correct
if actual_label == predicted_label:
    print("✅ **Model Prediction is CORRECT**")
else:
    print("❌ **Model Prediction is INCORRECT**")
