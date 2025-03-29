# Import required libraries
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
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
    print("❌ Model file not found. Run the training script first to save the model.")
    exit()

# Step 4: Select a Test Sample (Unseen Data)
i = random.randint(0, len(X_test) - 1)  # Select a random test sample
actual_label = "Malignant" if y_test.iloc[i] == 1 else "Benign"

# Get the test instance's original feature values
instance_df = pd.DataFrame(scaler.inverse_transform(X_test[i].reshape(1, -1)), columns=df.columns[1:])
instance_df.to_csv("lime_shap_test_sample.csv", index=False)

# Step 5: Perform Prediction
predicted_label = "Malignant" if ensemble_model.predict(X_test[i].reshape(1, -1))[0] == 1 else "Benign"
probabilities = ensemble_model.predict_proba(X_test[i].reshape(1, -1))[0]

# Display Prediction Results
print("\n🔹 **Selected Test Sample for LIME & SHAP Explanation** 🔹")
print(f"Instance Index: {i}")
print(f"Actual Diagnosis: {actual_label}\n")
print(instance_df)

print("\n🔹 **Model Prediction Results** 🔹")
print(f"Predicted Diagnosis: {predicted_label}")
print(f"Model Probabilities: Benign={probabilities[0]:.4f}, Malignant={probabilities[1]:.4f}")

# Step 6: SHAP Analysis (Global Feature Importance)
shap.initjs()

# Use SHAP on Gradient Boosting instead of VotingClassifier
gb_model = ensemble_model.named_estimators_["gradient_boosting"]
explainer = shap.Explainer(gb_model, X_train)
shap_values = explainer(X_test)

# Generate and Save SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=df.columns[1:], show=False)
plt.savefig("shap_summary_plot.png")
plt.show()
print("✅ SHAP summary plot saved as 'shap_summary_plot.png'")

# Step 7: SHAP Force Plot (Local Explanation for One Instance) - Save as HTML
force_plot = shap.force_plot(
    shap_values[i].base_values, 
    shap_values[i].values, 
    X_test[i, :], 
    feature_names=df.columns[1:]
)

shap.save_html("shap_force_plot.html", force_plot)
print("✅ SHAP force plot saved as 'shap_force_plot.html'")


# Step 8: LIME Explanation (Local Feature Importance)
explainer = lime.lime_tabular.LimeTabularExplainer(
    scaler.inverse_transform(X_train),  # Transform back to original range
    feature_names=df.columns[1:], 
    class_names=['Benign', 'Malignant'], 
    discretize_continuous=True
)

# Generate LIME Explanation for a Single Test Sample
exp = explainer.explain_instance(
    scaler.inverse_transform(X_test[i, :].reshape(1, -1)).flatten(),
    gb_model.predict_proba,
    num_features=10
)

# Save LIME Explanation Plot
lime_fig = exp.as_pyplot_figure()
lime_fig.savefig("lime_explanation.png")
plt.show()
print("✅ LIME explanation saved as 'lime_explanation.png'")
