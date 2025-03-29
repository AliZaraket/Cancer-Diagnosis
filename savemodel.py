# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
import joblib  # To save/load models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Step 3: Train SVM Models with Different Kernels
svm_linear = SVC(kernel='linear', probability=True, random_state=42)
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
svm_sigmoid = SVC(kernel='sigmoid', probability=True, random_state=42)

# Train models
svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)
svm_sigmoid.fit(X_train, y_train)

# Train Gradient Boosting Classifier
gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gradient_boosting.fit(X_train, y_train)

# Step 4: Train Ensemble Model (Voting Classifier)
ensemble_model = VotingClassifier(
    estimators=[
        ('svm_linear', svm_linear),
        ('svm_rbf', svm_rbf),
        ('svm_sigmoid', svm_sigmoid),
        ('gradient_boosting', gradient_boosting)
    ],
    voting='soft'
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Save the trained model for later use
joblib.dump(ensemble_model, "ensemble_model.pkl")
print("✅ Model trained and saved as 'ensemble_model.pkl'")

# Step 5: Make Predictions & Evaluate Models
y_pred_ensemble = ensemble_model.predict(X_test)

# Evaluate the ensemble model
print("✅ Ensemble Model Performance:\n", classification_report(y_test, y_pred_ensemble))

# Step 6: Individual Model Accuracies
models = ["SVM (Linear)", "SVM (RBF)", "SVM (Sigmoid)", "Gradient Boosting", "Ensemble"]
accuracies = [
    accuracy_score(y_test, svm_linear.predict(X_test)),
    accuracy_score(y_test, svm_rbf.predict(X_test)),
    accuracy_score(y_test, svm_sigmoid.predict(X_test)),
    accuracy_score(y_test, gradient_boosting.predict(X_test)),
    accuracy_score(y_test, y_pred_ensemble)
]

# Confusion Matrices
conf_matrices = [
    confusion_matrix(y_test, svm_linear.predict(X_test)),
    confusion_matrix(y_test, svm_rbf.predict(X_test)),
    confusion_matrix(y_test, svm_sigmoid.predict(X_test)),
    confusion_matrix(y_test, gradient_boosting.predict(X_test)),
    confusion_matrix(y_test, y_pred_ensemble)
]

for i, model in enumerate(models):
    print(f"\nConfusion Matrix for {model}:\n{conf_matrices[i]}")

# Step 7: Save the Accuracy Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies)
plt.ylim(min(accuracies) - 0.02, max(accuracies) + 0.02)
plt.title("Model Accuracies")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.savefig("model_accuracies.png")
plt.show()

print("✅ Accuracy plot saved as 'model_accuracies.png'")
